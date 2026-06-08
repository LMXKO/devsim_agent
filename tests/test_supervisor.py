from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tcad_agent.llm import LLMConfig
from tcad_agent.supervisor import (
    SupervisorAction,
    SupervisorActionKind,
    SupervisorActionStatus,
    SupervisorState,
    SupervisorStatus,
    choose_next_action,
    execute_action,
    run_supervisor,
    supervisor_action_from_agent,
    utc_timestamp,
)


def state_with_index(goal: str, recent_records: list[dict[str, object]] | None = None) -> SupervisorState:
    now = utc_timestamp()
    return SupervisorState(
        status=SupervisorStatus.RUNNING,
        supervisor_id="sup_unit",
        goal_text=goal,
        supervisor_dir="/tmp/sup_unit",
        created_at=now,
        updated_at=now,
        execute=True,
        max_cycles=3,
        last_index_summary={"records_indexed": 1},
        recent_records=recent_records
        if recent_records is not None
        else [
            {
                "experiment_id": "opt_a",
                "kind": "adaptive_optimization",
                "status": "completed",
                "state_path": "/tmp/opt_a/optimization_state.json",
            }
        ],
    )


class FakeSupervisorClient:
    config = LLMConfig(model="fake-supervisor-agent")

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        return self.response


class SupervisorTest(unittest.TestCase):
    def test_plan_only_writes_rebuild_index_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = run_supervisor(
                "做一个 MOS C-V",
                supervisor_id="sup_plan",
                supervisor_root=Path(tmp),
                execute=False,
            )

            self.assertEqual(state.status, SupervisorStatus.PLANNED)
            self.assertEqual(state.actions[0].kind, SupervisorActionKind.REBUILD_INDEX)
            self.assertTrue((Path(tmp) / "sup_plan" / "supervisor_state.json").exists())
            self.assertIn("planned_action", state.checkpoint)
            self.assertTrue(state.checkpoint["agent_first_policy"]["enabled"])

    def test_supervisor_agent_can_override_deterministic_candidate(self) -> None:
        state = state_with_index("把最近的优化结果做 dashboard")
        deterministic = choose_next_action(state)
        client = FakeSupervisorClient(
            json.dumps(
                {
                    "action": {
                        "kind": "generate_dashboard",
                        "reason": "agent sees a completed optimization state and the user asked for visual inspection.",
                        "request": {"state": "/tmp/opt_a/optimization_state.json"},
                    },
                    "observation_summary": "recent_records contains a completed adaptive optimization.",
                    "hypothesis_zh": "用户要看结果，不需要再跑新仿真。",
                    "evidence_used": ["goal_text", "recent_records", "deterministic_candidate"],
                }
            )
        )

        action, decision = supervisor_action_from_agent(state, deterministic, client=client)

        self.assertEqual(action.kind, SupervisorActionKind.GENERATE_DASHBOARD)
        self.assertFalse(decision["fallback_used"])
        self.assertEqual(decision["hypothesis_zh"], "用户要看结果，不需要再跑新仿真。")
        self.assertEqual(len(client.calls), 1)

    def test_supervisor_agent_falls_back_on_unsupported_or_command_response(self) -> None:
        state = state_with_index("做 MOS capacitor C-V")
        deterministic = choose_next_action(state)
        client = FakeSupervisorClient(
            json.dumps(
                {
                    "action": {
                        "kind": "run_shell",
                        "reason": "bad",
                        "request": {},
                        "command": "rm -rf runs",
                    }
                }
            )
        )

        action, decision = supervisor_action_from_agent(state, deterministic, client=client)

        self.assertEqual(action.kind, deterministic.kind)
        self.assertTrue(decision["fallback_used"])
        self.assertIn("unsupported action kind", decision["failure_reason"])

    def test_routes_mos_cv_after_index_refresh(self) -> None:
        action = choose_next_action(state_with_index("做 MOS capacitor C-V 从 -0.5V 到 0.5V 步长 0.25V 氧化层 5nm"))

        self.assertEqual(action.kind, SupervisorActionKind.RUN_MOS_CV)
        self.assertEqual(action.request["start"], -0.5)
        self.assertEqual(action.request["stop"], 0.5)
        self.assertEqual(action.request["step"], 0.25)
        self.assertEqual(action.request["oxide_thickness_nm"], 5.0)

    def test_routes_mos_cv_fixed_charge_natural_goal(self) -> None:
        action = choose_next_action(
            state_with_index("MOSCAP 曲线往负压偏，做 C-V gate sweep -2V 到 2V，tox 5nm，固定电荷 5e11 cm^-2")
        )

        self.assertEqual(action.kind, SupervisorActionKind.RUN_MOS_CV)
        self.assertEqual(action.request["start"], -2.0)
        self.assertEqual(action.request["stop"], 2.0)
        self.assertEqual(action.request["fixed_oxide_charge_cm2"], 5e11)

    def test_routes_mosfet_before_generic_mos_cv(self) -> None:
        action = choose_next_action(
            state_with_index("做 2D MOSFET Id-Vg gate_start 0V gate_stop 1V gate_step 0.25V drain_voltage 0.05V")
        )

        self.assertEqual(action.kind, SupervisorActionKind.RUN_MOSFET_2D)
        self.assertEqual(action.request["gate_start"], 0.0)
        self.assertEqual(action.request["gate_stop"], 1.0)
        self.assertEqual(action.request["gate_step"], 0.25)
        self.assertEqual(action.request["drain_voltage"], 0.05)

    def test_routes_mosfet_output_characteristic_natural_goal(self) -> None:
        action = choose_next_action(
            state_with_index("客户说 NMOS output characteristic 高 Vd 有 kink，固定 Vg=0.8/1.0/1.2，Vd 0 到 1.2V")
        )

        self.assertEqual(action.kind, SupervisorActionKind.RUN_MOSFET_2D)
        self.assertEqual(action.request["sweep_type"], "idvd")
        self.assertEqual(action.request["idvd_gate_voltage"], 1.2)
        self.assertEqual(action.request["drain_start"], 0.0)
        self.assertEqual(action.request["drain_stop"], 1.2)
        self.assertEqual(action.request["impact_ionization_model"], "none")
        self.assertEqual(action.request["x_divisions"], 12)
        self.assertEqual(action.request["silicon_y_divisions"], 4)

    def test_routes_mosfet_explicit_avalanche_model_when_requested(self) -> None:
        action = choose_next_action(
            state_with_index("做 2D NMOS Id-Vd，并显式打开 avalanche/impact ionization 模型检查高场电流")
        )

        self.assertEqual(action.kind, SupervisorActionKind.RUN_MOSFET_2D)
        self.assertEqual(action.request["impact_ionization_model"], "selberherr")

    def test_routes_mosfet_mobility_model_natural_goal(self) -> None:
        action = choose_next_action(
            state_with_index("这个 NMOS 的 Id-Vg 对 mobility model 敏感，先用 doping-dependent mobility 跑一条")
        )

        self.assertEqual(action.kind, SupervisorActionKind.RUN_MOSFET_2D)
        self.assertEqual(action.request["mobility_model"], "doping_dependent")
        self.assertEqual(action.request["tcad_deck_spec"]["physics_models"]["mobility_model"], "doping_dependent")

    def test_routes_mosfet_advanced_physics_into_deck(self) -> None:
        action = choose_next_action(
            state_with_index("NMOS 高温 Id-Vg，electron mobility 500，hole mobility 180，SRH lifetime 1e-7，Dit 1e11，并打开 avalanche")
        )

        self.assertEqual(action.kind, SupervisorActionKind.RUN_MOSFET_2D)
        self.assertEqual(action.request["temperature_k"], 350.0)
        self.assertEqual(action.request["electron_mobility_cm2_v_s"], 500.0)
        self.assertEqual(action.request["hole_mobility_cm2_v_s"], 180.0)
        self.assertEqual(action.request["electron_lifetime_s"], 1e-7)
        self.assertEqual(action.request["impact_ionization_model"], "selberherr")
        self.assertEqual(action.request["tcad_deck_spec"]["physics_models"]["coupling_status"], "needs_benchmark_confirmation")

    def test_routes_diode_high_temp_and_lifetime_natural_goal(self) -> None:
        action = choose_next_action(
            state_with_index("PN diode 高温漏电 baseline，lifetime 1e-7，先从 0V 扫到 -10V")
        )

        self.assertEqual(action.kind, SupervisorActionKind.RUN_DIODE_BREAKDOWN)
        self.assertEqual(action.request["temperature_k"], 350.0)
        self.assertEqual(action.request["electron_lifetime_s"], 1e-7)
        self.assertEqual(action.request["hole_lifetime_s"], 1e-7)

    def test_routes_structure_leakage_edits_to_power_deck_mutations(self) -> None:
        action = choose_next_action(
            state_with_index("这个结构漏电偏高，改 field plate / drift doping / lifetime 看看")
        )

        self.assertEqual(action.kind, SupervisorActionKind.RUN_EXTENDED_DEVICE)
        self.assertEqual(action.request["device_type"], "power_mosfet_bv_ron")
        self.assertEqual(action.request["fidelity"], "physics_1d")
        targets = {mutation["target"] for mutation in action.request["tcad_deck_mutations"]}
        self.assertEqual(targets, {"field_plate", "drift_doping", "lifetime"})

    def test_routes_diode_customer_leakage_investigation_natural_goal(self) -> None:
        action = choose_next_action(
            state_with_index("客户反馈 PN 二极管 -5V 漏电比实测高一个数量级，帮我先跑 baseline reverse IV，下一轮看 SRH lifetime 敏感性")
        )

        self.assertEqual(action.kind, SupervisorActionKind.RUN_DIODE_BREAKDOWN)
        self.assertEqual(action.request["leakage_voltage_v"], -5.0)
        self.assertLessEqual(action.request["stop"], -5.0)
        self.assertEqual(action.request["quality_max_leakage_abs_current_a"], 1e-6)

    def test_routes_diode_leakage_limit_and_bv_natural_goal(self) -> None:
        action = choose_next_action(
            state_with_index("请检查这个二极管在 5V 漏电要低于 10nA，同时提取 BV/击穿电压")
        )

        self.assertEqual(action.kind, SupervisorActionKind.RUN_DIODE_BREAKDOWN)
        self.assertEqual(action.request["leakage_voltage_v"], -5.0)
        self.assertAlmostEqual(action.request["quality_max_leakage_abs_current_a"], 10e-9)
        self.assertTrue(action.request["require_breakdown"])

    def test_routes_diode_breakdown_before_generic_pn_iv(self) -> None:
        action = choose_next_action(
            state_with_index("做 PN 二极管 breakdown 从 0V 到 -2V step 0.5V breakdown_current 1e-6")
        )

        self.assertEqual(action.kind, SupervisorActionKind.RUN_DIODE_BREAKDOWN)
        self.assertEqual(action.request["start"], 0.0)
        self.assertEqual(action.request["stop"], -2.0)
        self.assertEqual(action.request["step"], 0.5)
        self.assertEqual(action.request["breakdown_current_a"], 1e-6)

    def test_routes_mesh_convergence(self) -> None:
        action = choose_next_action(state_with_index("对 PN IV 做 mesh convergence relative_tolerance 0.01"))

        self.assertEqual(action.kind, SupervisorActionKind.RUN_MESH_CONVERGENCE)
        self.assertIn("base_task_text", action.request)
        self.assertEqual(action.request["convergence_request"]["relative_tolerance"], 0.01)

    def test_routes_schottky_extended_device_instead_of_generic_pn_iv(self) -> None:
        action = choose_next_action(state_with_index("做 Schottky diode forward IV 并提取 barrier height"))

        self.assertEqual(action.kind, SupervisorActionKind.RUN_EXTENDED_DEVICE)
        self.assertEqual(action.request["device_type"], "schottky_diode")

    def test_routes_schottky_calibration_before_extended_device(self) -> None:
        action = choose_next_action(
            state_with_index("校准 Schottky diode 到可信曲线 trusted_curve.csv，并用 DEVSIM 复核")
        )

        self.assertEqual(action.kind, SupervisorActionKind.RUN_SCHOTTKY_CALIBRATION)
        self.assertEqual(action.request["calibration_id"], "sup_unit_schottky_cal_001")
        self.assertEqual(action.request["target_curve_path"], "trusted_curve.csv")
        self.assertTrue(action.request["verify_with_devsim"])

    def test_routes_new_schottky_task_before_historical_conclusion(self) -> None:
        action = choose_next_action(
            state_with_index(
                "校准 Schottky diode barrier height，用 golden curve 拟合，给出异常点和下一轮建议",
                recent_records=[
                    {
                        "experiment_id": "old_mos",
                        "kind": "mosfet_2d_id_sweep",
                        "status": "completed",
                        "state_path": "/tmp/old_mos/state.json",
                    }
                ],
            )
        )

        self.assertEqual(action.kind, SupervisorActionKind.RUN_SCHOTTKY_CALIBRATION)

    def test_routes_new_breakdown_task_before_historical_repair(self) -> None:
        action = choose_next_action(
            state_with_index(
                "做 PN diode reverse leakage 和 breakdown，若收敛失败自动缩小 bias step 后重试",
                recent_records=[
                    {
                        "experiment_id": "old_mos_bad",
                        "kind": "mosfet_2d_id_sweep",
                        "status": "completed",
                        "quality_status": "suspicious",
                        "state_path": "/tmp/old_mos_bad/state.json",
                    }
                ],
            )
        )

        self.assertEqual(action.kind, SupervisorActionKind.RUN_DIODE_BREAKDOWN)

    def test_routes_power_mosfet_extended_device_instead_of_generic_mosfet(self) -> None:
        action = choose_next_action(state_with_index("做 power MOSFET BV 和 Ron 优化"))

        self.assertEqual(action.kind, SupervisorActionKind.RUN_EXTENDED_DEVICE)
        self.assertEqual(action.request["device_type"], "power_mosfet_bv_ron")

    def test_routes_conclusion_to_recent_tool_state(self) -> None:
        action = choose_next_action(
            state_with_index(
                "生成最近 MOSFET 的工程结论",
                recent_records=[
                    {
                        "experiment_id": "mos_a",
                        "kind": "mosfet_2d_id_sweep",
                        "status": "completed",
                        "state_path": "/tmp/mos_a/state.json",
                    }
                ],
            )
        )

        self.assertEqual(action.kind, SupervisorActionKind.GENERATE_CONCLUSION)
        self.assertEqual(action.request["state"], "/tmp/mos_a/state.json")

    def test_routes_conclusion_to_recent_schottky_calibration_state(self) -> None:
        action = choose_next_action(
            state_with_index(
                "生成最近 Schottky 校准的工程结论",
                recent_records=[
                    {
                        "experiment_id": "cal_a",
                        "kind": "schottky_iv_calibration",
                        "status": "completed",
                        "state_path": "/tmp/cal_a/state.json",
                    }
                ],
            )
        )

        self.assertEqual(action.kind, SupervisorActionKind.GENERATE_CONCLUSION)
        self.assertEqual(action.request["state"], "/tmp/cal_a/state.json")

    def test_routes_repair_plan_to_recent_tool_state(self) -> None:
        action = choose_next_action(
            state_with_index(
                "给最近失败的 TCAD run 生成修复策略",
                recent_records=[
                    {
                        "experiment_id": "pn_bad",
                        "kind": "pn_junction_iv_sweep",
                        "status": "failed",
                        "state_path": "/tmp/pn_bad/state.json",
                    }
                ],
            )
        )

        self.assertEqual(action.kind, SupervisorActionKind.GENERATE_REPAIR_PLAN)
        self.assertEqual(action.request["state"], "/tmp/pn_bad/state.json")

    def test_routes_report_to_recent_experiment(self) -> None:
        action = choose_next_action(state_with_index("生成最近优化的报告"))

        self.assertEqual(action.kind, SupervisorActionKind.GENERATE_REPORT)
        self.assertEqual(action.request["state"], "/tmp/opt_a/optimization_state.json")

    def test_routes_dashboard_to_recent_experiment(self) -> None:
        action = choose_next_action(state_with_index("生成最近优化的 dashboard"))

        self.assertEqual(action.kind, SupervisorActionKind.GENERATE_DASHBOARD)
        self.assertEqual(action.request["state"], "/tmp/opt_a/optimization_state.json")

    def test_ambiguous_goal_asks_user(self) -> None:
        action = choose_next_action(state_with_index("帮我看看下一步"))

        self.assertEqual(action.kind, SupervisorActionKind.ASK_USER)

    def test_execute_diode_breakdown_action_uses_tool(self) -> None:
        state = state_with_index("做 diode breakdown")
        action = choose_next_action(state_with_index("做 diode breakdown 从 0V 到 -1V step 0.5V"))
        fake_result = Mock()
        fake_result.model_dump.return_value = {"status": "completed", "run_id": "bd_unit"}

        with patch("tcad_agent.supervisor.run_diode_breakdown_sweep", return_value=fake_result) as runner:
            completed = execute_action(action, state)

        self.assertEqual(completed.status, SupervisorActionStatus.COMPLETED)
        self.assertEqual(completed.result["status"], "completed")
        self.assertEqual(completed.result["run_id"], "bd_unit")
        self.assertEqual(completed.result["tcad_deck_spec"]["device_family"], "pn_diode_breakdown_leakage")
        runner.assert_called_once()

    def test_execute_mosfet_action_uses_tool(self) -> None:
        state = state_with_index("做 2D MOSFET")
        action = choose_next_action(state_with_index("做 2D MOSFET Id-Vg"))
        fake_result = Mock()
        fake_result.model_dump.return_value = {"status": "completed", "run_id": "mos_unit"}

        with patch("tcad_agent.supervisor.run_mosfet_2d_id_sweep", return_value=fake_result) as runner:
            completed = execute_action(action, state)

        self.assertEqual(completed.status, SupervisorActionStatus.COMPLETED)
        self.assertEqual(completed.result["status"], "completed")
        self.assertEqual(completed.result["run_id"], "mos_unit")
        self.assertEqual(completed.result["tcad_deck_spec"]["device_family"], "2d_mosfet")
        runner.assert_called_once()

    def test_execute_mosfet_action_accepts_dict_tool_result(self) -> None:
        state = state_with_index("做 2D MOSFET")
        action = choose_next_action(state_with_index("做 2D MOSFET Id-Vg"))

        with patch(
            "tcad_agent.supervisor.run_mosfet_2d_id_sweep",
            return_value={"status": "completed", "run_id": "mos_dict_unit", "quality_report": {"status": "suspicious"}},
        ) as runner:
            completed = execute_action(action, state)

        self.assertEqual(completed.status, SupervisorActionStatus.COMPLETED)
        self.assertEqual(completed.result["run_id"], "mos_dict_unit")
        self.assertEqual(completed.result["quality_report"]["status"], "suspicious")
        self.assertIsNone(completed.error)
        runner.assert_called_once()

    def test_execute_mesh_convergence_action_uses_tool(self) -> None:
        state = state_with_index("做 mesh convergence")
        action = choose_next_action(state_with_index("对 PN IV 做 mesh convergence"))
        fake_result = Mock()
        fake_result.model_dump.return_value = {"status": "completed", "convergence_id": "mesh_unit"}

        with patch("tcad_agent.supervisor.run_mesh_convergence", return_value=fake_result) as runner:
            completed = execute_action(action, state)

        self.assertEqual(completed.status, SupervisorActionStatus.COMPLETED)
        self.assertEqual(completed.result, {"status": "completed", "convergence_id": "mesh_unit"})
        runner.assert_called_once()

    def test_execute_extended_device_action_uses_tool(self) -> None:
        state = state_with_index("做 Schottky diode")
        action = choose_next_action(state_with_index("做 Schottky diode forward IV"))
        fake_result = Mock()
        fake_result.model_dump.return_value = {"status": "completed", "run_id": "schottky_unit"}

        with patch("tcad_agent.supervisor.run_extended_device_sweep", return_value=fake_result) as runner:
            completed = execute_action(action, state)

        self.assertEqual(completed.status, SupervisorActionStatus.COMPLETED)
        self.assertEqual(completed.result["status"], "completed")
        self.assertEqual(completed.result["run_id"], "schottky_unit")
        self.assertIn("tcad_deck_spec", completed.result)
        runner.assert_called_once()

    def test_execute_schottky_calibration_action_uses_tool(self) -> None:
        state = state_with_index("校准 Schottky diode")
        action = choose_next_action(state_with_index("校准 Schottky diode 到可信曲线"))
        fake_result = Mock()
        fake_result.model_dump.return_value = {"status": "completed", "calibration_id": "cal_unit"}

        with patch("tcad_agent.supervisor.run_schottky_calibration", return_value=fake_result) as runner:
            completed = execute_action(action, state)

        self.assertEqual(completed.status, SupervisorActionStatus.COMPLETED)
        self.assertEqual(completed.result["status"], "completed")
        self.assertEqual(completed.result["calibration_id"], "cal_unit")
        self.assertIn("tcad_deck_spec", completed.result)
        runner.assert_called_once()

    def test_execute_conclusion_action_uses_tool(self) -> None:
        state = state_with_index("生成工程结论")
        action = SupervisorAction(
            index=1,
            kind=SupervisorActionKind.GENERATE_CONCLUSION,
            status=SupervisorActionStatus.PLANNED,
            reason="unit",
            request={"state": "/tmp/unit/state.json"},
            created_at=utc_timestamp(),
            updated_at=utc_timestamp(),
        )
        fake_result = Mock()
        fake_result.model_dump.return_value = {"status": "completed", "conclusion_path": "/tmp/unit/conclusion.md"}

        with patch("tcad_agent.supervisor.generate_experiment_conclusion", return_value=fake_result) as runner:
            completed = execute_action(action, state)

        self.assertEqual(completed.status, SupervisorActionStatus.COMPLETED)
        self.assertEqual(completed.result, {"status": "completed", "conclusion_path": "/tmp/unit/conclusion.md"})
        runner.assert_called_once_with(Path("/tmp/unit/state.json"))

    def test_execute_repair_plan_action_uses_tool(self) -> None:
        state = state_with_index("生成修复策略")
        action = SupervisorAction(
            index=1,
            kind=SupervisorActionKind.GENERATE_REPAIR_PLAN,
            status=SupervisorActionStatus.PLANNED,
            reason="unit",
            request={"state": "/tmp/unit/state.json"},
            created_at=utc_timestamp(),
            updated_at=utc_timestamp(),
        )
        fake_result = Mock()
        fake_result.model_dump.return_value = {"status": "planned", "next_action": "continuation_bias_ramp"}

        with patch("tcad_agent.supervisor.build_repair_plan", return_value=fake_result) as runner:
            completed = execute_action(action, state)

        self.assertEqual(completed.status, SupervisorActionStatus.COMPLETED)
        self.assertEqual(completed.result, {"status": "planned", "next_action": "continuation_bias_ramp"})
        runner.assert_called_once_with(Path("/tmp/unit/state.json"))


if __name__ == "__main__":
    unittest.main()
