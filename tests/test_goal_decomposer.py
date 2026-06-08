from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.goal_decomposer import (
    DecompositionStatus,
    GoalStepKind,
    decompose_goal_with_llm,
    deterministic_decompose_goal,
    replan_goal_after_issue,
    write_decomposition_result,
)
from tcad_agent.llm import LLMConfig


class FakeClient:
    config = LLMConfig(model="fake-goal")

    def __init__(self, response: str | Exception) -> None:
        self.response = response

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class GoalDecomposerTest(unittest.TestCase):
    def test_deterministic_plan_includes_convergence_repair_and_conclusion(self) -> None:
        result = deterministic_decompose_goal("做 MOSFET Id-Vg，并做 mesh convergence，最后给工程结论")
        kinds = [step.kind for step in result.steps]

        self.assertEqual(result.status, DecompositionStatus.COMPLETED)
        self.assertIn(GoalStepKind.RUN_SUPERVISOR, kinds)
        self.assertIn(GoalStepKind.RUN_TOOL_CONVERGENCE, kinds)
        self.assertIn(GoalStepKind.RUN_PHYSICAL_BENCHMARK, kinds)
        self.assertIn(GoalStepKind.RUN_REPAIR_EXECUTOR, kinds)
        self.assertEqual(result.steps[-1].kind, GoalStepKind.GENERATE_CONCLUSION)
        self.assertIn("engineering_intent", result.steps[0].request)

    def test_llm_plan_success(self) -> None:
        response = json.dumps(
            {
                "plan_id": "llm_plan",
                "steps": [
                    {
                        "index": 1,
                        "kind": "run_supervisor",
                        "title": "Run MOSFET",
                        "request": {"goal_text": "MOSFET Id-Vg"},
                    },
                    {
                        "index": 2,
                        "kind": "generate_conclusion",
                        "title": "Conclusion",
                        "depends_on": [1],
                    },
                ],
            }
        )

        result = decompose_goal_with_llm("MOSFET Id-Vg", client=FakeClient(response))

        self.assertEqual(result.status, DecompositionStatus.COMPLETED)
        self.assertEqual(result.model, "fake-goal")
        self.assertEqual(result.plan_id, "llm_plan")
        self.assertEqual(len(result.steps), 2)

    def test_llm_tool_convergence_request_is_normalized(self) -> None:
        response = json.dumps(
            {
                "steps": [
                    {
                        "index": 1,
                        "kind": "run_supervisor",
                        "title": "Run output curves",
                        "request": {"goal_text": "2D NMOS output characteristic"},
                    },
                    {
                        "index": 2,
                        "kind": "run_tool_convergence",
                        "title": "Check mesh",
                        "request": {
                            "tool_name": "mosfet_2d_id_sweep",
                            "base_request": {
                                "sweep_type": "output_characteristic",
                                "gate_values": [0.8, 1.0, 1.2],
                                "drain_start": 0.0,
                                "drain_stop": 1.2,
                                "drain_step": 0.05,
                            },
                            "axis_path": "mesh_refinement_level",
                            "values": [1, 2, 3],
                            "metric_path": "simulation_results.id_saturation",
                        },
                        "depends_on": [1],
                        "stop_on_failure": False,
                    },
                ]
            }
        )

        result = decompose_goal_with_llm("2D NMOS output characteristic", client=FakeClient(response))
        convergence = result.steps[1]

        self.assertEqual(convergence.request["base_request"]["sweep_type"], "idvd")
        self.assertEqual(convergence.request["base_request"]["idvd_gate_voltage"], 1.2)
        self.assertEqual(convergence.request["axis_path"], "x_divisions")
        self.assertEqual(convergence.request["values"], [8, 12, 16])
        self.assertIn("已归一化", " ".join(result.warnings))

    def test_llm_replan_returns_chinese_strategy_and_plan_patch(self) -> None:
        response = json.dumps(
            {
                "strategy_zh": "字段别名已经归一化，把可选收敛检查标记为非阻塞，继续生成结论。",
                "mark_soft_failed": [2],
                "append_steps": [
                    {
                        "index": 1,
                        "kind": "generate_conclusion",
                        "title": "Write conclusion with warning",
                        "request": {},
                        "depends_on": [2],
                        "stop_on_failure": False,
                    }
                ],
            }
        )

        decision = replan_goal_after_issue(
            "做 MOSFET output characteristic",
            current_plan={"steps": []},
            goal_step_statuses={"2": {"status": "failed"}},
            issue_context={"failure_reason": "sweep_type output_characteristic is invalid"},
            current_evidence={"state_path": "/tmp/current/state.json"},
            client=FakeClient(response),
        )

        self.assertEqual(decision.status, DecompositionStatus.COMPLETED)
        self.assertIn("归一化", decision.strategy_zh)
        self.assertEqual(decision.mark_soft_failed, [2])
        self.assertEqual(decision.append_steps[0].kind, GoalStepKind.GENERATE_CONCLUSION)

    def test_extended_device_template_runs_through_supervisor(self) -> None:
        result = deterministic_decompose_goal("做 Schottky diode IV 并提取 barrier height")
        kinds = [step.kind for step in result.steps]

        self.assertEqual(result.status, DecompositionStatus.COMPLETED)
        self.assertIn(GoalStepKind.RUN_SUPERVISOR, kinds)
        self.assertEqual(result.steps[-1].kind, GoalStepKind.GENERATE_CONCLUSION)

    def test_schottky_calibration_convergence_uses_calibration_tool(self) -> None:
        result = deterministic_decompose_goal("校准 Schottky diode 到可信曲线，并做 convergence，最后给结论")
        convergence_steps = [step for step in result.steps if step.kind == GoalStepKind.RUN_TOOL_CONVERGENCE]

        self.assertEqual(len(convergence_steps), 1)
        self.assertEqual(convergence_steps[0].request["tool_name"], "schottky_iv_calibration")
        self.assertEqual(convergence_steps[0].request["axis_path"], "step")

    def test_measured_curve_goal_adds_golden_comparison_step(self) -> None:
        result = deterministic_decompose_goal("做 MOSFET Id-Vg，measured_curve target.csv，最后给结论")
        golden_steps = [step for step in result.steps if step.kind == GoalStepKind.RUN_GOLDEN_COMPARISON]

        self.assertEqual(len(golden_steps), 1)
        self.assertEqual(golden_steps[0].request["reference_curve_path"], "target.csv")
        self.assertIn(GoalStepKind.RUN_PHYSICAL_BENCHMARK, [step.kind for step in result.steps])

    def test_output_characteristic_signoff_uses_idvd_convergence(self) -> None:
        result = deterministic_decompose_goal("客户说 NMOS output characteristic 有 kink，做 signoff 并给工程结论")
        convergence_steps = [step for step in result.steps if step.kind == GoalStepKind.RUN_TOOL_CONVERGENCE]

        self.assertEqual(len(convergence_steps), 1)
        self.assertEqual(convergence_steps[0].request["tool_name"], "mosfet_2d_id_sweep")
        self.assertEqual(convergence_steps[0].request["base_request"]["sweep_type"], "idvd")
        self.assertEqual(convergence_steps[0].request["metric_path"], "quality_report.metrics.idvd_final_current_a")

    def test_dibl_goal_uses_drain_voltage_split(self) -> None:
        result = deterministic_decompose_goal("短沟道 NMOS 我担心 DIBL，低 Vd 和高 Vd 各跑 Id-Vg，Vg 到 1.2V")
        convergence = next(step for step in result.steps if step.kind == GoalStepKind.RUN_TOOL_CONVERGENCE)

        self.assertEqual(convergence.request["tool_name"], "mosfet_2d_id_sweep")
        self.assertEqual(convergence.request["base_request"]["sweep_type"], "idvg")
        self.assertEqual(convergence.request["axis_path"], "drain_voltage")
        self.assertEqual(convergence.request["values"], [0.05, 1.0])

    def test_mobility_ab_goal_uses_mobility_model_split(self) -> None:
        result = deterministic_decompose_goal("NMOS mobility model A/B，对比 constant mobility 和 doping-dependent mobility")
        convergence = next(step for step in result.steps if step.kind == GoalStepKind.RUN_TOOL_CONVERGENCE)

        self.assertEqual(convergence.request["axis_path"], "mobility_model")
        self.assertEqual(convergence.request["values"], ["constant", "doping_dependent"])

    def test_moscap_tox_qf_goal_uses_moscap_split(self) -> None:
        result = deterministic_decompose_goal("MOSCAP 不确定是 tox 偏厚还是 Qf 固定电荷，做 C-V sanity")
        convergence = next(step for step in result.steps if step.kind == GoalStepKind.RUN_TOOL_CONVERGENCE)

        self.assertEqual(convergence.request["tool_name"], "mos_capacitor_cv_sweep")
        self.assertEqual(convergence.request["axis_path"], "oxide_thickness_nm")

    def test_diode_temperature_corner_uses_temperature_split(self) -> None:
        result = deterministic_decompose_goal("PN diode 高温 leakage temperature corner，先看 300K/350K/400K")
        convergence = next(step for step in result.steps if step.kind == GoalStepKind.RUN_TOOL_CONVERGENCE)

        self.assertEqual(convergence.request["tool_name"], "diode_breakdown_leakage_sweep")
        self.assertEqual(convergence.request["axis_path"], "temperature_k")

    def test_advanced_industrial_template_routes_to_physics_runner(self) -> None:
        result = deterministic_decompose_goal("GaN HEMT 输出特性和 current collapse 风险，帮我扫栅压和漏压")

        self.assertIn(GoalStepKind.RUN_SUPERVISOR, [step.kind for step in result.steps])
        self.assertIn(GoalStepKind.RUN_TOOL_CONVERGENCE, [step.kind for step in result.steps])
        primary = next(step for step in result.steps if step.kind == GoalStepKind.RUN_SUPERVISOR)
        convergence = next(step for step in result.steps if step.kind == GoalStepKind.RUN_TOOL_CONVERGENCE)
        self.assertEqual(primary.request["request_hint"]["device_type"], "gan_hemt_id_bv")
        self.assertEqual(primary.request["request_hint"]["fidelity"], "physics_1d")
        self.assertEqual(convergence.request["base_request"]["fidelity"], "physics_1d")
        self.assertEqual(convergence.request["axis_path"], "gan_2deg_density_cm2")
        self.assertFalse(any("尚未实现" in warning for warning in result.warnings))

    def test_abstract_goal_asks_for_required_tcad_details(self) -> None:
        result = deterministic_decompose_goal("TCAD仿真工程师通过自然语言描述完成工作，需要agent长时间、自动解决遇到的问题")

        self.assertEqual(result.steps[0].kind, GoalStepKind.ASK_USER)
        self.assertGreaterEqual(len(result.steps[0].request["questions"]), 2)
        self.assertIn("澄清", result.warnings[0])

    def test_power_mosfet_goal_routes_to_physics_path_without_compact_warning(self) -> None:
        result = deterministic_decompose_goal("做 power MOSFET BV 和 Ron tradeoff，最后给工程结论")

        self.assertIn(GoalStepKind.RUN_SUPERVISOR, [step.kind for step in result.steps])
        primary = next(step for step in result.steps if step.kind == GoalStepKind.RUN_SUPERVISOR)
        self.assertEqual(primary.request["request_hint"]["fidelity"], "physics_1d")
        self.assertFalse(any("compact baseline" in warning for warning in result.warnings))

    def test_structure_edit_goal_creates_deck_mutation_sweeps(self) -> None:
        result = deterministic_decompose_goal("这个结构漏电偏高，改 field plate / drift doping / lifetime 看看")
        primary = next(step for step in result.steps if step.kind == GoalStepKind.RUN_SUPERVISOR)
        convergence_steps = [step for step in result.steps if step.kind == GoalStepKind.RUN_TOOL_CONVERGENCE]

        self.assertEqual(primary.request["engineering_intent"]["device_family"], "power_mosfet")
        self.assertEqual(primary.request["request_hint"]["device_type"], "power_mosfet_bv_ron")
        self.assertEqual(len(convergence_steps), 3)
        self.assertEqual(
            {step.request["axis_path"] for step in convergence_steps},
            {
                "power_mos_field_plate_length_um",
                "power_mos_drift_region_doping_cm3",
                "power_mos_carrier_lifetime_s",
            },
        )
        self.assertTrue(all(step.request["base_request"].get("tcad_deck_mutations") for step in convergence_steps))

    def test_deterministic_replan_classifies_schema_alias_issue(self) -> None:
        decision = replan_goal_after_issue(
            "做 MOSFET output characteristic",
            current_plan={"steps": []},
            goal_step_statuses={"2": {"status": "failed"}},
            issue_context={"failure_reason": "sweep_type output_characteristic string_pattern_mismatch"},
            client=FakeClient(RuntimeError("offline")),
        )

        self.assertEqual(decision.status, DecompositionStatus.FALLBACK)
        self.assertEqual(decision.issue_family, "schema_or_field_alias")
        self.assertIn("normalize_tool_fields", decision.recommended_actions)

    def test_llm_invalid_json_falls_back(self) -> None:
        result = decompose_goal_with_llm("PN IV", client=FakeClient("not json"))

        self.assertEqual(result.status, DecompositionStatus.FALLBACK)
        self.assertTrue(result.fallback_used)
        self.assertGreaterEqual(len(result.steps), 2)

    def test_write_decomposition_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "goal.json"
            result = deterministic_decompose_goal("PN IV")

            write_decomposition_result(result, output)

            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
