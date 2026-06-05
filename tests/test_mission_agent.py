from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tcad_agent.llm import LLMConfig
from tcad_agent.mission_agent import MissionStatus, MissionStepKind, run_mission_agent


class FakeLLMClient:
    config = LLMConfig(model="fake-mission-planner")

    def __init__(self, response: str | Exception) -> None:
        self.response = response

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class SequenceLLMClient:
    config = LLMConfig(model="fake-sequence-planner")

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.users: list[str] = []

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        self.calls += 1
        self.users.append(user)
        if not self.responses:
            raise RuntimeError("no fake response left")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class MissionAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def fake_supervisor_state(self, tool_result: dict[str, object] | None = None) -> Mock:
        result = Mock()
        payload: dict[str, object] = {
            "status": "completed",
            "supervisor_id": "sup_unit",
            "supervisor_dir": str(self.root / "supervisor" / "sup_unit"),
        }
        if tool_result:
            last_action = {
                "index": 1,
                "kind": "run_schottky_calibration",
                "status": "completed",
                "result": tool_result,
            }
            payload["actions"] = [last_action]
            payload["checkpoint"] = {"last_action": last_action}
        result.model_dump.return_value = payload
        return result

    def fake_conclusion(self) -> Mock:
        result = Mock()
        result.model_dump.return_value = {
            "status": "completed",
            "conclusion_path": str(self.root / "state" / "conclusion.md"),
        }
        return result

    def fake_repair(self, requires_confirmation: bool) -> Mock:
        result = Mock()
        result.model_dump.return_value = {
            "status": "planned",
            "output_path": str(self.root / "state" / "repair_plan.json"),
            "next_action": "geometry_sanity_repair" if requires_confirmation else "continuation_bias_ramp",
            "actions": [
                {
                    "name": "geometry_sanity_repair" if requires_confirmation else "continuation_bias_ramp",
                    "user_confirmation_required": requires_confirmation,
                }
            ],
        }
        return result

    def fake_repair_execution(self) -> Mock:
        result = Mock()
        result.model_dump.return_value = {
            "status": "completed",
            "execution_dir": str(self.root / "repair_exec"),
            "final_state_path": str(self.root / "state" / "repaired_state.json"),
            "current_state_path": str(self.root / "state" / "repaired_state.json"),
            "final_quality_status": "passed",
        }
        return result

    def fake_tool_convergence(self) -> Mock:
        result = Mock()
        result.model_dump.return_value = {
            "status": "completed",
            "convergence_id": "conv_unit",
            "convergence_dir": str(self.root / "tool_convergence" / "conv_unit"),
            "quality_report": {
                "status": "passed",
                "metrics": {"relative_delta": 0.0},
            },
        }
        return result

    def fake_benchmark(self, status: str = "passed") -> Mock:
        result = Mock()
        result.model_dump.return_value = {
            "tool_name": "physical_benchmark",
            "status": status,
            "source_state_path": str(self.root / "state" / "state.json"),
            "benchmark_path": str(self.root / "state" / "benchmark.json"),
            "checks": [],
            "summary": {"counts": {"pass": 1, "warning": 0, "error": 0}},
        }
        return result

    def fake_failed_tool_convergence(self) -> Mock:
        result = Mock()
        result.model_dump.return_value = {
            "status": "failed",
            "convergence_id": "conv_failed",
            "convergence_dir": str(self.root / "tool_convergence" / "conv_failed"),
            "quality_report": {
                "status": "failed",
                "issues": [{"code": "too_few_completed_convergence_cases", "severity": "error"}],
                "metrics": {"cases": 3, "completed_cases": 0},
                "recommended_next_action": "rerun failed convergence cases before trusting the result",
            },
            "failure_reason": "validation failed",
        }
        return result

    def test_plan_only_writes_rebuild_step(self) -> None:
        state = run_mission_agent(
            "完成一个 MOSFET 任务",
            mission_id="mission_plan",
            mission_root=self.root,
            execute=False,
        )

        self.assertEqual(state.status, MissionStatus.PLANNED)
        self.assertEqual(state.steps[0].kind, MissionStepKind.REBUILD_INDEX)
        self.assertTrue((self.root / "mission_plan" / "mission_state.json").exists())

    def test_plan_only_can_use_llm_decomposition(self) -> None:
        response = json.dumps(
            {
                "plan_id": "mission_llm",
                "steps": [
                    {
                        "index": 1,
                        "kind": "query_history",
                        "title": "Inspect history",
                        "request": {"limit": 5},
                        "stop_on_failure": False,
                    },
                    {
                        "index": 2,
                        "kind": "run_supervisor",
                        "title": "Run TCAD",
                        "request": {"goal_text": "MOSFET Id-Vg"},
                        "depends_on": [1],
                    },
                ],
            }
        )

        state = run_mission_agent(
            "先看历史再做 MOSFET",
            mission_id="mission_llm",
            mission_root=self.root,
            execute=False,
            use_llm_decomposer=True,
            llm_client=FakeLLMClient(response),
        )

        self.assertEqual(state.status, MissionStatus.PLANNED)
        self.assertEqual(state.checkpoint["goal_decomposer"], "llm")
        self.assertEqual(state.checkpoint["goal_decomposer_model"], "fake-mission-planner")
        self.assertFalse(state.checkpoint["goal_decomposer_fallback_used"])
        self.assertEqual(state.checkpoint["goal_decomposition"]["steps"][0]["kind"], "query_history")

    def test_mission_state_is_written_before_llm_decomposition_returns(self) -> None:
        mission_id = "mission_probe"
        state_path = self.root / mission_id / "mission_state.json"
        test_case = self

        class InspectingLLMClient(FakeLLMClient):
            def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
                test_case.assertTrue(state_path.exists())
                snapshot = json.loads(state_path.read_text(encoding="utf-8"))
                test_case.assertEqual(snapshot["checkpoint"]["goal_decomposition_status"], "running")
                test_case.assertEqual(snapshot["next_action"], "decompose goal")
                return super().chat(system, user, temperature)

        response = json.dumps(
            {
                "steps": [
                    {
                        "index": 1,
                        "kind": "run_supervisor",
                        "title": "Run MOSFET",
                        "reason": "execute task",
                        "tool_name": "supervisor",
                        "request": {},
                    }
                ]
            }
        )

        state = run_mission_agent(
            "做 MOSFET Id-Vg",
            mission_id=mission_id,
            mission_root=self.root,
            execute=False,
            use_llm_decomposer=True,
            llm_client=InspectingLLMClient(response),
        )

        self.assertEqual(state.checkpoint["goal_decomposition_status"], "completed")
        self.assertEqual(state.checkpoint["goal_decomposition"]["steps"][0]["kind"], "run_supervisor")

    def test_llm_decomposition_falls_back_by_default(self) -> None:
        state = run_mission_agent(
            "完成一个 MOSFET 任务",
            mission_id="mission_llm_fallback",
            mission_root=self.root,
            execute=False,
            use_llm_decomposer=True,
            llm_client=FakeLLMClient(RuntimeError("planner offline")),
        )

        self.assertEqual(state.status, MissionStatus.PLANNED)
        self.assertTrue(state.checkpoint["goal_decomposer_fallback_used"])
        self.assertEqual(state.checkpoint["goal_decomposition"]["status"], "fallback")

    def test_llm_decomposition_can_fail_without_fallback(self) -> None:
        state = run_mission_agent(
            "完成一个 MOSFET 任务",
            mission_id="mission_llm_strict",
            mission_root=self.root,
            execute=True,
            use_llm_decomposer=True,
            allow_llm_fallback=False,
            llm_client=FakeLLMClient(RuntimeError("planner offline")),
        )

        self.assertEqual(state.status, MissionStatus.FAILED)
        self.assertEqual(state.failure_reason, "goal decomposition failed")
        self.assertEqual(state.steps, [])

    def test_execute_generates_conclusion_for_passed_result(self) -> None:
        state_path = str(self.root / "state" / "state.json")
        recent_record = {
            "experiment_id": "mos_ok",
            "kind": "mosfet_2d_id_sweep",
            "status": "completed",
            "quality_status": "passed",
            "state_path": state_path,
        }
        tool_result = {
            "tool_name": "mosfet_2d_id_sweep",
            "run_id": "mos_ok",
            "status": "completed",
            "state_path": state_path,
            "quality_report": {"status": "passed"},
        }

        with (
            patch("tcad_agent.mission_agent.rebuild_index", return_value={"records_indexed": 1}),
            patch("tcad_agent.mission_agent.list_records", side_effect=[[], [recent_record]]),
            patch("tcad_agent.mission_agent.run_supervisor", return_value=self.fake_supervisor_state(tool_result)),
            patch("tcad_agent.mission_agent.run_physical_benchmark", return_value=self.fake_benchmark()),
            patch("tcad_agent.mission_agent.generate_experiment_conclusion", return_value=self.fake_conclusion()),
        ):
            state = run_mission_agent(
                "完成一个 MOSFET 任务并给结论",
                mission_id="mission_conclusion",
                mission_root=self.root,
                execute=True,
                max_cycles=6,
            )

        self.assertEqual(state.status, MissionStatus.COMPLETED)
        self.assertEqual([step.kind for step in state.steps], [
            MissionStepKind.REBUILD_INDEX,
            MissionStepKind.RUN_SUPERVISOR,
            MissionStepKind.REBUILD_INDEX,
            MissionStepKind.RUN_PHYSICAL_BENCHMARK,
            MissionStepKind.SKIP_GOAL_STEP,
            MissionStepKind.GENERATE_CONCLUSION,
        ])
        self.assertIn("conclusion_path", state.checkpoint)
        self.assertEqual(state.checkpoint["goal_step_statuses"]["2"]["status"], "completed")
        self.assertEqual(state.checkpoint["goal_step_statuses"]["3"]["status"], "skipped")
        self.assertEqual(state.checkpoint["goal_step_statuses"]["4"]["status"], "completed")
        self.assertEqual(len(state.checkpoint["controller_cycles"]), len(state.steps))
        last_decision = state.checkpoint["controller_cycles"][-1]["decision"]
        self.assertEqual(last_decision["action"], "finish")
        self.assertIn("工程结论", last_decision["reason_zh"])

    def test_primary_supervisor_without_tcad_result_does_not_use_global_history(self) -> None:
        stale_record = {
            "experiment_id": "stale_mos_bad",
            "kind": "mosfet_2d_id_sweep",
            "status": "completed",
            "quality_status": "suspicious",
            "state_path": str(self.root / "stale" / "state.json"),
        }

        with (
            patch("tcad_agent.mission_agent.rebuild_index", return_value={"records_indexed": 1}),
            patch("tcad_agent.mission_agent.list_records", side_effect=[[], [stale_record]]),
            patch("tcad_agent.mission_agent.run_supervisor", return_value=self.fake_supervisor_state()),
            patch("tcad_agent.mission_agent.build_repair_plan") as repair_plan,
            patch("tcad_agent.mission_agent.generate_experiment_conclusion") as conclusion,
        ):
            state = run_mission_agent(
                "完成一个 MOSFET 任务并给结论",
                mission_id="mission_no_current_tcad",
                mission_root=self.root,
                execute=True,
                max_cycles=6,
            )

        self.assertEqual(state.status, MissionStatus.WAITING_FOR_USER)
        self.assertEqual(state.steps[-1].kind, MissionStepKind.ASK_USER)
        self.assertIn("物理 benchmark", state.steps[-1].request["question"])
        repair_plan.assert_not_called()
        conclusion.assert_not_called()

    def test_execute_consumes_query_history_goal_step(self) -> None:
        state_path = str(self.root / "state" / "state.json")
        recent_record = {
            "experiment_id": "mos_ok",
            "kind": "mosfet_2d_id_sweep",
            "status": "completed",
            "quality_status": "passed",
            "state_path": state_path,
        }
        tool_result = {
            "tool_name": "mosfet_2d_id_sweep",
            "run_id": "mos_ok",
            "status": "completed",
            "state_path": state_path,
            "quality_report": {"status": "passed"},
        }

        with (
            patch("tcad_agent.mission_agent.rebuild_index", return_value={"records_indexed": 1}),
            patch("tcad_agent.mission_agent.list_records", side_effect=[[], [recent_record], [recent_record]]),
            patch("tcad_agent.mission_agent.run_supervisor", return_value=self.fake_supervisor_state(tool_result)),
            patch("tcad_agent.mission_agent.run_physical_benchmark", return_value=self.fake_benchmark()),
            patch("tcad_agent.mission_agent.generate_experiment_conclusion", return_value=self.fake_conclusion()),
        ):
            state = run_mission_agent(
                "先检索历史，再完成一个 MOSFET 任务并给结论",
                mission_id="mission_history_dag",
                mission_root=self.root,
                execute=True,
                max_cycles=7,
            )

        self.assertEqual(state.status, MissionStatus.COMPLETED)
        self.assertEqual(state.steps[1].kind, MissionStepKind.QUERY_HISTORY)
        self.assertEqual(state.checkpoint["goal_step_statuses"]["1"]["kind"], "query_history")
        self.assertEqual(state.checkpoint["goal_step_statuses"]["1"]["status"], "completed")

    def test_execute_waits_for_confirmation_after_sensitive_repair(self) -> None:
        state_path = str(self.root / "state" / "state.json")
        recent_record = {
            "experiment_id": "pn_bad",
            "kind": "pn_junction_iv_sweep",
            "status": "completed",
            "quality_status": "suspicious",
            "state_path": state_path,
        }
        tool_result = {
            "tool_name": "pn_junction_iv_sweep",
            "run_id": "pn_bad",
            "status": "completed",
            "state_path": state_path,
            "quality_report": {"status": "suspicious"},
        }

        with (
            patch("tcad_agent.mission_agent.rebuild_index", return_value={"records_indexed": 1}),
            patch("tcad_agent.mission_agent.list_records", side_effect=[[], [recent_record]]),
            patch("tcad_agent.mission_agent.run_supervisor", return_value=self.fake_supervisor_state(tool_result)),
            patch("tcad_agent.mission_agent.run_physical_benchmark", return_value=self.fake_benchmark("suspicious")),
            patch("tcad_agent.mission_agent.build_repair_plan", return_value=self.fake_repair(True)),
        ):
            state = run_mission_agent(
                "完成 PN 任务，失败后自动换策略",
                mission_id="mission_repair",
                mission_root=self.root,
                execute=True,
                max_cycles=6,
            )

        self.assertEqual(state.status, MissionStatus.WAITING_FOR_USER)
        self.assertEqual(state.steps[-1].kind, MissionStepKind.ASK_USER)
        self.assertTrue(state.checkpoint["repair_requires_confirmation"])

    def test_execute_runs_non_sensitive_repair_before_conclusion(self) -> None:
        state_path = str(self.root / "state" / "state.json")
        recent_record = {
            "experiment_id": "mos_bad",
            "kind": "mosfet_2d_id_sweep",
            "status": "completed",
            "quality_status": "suspicious",
            "state_path": state_path,
        }
        tool_result = {
            "tool_name": "mosfet_2d_id_sweep",
            "run_id": "mos_bad",
            "status": "completed",
            "state_path": state_path,
            "quality_report": {"status": "suspicious"},
        }

        with (
            patch("tcad_agent.mission_agent.rebuild_index", return_value={"records_indexed": 1}),
            patch("tcad_agent.mission_agent.list_records", side_effect=[[], [recent_record]]),
            patch("tcad_agent.mission_agent.run_supervisor", return_value=self.fake_supervisor_state(tool_result)),
            patch("tcad_agent.mission_agent.run_physical_benchmark", return_value=self.fake_benchmark("suspicious")),
            patch("tcad_agent.mission_agent.build_repair_plan", return_value=self.fake_repair(False)),
            patch("tcad_agent.mission_agent.run_repair_executor", return_value=self.fake_repair_execution()),
            patch("tcad_agent.mission_agent.generate_experiment_conclusion", return_value=self.fake_conclusion()) as conclusion,
        ):
            state = run_mission_agent(
                "完成 MOSFET 任务，自动修复后给结论",
                mission_id="mission_auto_repair",
                mission_root=self.root,
                execute=True,
                max_cycles=7,
            )

        self.assertEqual(state.status, MissionStatus.COMPLETED)
        self.assertIn(MissionStepKind.EXECUTE_REPAIR, [step.kind for step in state.steps])
        self.assertEqual(state.checkpoint["repaired_state_path"], str(self.root / "state" / "repaired_state.json"))
        conclusion.assert_called_once_with(Path(self.root / "state" / "repaired_state.json"))

    def test_repair_step_prefers_current_supervisor_record_over_global_history(self) -> None:
        stale_record = {
            "experiment_id": "old_mos_bad",
            "kind": "mosfet_2d_id_sweep",
            "status": "completed",
            "quality_status": "suspicious",
            "state_path": str(self.root / "state" / "old_mos_state.json"),
        }
        current_state_path = self.root / "state" / "current_schottky_state.json"
        plan = json.dumps(
            {
                "steps": [
                    {"index": 1, "kind": "run_supervisor", "title": "Run current TCAD", "request": {}, "depends_on": []},
                    {
                        "index": 2,
                        "kind": "run_repair_executor",
                        "title": "Repair only if current result needs it",
                        "request": {},
                        "depends_on": [1],
                    },
                    {"index": 3, "kind": "generate_conclusion", "title": "Conclude", "request": {}, "depends_on": [2]},
                ]
            }
        )
        tool_result = {
            "tool_name": "schottky_iv_calibration",
            "status": "completed",
            "calibration_id": "current_schottky",
            "quality_report": {"status": "passed"},
            "final_summary": {"artifacts": {"state": str(current_state_path)}},
        }

        with (
            patch("tcad_agent.mission_agent.rebuild_index", return_value={"records_indexed": 1}),
            patch("tcad_agent.mission_agent.list_records", side_effect=[[], [stale_record]]),
            patch("tcad_agent.mission_agent.run_supervisor", return_value=self.fake_supervisor_state(tool_result)),
            patch("tcad_agent.mission_agent.generate_experiment_conclusion", return_value=self.fake_conclusion()) as conclusion,
            patch("tcad_agent.mission_agent.build_repair_plan") as repair_plan,
        ):
            state = run_mission_agent(
                "校准 Schottky 并给结论",
                mission_id="mission_current_record",
                mission_root=self.root,
                execute=True,
                max_cycles=6,
                use_llm_decomposer=True,
                llm_client=FakeLLMClient(plan),
            )

        self.assertEqual(state.status, MissionStatus.COMPLETED)
        self.assertEqual(state.checkpoint["primary_tcad_record"]["state_path"], str(current_state_path))
        self.assertIn(MissionStepKind.SKIP_GOAL_STEP, [step.kind for step in state.steps])
        repair_plan.assert_not_called()
        conclusion.assert_called_once_with(current_state_path)

    def test_execute_goal_decomposition_tool_convergence_before_conclusion(self) -> None:
        calibration_record = {
            "experiment_id": "schottky_cal",
            "kind": "schottky_iv_calibration",
            "status": "completed",
            "quality_status": "passed",
            "state_path": str(self.root / "state" / "schottky_cal_state.json"),
        }
        convergence_record = {
            "experiment_id": "schottky_conv",
            "kind": "tool_convergence",
            "status": "completed",
            "quality_status": "passed",
            "state_path": str(self.root / "state" / "tool_convergence_state.json"),
        }

        with (
            patch("tcad_agent.mission_agent.rebuild_index", return_value={"records_indexed": 1}),
            patch("tcad_agent.mission_agent.list_records", side_effect=[[], [calibration_record], [calibration_record, convergence_record]]),
            patch("tcad_agent.mission_agent.run_supervisor", return_value=self.fake_supervisor_state()),
            patch("tcad_agent.mission_agent.run_tool_convergence", return_value=self.fake_tool_convergence()) as convergence,
            patch("tcad_agent.mission_agent.run_physical_benchmark", return_value=self.fake_benchmark()),
            patch("tcad_agent.mission_agent.generate_experiment_conclusion", return_value=self.fake_conclusion()) as conclusion,
        ):
            state = run_mission_agent(
                "校准 Schottky diode 到可信曲线，并做 convergence，最后给结论",
                mission_id="mission_schottky_convergence",
                mission_root=self.root,
                execute=True,
                max_cycles=8,
            )

        self.assertEqual(state.status, MissionStatus.COMPLETED)
        self.assertEqual(
            [step.kind for step in state.steps],
            [
                MissionStepKind.REBUILD_INDEX,
                MissionStepKind.RUN_SUPERVISOR,
                MissionStepKind.REBUILD_INDEX,
                MissionStepKind.RUN_TOOL_CONVERGENCE,
                MissionStepKind.REBUILD_INDEX,
                MissionStepKind.RUN_PHYSICAL_BENCHMARK,
                MissionStepKind.SKIP_GOAL_STEP,
                MissionStepKind.GENERATE_CONCLUSION,
            ],
        )
        self.assertEqual(state.checkpoint["tool_convergence_quality_status"], "passed")
        self.assertEqual(state.checkpoint["goal_step_statuses"]["2"]["status"], "completed")
        self.assertEqual(state.checkpoint["goal_step_statuses"]["3"]["status"], "completed")
        self.assertEqual(state.checkpoint["goal_step_statuses"]["4"]["status"], "skipped")
        self.assertEqual(state.checkpoint["goal_step_statuses"]["5"]["status"], "completed")
        convergence.assert_called_once()
        conclusion.assert_called_once_with(Path(self.root / "tool_convergence" / "conv_unit" / "state.json"))

    def test_nonblocking_convergence_failure_continues_to_primary_conclusion(self) -> None:
        primary_state = self.root / "state" / "primary_state.json"
        tool_result = {
            "tool_name": "mosfet_2d_id_sweep",
            "run_id": "current_mos",
            "status": "completed",
            "state_path": str(primary_state),
            "quality_report": {"status": "passed"},
        }
        plan = json.dumps(
            {
                "steps": [
                    {"index": 1, "kind": "run_supervisor", "title": "Run current TCAD", "request": {}, "depends_on": []},
                    {
                        "index": 2,
                        "kind": "run_tool_convergence",
                        "title": "Optional convergence",
                        "request": {
                            "tool_name": "mosfet_2d_id_sweep",
                            "base_request": {"sweep_type": "output_characteristic"},
                            "axis_path": "mesh_refinement_level",
                            "values": [1, 2, 3],
                        },
                        "depends_on": [1],
                        "stop_on_failure": False,
                    },
                    {
                        "index": 3,
                        "kind": "run_repair_executor",
                        "title": "Repair if needed",
                        "request": {},
                        "depends_on": [2],
                        "stop_on_failure": False,
                    },
                    {"index": 4, "kind": "generate_conclusion", "title": "Conclude", "request": {}, "depends_on": [3]},
                ]
            }
        )

        with (
            patch("tcad_agent.mission_agent.rebuild_index", return_value={"records_indexed": 1}),
            patch("tcad_agent.mission_agent.list_records", return_value=[]),
            patch("tcad_agent.mission_agent.run_supervisor", return_value=self.fake_supervisor_state(tool_result)),
            patch("tcad_agent.mission_agent.run_tool_convergence", return_value=self.fake_failed_tool_convergence()),
            patch("tcad_agent.mission_agent.generate_experiment_conclusion", return_value=self.fake_conclusion()) as conclusion,
            patch("tcad_agent.mission_agent.build_repair_plan") as repair_plan,
        ):
            state = run_mission_agent(
                "2D NMOS output characteristic with optional convergence",
                mission_id="mission_soft_convergence",
                mission_root=self.root,
                execute=True,
                max_cycles=8,
                use_llm_decomposer=True,
                llm_client=FakeLLMClient(plan),
            )

        self.assertEqual(state.status, MissionStatus.COMPLETED)
        self.assertEqual(state.checkpoint["goal_step_statuses"]["2"]["status"], "soft_failed")
        self.assertEqual(state.checkpoint["goal_step_statuses"]["3"]["status"], "skipped")
        self.assertTrue(state.checkpoint["soft_failures"])
        repair_plan.assert_not_called()
        conclusion.assert_called_once_with(primary_state)

    def test_agent_replans_with_llm_after_optional_convergence_failure(self) -> None:
        primary_state = self.root / "state" / "primary_state.json"
        primary_state.parent.mkdir(parents=True, exist_ok=True)
        primary_state.write_text(
            json.dumps(
                {
                    "tool_name": "mosfet_2d_id_sweep",
                    "status": "completed",
                    "run_id": "current_mos",
                    "request": {
                        "sweep_type": "idvd",
                        "tcad_deck_spec": {
                            "device_family": "2d_mosfet",
                            "physics_models": {"coupling_status": "needs_benchmark_confirmation"},
                            "signoff_requirements": {"required_level": "engineering_signoff"},
                        },
                    },
                    "quality_report": {"status": "passed", "issues": [], "metrics": {"idvd_final_current_a": 1e-5}},
                }
            ),
            encoding="utf-8",
        )
        tool_result = {
            "tool_name": "mosfet_2d_id_sweep",
            "run_id": "current_mos",
            "status": "completed",
            "state_path": str(primary_state),
            "quality_report": {"status": "passed"},
        }
        initial_plan = json.dumps(
            {
                "steps": [
                    {"index": 1, "kind": "run_supervisor", "title": "Run current TCAD", "request": {}, "depends_on": []},
                    {
                        "index": 2,
                        "kind": "run_tool_convergence",
                        "title": "Optional convergence",
                        "request": {
                            "tool_name": "mosfet_2d_id_sweep",
                            "base_request": {"sweep_type": "output_characteristic"},
                            "axis_path": "mesh_refinement_level",
                            "values": [1, 2, 3],
                        },
                        "depends_on": [1],
                        "stop_on_failure": False,
                    },
                    {
                        "index": 3,
                        "kind": "run_repair_executor",
                        "title": "Repair if needed",
                        "request": {},
                        "depends_on": [2],
                        "stop_on_failure": False,
                    },
                    {"index": 4, "kind": "generate_conclusion", "title": "Conclude", "request": {}, "depends_on": [3]},
                ]
            }
        )
        replan_response = json.dumps(
            {
                "strategy_zh": "收敛验证是非关键失败，字段已归一化；先保留主仿真结果继续生成结论，并把收敛风险写入结论。",
                "mark_soft_failed": [2],
                "append_steps": [],
            }
        )
        client = SequenceLLMClient([initial_plan, replan_response])

        with (
            patch("tcad_agent.mission_agent.rebuild_index", return_value={"records_indexed": 1}),
            patch("tcad_agent.mission_agent.list_records", return_value=[]),
            patch("tcad_agent.mission_agent.run_supervisor", return_value=self.fake_supervisor_state(tool_result)),
            patch("tcad_agent.mission_agent.run_tool_convergence", return_value=self.fake_failed_tool_convergence()),
            patch("tcad_agent.mission_agent.generate_experiment_conclusion", return_value=self.fake_conclusion()),
            patch("tcad_agent.mission_agent.build_repair_plan"),
        ):
            state = run_mission_agent(
                "2D NMOS output characteristic with optional convergence",
                mission_id="mission_llm_replan",
                mission_root=self.root,
                execute=True,
                max_cycles=8,
                use_llm_decomposer=True,
                llm_client=client,
            )

        self.assertEqual(state.status, MissionStatus.COMPLETED)
        self.assertEqual(client.calls, 2)
        self.assertIn(MissionStepKind.REPLAN, [step.kind for step in state.steps])
        self.assertEqual(state.checkpoint["agent_replans"][0]["strategy_zh"], "收敛验证是非关键失败，字段已归一化；先保留主仿真结果继续生成结论，并把收敛风险写入结论。")
        self.assertEqual(state.checkpoint["goal_step_statuses"]["2"]["status"], "soft_failed")
        replan_prompt = json.loads(client.users[1])
        digest = replan_prompt["issue_context"]["current_evidence_digest"]
        self.assertEqual(digest["tcad_deck_spec"]["device_family"], "2d_mosfet")
        self.assertEqual(digest["quality_report"]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
