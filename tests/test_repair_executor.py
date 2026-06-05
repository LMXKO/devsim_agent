from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.repair_executor import RepairExecutionStatus, run_repair_executor


class RepairExecutorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run_root = self.root / "agent_tools"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_source_state(self, *, issue_code: str = "current_not_monotonic", failure_class: str | None = None) -> Path:
        run_dir = self.run_root / "pn_junction_iv" / "source_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "tool_name": "pn_junction_iv_sweep",
            "status": "completed" if failure_class is None else "failed",
            "run_id": "source_run",
            "run_dir": str(run_dir),
            "request": {
                "start": 0.0,
                "stop": 1.0,
                "step": 0.5,
                "min_step": 0.125,
                "max_attempts": 2,
                "run_root": str(self.run_root),
            },
            "attempts": [{"failure_class": failure_class}] if failure_class else [],
            "quality_report": {
                "status": "suspicious",
                "issues": [{"code": issue_code, "severity": "warning"}],
                "metrics": {},
            },
        }
        state_path = run_dir / "state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        return state_path

    def write_mosfet_convergence_state(self) -> Path:
        run_dir = self.run_root / "mosfet_2d_id" / "mos_bad"
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "tool_name": "mosfet_2d_id_sweep",
            "status": "failed",
            "run_id": "mos_bad",
            "run_dir": str(run_dir),
            "request": {
                "sweep_type": "idvd",
                "gate_start": 0.8,
                "gate_stop": 1.2,
                "gate_step": 0.2,
                "min_gate_step": 0.05,
                "drain_start": 0.0,
                "drain_stop": 1.2,
                "drain_step": 0.05,
                "min_drain_step": 0.0125,
                "idvd_gate_voltage": 1.2,
                "impact_ionization_model": "selberherr",
                "x_divisions": 8,
                "silicon_y_divisions": 3,
                "run_root": str(self.run_root),
            },
            "attempts": [{"failure_class": "convergence", "failure_reason": "DEVSIM solver did not converge."}],
            "quality_report": {
                "status": "failed",
                "issues": [{"code": "too_many_convergence_failures", "severity": "error"}],
                "metrics": {},
            },
        }
        state_path = run_dir / "state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        return state_path

    def write_tool_convergence_failed_case_state(self) -> Path:
        run_dir = self.root / "tool_convergence" / "conv_bad"
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "tool_name": "tool_convergence",
            "status": "failed",
            "convergence_id": "conv_bad",
            "convergence_dir": str(run_dir),
            "target_tool": "mosfet_2d_id_sweep",
            "axis_path": "x_divisions",
            "values": [8, 12, 16],
            "cases": [
                {
                    "index": 1,
                    "status": "failed",
                    "failure_reason": "DEVSIM solver did not converge.",
                    "request": {
                        "sweep_type": "idvd",
                        "drain_start": 0.0,
                        "drain_stop": 1.2,
                        "drain_step": 0.1,
                        "min_drain_step": 0.025,
                        "gate_step": 0.2,
                        "min_gate_step": 0.05,
                        "run_root": str(self.run_root),
                    },
                }
            ],
            "quality_report": {
                "status": "failed",
                "issues": [{"code": "too_few_completed_convergence_cases", "severity": "error"}],
                "metrics": {"cases": 3, "completed_cases": 0},
            },
        }
        state_path = run_dir / "state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        return state_path

    def passing_runner(self, request: dict[str, object]) -> dict[str, object]:
        run_dir = Path(str(request["run_root"])) / "pn_junction_iv" / str(request["run_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "tool_name": "pn_junction_iv_sweep",
            "status": "completed",
            "run_id": request["run_id"],
            "run_dir": str(run_dir),
            "request": request,
            "attempts": [],
            "quality_report": {"status": "passed", "issues": [], "metrics": {}},
        }
        (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
        return state

    def suspicious_runner(self, request: dict[str, object]) -> dict[str, object]:
        run_dir = Path(str(request["run_root"])) / "pn_junction_iv" / str(request["run_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "tool_name": "pn_junction_iv_sweep",
            "status": "completed",
            "run_id": request["run_id"],
            "run_dir": str(run_dir),
            "request": request,
            "attempts": [],
            "quality_report": {
                "status": "suspicious",
                "issues": [{"code": "current_not_monotonic", "severity": "warning"}],
                "metrics": {},
            },
        }
        (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
        return state

    def mosfet_runner_passes_only_after_model_staging(self, request: dict[str, object]) -> dict[str, object]:
        run_dir = Path(str(request["run_root"])) / "mosfet_2d_id" / str(request["run_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        staged_model = (
            request.get("impact_ionization_model") == "none"
            and request.get("deferred_impact_ionization_model") == "selberherr"
        )
        state = {
            "tool_name": "mosfet_2d_id_sweep",
            "status": "completed" if staged_model else "failed",
            "run_id": request["run_id"],
            "run_dir": str(run_dir),
            "request": request,
            "attempts": [] if staged_model else [{"failure_class": "convergence"}],
            "quality_report": {
                "status": "passed" if staged_model else "failed",
                "issues": [] if staged_model else [{"code": "too_many_convergence_failures", "severity": "error"}],
                "metrics": {},
            },
        }
        (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
        return state

    def mosfet_passing_runner(self, request: dict[str, object]) -> dict[str, object]:
        run_dir = Path(str(request["run_root"])) / "mosfet_2d_id" / str(request["run_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "tool_name": "mosfet_2d_id_sweep",
            "status": "completed",
            "run_id": request["run_id"],
            "run_dir": str(run_dir),
            "request": request,
            "attempts": [],
            "quality_report": {"status": "passed", "issues": [], "metrics": {}},
        }
        (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
        return state

    def test_plan_only_builds_next_repair_request(self) -> None:
        source = self.write_source_state()

        state = run_repair_executor(source, execution_id="repair_plan", execute=False)

        self.assertEqual(state.status, RepairExecutionStatus.PLANNED)
        self.assertEqual(len(state.attempts), 1)
        self.assertEqual(state.attempts[0].action_name, "local_bias_step_refinement")
        self.assertEqual(state.attempts[0].next_request["step"], 0.25)
        self.assertTrue((source.parent / "repair_execution" / "repair_plan" / "repair_execution_state.json").exists())

    def test_execute_runs_repair_and_accepts_passed_result(self) -> None:
        source = self.write_source_state()

        state = run_repair_executor(
            source,
            execution_id="repair_exec",
            execute=True,
            registry={"pn_junction_iv_sweep": self.passing_runner},
        )

        self.assertEqual(state.status, RepairExecutionStatus.COMPLETED)
        self.assertEqual(state.final_quality_status, "passed")
        self.assertEqual(len(state.attempts), 1)
        self.assertTrue(Path(state.final_state_path).exists())

    def test_sensitive_repair_waits_for_user_by_default(self) -> None:
        source = self.write_source_state(issue_code="junction_not_inside_device")

        state = run_repair_executor(
            source,
            execution_id="repair_wait",
            execute=True,
            registry={"pn_junction_iv_sweep": self.passing_runner},
        )

        self.assertEqual(state.status, RepairExecutionStatus.WAITING_FOR_USER)
        self.assertEqual(state.attempts, [])
        self.assertIn("blocked_repair_plan_path", state.checkpoint)

    def test_max_rounds_fails_if_quality_never_passes(self) -> None:
        source = self.write_source_state()

        state = run_repair_executor(
            source,
            execution_id="repair_budget",
            execute=True,
            max_rounds=2,
            registry={"pn_junction_iv_sweep": self.suspicious_runner},
        )

        self.assertEqual(state.status, RepairExecutionStatus.FAILED)
        self.assertIn("maximum repair rounds", state.failure_reason)
        self.assertEqual(len(state.attempts), 2)
        self.assertEqual(state.attempts[1].next_request["run_id"].count("_repair_"), 1)
        self.assertLess(len(str(state.attempts[1].next_request["run_id"])), 120)

    def test_execute_tries_next_repair_strategy_after_failed_attempt(self) -> None:
        source = self.write_mosfet_convergence_state()

        state = run_repair_executor(
            source,
            execution_id="repair_model_staging",
            execute=True,
            max_rounds=3,
            registry={"mosfet_2d_id_sweep": self.mosfet_runner_passes_only_after_model_staging},
        )

        self.assertEqual(state.status, RepairExecutionStatus.COMPLETED)
        self.assertEqual(
            [attempt.action_name for attempt in state.attempts],
            ["continuation_bias_ramp", "reuse_last_successful_initial_solution", "model_switch_staging"],
        )
        self.assertTrue(state.attempts[1].next_request["resume"])
        self.assertEqual(state.attempts[2].next_request["impact_ionization_model"], "none")
        self.assertEqual(state.attempts[2].next_request["deferred_impact_ionization_model"], "selberherr")

    def test_tool_convergence_failed_case_executes_target_tool_retry(self) -> None:
        source = self.write_tool_convergence_failed_case_state()

        state = run_repair_executor(
            source,
            execution_id="repair_toolconv_case",
            execute=True,
            max_rounds=2,
            registry={"mosfet_2d_id_sweep": self.mosfet_passing_runner},
        )

        self.assertEqual(state.status, RepairExecutionStatus.COMPLETED)
        self.assertEqual(state.attempts[0].target_tool, "mosfet_2d_id_sweep")
        self.assertEqual(state.attempts[0].action_name, "rerun_failed_convergence_cases_with_safe_bias")
        self.assertLess(state.attempts[0].next_request["drain_step"], 0.1)
        self.assertTrue(state.attempts[0].next_request["resume"])


if __name__ == "__main__":
    unittest.main()
