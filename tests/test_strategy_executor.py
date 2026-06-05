from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.tools.strategy_executor import StrategyStatus, build_strategy_plan


class StrategyExecutorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run_root = self.root / "agent_tools"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_state(self, quality_status: str, issues: list[dict[str, object]]) -> Path:
        run_dir = self.run_root / "pn_junction_iv" / "source_run"
        run_dir.mkdir(parents=True)
        state = {
            "tool_name": "pn_junction_iv_sweep",
            "status": "completed",
            "run_id": "source_run",
            "run_dir": str(run_dir),
            "request": {
                "start": 0.0,
                "stop": 5.0,
                "step": 5.0,
                "min_step": 1.25,
                "max_attempts": 3,
                "timeout_seconds": 60.0,
                "quality_min_points": 3,
                "quality_max_abs_current_a": 1.0,
                "quality_max_convergence_failures": 0,
                "run_id": "source_run",
                "run_root": str(self.run_root),
                "resume": False,
            },
            "attempts": [],
            "checkpoint": {"quality_status": quality_status},
            "quality_report": {
                "status": quality_status,
                "issues": issues,
                "metrics": {},
            },
        }
        state_path = run_dir / "state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        return state_path

    def write_diagnosis(self, state_path: Path, next_tool_command: str | None) -> Path:
        diagnosis = {
            "status": "completed",
            "parsed_response": {
                "recommended_next_action": "rerun",
                "next_tool_command": next_tool_command,
            },
        }
        diagnosis_path = state_path.parent / "llm_diagnosis.json"
        diagnosis_path.write_text(json.dumps(diagnosis), encoding="utf-8")
        return diagnosis_path

    def test_passed_quality_skips_followup(self) -> None:
        state_path = self.write_state("passed", [])

        plan = build_strategy_plan(state_path)

        self.assertEqual(plan.status, StrategyStatus.SKIPPED)
        self.assertIsNone(plan.next_request)

    def test_suspicious_current_narrows_voltage_range(self) -> None:
        state_path = self.write_state(
            "suspicious",
            [
                {
                    "code": "current_exceeds_policy",
                    "severity": "warning",
                    "message": "too high",
                    "evidence": {},
                }
            ],
        )

        plan = build_strategy_plan(state_path)

        self.assertEqual(plan.status, StrategyStatus.PLANNED)
        self.assertEqual(plan.next_request["stop"], 0.5)
        self.assertEqual(plan.next_request["step"], 0.1)
        self.assertEqual(plan.next_request["run_id"], "source_run_followup_001")

    def test_rejects_arbitrary_llm_command_and_uses_policy(self) -> None:
        state_path = self.write_state(
            "suspicious",
            [
                {
                    "code": "current_exceeds_policy",
                    "severity": "warning",
                    "message": "too high",
                    "evidence": {},
                }
            ],
        )
        diagnosis_path = self.write_diagnosis(state_path, "cat /tmp/file")

        plan = build_strategy_plan(state_path, diagnosis_path=diagnosis_path)

        self.assertEqual(plan.status, StrategyStatus.PLANNED)
        self.assertIn("LLM next_tool_command was not on the allowed tool command list.", plan.warnings)
        self.assertEqual(plan.next_request["stop"], 0.5)

    def test_accepts_whitelisted_llm_pn_command(self) -> None:
        state_path = self.write_state("suspicious", [])
        command = "python3.11 -m tcad_agent.tools.pn_junction_iv --stop 0.4 --step 0.05 --min-step 0.0125 --max-attempts 4"
        diagnosis_path = self.write_diagnosis(state_path, command)

        plan = build_strategy_plan(state_path, diagnosis_path=diagnosis_path)

        self.assertEqual(plan.status, StrategyStatus.PLANNED)
        self.assertEqual(plan.reason, "follow-up request derived from whitelisted LLM next_tool_command")
        self.assertEqual(plan.next_request["stop"], 0.4)
        self.assertEqual(plan.next_request["step"], 0.05)
        self.assertEqual(plan.next_request["max_attempts"], 4)

    def test_adjusts_inherited_min_step_when_llm_reduces_step(self) -> None:
        state_path = self.write_state("suspicious", [])
        command = "python3.11 -m tcad_agent.tools.pn_junction_iv --stop 0.4 --step 0.05"
        diagnosis_path = self.write_diagnosis(state_path, command)

        plan = build_strategy_plan(state_path, diagnosis_path=diagnosis_path)

        self.assertEqual(plan.status, StrategyStatus.PLANNED)
        self.assertEqual(plan.next_request["step"], 0.05)
        self.assertEqual(plan.next_request["min_step"], 0.0125)
        self.assertIn("Adjusted min_step", plan.warnings[0])


if __name__ == "__main__":
    unittest.main()
