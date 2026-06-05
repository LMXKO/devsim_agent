from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.tools.autonomous_loop import (
    AutonomousLoopRequest,
    LoopStatus,
    run_autonomous_loop,
)
from tcad_agent.tools.llm_diagnose import DiagnosisStatus, LLMDiagnosisResult
from tcad_agent.tools.pn_junction_iv import PNJunctionIVRequest


class FakePNRunner:
    def __init__(self, pass_when_stop_at_most: float = 0.5) -> None:
        self.pass_when_stop_at_most = pass_when_stop_at_most
        self.calls: list[PNJunctionIVRequest] = []

    def __call__(self, request: PNJunctionIVRequest) -> dict[str, object]:
        self.calls.append(request)
        run_dir = request.run_root / "pn_junction_iv" / (request.run_id or "fake_run")
        run_dir.mkdir(parents=True, exist_ok=True)

        passed = request.stop <= self.pass_when_stop_at_most
        quality_report = {
            "status": "passed" if passed else "suspicious",
            "issues": []
            if passed
            else [
                {
                    "code": "current_exceeds_policy",
                    "severity": "warning",
                    "message": "current too high",
                    "evidence": {"max_abs_current_a": 10.0},
                }
            ],
            "metrics": {"max_abs_current_a": 0.1 if passed else 10.0},
            "recommended_next_action": "accept" if passed else "narrow voltage range",
        }
        state = {
            "tool_name": "pn_junction_iv_sweep",
            "status": "completed",
            "run_id": request.run_id,
            "run_dir": str(run_dir),
            "request": request.model_dump(mode="json"),
            "attempts": [],
            "checkpoint": {"quality_status": quality_report["status"]},
            "final_summary": {"status": "completed", "artifacts": {}},
            "quality_report": quality_report,
        }
        (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
        return state


class FakeDiagnosisRunner:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def __call__(self, **kwargs: object) -> LLMDiagnosisResult:
        state_path = Path(kwargs["state_path"])
        self.calls.append(state_path)
        output_path = state_path.parent / "llm_diagnosis.json"
        diagnosis = LLMDiagnosisResult(
            status=DiagnosisStatus.COMPLETED,
            state_path=str(state_path),
            output_path=str(output_path),
            quality_status="suspicious",
            parsed_response={
                "recommended_next_action": "rerun at 0.4 V",
                "next_tool_command": "python3.11 -m tcad_agent.tools.pn_junction_iv --stop 0.4 --step 0.05",
            },
            recommended_next_action="rerun at 0.4 V",
        )
        output_path.write_text(json.dumps(diagnosis.model_dump(mode="json")), encoding="utf-8")
        return diagnosis


class AutonomousLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def request(self, **overrides: object) -> AutonomousLoopRequest:
        data = {
            "loop_id": "loop_unit",
            "loop_root": self.root / "loops",
            "run_root": self.root / "agent_tools",
            "stop": 5.0,
            "step": 5.0,
            "min_step": 1.25,
            "max_attempts": 3,
            "max_cycles": 3,
            "use_llm": False,
        }
        data.update(overrides)
        return AutonomousLoopRequest.model_validate(data)

    def test_no_llm_loop_runs_followup_until_quality_passes(self) -> None:
        runner = FakePNRunner()

        result = run_autonomous_loop(self.request(), pn_runner=runner)

        self.assertEqual(result["status"], LoopStatus.COMPLETED)
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(result["final_quality_report"]["status"], "passed")
        self.assertEqual(result["cycles"][0]["quality_status"], "suspicious")
        self.assertEqual(result["cycles"][0]["next_request"]["stop"], 0.5)
        self.assertEqual(result["cycles"][1]["quality_status"], "passed")
        self.assertTrue((self.root / "loops" / "loop_unit" / "loop_state.json").exists())

    def test_max_cycles_fails_without_accepted_result(self) -> None:
        runner = FakePNRunner(pass_when_stop_at_most=-1.0)

        result = run_autonomous_loop(self.request(max_cycles=1), pn_runner=runner)

        self.assertEqual(result["status"], LoopStatus.FAILED)
        self.assertEqual(len(runner.calls), 1)
        self.assertIn("maximum autonomous cycles", result["failure_reason"])

    def test_use_llm_feeds_diagnosis_into_strategy(self) -> None:
        runner = FakePNRunner()
        diagnosis = FakeDiagnosisRunner()

        result = run_autonomous_loop(
            self.request(use_llm=True),
            pn_runner=runner,
            diagnosis_runner=diagnosis,
        )

        self.assertEqual(result["status"], LoopStatus.COMPLETED)
        self.assertEqual(len(diagnosis.calls), 1)
        self.assertEqual(runner.calls[1].stop, 0.4)
        self.assertEqual(result["cycles"][0]["diagnosis_status"], DiagnosisStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()
