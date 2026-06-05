from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.llm import LLMConfig
from tcad_agent.tools.llm_diagnose import (
    DiagnosisStatus,
    build_context,
    diagnose_state,
    parse_json_response,
)


class FakeClient:
    config = LLMConfig(model="fake-model")

    def __init__(self) -> None:
        self.calls: list[dict[str, str | float]] = []

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        return json.dumps(
            {
                "diagnosis": "Run completed but is physically suspicious.",
                "risk_level": "high",
                "recommended_next_action": "rerun with stop voltage reduced to 0.5 V",
                "next_tool_command": "python3.11 -m tcad_agent.tools.pn_junction_iv --stop 0.5 --step 0.1",
                "rationale": ["current exceeds threshold"],
                "follow_up_checks": ["verify quality_report.status is passed"],
            }
        )


class LLMDiagnoseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_state(self, quality_status: str) -> Path:
        run_dir = self.root / "run"
        log_dir = run_dir / "attempt_runs" / "pn_junction" / "attempt_001"
        log_dir.mkdir(parents=True)
        log_path = log_dir / "devsim.log"
        log_path.write_text("Iteration: 0\nConvergence failure!\n", encoding="utf-8")

        state = {
            "tool_name": "pn_junction_iv_sweep",
            "status": "completed",
            "run_id": "unit_run",
            "run_dir": str(run_dir),
            "request": {"stop": 5.0, "step": 5.0},
            "checkpoint": {"quality_status": quality_status},
            "attempts": [
                {
                    "index": 1,
                    "status": "failed",
                    "step_v": 5.0,
                    "failure_class": "convergence",
                    "failure_reason": "DEVSIM solver did not converge.",
                    "stderr_tail": "devsim_py3.error: Convergence failure!",
                }
            ],
            "final_summary": {
                "status": "completed",
                "artifacts": {"log": str(log_path)},
            },
            "quality_report": {
                "status": quality_status,
                "issues": [
                    {
                        "code": "current_exceeds_policy",
                        "severity": "warning",
                        "message": "Total current exceeds policy.",
                        "evidence": {"max_abs_current_a": 10.0},
                    }
                ],
                "metrics": {"max_abs_current_a": 10.0, "convergence_failures": 1},
                "recommended_next_action": "narrow the voltage range",
            },
        }
        state_path = run_dir / "state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        return state_path

    def test_parse_json_response_extracts_json(self) -> None:
        parsed = parse_json_response('prefix {"recommended_next_action":"rerun"} suffix')
        self.assertEqual(parsed, {"recommended_next_action": "rerun"})

    def test_parse_json_response_normalizes_null_command(self) -> None:
        parsed = parse_json_response('{"next_tool_command":"null"}')
        self.assertEqual(parsed, {"next_tool_command": None})

    def test_parse_json_response_rejects_arbitrary_shell_command(self) -> None:
        parsed = parse_json_response('{"next_tool_command":"cat /tmp/file"}')
        self.assertEqual(parsed["next_tool_command"], None)
        self.assertEqual(parsed["rejected_next_tool_command"], "cat /tmp/file")

    def test_parse_json_response_allows_known_tool_command(self) -> None:
        command = "python3.11 -m tcad_agent.tools.pn_junction_iv --stop 0.5 --step 0.1"
        parsed = parse_json_response(json.dumps({"next_tool_command": command}))
        self.assertEqual(parsed["next_tool_command"], command)

    def test_passed_state_skips_without_calling_llm(self) -> None:
        state_path = self.write_state("passed")
        client = FakeClient()

        result = diagnose_state(state_path, client=client)

        self.assertEqual(result.status, DiagnosisStatus.SKIPPED)
        self.assertEqual(client.calls, [])
        self.assertTrue((state_path.parent / "llm_diagnosis.json").exists())

    def test_suspicious_state_calls_llm_and_writes_result(self) -> None:
        state_path = self.write_state("suspicious")
        client = FakeClient()

        result = diagnose_state(state_path, client=client)

        self.assertEqual(result.status, DiagnosisStatus.COMPLETED)
        self.assertEqual(result.model, "fake-model")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(result.recommended_next_action, "rerun with stop voltage reduced to 0.5 V")
        self.assertTrue((state_path.parent / "llm_diagnosis.json").exists())

    def test_context_includes_log_excerpts(self) -> None:
        state_path = self.write_state("suspicious")

        context = build_context(state_path, max_log_chars=100)

        self.assertEqual(context.quality_status, "suspicious")
        self.assertGreaterEqual(len(context.log_excerpts), 1)
        self.assertIn("Convergence failure", context.log_excerpts[0]["text_tail"])


if __name__ == "__main__":
    unittest.main()
