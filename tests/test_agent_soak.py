from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.agent_soak import AgentSoakRequest, AgentSoakStatus, run_agent_soak


def write_state(path: Path) -> Path:
    csv_path = path.parent / "curve.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("voltage_v,current_a\n0,0\n1,1e-9\n", encoding="utf-8")
    path.write_text(
        json.dumps(
            {
                "tool_name": "extended_device_sweep",
                "status": "completed",
                "run_id": "soak_passed",
                "run_dir": str(path.parent),
                "final_summary": {
                    "artifacts": {"csv": str(csv_path)},
                    "metrics": {"leakage_current_a": 1e-9, "points": 2},
                },
                "quality_report": {"status": "passed", "issues": [], "metrics": {"leakage_current_a": 1e-9}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


class AgentSoakTest(unittest.TestCase):
    def test_soak_resumes_agent_across_step_slices_and_writes_cockpit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_state = write_state(root / "runs" / "passed" / "state.json")
            calls: list[str] = []
            state = run_agent_soak(
                AgentSoakRequest(
                    goal_text="Run a sliced deterministic soak.",
                    soak_id="unit_soak",
                    soak_root=root,
                    execute=True,
                    duration_hours=0,
                    max_steps=3,
                    step_slice=1,
                    autonomous_request={
                        "use_llm": False,
                        "initial_tool_name": "extended_device_sweep",
                        "initial_request": {"run_id": "unit_soak_tool"},
                        "generate_report": False,
                        "generate_dashboard": False,
                    },
                ),
                runner_registry={
                    "extended_device_sweep": lambda request: calls.append("tool") or {"status": "completed", "state_path": str(result_state)},
                    "physical_benchmark": lambda request: calls.append("benchmark") or {"status": "completed", "benchmark_path": str(root / "benchmark.json")},
                },
            )

            self.assertEqual(state.status, AgentSoakStatus.COMPLETED)
            self.assertEqual([cycle.new_steps for cycle in state.cycles], [1, 1, 1])
            self.assertEqual(calls, ["tool", "benchmark"])
            self.assertEqual(state.completed_steps, 3)
            self.assertIsNone(state.failure_reason)
            self.assertTrue(Path(str(state.agent_state_path)).exists())
            self.assertTrue(Path(str(state.latest_cockpit_path)).exists())

    def test_soak_observes_cancel_file_before_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cancel_file = root / "cancel.requested"
            cancel_file.write_text("cancel\n", encoding="utf-8")
            state = run_agent_soak(
                AgentSoakRequest(
                    goal_text="Cancel this soak.",
                    soak_id="unit_cancel",
                    soak_root=root,
                    execute=True,
                    max_steps=2,
                    cancel_file=cancel_file,
                    autonomous_request={"use_llm": False},
                ),
            )

            self.assertEqual(state.status, AgentSoakStatus.CANCELLED)
            self.assertEqual(state.cycles, [])
            self.assertTrue(Path(state.heartbeat_path).exists())


if __name__ == "__main__":
    unittest.main()
