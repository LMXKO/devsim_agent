from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.agent_cockpit import generate_agent_cockpit


class AgentCockpitTest(unittest.TestCase):
    def test_generates_minimal_agent_cockpit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "agent_state.json"
            output = root / "cockpit.html"
            source.write_text(
                json.dumps(
                    {
                        "tool_name": "autonomous_devsim_agent",
                        "status": "running",
                        "agent_id": "agent_ui",
                        "goal_text": "Power MOSFET BV Ron",
                        "next_action": "run signoff",
                        "checkpoint": {
                            "agent_decision_ledger": [
                                {
                                    "step_index": 1,
                                    "action": {"kind": "run_tool", "tool_name": "power_mosfet_signoff"},
                                    "hypothesis_zh": "验证 2D 场板收敛缺口",
                                    "fallback_used": False,
                                }
                            ]
                        },
                        "signoff_gate": {
                            "verdict": "conditional",
                            "missing_evidence": ["golden_or_measured_correlation"],
                            "next_actions": [{"action": "add_golden", "reason": "补实测曲线"}],
                        },
                        "mutation_effect_analysis": {
                            "decision": "continue_refine",
                            "primary_metric": "leakage_current_a",
                            "improved_metrics": ["leakage_current_a"],
                        },
                        "final_summary": {"artifacts": {"signoff_gate": str(root / "signoff_gate.json")}},
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            result = generate_agent_cockpit(source, output)
            html = output.read_text(encoding="utf-8")

        self.assertEqual(result.status, "completed")
        self.assertIn("Decisions", html)
        self.assertIn("Lineage", html)
        self.assertIn("golden_or_measured_correlation", html)
        self.assertNotIn("hero", html.lower())


if __name__ == "__main__":
    unittest.main()

