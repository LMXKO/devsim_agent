from __future__ import annotations

import unittest

from tcad_agent.engineering_intent import parse_engineering_intent
from tcad_agent.long_horizon_agent import build_long_horizon_snapshot, decide_long_horizon_action


class LongHorizonAgentTest(unittest.TestCase):
    def test_replans_soft_failure_before_asking_user(self) -> None:
        goal = "MOSFET Id-Vg 签核，失败时自动解决"
        checkpoint = {
            "engineering_intent": parse_engineering_intent(goal).model_dump(mode="json"),
            "agent_replan_attempts": 1,
            "agent_replan_max_attempts": 3,
        }
        observation = {
            "soft_failure_count": 1,
            "blocked_goal_steps": [],
            "pending_goal_kinds": ["run_repair_executor", "generate_conclusion"],
        }

        decision = decide_long_horizon_action(build_long_horizon_snapshot(goal, checkpoint, observation))

        self.assertEqual(decision.action, "replan")
        self.assertTrue(decision.should_replan)
        self.assertFalse(decision.needs_user)

    def test_signoff_missing_evidence_continues_with_risk_at_conclusion(self) -> None:
        goal = "NMOS signoff，要 mesh convergence 和工程结论"
        checkpoint = {
            "engineering_intent": parse_engineering_intent(goal).model_dump(mode="json"),
            "agent_replan_attempts": 0,
            "agent_replan_max_attempts": 3,
        }
        observation = {
            "soft_failure_count": 0,
            "blocked_goal_steps": [],
            "pending_goal_kinds": ["generate_conclusion"],
            "physical_benchmark": {"status": "suspicious"},
            "tool_convergence": {"status": "failed", "quality_status": "failed"},
        }

        decision = decide_long_horizon_action(build_long_horizon_snapshot(goal, checkpoint, observation))

        self.assertEqual(decision.action, "continue_with_risk")
        self.assertIn("mesh_or_tool_convergence", decision.missing_evidence)


if __name__ == "__main__":
    unittest.main()
