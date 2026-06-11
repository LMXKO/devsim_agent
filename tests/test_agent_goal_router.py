from __future__ import annotations

import unittest

from tcad_agent.agent_goal_router import AgentGoalRouteRequest, route_agent_goal
from tcad_agent.device_templates import route_device_goal


class AgentGoalRouterTest(unittest.TestCase):
    def test_routes_meta_agent_goal_to_power_mosfet_autonomous_request(self) -> None:
        result = route_agent_goal(
            AgentGoalRouteRequest(
                goal_text="AI 长时间自主操作 DEVSIM/Sentaurus 完成功率器件优化任务",
                execute=False,
            )
        )

        self.assertEqual(result.status, "matched")
        self.assertEqual(result.selected_template_id, "power_mosfet_bv_ron")
        self.assertEqual(result.primary_tool, "autonomous_devsim_agent")
        self.assertEqual(result.simulator_strategy, "devsim_primary_sentaurus_optional")
        self.assertTrue(result.autonomous_request["require_capability_audit"])
        self.assertTrue(result.autonomous_request["enable_experiment_design"])
        self.assertEqual(result.autonomous_request["initial_request"]["fidelity"], "devsim_2d_field_plate")
        self.assertIn("experiment_design", {step["id"] for step in result.evidence_plan})

    def test_device_template_accepts_generic_power_device_goal(self) -> None:
        route = route_device_goal("长时间自主优化高压功率器件 BV Ron 漏电")

        self.assertEqual(route.template.template_id, "power_mosfet_bv_ron")
        self.assertEqual(route.request_hint["fidelity"], "devsim_2d_field_plate")

    def test_sentaurus_only_goal_requires_external_workspace(self) -> None:
        result = route_agent_goal(
            AgentGoalRouteRequest(
                goal_text="用 Sentaurus 自主优化 GaN HEMT current collapse",
                simulator="sentaurus",
            )
        )

        self.assertEqual(result.status, "needs_input")
        self.assertIn("sentaurus_project_path", result.missing_inputs)
        self.assertEqual(result.simulator_strategy, "sentaurus_external_workspace_required")


if __name__ == "__main__":
    unittest.main()

