from __future__ import annotations

import unittest

from tcad_agent.engineering_intent import DeviceSupport, parse_engineering_intent


class EngineeringIntentTest(unittest.TestCase):
    def test_parses_natural_language_mosfet_business_goal(self) -> None:
        intent = parse_engineering_intent(
            "帮我看一下这个 2D NMOS 的线性区和饱和区 Id-Vg。Vd 用 0.05V 和 1.0V 两个点，"
            "Vg 从 0 扫到 1.2V。我要 Vth、SS、Ion/Ioff，还有 DIBL；中间收敛失败你自己调步长重跑。"
        )

        self.assertEqual(intent.device_family, "mosfet_2d")
        self.assertEqual(intent.support, DeviceSupport.EXECUTABLE)
        self.assertIn("idvg", intent.analyses)
        self.assertIn("idvd", intent.analyses)
        self.assertIn("vth", intent.metrics)
        self.assertIn("dibl", intent.metrics)
        self.assertIn("auto_repair_without_user_until_budget_exhausted", intent.repair_preferences)
        self.assertEqual(intent.sweep_hints["gate_stop"], 1.2)
        self.assertEqual(intent.request_hint["sweep_type"], "both")

    def test_advanced_industrial_template_is_executable(self) -> None:
        intent = parse_engineering_intent("GaN HEMT 输出特性有 current collapse 风险，帮我扫栅压和漏压")

        self.assertEqual(intent.device_family, "gan_hemt")
        self.assertEqual(intent.support, DeviceSupport.EXECUTABLE)
        self.assertEqual(intent.request_hint["fidelity"], "physics_1d")
        self.assertIn("idvd", intent.analyses)
        self.assertEqual(intent.evidence_policy, "executable_exploratory")
        self.assertEqual(intent.capability_warnings, [])

    def test_power_mosfet_routes_to_physics_fidelity(self) -> None:
        intent = parse_engineering_intent("LDMOS / power MOSFET BV 和 Ron tradeoff，自动看场峰值风险")

        self.assertEqual(intent.device_family, "power_mosfet")
        self.assertEqual(intent.support, DeviceSupport.EXECUTABLE)
        self.assertEqual(intent.request_hint["fidelity"], "physics_1d")
        self.assertEqual(intent.evidence_policy, "executable_exploratory")
        self.assertFalse(intent.capability_warnings)

    def test_abstract_tcad_engineer_goal_asks_for_clarification(self) -> None:
        intent = parse_engineering_intent("TCAD仿真工程师通过自然语言描述完成工作，需要agent长时间、自动解决遇到的问题")

        self.assertEqual(intent.device_family, "unknown")
        self.assertEqual(intent.support, DeviceSupport.UNKNOWN)
        self.assertEqual(intent.evidence_policy, "needs_clarification")
        self.assertGreaterEqual(len(intent.clarification_questions), 2)
        self.assertIn("auto_repair_without_user_until_budget_exhausted", intent.repair_preferences)

    def test_parses_simple_spec_constraints(self) -> None:
        intent = parse_engineering_intent("做 PN diode BV 至少 30V，漏电低于 10nA，Ion/Ioff 至少 1e4")

        self.assertIn("leakage_current_limit=10nA", intent.constraints)
        self.assertIn("ion_ioff_min=1e4", intent.constraints)
        self.assertIn("breakdown_voltage_min=30V", intent.constraints)


if __name__ == "__main__":
    unittest.main()
