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

    def test_planned_industrial_template_is_explicit(self) -> None:
        intent = parse_engineering_intent("GaN HEMT 输出特性有 current collapse 风险，帮我扫栅压和漏压")

        self.assertEqual(intent.device_family, "gan_hemt")
        self.assertEqual(intent.support, DeviceSupport.PLANNED)
        self.assertEqual(intent.risk_level, "high")
        self.assertIn("idvd", intent.analyses)


if __name__ == "__main__":
    unittest.main()
