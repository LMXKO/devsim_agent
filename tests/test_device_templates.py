from __future__ import annotations

import unittest

from tcad_agent.device_templates import RouteStatus, TemplateSupport, list_device_templates, route_device_goal


class DeviceTemplatesTest(unittest.TestCase):
    def test_routes_executable_mosfet_template(self) -> None:
        result = route_device_goal("做 2D MOSFET Id-Vg 和 Id-Vd")

        self.assertEqual(result.status, RouteStatus.MATCHED)
        self.assertTrue(result.executable)
        self.assertEqual(result.suggested_tool, "mosfet_2d_id_sweep")
        self.assertEqual(result.template.template_id, "mosfet_2d_id")

    def test_routes_extended_schottky_template_as_executable(self) -> None:
        result = route_device_goal("做 Schottky diode forward IV 并提取 barrier height")

        self.assertEqual(result.status, RouteStatus.MATCHED)
        self.assertTrue(result.executable)
        self.assertEqual(result.suggested_tool, "extended_device_sweep")
        self.assertEqual(result.template.support, TemplateSupport.EXECUTABLE)
        self.assertEqual(result.request_hint["device_type"], "schottky_diode")

    def test_specific_power_mosfet_beats_generic_mosfet_alias(self) -> None:
        result = route_device_goal("做 power MOSFET BV 和 Ron 优化")

        self.assertEqual(result.status, RouteStatus.MATCHED)
        self.assertEqual(result.template.template_id, "power_mosfet_bv_ron")
        self.assertTrue(result.executable)

    def test_lists_extended_templates_as_executable(self) -> None:
        executable = list_device_templates(support=TemplateSupport.EXECUTABLE)
        ids = {item["template_id"] for item in executable}

        self.assertIn("bjt_gummel_output", ids)
        self.assertIn("power_mosfet_bv_ron", ids)
        self.assertIn("mosfet_2d_id", ids)


if __name__ == "__main__":
    unittest.main()
