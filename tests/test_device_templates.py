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
        self.assertEqual(result.tcad_fidelity, "devsim_2d_drift_diffusion")
        self.assertIn("mesh_model_convergence", result.signoff_workflow)
        self.assertIn("mosfet_id_dibl", result.public_source_category_ids)
        self.assertIn("split_low_high_drain_idvg_for_dibl", result.recommended_convergence)
        self.assertIn("devsim_3dmos", {source["source_id"] for source in result.public_sources})

    def test_routes_extended_schottky_template_as_executable(self) -> None:
        result = route_device_goal("做 Schottky diode forward IV 并提取 barrier height")

        self.assertEqual(result.status, RouteStatus.MATCHED)
        self.assertTrue(result.executable)
        self.assertEqual(result.suggested_tool, "extended_device_sweep")
        self.assertEqual(result.template.support, TemplateSupport.EXECUTABLE)
        self.assertEqual(result.request_hint["device_type"], "schottky_diode")
        self.assertEqual(result.tcad_fidelity, "devsim_1d_thermionic_contact")

    def test_specific_power_mosfet_beats_generic_mosfet_alias(self) -> None:
        result = route_device_goal("做 power MOSFET BV 和 Ron 优化")

        self.assertEqual(result.status, RouteStatus.MATCHED)
        self.assertEqual(result.template.template_id, "power_mosfet_bv_ron")
        self.assertTrue(result.executable)
        self.assertTrue(result.runnable)
        self.assertTrue(result.signoff_ready)
        self.assertTrue(result.runner_promotion_required)
        self.assertIn("runner_contract", result.runner_promotion_stage_ids)
        self.assertEqual(result.template.support, TemplateSupport.EXECUTABLE)
        self.assertEqual(result.request_hint["fidelity"], "devsim_2d_field_plate")
        self.assertEqual(result.tcad_fidelity, "devsim_2d_field_plate_layout_prototype")
        runner_ids = {runner["runner_id"] for runner in result.industrial_runner_coverage}
        self.assertIn("power_mosfet_bv_ron_devsim_1d", runner_ids)
        self.assertIn("power_mosfet_bv_ron_devsim_2d_field_plate", runner_ids)
        self.assertIn("TCAD evidence", result.message)

    def test_lists_extended_templates_by_support_boundary(self) -> None:
        executable = list_device_templates(support=TemplateSupport.EXECUTABLE)
        executable_ids = {item["template_id"] for item in executable}
        compact = list_device_templates(support=TemplateSupport.COMPACT_BASELINE)
        compact_ids = {item["template_id"] for item in compact}
        planned = list_device_templates(support=TemplateSupport.PLANNED)
        planned_ids = {item["template_id"] for item in planned}

        self.assertIn("mosfet_2d_id", executable_ids)
        self.assertIn("schottky_diode", executable_ids)
        self.assertIn("bjt_gummel_output", executable_ids)
        self.assertIn("power_mosfet_bv_ron", executable_ids)
        self.assertIn("gan_hemt_id_bv", executable_ids)
        self.assertIn("sic_power_diode_bv_leakage", executable_ids)
        self.assertIn("finfet_id_cv", executable_ids)
        self.assertIn("igbt_output_turnoff", executable_ids)
        self.assertNotIn("bjt_gummel_output", compact_ids)
        self.assertNotIn("power_mosfet_bv_ron", compact_ids)
        self.assertEqual(planned_ids, set())

    def test_advanced_industrial_template_routes_to_physics_runner(self) -> None:
        result = route_device_goal("GaN HEMT 输出特性和 current collapse 风险")

        self.assertEqual(result.status, RouteStatus.MATCHED)
        self.assertEqual(result.template.support, TemplateSupport.EXECUTABLE)
        self.assertEqual(result.tcad_fidelity, "physics_1d_algan_gan_polarization_trap")
        self.assertEqual(result.request_hint["fidelity"], "physics_1d")
        self.assertIn("polarization_trap_convergence", result.signoff_workflow)
        self.assertIn("gan_algan_hemt", result.public_source_category_ids)
        self.assertIn("solve_heterojunction_equilibrium_with_fixed_polarization_first", result.recommended_convergence)
        self.assertIn("genius_tcad_open", {source["source_id"] for source in result.public_sources})
        self.assertTrue(result.executable)
        self.assertTrue(result.runnable)
        self.assertEqual(result.suggested_tool, "extended_device_sweep")
        self.assertEqual(result.capability_warnings, [])
        self.assertTrue(result.runner_promotion_required)
        self.assertIn("golden_correlation_and_signoff", result.runner_promotion_stage_ids)


if __name__ == "__main__":
    unittest.main()
