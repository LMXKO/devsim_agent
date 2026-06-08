from __future__ import annotations

import unittest

from tcad_agent.tcad_spec import parse_tcad_spec


class TCADSpecTest(unittest.TestCase):
    def test_parses_mosfet_signoff_spec_from_natural_language(self) -> None:
        spec = parse_tcad_spec(
            "客户 signoff 2D NMOS output characteristic 和 Id-Vg，tox 5nm，L 0.18um，"
            "Ion/Ioff 至少 1e4，measured_curve target.csv，interface trap 1e11"
        )

        self.assertEqual(spec.device_family, "mosfet_2d")
        self.assertEqual(spec.execution_profile, "tcad_signoff_candidate")
        self.assertEqual(spec.tcad_fidelity, "devsim_2d_drift_diffusion")
        self.assertIn("mesh_model_convergence", spec.signoff_workflow)
        self.assertTrue(spec.signoff_required)
        self.assertEqual(spec.measured_or_golden_reference, "target.csv")
        self.assertEqual(spec.geometry["oxide_thickness_nm"], 5.0)
        self.assertEqual(spec.geometry["length_um"], 0.18)
        self.assertEqual(spec.models["interface_trap_density_cm2"], 1e11)
        self.assertEqual(spec.constraints["ion_ioff_min"], "1e4")
        self.assertEqual(spec.spec_limits["ion_ioff_min"], 1e4)
        self.assertNotIn("measured_or_golden_reference", spec.missing_inputs)
        self.assertEqual(spec.calibration["reference_path"], "target.csv")
        self.assertIn("golden_or_measured_comparison", spec.deliverables)

    def test_parses_engineering_limits_bias_splits_and_temperature_corners(self) -> None:
        spec = parse_tcad_spec(
            "2D NMOS DIBL split：Vd 用 0.05V 和 1.0V，Vg 从 0V 到 1.2V，"
            "Ion/Ioff 至少 1e4，漏电低于 10nA，高温 125C，最后给结论"
        )

        self.assertEqual(spec.device_family, "mosfet_2d")
        self.assertEqual(spec.spec_limits["ion_ioff_min"], 1e4)
        self.assertAlmostEqual(spec.spec_limits["leakage_current_max_a"], 10e-9)
        self.assertEqual(spec.bias["drain_voltage_values_v"], [0.05, 1.0])
        self.assertIn("low_high_drain_idvg", spec.corner_plan["bias_splits"])
        self.assertAlmostEqual(spec.corner_plan["temperature_values_k"][0], 398.15)
        self.assertIn("engineering_conclusion", spec.deliverables)

    def test_abstract_goal_keeps_clarification_requirements(self) -> None:
        spec = parse_tcad_spec("TCAD仿真工程师通过自然语言描述完成工作，需要agent长时间自动解决问题")

        self.assertEqual(spec.execution_profile, "needs_clarification")
        self.assertIn("device_family", spec.missing_inputs)
        self.assertIn("analysis_type", spec.missing_inputs)
        self.assertGreaterEqual(len(spec.clarification_questions), 2)

    def test_advanced_industrial_spec_routes_to_executable_runner(self) -> None:
        spec = parse_tcad_spec("GaN HEMT 输出特性和 current collapse 风险，帮我扫栅压和漏压")

        self.assertEqual(spec.execution_profile, "tcad_executable")
        self.assertEqual(spec.tcad_fidelity, "physics_1d_algan_gan_polarization_trap")
        self.assertEqual(spec.request_hint["device_type"], "gan_hemt_id_bv")
        self.assertEqual(spec.request_hint["fidelity"], "physics_1d")
        self.assertNotIn("tcad_runner", spec.missing_inputs)
        self.assertIn("polarization_trap_convergence", spec.signoff_workflow)


if __name__ == "__main__":
    unittest.main()
