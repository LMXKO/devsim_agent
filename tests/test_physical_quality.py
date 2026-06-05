from __future__ import annotations

import unittest

from tcad_agent.physical_quality import (
    check_diode_breakdown_physics,
    check_mos_capacitor_physics,
    check_mosfet_physics,
    check_parameter_sanity,
    check_voltage_span,
    oxide_capacitance_f_per_cm2,
)


class PhysicalQualityTest(unittest.TestCase):
    def test_oxide_capacitance_estimate(self) -> None:
        cox = oxide_capacitance_f_per_cm2(5.0)

        self.assertAlmostEqual(cox, 6.906e-7, delta=1e-9)

    def test_parameter_sanity_flags_bad_doping_and_geometry(self) -> None:
        issues = check_parameter_sanity(
            {
                "p_doping_cm3": 1e25,
                "n_doping_cm3": 1e18,
                "length_um": 0.1,
                "junction_um": 0.2,
            },
            check_temperature=False,
        )
        codes = {issue["code"] for issue in issues}

        self.assertIn("doping_out_of_expected_range", codes)
        self.assertIn("junction_not_inside_device", codes)

    def test_parameter_sanity_flags_transport_model_ranges(self) -> None:
        issues = check_parameter_sanity(
            {
                "electron_lifetime_s": 1e-15,
                "hole_lifetime_s": 1e-2,
                "electron_mobility_cm2_v_s": 5000,
                "interface_trap_density_cm2": 5e13,
                "fixed_oxide_charge_cm2": 2e13,
            },
            check_temperature=False,
        )
        codes = {issue["code"] for issue in issues}

        self.assertIn("lifetime_out_of_expected_range", codes)
        self.assertIn("mobility_out_of_expected_range", codes)
        self.assertIn("interface_trap_density_extreme", codes)
        self.assertIn("fixed_oxide_charge_extreme", codes)

    def test_voltage_span_flags_unit_suspicion(self) -> None:
        issues = check_voltage_span([0, 1000], max_span_v=50)

        self.assertEqual(issues[0]["code"], "voltage_span_unusually_large")

    def test_mos_capacitance_flags_values_above_cox(self) -> None:
        metrics = {
            "voltage_range_v": [-1, 1],
            "min_capacitance_f_per_cm2": 1e-8,
            "max_capacitance_f_per_cm2": 1e-4,
            "final_capacitance_f_per_cm2": 1e-8,
        }
        issues = check_mos_capacitor_physics(metrics, {"oxide_thickness_nm": 5.0, "substrate_doping_cm3": 1e17})

        self.assertIn("capacitance_exceeds_oxide_capacitance", {issue["code"] for issue in issues})
        self.assertIn("oxide_capacitance_estimate_f_per_cm2", metrics)

    def test_mos_capacitance_flags_flat_curve_over_wide_sweep(self) -> None:
        metrics = {
            "voltage_range_v": [-2, 2],
            "min_capacitance_f_per_cm2": 4.0e-8,
            "max_capacitance_f_per_cm2": 4.01e-8,
            "final_capacitance_f_per_cm2": 4.0e-8,
        }

        issues = check_mos_capacitor_physics(metrics, {"oxide_thickness_nm": 5.0})

        self.assertIn("moscap_cv_dynamic_range_too_low", {issue["code"] for issue in issues})
        self.assertIn("capacitance_dynamic_range", metrics)

    def test_mosfet_flags_idvd_kink_and_negative_differential_segments(self) -> None:
        issues = check_mosfet_physics(
            {
                "idvd_points": 6,
                "idvd_negative_differential_segments": 1,
                "idvd_kink_slope_jumps": 1,
                "idvd_saturation_ratio": 0.9,
                "idvd_max_drain_span_v": 1.2,
            },
            {"length_um": 0.2},
            drain_start_v=0.0,
            drain_stop_v=1.2,
        )
        codes = {issue["code"] for issue in issues}

        self.assertIn("idvd_negative_differential_conductance", codes)
        self.assertIn("idvd_kink_suspected", codes)
        self.assertIn("idvd_saturation_not_observed", codes)

    def test_diode_breakdown_flags_wrong_breakdown_polarity(self) -> None:
        issues = check_diode_breakdown_physics(
            {
                "voltage_range_v": [0.0, -5.0],
                "min_reverse_voltage_v": -5.0,
                "breakdown_voltage_at_threshold_v": 2.0,
            },
            {"length_um": 0.1, "junction_um": 0.05},
        )

        self.assertIn("breakdown_voltage_wrong_polarity", {issue["code"] for issue in issues})


if __name__ == "__main__":
    unittest.main()
