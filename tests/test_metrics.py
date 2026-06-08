from __future__ import annotations

import unittest

from tcad_agent.metrics import (
    IVPoint,
    MOSFETPoint,
    extract_diode_reverse_metrics,
    extract_mosfet_metrics,
    extract_pn_iv_metrics,
)


class MetricsTest(unittest.TestCase):
    def test_extracts_pn_iv_engineering_metrics(self) -> None:
        metrics = extract_pn_iv_metrics(
            [
                IVPoint(voltage_v=-1.0, electron_current_a=0.0, hole_current_a=-1e-10, total_current_a=-1e-10),
                IVPoint(voltage_v=0.0, electron_current_a=0.0, hole_current_a=1e-12, total_current_a=1e-12),
                IVPoint(voltage_v=0.1, electron_current_a=0.0, hole_current_a=1e-8, total_current_a=1e-8),
                IVPoint(voltage_v=0.2, electron_current_a=0.0, hole_current_a=1e-6, total_current_a=1e-6),
            ],
            temperature_k=300.0,
        )

        self.assertEqual(metrics["points"], 4)
        self.assertEqual(metrics["voltage_range_v"], [-1.0, 0.2])
        self.assertAlmostEqual(metrics["leakage_current_a"], -1e-10)
        self.assertAlmostEqual(metrics["turn_on_voltage_at_1ua_v"], 0.2)
        self.assertAlmostEqual(metrics["rectification_ratio_final_to_leakage"], 1e4)
        self.assertIsNotNone(metrics["ideality_factor_estimate"])
        self.assertIsNotNone(metrics["differential_resistance_last_ohm"])

    def test_extracts_reverse_leakage_and_breakdown_metrics(self) -> None:
        points = [
            IVPoint(voltage_v=0.0, electron_current_a=0.0, hole_current_a=-1e-12, total_current_a=-1e-12),
            IVPoint(voltage_v=-0.5, electron_current_a=0.0, hole_current_a=-1e-9, total_current_a=-1e-9),
            IVPoint(voltage_v=-1.0, electron_current_a=0.0, hole_current_a=-1e-7, total_current_a=-1e-7),
            IVPoint(voltage_v=-2.0, electron_current_a=0.0, hole_current_a=-1e-5, total_current_a=-1e-5),
        ]

        metrics = extract_diode_reverse_metrics(
            points,
            leakage_voltage_v=-1.0,
            breakdown_current_a=1e-6,
        )

        self.assertEqual(metrics["reverse_points"], 3)
        self.assertAlmostEqual(metrics["leakage_abs_current_at_target_a"], 1e-7)
        self.assertTrue(metrics["breakdown_detected"])
        self.assertAlmostEqual(metrics["breakdown_voltage_at_threshold_v"], -1.0909090909090908)
        self.assertEqual(metrics["breakdown_bracket_v"], [-1.0, -2.0])
        self.assertIsNotNone(metrics["reverse_curve_knee_voltage_v"])
        self.assertIn("threshold bracket", metrics["curve_shape_summary"])
        self.assertEqual(metrics["reverse_current_shape_violations"], 0)

    def test_extracts_mosfet_id_metrics(self) -> None:
        metrics = extract_mosfet_metrics(
            [
                MOSFETPoint(
                    sweep_type="idvg",
                    gate_voltage_v=0.0,
                    drain_voltage_v=0.05,
                    drain_electron_current_a=1e-12,
                    drain_hole_current_a=0.0,
                    drain_total_current_a=1e-12,
                ),
                MOSFETPoint(
                    sweep_type="idvg",
                    gate_voltage_v=0.5,
                    drain_voltage_v=0.05,
                    drain_electron_current_a=1e-6,
                    drain_hole_current_a=0.0,
                    drain_total_current_a=1e-6,
                ),
                MOSFETPoint(
                    sweep_type="idvg",
                    gate_voltage_v=1.0,
                    drain_voltage_v=0.05,
                    drain_electron_current_a=1e-4,
                    drain_hole_current_a=0.0,
                    drain_total_current_a=1e-4,
                ),
                MOSFETPoint(
                    sweep_type="idvd",
                    gate_voltage_v=1.0,
                    drain_voltage_v=0.0,
                    drain_electron_current_a=0.0,
                    drain_hole_current_a=0.0,
                    drain_total_current_a=0.0,
                ),
                MOSFETPoint(
                    sweep_type="idvd",
                    gate_voltage_v=1.0,
                    drain_voltage_v=0.1,
                    drain_electron_current_a=2e-4,
                    drain_hole_current_a=0.0,
                    drain_total_current_a=2e-4,
                ),
            ],
            threshold_current_a=1e-6,
        )

        self.assertEqual(metrics["idvg_points"], 3)
        self.assertEqual(metrics["idvd_points"], 2)
        self.assertAlmostEqual(metrics["vth_at_threshold_current_v"], 0.5)
        self.assertAlmostEqual(metrics["ion_ioff_ratio"], 1e8)
        self.assertIsNotNone(metrics["subthreshold_swing_mv_dec"])
        self.assertIsNotNone(metrics["max_transconductance_s"])
        self.assertIsNotNone(metrics["output_conductance_last_s"])
        self.assertIn("idvd_kink_peak_voltage_v", metrics)


if __name__ == "__main__":
    unittest.main()
