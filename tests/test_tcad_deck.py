from __future__ import annotations

import unittest

from tcad_agent.tcad_deck import build_tcad_deck_spec, compact_tcad_deck_spec


class TCADDeckSpecTest(unittest.TestCase):
    def test_builds_mosfet_deck_with_physics_and_signoff(self) -> None:
        deck = build_tcad_deck_spec(
            "客户要 signoff：NMOS Id-Vd kink，打开 interface trap density 1e11 和 avalanche",
            "mosfet_2d_id_sweep",
            {
                "sweep_type": "idvd",
                "gate_start": 0.0,
                "gate_stop": 1.2,
                "gate_step": 0.2,
                "drain_start": 0.0,
                "drain_stop": 1.2,
                "drain_step": 0.05,
                "idvd_gate_voltage": 1.2,
                "substrate_doping_cm3": 1e17,
                "source_drain_doping_cm3": 1e20,
                "interface_trap_density_cm2": 1e11,
                "impact_ionization_model": "selberherr",
                "model_strategy": "poisson_then_dd",
                "x_divisions": 12,
                "silicon_y_divisions": 4,
            },
        )

        self.assertEqual(deck["device_family"], "2d_mosfet")
        self.assertEqual(deck["signoff_requirements"]["required_level"], "engineering_signoff")
        self.assertEqual(deck["physics_models"]["coupling_status"], "needs_benchmark_confirmation")
        self.assertTrue(deck["warnings"])
        compact = compact_tcad_deck_spec(deck)
        self.assertEqual(compact["bias_sequence"][0]["name"], "Id-Vd")

    def test_builds_diode_deck_with_leakage_spec(self) -> None:
        deck = build_tcad_deck_spec(
            "PN diode -5V 漏电低于 10nA，同时提取 BV",
            "diode_breakdown_leakage_sweep",
            {
                "start": 0.0,
                "stop": -10.0,
                "step": 0.5,
                "leakage_voltage_v": -5.0,
                "quality_max_leakage_abs_current_a": 10e-9,
                "breakdown_current_a": 1e-6,
                "require_breakdown": True,
            },
        )

        self.assertEqual(deck["device_family"], "pn_diode_breakdown_leakage")
        self.assertEqual(deck["signoff_requirements"]["leakage_voltage_v"], -5.0)
        self.assertTrue(deck["signoff_requirements"]["require_breakdown"])


if __name__ == "__main__":
    unittest.main()
