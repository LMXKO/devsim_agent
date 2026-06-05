from __future__ import annotations

import unittest

from tcad_agent.examples.mosfet_2d.run import (
    MOSFET2DParameters,
    build_gmsh_like_mesh,
    charge_coupled_gate_shift_v,
    impact_ionization_multiplier,
    physics_model_summary,
    voltage_targets,
)


class MOSFET2DRunnerTest(unittest.TestCase):
    def test_voltage_targets_decrease(self) -> None:
        self.assertEqual(voltage_targets(1.0, 0.0, 0.5), [1.0, 0.5, 0.0])

    def test_build_gmsh_like_mesh_contains_regions_contacts_and_interface(self) -> None:
        coordinates, elements, physical_names = build_gmsh_like_mesh(
            MOSFET2DParameters(x_divisions=4, silicon_y_divisions=3)
        )

        self.assertIn("oxide", physical_names)
        self.assertIn("silicon", physical_names)
        self.assertIn("gate", physical_names)
        self.assertIn("source", physical_names)
        self.assertIn("drain", physical_names)
        self.assertIn("body", physical_names)
        self.assertIn("oxide_silicon", physical_names)
        self.assertGreater(len(coordinates), 0)
        self.assertIn(2, elements)
        self.assertIn(1, elements)

    def test_physics_model_summary_supports_doping_dependent_mobility(self) -> None:
        summary = physics_model_summary(
            MOSFET2DParameters(
                mobility_model="doping_dependent",
                substrate_doping_cm3=1e17,
                recombination_model="none",
            )
        )

        self.assertEqual(summary["mobility_model_used"], "doping_dependent_effective")
        self.assertGreater(summary["electron_mobility_cm2_v_s"], summary["hole_mobility_cm2_v_s"])
        self.assertEqual(summary["electron_lifetime_s"], 1e30)

    def test_advanced_model_summary_includes_compact_coupling_terms(self) -> None:
        params = MOSFET2DParameters(
            fixed_oxide_charge_cm2=1e11,
            interface_trap_density_cm2=1e11,
            impact_ionization_model="selberherr",
        )
        summary = physics_model_summary(params)

        self.assertEqual(summary["advanced_model_coupling"], "compact_equivalent_bias_and_avalanche")
        self.assertGreater(charge_coupled_gate_shift_v(params), 0.0)
        self.assertGreater(impact_ionization_multiplier(params, 10.0), 1.0)


if __name__ == "__main__":
    unittest.main()
