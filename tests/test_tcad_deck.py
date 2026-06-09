from __future__ import annotations

import unittest

from tcad_agent.tcad_deck import (
    build_tcad_deck_spec,
    compact_tcad_deck_spec,
    parse_tcad_deck_source,
    semantic_patch_tcad_deck_source,
)


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

    def test_power_device_deck_records_open_mutations(self) -> None:
        deck = build_tcad_deck_spec(
            "这个结构漏电偏高，改 field plate / drift doping / lifetime 看看",
            "extended_device_sweep",
            {
                "device_type": "power_mosfet_bv_ron",
                "fidelity": "physics_1d",
                "power_mos_field_plate_length_um": 1.5,
                "power_mos_drift_region_doping_cm3": 1e16,
                "power_mos_carrier_lifetime_s": 1e-6,
            },
        )

        targets = {mutation["target"] for mutation in deck["planned_mutations"]}
        self.assertEqual(deck["device_family"], "power_mosfet_bv_ron")
        self.assertEqual(targets, {"field_plate", "drift_doping", "lifetime"})
        self.assertEqual(deck["physics_models"]["carrier_lifetime_s"], 1e-6)
        compact = compact_tcad_deck_spec(deck)
        self.assertEqual(len(compact["planned_mutations"]), 3)

    def test_parses_existing_devsim_deck_sections_and_semantic_patch_diff(self) -> None:
        source = "\n".join(
            [
                "field_plate_length_um = 1.5",
                "drift_region_doping_cm3 = 1e16",
                "def build_mesh():",
                "    create_1d_mesh(mesh='m', ps=0.01)",
                "set_parameter(name='carrier_lifetime_s', value=1e-6)",
                "solve(type='dc', absolute_error=1e10)",
            ]
        )

        ir = parse_tcad_deck_source(source, source_path="user_deck.py")
        section_names = {section.name for section in ir.sections}

        self.assertIn("geometry", section_names)
        self.assertIn("model", section_names)
        self.assertIn("bias", section_names)

        result = semantic_patch_tcad_deck_source(
            source,
            [
                {"deck_path": "geometry.field_plate_length_um", "request_path": "power_mos_field_plate_length_um", "value": 2.0},
                {"deck_path": "physics_models.carrier_lifetime_s", "request_path": "power_mos_carrier_lifetime_s", "value": 1e-5},
            ],
            source_path="user_deck.py",
        )

        self.assertIn("-field_plate_length_um = 1.5", result.unified_diff)
        self.assertIn("+field_plate_length_um = 2", result.unified_diff)
        self.assertIn("value=1e-05", result.patched_source)
        self.assertEqual(len(result.applied_patches), 2)

    def test_extended_mutation_vocabulary_is_verifiable(self) -> None:
        deck = build_tcad_deck_spec(
            "power MOSFET 漏电偏高，扫 guard ring、junction depth、oxide thickness、implant dose、trench corner radius、trap density、region-specific lifetime",
            "extended_device_sweep",
            {
                "device_type": "power_mosfet_bv_ron",
                "fidelity": "physics_1d",
                "power_mos_guard_ring_spacing_um": 1.0,
                "power_mos_junction_depth_um": 0.35,
                "power_mos_gate_oxide_thickness_nm": 50.0,
                "power_mos_implant_dose_cm2": 1e13,
                "power_mos_trench_corner_radius_um": 0.08,
                "power_mos_trap_density_cm2": 1e11,
                "power_mos_drift_region_lifetime_s": 1e-6,
            },
        )

        mutations = {mutation["target"]: mutation for mutation in deck["planned_mutations"]}

        self.assertIn("guard_ring", mutations)
        self.assertIn("junction_depth", mutations)
        self.assertIn("oxide_thickness", mutations)
        self.assertIn("implant_dose", mutations)
        self.assertIn("trench_corner_radius", mutations)
        self.assertIn("trap_density", mutations)
        self.assertIn("region_lifetime", mutations)
        self.assertTrue(mutations["trap_density"]["validation_metric_paths"])
        self.assertTrue(mutations["guard_ring"]["requires_user_confirmation"])

    def test_semantic_patch_handles_nested_dict_calls_mesh_and_bias(self) -> None:
        source = "\n".join(
            [
                "device = {",
                "    'geometry': {'field_plate_length_um': 1.5, 'guard_ring_spacing_um': 0.8},",
                "    'physics_models': {'regions': {'drift': {'carrier_lifetime_s': 1e-6}}, 'trap_density_cm2': 1e11},",
                "}",
                "set_parameter(device=device, name='drift_region_doping_cm3', value=1e16)",
                "create_2d_mesh(mesh='m', min_spacing=0.01, refine_at='junction')",
                "solve(type='dc', drain_voltage=20.0)",
            ]
        )

        ir = parse_tcad_deck_source(source, source_path="complex_user_deck.py")
        section_names = {section.name for section in ir.sections}

        self.assertIn("geometry", section_names)
        self.assertIn("mesh", section_names)
        self.assertIn("bias", section_names)

        result = semantic_patch_tcad_deck_source(
            source,
            [
                {"deck_path": "geometry.field_plate_length_um", "request_path": "power_mos_field_plate_length_um", "value": 2.25},
                {"deck_path": "physics_models.regions.drift.carrier_lifetime_s", "request_path": "power_mos_drift_region_lifetime_s", "value": 1e-5},
                {"deck_path": "doping.drift_region_doping_cm3", "request_path": "power_mos_drift_region_doping_cm3", "value": 8e15},
                {"deck_path": "mesh.min_spacing", "request_path": "mesh_min_spacing_um", "value": 0.005},
                {"deck_path": "bias.drain_voltage", "request_path": "drain_stop", "value": 30.0},
            ],
            source_path="complex_user_deck.py",
        )

        self.assertEqual(len(result.unapplied_patches), 0)
        self.assertIn("'field_plate_length_um': 2.25", result.patched_source)
        self.assertIn("'carrier_lifetime_s': 1e-05", result.patched_source)
        self.assertIn("value=8e+15", result.patched_source)
        self.assertIn("min_spacing=0.005", result.patched_source)
        self.assertIn("drain_voltage=30", result.patched_source)
        self.assertTrue(result.all_patches_verified)
        self.assertEqual(len(result.verified_patches), 5)

    def test_semantic_patch_aliases_devsim_named_parameters(self) -> None:
        source = "\n".join(
            [
                "set_parameter(device='dev', region='drift', name='NetDoping', value=1e16)",
                "set_parameter(device='dev', region='drift', name='taun', value=1e-6)",
            ]
        )

        result = semantic_patch_tcad_deck_source(
            source,
            [
                {"deck_path": "doping.drift_region_doping_cm3", "request_path": "power_mos_drift_region_doping_cm3", "value": 8e15},
                {"deck_path": "physics_models.carrier_lifetime_s", "request_path": "power_mos_carrier_lifetime_s", "value": 2e-6},
            ],
            source_path="named_devsim_deck.py",
        )

        self.assertTrue(result.all_patches_verified)
        self.assertEqual(len(result.unverified_patches), 0)
        self.assertIn("name='NetDoping', value=8e+15", result.patched_source)
        self.assertIn("name='taun', value=2e-06", result.patched_source)

    def test_semantic_patch_marks_fallback_append_unverified(self) -> None:
        result = semantic_patch_tcad_deck_source(
            "solve(type='dc')\n",
            {"deck_path": "geometry.field_plate_length_um", "request_path": "power_mos_field_plate_length_um", "value": 2.0},
            source_path="unmatched_deck.py",
        )

        self.assertFalse(result.all_patches_verified)
        self.assertEqual(len(result.verified_patches), 0)
        self.assertEqual(len(result.unverified_patches), 1)
        self.assertEqual(result.applied_patches[0]["semantic_status"], "unverified_fallback_append")


if __name__ == "__main__":
    unittest.main()
