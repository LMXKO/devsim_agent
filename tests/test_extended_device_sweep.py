from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tcad_agent.physical_benchmark import BenchmarkStatus, run_physical_benchmark
from tcad_agent.tools.extended_device_sweep import (
    ExtendedDeviceFidelity,
    ExtendedDeviceRequest,
    ExtendedDeviceStatus,
    ExtendedDeviceType,
    run_extended_device_sweep,
)


class ExtendedDeviceSweepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_device(self, device_type: ExtendedDeviceType):
        return run_extended_device_sweep(
            ExtendedDeviceRequest(device_type=device_type, run_id=f"unit_{device_type.value}", run_root=self.root)
        )

    def test_runs_all_extended_device_templates(self) -> None:
        expected_metric = {
            ExtendedDeviceType.SCHOTTKY_DIODE: "barrier_height_ev",
            ExtendedDeviceType.BJT_GUMMEL_OUTPUT: "current_gain_beta",
            ExtendedDeviceType.JFET_TRANSFER_OUTPUT: "pinch_off_voltage_v",
            ExtendedDeviceType.POWER_MOSFET_BV_RON: "specific_on_resistance_ohm_cm2",
            ExtendedDeviceType.PHOTODIODE_IV: "responsivity_a_per_w",
        }

        for device_type, metric in expected_metric.items():
            with self.subTest(device_type=device_type):
                state = self.run_device(device_type)
                metrics = state.quality_report["metrics"]
                artifacts = state.final_summary["artifacts"]

                self.assertEqual(state.status, ExtendedDeviceStatus.COMPLETED)
                self.assertEqual(state.quality_report["status"], "passed")
                self.assertIn(metric, metrics)
                self.assertTrue(Path(artifacts["csv"]).exists())
                self.assertTrue((Path(state.run_dir) / "state.json").exists())

    def test_physical_benchmark_supports_extended_device(self) -> None:
        state = self.run_device(ExtendedDeviceType.SCHOTTKY_DIODE)

        benchmark = run_physical_benchmark(Path(state.run_dir))

        self.assertEqual(benchmark.status, BenchmarkStatus.SUSPICIOUS)
        self.assertIn("schottky_barrier_height", {check.code for check in benchmark.checks})
        self.assertIn("compact_baseline_not_signoff_evidence", {check.code for check in benchmark.checks})

    def test_schottky_devsim_fidelity_uses_runner_artifacts(self) -> None:
        def fake_run(command, **kwargs):
            run_root = Path(command[command.index("--run-root") + 1])
            run_id = command[command.index("--run-id") + 1]
            runner_dir = run_root / "schottky_1d" / run_id
            runner_dir.mkdir(parents=True, exist_ok=True)
            (runner_dir / "sweep.csv").write_text(
                "\n".join(
                    [
                        "voltage_v,devsim_total_current_a,current_a,abs_current_a",
                        "-0.1,-1e-7,-1e-13,1e-13",
                        "0.0,1e-12,1e-18,1e-18",
                        "0.1,1e-6,1e-12,1e-12",
                    ]
                ),
                encoding="utf-8",
            )
            (runner_dir / "device_tecplot.dat").write_text("tecplot", encoding="utf-8")
            (runner_dir / "devsim.log").write_text("solver log", encoding="utf-8")
            summary = {
                "metrics": {
                    "device_type": "schottky_diode",
                    "fidelity": "devsim_1d",
                    "solver_backend": "devsim_1d_thermionic_emission_contact_model",
                    "schottky_contact_model": "thermionic_emission",
                    "schottky_contact_coupling_mode": "residual",
                    "thermionic_residual_coupled": True,
                    "points": 3,
                    "barrier_height_ev": 0.72,
                    "ideality_factor_estimate": 1.08,
                    "reverse_leakage_current_a": 1e-13,
                    "devsim_thermionic_contact_current_max_abs_a": 1e-12,
                }
            }
            (runner_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

            class Completed:
                returncode = 0
                stdout = json.dumps({"status": "completed", "run_dir": str(runner_dir)})
                stderr = ""

            return Completed()

        with patch("tcad_agent.tools.extended_device_sweep.subprocess.run", side_effect=fake_run):
            state = run_extended_device_sweep(
                ExtendedDeviceRequest(
                    device_type=ExtendedDeviceType.SCHOTTKY_DIODE,
                    fidelity=ExtendedDeviceFidelity.DEVSIM_1D,
                    start=-0.1,
                    stop=0.1,
                    step=0.1,
                    run_id="unit_schottky_devsim",
                    run_root=self.root,
                )
            )

        self.assertEqual(state.status, ExtendedDeviceStatus.COMPLETED)
        self.assertEqual(state.final_summary["fidelity"], "devsim_1d")
        self.assertTrue(state.quality_report["metrics"]["tcad_solver_invoked"])
        self.assertEqual(
            state.quality_report["metrics"]["solver_backend"],
            "devsim_1d_thermionic_emission_contact_model",
        )
        self.assertIn("tecplot", state.final_summary["artifacts"])

        benchmark = run_physical_benchmark(Path(state.run_dir))
        self.assertEqual(benchmark.status, BenchmarkStatus.PASSED)
        codes = {check.code for check in benchmark.checks}
        self.assertIn("schottky_devsim_solver_invoked", codes)
        self.assertIn("schottky_thermionic_residual_coupled", codes)

    def test_bjt_physics_fidelity_has_three_terminal_evidence(self) -> None:
        state = run_extended_device_sweep(
            ExtendedDeviceRequest(
                device_type=ExtendedDeviceType.BJT_GUMMEL_OUTPUT,
                fidelity=ExtendedDeviceFidelity.PHYSICS_1D,
                run_id="unit_bjt_physics",
                run_root=self.root,
            )
        )

        self.assertEqual(state.status, ExtendedDeviceStatus.COMPLETED)
        self.assertEqual(state.quality_report["status"], "passed")
        metrics = state.quality_report["metrics"]
        self.assertEqual(metrics["evidence_level"], "tcad_executable")
        self.assertTrue(metrics["equation_coupled_transport"])
        self.assertTrue(metrics["three_terminal_output_family"])
        self.assertTrue(metrics["mesh_resolved_geometry"])
        self.assertTrue(metrics["doping_profile_defined"])
        self.assertEqual(state.tcad_deck_spec["device_family"], "bjt_gummel_output")
        self.assertEqual(state.tcad_deck_spec["physics_models"]["coupling_status"], "equation_coupled")
        self.assertIn("emitter", {region["name"] for region in state.tcad_deck_spec["regions"]})
        self.assertIn("junction_spacing_um", state.tcad_deck_spec["mesh"])
        self.assertTrue(Path(state.final_summary["artifacts"]["tcad_deck_spec"]).exists())

        benchmark = run_physical_benchmark(Path(state.run_dir))
        self.assertEqual(benchmark.status, BenchmarkStatus.PASSED)
        self.assertEqual(benchmark.summary["evidence_matrix"]["deck_spec"], "present")
        self.assertEqual(benchmark.summary["evidence_matrix"]["model_coupling_risk"], "not_detected")
        self.assertNotIn("structured_tcad_spec", benchmark.summary["signoff_evidence_pack"]["missing_evidence"])
        codes = {check.code for check in benchmark.checks}
        self.assertIn("bjt_physics_transport_coupled", codes)
        self.assertIn("bjt_three_terminal_output_family_present", codes)
        self.assertIn("bjt_mesh_resolved_deck_present", codes)
        self.assertNotIn("compact_baseline_not_signoff_evidence", codes)

    def test_power_mosfet_physics_fidelity_has_avalanche_evidence(self) -> None:
        state = run_extended_device_sweep(
            ExtendedDeviceRequest(
                device_type=ExtendedDeviceType.POWER_MOSFET_BV_RON,
                fidelity=ExtendedDeviceFidelity.PHYSICS_1D,
                run_id="unit_power_physics",
                run_root=self.root,
            )
        )

        self.assertEqual(state.status, ExtendedDeviceStatus.COMPLETED)
        self.assertEqual(state.quality_report["status"], "passed")
        metrics = state.quality_report["metrics"]
        self.assertEqual(metrics["evidence_level"], "tcad_executable")
        self.assertTrue(metrics["impact_ionization_coupled"])
        self.assertIn("drift_specific_on_resistance_ohm_cm2", metrics)
        self.assertTrue(metrics["mesh_resolved_drift_region"])
        self.assertTrue(metrics["field_plate_geometry_defined"])
        self.assertEqual(state.tcad_deck_spec["device_family"], "power_mosfet_bv_ron")
        self.assertEqual(state.tcad_deck_spec["physics_models"]["coupling_status"], "equation_coupled")
        self.assertIn("field_plate", state.tcad_deck_spec["contacts"])
        self.assertIn("field_plate_edge", state.tcad_deck_spec["mesh"]["refined_regions"])
        self.assertTrue(Path(state.final_summary["artifacts"]["tcad_deck_spec"]).exists())
        self.assertTrue(Path(state.final_summary["artifacts"]["generated_deck"]).exists())
        self.assertTrue(Path(state.final_summary["artifacts"]["deck_request"]).exists())
        self.assertEqual(metrics["carrier_lifetime_s"], 1.0e-6)

        benchmark = run_physical_benchmark(Path(state.run_dir))
        self.assertEqual(benchmark.status, BenchmarkStatus.PASSED)
        self.assertEqual(benchmark.summary["evidence_matrix"]["deck_spec"], "present")
        self.assertNotIn("structured_tcad_spec", benchmark.summary["signoff_evidence_pack"]["missing_evidence"])
        codes = {check.code for check in benchmark.checks}
        self.assertIn("power_mos_impact_ionization_coupled", codes)
        self.assertIn("power_mos_ron_components_present", codes)
        self.assertIn("power_mos_mesh_resolved_deck_present", codes)
        self.assertNotIn("compact_baseline_not_signoff_evidence", codes)

    def test_power_mosfet_lifetime_mutation_changes_leakage_and_writes_history(self) -> None:
        short_lifetime = run_extended_device_sweep(
            ExtendedDeviceRequest(
                device_type=ExtendedDeviceType.POWER_MOSFET_BV_RON,
                fidelity=ExtendedDeviceFidelity.PHYSICS_1D,
                start=0.0,
                stop=-30.0,
                step=5.0,
                power_mos_carrier_lifetime_s=1e-7,
                tcad_deck_mutations=[
                    {
                        "name": "sweep_power_carrier_lifetime",
                        "target": "lifetime",
                        "request_path": "power_mos_carrier_lifetime_s",
                        "deck_path": "physics_models.carrier_lifetime_s",
                        "values": [1e-7, 1e-6, 1e-5],
                    }
                ],
                run_id="unit_power_lifetime_short",
                run_root=self.root,
            )
        )
        long_lifetime = run_extended_device_sweep(
            ExtendedDeviceRequest(
                device_type=ExtendedDeviceType.POWER_MOSFET_BV_RON,
                fidelity=ExtendedDeviceFidelity.PHYSICS_1D,
                start=0.0,
                stop=-30.0,
                step=5.0,
                power_mos_carrier_lifetime_s=1e-5,
                run_id="unit_power_lifetime_long",
                run_root=self.root,
            )
        )

        self.assertGreater(
            short_lifetime.quality_report["metrics"]["leakage_current_a"],
            long_lifetime.quality_report["metrics"]["leakage_current_a"],
        )
        self.assertTrue(Path(short_lifetime.final_summary["artifacts"]["tcad_deck_mutations"]).exists())
        self.assertTrue(Path(short_lifetime.final_summary["artifacts"]["deck_patch_history"]).exists())

    def test_remaining_extended_devices_have_physics_fidelity_benchmarks(self) -> None:
        expected_codes = {
            ExtendedDeviceType.JFET_TRANSFER_OUTPUT: "jfet_depletion_model_coupled",
            ExtendedDeviceType.PHOTODIODE_IV: "photodiode_optical_generation_coupled",
            ExtendedDeviceType.FINFET_ID_CV: "finfet_density_gradient_coupled",
            ExtendedDeviceType.SIC_POWER_DIODE_BV_LEAKAGE: "sic_impact_ionization_coupled",
            ExtendedDeviceType.GAN_HEMT_ID_BV: "gan_polarization_charge_coupled",
            ExtendedDeviceType.IGBT_OUTPUT_TURNOFF: "igbt_transient_turnoff_simulated",
        }

        for device_type, expected_code in expected_codes.items():
            with self.subTest(device_type=device_type):
                state = run_extended_device_sweep(
                    ExtendedDeviceRequest(
                        device_type=device_type,
                        fidelity=ExtendedDeviceFidelity.PHYSICS_1D,
                        run_id=f"unit_{device_type.value}_physics",
                        run_root=self.root,
                    )
                )

                self.assertEqual(state.status, ExtendedDeviceStatus.COMPLETED)
                self.assertEqual(state.quality_report["status"], "passed")
                self.assertEqual(state.quality_report["metrics"]["evidence_level"], "tcad_executable")
                self.assertEqual(state.tcad_deck_spec["device_family"], device_type.value)

                benchmark = run_physical_benchmark(Path(state.run_dir))
                self.assertEqual(benchmark.status, BenchmarkStatus.PASSED)
                self.assertEqual(benchmark.summary["evidence_matrix"]["capability_boundary"], "tcad_executable")
                codes = {check.code for check in benchmark.checks}
                self.assertIn(expected_code, codes)
                self.assertNotIn("compact_baseline_not_signoff_evidence", codes)
                self.assertNotIn("planned_industrial_template_runner_missing", codes)


if __name__ == "__main__":
    unittest.main()
