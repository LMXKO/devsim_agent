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

        self.assertEqual(benchmark.status, BenchmarkStatus.PASSED)
        self.assertIn("schottky_barrier_height", {check.code for check in benchmark.checks})

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


if __name__ == "__main__":
    unittest.main()
