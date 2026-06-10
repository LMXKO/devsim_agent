from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.golden_curve import GoldenCurveComparisonRequest, run_golden_curve_comparison
from tcad_agent.physical_benchmark import BenchmarkStatus, run_physical_benchmark
from tcad_agent.tools.extended_device_sweep import ExtendedDeviceRequest, ExtendedDeviceType, run_extended_device_sweep


class GoldenCurveComparisonTest(unittest.TestCase):
    def test_compares_extended_device_curve_against_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = run_extended_device_sweep(
                ExtendedDeviceRequest(
                    device_type=ExtendedDeviceType.SCHOTTKY_DIODE,
                    start=-0.1,
                    stop=0.1,
                    step=0.1,
                    run_id="source_schottky",
                    run_root=root,
                )
            )
            source_csv = Path(source.final_summary["artifacts"]["csv"])
            reference_csv = root / "reference.csv"
            with source_csv.open(newline="", encoding="utf-8") as source_handle, reference_csv.open(
                "w", newline="", encoding="utf-8"
            ) as reference_handle:
                reader = csv.DictReader(source_handle)
                writer = csv.DictWriter(reference_handle, fieldnames=["voltage_v", "current_a"])
                writer.writeheader()
                for row in reader:
                    writer.writerow({"voltage_v": row["voltage_v"], "current_a": row["current_a"]})

            comparison = run_golden_curve_comparison(
                GoldenCurveComparisonRequest(
                    comparison_id="cmp_unit",
                    source_state_path=Path(source.run_dir) / "state.json",
                    reference_curve_path=reference_csv,
                    run_root=root / "comparisons",
                )
            )

            self.assertEqual(comparison.status, "completed")
            self.assertEqual(comparison.quality_report["status"], "passed")
            self.assertEqual(comparison.quality_report["metrics"]["golden_curve_rmse_log_dec"], 0.0)

            benchmark = run_physical_benchmark(Path(comparison.comparison_dir))
            self.assertEqual(benchmark.status, BenchmarkStatus.PASSED)
            self.assertEqual(benchmark.summary["evidence_matrix"]["golden_or_measured_comparison"], "present")
            self.assertIn("golden_or_measured_comparison_present", {check.code for check in benchmark.checks})

    def test_interpolates_and_normalizes_measured_curve_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "source"
            run_dir.mkdir()
            source_csv = run_dir / "curve.csv"
            source_csv.write_text(
                "voltage_v,current_a\n0,1e-6\n1,2e-6\n2,3e-6\n",
                encoding="utf-8",
            )
            state_path = run_dir / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "tool_name": "unit_source",
                        "status": "completed",
                        "run_dir": str(run_dir),
                        "final_summary": {"artifacts": {"csv": str(source_csv)}, "metrics": {}},
                        "quality_report": {"status": "passed", "metrics": {}},
                    }
                ),
                encoding="utf-8",
            )
            reference_csv = root / "measured.csv"
            reference_csv.write_text(
                "voltage_v,current_mA\n0.5,0.0015\n1.5,0.0025\n",
                encoding="utf-8",
            )

            comparison = run_golden_curve_comparison(
                GoldenCurveComparisonRequest(
                    comparison_id="cmp_interp_units",
                    source_state_path=state_path,
                    reference_curve_path=reference_csv,
                    run_root=root / "comparisons",
                )
            )

            self.assertEqual(comparison.status, "completed")
            metrics = comparison.quality_report["metrics"]
            self.assertAlmostEqual(metrics["golden_curve_rmse_log_dec"], 0.0, places=10)
            self.assertEqual(metrics["matched_points"], 2.0)
            self.assertEqual(metrics["golden_curve_reference_y_scale"], 1e-3)
            self.assertTrue(Path(comparison.aligned_curve_path).exists())
            self.assertTrue(Path(comparison.calibration_path).exists())


if __name__ == "__main__":
    unittest.main()
