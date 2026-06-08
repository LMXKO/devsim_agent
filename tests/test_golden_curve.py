from __future__ import annotations

import csv
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


if __name__ == "__main__":
    unittest.main()
