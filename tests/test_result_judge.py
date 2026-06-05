from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from tcad_agent.tools.result_judge import QualityStatus, judge_pn_junction_iv


class ResultJudgeTest(unittest.TestCase):
    def write_case(self, rows: list[dict[str, str | float]]) -> dict[str, object]:
        root = Path(self.tmp.name)
        csv_path = root / "iv_sweep.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "voltage_v",
                    "electron_current_a",
                    "hole_current_a",
                    "total_current_a",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        artifacts = {
            "csv": str(csv_path),
            "log": str(root / "devsim.log"),
            "plot": str(root / "iv_curve.png"),
            "tecplot": str(root / "device_tecplot.dat"),
        }
        for path in artifacts.values():
            Path(path).touch()

        return {
            "task": "pn_junction_iv_sweep",
            "status": "completed",
            "artifacts": artifacts,
        }

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_passes_clean_forward_iv(self) -> None:
        summary = self.write_case(
            [
                {"voltage_v": 0.0, "electron_current_a": 0.0, "hole_current_a": 1e-10, "total_current_a": 1e-10},
                {"voltage_v": 0.1, "electron_current_a": 0.0, "hole_current_a": 1e-8, "total_current_a": 1e-8},
                {"voltage_v": 0.2, "electron_current_a": 0.0, "hole_current_a": 1e-6, "total_current_a": 1e-6},
            ]
        )

        report = judge_pn_junction_iv(summary)

        self.assertEqual(report.status, QualityStatus.PASSED)
        self.assertEqual(report.issues, [])

    def test_passes_clean_reverse_iv(self) -> None:
        summary = self.write_case(
            [
                {"voltage_v": 0.0, "electron_current_a": 0.0, "hole_current_a": -1e-12, "total_current_a": -1e-12},
                {"voltage_v": -0.1, "electron_current_a": 0.0, "hole_current_a": -1e-10, "total_current_a": -1e-10},
                {"voltage_v": -0.2, "electron_current_a": 0.0, "hole_current_a": -1e-8, "total_current_a": -1e-8},
            ]
        )

        report = judge_pn_junction_iv(summary)

        self.assertEqual(report.status, QualityStatus.PASSED)
        self.assertNotIn("voltage_not_monotonic", {issue.code for issue in report.issues})

    def test_marks_disordered_voltage_failed(self) -> None:
        summary = self.write_case(
            [
                {"voltage_v": 0.0, "electron_current_a": 0.0, "hole_current_a": 1e-12, "total_current_a": 1e-12},
                {"voltage_v": 0.2, "electron_current_a": 0.0, "hole_current_a": 1e-8, "total_current_a": 1e-8},
                {"voltage_v": 0.1, "electron_current_a": 0.0, "hole_current_a": 1e-7, "total_current_a": 1e-7},
            ]
        )

        report = judge_pn_junction_iv(summary)

        self.assertEqual(report.status, QualityStatus.FAILED)
        self.assertIn("voltage_not_monotonic", {issue.code for issue in report.issues})

    def test_marks_huge_current_suspicious(self) -> None:
        summary = self.write_case(
            [
                {"voltage_v": 0.0, "electron_current_a": 0.0, "hole_current_a": 1e-10, "total_current_a": 1e-10},
                {"voltage_v": 0.1, "electron_current_a": 0.0, "hole_current_a": 2.0, "total_current_a": 2.0},
                {"voltage_v": 0.2, "electron_current_a": 0.0, "hole_current_a": 3.0, "total_current_a": 3.0},
            ]
        )

        report = judge_pn_junction_iv(summary)

        self.assertEqual(report.status, QualityStatus.SUSPICIOUS)
        self.assertIn("current_exceeds_policy", {issue.code for issue in report.issues})

    def test_marks_convergence_recovery_suspicious(self) -> None:
        summary = self.write_case(
            [
                {"voltage_v": 0.0, "electron_current_a": 0.0, "hole_current_a": 1e-10, "total_current_a": 1e-10},
                {"voltage_v": 0.1, "electron_current_a": 0.0, "hole_current_a": 1e-8, "total_current_a": 1e-8},
                {"voltage_v": 0.2, "electron_current_a": 0.0, "hole_current_a": 1e-6, "total_current_a": 1e-6},
            ]
        )
        attempts = [
            {"failure_class": "convergence"},
            {"failure_class": "none"},
        ]

        report = judge_pn_junction_iv(summary, attempts=attempts)

        self.assertEqual(report.status, QualityStatus.SUSPICIOUS)
        self.assertIn("too_many_convergence_failures", {issue.code for issue in report.issues})

    def test_marks_nonfinite_current_failed(self) -> None:
        summary = self.write_case(
            [
                {"voltage_v": 0.0, "electron_current_a": 0.0, "hole_current_a": 1e-10, "total_current_a": 1e-10},
                {"voltage_v": 0.1, "electron_current_a": 0.0, "hole_current_a": "nan", "total_current_a": "nan"},
                {"voltage_v": 0.2, "electron_current_a": 0.0, "hole_current_a": 1e-6, "total_current_a": 1e-6},
            ]
        )

        report = judge_pn_junction_iv(summary)

        self.assertEqual(report.status, QualityStatus.FAILED)
        self.assertIn("nonfinite_value", {issue.code for issue in report.issues})

    def test_marks_unphysical_ideality_suspicious(self) -> None:
        summary = self.write_case(
            [
                {"voltage_v": 0.0, "electron_current_a": 0.0, "hole_current_a": 1e-12, "total_current_a": 1e-12},
                {"voltage_v": 0.1, "electron_current_a": 0.0, "hole_current_a": 1.1e-12, "total_current_a": 1.1e-12},
                {"voltage_v": 0.2, "electron_current_a": 0.0, "hole_current_a": 1.2e-12, "total_current_a": 1.2e-12},
            ]
        )

        report = judge_pn_junction_iv(summary)

        self.assertEqual(report.status, QualityStatus.SUSPICIOUS)
        self.assertIn("ideality_factor_out_of_range", {issue.code for issue in report.issues})
        self.assertIn("ideality_factor_estimate", report.metrics)

    def test_marks_temperature_unit_suspicious(self) -> None:
        summary = self.write_case(
            [
                {"voltage_v": 0.0, "electron_current_a": 0.0, "hole_current_a": 1e-10, "total_current_a": 1e-10},
                {"voltage_v": 0.1, "electron_current_a": 0.0, "hole_current_a": 1e-8, "total_current_a": 1e-8},
                {"voltage_v": 0.2, "electron_current_a": 0.0, "hole_current_a": 1e-6, "total_current_a": 1e-6},
            ]
        )
        summary["parameters"] = {"temperature_k": 27.0}

        report = judge_pn_junction_iv(summary)

        self.assertEqual(report.status, QualityStatus.SUSPICIOUS)
        self.assertIn("temperature_out_of_expected_range", {issue.code for issue in report.issues})

    def test_marks_bad_device_parameter_sanity(self) -> None:
        summary = self.write_case(
            [
                {"voltage_v": 0.0, "electron_current_a": 0.0, "hole_current_a": 1e-10, "total_current_a": 1e-10},
                {"voltage_v": 0.1, "electron_current_a": 0.0, "hole_current_a": 1e-8, "total_current_a": 1e-8},
                {"voltage_v": 0.2, "electron_current_a": 0.0, "hole_current_a": 1e-6, "total_current_a": 1e-6},
            ]
        )
        summary["parameters"] = {
            "p_doping_cm3": 1e25,
            "n_doping_cm3": 1e18,
            "length_um": 0.1,
            "junction_um": 0.2,
        }

        report = judge_pn_junction_iv(summary)
        codes = {issue.code for issue in report.issues}

        self.assertEqual(report.status, QualityStatus.FAILED)
        self.assertIn("doping_out_of_expected_range", codes)
        self.assertIn("junction_not_inside_device", codes)


if __name__ == "__main__":
    unittest.main()
