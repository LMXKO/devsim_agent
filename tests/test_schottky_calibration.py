from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from tcad_agent.schottky_calibration import (
    SchottkyCalibrationRequest,
    run_schottky_calibration,
    simulate_curve,
)


class SchottkyCalibrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_builtin_trusted_curve_recovers_default_parameters(self) -> None:
        state = run_schottky_calibration(
            SchottkyCalibrationRequest(
                calibration_id="unit_builtin",
                run_root=self.root,
            )
        )

        self.assertEqual(state.status, "completed")
        self.assertEqual(state.quality_report["status"], "passed")
        self.assertIsNotNone(state.best_candidate)
        self.assertEqual(state.best_candidate.barrier_height_ev, 0.72)
        self.assertEqual(state.best_candidate.ideality_factor, 1.08)
        self.assertEqual(state.best_candidate.series_resistance_ohm, 5.0)
        self.assertEqual(state.best_candidate.image_force_lowering_ev, 0.01)
        self.assertLess(state.best_candidate.rmse_log_current_dec, 1e-12)
        self.assertTrue((Path(state.run_dir) / "target_curve.csv").exists())
        self.assertTrue((Path(state.run_dir) / "candidates.csv").exists())
        self.assertTrue((Path(state.run_dir) / "state.json").exists())

    def test_loads_custom_csv_target_curve(self) -> None:
        request = SchottkyCalibrationRequest(
            calibration_id="unit_csv",
            run_root=self.root,
            barrier_values_ev=[0.70, 0.74],
            ideality_values=[1.08],
            series_resistance_values_ohm=[0.0],
            image_force_lowering_values_ev=[0.0],
            start=-0.1,
            stop=0.2,
            step=0.1,
        )
        target = simulate_curve(
            [-0.1, 0.0, 0.1, 0.2],
            barrier_height_ev=0.74,
            ideality_factor=1.08,
            series_resistance_ohm=0.0,
            image_force_lowering_ev=0.0,
            request=request,
        )
        csv_path = self.root / "target.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["voltage_v", "current_a"])
            writer.writeheader()
            for point in target:
                writer.writerow(point.model_dump(mode="json"))

        state = run_schottky_calibration(request.model_copy(update={"target_curve_path": csv_path}))

        self.assertEqual(state.status, "completed")
        self.assertIsNotNone(state.best_candidate)
        self.assertEqual(state.best_candidate.barrier_height_ev, 0.74)


if __name__ == "__main__":
    unittest.main()
