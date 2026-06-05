from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from tcad_agent.tools.diode_breakdown import (
    DiodeBreakdownRequest,
    DiodeBreakdownStatus,
    run_diode_breakdown_sweep,
)


class DiodeBreakdownToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_reverse_summary(self) -> dict[str, object]:
        run_dir = self.root / "pn_source"
        run_dir.mkdir(parents=True)
        csv_path = run_dir / "iv_sweep.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["voltage_v", "electron_current_a", "hole_current_a", "total_current_a"],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {"voltage_v": 0.0, "electron_current_a": 0.0, "hole_current_a": -1e-12, "total_current_a": -1e-12},
                    {"voltage_v": -0.5, "electron_current_a": 0.0, "hole_current_a": -1e-9, "total_current_a": -1e-9},
                    {"voltage_v": -1.0, "electron_current_a": 0.0, "hole_current_a": -1e-7, "total_current_a": -1e-7},
                    {"voltage_v": -2.0, "electron_current_a": 0.0, "hole_current_a": -1e-5, "total_current_a": -1e-5},
                ]
            )
        plot_path = run_dir / "iv_curve.png"
        log_path = run_dir / "devsim.log"
        tecplot_path = run_dir / "device_tecplot.dat"
        for path in [plot_path, log_path, tecplot_path]:
            path.touch()
        return {
            "task": "pn_junction_iv_sweep",
            "status": "completed",
            "parameters": {"temperature_k": 300.0},
            "artifacts": {
                "csv": str(csv_path),
                "plot": str(plot_path),
                "log": str(log_path),
                "tecplot": str(tecplot_path),
            },
        }

    def test_wraps_reverse_pn_sweep_and_extracts_breakdown_metrics(self) -> None:
        summary = self.write_reverse_summary()
        pn_state = {
            "status": "completed",
            "attempts": [{"failure_class": "none"}],
            "final_summary": summary,
            "quality_report": {"status": "passed", "metrics": {}},
        }

        with patch("tcad_agent.tools.diode_breakdown.run_pn_junction_iv_sweep", return_value=pn_state):
            state = run_diode_breakdown_sweep(
                DiodeBreakdownRequest(
                    run_id="diode_unit",
                    run_root=self.root / "agent_tools",
                    stop=-2.0,
                    step=0.5,
                    breakdown_current_a=1e-6,
                    leakage_voltage_v=-1.0,
                )
            )

        self.assertEqual(state["status"], DiodeBreakdownStatus.COMPLETED)
        self.assertEqual(state["quality_report"]["status"], "passed")
        metrics = state["quality_report"]["metrics"]
        self.assertTrue(metrics["breakdown_detected"])
        self.assertAlmostEqual(metrics["leakage_abs_current_at_target_a"], 1e-7)
        self.assertAlmostEqual(metrics["breakdown_voltage_at_threshold_v"], -1.0909090909090908)
        self.assertEqual(state["final_summary"]["task"], "diode_breakdown_leakage_sweep")
        self.assertTrue((self.root / "agent_tools" / "diode_breakdown" / "diode_unit" / "state.json").exists())

    def test_requires_reverse_bias_range(self) -> None:
        with self.assertRaises(ValidationError):
            DiodeBreakdownRequest(start=0.0, stop=1.0)


if __name__ == "__main__":
    unittest.main()
