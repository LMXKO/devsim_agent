from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tcad_agent.tool_convergence import (
    ToolConvergenceQuality,
    ToolConvergenceRequest,
    ToolConvergenceStatus,
    run_tool_convergence,
)


class ToolConvergenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def runner_from_values(self, metrics: dict[object, float]):
        def runner(request: dict[str, object]) -> dict[str, object]:
            run_dir = Path(str(request["run_root"])) / "fake_tool" / str(request["run_id"])
            run_dir.mkdir(parents=True, exist_ok=True)
            metric = metrics[request["x_divisions"]]
            state = {
                "tool_name": "mosfet_2d_id_sweep",
                "status": "completed",
                "run_id": request["run_id"],
                "run_dir": str(run_dir),
                "request": request,
                "quality_report": {
                    "status": "passed",
                    "issues": [],
                    "metrics": {"ion_ioff_ratio": metric},
                },
            }
            (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
            return state

        return runner

    def request(self, **overrides: object) -> ToolConvergenceRequest:
        data = {
            "convergence_id": "conv_unit",
            "tool_name": "mosfet_2d_id_sweep",
            "base_request": {"run_id": "mos_base", "x_divisions": 4},
            "axis_path": "x_divisions",
            "values": [4, 8, 12],
            "metric_path": "quality_report.metrics.ion_ioff_ratio",
            "relative_tolerance": 0.05,
            "execute": True,
            "overwrite": True,
            "convergence_root": self.root,
        }
        data.update(overrides)
        return ToolConvergenceRequest.model_validate(data)

    def test_dry_run_plans_cases(self) -> None:
        state = run_tool_convergence(self.request(execute=False))

        self.assertEqual(state.status, ToolConvergenceStatus.COMPLETED)
        self.assertEqual(state.quality_report["status"], ToolConvergenceQuality.PLANNED)
        self.assertEqual(len(state.cases), 3)

    def test_passes_when_last_two_metrics_are_close(self) -> None:
        state = run_tool_convergence(
            self.request(),
            registry={"mosfet_2d_id_sweep": self.runner_from_values({4: 1.0, 8: 1.02, 12: 1.03})},
        )

        self.assertEqual(state.status, ToolConvergenceStatus.COMPLETED)
        self.assertEqual(state.quality_report["status"], ToolConvergenceQuality.PASSED)
        self.assertLess(state.quality_report["metrics"]["relative_delta"], 0.05)
        self.assertTrue((self.root / "conv_unit" / "state.json").exists())

    def test_marks_suspicious_when_last_two_metrics_differ(self) -> None:
        state = run_tool_convergence(
            self.request(),
            registry={"mosfet_2d_id_sweep": self.runner_from_values({4: 1.0, 8: 1.0, 12: 2.0})},
        )

        self.assertEqual(state.status, ToolConvergenceStatus.COMPLETED)
        self.assertEqual(state.quality_report["status"], ToolConvergenceQuality.SUSPICIOUS)
        self.assertIn("tool_not_converged", {issue["code"] for issue in state.quality_report["issues"]})

    def test_default_registry_accepts_model_dump_results(self) -> None:
        fake_state = Mock()
        fake_state.model_dump.return_value = {
            "tool_name": "schottky_iv_calibration",
            "status": "completed",
            "run_dir": str(self.root / "schottky_case"),
            "quality_report": {
                "status": "passed",
                "metrics": {"best_rmse_log_current_dec": 0.0},
            },
        }
        request = self.request(
            tool_name="schottky_iv_calibration",
            base_request={"start": -0.2, "stop": 0.4, "step": 0.1},
            axis_path="step",
            values=[0.2, 0.1],
            metric_path="quality_report.metrics.best_rmse_log_current_dec",
        )

        with patch("tcad_agent.tool_convergence.run_schottky_calibration", return_value=fake_state):
            state = run_tool_convergence(request)

        self.assertEqual(state.status, ToolConvergenceStatus.COMPLETED)
        self.assertEqual(state.quality_report["status"], ToolConvergenceQuality.PASSED)
        self.assertEqual([case["status"] for case in state.cases], ["completed", "completed"])

    def test_normalizes_mosfet_llm_aliases_before_execution(self) -> None:
        request = self.request(
            base_request={
                "sweep_type": "output_characteristic",
                "gate_values": [0.8, 1.0, 1.2],
                "drain_start": 0.0,
                "drain_stop": 1.2,
                "drain_step": 0.05,
            },
            axis_path="mesh_refinement_level",
            values=[1, 2, 3],
            metric_path="simulation_results.id_saturation",
        )

        self.assertEqual(request.base_request["sweep_type"], "idvd")
        self.assertEqual(request.base_request["idvd_gate_voltage"], 1.2)
        self.assertNotIn("gate_values", request.base_request)
        self.assertEqual(request.axis_path, "x_divisions")
        self.assertEqual(request.values, [8, 12, 16])
        self.assertEqual(request.metric_path, "quality_report.metrics.idvd_final_current_a")

    def test_mosfet_dibl_split_metric_from_drain_voltage_cases(self) -> None:
        def runner(request: dict[str, object]) -> dict[str, object]:
            drain_voltage = float(request["drain_voltage"])
            vth = 0.45 if drain_voltage < 0.5 else 0.39
            return {
                "tool_name": "mosfet_2d_id_sweep",
                "status": "completed",
                "run_id": request["run_id"],
                "run_dir": str(self.root / "fake_dibl" / str(request["run_id"])),
                "quality_report": {
                    "status": "passed",
                    "metrics": {"vth_at_threshold_current_v": vth},
                },
            }

        state = run_tool_convergence(
            self.request(
                base_request={"run_id": "dibl_base", "sweep_type": "idvg", "drain_voltage": 0.05},
                axis_path="drain_voltage",
                values=[0.05, 1.0],
                metric_path="quality_report.metrics.vth_at_threshold_current_v",
                relative_tolerance=1.0,
            ),
            registry={"mosfet_2d_id_sweep": runner},
        )

        metrics = state.quality_report["metrics"]
        self.assertEqual(metrics["engineering_metric"], "dibl")
        self.assertAlmostEqual(metrics["dibl_mv_per_v"], (0.45 - 0.39) / 0.95 * 1000.0)


if __name__ == "__main__":
    unittest.main()
