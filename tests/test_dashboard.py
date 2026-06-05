from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.dashboard import DashboardStatus, generate_experiment_dashboard
from tcad_agent.reporting import ReportKind


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class DashboardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_final_state(self, name: str, current: float) -> Path:
        run_dir = self.root / "runs" / name
        plot = run_dir / "iv_curve.png"
        csv = run_dir / "iv_sweep.csv"
        log = run_dir / "devsim.log"
        plot.parent.mkdir(parents=True, exist_ok=True)
        plot.write_bytes(b"png")
        csv.write_text("voltage,current\n0,0\n", encoding="utf-8")
        log.write_text("completed\n", encoding="utf-8")
        state_path = run_dir / "state.json"
        write_json(
            state_path,
            {
                "final_summary": {
                    "artifacts": {
                        "plot": str(plot),
                        "csv": str(csv),
                        "log": str(log),
                    },
                    "final_total_current_a": current,
                    "points": 3,
                },
                "quality_report": {
                    "metrics": {
                        "final_total_current_a": current,
                        "max_abs_current_a": abs(current),
                        "points": 3,
                    }
                },
            },
        )
        return state_path

    def test_generate_optimization_dashboard(self) -> None:
        final_state = self.write_final_state("best", 1.0e-6)
        opt_dir = self.root / "opt"
        write_json(
            opt_dir / "optimization_state.json",
            {
                "tool_name": "adaptive_optimizer",
                "status": "completed",
                "optimize_id": "opt_dash",
                "axis": {
                    "path": "parameters.p_doping_cm3",
                    "scale": "log",
                    "min_value": 1e16,
                    "max_value": 1e18,
                },
                "objective": {
                    "metric_path": "final_quality_report.metrics.final_total_current_a",
                    "direction": "minimize",
                    "absolute": True,
                },
                "rounds": [
                    {
                        "index": 1,
                        "sweep_id": "round_001",
                        "status": "completed",
                        "values": [1e16, 1e17],
                        "summary_csv_path": str(opt_dir / "summary.csv"),
                        "sweep_state_path": str(opt_dir / "sweep_state.json"),
                    }
                ],
                "observations": [
                    {
                        "round_index": 1,
                        "case_index": 1,
                        "task_id": "case_001",
                        "value": 1e16,
                        "status": "completed",
                        "quality_status": "passed",
                        "objective_value": 2.0e-6,
                    },
                    {
                        "round_index": 1,
                        "case_index": 2,
                        "task_id": "case_002",
                        "value": 1e17,
                        "status": "completed",
                        "quality_status": "passed",
                        "objective_value": 1.0e-6,
                        "final_state_path": str(final_state),
                    },
                ],
                "best_observation": {
                    "round_index": 1,
                    "case_index": 2,
                    "task_id": "case_002",
                    "value": 1e17,
                    "status": "completed",
                    "quality_status": "passed",
                    "objective_value": 1.0e-6,
                    "final_state_path": str(final_state),
                },
                "next_action": "done",
            },
        )

        result = generate_experiment_dashboard(opt_dir)

        self.assertEqual(result.status, DashboardStatus.COMPLETED)
        self.assertEqual(result.kind, ReportKind.ADAPTIVE_OPTIMIZATION)
        html = Path(result.dashboard_path).read_text(encoding="utf-8")
        self.assertIn("TCAD Optimization Dashboard: opt_dash", html)
        self.assertIn("trend-chart", html)
        self.assertIn("Best IV curve", html)
        self.assertIn("iv_curve.png", html)
        self.assertIn("case_002", html)

    def test_generate_sweep_dashboard(self) -> None:
        final_state = self.write_final_state("sweep_best", 3.0e-6)
        sweep_dir = self.root / "sweep"
        write_json(
            sweep_dir / "sweep_state.json",
            {
                "tool_name": "parameter_sweep",
                "status": "completed",
                "sweep_id": "sweep_dash",
                "axes": [{"path": "parameters.p_doping_cm3", "values": [1e16, 1e17]}],
                "objective": {
                    "metric_path": "final_quality_report.metrics.final_total_current_a",
                    "direction": "minimize",
                    "absolute": True,
                },
                "cases": [
                    {
                        "index": 1,
                        "task_id": "case_001",
                        "values": {"parameters.p_doping_cm3": 1e16},
                        "status": "completed",
                        "quality_status": "passed",
                        "objective_value": 5.0e-6,
                    },
                    {
                        "index": 2,
                        "task_id": "case_002",
                        "values": {"parameters.p_doping_cm3": 1e17},
                        "status": "completed",
                        "quality_status": "passed",
                        "objective_value": 3.0e-6,
                        "final_state_path": str(final_state),
                    },
                ],
                "best_case": {
                    "index": 2,
                    "task_id": "case_002",
                    "values": {"parameters.p_doping_cm3": 1e17},
                    "status": "completed",
                    "quality_status": "passed",
                    "objective_value": 3.0e-6,
                    "final_state_path": str(final_state),
                },
            },
        )

        result = generate_experiment_dashboard(sweep_dir)

        self.assertEqual(result.status, DashboardStatus.COMPLETED)
        self.assertEqual(result.kind, ReportKind.PARAMETER_SWEEP)
        html = Path(result.dashboard_path).read_text(encoding="utf-8")
        self.assertIn("TCAD Sweep Dashboard: sweep_dash", html)
        self.assertIn("Ranked Results", html)
        self.assertIn("parameters.p_doping_cm3", html)

    def test_generate_multidim_dashboard_with_heatmap(self) -> None:
        final_state = self.write_final_state("multi_best", 1.0e-6)
        opt_dir = self.root / "multi"
        write_json(
            opt_dir / "optimization_state.json",
            {
                "tool_name": "multidim_optimizer",
                "status": "completed",
                "optimize_id": "multi_dash",
                "axes": [
                    {"path": "parameters.p_doping_cm3", "scale": "log"},
                    {"path": "parameters.junction_um", "scale": "linear"},
                ],
                "objective": {
                    "metric_path": "final_quality_report.metrics.final_total_current_a",
                    "direction": "minimize",
                    "absolute": True,
                },
                "rounds": [
                    {
                        "index": 1,
                        "round_id": "multi_dash_round_001",
                        "status": "completed",
                        "candidate_values": [
                            {"parameters.p_doping_cm3": 1e16, "parameters.junction_um": 0.04},
                            {"parameters.p_doping_cm3": 1e17, "parameters.junction_um": 0.05},
                        ],
                        "summary_csv_path": str(opt_dir / "summary.csv"),
                        "sweep_state_paths": [str(opt_dir / "sweep_state.json")],
                    }
                ],
                "observations": [
                    {
                        "round_index": 1,
                        "point_index": 1,
                        "case_index": 1,
                        "task_id": "case_001",
                        "values": {"parameters.p_doping_cm3": 1e16, "parameters.junction_um": 0.04},
                        "status": "completed",
                        "quality_status": "passed",
                        "objective_value": 2.0e-6,
                    },
                    {
                        "round_index": 1,
                        "point_index": 2,
                        "case_index": 1,
                        "task_id": "case_002",
                        "values": {"parameters.p_doping_cm3": 1e17, "parameters.junction_um": 0.05},
                        "status": "completed",
                        "quality_status": "passed",
                        "objective_value": 1.0e-6,
                        "final_state_path": str(final_state),
                    },
                ],
                "best_observation": {
                    "round_index": 1,
                    "point_index": 2,
                    "case_index": 1,
                    "task_id": "case_002",
                    "values": {"parameters.p_doping_cm3": 1e17, "parameters.junction_um": 0.05},
                    "status": "completed",
                    "quality_status": "passed",
                    "objective_value": 1.0e-6,
                    "final_state_path": str(final_state),
                },
            },
        )

        result = generate_experiment_dashboard(opt_dir)

        self.assertEqual(result.status, DashboardStatus.COMPLETED)
        self.assertEqual(result.kind, ReportKind.MULTIDIM_OPTIMIZATION)
        html = Path(result.dashboard_path).read_text(encoding="utf-8")
        self.assertIn("TCAD Multi-Dimensional Optimization Dashboard: multi_dash", html)
        self.assertIn("Objective Heatmap", html)
        self.assertIn("heatmap-chart", html)
        self.assertIn("parameters.junction_um", html)
        self.assertIn("case_002", html)


if __name__ == "__main__":
    unittest.main()
