from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.reporting import ReportKind, ReportStatus, generate_experiment_report, resolve_state_path


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class ReportingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_final_state(self, name: str, current: float) -> Path:
        run_dir = self.root / "runs" / name
        plot = run_dir / "iv_curve.png"
        csv = run_dir / "iv_sweep.csv"
        plot.parent.mkdir(parents=True, exist_ok=True)
        plot.write_bytes(b"png")
        csv.write_text("voltage,current\n0,0\n", encoding="utf-8")
        state_path = run_dir / "state.json"
        write_json(
            state_path,
            {
                "final_summary": {
                    "artifacts": {
                        "plot": str(plot),
                        "csv": str(csv),
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

    def test_generate_optimization_report(self) -> None:
        final_state = self.write_final_state("best", 1.2e-6)
        opt_dir = self.root / "opt"
        state_path = opt_dir / "optimization_state.json"
        write_json(
            state_path,
            {
                "tool_name": "adaptive_optimizer",
                "status": "completed",
                "optimize_id": "opt_report",
                "execute": True,
                "axis": {"path": "parameters.p_doping_cm3", "scale": "log"},
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
                        "objective_value": 1.2e-6,
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
                    "objective_value": 1.2e-6,
                    "final_state_path": str(final_state),
                },
                "next_action": "done",
            },
        )

        result = generate_experiment_report(opt_dir)

        self.assertEqual(result.status, ReportStatus.COMPLETED)
        self.assertEqual(result.kind, ReportKind.ADAPTIVE_OPTIMIZATION)
        report = Path(result.report_path).read_text(encoding="utf-8")
        self.assertIn("# TCAD Optimization Report: opt_report", report)
        self.assertIn("case_002", report)
        self.assertIn("![Best IV curve]", report)
        self.assertIn("iv_curve.png", report)
        self.assertEqual(resolve_state_path(opt_dir), state_path)

    def test_generate_sweep_report(self) -> None:
        final_state = self.write_final_state("sweep_best", 3.4e-6)
        sweep_dir = self.root / "sweep"
        state_path = sweep_dir / "sweep_state.json"
        write_json(
            state_path,
            {
                "tool_name": "parameter_sweep",
                "status": "completed",
                "sweep_id": "sweep_report",
                "execute": True,
                "axes": [{"path": "parameters.p_doping_cm3", "values": [1e16, 1e17]}],
                "objective": {
                    "metric_path": "final_quality_report.metrics.final_total_current_a",
                    "direction": "minimize",
                    "absolute": True,
                },
                "summary_csv_path": str(sweep_dir / "summary.csv"),
                "cases": [
                    {
                        "index": 1,
                        "task_id": "case_001",
                        "values": {"parameters.p_doping_cm3": 1e16},
                        "status": "completed",
                        "quality_status": "passed",
                        "objective_value": 4.0e-6,
                    },
                    {
                        "index": 2,
                        "task_id": "case_002",
                        "values": {"parameters.p_doping_cm3": 1e17},
                        "status": "completed",
                        "quality_status": "passed",
                        "objective_value": 3.4e-6,
                        "final_state_path": str(final_state),
                    },
                ],
                "best_case": {
                    "index": 2,
                    "task_id": "case_002",
                    "values": {"parameters.p_doping_cm3": 1e17},
                    "status": "completed",
                    "quality_status": "passed",
                    "objective_value": 3.4e-6,
                    "final_state_path": str(final_state),
                },
            },
        )

        result = generate_experiment_report(state_path)

        self.assertEqual(result.status, ReportStatus.COMPLETED)
        self.assertEqual(result.kind, ReportKind.PARAMETER_SWEEP)
        report = Path(result.report_path).read_text(encoding="utf-8")
        self.assertIn("# TCAD Sweep Report: sweep_report", report)
        self.assertIn("Ranked Cases", report)
        self.assertIn("case_002", report)

    def test_generate_multidim_optimization_report(self) -> None:
        final_state = self.write_final_state("multi_best", 2.1e-6)
        opt_dir = self.root / "multi"
        state_path = opt_dir / "optimization_state.json"
        write_json(
            state_path,
            {
                "tool_name": "multidim_optimizer",
                "status": "completed",
                "optimize_id": "multi_report",
                "execute": True,
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
                        "round_id": "multi_report_round_001",
                        "status": "completed",
                        "candidate_values": [{"parameters.p_doping_cm3": 1e17, "parameters.junction_um": 0.05}],
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
                        "values": {"parameters.p_doping_cm3": 1e17, "parameters.junction_um": 0.05},
                        "status": "completed",
                        "quality_status": "passed",
                        "objective_value": 2.1e-6,
                        "final_state_path": str(final_state),
                    }
                ],
                "best_observation": {
                    "round_index": 1,
                    "point_index": 1,
                    "case_index": 1,
                    "task_id": "case_001",
                    "values": {"parameters.p_doping_cm3": 1e17, "parameters.junction_um": 0.05},
                    "status": "completed",
                    "quality_status": "passed",
                    "objective_value": 2.1e-6,
                    "final_state_path": str(final_state),
                },
            },
        )

        result = generate_experiment_report(opt_dir)

        self.assertEqual(result.status, ReportStatus.COMPLETED)
        self.assertEqual(result.kind, ReportKind.MULTIDIM_OPTIMIZATION)
        report = Path(result.report_path).read_text(encoding="utf-8")
        self.assertIn("# TCAD Multi-Dimensional Optimization Report: multi_report", report)
        self.assertIn("parameters.junction_um", report)
        self.assertIn("Ranked Observations", report)
        self.assertIn("case_001", report)


if __name__ == "__main__":
    unittest.main()
