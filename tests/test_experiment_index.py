from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.experiment_index import list_records, rebuild_index


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class ExperimentIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / "index.sqlite"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_rebuild_indexes_known_state_types(self) -> None:
        write_json(
            self.root / "tasks" / "task_a" / "task_run_state.json",
            {
                "tool_name": "tcad_task_runner",
                "task_id": "task_a",
                "status": "completed",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:01:00Z",
                "final_quality_report": {
                    "status": "passed",
                    "metrics": {"final_total_current_a": 1e-6},
                },
            },
        )
        write_json(
            self.root / "sweeps" / "sweep_a" / "sweep_state.json",
            {
                "tool_name": "parameter_sweep",
                "sweep_id": "sweep_a",
                "status": "completed",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:02:00Z",
                "axes": [{"path": "parameters.p_doping_cm3"}],
                "best_case": {
                    "values": {"parameters.p_doping_cm3": 1e17},
                    "objective_value": 2e-6,
                    "quality_status": "passed",
                },
            },
        )
        write_json(
            self.root / "optimizations" / "opt_a" / "optimization_state.json",
            {
                "tool_name": "adaptive_optimizer",
                "optimize_id": "opt_a",
                "status": "completed",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:03:00Z",
                "axis": {"path": "parameters.p_doping_cm3"},
                "best_observation": {
                    "value": 3e17,
                    "objective_value": 3e-6,
                    "quality_status": "passed",
                },
            },
        )
        write_json(
            self.root / "agent_tools" / "mos_capacitor_cv" / "mos_a" / "state.json",
            {
                "tool_name": "mos_capacitor_cv_sweep",
                "run_id": "mos_a",
                "status": "completed",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:04:00Z",
                "quality_report": {
                    "status": "passed",
                    "metrics": {"final_capacitance_f_per_cm2": 4e-8},
                },
            },
        )
        write_json(
            self.root / "optimizations" / "multi_a" / "optimization_state.json",
            {
                "tool_name": "multidim_optimizer",
                "optimize_id": "multi_a",
                "status": "completed",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:05:00Z",
                "axes": [
                    {"path": "parameters.p_doping_cm3"},
                    {"path": "parameters.junction_um"},
                ],
                "best_observation": {
                    "values": {"parameters.p_doping_cm3": 1e17, "parameters.junction_um": 0.05},
                    "objective_value": 5e-6,
                    "quality_status": "passed",
                },
            },
        )
        write_json(
            self.root / "agent_tools" / "diode_breakdown" / "bd_a" / "state.json",
            {
                "tool_name": "diode_breakdown_leakage_sweep",
                "run_id": "bd_a",
                "status": "completed",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:06:00Z",
                "quality_report": {
                    "status": "passed",
                    "metrics": {"leakage_abs_current_at_target_a": 7e-10},
                },
            },
        )
        write_json(
            self.root / "mesh_convergence" / "mesh_a" / "state.json",
            {
                "tool_name": "mesh_convergence",
                "run_id": "mesh_a",
                "status": "completed",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:07:00Z",
                "quality_report": {
                    "status": "passed",
                    "metrics": {"relative_delta": 0.012},
                },
            },
        )
        write_json(
            self.root / "agent_tools" / "mosfet_2d_id" / "fet_a" / "state.json",
            {
                "tool_name": "mosfet_2d_id_sweep",
                "run_id": "fet_a",
                "status": "completed",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:08:00Z",
                "quality_report": {
                    "status": "passed",
                    "metrics": {"ion_ioff_ratio": 1e5},
                },
            },
        )
        write_json(
            self.root / "tool_convergence" / "conv_a" / "state.json",
            {
                "tool_name": "tool_convergence",
                "convergence_id": "conv_a",
                "status": "completed",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:09:00Z",
                "quality_report": {
                    "status": "passed",
                    "metrics": {"relative_delta": 0.02},
                },
            },
        )
        write_json(
            self.root / "agent_tools" / "mosfet_2d_id" / "fet_a" / "benchmark.json",
            {
                "tool_name": "physical_benchmark",
                "status": "passed",
                "source_state_path": str(self.root / "agent_tools" / "mosfet_2d_id" / "fet_a" / "state.json"),
                "source_tool_name": "mosfet_2d_id_sweep",
                "summary": {"generated_at": "2026-01-01T00:10:00Z", "counts": {"pass": 3, "warning": 0, "error": 0}},
                "checks": [],
            },
        )
        write_json(
            self.root / "agent_tools" / "extended_devices" / "schottky" / "state.json",
            {
                "tool_name": "extended_device_sweep",
                "run_id": "schottky_a",
                "status": "completed",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:11:00Z",
                "quality_report": {
                    "status": "passed",
                    "metrics": {"device_type": "schottky_diode", "barrier_height_ev": 0.72},
                },
            },
        )
        write_json(
            self.root / "optimizations" / "eng_a" / "engineering_objectives.json",
            {
                "tool_name": "engineering_objective_evaluation",
                "status": "completed",
                "best_candidate": {"candidate_id": "case_a", "score": -1e6},
            },
        )
        write_json(
            self.root / "agent_tools" / "schottky_calibration" / "cal_a" / "state.json",
            {
                "tool_name": "schottky_iv_calibration",
                "calibration_id": "cal_a",
                "status": "completed",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:12:00Z",
                "quality_report": {
                    "status": "passed",
                    "metrics": {"best_rmse_log_current_dec": 0.01},
                },
            },
        )

        result = rebuild_index(self.root, self.db)
        records = list_records(self.db)
        optimizations = list_records(self.db, kind="adaptive_optimization")
        multidim = list_records(self.db, kind="multidim_optimization")
        mos_tools = list_records(self.db, kind="mos_capacitor_cv_sweep")
        diode_tools = list_records(self.db, kind="diode_breakdown_leakage_sweep")
        mesh_tools = list_records(self.db, kind="mesh_convergence")
        mosfet_tools = list_records(self.db, kind="mosfet_2d_id_sweep")
        tool_convergence = list_records(self.db, kind="tool_convergence")
        benchmarks = list_records(self.db, kind="physical_benchmark")
        extended = list_records(self.db, kind="extended_device_sweep")
        engineering = list_records(self.db, kind="engineering_objective_evaluation")
        calibration = list_records(self.db, kind="schottky_iv_calibration")

        self.assertEqual(result["records_indexed"], 13)
        self.assertEqual(len(records), 13)
        self.assertEqual(len(optimizations), 1)
        self.assertEqual(len(multidim), 1)
        self.assertEqual(len(mos_tools), 1)
        self.assertEqual(len(diode_tools), 1)
        self.assertEqual(len(mesh_tools), 1)
        self.assertEqual(len(mosfet_tools), 1)
        self.assertEqual(len(tool_convergence), 1)
        self.assertEqual(len(benchmarks), 1)
        self.assertEqual(len(extended), 1)
        self.assertEqual(len(engineering), 1)
        self.assertEqual(len(calibration), 1)
        self.assertEqual(optimizations[0]["experiment_id"], "opt_a")
        self.assertEqual(optimizations[0]["best_axis_path"], "parameters.p_doping_cm3")
        self.assertEqual(optimizations[0]["best_axis_value"], 3e17)
        self.assertEqual(multidim[0]["experiment_id"], "multi_a")
        self.assertEqual(multidim[0]["objective_value"], 5e-6)
        self.assertIn("parameters.junction_um", multidim[0]["best_axis_path"])
        self.assertEqual(mos_tools[0]["objective_value"], 4e-8)
        self.assertEqual(diode_tools[0]["objective_value"], 7e-10)
        self.assertEqual(mesh_tools[0]["objective_value"], 0.012)
        self.assertEqual(mosfet_tools[0]["objective_value"], 1e5)
        self.assertEqual(tool_convergence[0]["objective_value"], 0.02)
        self.assertEqual(benchmarks[0]["quality_status"], "passed")
        self.assertEqual(extended[0]["objective_value"], 0.72)
        self.assertEqual(engineering[0]["quality_status"], "feasible")


if __name__ == "__main__":
    unittest.main()
