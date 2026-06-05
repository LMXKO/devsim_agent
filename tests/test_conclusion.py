from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.conclusion import ConclusionStatus, generate_experiment_conclusion


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class ConclusionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_generates_optimization_conclusion(self) -> None:
        state_path = self.root / "opt" / "optimization_state.json"
        write_json(
            state_path,
            {
                "tool_name": "adaptive_optimizer",
                "optimize_id": "opt_conclusion",
                "status": "completed",
                "axis": {"path": "parameters.p_doping_cm3"},
                "objective": {"direction": "minimize"},
                "observations": [
                    {
                        "task_id": "case_001",
                        "value": 1e16,
                        "status": "completed",
                        "quality_status": "passed",
                        "objective_value": 1e-4,
                    },
                    {
                        "task_id": "case_002",
                        "value": 1e17,
                        "status": "completed",
                        "quality_status": "passed",
                        "objective_value": 1e-5,
                    },
                ],
                "best_observation": {
                    "task_id": "case_002",
                    "value": 1e17,
                    "status": "completed",
                    "quality_status": "passed",
                    "objective_value": 1e-5,
                },
            },
        )

        result = generate_experiment_conclusion(self.root / "opt")

        self.assertEqual(result.status, ConclusionStatus.COMPLETED)
        text = Path(result.conclusion_path).read_text(encoding="utf-8")
        self.assertIn("# TCAD 工程结论：opt_conclusion", text)
        self.assertIn("最优任务/结果", text)
        self.assertIn("趋势解读", text)
        self.assertIn("工程判断", text)
        self.assertIn("物理可信度检查", text)
        self.assertIn("下一轮实验计划", text)
        self.assertIn("继续做若干轮自适应优化", text)

    def test_generates_mos_tool_conclusion(self) -> None:
        state_path = self.root / "mos" / "state.json"
        write_json(
            state_path,
            {
                "tool_name": "mos_capacitor_cv_sweep",
                "run_id": "mos_conclusion",
                "status": "completed",
                "quality_report": {
                    "status": "passed",
                    "metrics": {
                        "points": 3,
                        "final_capacitance_f_per_cm2": 1.2e-7,
                        "max_capacitance_f_per_cm2": 1.7e-7,
                    },
                },
            },
        )

        result = generate_experiment_conclusion(state_path)

        self.assertEqual(result.status, ConclusionStatus.COMPLETED)
        text = Path(result.conclusion_path).read_text(encoding="utf-8")
        self.assertIn("mos_conclusion", text)
        self.assertIn("final_capacitance_f_per_cm2", text)
        self.assertIn("物理 benchmark 状态", text)
        self.assertIn("签核状态", text)
        self.assertIn("置信分数", text)
        self.assertIn("可信度等级", text)
        self.assertIn("缺失证据", text)
        self.assertIn("benchmark 建议", text)
        self.assertIn("扫描氧化层厚度", text)

    def test_generates_multidim_optimization_conclusion(self) -> None:
        state_path = self.root / "multi" / "optimization_state.json"
        write_json(
            state_path,
            {
                "tool_name": "multidim_optimizer",
                "optimize_id": "multi_conclusion",
                "status": "completed",
                "axes": [
                    {"path": "parameters.p_doping_cm3"},
                    {"path": "parameters.junction_um"},
                ],
                "objective": {"direction": "minimize"},
                "observations": [
                    {
                        "task_id": "case_001",
                        "values": {"parameters.p_doping_cm3": 1e16, "parameters.junction_um": 0.04},
                        "status": "completed",
                        "quality_status": "passed",
                        "objective_value": 2e-5,
                    },
                    {
                        "task_id": "case_002",
                        "values": {"parameters.p_doping_cm3": 1e17, "parameters.junction_um": 0.05},
                        "status": "completed",
                        "quality_status": "passed",
                        "objective_value": 1e-5,
                    },
                ],
                "best_observation": {
                    "task_id": "case_002",
                    "values": {"parameters.p_doping_cm3": 1e17, "parameters.junction_um": 0.05},
                    "status": "completed",
                    "quality_status": "passed",
                    "objective_value": 1e-5,
                },
            },
        )

        result = generate_experiment_conclusion(self.root / "multi")

        self.assertEqual(result.status, ConclusionStatus.COMPLETED)
        text = Path(result.conclusion_path).read_text(encoding="utf-8")
        self.assertIn("# TCAD 工程结论：multi_conclusion", text)
        self.assertIn("多参数响应面", text)
        self.assertIn("parameters.junction_um", text)
        self.assertIn("多维局部细化", text)

    def test_generates_diode_breakdown_conclusion(self) -> None:
        state_path = self.root / "diode" / "state.json"
        write_json(
            state_path,
            {
                "tool_name": "diode_breakdown_leakage_sweep",
                "run_id": "diode_conclusion",
                "status": "completed",
                "quality_report": {
                    "status": "passed",
                    "metrics": {
                        "leakage_abs_current_at_target_a": 7e-10,
                        "breakdown_voltage_at_threshold_v": -12.5,
                        "max_reverse_abs_current_a": 2e-6,
                        "reverse_current_shape_violations": 0,
                    },
                },
            },
        )

        result = generate_experiment_conclusion(state_path)

        self.assertEqual(result.status, ConclusionStatus.COMPLETED)
        text = Path(result.conclusion_path).read_text(encoding="utf-8")
        self.assertIn("diode_conclusion", text)
        self.assertIn("breakdown_voltage_at_threshold_v", text)
        self.assertIn("扩展反偏范围", text)

    def test_generates_mesh_convergence_conclusion(self) -> None:
        state_path = self.root / "mesh" / "state.json"
        write_json(
            state_path,
            {
                "tool_name": "mesh_convergence",
                "convergence_id": "mesh_conclusion",
                "status": "completed",
                "quality_report": {
                    "status": "passed",
                    "metrics": {
                        "relative_delta": 0.012,
                        "relative_tolerance": 0.05,
                        "finest_mesh_value": 5e-6,
                        "previous_mesh_value": 1e-5,
                        "finest_objective": 1.0e-6,
                        "previous_objective": 1.01e-6,
                    },
                },
            },
        )

        result = generate_experiment_conclusion(state_path)

        self.assertEqual(result.status, ConclusionStatus.COMPLETED)
        text = Path(result.conclusion_path).read_text(encoding="utf-8")
        self.assertIn("mesh_conclusion", text)
        self.assertIn("relative_delta", text)
        self.assertIn("沿用当前网格", text)

    def test_generates_mosfet_2d_conclusion(self) -> None:
        state_path = self.root / "mosfet" / "state.json"
        write_json(
            state_path,
            {
                "tool_name": "mosfet_2d_id_sweep",
                "run_id": "mosfet_conclusion",
                "status": "completed",
                "quality_report": {
                    "status": "passed",
                    "metrics": {
                        "vth_at_threshold_current_v": 0.45,
                        "subthreshold_swing_mv_dec": 90.0,
                        "ion_current_a": 1e-4,
                        "ioff_current_a": 1e-12,
                        "ion_ioff_ratio": 1e8,
                        "max_transconductance_s": 2e-4,
                        "idvd_final_current_a": 2e-4,
                        "output_conductance_last_s": 1e-3,
                    },
                },
            },
        )

        result = generate_experiment_conclusion(state_path)

        self.assertEqual(result.status, ConclusionStatus.COMPLETED)
        text = Path(result.conclusion_path).read_text(encoding="utf-8")
        self.assertIn("mosfet_conclusion", text)
        self.assertIn("ion_ioff_ratio", text)
        self.assertIn("MOSFET 网格/模型验证", text)
        self.assertIn("复核 Vth", text)


if __name__ == "__main__":
    unittest.main()
