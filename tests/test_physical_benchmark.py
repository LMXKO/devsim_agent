from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.physical_benchmark import BenchmarkStatus, run_physical_benchmark


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class PhysicalBenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_mos_capacitor_passes_cox_benchmark(self) -> None:
        state_path = self.root / "moscap" / "state.json"
        write_json(
            state_path,
            {
                "tool_name": "mos_capacitor_cv_sweep",
                "status": "completed",
                "request": {"oxide_thickness_nm": 5.0},
                "quality_report": {
                    "status": "passed",
                    "metrics": {
                        "min_capacitance_f_per_cm2": 2.0e-8,
                        "max_capacitance_f_per_cm2": 5.0e-7,
                        "final_capacitance_f_per_cm2": 3.0e-7,
                    },
                },
            },
        )

        result = run_physical_benchmark(state_path)
        codes = {check.code for check in result.checks}

        self.assertEqual(result.status, BenchmarkStatus.PASSED)
        self.assertIn("moscap_capacitance_below_cox", codes)
        self.assertEqual(result.summary["signoff_status"], "ready")
        self.assertEqual(result.summary["signoff_label_zh"], "可作为本轮工程证据")
        self.assertGreater(result.summary["confidence_score"], 0.9)
        self.assertTrue(Path(result.benchmark_path).exists())

    def test_mos_capacitor_fails_when_capacitance_exceeds_cox(self) -> None:
        state_path = self.root / "bad_moscap" / "state.json"
        write_json(
            state_path,
            {
                "tool_name": "mos_capacitor_cv_sweep",
                "status": "completed",
                "request": {"oxide_thickness_nm": 5.0},
                "quality_report": {
                    "status": "passed",
                    "metrics": {
                        "min_capacitance_f_per_cm2": 2.0e-8,
                        "max_capacitance_f_per_cm2": 2.0e-6,
                        "final_capacitance_f_per_cm2": 3.0e-7,
                    },
                },
            },
        )

        result = run_physical_benchmark(state_path)
        codes = {check.code for check in result.checks}

        self.assertEqual(result.status, BenchmarkStatus.FAILED)
        self.assertIn("moscap_capacitance_exceeds_cox", codes)
        self.assertEqual(result.summary["signoff_status"], "blocked")
        self.assertIn("moscap_capacitance_exceeds_cox", result.summary["blocking_codes"])

    def test_mosfet_flags_subthermal_subthreshold_swing(self) -> None:
        state_path = self.root / "mosfet" / "state.json"
        write_json(
            state_path,
            {
                "tool_name": "mosfet_2d_id_sweep",
                "status": "completed",
                "request": {"temperature_k": 300.0, "gate_start": 0.0, "gate_stop": 1.0},
                "quality_report": {
                    "status": "passed",
                    "metrics": {
                        "subthreshold_swing_mv_dec": 40.0,
                        "ion_ioff_ratio": 1.0e7,
                        "vth_at_threshold_current_v": 0.45,
                    },
                },
            },
        )

        result = run_physical_benchmark(state_path)

        self.assertEqual(result.status, BenchmarkStatus.SUSPICIOUS)
        self.assertIn("mosfet_subthreshold_swing_below_thermal_limit", {check.code for check in result.checks})

    def test_mosfet_flags_idvd_kink_benchmark(self) -> None:
        state_path = self.root / "mosfet_kink" / "state.json"
        write_json(
            state_path,
            {
                "tool_name": "mosfet_2d_id_sweep",
                "status": "completed",
                "request": {"drain_start": 0.0, "drain_stop": 1.2},
                "quality_report": {
                    "status": "passed",
                    "metrics": {
                        "idvd_points": 6,
                        "idvd_negative_differential_segments": 1,
                        "idvd_kink_slope_jumps": 1,
                        "idvd_saturation_ratio": 0.9,
                        "idvd_max_drain_span_v": 1.2,
                    },
                },
            },
        )

        result = run_physical_benchmark(state_path)
        codes = {check.code for check in result.checks}

        self.assertEqual(result.status, BenchmarkStatus.SUSPICIOUS)
        self.assertIn("mosfet_idvd_kink_suspected", codes)
        self.assertIn("mosfet_idvd_saturation_not_observed", codes)

    def test_deck_signoff_requires_convergence_and_records_evidence_matrix(self) -> None:
        state_path = self.root / "mosfet_signoff" / "state.json"
        write_json(
            state_path,
            {
                "tool_name": "mosfet_2d_id_sweep",
                "status": "completed",
                "request": {
                    "gate_start": 0.0,
                    "gate_stop": 1.0,
                    "tcad_deck_spec": {
                        "device_family": "2d_mosfet",
                        "physics_models": {"coupling_status": "needs_benchmark_confirmation", "interface_trap_density_cm2": 1e11},
                        "signoff_requirements": {
                            "required_level": "engineering_signoff",
                            "require_convergence_evidence": True,
                            "require_physical_benchmark": True,
                        },
                        "warnings": ["界面态模型需要确认耦合状态。"],
                    },
                },
                "quality_report": {
                    "status": "passed",
                    "metrics": {
                        "subthreshold_swing_mv_dec": 80.0,
                        "ion_ioff_ratio": 1.0e5,
                        "vth_at_threshold_current_v": 0.45,
                    },
                },
            },
        )

        result = run_physical_benchmark(state_path)
        codes = {check.code for check in result.checks}

        self.assertEqual(result.status, BenchmarkStatus.SUSPICIOUS)
        self.assertIn("deck_signoff_convergence_evidence_missing", codes)
        self.assertIn("deck_physics_model_coupling_needs_confirmation", codes)
        self.assertEqual(result.summary["signoff_status"], "conditional")
        self.assertEqual(result.summary["evidence_matrix"]["deck_spec"], "present")
        self.assertEqual(result.summary["evidence_matrix"]["convergence_evidence"], "missing")

    def test_aggregate_benchmarks_best_child_state(self) -> None:
        child_path = self.root / "child" / "state.json"
        write_json(
            child_path,
            {
                "tool_name": "diode_breakdown_leakage_sweep",
                "status": "completed",
                "request": {"quality_max_leakage_abs_current_a": 1e-6},
                "quality_report": {
                    "status": "passed",
                    "metrics": {
                        "leakage_abs_current_at_target_a": 1e-8,
                        "breakdown_voltage_at_threshold_v": -5.0,
                        "reverse_current_shape_violations": 0,
                    },
                },
            },
        )
        opt_path = self.root / "opt" / "optimization_state.json"
        write_json(
            opt_path,
            {
                "tool_name": "adaptive_optimizer",
                "status": "completed",
                "objective": {"direction": "minimize"},
                "observations": [
                    {
                        "task_id": "case_a",
                        "status": "completed",
                        "quality_status": "passed",
                        "objective_value": 1e-8,
                        "final_state_path": str(child_path),
                    }
                ],
                "best_observation": {
                    "task_id": "case_a",
                    "status": "completed",
                    "quality_status": "passed",
                    "objective_value": 1e-8,
                    "final_state_path": str(child_path),
                },
            },
        )

        result = run_physical_benchmark(opt_path.parent)
        codes = {check.code for check in result.checks}

        self.assertEqual(result.status, BenchmarkStatus.PASSED)
        self.assertIn("aggregate_has_completed_cases", codes)
        self.assertIn("best_case_diode_reverse_current_shape_ok", codes)

    def test_default_golden_profile_checks_extended_device_metrics(self) -> None:
        state_path = self.root / "schottky" / "state.json"
        write_json(
            state_path,
            {
                "tool_name": "extended_device_sweep",
                "status": "completed",
                "request": {"device_type": "schottky_diode"},
                "quality_report": {
                    "status": "passed",
                    "metrics": {
                        "device_type": "schottky_diode",
                        "barrier_height_ev": 0.72,
                        "ideality_factor_estimate": 1.08,
                    },
                },
            },
        )

        result = run_physical_benchmark(state_path)
        codes = {check.code for check in result.checks}

        self.assertEqual(result.status, BenchmarkStatus.PASSED)
        self.assertIn("golden_metric_barrier_height_ev_within_tolerance", codes)

    def test_schottky_calibration_has_physical_benchmark(self) -> None:
        state_path = self.root / "schottky_cal" / "state.json"
        write_json(
            state_path,
            {
                "tool_name": "schottky_iv_calibration",
                "status": "completed",
                "request": {"max_pass_rmse_log_current_dec": 0.15},
                "quality_report": {
                    "status": "passed",
                    "metrics": {
                        "best_barrier_height_ev": 0.72,
                        "best_ideality_factor": 1.1,
                        "best_series_resistance_ohm": 2.0,
                        "best_rmse_log_current_dec": 0.08,
                    },
                },
            },
        )

        result = run_physical_benchmark(state_path)
        codes = {check.code for check in result.checks}

        self.assertEqual(result.status, BenchmarkStatus.PASSED)
        self.assertIn("schottky_calibration_barrier_height", codes)
        self.assertIn("schottky_calibration_rmse_within_threshold", codes)

    def test_custom_golden_metric_can_fail_result(self) -> None:
        state_path = self.root / "custom_golden" / "state.json"
        write_json(
            state_path,
            {
                "tool_name": "extended_device_sweep",
                "status": "completed",
                "request": {
                    "device_type": "photodiode_iv",
                    "golden_metrics": {"responsivity_a_per_w": {"expected": 0.9, "relative_tolerance": 0.05}},
                },
                "quality_report": {
                    "status": "passed",
                    "metrics": {
                        "device_type": "photodiode_iv",
                        "photocurrent_a": 5e-7,
                        "responsivity_a_per_w": 0.5,
                    },
                },
            },
        )

        result = run_physical_benchmark(state_path)

        self.assertEqual(result.status, BenchmarkStatus.FAILED)
        self.assertIn("golden_metric_responsivity_a_per_w_far_outside_tolerance", {check.code for check in result.checks})


if __name__ == "__main__":
    unittest.main()
