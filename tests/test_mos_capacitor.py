from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tcad_agent.examples.mos_capacitor.run import (
    CVPoint,
    MOSCapacitorParameters,
    fixed_charge_voltage_shift_v,
    summarize_cv,
    voltage_targets,
)
from tcad_agent.tools.mos_capacitor_cv import (
    MOSCapacitorCVRequest,
    build_runner_command,
    classify_failure,
    create_initial_state,
    judge_summary_quality,
)


class MOSCapacitorRunnerTest(unittest.TestCase):
    def test_voltage_targets_support_descending_sweep(self) -> None:
        self.assertEqual(voltage_targets(0.5, -0.5, 0.5), [0.5, 0.0, -0.5])

    def test_summarize_cv(self) -> None:
        summary = summarize_cv(
            [
                CVPoint(gate_voltage_v=-0.5, gate_charge_c_per_cm2=-1e-8, capacitance_f_per_cm2=None),
                CVPoint(gate_voltage_v=0.0, gate_charge_c_per_cm2=1e-8, capacitance_f_per_cm2=4e-8),
                CVPoint(gate_voltage_v=0.5, gate_charge_c_per_cm2=3e-8, capacitance_f_per_cm2=4e-8),
            ]
        )

        self.assertEqual(summary["points"], 3)
        self.assertEqual(summary["voltage_range_v"], [-0.5, 0.5])
        self.assertEqual(summary["min_gate_charge_c_per_cm2"], -1e-8)
        self.assertEqual(summary["max_capacitance_f_per_cm2"], 4e-8)

    def test_tool_quality_report_passes_clean_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = {}
            for name in ["csv", "plot", "tecplot", "log"]:
                path = root / f"{name}.txt"
                path.touch()
                artifacts[name] = str(path)
            summary = {
                "points": 3,
                "voltage_range_v": [-0.5, 0.5],
                "min_gate_charge_c_per_cm2": -1e-8,
                "max_gate_charge_c_per_cm2": 2e-8,
                "min_capacitance_f_per_cm2": 1e-8,
                "max_capacitance_f_per_cm2": 4e-8,
                "final_capacitance_f_per_cm2": 2e-8,
                "artifacts": artifacts,
            }

            report = judge_summary_quality(summary, MOSCapacitorCVRequest())

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["issues"], [])

    def test_tool_quality_report_flags_capacitance_above_cox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = {}
            for name in ["csv", "plot", "tecplot", "log"]:
                path = root / f"{name}.txt"
                path.touch()
                artifacts[name] = str(path)
            summary = {
                "points": 3,
                "voltage_range_v": [-0.5, 0.5],
                "parameters": {"oxide_thickness_nm": 5.0, "substrate_doping_cm3": 1e17},
                "min_gate_charge_c_per_cm2": -1e-8,
                "max_gate_charge_c_per_cm2": 2e-8,
                "min_capacitance_f_per_cm2": 1e-8,
                "max_capacitance_f_per_cm2": 1e-4,
                "final_capacitance_f_per_cm2": 2e-8,
                "artifacts": artifacts,
            }

            report = judge_summary_quality(summary, MOSCapacitorCVRequest())

        self.assertEqual(report["status"], "suspicious")
        self.assertIn("capacitance_exceeds_oxide_capacitance", {issue["code"] for issue in report["issues"]})
        self.assertIn("oxide_capacitance_estimate_f_per_cm2", report["metrics"])

    def test_fixed_oxide_charge_records_equivalent_voltage_shift(self) -> None:
        shift = fixed_charge_voltage_shift_v(MOSCapacitorParameters(oxide_thickness_nm=5.0, fixed_oxide_charge_cm2=5e11))

        self.assertGreater(shift, 0.0)
        self.assertAlmostEqual(shift, 0.116, delta=0.01)

    def test_tool_command_includes_fixed_oxide_charge(self) -> None:
        command = build_runner_command(
            MOSCapacitorCVRequest(fixed_oxide_charge_cm2=5e11),
            attempt_index=1,
            step=0.25,
            run_dir=Path("/tmp/moscap"),
        )

        self.assertIn("--fixed-oxide-charge-cm2", command)
        self.assertIn("500000000000.0", command)

    def test_initial_state_preserves_tcad_deck_spec(self) -> None:
        deck = {"device_family": "mos_capacitor", "signoff_requirements": {"required_level": "engineering_signoff"}}
        state = create_initial_state(
            MOSCapacitorCVRequest(tcad_deck_spec=deck),
            run_id="moscap_deck",
            run_dir=Path("/tmp/moscap_deck"),
        )

        self.assertEqual(state.tcad_deck_spec, deck)
        self.assertEqual(state.request["tcad_deck_spec"], deck)

    def test_classifies_convergence_failure(self) -> None:
        failure_class, _ = classify_failure(1, "maximum_iterations exceeded", "")

        self.assertEqual(failure_class.value, "convergence")


if __name__ == "__main__":
    unittest.main()
