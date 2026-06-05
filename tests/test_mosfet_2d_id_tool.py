from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tcad_agent.tools.mosfet_2d_id import (
    AttemptRecord,
    MOSFET2DIDRequest,
    ToolStatus,
    build_runner_command,
    judge_summary_quality,
    run_mosfet_2d_id_sweep,
)


class MOSFET2DIDToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_summary(self, run_dir: Path) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        csv_path = run_dir / "mosfet_id_sweep.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sweep_type",
                    "gate_voltage_v",
                    "drain_voltage_v",
                    "drain_electron_current_a",
                    "drain_hole_current_a",
                    "drain_total_current_a",
                    "abs_drain_current_a",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "sweep_type": "idvg",
                        "gate_voltage_v": 0.0,
                        "drain_voltage_v": 0.05,
                        "drain_electron_current_a": 1e-12,
                        "drain_hole_current_a": 0.0,
                        "drain_total_current_a": 1e-12,
                        "abs_drain_current_a": 1e-12,
                    },
                    {
                        "sweep_type": "idvg",
                        "gate_voltage_v": 1.0,
                        "drain_voltage_v": 0.05,
                        "drain_electron_current_a": 1e-4,
                        "drain_hole_current_a": 0.0,
                        "drain_total_current_a": 1e-4,
                        "abs_drain_current_a": 1e-4,
                    },
                    {
                        "sweep_type": "idvd",
                        "gate_voltage_v": 1.0,
                        "drain_voltage_v": 0.0,
                        "drain_electron_current_a": 0.0,
                        "drain_hole_current_a": 0.0,
                        "drain_total_current_a": 0.0,
                        "abs_drain_current_a": 0.0,
                    },
                    {
                        "sweep_type": "idvd",
                        "gate_voltage_v": 1.0,
                        "drain_voltage_v": 0.1,
                        "drain_electron_current_a": 2e-4,
                        "drain_hole_current_a": 0.0,
                        "drain_total_current_a": 2e-4,
                        "abs_drain_current_a": 2e-4,
                    },
                ]
            )
        for name in ["mosfet_id_curves.png", "device_tecplot.dat", "devsim.log"]:
            (run_dir / name).touch()
        summary = {
            "task": "mosfet_2d_id_sweep",
            "status": "completed",
            "metrics": {"points": 4},
            "artifacts": {
                "csv": str(csv_path),
                "plot": str(run_dir / "mosfet_id_curves.png"),
                "tecplot": str(run_dir / "device_tecplot.dat"),
                "log": str(run_dir / "devsim.log"),
            },
        }
        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return summary_path

    def test_wraps_runner_and_extracts_quality_metrics(self) -> None:
        def fake_attempt(request, state, state_path, attempt_index, gate_step, drain_step):
            run_dir = self.root / "runner" / "attempt_001"
            summary_path = self.write_summary(run_dir)
            return AttemptRecord(
                index=attempt_index,
                status=ToolStatus.COMPLETED,
                gate_step_v=gate_step,
                drain_step_v=drain_step,
                started_at="2026-01-01T00:00:00Z",
                completed_at="2026-01-01T00:00:01Z",
                command=[],
                returncode=0,
                run_dir=str(run_dir),
                summary_path=str(summary_path),
            )

        with patch("tcad_agent.tools.mosfet_2d_id.run_attempt", side_effect=fake_attempt):
            state = run_mosfet_2d_id_sweep(
                MOSFET2DIDRequest(run_id="mosfet_unit", run_root=self.root / "agent_tools")
            )

        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["quality_report"]["status"], "passed")
        self.assertAlmostEqual(state["quality_report"]["metrics"]["ion_ioff_ratio"], 1e8)
        self.assertIn("vth_at_threshold_current_v", state["quality_report"]["metrics"])
        self.assertTrue((self.root / "agent_tools" / "mosfet_2d_id" / "mosfet_unit" / "state.json").exists())

    def test_quality_report_flags_subthermal_subthreshold_swing(self) -> None:
        run_dir = self.root / "subthermal"
        run_dir.mkdir(parents=True, exist_ok=True)
        csv_path = run_dir / "mosfet_id_sweep.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sweep_type",
                    "gate_voltage_v",
                    "drain_voltage_v",
                    "drain_electron_current_a",
                    "drain_hole_current_a",
                    "drain_total_current_a",
                    "abs_drain_current_a",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "sweep_type": "idvg",
                        "gate_voltage_v": 0.0,
                        "drain_voltage_v": 0.05,
                        "drain_electron_current_a": 1e-12,
                        "drain_hole_current_a": 0.0,
                        "drain_total_current_a": 1e-12,
                        "abs_drain_current_a": 1e-12,
                    },
                    {
                        "sweep_type": "idvg",
                        "gate_voltage_v": 0.005,
                        "drain_voltage_v": 0.05,
                        "drain_electron_current_a": 1e-7,
                        "drain_hole_current_a": 0.0,
                        "drain_total_current_a": 1e-7,
                        "abs_drain_current_a": 1e-7,
                    },
                    {
                        "sweep_type": "idvg",
                        "gate_voltage_v": 0.01,
                        "drain_voltage_v": 0.05,
                        "drain_electron_current_a": 1e-3,
                        "drain_hole_current_a": 0.0,
                        "drain_total_current_a": 1e-3,
                        "abs_drain_current_a": 1e-3,
                    },
                ]
            )
        for name in ["mosfet_id_curves.png", "device_tecplot.dat", "devsim.log"]:
            (run_dir / name).touch()
        summary = {
            "task": "mosfet_2d_id_sweep",
            "status": "completed",
            "metrics": {"points": 3},
            "artifacts": {
                "csv": str(csv_path),
                "plot": str(run_dir / "mosfet_id_curves.png"),
                "tecplot": str(run_dir / "device_tecplot.dat"),
                "log": str(run_dir / "devsim.log"),
            },
        }

        report = judge_summary_quality(
            summary,
            MOSFET2DIDRequest(gate_start=0.0, gate_stop=0.01, gate_step=0.005, min_gate_step=0.001),
        )

        self.assertEqual(report["status"], "suspicious")
        self.assertIn("subthreshold_swing_below_thermal_limit", {issue["code"] for issue in report["issues"]})

    def test_runner_command_includes_physics_model_flags(self) -> None:
        request = MOSFET2DIDRequest(
            mobility_model="doping_dependent",
            recombination_model="none",
            electron_lifetime_s=1e-7,
            hole_lifetime_s=2e-7,
            interface_trap_density_cm2=1e11,
            fixed_oxide_charge_cm2=1e10,
            impact_ionization_model="selberherr",
            electron_mobility_cm2_v_s=300.0,
            solver_max_iterations=160,
        )

        command = build_runner_command(request, 1, request.gate_step, request.drain_step, self.root)

        self.assertIn("--mobility-model", command)
        self.assertIn("doping_dependent", command)
        self.assertIn("--recombination-model", command)
        self.assertIn("none", command)
        self.assertIn("--electron-mobility-cm2-v-s", command)
        self.assertIn("300.0", command)
        self.assertIn("--solver-max-iterations", command)
        self.assertIn("160", command)

    def test_quality_report_flags_metadata_only_advanced_models(self) -> None:
        run_dir = self.root / "metadata_models"
        summary_path = self.write_summary(run_dir)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["physics_models"] = {
            "interface_trap_density_cm2": 1e11,
            "fixed_oxide_charge_cm2": 1e10,
            "impact_ionization_model": "selberherr",
        }

        report = judge_summary_quality(summary, MOSFET2DIDRequest())
        codes = {issue["code"] for issue in report["issues"]}

        self.assertEqual(report["status"], "suspicious")
        self.assertIn("interface_trap_model_metadata_only", codes)
        self.assertIn("fixed_oxide_charge_metadata_only", codes)
        self.assertIn("impact_ionization_model_metadata_only", codes)

    def test_quality_report_accepts_compact_coupled_advanced_models(self) -> None:
        run_dir = self.root / "coupled_models"
        summary_path = self.write_summary(run_dir)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["physics_models"] = {
            "interface_trap_density_cm2": 1e11,
            "fixed_oxide_charge_cm2": 1e10,
            "impact_ionization_model": "selberherr",
            "advanced_model_coupling": "compact_equivalent_bias_and_avalanche",
        }

        report = judge_summary_quality(summary, MOSFET2DIDRequest())
        codes = {issue["code"] for issue in report["issues"]}

        self.assertEqual(report["status"], "passed")
        self.assertNotIn("interface_trap_model_metadata_only", codes)
        self.assertNotIn("fixed_oxide_charge_metadata_only", codes)
        self.assertNotIn("impact_ionization_model_metadata_only", codes)

    def test_quality_report_flags_idvd_kink_from_curve_shape(self) -> None:
        run_dir = self.root / "idvd_kink"
        run_dir.mkdir(parents=True, exist_ok=True)
        csv_path = run_dir / "mosfet_id_sweep.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sweep_type",
                    "gate_voltage_v",
                    "drain_voltage_v",
                    "drain_electron_current_a",
                    "drain_hole_current_a",
                    "drain_total_current_a",
                    "abs_drain_current_a",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {"sweep_type": "idvd", "gate_voltage_v": 1.0, "drain_voltage_v": 0.0, "drain_electron_current_a": 0.0, "drain_hole_current_a": 0.0, "drain_total_current_a": 0.0, "abs_drain_current_a": 0.0},
                    {"sweep_type": "idvd", "gate_voltage_v": 1.0, "drain_voltage_v": 0.2, "drain_electron_current_a": 1e-5, "drain_hole_current_a": 0.0, "drain_total_current_a": 1e-5, "abs_drain_current_a": 1e-5},
                    {"sweep_type": "idvd", "gate_voltage_v": 1.0, "drain_voltage_v": 0.4, "drain_electron_current_a": 1.2e-5, "drain_hole_current_a": 0.0, "drain_total_current_a": 1.2e-5, "abs_drain_current_a": 1.2e-5},
                    {"sweep_type": "idvd", "gate_voltage_v": 1.0, "drain_voltage_v": 0.6, "drain_electron_current_a": 9e-6, "drain_hole_current_a": 0.0, "drain_total_current_a": 9e-6, "abs_drain_current_a": 9e-6},
                    {"sweep_type": "idvd", "gate_voltage_v": 1.0, "drain_voltage_v": 0.8, "drain_electron_current_a": 4e-5, "drain_hole_current_a": 0.0, "drain_total_current_a": 4e-5, "abs_drain_current_a": 4e-5},
                    {"sweep_type": "idvd", "gate_voltage_v": 1.0, "drain_voltage_v": 1.2, "drain_electron_current_a": 9e-5, "drain_hole_current_a": 0.0, "drain_total_current_a": 9e-5, "abs_drain_current_a": 9e-5},
                ]
            )
        for name in ["mosfet_id_curves.png", "device_tecplot.dat", "devsim.log"]:
            (run_dir / name).touch()
        summary = {
            "task": "mosfet_2d_id_sweep",
            "status": "completed",
            "metrics": {"points": 6},
            "artifacts": {
                "csv": str(csv_path),
                "plot": str(run_dir / "mosfet_id_curves.png"),
                "tecplot": str(run_dir / "device_tecplot.dat"),
                "log": str(run_dir / "devsim.log"),
            },
        }

        report = judge_summary_quality(summary, MOSFET2DIDRequest(sweep_type="idvd", drain_start=0.0, drain_stop=1.2))
        codes = {issue["code"] for issue in report["issues"]}

        self.assertEqual(report["status"], "suspicious")
        self.assertIn("idvd_negative_differential_conductance", codes)
        self.assertIn("idvd_kink_suspected", codes)


if __name__ == "__main__":
    unittest.main()
