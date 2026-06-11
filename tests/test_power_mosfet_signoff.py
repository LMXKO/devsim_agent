from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tcad_agent.agent_experiment_design import build_agent_experiment_design_plan
from tcad_agent.power_mosfet_signoff import PowerMOSFETSignoffRequest, run_power_mosfet_signoff


class PowerMOSFETSignoffTest(unittest.TestCase):
    def test_planned_signoff_pack_records_missing_evidence_without_failing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = run_power_mosfet_signoff(
                PowerMOSFETSignoffRequest(
                    run_id="power_plan",
                    run_root=Path(tmp),
                    execute=False,
                )
            )

        self.assertEqual(state.status, "planned")
        self.assertEqual(state.quality_report["status"], "planned")
        self.assertIn("golden_or_measured_correlation", state.signoff_gate["missing_evidence"])
        self.assertIn("planned_baseline_request", state.artifacts)

    def test_agent_experiment_design_selects_power_signoff_for_2d_gap_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "tool_name": "extended_device_sweep",
                        "status": "completed",
                        "run_id": "power_2d",
                        "request": {
                            "device_type": "power_mosfet_bv_ron",
                            "fidelity": "devsim_2d_field_plate",
                        },
                        "quality_report": {"status": "suspicious", "metrics": {}},
                        "final_summary": {
                            "metrics": {
                                "device_type": "power_mosfet_bv_ron",
                                "fidelity": "devsim_2d_field_plate",
                                "signoff_gaps": ["mesh_convergence", "golden_or_measured_correlation"],
                            },
                            "artifacts": {"csv": str(Path(tmp) / "curve.csv")},
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            with patch("tcad_agent.agent_experiment_design.benchmark_for_state") as benchmark:
                fake = Mock()
                fake.summary = {"signoff_evidence_pack": {"missing_evidence": [], "verdict": "conditional"}}
                fake.benchmark_path = None
                benchmark.return_value = fake
                plan = build_agent_experiment_design_plan(state_path)

        self.assertEqual(plan.selected_candidate.candidate_id, "power_mosfet_2d_signoff_evidence_pack")
        self.assertEqual(plan.selected_candidate.tool_name, "power_mosfet_signoff")

    def test_agent_experiment_design_refines_when_mutation_effect_helped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "mutation_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "tool_name": "extended_device_sweep",
                        "status": "completed",
                        "run_id": "mutation",
                        "request": {"device_type": "power_mosfet_bv_ron", "fidelity": "physics_1d"},
                        "quality_report": {"status": "passed", "metrics": {}},
                        "final_summary": {"metrics": {"device_type": "power_mosfet_bv_ron", "fidelity": "physics_1d"}},
                        "mutation_effect_analysis": {
                            "decision": "continue_refine",
                            "worth_continuing": True,
                            "primary_metric": "leakage_current_a",
                            "recommended_next_target": "lifetime",
                            "improved_metrics": ["leakage_current_a"],
                            "regressed_metrics": [],
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            with patch("tcad_agent.agent_experiment_design.benchmark_for_state") as benchmark:
                fake = Mock()
                fake.summary = {"signoff_evidence_pack": {"missing_evidence": [], "verdict": "conditional"}}
                fake.benchmark_path = None
                benchmark.return_value = fake
                plan = build_agent_experiment_design_plan(state_path)

        self.assertEqual(plan.selected_candidate.candidate_id, "refine_effective_mutation_direction")
        self.assertEqual(plan.selected_candidate.action_kind, "plan_mutation_refinement")
        self.assertEqual(plan.curve_engineering_review["decision"], "continue_refine")


if __name__ == "__main__":
    unittest.main()
