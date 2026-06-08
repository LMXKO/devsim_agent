from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.repair_strategy import RepairPlanStatus, build_repair_plan, repair_request


class RepairStrategyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_state(self, data: dict[str, object]) -> Path:
        path = self.root / "state.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_convergence_failure_plans_tcad_repair_sequence(self) -> None:
        state_path = self.write_state(
            {
                "tool_name": "pn_junction_iv_sweep",
                "status": "failed",
                "run_id": "pn_bad",
                "request": {
                    "start": 0.0,
                    "stop": 5.0,
                    "step": 1.0,
                    "min_step": 0.25,
                    "max_attempts": 2,
                    "contact_spacing_um": 0.001,
                    "junction_spacing_um": 1e-5,
                    "solver_max_iterations": 80,
                },
                "attempts": [{"failure_class": "convergence"}],
                "quality_report": {"status": "failed", "issues": []},
            }
        )

        plan = build_repair_plan(state_path)
        actions = {action.name: action for action in plan.actions}

        self.assertEqual(plan.status, RepairPlanStatus.PLANNED)
        self.assertEqual(plan.next_action, "continuation_bias_ramp")
        self.assertEqual(actions["continuation_bias_ramp"].request_patch["step"], 0.5)
        self.assertEqual(actions["continuation_bias_ramp"].request_patch["max_attempts"], 5)
        self.assertIn("solver_parameter_adjustment", actions)
        self.assertGreaterEqual(actions["solver_parameter_adjustment"].request_patch["solver_max_iterations"], 160)
        self.assertIn("solver_relative_error", actions["solver_parameter_adjustment"].request_patch)
        self.assertIn("reuse_last_successful_initial_solution", actions)
        self.assertTrue(actions["reuse_last_successful_initial_solution"].request_patch["resume"])
        self.assertIn("model_switch_staging", actions)
        self.assertEqual(actions["model_switch_staging"].request_patch["model_strategy"], "poisson_then_dd")
        self.assertTrue((self.root / "repair_plan.json").exists())

    def test_mesh_and_geometry_issues_plan_specific_repairs(self) -> None:
        state_path = self.write_state(
            {
                "tool_name": "pn_junction_iv_sweep",
                "status": "completed",
                "run_id": "pn_suspicious",
                "request": {
                    "length_um": 0.1,
                    "junction_um": 0.2,
                    "junction_spacing_um": 2e-5,
                    "contact_spacing_um": 1e-3,
                    "x_divisions": 8,
                },
                "quality_report": {
                    "status": "suspicious",
                    "issues": [
                        {"code": "mesh_not_converged", "severity": "warning"},
                        {"code": "junction_not_inside_device", "severity": "error"},
                    ],
                },
            }
        )

        plan = build_repair_plan(state_path)
        actions = {action.name: action for action in plan.actions}

        self.assertIn("mesh_refinement_and_convergence_check", actions)
        self.assertEqual(actions["mesh_refinement_and_convergence_check"].request_patch["junction_spacing_um"], 1e-5)
        self.assertGreater(actions["mesh_refinement_and_convergence_check"].request_patch["x_divisions"], 8)
        self.assertIn("geometry_sanity_repair", actions)
        self.assertEqual(actions["geometry_sanity_repair"].request_patch["junction_um"], 0.05)
        self.assertTrue(actions["geometry_sanity_repair"].user_confirmation_required)

    def test_mosfet_threshold_issue_extends_gate_sweep(self) -> None:
        state_path = self.write_state(
            {
                "tool_name": "mosfet_2d_id_sweep",
                "status": "completed",
                "run_id": "mos_bad",
                "request": {"gate_stop": 0.5},
                "quality_report": {
                    "status": "suspicious",
                    "issues": [{"code": "mosfet_threshold_not_crossed", "severity": "warning"}],
                },
            }
        )

        plan = build_repair_plan(state_path)
        actions = {action.name: action for action in plan.actions}

        self.assertIn("mosfet_sweep_range_extension", actions)
        self.assertEqual(actions["mosfet_sweep_range_extension"].request_patch["gate_stop"], 1.0)

    def test_validation_failure_plans_schema_normalization(self) -> None:
        state_path = self.write_state(
            {
                "tool_name": "mosfet_2d_id_sweep",
                "status": "failed",
                "run_id": "schema_bad",
                "request": {"sweep_type": "output_characteristic"},
                "attempts": [{"failure_class": "validation"}],
                "quality_report": {"status": "failed", "issues": []},
            }
        )

        plan = build_repair_plan(state_path)
        actions = {action.name: action for action in plan.actions}

        self.assertIn("schema_field_alias_normalization", actions)
        self.assertEqual(plan.next_action, "schema_field_alias_normalization")
        self.assertEqual(actions["schema_field_alias_normalization"].request_patch["sweep_type"], "idvd")

    def test_repair_request_inherits_top_level_tcad_deck_spec(self) -> None:
        deck = {"device_family": "2d_mosfet"}
        request = repair_request(
            {
                "tool_name": "mosfet_2d_id_sweep",
                "request": {"gate_stop": 0.5},
                "tcad_deck_spec": deck,
            }
        )

        self.assertEqual(request["tcad_deck_spec"], deck)

    def test_mosfet_output_kink_plans_output_triage(self) -> None:
        state_path = self.write_state(
            {
                "tool_name": "mosfet_2d_id_sweep",
                "status": "completed",
                "run_id": "kink_bad",
                "request": {"drain_step": 0.1, "min_drain_step": 0.025},
                "quality_report": {
                    "status": "suspicious",
                    "issues": [{"code": "idvd_kink_suspected", "severity": "warning"}],
                },
            }
        )

        plan = build_repair_plan(state_path)
        actions = {action.name: action for action in plan.actions}

        self.assertIn("mosfet_output_physics_triage", actions)
        self.assertIn("local_bias_step_refinement", actions)

    def test_moscap_fixed_charge_issue_plans_bias_window_review(self) -> None:
        state_path = self.write_state(
            {
                "tool_name": "mos_capacitor_cv_sweep",
                "status": "completed",
                "run_id": "moscap_charge",
                "request": {"start": -0.5, "stop": 0.5, "step": 0.25},
                "quality_report": {
                    "status": "suspicious",
                    "issues": [{"code": "fixed_charge_shift_exceeds_sweep_window", "severity": "warning"}],
                },
            }
        )

        plan = build_repair_plan(state_path)
        actions = {action.name: action for action in plan.actions}

        self.assertIn("moscap_bias_and_charge_window_review", actions)
        self.assertEqual(actions["moscap_bias_and_charge_window_review"].request_patch["start"], -2.0)

    def test_moscap_cox_benchmark_issue_plans_unit_reconciliation(self) -> None:
        state_path = self.write_state(
            {
                "tool_name": "mos_capacitor_cv_sweep",
                "status": "completed",
                "run_id": "moscap_cox",
                "request": {"oxide_thickness_nm": 5.0},
                "quality_report": {
                    "status": "suspicious",
                    "issues": [{"code": "moscap_capacitance_exceeds_cox", "severity": "error"}],
                },
            }
        )

        plan = build_repair_plan(state_path)
        actions = {action.name: action for action in plan.actions}

        self.assertIn("analytic_cox_unit_reconciliation", actions)
        self.assertTrue(actions["analytic_cox_unit_reconciliation"].user_confirmation_required)

    def test_diode_breakdown_and_leakage_issues_plan_reverse_and_lifetime_repairs(self) -> None:
        state_path = self.write_state(
            {
                "tool_name": "diode_breakdown_leakage_sweep",
                "status": "completed",
                "run_id": "diode_leak",
                "request": {"start": 0.0, "stop": -5.0, "step": 0.5, "electron_lifetime_s": 1e-8},
                "quality_report": {
                    "status": "suspicious",
                    "issues": [
                        {"code": "diode_breakdown_not_reached", "severity": "warning"},
                        {"code": "diode_leakage_above_policy", "severity": "warning"},
                    ],
                },
            }
        )

        plan = build_repair_plan(state_path)
        actions = {action.name: action for action in plan.actions}

        self.assertIn("extend_reverse_bias_window", actions)
        self.assertLess(actions["extend_reverse_bias_window"].request_patch["stop"], -5.0)
        self.assertIn("srh_lifetime_and_boundary_sanity", actions)
        self.assertEqual(actions["srh_lifetime_and_boundary_sanity"].request_patch["electron_lifetime_s"], 1e-7)

    def test_tool_convergence_failure_uses_failed_case_request_and_target_tool(self) -> None:
        state_path = self.write_state(
            {
                "tool_name": "tool_convergence",
                "status": "failed",
                "convergence_id": "conv_bad",
                "target_tool": "mosfet_2d_id_sweep",
                "axis_path": "x_divisions",
                "values": [8, 12, 16],
                "cases": [
                    {
                        "index": 1,
                        "status": "failed",
                        "failure_reason": "DEVSIM solver did not converge.",
                        "request": {
                            "sweep_type": "idvd",
                            "drain_step": 0.1,
                            "min_drain_step": 0.025,
                            "gate_step": 0.2,
                            "min_gate_step": 0.05,
                        },
                    }
                ],
                "quality_report": {
                    "status": "failed",
                    "issues": [{"code": "too_few_completed_convergence_cases", "severity": "error"}],
                },
            }
        )

        plan = build_repair_plan(state_path)
        actions = {action.name: action for action in plan.actions}

        self.assertEqual(plan.next_action, "rerun_failed_convergence_cases_with_safe_bias")
        self.assertIn("rerun_failed_convergence_cases_with_safe_bias", actions)
        self.assertEqual(actions["rerun_failed_convergence_cases_with_safe_bias"].target_tool, "mosfet_2d_id_sweep")
        self.assertLess(actions["rerun_failed_convergence_cases_with_safe_bias"].request_patch["drain_step"], 0.1)
        self.assertTrue(actions["rerun_failed_convergence_cases_with_safe_bias"].request_patch["resume"])

    def test_deck_signoff_and_model_coupling_issues_plan_repairs(self) -> None:
        state_path = self.write_state(
            {
                "tool_name": "mosfet_2d_id_sweep",
                "status": "completed",
                "run_id": "mos_signoff",
                "request": {
                    "gate_step": 0.2,
                    "min_gate_step": 0.05,
                    "drain_step": 0.1,
                    "min_drain_step": 0.025,
                    "x_divisions": 8,
                    "silicon_y_divisions": 3,
                },
                "quality_report": {
                    "status": "suspicious",
                    "issues": [
                        {"code": "deck_signoff_convergence_evidence_missing", "severity": "warning"},
                        {"code": "deck_physics_model_coupling_needs_confirmation", "severity": "warning"},
                    ],
                },
            }
        )

        plan = build_repair_plan(state_path)
        actions = {action.name: action for action in plan.actions}

        self.assertIn("signoff_evidence_density_retry", actions)
        self.assertLess(actions["signoff_evidence_density_retry"].request_patch["gate_step"], 0.2)
        self.assertGreater(actions["signoff_evidence_density_retry"].request_patch["x_divisions"], 8)
        self.assertIn("model_coupling_and_extraction_review", actions)

    def test_compact_baseline_issue_requires_runner_promotion(self) -> None:
        state_path = self.write_state(
            {
                "tool_name": "extended_device_sweep",
                "status": "completed",
                "run_id": "compact_power",
                "request": {"device_type": "power_mosfet_bv_ron"},
                "quality_report": {
                    "status": "suspicious",
                    "issues": [{"code": "compact_baseline_not_signoff_evidence", "severity": "warning"}],
                },
            }
        )

        plan = build_repair_plan(state_path)
        actions = {action.name: action for action in plan.actions}

        self.assertIn("promote_compact_baseline_to_tcad_runner", actions)
        self.assertTrue(actions["promote_compact_baseline_to_tcad_runner"].user_confirmation_required)

    def test_deck_mutation_repairs_are_planned_for_power_leakage_issue(self) -> None:
        state_path = self.write_state(
            {
                "tool_name": "extended_device_sweep",
                "status": "completed",
                "run_id": "power_leak",
                "request": {
                    "device_type": "power_mosfet_bv_ron",
                    "fidelity": "physics_1d",
                    "power_mos_field_plate_length_um": 1.5,
                    "tcad_deck_mutations": [
                        {
                            "name": "sweep_field_plate_length",
                            "target": "field_plate",
                            "request_path": "power_mos_field_plate_length_um",
                            "deck_path": "geometry.field_plate_length_um",
                            "values": [0.9, 1.5, 2.1],
                            "reason": "vary field plate",
                        }
                    ],
                },
                "quality_report": {
                    "status": "suspicious",
                    "issues": [{"code": "power_mos_leakage_above_policy", "severity": "warning"}],
                },
            }
        )

        plan = build_repair_plan(state_path)
        actions = {action.name: action for action in plan.actions}

        self.assertIn("deck_mutation_field_plate", actions)
        self.assertEqual(actions["deck_mutation_field_plate"].request_patch["power_mos_field_plate_length_um"], 0.9)
        self.assertEqual(actions["deck_mutation_field_plate"].deck_patch["deck_path"], "geometry.field_plate_length_um")

    def test_curve_guided_mutation_refines_effective_direction(self) -> None:
        state_path = self.write_state(
            {
                "tool_name": "extended_device_sweep",
                "status": "completed",
                "run_id": "power_field_plate_repair",
                "request": {
                    "device_type": "power_mosfet_bv_ron",
                    "fidelity": "physics_1d",
                    "power_mos_field_plate_length_um": 2.1,
                    "deck_patch_history": [
                        {
                            "request_path": "power_mos_field_plate_length_um",
                            "deck_path": "geometry.field_plate_length_um",
                            "value": 2.1,
                        }
                    ],
                    "tcad_deck_mutations": [
                        {
                            "name": "sweep_field_plate_length",
                            "target": "field_plate",
                            "request_path": "power_mos_field_plate_length_um",
                            "deck_path": "geometry.field_plate_length_um",
                            "values": [0.9, 1.5, 2.1],
                            "reason": "vary field plate",
                        },
                        {
                            "name": "sweep_power_carrier_lifetime",
                            "target": "lifetime",
                            "request_path": "power_mos_carrier_lifetime_s",
                            "deck_path": "physics_models.carrier_lifetime_s",
                            "values": [1e-7, 1e-6, 1e-5],
                            "reason": "vary lifetime",
                        },
                    ],
                },
                "mutation_effect_analysis": {
                    "worth_continuing": True,
                    "recommended_next_target": "field_plate",
                    "baseline_value": 1.5,
                    "mutation_value": 2.1,
                    "decision": "continue_same_target",
                    "rationale": "max_electric_field_v_per_cm improved without blocking tradeoffs",
                },
                "quality_report": {
                    "status": "suspicious",
                    "issues": [{"code": "power_mos_field_exceeds_critical_margin", "severity": "warning"}],
                },
            }
        )

        plan = build_repair_plan(state_path)
        actions = {action.name: action for action in plan.actions}

        self.assertIn("deck_mutation_field_plate", actions)
        self.assertAlmostEqual(actions["deck_mutation_field_plate"].request_patch["power_mos_field_plate_length_um"], 2.4)
        self.assertEqual(actions["deck_mutation_field_plate"].deck_patch["curve_guided_decision"], "continue_same_target")

    def test_solver_log_signature_plans_initialization_backoff(self) -> None:
        state_path = self.write_state(
            {
                "tool_name": "mosfet_2d_id_sweep",
                "status": "failed",
                "run_id": "solver_log",
                "request": {"sweep_type": "idvd", "drain_step": 0.1, "gate_step": 0.2},
                "attempts": [{"failure_class": "runtime_exception", "failure_reason": "Newton failed: singular matrix and NaN residual"}],
                "quality_report": {"status": "failed", "issues": []},
            }
        )

        plan = build_repair_plan(state_path)
        actions = {action.name: action for action in plan.actions}

        self.assertIn("solver_initialization_bias_backoff", actions)
        self.assertEqual(actions["solver_initialization_bias_backoff"].request_patch["model_strategy"], "poisson_then_dd")
        self.assertEqual(
            actions["solver_initialization_bias_backoff"].request_patch["initial_condition_strategy"],
            "zero_bias_poisson_then_ramp",
        )

    def test_planned_industrial_template_issue_requires_implementation_first(self) -> None:
        state_path = self.write_state(
            {
                "tool_name": "extended_device_sweep",
                "status": "completed",
                "run_id": "gan_surrogate",
                "request": {"device_type": "gan_hemt_id_bv"},
                "quality_report": {
                    "status": "failed",
                    "issues": [{"code": "planned_industrial_template_runner_missing", "severity": "error"}],
                },
            }
        )

        plan = build_repair_plan(state_path)
        actions = {action.name: action for action in plan.actions}

        self.assertEqual(plan.next_action, "implement_planned_industrial_runner_first")
        self.assertIn("implement_planned_industrial_runner_first", actions)
        self.assertTrue(actions["implement_planned_industrial_runner_first"].user_confirmation_required)

    def test_passed_state_has_no_action(self) -> None:
        state_path = self.write_state(
            {
                "tool_name": "pn_junction_iv_sweep",
                "status": "completed",
                "run_id": "pn_ok",
                "request": {},
                "quality_report": {"status": "passed", "issues": []},
            }
        )

        plan = build_repair_plan(state_path)

        self.assertEqual(plan.status, RepairPlanStatus.NO_ACTION)
        self.assertEqual(plan.actions, [])


if __name__ == "__main__":
    unittest.main()
