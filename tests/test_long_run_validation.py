from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tcad_agent.long_run_validation import (
    DEFAULT_AUTONOMOUS_E2E_SCENARIOS,
    LongRunValidationRequest,
    LongRunValidationStatus,
    LongRunValidationSuite,
    SCENARIO_REGISTRY,
    run_long_run_validation,
)


class LongRunValidationTest(unittest.TestCase):
    def test_live_llm_user_deck_acceptance_is_explicit_not_default(self) -> None:
        self.assertIn("public_user_deck_live_llm_acceptance", SCENARIO_REGISTRY)
        self.assertIn("public_user_deck_live_llm_soak", SCENARIO_REGISTRY)
        self.assertIn("public_curve_decision_live_llm_eval", SCENARIO_REGISTRY)
        self.assertIn("public_curve_decision_live_llm_agent_loop", SCENARIO_REGISTRY)
        self.assertIn("public_curve_decision_live_llm_devsim_soak", SCENARIO_REGISTRY)
        self.assertIn("public_sentaurus_live_llm_contract_soak", SCENARIO_REGISTRY)
        self.assertNotIn("public_user_deck_live_llm_acceptance", DEFAULT_AUTONOMOUS_E2E_SCENARIOS)
        self.assertNotIn("public_user_deck_live_llm_soak", DEFAULT_AUTONOMOUS_E2E_SCENARIOS)
        self.assertNotIn("public_curve_decision_live_llm_eval", DEFAULT_AUTONOMOUS_E2E_SCENARIOS)
        self.assertNotIn("public_curve_decision_live_llm_agent_loop", DEFAULT_AUTONOMOUS_E2E_SCENARIOS)
        self.assertNotIn("public_curve_decision_live_llm_devsim_soak", DEFAULT_AUTONOMOUS_E2E_SCENARIOS)
        self.assertNotIn("public_sentaurus_live_llm_contract_soak", DEFAULT_AUTONOMOUS_E2E_SCENARIOS)

    def test_runs_queue_daemon_benchmarks_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = run_long_run_validation(
                LongRunValidationRequest(validation_id="unit_longrun", validation_root=Path(tmp))
            )

        self.assertEqual(state.status, LongRunValidationStatus.COMPLETED)
        self.assertEqual(state.daemon_result["completed"], 5)
        self.assertEqual(len(state.benchmark_results), 5)
        self.assertGreaterEqual(state.index_summary["records_indexed"], 2)
        self.assertTrue(all(item["status"] == "completed" for item in state.queued_items))
        self.assertIn("longrun_power_mosfet_convergence", {item["queue_id"] for item in state.queued_items})
        self.assertIn("longrun_bjt_convergence", {item["queue_id"] for item in state.queued_items})

    def test_runs_autonomous_e2e_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = run_long_run_validation(
                LongRunValidationRequest(
                    validation_id="unit_autonomous_e2e",
                    validation_root=Path(tmp),
                    suite=LongRunValidationSuite.AUTONOMOUS_E2E,
                    agent_max_steps=8,
                )
            )

        self.assertEqual(state.status, LongRunValidationStatus.COMPLETED)
        scenario_by_id = {item["scenario_id"]: item for item in state.scenario_results}
        self.assertEqual(
            set(scenario_by_id),
            {
                "agent_confirmation_pause",
                "agent_cancel_boundary",
                "agent_repair_report",
                "mutation_refinement_multiround",
                "sentaurus_autonomous_refinement",
                "natural_language_power_marathon",
                "public_user_deck_acceptance",
                "public_user_deck_corpus_acceptance",
                "public_curve_decision_eval",
                "queue_confirmation_resume",
                "queue_interruption_recovery",
            },
        )
        self.assertTrue(all(item["status"] == "completed" for item in scenario_by_id.values()))
        self.assertEqual(
            scenario_by_id["mutation_refinement_multiround"]["details"]["refinement_values"],
            [2.25, 2.375],
        )
        self.assertEqual(
            scenario_by_id["sentaurus_autonomous_refinement"]["details"]["sentaurus_run_count"],
            3,
        )
        self.assertEqual(
            scenario_by_id["sentaurus_autonomous_refinement"]["details"]["lineage_entries"],
            3,
        )
        self.assertTrue(
            any(artifact["name"] == "report" and artifact["exists"] for artifact in scenario_by_id["agent_repair_report"]["artifacts"])
        )
        marathon = scenario_by_id["natural_language_power_marathon"]["details"]
        self.assertEqual(marathon["route_template"], "power_mosfet_bv_ron")
        self.assertEqual(marathon["initial_fidelity"], "devsim_2d_field_plate")
        self.assertEqual(marathon["selected_experiment_candidate"], "power_mosfet_2d_signoff_evidence_pack")
        self.assertEqual(marathon["signoff_verdict"], "conditional")
        self.assertEqual(marathon["resume_status"], "completed")
        self.assertEqual(marathon["cancel_status"], "cancelled")
        self.assertTrue(
            any(
                artifact["name"] == "agent_cockpit" and artifact["exists"]
                for artifact in scenario_by_id["natural_language_power_marathon"]["artifacts"]
            )
        )
        public_deck = scenario_by_id["public_user_deck_acceptance"]["details"]
        self.assertTrue(public_deck["deck_patch_verified"])
        self.assertEqual(public_deck["updated_n_doping_cm3"], 8e17)
        self.assertEqual(public_deck["quality_status"], "passed")
        corpus = scenario_by_id["public_user_deck_corpus_acceptance"]["details"]
        self.assertEqual(corpus["case_count"], 3)
        self.assertTrue(all(item["deck_patch_verified"] for item in corpus["cases"]))
        self.assertEqual(
            {item["shape"] for item in corpus["cases"]},
            {"function_wrapped_config", "package_imports_with_local_overrides", "multi_sweep_bias_sequence"},
        )
        curve_eval = scenario_by_id["public_curve_decision_eval"]["details"]
        self.assertEqual(curve_eval["case_count"], 4)
        self.assertEqual(curve_eval["failed_count"], 0)
        self.assertEqual(curve_eval["fallback_count"], 0)
        self.assertEqual(
            {item["recommended_action"] for item in curve_eval["cases"]},
            {
                "refine_effective_mutation",
                "pareto_review_before_next_patch",
                "switch_mutation_target",
                "repair_curve_shape",
            },
        )
        self.assertEqual(
            scenario_by_id["queue_confirmation_resume"]["details"]["completed_status"],
            "completed",
        )
        self.assertGreaterEqual(state.index_summary["records_indexed"], 2)


if __name__ == "__main__":
    unittest.main()
