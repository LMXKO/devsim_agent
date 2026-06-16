from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.llm import LLMConfig
from tcad_agent.mutation_schema_agent import MutationSchemaExtensionRequest, run_mutation_schema_extension


class FakeSchemaClient:
    config = LLMConfig(model="fake-schema-agent")

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        return json.dumps(self.payload)


class MutationSchemaAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_project(self, text: str) -> Path:
        project = self.root / "project"
        project.mkdir()
        (project / "device.cmd").write_text(text.lstrip(), encoding="utf-8")
        return project

    def test_builds_review_package_for_unknown_deck_binding(self) -> None:
        project = self.write_project(
            """
set SURFACE_RECOMB_VELOCITY 1e5

Physics {
  Mobility( DopingDep )
}
"""
        )

        result = run_mutation_schema_extension(
            MutationSchemaExtensionRequest(
                goal_text="Reduce leakage by tuning surface recombination velocity without hurting BV.",
                project_path=project,
                deck_files=["device.cmd"],
                proposed_target="surface recombination velocity",
                output_dir=self.root / "schema",
            )
        )

        self.assertEqual(result.status, "completed")
        self.assertIsNotNone(result.selected_candidate)
        candidate = result.selected_candidate
        self.assertEqual(candidate.class_id, "surface_recombination_velocity")
        self.assertTrue(candidate.ready_for_review)
        self.assertEqual(candidate.validation_patch["operation"], "sentaurus_set_variable")
        self.assertEqual(candidate.validation_patch["variable"], "SURFACE_RECOMB_VELOCITY")
        self.assertEqual(candidate.verified_patch_count, 1)
        self.assertTrue(Path(candidate.fixture_deck_path).exists())
        self.assertTrue(Path(result.output_path).exists())
        self.assertTrue(result.public_evidence_dossier["evidence_gate"]["passed"])
        self.assertTrue(result.final_summary["does_not_modify_static_vocabulary"])
        self.assertTrue(result.final_summary["does_not_execute_solver"])

    def test_llm_can_propose_schema_inside_validation_guardrails(self) -> None:
        project = self.write_project("set GATE_WORKFUNCTION_SHIFT 0.05\n")
        client = FakeSchemaClient(
            {
                "schema": {
                    "class_id": "gate_workfunction_shift",
                    "display_name": "Gate workfunction shift",
                    "target_kind": "process_parameter",
                    "default_risk_level": "high",
                    "requires_user_confirmation": True,
                    "variable_name_tokens": [["GATE", "WORKFUNCTION"]],
                    "semantic_patch_operations": ["sentaurus_set_variable"],
                    "rationale": "Local deck exposes a numeric gate workfunction variable; keep as review-only.",
                    "confidence": 0.77,
                }
            }
        )

        result = run_mutation_schema_extension(
            MutationSchemaExtensionRequest(
                goal_text="Tune gate workfunction shift to improve threshold and leakage tradeoff.",
                project_path=project,
                deck_files=["device.cmd"],
                proposed_target="gate workfunction shift",
                output_dir=self.root / "schema_llm",
                use_llm=True,
                allow_llm_fallback=False,
            ),
            llm_client=client,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.model_decision["status"], "completed")
        self.assertFalse(result.model_decision["fallback_used"])
        self.assertEqual(result.selected_candidate.class_id, "gate_workfunction_shift")
        self.assertEqual(result.selected_candidate.default_risk_level, "high")
        self.assertTrue(result.selected_candidate.requires_user_confirmation)
        self.assertIn("Local deck exposes", result.selected_candidate.llm_rationale)

    def test_blocks_when_no_unknown_deck_binding_matches_target(self) -> None:
        project = self.write_project("set LIFETIME_SCALE 1.0\n")

        result = run_mutation_schema_extension(
            MutationSchemaExtensionRequest(
                goal_text="Reduce reverse leakage by adding a new surface recombination schema.",
                project_path=project,
                deck_files=["device.cmd"],
                proposed_target="surface recombination velocity",
                output_dir=self.root / "schema_blocked",
            )
        )

        self.assertEqual(result.status, "blocked_no_deck_binding")
        self.assertIsNone(result.selected_candidate)
        self.assertEqual(result.final_summary["candidate_count"], 0)
        self.assertTrue(Path(result.output_path).exists())


if __name__ == "__main__":
    unittest.main()
