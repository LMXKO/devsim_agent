from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.sentaurus_patch_refiner import SentaurusPatchRefinerRequest, build_sentaurus_patch_refinement_plan
from tcad_agent.llm import LLMConfig


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class FakeRefinerClient:
    config = LLMConfig(model="fake-refiner")

    def __init__(self, selected_candidate_id: str) -> None:
        self.selected_candidate_id = selected_candidate_id
        self.calls: list[dict[str, str]] = []

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        self.calls.append({"system": system, "user": user})
        return json.dumps(
            {
                "selected_candidate_id": self.selected_candidate_id,
                "rationale": "Prefer the candidate that tests trap-assisted leakage without repeating the failed drift patch.",
                "expected_observation": "Leakage should move if traps dominate.",
                "stop_condition": "Stop if leakage and slope do not change.",
                "rejected_candidate_ids": [],
            }
        )


class SentaurusPatchRefinerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_state_with_effect(self, project: Path, analysis: dict) -> Path:
        state_path = self.root / "mutation" / "sentaurus_state.json"
        write_json(
            state_path,
            {
                "tool_name": "sentaurus_run",
                "status": "completed",
                "run_id": "mutation",
                "project_copy_path": str(project),
                "request": {"goal_text": "Reduce leakage without hurting BV/Ron.", "deck_files": ["device.cmd"]},
                "quality_report": {"status": "passed", "metrics": {"leakage_abs_current_at_target_a": 4e-10}},
                "final_summary": {
                    "artifacts": {"project_copy": str(project)},
                    "metrics": {"solver_backend": "sentaurus"},
                    "parameters": {"deck_files": ["device.cmd"]},
                },
                "sentaurus_mutation_effect_analysis": analysis,
            },
        )
        return state_path

    def test_continue_refine_generates_half_step_beyond_verified_patch(self) -> None:
        project = self.root / "project"
        project.mkdir()
        (project / "device.cmd").write_text("set LIFETIME_SCALE 2.0\n", encoding="utf-8")
        analysis = {
            "decision": "continue_refine",
            "candidate_id": "device.cmd:lifetime:LIFETIME_SCALE",
            "primary_metric": "leakage_abs_current_at_target_a",
            "rationale": "Leakage improved without tradeoff.",
            "candidate": {
                "candidate_id": "device.cmd:lifetime:LIFETIME_SCALE",
                "title": "LIFETIME_SCALE 2x lifetime probe",
                "risk_level": "low",
                "patches": [
                    {
                        "file": "device.cmd",
                        "operation": "sentaurus_set_variable",
                        "variable": "LIFETIME_SCALE",
                        "value": "2",
                    }
                ],
                "validation_records": [
                    {
                        "file": "device.cmd",
                        "operation": "sentaurus_set_variable",
                        "variable": "LIFETIME_SCALE",
                        "verified": True,
                        "old_value": "1.0",
                        "value": "2",
                    }
                ],
            },
        }
        state_path = self.write_state_with_effect(project, analysis)

        plan = build_sentaurus_patch_refinement_plan(
            SentaurusPatchRefinerRequest(
                source_state_path=state_path,
                output_path=self.root / "refinement.json",
            )
        )

        self.assertEqual(plan.status, "completed")
        self.assertTrue(Path(plan.output_path).exists())
        self.assertIsNotNone(plan.selected_candidate)
        patch = plan.selected_candidate.patches[0]
        self.assertEqual(patch["operation"], "sentaurus_set_variable")
        self.assertEqual(patch["variable"], "LIFETIME_SCALE")
        self.assertEqual(patch["value"], "2.5")
        self.assertEqual(plan.selected_candidate.verified_patch_count, 1)

    def test_blocks_pareto_review_without_new_patch(self) -> None:
        project = self.root / "project"
        project.mkdir()
        (project / "device.cmd").write_text("set LIFETIME_SCALE 2.0\n", encoding="utf-8")
        state_path = self.write_state_with_effect(
            project,
            {
                "decision": "blocked_for_pareto_review",
                "candidate_id": "device.cmd:lifetime:LIFETIME_SCALE",
                "tradeoff_violations": [{"metric": "max_electric_field_v_per_cm"}],
            },
        )

        plan = build_sentaurus_patch_refinement_plan(SentaurusPatchRefinerRequest(source_state_path=state_path))

        self.assertEqual(plan.status, "blocked_for_pareto_review")
        self.assertIsNone(plan.selected_candidate)
        self.assertEqual(plan.candidates, [])

    def test_switch_target_uses_different_verified_candidate(self) -> None:
        project = self.root / "project"
        project.mkdir()
        (project / "device.cmd").write_text(
            """
set LIFETIME_SCALE 2.0
set DRIFT_DOPING 1e15
""".lstrip(),
            encoding="utf-8",
        )
        state_path = self.write_state_with_effect(
            project,
            {
                "decision": "switch_target",
                "candidate_id": "device.cmd:lifetime:LIFETIME_SCALE",
                "primary_metric": "leakage_abs_current_at_target_a",
                "recommended_next_target": "drift_doping",
                "rationale": "Lifetime did not improve leakage.",
                "candidate": {
                    "candidate_id": "device.cmd:lifetime:LIFETIME_SCALE",
                    "patches": [
                        {
                            "file": "device.cmd",
                            "operation": "sentaurus_set_variable",
                            "variable": "LIFETIME_SCALE",
                            "value": "2",
                        }
                    ],
                },
            },
        )

        plan = build_sentaurus_patch_refinement_plan(
            SentaurusPatchRefinerRequest(
                source_state_path=state_path,
                goal_text="Improve BV/field by changing drift doping, not lifetime.",
            )
        )

        self.assertEqual(plan.status, "completed")
        self.assertIsNotNone(plan.selected_candidate)
        selected_text = json.dumps(plan.selected_candidate.model_dump(mode="json"))
        self.assertIn("DRIFT_DOPING", selected_text)
        self.assertNotIn('"variable": "LIFETIME_SCALE"', selected_text)

    def test_llm_policy_can_choose_among_verified_switch_candidates(self) -> None:
        project = self.root / "project"
        project.mkdir()
        (project / "device.cmd").write_text(
            """
set LIFETIME_SCALE 1.0
set TRAP_DENSITY 1e12
set DRIFT_DOPING 8e14
""".lstrip(),
            encoding="utf-8",
        )
        state_path = self.write_state_with_effect(
            project,
            {
                "decision": "switch_target",
                "candidate_id": "device.cmd:drift_doping:DRIFT_DOPING",
                "primary_metric": "leakage_abs_current_at_target_a",
                "rationale": "Drift doping did not reduce leakage.",
                "candidate": {
                    "candidate_id": "device.cmd:drift_doping:DRIFT_DOPING",
                    "patches": [
                        {
                            "file": "device.cmd",
                            "operation": "sentaurus_set_variable",
                            "variable": "DRIFT_DOPING",
                            "value": "8e14",
                        }
                    ],
                },
            },
        )
        selected_id = "device.cmd:trap_density:TRAP_DENSITY:switch_from:device.cmd:drift_doping:DRIFT_DOPING"
        client = FakeRefinerClient(selected_id)

        plan = build_sentaurus_patch_refinement_plan(
            SentaurusPatchRefinerRequest(
                source_state_path=state_path,
                goal_text="Reduce leakage by testing lifetime or trap density; do not repeat drift doping.",
                use_llm=True,
            ),
            llm_client=client,
        )

        self.assertEqual(plan.status, "completed")
        self.assertEqual(plan.selected_candidate.candidate_id, selected_id)
        self.assertEqual(plan.agent_policy["status"], "completed")
        self.assertEqual(len(client.calls), 1)
        self.assertIn("eligible_candidates", client.calls[0]["user"])

    def test_strict_llm_policy_rejects_unknown_candidate(self) -> None:
        project = self.root / "project"
        project.mkdir()
        (project / "device.cmd").write_text("set LIFETIME_SCALE 2.0\n", encoding="utf-8")
        state_path = self.write_state_with_effect(
            project,
            {
                "decision": "continue_refine",
                "candidate_id": "device.cmd:lifetime:LIFETIME_SCALE",
                "primary_metric": "leakage_abs_current_at_target_a",
                "candidate": {
                    "candidate_id": "device.cmd:lifetime:LIFETIME_SCALE",
                    "risk_level": "low",
                    "patches": [{"file": "device.cmd", "operation": "sentaurus_set_variable", "variable": "LIFETIME_SCALE", "value": "2"}],
                    "validation_records": [
                        {
                            "file": "device.cmd",
                            "operation": "sentaurus_set_variable",
                            "variable": "LIFETIME_SCALE",
                            "verified": True,
                            "old_value": "1.0",
                            "value": "2",
                        }
                    ],
                },
            },
        )

        plan = build_sentaurus_patch_refinement_plan(
            SentaurusPatchRefinerRequest(
                source_state_path=state_path,
                use_llm=True,
                allow_llm_fallback=False,
            ),
            llm_client=FakeRefinerClient("invented_candidate"),
        )

        self.assertEqual(plan.status, "failed")
        self.assertIsNone(plan.selected_candidate)
        self.assertEqual(plan.agent_policy["status"], "invalid_selection")


if __name__ == "__main__":
    unittest.main()
