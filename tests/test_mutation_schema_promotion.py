from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from tcad_agent.mutation_schema_agent import MutationSchemaExtensionRequest, run_mutation_schema_extension
from tcad_agent.mutation_schema_promotion import MutationSchemaPromotionRequest, run_mutation_schema_promotion


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MutationSchemaPromotionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_schema_extension(self) -> Path:
        project = self.root / "project"
        project.mkdir()
        (project / "device.cmd").write_text(
            """
set SURFACE_RECOMB_VELOCITY 1e5

Physics {
  Mobility( DopingDep )
}
""".lstrip(),
            encoding="utf-8",
        )
        extension = run_mutation_schema_extension(
            MutationSchemaExtensionRequest(
                goal_text="Reduce reverse leakage by tuning surface recombination velocity without hurting BV.",
                project_path=project,
                deck_files=["device.cmd"],
                proposed_target="surface recombination velocity",
                output_dir=self.root / "schema_extension",
            )
        )
        self.assertEqual(extension.status, "completed")
        return Path(extension.output_path)

    def test_promotion_gate_generates_patch_and_test_without_applying(self) -> None:
        extension_path = self.write_schema_extension()

        result = run_mutation_schema_promotion(
            MutationSchemaPromotionRequest(
                schema_extension_path=extension_path,
                output_dir=self.root / "promotion",
            )
        )

        self.assertEqual(result.status, "ready_for_confirmation")
        self.assertFalse(result.applied)
        self.assertEqual(result.selected_class_id, "surface_recombination_velocity")
        codes = {check.code for check in result.checks if check.status == "passed"}
        self.assertIn("public_evidence_gate_passed", codes)
        self.assertIn("deck_patch_validation_verified", codes)
        self.assertIn("fixture_validation_verified", codes)
        self.assertIn("schema_patch_validates", codes)
        self.assertIn('+        class_id="surface_recombination_velocity",', result.mutation_vocabulary_patch)
        self.assertIn("test_surface_recombination_velocity_entry_is_classifiable", result.generated_test_source)
        self.assertTrue(Path(result.artifacts["mutation_vocabulary_patch"]).exists())
        self.assertTrue(Path(result.artifacts["generated_test_source"]).exists())
        self.assertTrue(Path(result.output_path).exists())

    def test_apply_requires_explicit_confirmation(self) -> None:
        extension_path = self.write_schema_extension()

        result = run_mutation_schema_promotion(
            MutationSchemaPromotionRequest(
                schema_extension_path=extension_path,
                output_dir=self.root / "promotion_blocked",
                apply=True,
                confirmed=False,
            )
        )

        self.assertEqual(result.status, "blocked_needs_confirmation")
        self.assertEqual(result.failure_reason, "apply_requires_confirmed_true")
        self.assertFalse(result.applied)

    def test_confirmed_apply_updates_only_requested_vocabulary_file(self) -> None:
        extension_path = self.write_schema_extension()
        vocabulary_copy = self.root / "mutation_vocabulary_copy.py"
        shutil.copy2(PROJECT_ROOT / "tcad_agent" / "mutation_vocabulary.py", vocabulary_copy)

        result = run_mutation_schema_promotion(
            MutationSchemaPromotionRequest(
                schema_extension_path=extension_path,
                vocabulary_path=vocabulary_copy,
                output_dir=self.root / "promotion_apply",
                apply=True,
                confirmed=True,
            )
        )

        self.assertEqual(result.status, "applied")
        self.assertTrue(result.applied)
        updated = vocabulary_copy.read_text(encoding="utf-8")
        self.assertIn('class_id="surface_recombination_velocity"', updated)
        original = (PROJECT_ROOT / "tcad_agent" / "mutation_vocabulary.py").read_text(encoding="utf-8")
        self.assertNotIn('class_id="surface_recombination_velocity"', original)


if __name__ == "__main__":
    unittest.main()
