from __future__ import annotations

import unittest

from tcad_agent.mutation_vocabulary import (
    classify_mutation_variable,
    list_mutation_vocabulary,
    mutation_class_ids,
    mutation_entry,
)


class MutationVocabularyTest(unittest.TestCase):
    def test_schema_covers_open_engineering_mutation_terms(self) -> None:
        expected = {
            "guard_ring",
            "junction_depth",
            "oxide_thickness",
            "implant_dose",
            "trench_corner_radius",
            "trap_density",
            "region_specific_lifetime",
            "field_plate",
            "drift_doping",
            "lifetime",
        }

        self.assertTrue(expected.issubset(set(mutation_class_ids())))
        for class_id in expected:
            entry = mutation_entry(class_id)
            self.assertIsNotNone(entry)
            self.assertTrue(entry.primary_metrics)
            self.assertTrue(entry.expected_curve_evidence)
            self.assertTrue(entry.stop_conditions)
            self.assertTrue(entry.public_source_ids)

    def test_variable_classifier_uses_schema_tokens(self) -> None:
        cases = {
            "GUARD_RING_SPACING": "guard_ring",
            "JUNCTION_DEPTH": "junction_depth",
            "TOX_GATE": "oxide_thickness",
            "PPLUS_IMPLANT_DOSE": "implant_dose",
            "TRENCH_CORNER_RADIUS": "trench_corner_radius",
            "TRAP_DENSITY": "trap_density",
            "N_DRIFT_TAU": "region_specific_lifetime",
            "NDRIFT_DOPING": "drift_doping",
        }

        for variable, expected in cases.items():
            with self.subTest(variable=variable):
                self.assertIn(expected, classify_mutation_variable(variable))

    def test_list_mutation_vocabulary_is_json_ready(self) -> None:
        vocabulary = list_mutation_vocabulary()

        self.assertTrue(all("class_id" in item for item in vocabulary))
        self.assertTrue(all("semantic_patch_operations" in item for item in vocabulary))


if __name__ == "__main__":
    unittest.main()
