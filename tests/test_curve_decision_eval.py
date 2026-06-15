from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.curve_decision_eval import (
    CurveDecisionEvalRequest,
    CurveDecisionEvalStatus,
    default_curve_decision_cases,
    run_curve_decision_eval,
)
from tcad_agent.llm import LLMConfig


class FakeCurveDecisionClient:
    config = LLMConfig(base_url="http://unit.test/v1", model="unit-curve-model", api_key="")

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        del system, temperature
        payload = json.loads(user)
        case_id = payload["case"]["case_id"]
        responses = {
            "lifetime_leakage_improved": {
                "recommended_action": "refine_effective_mutation",
                "recommended_target": "region_specific_lifetime",
                "rationale": "Leakage fell while BV, Ron, and field stayed within tradeoff limits.",
                "evidence_used": ["metric_deltas", "tradeoff_violations", "curve_shape"],
            },
            "field_plate_ron_tradeoff": {
                "recommended_action": "pareto_review_before_next_patch",
                "recommended_target": "guard_ring",
                "rationale": "Field peak improved, but Ron regressed beyond tolerance, so constraints need review.",
                "evidence_used": ["metric_deltas", "tradeoff_violations", "overlay"],
            },
            "drift_doping_ron_not_improved": {
                "recommended_action": "switch_mutation_target",
                "recommended_target": "implant_dose",
                "rationale": "The drift doping probe worsened the primary Ron metric.",
                "evidence_used": ["metric_deltas", "primary_metric", "curve_shape"],
            },
            "nonmonotonic_curve_requires_repair": {
                "recommended_action": "repair_curve_shape",
                "recommended_target": "bias_or_mesh_refinement",
                "rationale": "The mutation curve has a monotonicity break, so numerical evidence comes first.",
                "evidence_used": ["curve_shape", "overlay", "monotonicity"],
            },
        }
        return json.dumps(responses[case_id])


class InvalidCurveDecisionClient:
    config = LLMConfig(base_url="http://unit.test/v1", model="unit-invalid-model", api_key="")

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        del system, user, temperature
        return "{}"


class CurveDecisionEvalTest(unittest.TestCase):
    def test_deterministic_eval_covers_expected_curve_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_curve_decision_eval(
                CurveDecisionEvalRequest(eval_id="unit_curve_eval", eval_root=Path(tmp))
            )

            self.assertEqual(result.status, CurveDecisionEvalStatus.COMPLETED)
            self.assertEqual(result.case_count, 4)
            self.assertEqual(result.passed_count, 4)
            by_case = {item.case_id: item for item in result.cases}
            self.assertEqual(by_case["lifetime_leakage_improved"].recommended_action, "refine_effective_mutation")
            self.assertEqual(by_case["field_plate_ron_tradeoff"].recommended_action, "pareto_review_before_next_patch")
            self.assertEqual(by_case["drift_doping_ron_not_improved"].recommended_action, "switch_mutation_target")
            self.assertEqual(by_case["nonmonotonic_curve_requires_repair"].recommended_action, "repair_curve_shape")
            self.assertTrue(all(item.overlay_svg_path and Path(item.overlay_svg_path).exists() for item in result.cases))
            self.assertTrue(result.result_path and Path(result.result_path).exists())

    def test_llm_eval_records_model_decisions_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_curve_decision_eval(
                CurveDecisionEvalRequest(
                    eval_id="unit_curve_llm_eval",
                    eval_root=Path(tmp),
                    use_llm=True,
                    allow_llm_fallback=False,
                    cases=default_curve_decision_cases(),
                ),
                llm_client=FakeCurveDecisionClient(),
            )

        self.assertEqual(result.status, CurveDecisionEvalStatus.COMPLETED)
        self.assertEqual(result.llm_decision_count, result.case_count)
        self.assertEqual(result.fallback_count, 0)
        self.assertEqual(result.raw_response_count, result.case_count)
        self.assertEqual(result.models, ["unit-curve-model"])

    def test_invalid_llm_without_fallback_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_curve_decision_eval(
                CurveDecisionEvalRequest(
                    eval_id="unit_curve_invalid_eval",
                    eval_root=Path(tmp),
                    use_llm=True,
                    allow_llm_fallback=False,
                ),
                llm_client=InvalidCurveDecisionClient(),
            )

        self.assertEqual(result.status, CurveDecisionEvalStatus.FAILED)
        self.assertEqual(result.passed_count, 0)
        self.assertTrue(all(item.decision_source == "llm" for item in result.cases))
        self.assertTrue(all(item.failure_reason for item in result.cases))


if __name__ == "__main__":
    unittest.main()
