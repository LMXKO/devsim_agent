from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.agent_curve_decision import (
    CurveDecisionNextAgentAction,
    CurveDecisionPlannerRequest,
    CurveDecisionPlannerStatus,
    build_curve_decision_plan,
)
from tcad_agent.llm import LLMConfig


class FakeCurveDecisionPlannerClient:
    config = LLMConfig(base_url="http://unit.test/v1", model="unit-curve-decision-model", api_key="")

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        del system, user, temperature
        return json.dumps(
            {
                "recommended_action": "switch_mutation_target",
                "recommended_target": "implant_dose",
                "recommended_direction": "probe_alternate",
                "rationale": "The tested drift doping mutation worsened Ron, so another process target should be probed.",
                "evidence_used": ["mutation_effect_analysis", "metric_deltas", "curve_shape"],
            }
        )


class InvalidCurveDecisionPlannerClient:
    config = LLMConfig(base_url="http://unit.test/v1", model="unit-invalid-curve-decision-model", api_key="")

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        del system, user, temperature
        return "{}"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_state(root: Path) -> Path:
    csv_path = root / "curve.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "reverse_voltage_v,current_a,electric_field_v_per_cm\n0,1e-12,1e4\n-20,2e-10,8e4\n-40,1e-8,1.8e5\n",
        encoding="utf-8",
    )
    state_path = root / "state.json"
    write_json(
        state_path,
        {
            "tool_name": "extended_device_sweep",
            "status": "completed",
            "run_id": "curve_decision_source",
            "request": {
                "device_type": "power_mosfet_bv_ron",
                "fidelity": "devsim_2d_field_plate",
                "power_mos_drift_region_doping_cm3": 1.0e16,
            },
            "final_summary": {
                "artifacts": {"csv": str(csv_path)},
                "metrics": {
                    "leakage_current_a": 1.05e-8,
                    "breakdown_voltage_v": 79.0,
                    "specific_on_resistance_ohm_cm2": 5.8e-3,
                    "max_electric_field_v_per_cm": 2.9e5,
                },
            },
            "quality_report": {"status": "passed", "metrics": {"specific_on_resistance_ohm_cm2": 5.8e-3}},
            "mutation_effect_analysis": {
                "mutation_target": "drift_doping",
                "primary_metric": "specific_on_resistance_ohm_cm2",
                "primary_improved": False,
                "worth_continuing": False,
                "decision": "switch_target",
                "rationale": "Ron worsened after drift doping change.",
                "recommended_next_target": "implant_dose",
                "recommended_next_direction": "probe_alternate",
                "improved_metrics": [],
                "regressed_metrics": ["specific_on_resistance_ohm_cm2"],
                "tradeoff_violations": [],
            },
        },
    )
    return state_path


class AgentCurveDecisionTest(unittest.TestCase):
    def test_llm_curve_decision_plan_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_state(root / "source")
            output = root / "curve_decision.json"
            plan = build_curve_decision_plan(
                CurveDecisionPlannerRequest(
                    source_state_path=source,
                    goal_text="Optimize Ron/BV tradeoff from curve evidence.",
                    output_path=output,
                    use_llm=True,
                    allow_llm_fallback=False,
                ),
                llm_client=FakeCurveDecisionPlannerClient(),
            )

            self.assertEqual(plan.status, CurveDecisionPlannerStatus.COMPLETED)
            self.assertEqual(plan.decision_source, "llm")
            self.assertFalse(plan.fallback_used)
            self.assertEqual(plan.model, "unit-curve-decision-model")
            self.assertEqual(plan.recommended_action, "switch_mutation_target")
            self.assertEqual(plan.recommended_target, "implant_dose")
            self.assertEqual(plan.next_agent_action, CurveDecisionNextAgentAction.PLAN_GUIDANCE_PATCH)
            self.assertEqual(plan.curve_guidance["recommended_target"], "implant_dose")
            self.assertTrue(output.exists())

    def test_invalid_llm_without_fallback_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = write_state(Path(tmp) / "source")
            plan = build_curve_decision_plan(
                CurveDecisionPlannerRequest(
                    source_state_path=source,
                    goal_text="Optimize Ron/BV tradeoff from curve evidence.",
                    use_llm=True,
                    allow_llm_fallback=False,
                ),
                llm_client=InvalidCurveDecisionPlannerClient(),
            )

            self.assertEqual(plan.status, CurveDecisionPlannerStatus.FAILED)
            self.assertEqual(plan.decision_source, "llm")
            self.assertIn("allowed action", plan.failure_reason or "")


if __name__ == "__main__":
    unittest.main()
