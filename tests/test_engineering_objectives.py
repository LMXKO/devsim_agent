from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.engineering_objectives import (
    ConstraintOperator,
    EngineeringConstraint,
    EngineeringObjective,
    ObjectiveDirection,
    evaluate_engineering_objectives,
)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class EngineeringObjectivesTest(unittest.TestCase):
    def test_filters_constraints_and_selects_best_feasible_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "optimization_state.json"
            write_json(
                state_path,
                {
                    "tool_name": "multidim_optimizer",
                    "status": "completed",
                    "observations": [
                        {
                            "task_id": "fast_but_leaky",
                            "status": "completed",
                            "metrics": {"ion_ioff_ratio": 1e8, "ioff_current_a": 1e-8},
                        },
                        {
                            "task_id": "balanced",
                            "status": "completed",
                            "metrics": {"ion_ioff_ratio": 1e6, "ioff_current_a": 1e-12},
                        },
                    ],
                },
            )

            result = evaluate_engineering_objectives(
                state_path,
                objectives=[
                    EngineeringObjective(metric_path="ion_ioff_ratio", direction=ObjectiveDirection.MAXIMIZE)
                ],
                constraints=[
                    EngineeringConstraint(
                        metric_path="ioff_current_a",
                        operator=ConstraintOperator.LE,
                        value=1e-10,
                    )
                ],
            )
            output_exists = Path(result.output_path).exists()

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.best_candidate.candidate_id, "balanced")
        self.assertFalse(next(item for item in result.candidates if item.candidate_id == "fast_but_leaky").feasible)
        self.assertEqual(result.decision["action"], "review_constraints")
        self.assertEqual(result.decision["best_candidate_id"], "balanced")
        self.assertIn("balanced", result.decision["feasible_candidate_ids"])
        self.assertTrue(output_exists)

    def test_pareto_front_keeps_tradeoff_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "optimization_state.json"
            write_json(
                state_path,
                {
                    "tool_name": "multidim_optimizer",
                    "status": "completed",
                    "observations": [
                        {"task_id": "high_bv", "metrics": {"breakdown_voltage_v": -100.0, "specific_on_resistance_ohm_cm2": 0.2}},
                        {"task_id": "low_ron", "metrics": {"breakdown_voltage_v": -50.0, "specific_on_resistance_ohm_cm2": 0.03}},
                        {"task_id": "dominated", "metrics": {"breakdown_voltage_v": -40.0, "specific_on_resistance_ohm_cm2": 0.2}},
                    ],
                },
            )

            result = evaluate_engineering_objectives(
                state_path,
                objectives=[
                    EngineeringObjective(metric_path="breakdown_voltage_v", direction=ObjectiveDirection.MAXIMIZE_ABS),
                    EngineeringObjective(metric_path="specific_on_resistance_ohm_cm2", direction=ObjectiveDirection.MINIMIZE),
                ],
            )

        front = {item.candidate_id for item in result.pareto_front}

        self.assertEqual(front, {"high_bv", "low_ron"})
        self.assertEqual(result.decision["action"], "continue_with_best_candidate")
        self.assertTrue(result.decision["best_on_pareto_front"])


if __name__ == "__main__":
    unittest.main()
