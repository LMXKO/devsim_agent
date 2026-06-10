from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.sentaurus_mutation_effect import SentaurusMutationEffectRequest, analyze_sentaurus_mutation_effect


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_curve(path: Path, rows: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["voltage_v,current_a,electric_field_v_per_cm"]
    lines.extend(f"{voltage},{current},{field}" for voltage, current, field in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sentaurus_state(
    path: Path,
    *,
    leakage: float,
    breakdown: float,
    field: float,
    ron: float,
    quality: str = "passed",
) -> Path:
    curve = path.parent / "sentaurus_extract.csv"
    write_curve(
        curve,
        [
            (0.0, leakage, field * 0.1),
            (-50.0, leakage * 5, field * 0.5),
            (breakdown, 1e-6, field),
        ],
    )
    write_json(
        path,
        {
            "tool_name": "sentaurus_run",
            "status": "completed" if quality == "passed" else "failed",
            "run_id": path.parent.name,
            "quality_report": {
                "status": quality,
                "metrics": {
                    "solver_backend": "sentaurus",
                    "tcad_solver_invoked": True,
                    "curve_points": 3,
                    "curve_x_key": "voltage_v",
                    "curve_y_key": "current_a",
                    "curve_field_key": "electric_field_v_per_cm",
                    "breakdown_current_threshold_a": 1e-6,
                    "leakage_abs_current_at_target_a": leakage,
                    "breakdown_voltage_at_threshold_v": breakdown,
                    "max_electric_field_v_per_cm": field,
                    "specific_on_resistance_ohm_cm2": ron,
                },
            },
            "final_summary": {
                "artifacts": {"sentaurus_curve_csv": str(curve)},
                "metrics": {
                    "solver_backend": "sentaurus",
                    "curve_points": 3,
                },
            },
        },
    )
    return path


class SentaurusMutationEffectTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_continue_refine_when_leakage_improves_without_tradeoff(self) -> None:
        baseline = write_sentaurus_state(
            self.root / "baseline" / "sentaurus_state.json",
            leakage=1e-9,
            breakdown=-100.0,
            field=8e5,
            ron=0.05,
        )
        mutation = write_sentaurus_state(
            self.root / "mutation" / "sentaurus_state.json",
            leakage=4e-10,
            breakdown=-105.0,
            field=7.5e5,
            ron=0.052,
        )

        result = analyze_sentaurus_mutation_effect(
            SentaurusMutationEffectRequest(
                baseline_state_path=baseline,
                mutation_state_path=mutation,
                candidate={"candidate_id": "device.cmd:lifetime:LIFETIME_SCALE", "patches": [{"variable": "LIFETIME_SCALE"}]},
                goal_text="Reduce leakage while BV and Ron must not get worse.",
                output_path=self.root / "effect.json",
            )
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.decision, "continue_refine")
        self.assertTrue(result.worth_continuing)
        self.assertEqual(result.primary_metric, "leakage_abs_current_at_target_a")
        self.assertTrue(result.primary_improved)
        self.assertFalse(result.tradeoff_violations)
        self.assertTrue(Path(result.output_path).exists())
        self.assertTrue(Path(result.overlay_svg_path).exists())

    def test_blocks_for_pareto_when_primary_improves_but_field_regresses(self) -> None:
        baseline = write_sentaurus_state(
            self.root / "baseline" / "sentaurus_state.json",
            leakage=1e-9,
            breakdown=-100.0,
            field=8e5,
            ron=0.05,
        )
        mutation = write_sentaurus_state(
            self.root / "mutation" / "sentaurus_state.json",
            leakage=5e-10,
            breakdown=-102.0,
            field=1.1e6,
            ron=0.051,
        )

        result = analyze_sentaurus_mutation_effect(
            SentaurusMutationEffectRequest(
                baseline_state_path=baseline,
                mutation_state_path=mutation,
                candidate={"candidate_id": "device.cmd:lifetime:LIFETIME_SCALE"},
                goal_text="Reduce leakage without making field peak worse.",
            )
        )

        self.assertEqual(result.decision, "blocked_for_pareto_review")
        self.assertFalse(result.worth_continuing)
        self.assertEqual(result.tradeoff_violations[0]["metric"], "max_electric_field_v_per_cm")
        self.assertEqual(result.recommended_next_action, "pareto_or_constraint_review")

    def test_switches_target_when_primary_metric_worsens(self) -> None:
        baseline = write_sentaurus_state(
            self.root / "baseline" / "sentaurus_state.json",
            leakage=1e-9,
            breakdown=-100.0,
            field=8e5,
            ron=0.05,
        )
        mutation = write_sentaurus_state(
            self.root / "mutation" / "sentaurus_state.json",
            leakage=2e-9,
            breakdown=-100.0,
            field=8e5,
            ron=0.05,
        )

        result = analyze_sentaurus_mutation_effect(
            SentaurusMutationEffectRequest(
                baseline_state_path=baseline,
                mutation_state_path=mutation,
                candidate={"candidate_id": "device.cmd:lifetime:LIFETIME_SCALE"},
                goal_text="Reduce leakage.",
            )
        )

        self.assertEqual(result.decision, "switch_target")
        self.assertFalse(result.primary_improved)
        self.assertIn("leakage_abs_current_at_target_a", result.regressed_metrics)


if __name__ == "__main__":
    unittest.main()
