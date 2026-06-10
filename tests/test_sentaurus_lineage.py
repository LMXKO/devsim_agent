from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.sentaurus_lineage import SentaurusLineageArchiveRequest, build_sentaurus_lineage_archive


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_state(path: Path, *, leakage: float, field: float, baseline: Path | None = None, decision: str | None = None) -> Path:
    state = {
        "tool_name": "sentaurus_run",
        "status": "completed",
        "run_id": path.parent.name,
        "quality_report": {
            "status": "passed",
            "metrics": {
                "leakage_abs_current_at_target_a": leakage,
                "max_electric_field_v_per_cm": field,
                "breakdown_voltage_at_threshold_v": -100.0,
                "specific_on_resistance_ohm_cm2": 0.05,
                "curve_points": 3,
            },
        },
        "final_summary": {"artifacts": {}, "metrics": {"solver_backend": "sentaurus"}},
    }
    if baseline:
        state["repair_context"] = {"baseline_state_path": str(baseline)}
    if decision:
        state["sentaurus_mutation_effect_analysis"] = {
            "decision": decision,
            "candidate_id": "device.cmd:lifetime:LIFETIME_SCALE",
            "candidate": {
                "candidate_id": "device.cmd:lifetime:LIFETIME_SCALE",
                "patches": [{"file": "device.cmd", "operation": "sentaurus_set_variable", "variable": "LIFETIME_SCALE", "value": "2"}],
            },
            "worth_continuing": decision == "continue_refine",
            "primary_metric": "leakage_abs_current_at_target_a",
            "improved_metrics": ["leakage_abs_current_at_target_a"] if decision == "continue_refine" else [],
            "regressed_metrics": ["max_electric_field_v_per_cm"] if decision == "blocked_for_pareto_review" else [],
            "tradeoff_violations": [{"metric": "max_electric_field_v_per_cm"}] if decision == "blocked_for_pareto_review" else [],
            "rationale": "test lineage",
        }
    write_json(path, state)
    return path


class SentaurusLineageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_archive_collects_chain_and_marks_pareto_front(self) -> None:
        baseline = write_state(self.root / "baseline" / "sentaurus_state.json", leakage=1e-9, field=8e5)
        mutation = write_state(
            self.root / "mutation" / "sentaurus_state.json",
            leakage=4e-10,
            field=7.8e5,
            baseline=baseline,
            decision="continue_refine",
        )

        archive = build_sentaurus_lineage_archive(
            SentaurusLineageArchiveRequest(
                source_state_path=mutation,
                output_path=self.root / "lineage.json",
            )
        )

        self.assertEqual(archive.status, "completed")
        self.assertTrue(Path(archive.output_path).exists())
        self.assertEqual(len(archive.entries), 2)
        self.assertEqual(archive.entries[-1].decision, "continue_refine")
        self.assertIn("sentaurus_002", archive.pareto_front)
        self.assertEqual(archive.best_entry.lineage_id, "sentaurus_002")


if __name__ == "__main__":
    unittest.main()
