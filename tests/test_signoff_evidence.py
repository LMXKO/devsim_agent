from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.physical_benchmark import run_physical_benchmark


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class SignoffEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_ready_pack_when_required_evidence_is_present(self) -> None:
        state_path = self.root / "mosfet" / "state.json"
        write_json(
            state_path,
            {
                "tool_name": "mosfet_2d_id_sweep",
                "status": "completed",
                "request": {
                    "golden_metrics": {"ion_ioff_ratio": {"expected": 1e5, "relative_tolerance": 0.1}},
                    "tcad_deck_spec": {
                        "device_family": "2d_mosfet",
                        "signoff_requirements": {
                            "required_level": "engineering_signoff",
                            "require_convergence_evidence": True,
                            "golden_metrics": {"ion_ioff_ratio": {"expected": 1e5, "relative_tolerance": 0.1}},
                        },
                    },
                },
                "final_summary": {"artifacts": {"csv": "curve.csv", "plot": "curve.svg"}},
                "quality_report": {
                    "status": "passed",
                    "metrics": {
                        "subthreshold_swing_mv_dec": 80.0,
                        "ion_ioff_ratio": 1e5,
                        "vth_at_threshold_current_v": 0.45,
                        "relative_delta": 0.02,
                    },
                },
            },
        )

        result = run_physical_benchmark(state_path)
        pack = result.summary["signoff_evidence_pack"]

        self.assertEqual(pack["verdict"], "ready")
        self.assertEqual(pack["label_zh"], "签核证据齐套")
        self.assertEqual(pack["missing_evidence"], [])
        item_status = {item["name"]: item["status"] for item in pack["required_items"]}
        self.assertEqual(item_status["convergence_evidence"], "present")
        self.assertEqual(item_status["golden_or_measured_comparison"], "present")

    def test_pack_blocks_planned_surrogate(self) -> None:
        state_path = self.root / "gan" / "state.json"
        write_json(
            state_path,
            {
                "tool_name": "extended_device_sweep",
                "status": "completed",
                "request": {"device_type": "gan_hemt_id_bv", "fidelity": "compact"},
                "quality_report": {"status": "passed", "metrics": {"device_type": "gan_hemt_id_bv", "fidelity": "compact"}},
            },
        )

        result = run_physical_benchmark(state_path)
        pack = result.summary["signoff_evidence_pack"]

        self.assertEqual(pack["verdict"], "blocked")
        self.assertIn("planned industrial runner missing", pack["blocking_reasons"])


if __name__ == "__main__":
    unittest.main()
