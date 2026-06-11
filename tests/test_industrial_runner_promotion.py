from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.industrial_runner_promotion import build_industrial_runner_promotion_plan


class IndustrialRunnerPromotionTest(unittest.TestCase):
    def test_builds_runner_promotion_work_package_for_gan_hemt(self) -> None:
        live_lookup = {
            "status": "completed",
            "verified_source_ids": ["genius_tcad_open"],
            "findings": [{"source_id": "genius_tcad_open", "status": "verified"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "promotion.json"
            plan = build_industrial_runner_promotion_plan(
                "GaN HEMT current collapse and BV signoff",
                template_id="gan_hemt_id_bv",
                simulator="devsim",
                live_lookup_result=live_lookup,
                output_path=output,
            )
            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(plan.status, "completed")
        self.assertTrue(plan.promotion_required)
        self.assertEqual(plan.template_id, "gan_hemt_id_bv")
        self.assertEqual(saved["evidence_dossier"]["verified_source_ids"], ["genius_tcad_open"])
        stage_ids = [stage.stage_id for stage in plan.stages]
        self.assertEqual(
            stage_ids,
            [
                "public_evidence_and_license_gate",
                "runner_contract",
                "geometry_mesh_model_implementation",
                "metric_extraction",
                "convergence_and_quality",
                "golden_correlation_and_signoff",
                "autonomous_e2e_validation",
            ],
        )
        self.assertTrue(any("polarization" in " ".join(stage.actions).lower() for stage in plan.stages))
        self.assertTrue(any("long_run_validation" in item for item in plan.acceptance_tests))

    def test_power_mosfet_plan_exposes_real_devsim_runner(self) -> None:
        plan = build_industrial_runner_promotion_plan(
            "Power MOSFET LDMOS BV Ron field peak",
            template_id="power_mosfet_bv_ron",
            simulator="devsim",
        )

        self.assertEqual(plan.status, "completed")
        self.assertTrue(plan.promotion_required)
        self.assertTrue(plan.real_runner_available)
        self.assertEqual(plan.real_runner_id, "power_mosfet_bv_ron_devsim_1d")
        self.assertIn("extended_device_sweep", plan.real_runner_command)
        self.assertNotIn("run_id", plan.real_runner_command)
        self.assertEqual(plan.next_action, "run_real_runner_and_close_convergence_gaps")
        stages = {stage.stage_id: stage for stage in plan.stages}
        self.assertEqual(stages["runner_contract"].status, "completed")
        self.assertTrue(any("power_mosfet_bv_ron_devsim_1d" in action for action in stages["runner_contract"].actions))


if __name__ == "__main__":
    unittest.main()
