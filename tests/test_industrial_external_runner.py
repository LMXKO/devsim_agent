from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tcad_agent.industrial_external_runner import IndustrialExternalRunnerRequest, run_industrial_external_runner
from tcad_agent.industrial_runner_registry import preferred_runner_for_template, runner_descriptors_for_template


class IndustrialExternalRunnerTest(unittest.TestCase):
    def test_gan_registry_prefers_real_external_runner_before_surrogate(self) -> None:
        preferred = preferred_runner_for_template("gan_hemt_id_bv")
        runner_ids = {runner.runner_id for runner in runner_descriptors_for_template("gan_hemt_id_bv")}

        self.assertEqual(preferred.runner_id, "gan_hemt_sentaurus_external")
        self.assertIn("gan_hemt_id_bv_physics_1d", runner_ids)

    def test_external_runner_waits_for_user_owned_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = run_industrial_external_runner(
                IndustrialExternalRunnerRequest(
                    goal_text="GaN HEMT BV current collapse",
                    template_id="gan_hemt_id_bv",
                    run_id="gan_wait",
                    run_root=Path(tmp),
                )
            )

        self.assertEqual(state.status, "waiting_for_external_workspace")
        self.assertEqual(state.quality_report["status"], "suspicious")
        self.assertTrue(state.final_summary["metrics"]["external_workspace_required"])
        self.assertIn("gan_hemt_sentaurus_external", state.runner_contract["external_runner_ids"])
        self.assertTrue(state.runner_contract["requires_user_owned_workspace"])


if __name__ == "__main__":
    unittest.main()

