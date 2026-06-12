from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from tcad_agent.autonomous_devsim_agent import (
    AutonomousDevsimRequest,
    DevsimAgentActionKind,
    DevsimAgentStatus,
    run_autonomous_devsim_agent,
)
from tcad_agent.user_deck_runner import UserDeckRunRequest, run_user_deck


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DECK = PROJECT_ROOT / "tcad_agent" / "examples" / "user_deck_acceptance" / "pn_diode_acceptance_deck.py"


class PublicUserDeckAcceptanceTest(unittest.TestCase):
    def run_with_acceptance_root(self, root: Path, callback):
        previous = os.environ.get("ACTSOFT_USER_DECK_ACCEPTANCE_ROOT")
        os.environ["ACTSOFT_USER_DECK_ACCEPTANCE_ROOT"] = str(root)
        try:
            return callback()
        finally:
            if previous is None:
                os.environ.pop("ACTSOFT_USER_DECK_ACCEPTANCE_ROOT", None)
            else:
                os.environ["ACTSOFT_USER_DECK_ACCEPTANCE_ROOT"] = previous

    def test_public_devsim_user_deck_runs_and_reports_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def run():
                return run_user_deck(
                    UserDeckRunRequest(
                        deck_path=str(PUBLIC_DECK),
                        run_id="public_deck_direct",
                        run_root=root / "user_deck_states",
                    )
                )

            result = self.run_with_acceptance_root(root / "deck_runs", run)

            self.assertEqual(result["status"], "completed")
            metrics = result["quality_report"]["metrics"]
            artifacts = result["final_summary"]["artifacts"]
            self.assertEqual(metrics["curve_points"], 3)
            self.assertEqual(metrics["n_doping_cm3"], 1.0e18)
            self.assertTrue(Path(artifacts["csv"]).exists())
            self.assertTrue(Path(artifacts["plot"]).exists())
            self.assertTrue(Path(artifacts["reported_summary"]).exists())

    def test_autonomous_agent_ingests_patches_and_executes_public_devsim_deck(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def run():
                return run_autonomous_devsim_agent(
                    AutonomousDevsimRequest(
                        goal_text="读取公开 PN diode DEVSIM deck，把 N 区掺杂调低后运行并输出验收证据",
                        agent_id="public_user_deck_acceptance",
                        agent_root=root / "agents",
                        execute=True,
                        use_llm=False,
                        max_steps=6,
                        source_deck_path=str(PUBLIC_DECK),
                        deck_patches=[
                            {
                                "deck_path": "doping.n_doping_cm3",
                                "request_path": "n_doping_cm3",
                                "value": 8e17,
                            }
                        ],
                        allow_user_confirmation_actions=True,
                        generate_report=False,
                        generate_dashboard=False,
                    )
                )

            state = self.run_with_acceptance_root(root / "deck_runs", run)

            final_state = json.loads(Path(str(state.final_state_path)).read_text(encoding="utf-8"))

            self.assertEqual(state.status, DevsimAgentStatus.COMPLETED)
            self.assertEqual(
                [step.kind for step in state.steps],
                [
                    DevsimAgentActionKind.INGEST_DECK,
                    DevsimAgentActionKind.APPLY_DECK_PATCH,
                    DevsimAgentActionKind.RUN_USER_DECK,
                    DevsimAgentActionKind.RUN_PHYSICAL_BENCHMARK,
                    DevsimAgentActionKind.STOP_SUCCESS,
                ],
            )
            self.assertTrue(state.checkpoint["deck_patch_verified"])
            self.assertFalse(state.checkpoint["deck_patch_unverified"])
            self.assertTrue(Path(state.checkpoint["semantic_deck_diff"]).exists())
            self.assertEqual(final_state["quality_report"]["metrics"]["n_doping_cm3"], 8e17)
            self.assertEqual(final_state["quality_report"]["status"], "passed")
            self.assertTrue(Path(final_state["final_summary"]["artifacts"]["csv"]).exists())


if __name__ == "__main__":
    unittest.main()
