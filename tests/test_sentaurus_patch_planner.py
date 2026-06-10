from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.autonomous_devsim_agent import (
    AutonomousDevsimRequest,
    DevsimAgentActionKind,
    DevsimAgentStatus,
    run_autonomous_devsim_agent,
)
from tcad_agent.sentaurus_patch_planner import SentaurusPatchPlannerRequest, plan_sentaurus_patches


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT_ROOT / "tcad_agent" / "examples" / "sentaurus_fixtures"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class SentaurusPatchPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_plans_verified_bv_goal_and_convergence_step_patches(self) -> None:
        plan = plan_sentaurus_patches(
            SentaurusPatchPlannerRequest(
                goal_text="Ramp reverse BV to 1200V and reduce the step when convergence is difficult.",
                project_path=FIXTURES / "power_diode_bv",
                deck_files=["device.cmd"],
                output_path=self.root / "plan.json",
            )
        )

        self.assertEqual(plan.status, "completed")
        self.assertTrue(Path(plan.output_path).exists())
        candidates = {candidate.candidate_id: candidate for candidate in plan.candidates}
        convergence = candidates["device.cmd:convergence_step_control"]
        self.assertEqual(convergence.verified_patch_count, len(convergence.patches))
        self.assertTrue(any(patch["parameter"] == "InitialStep" and patch["value"] == "0.0001" for patch in convergence.patches))
        self.assertTrue(any(patch["parameter"] == "Iterations" for patch in convergence.patches))

        bias_candidates = [candidate for candidate in plan.candidates if ":bv_goal_cathode_" in candidate.candidate_id]
        self.assertEqual(len(bias_candidates), 1)
        bias_patch = bias_candidates[0].patches[0]
        self.assertEqual(bias_patch["operation"], "sentaurus_update_assignment")
        self.assertEqual(bias_patch["section_path"], ["Solve", "Quasistationary", "Goal"])
        self.assertEqual(bias_patch["selector"], {"Name": "cathode"})
        self.assertEqual(bias_patch["parameter"], "Voltage")
        self.assertEqual(bias_patch["value"], "-1200")
        self.assertEqual(bias_candidates[0].verified_patch_count, 1)

    def test_prefers_lifetime_for_leakage_goal_with_tradeoff_guard(self) -> None:
        plan = plan_sentaurus_patches(
            SentaurusPatchPlannerRequest(
                goal_text="Reduce leakage while BV and Ron must not get worse.",
                project_path=FIXTURES / "power_diode_bv",
                deck_files=["device.cmd"],
            )
        )

        self.assertEqual(plan.status, "completed")
        self.assertIsNotNone(plan.selected_candidate)
        selected = plan.selected_candidate
        self.assertIn("LIFETIME_SCALE", json.dumps(selected.patches))
        self.assertEqual(selected.risk_level, "low")
        self.assertFalse(selected.requires_user_confirmation)

    def test_blocks_high_risk_geometry_candidate_without_confirmation(self) -> None:
        project = self.root / "field_plate_project"
        project.mkdir()
        (project / "device.cmd").write_text(
            """
set FIELD_PLATE_LENGTH 1.0

Physics {
  Mobility( DopingDep )
}
""".lstrip(),
            encoding="utf-8",
        )

        plan = plan_sentaurus_patches(
            SentaurusPatchPlannerRequest(
                goal_text="Reduce electric field peak by tuning the field plate.",
                project_path=project,
                deck_files=["device.cmd"],
            )
        )

        self.assertEqual(plan.status, "blocked_for_user_confirmation")
        self.assertIsNone(plan.selected_candidate)
        self.assertEqual(plan.candidates[0].risk_level, "high")
        self.assertTrue(plan.candidates[0].requires_user_confirmation)

    def test_autonomous_agent_plans_and_executes_safe_sentaurus_patch_candidate(self) -> None:
        project = self.root / "sentaurus_project"
        project.mkdir()
        (project / "device.cmd").write_text(
            """
set LIFETIME_SCALE 1.0

Math {
  Iterations=20
}
""".lstrip(),
            encoding="utf-8",
        )
        calls: list[dict] = []
        benchmark_sources: list[str] = []

        def fake_sentaurus(request: dict) -> dict:
            calls.append(request)
            run_dir = self.root / f"sentaurus_run_{len(calls)}"
            state_path = run_dir / "sentaurus_state.json"
            write_json(
                state_path,
                {
                    "tool_name": "sentaurus_run",
                    "status": "completed",
                    "run_id": run_dir.name,
                    "run_dir": str(run_dir),
                    "project_path": str(project),
                    "project_copy_path": str(project),
                    "request": {
                        "goal_text": request["goal_text"],
                        "project_path": str(project),
                        "deck_files": request.get("deck_files") or ["device.cmd"],
                    },
                    "quality_report": {
                        "status": "passed",
                        "issues": [],
                        "metrics": {
                            "solver_backend": "sentaurus",
                            "tcad_solver_invoked": True,
                            "curve_points": 3,
                            "leakage_current_a": 1e-10 if len(calls) == 1 else 5e-11,
                        },
                    },
                    "final_summary": {
                        "artifacts": {"project_copy": str(project)},
                        "metrics": {
                            "solver_backend": "sentaurus",
                            "tcad_solver_invoked": True,
                            "curve_points": 3,
                        },
                        "parameters": {"deck_files": ["device.cmd"]},
                    },
                },
            )
            return {"status": "completed", "state_path": str(state_path)}

        def fake_benchmark(request: dict) -> dict:
            benchmark_sources.append(str(request["source"]))
            return {"status": "completed", "benchmark_path": str(self.root / "benchmark.json")}

        request = AutonomousDevsimRequest(
            goal_text="Reduce Sentaurus leakage without changing geometry.",
            agent_id="agent_sentaurus_patch",
            agent_root=self.root / "agents",
            execute=True,
            use_llm=False,
            max_steps=5,
            sentaurus_project_path=project,
            sentaurus_request={"flow": ["sdevice"], "deck_files": ["device.cmd"]},
            enable_experiment_design=True,
            max_experiment_design_rounds=1,
            generate_report=False,
            generate_dashboard=False,
        )

        state = run_autonomous_devsim_agent(
            request,
            runner_registry={
                "sentaurus_run": fake_sentaurus,
                "physical_benchmark": fake_benchmark,
            },
        )

        self.assertEqual(state.status, DevsimAgentStatus.COMPLETED)
        self.assertIn(DevsimAgentActionKind.PLAN_SENTAURUS_PATCH, [step.kind for step in state.steps])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["sentaurus_patch_candidate_id"], "device.cmd:lifetime:LIFETIME_SCALE")
        self.assertEqual(calls[1]["patches"][0]["operation"], "sentaurus_set_variable")
        self.assertEqual(calls[1]["patches"][0]["variable"], "LIFETIME_SCALE")
        self.assertEqual(calls[1]["patches"][0]["value"], "2")
        self.assertEqual(len(benchmark_sources), 1)
        self.assertTrue(Path(state.checkpoint["sentaurus_patch_plan_path"]).exists())
        self.assertEqual(state.checkpoint["executed_sentaurus_patch_candidates"], 1)
        mutation_state = json.loads(Path(state.latest_state_path).read_text(encoding="utf-8"))
        effect = mutation_state["sentaurus_mutation_effect_analysis"]
        self.assertEqual(effect["decision"], "continue_refine")
        self.assertEqual(effect["primary_metric"], "leakage_current_a")
        self.assertTrue(effect["worth_continuing"])
        self.assertEqual(state.checkpoint["latest_sentaurus_mutation_effect_analysis"]["decision"], "continue_refine")


if __name__ == "__main__":
    unittest.main()
