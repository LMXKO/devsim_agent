from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from tcad_agent.autonomous_devsim_agent import AutonomousDevsimRequest, DevsimAgentActionKind, DevsimAgentStatus, run_autonomous_devsim_agent
from tcad_agent.physical_benchmark import BenchmarkStatus, run_physical_benchmark
from tcad_agent.sentaurus import SentaurusRunRequest, SentaurusRuntimeProfile, run_sentaurus


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class SentaurusRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_fake_script(self, name: str, body: str) -> Path:
        path = self.root / name
        path.write_text(body, encoding="utf-8")
        return path

    def write_project(self) -> Path:
        project = self.root / "project"
        project.mkdir()
        (project / "device.cmd").write_text("set DRIFT_DOPING 1e15\n", encoding="utf-8")
        return project

    def test_sentaurus_runner_executes_configured_command_applies_patch_and_extracts_curve(self) -> None:
        project = self.write_project()
        script = self.write_fake_script(
            "fake_sdevice.py",
            """
import pathlib
import sys

deck = pathlib.Path("device.cmd").read_text(encoding="utf-8")
if "2e15" not in deck:
    print("patch missing", file=sys.stderr)
    raise SystemExit(3)
pathlib.Path("n1_des.log").write_text("Sentaurus Device finished\\n", encoding="utf-8")
pathlib.Path(sys.argv[1]).write_text(
    "voltage_v,current_a,electric_field_v_per_cm\\n"
    "0,1e-12,1e4\\n"
    "-10,1e-9,2e5\\n"
    "-20,1e-6,8e5\\n",
    encoding="utf-8",
)
print("finished")
""".lstrip(),
        )
        profile = SentaurusRuntimeProfile(
            profile_id="unit_fake",
            commands={"sdevice": sys.executable},
            allowed_project_roots=[self.root],
            run_root=self.root / "runs",
        )

        state = run_sentaurus(
            SentaurusRunRequest(
                goal_text="把 BV 推到 20V 并观察 field peak",
                project_path=project,
                profile=profile,
                command_args={"sdevice": [str(script), "sentaurus_extract.csv"]},
                patches=[
                    {
                        "file": "device.cmd",
                        "pattern": "set DRIFT_DOPING 1e15",
                        "replacement": "set DRIFT_DOPING 2e15",
                        "reason": "unit semantic patch",
                    }
                ],
                timeout_seconds=10,
            )
        )

        metrics = state.quality_report["metrics"]
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.quality_report["status"], "passed")
        self.assertEqual(metrics["sentaurus_patches_verified"], 1)
        self.assertEqual(metrics["curve_points"], 3)
        self.assertEqual(metrics["solver_backend"], "sentaurus")
        self.assertTrue(metrics["curve_shape"]["threshold_bracket_x"])
        self.assertIn("sentaurus_curve_csv", state.artifacts)
        self.assertTrue(Path(state.state_path).exists())
        self.assertTrue((Path(state.run_dir) / "sentaurus_patch.diff").exists())

        benchmark = run_physical_benchmark(Path(state.state_path))
        codes = {check.code for check in benchmark.checks}
        self.assertEqual(benchmark.status, BenchmarkStatus.PASSED)
        self.assertIn("sentaurus_external_solver_invoked", codes)
        self.assertIn("sentaurus_curve_extracted", codes)
        self.assertIn("sentaurus_patches_verified", codes)

    def test_sentaurus_runner_classifies_convergence_failure_from_realistic_log_text(self) -> None:
        project = self.write_project()
        script = self.write_fake_script(
            "fake_fail.py",
            """
import sys
print("Newton failed: failed to converge at bias step", file=sys.stderr)
raise SystemExit(7)
""".lstrip(),
        )
        profile = SentaurusRuntimeProfile(
            profile_id="unit_fake_fail",
            commands={"sdevice": sys.executable},
            allowed_project_roots=[self.root],
            run_root=self.root / "runs",
        )

        state = run_sentaurus(
            SentaurusRunRequest(
                goal_text="修复 Sentaurus 击穿扫描收敛问题",
                project_path=project,
                profile=profile,
                command_args={"sdevice": [str(script)]},
                timeout_seconds=10,
            )
        )

        codes = {issue["code"] for issue in state.quality_report["issues"]}
        self.assertEqual(state.status, "failed")
        self.assertEqual(state.quality_report["status"], "failed")
        self.assertIn("sentaurus_convergence_issue", codes)
        self.assertIn("sentaurus_nonzero_returncode", codes)
        self.assertTrue(state.repair_context["candidate_next_actions"])

    def test_sentaurus_runner_blocks_required_unverified_patch(self) -> None:
        project = self.write_project()
        script = self.write_fake_script(
            "fake_curve.py",
            """
import pathlib
pathlib.Path("sentaurus_extract.csv").write_text(
    "voltage_v,current_a\\n0,1e-12\\n-1,1e-9\\n",
    encoding="utf-8",
)
""".lstrip(),
        )
        profile = SentaurusRuntimeProfile(
            profile_id="unit_patch_fail",
            commands={"sdevice": sys.executable},
            allowed_project_roots=[self.root],
            run_root=self.root / "runs",
        )

        state = run_sentaurus(
            SentaurusRunRequest(
                goal_text="验证 unmatched patch 不应被当成可信 mutation",
                project_path=project,
                profile=profile,
                command_args={"sdevice": [str(script)]},
                patches=[
                    {
                        "file": "device.cmd",
                        "pattern": "set DRIFT_DOPING 9e99",
                        "replacement": "set DRIFT_DOPING 8e14",
                    }
                ],
                timeout_seconds=10,
            )
        )

        codes = {issue["code"] for issue in state.quality_report["issues"]}
        self.assertEqual(state.status, "failed")
        self.assertEqual(state.quality_report["status"], "failed")
        self.assertIn("sentaurus_patch_unverified", codes)

    def test_autonomous_agent_routes_natural_language_to_sentaurus_when_project_context_exists(self) -> None:
        project = self.write_project()
        state_path = self.root / "fake_sentaurus" / "sentaurus_state.json"
        calls: list[dict] = []

        def fake_sentaurus(request: dict) -> dict:
            calls.append(request)
            write_json(
                state_path,
                {
                    "tool_name": "sentaurus_run",
                    "status": "completed",
                    "run_id": "fake_sentaurus",
                    "run_dir": str(state_path.parent),
                    "quality_report": {
                        "status": "passed",
                        "issues": [],
                        "metrics": {
                            "solver_backend": "sentaurus",
                            "tcad_solver_invoked": True,
                            "curve_points": 3,
                        },
                    },
                    "final_summary": {
                        "artifacts": {},
                        "metrics": {
                            "solver_backend": "sentaurus",
                            "tcad_solver_invoked": True,
                            "curve_points": 3,
                        },
                    },
                },
            )
            return {"status": "completed", "state_path": str(state_path)}

        def fake_benchmark(request: dict) -> dict:
            return {"status": "completed", "benchmark_path": str(self.root / "benchmark.json")}

        request = AutonomousDevsimRequest(
            goal_text="用 Sentaurus 跑这个项目，降低漏电并检查 BV/Ron 权衡",
            agent_id="agent_sentaurus",
            agent_root=self.root / "agents",
            execute=True,
            use_llm=False,
            max_steps=3,
            sentaurus_project_path=project,
            sentaurus_profile_path=self.root / "sentaurus_profile.json",
            sentaurus_request={"flow": ["sdevice"], "deck_files": ["device.cmd"]},
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
        self.assertEqual(state.steps[0].kind, DevsimAgentActionKind.RUN_TOOL)
        self.assertEqual(state.steps[0].action["tool_name"], "sentaurus_run")
        self.assertEqual(calls[0]["goal_text"], request.goal_text)
        self.assertEqual(calls[0]["project_path"], str(project))
        self.assertEqual(calls[0]["profile_path"], str(self.root / "sentaurus_profile.json"))
        self.assertEqual(calls[0]["deck_files"], ["device.cmd"])
        self.assertTrue(state.checkpoint["sentaurus_initial_run_done"])


if __name__ == "__main__":
    unittest.main()
