from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tcad_agent.autonomous_devsim_agent import AutonomousDevsimRequest, DevsimAgentActionKind, DevsimAgentStatus, run_autonomous_devsim_agent
from tcad_agent.run_queue import QueueStatus, default_runner_registry, enqueue_run, get_item, run_queue_worker
from tcad_agent.sentaurus_profile_onboarding import (
    SentaurusProfileOnboardingRequest,
    run_sentaurus_profile_onboarding,
)


class SentaurusProfileOnboardingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_project(self) -> Path:
        project = self.root / "sentaurus_project"
        project.mkdir()
        (project / "device.cmd").write_text("set LIFETIME_SCALE 1.0\n", encoding="utf-8")
        return project

    def write_fake_script(self, name: str, body: str) -> Path:
        path = self.root / name
        path.write_text(body, encoding="utf-8")
        return path

    def write_fake_remote_transport(self) -> tuple[Path, Path, Path]:
        remote_root = self.root / "fake_remote"
        remote_root.mkdir()
        remote_prefix = "/remote/actsoft"
        fake_ssh = self.write_fake_script(
            "fake_ssh.py",
            f"""
import pathlib
import subprocess
import sys

remote_root = pathlib.Path({str(remote_root)!r})
remote_prefix = {remote_prefix!r}
command = sys.argv[-1].replace(remote_prefix, str(remote_root))
completed = subprocess.run(command, shell=True, executable="/bin/bash", capture_output=True, text=True)
sys.stdout.write(completed.stdout or "")
sys.stderr.write(completed.stderr or "")
raise SystemExit(completed.returncode)
""".lstrip(),
        )
        fake_rsync = self.write_fake_script(
            "fake_rsync.py",
            f"""
import pathlib
import shutil
import sys

remote_root = pathlib.Path({str(remote_root)!r})
remote_prefix = {remote_prefix!r}
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
source, destination = args[-2], args[-1]

def map_path(raw):
    raw = raw.rstrip("/")
    if ":" in raw:
        _, path = raw.split(":", 1)
        if path.startswith(remote_prefix):
            path = path[len(remote_prefix):].lstrip("/")
        return remote_root / path
    return pathlib.Path(raw)

src = map_path(source)
dst = map_path(destination)
if dst.exists():
    if dst.is_dir():
        shutil.rmtree(dst)
    else:
        dst.unlink()
dst.parent.mkdir(parents=True, exist_ok=True)
if src.is_dir():
    shutil.copytree(src, dst, symlinks=True)
else:
    shutil.copy2(src, dst)
""".lstrip(),
        )
        return fake_ssh, fake_rsync, remote_root

    def test_generates_remote_slurm_template_and_missing_inputs_without_secrets(self) -> None:
        project = self.write_project()
        with patch("tcad_agent.sentaurus_profile_onboarding.shutil.which", return_value="/usr/bin/fake"):
            result = run_sentaurus_profile_onboarding(
                SentaurusProfileOnboardingRequest(
                    goal_text="用集群 Slurm 跑 Sentaurus，先帮我生成远端 profile",
                    project_path=project,
                    deck_files=["device.cmd"],
                    output_dir=self.root / "onboarding",
                )
            )

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.inferred_execution_mode, "remote_slurm")
        self.assertIn("remote_host", result.missing_inputs)
        self.assertIn("remote_run_root", result.missing_inputs)
        self.assertIn("sentaurus_profile_path_or_accepted_generated_template", result.missing_inputs)
        self.assertEqual(result.blocked_code, "blocked_missing_remote_host")
        template = json.loads(Path(result.profile_template_path).read_text(encoding="utf-8"))
        self.assertEqual(template["execution_mode"], "remote_slurm")
        self.assertIsNone(template["remote"]["host"])
        self.assertEqual(template["env"], {})
        self.assertTrue(Path(result.output_path).exists())
        self.assertTrue(Path(result.report_path).exists())
        serialized = Path(result.output_path).read_text(encoding="utf-8")
        self.assertNotIn("sk-", serialized)
        self.assertNotIn("unit-license-secret-value", serialized)

    def test_agent_runs_onboarding_before_remote_sentaurus_profile_exists(self) -> None:
        calls: list[dict] = []

        def fake_onboarding(request: dict) -> dict:
            calls.append(request)
            return {
                "tool_name": "sentaurus_profile_onboarding",
                "status": "blocked",
                "ready_to_execute_real_sentaurus": False,
                "blocked_code": "blocked_missing_remote_host",
                "missing_inputs": ["remote_host", "remote_run_root", "sentaurus_profile_path_or_accepted_generated_template"],
                "profile_template_path": str(self.root / "profile_template.json"),
                "output_path": str(self.root / "onboarding.json"),
                "report_path": str(self.root / "onboarding.md"),
            }

        state = run_autonomous_devsim_agent(
            AutonomousDevsimRequest(
                goal_text="用远端集群跑 Sentaurus，自动配置 profile 并检查 ssh rsync slurm",
                agent_id="agent_remote_sentaurus_onboarding",
                agent_root=self.root / "agents",
                execute=True,
                use_llm=False,
                max_steps=3,
                generate_report=False,
                generate_dashboard=False,
            ),
            runner_registry={"sentaurus_profile_onboarding": fake_onboarding},
        )

        self.assertEqual(state.status, DevsimAgentStatus.WAITING_FOR_USER)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["goal_text"], state.goal_text)
        self.assertTrue(state.checkpoint["sentaurus_profile_onboarding_done"])
        self.assertEqual(state.checkpoint["sentaurus_profile_onboarding_blocked_code"], "blocked_missing_remote_host")
        self.assertEqual([step.kind for step in state.steps], [DevsimAgentActionKind.RUN_TOOL, DevsimAgentActionKind.ASK_USER])
        self.assertEqual((state.steps[-1].action.get("request") or {}).get("gate"), "sentaurus_profile_onboarding")

    def test_queue_registry_exposes_onboarding_runner(self) -> None:
        self.assertIn("sentaurus_profile_onboarding", default_runner_registry())

    def test_queue_soak_closes_remote_sentaurus_run_patch_report_loop(self) -> None:
        project = self.write_project()
        (project / "fake_remote_sdevice.py").write_text(
            """
import pathlib
import re
import sys

deck = pathlib.Path("device.cmd").read_text(encoding="utf-8")
match = re.search(r"set\\s+LIFETIME_SCALE\\s+([0-9.]+)", deck)
lifetime = float(match.group(1)) if match else 1.0
leakage = 1e-9 / max(lifetime, 1.0)
field = 8e5 * (1.0 - min((lifetime - 1.0) * 0.02, 0.04))
pathlib.Path("remote_des.log").write_text(f"remote wrapper lifetime={lifetime}\\n", encoding="utf-8")
pathlib.Path(sys.argv[1]).write_text(
    "voltage_v,current_a,electric_field_v_per_cm\\n"
    f"0,{leakage},1e4\\n"
    f"-10,{leakage * 1000},{field * 0.5}\\n"
    f"-30,{max(leakage * 1e6, 2e-6)},{field}\\n",
    encoding="utf-8",
)
print("remote sentaurus wrapper completed")
""".lstrip(),
            encoding="utf-8",
        )
        fake_ssh, fake_rsync, remote_root = self.write_fake_remote_transport()
        profile = {
            "profile_id": "queue_remote_ssh",
            "execution_mode": "remote_ssh",
            "commands": {"sdevice": sys.executable},
            "allowed_project_roots": [str(self.root)],
            "run_root": str(self.root / "sentaurus_runs"),
            "env": {"LM_LICENSE_FILE": "unit-license-secret-value"},
            "default_flow": ["sdevice"],
            "curve_globs": ["*.csv", "*_extract.csv", "*_iv.csv"],
            "artifact_globs": ["*.log", "*.out", "*.plt", "*.tdr", "*.csv"],
            "remote": {
                "host": "fakehost",
                "remote_run_root": "/remote/actsoft",
                "ssh_command": f"{sys.executable} {fake_ssh}",
                "rsync_command": f"{sys.executable} {fake_rsync}",
            },
        }
        queue_db = self.root / "queue.sqlite"
        item = enqueue_run(
            queue_db,
            tool_name="agent_soak",
            queue_id="remote_sentaurus_queue_loop",
            request={
                "goal_text": "Use remote Sentaurus SSH profile to reduce reverse leakage and write an engineer report.",
                "execute": True,
                "duration_hours": 0.05,
                "soak_root": str(self.root / "agent_soak"),
                "max_steps": 24,
                "step_slice": 24,
                "poll_interval_seconds": 0,
                "generate_cockpit": False,
                "compile_mission_spec": False,
                "enable_agent_memory": False,
                "enable_curve_guidance": False,
                "autonomous_request": {
                    "use_llm": False,
                    "sentaurus_project_path": str(project),
                    "sentaurus_request": {
                        "profile": profile,
                        "flow": ["sdevice"],
                        "deck_files": ["device.cmd"],
                        "command_args": {"sdevice": ["fake_remote_sdevice.py", "sentaurus_extract.csv"]},
                    },
                    "enable_experiment_design": True,
                    "max_experiment_design_rounds": 2,
                    "generate_report": True,
                    "generate_dashboard": False,
                },
            },
            priority=10,
        )

        worker = run_queue_worker(queue_db, owner="unit-worker", max_items=1)
        finished = get_item(queue_db, item.queue_id)

        self.assertEqual(worker.completed, 1)
        self.assertIsNotNone(finished)
        self.assertEqual(finished.status, QueueStatus.COMPLETED)
        result = finished.result
        self.assertEqual(result["status"], "completed")
        agent_state = json.loads(Path(result["agent_state_path"]).read_text(encoding="utf-8"))
        tool_names = [
            ((step.get("action") or {}).get("tool_name"))
            for step in agent_state.get("steps") or []
            if isinstance(step, dict)
        ]
        self.assertIn("sentaurus_preflight", tool_names)
        self.assertGreaterEqual(tool_names.count("sentaurus_run"), 2)
        self.assertTrue(agent_state["checkpoint"]["report_done"])
        final_state = json.loads(Path(result["final_state_path"]).read_text(encoding="utf-8"))
        metrics = final_state["quality_report"]["metrics"]
        self.assertTrue(metrics["remote_execution"])
        self.assertEqual(metrics["remote_execution_mode"], "remote_ssh")
        self.assertIn("sentaurus_curve_csv", final_state["artifacts"])
        self.assertTrue(Path(final_state["artifacts"]["sentaurus_curve_csv"]).exists())
        self.assertTrue(list(remote_root.glob("*/project/sentaurus_extract.csv")))
        serialized = Path(result["agent_state_path"]).read_text(encoding="utf-8")
        self.assertNotIn("unit-license-secret-value", serialized)


if __name__ == "__main__":
    unittest.main()
