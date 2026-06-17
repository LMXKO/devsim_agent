from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from tcad_agent.autonomous_devsim_agent import AutonomousDevsimRequest, DevsimAgentActionKind, DevsimAgentStatus, run_autonomous_devsim_agent
from tcad_agent.physical_benchmark import BenchmarkStatus, run_physical_benchmark
from tcad_agent.sentaurus import SentaurusRunRequest, SentaurusRuntimeProfile, run_sentaurus
from tcad_agent.sentaurus_deck import apply_sentaurus_semantic_patch_text, parse_sentaurus_deck_text


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

    def write_project(self) -> Path:
        project = self.root / "project"
        project.mkdir()
        (project / "device.cmd").write_text("set DRIFT_DOPING 1e15\n", encoding="utf-8")
        return project

    def write_structured_project(self) -> Path:
        project = self.root / "structured_project"
        project.mkdir()
        (project / "device.cmd").write_text(
            """
set DRIFT_DOPING 1e15
File {
  Grid="@tdr@"
  Plot="@tdrdat@"
}
Electrode {
  { Name="source" Voltage=0.0 }
  { Name="drain" Voltage=0.0 }
}
Physics {
  Mobility( DopingDep HighFieldSaturation )
  Recombination( SRH )
}
Math {
  Iterations=20
}
Solve {
  Coupled { Poisson Electron Hole }
}
""".lstrip(),
            encoding="utf-8",
        )
        return project

    def test_sentaurus_deck_ir_and_semantic_patch_update_common_sections(self) -> None:
        deck = (self.write_structured_project() / "device.cmd").read_text(encoding="utf-8")
        ir = parse_sentaurus_deck_text(deck, source_path="device.cmd")

        self.assertIn("Electrode", [section.name for section in ir.sections])
        self.assertIn("Math", [section.name for section in ir.sections])
        self.assertIn("DRIFT_DOPING", [variable.key for variable in ir.set_variables])

        updated, record, _ = apply_sentaurus_semantic_patch_text(
            deck,
            {"operation": "sentaurus_set_variable", "variable": "DRIFT_DOPING", "value": "2e15"},
            source_path="device.cmd",
        )
        self.assertTrue(record["verified"])
        self.assertIn("set DRIFT_DOPING 2e15", updated)

        updated, record, _ = apply_sentaurus_semantic_patch_text(
            updated,
            {
                "operation": "sentaurus_update_assignment",
                "section_path": ["Electrode"],
                "selector": {"Name": "drain"},
                "parameter": "Voltage",
                "value": -20,
            },
            source_path="device.cmd",
        )
        self.assertTrue(record["verified"])
        self.assertIn('{ Name="drain" Voltage=-20 }', updated)

        updated, record, _ = apply_sentaurus_semantic_patch_text(
            updated,
            {
                "operation": "sentaurus_upsert_assignment",
                "section_path": ["Math"],
                "parameter": "Digits",
                "value": 5,
            },
            source_path="device.cmd",
        )
        self.assertTrue(record["verified"])
        self.assertIn("Digits=5", updated)

        quasi = 'Solve { Quasistationary( InitialStep=1e-3 Goal { Name="drain" Voltage=5.0 } ) { Coupled { Poisson Electron Hole } } }\n'
        quasi_ir = parse_sentaurus_deck_text(quasi, source_path="quasi.cmd")
        self.assertIn("Quasistationary", [section.name for section in quasi_ir.sections])
        updated, record, _ = apply_sentaurus_semantic_patch_text(
            quasi,
            {
                "operation": "sentaurus_update_assignment",
                "section_path": ["Solve", "Quasistationary"],
                "parameter": "InitialStep",
                "value": "1e-4",
            },
            source_path="quasi.cmd",
        )
        self.assertTrue(record["verified"])
        self.assertIn("InitialStep=1e-4", updated)

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

    def test_sentaurus_runner_executes_remote_ssh_profile_and_redacts_env_values(self) -> None:
        project = self.write_project()
        (project / "fake_remote_sdevice.py").write_text(
            """
import pathlib
import sys

deck = pathlib.Path("device.cmd").read_text(encoding="utf-8")
if "3e15" not in deck:
    print("patch missing", file=sys.stderr)
    raise SystemExit(7)
pathlib.Path("remote_des.log").write_text("remote Sentaurus wrapper finished\\n", encoding="utf-8")
pathlib.Path(sys.argv[1]).write_text(
    "voltage_v,current_a,electric_field_v_per_cm\\n"
    "0,2e-12,1e4\\n"
    "-5,2e-9,3e5\\n"
    "-15,2e-6,9e5\\n",
    encoding="utf-8",
)
print("remote finished")
""".lstrip(),
            encoding="utf-8",
        )
        fake_ssh, fake_rsync, remote_root = self.write_fake_remote_transport()
        secret_license = "unit-license-secret-value"
        profile = SentaurusRuntimeProfile(
            profile_id="unit_remote_ssh",
            execution_mode="remote_ssh",
            commands={"sdevice": sys.executable},
            allowed_project_roots=[self.root],
            run_root=self.root / "runs",
            env={"LM_LICENSE_FILE": secret_license},
            remote={
                "host": "fakehost",
                "remote_run_root": "/remote/actsoft",
                "ssh_command": f"{sys.executable} {fake_ssh}",
                "rsync_command": f"{sys.executable} {fake_rsync}",
            },
        )

        state = run_sentaurus(
            SentaurusRunRequest(
                goal_text="通过远端 SSH profile 跑 Sentaurus，并回拉 CSV 曲线",
                project_path=project,
                profile=profile,
                run_id="remote_ssh_unit",
                command_args={"sdevice": ["fake_remote_sdevice.py", "sentaurus_extract.csv"]},
                patches=[
                    {
                        "file": "device.cmd",
                        "pattern": "set DRIFT_DOPING 1e15",
                        "replacement": "set DRIFT_DOPING 3e15",
                    }
                ],
                timeout_seconds=10,
            )
        )

        self.assertEqual(state.status, "completed")
        metrics = state.quality_report["metrics"]
        self.assertTrue(metrics["remote_execution"])
        self.assertEqual(metrics["remote_execution_mode"], "remote_ssh")
        self.assertEqual(metrics["sentaurus_steps"], 1)
        self.assertEqual(metrics["curve_points"], 3)
        self.assertIn("sentaurus_curve_csv", state.artifacts)
        self.assertTrue((remote_root / "remote_ssh_unit" / "project" / "sentaurus_extract.csv").exists())
        state_text = Path(state.state_path).read_text(encoding="utf-8")
        self.assertNotIn(secret_license, state_text)
        self.assertIn("<redacted>", state_text)
        self.assertTrue(state.runtime_profile["remote"]["host_configured"])

    def test_sentaurus_runner_executes_remote_slurm_profile(self) -> None:
        project = self.write_project()
        fake_ssh, fake_rsync, remote_root = self.write_fake_remote_transport()
        (project / "fake_remote_sdevice.py").write_text(
            """
import pathlib
import sys

deck = pathlib.Path("device.cmd").read_text(encoding="utf-8")
if "4e15" not in deck:
    print("patch missing", file=sys.stderr)
    raise SystemExit(8)
pathlib.Path("slurm_des.log").write_text("slurm Sentaurus wrapper finished\\n", encoding="utf-8")
pathlib.Path(sys.argv[1]).write_text(
    "voltage_v,current_a,electric_field_v_per_cm\\n"
    "0,3e-12,1e4\\n"
    "-8,3e-9,4e5\\n"
    "-18,3e-6,1e6\\n",
    encoding="utf-8",
)
""".lstrip(),
            encoding="utf-8",
        )
        (project / "fake_sbatch.py").write_text(
            f"""
#!/usr/bin/env python3
import os
import pathlib
import subprocess
import sys

remote_root = pathlib.Path({str(remote_root)!r})
script = pathlib.Path(sys.argv[-1])
patched = script.with_suffix(script.suffix + ".local")
patched.write_text(script.read_text(encoding="utf-8").replace("/remote/actsoft", str(remote_root)), encoding="utf-8")
os.chmod(patched, 0o755)
completed = subprocess.run(["bash", str(patched)], capture_output=True, text=True)
sys.stderr.write(completed.stderr or "")
print("12345")
raise SystemExit(0)
""".lstrip(),
            encoding="utf-8",
        )
        (project / "fake_squeue.py").write_text("#!/usr/bin/env python3\nraise SystemExit(0)\n", encoding="utf-8")
        (project / "fake_scancel.py").write_text("#!/usr/bin/env python3\nraise SystemExit(0)\n", encoding="utf-8")
        for name in ["fake_sbatch.py", "fake_squeue.py", "fake_scancel.py"]:
            (project / name).chmod(0o755)
        profile = SentaurusRuntimeProfile(
            profile_id="unit_remote_slurm",
            execution_mode="remote_slurm",
            commands={"sdevice": sys.executable},
            allowed_project_roots=[self.root],
            run_root=self.root / "runs",
            remote={
                "host": "fakehost",
                "remote_run_root": "/remote/actsoft",
                "ssh_command": f"{sys.executable} {fake_ssh}",
                "rsync_command": f"{sys.executable} {fake_rsync}",
                "slurm_submit_command": "./fake_sbatch.py",
                "slurm_status_command": "./fake_squeue.py",
                "slurm_cancel_command": "./fake_scancel.py",
                "slurm_poll_interval_seconds": 0.01,
            },
        )

        state = run_sentaurus(
            SentaurusRunRequest(
                goal_text="通过远端 Slurm profile 跑 Sentaurus，并回拉 CSV 曲线",
                project_path=project,
                profile=profile,
                run_id="remote_slurm_unit",
                command_args={"sdevice": ["fake_remote_sdevice.py", "sentaurus_extract.csv"]},
                patches=[
                    {
                        "file": "device.cmd",
                        "pattern": "set DRIFT_DOPING 1e15",
                        "replacement": "set DRIFT_DOPING 4e15",
                    }
                ],
                timeout_seconds=10,
            )
        )

        self.assertEqual(state.status, "completed")
        metrics = state.quality_report["metrics"]
        self.assertTrue(metrics["remote_execution"])
        self.assertEqual(metrics["remote_execution_mode"], "remote_slurm")
        self.assertEqual(metrics["remote_scheduler"], "slurm")
        self.assertEqual(metrics["sentaurus_steps"], 1)
        self.assertEqual(metrics["curve_points"], 3)
        self.assertIn("sentaurus_curve_csv", state.artifacts)

    def test_sentaurus_runner_applies_semantic_deck_patches_and_writes_ir(self) -> None:
        project = self.write_structured_project()
        script = self.write_fake_script(
            "fake_semantic_sdevice.py",
            """
import pathlib
import sys

deck = pathlib.Path("device.cmd").read_text(encoding="utf-8")
required = ["set DRIFT_DOPING 2e15", '{ Name="drain" Voltage=-20 }', "Iterations=40", "Digits=5"]
missing = [item for item in required if item not in deck]
if missing:
    print("missing semantic edits: " + ",".join(missing), file=sys.stderr)
    raise SystemExit(5)
pathlib.Path("sentaurus_extract.csv").write_text(
    "voltage_v,current_a,electric_field_v_per_cm\\n"
    "0,1e-12,1e4\\n"
    "-10,1e-9,2e5\\n"
    "-20,1e-6,8e5\\n",
    encoding="utf-8",
)
""".lstrip(),
        )
        profile = SentaurusRuntimeProfile(
            profile_id="unit_semantic_fake",
            commands={"sdevice": sys.executable},
            allowed_project_roots=[self.root],
            run_root=self.root / "runs",
        )

        state = run_sentaurus(
            SentaurusRunRequest(
                goal_text="用语义 patch 改 Sentaurus deck 中的漂移区掺杂、漏极偏压和 Math 精度",
                project_path=project,
                profile=profile,
                deck_files=["device.cmd"],
                command_args={"sdevice": [str(script)]},
                patches=[
                    {
                        "file": "device.cmd",
                        "operation": "sentaurus_set_variable",
                        "variable": "DRIFT_DOPING",
                        "value": "2e15",
                    },
                    {
                        "file": "device.cmd",
                        "operation": "sentaurus_update_assignment",
                        "section_path": ["Electrode"],
                        "selector": {"Name": "drain"},
                        "parameter": "Voltage",
                        "value": -20,
                    },
                    {
                        "file": "device.cmd",
                        "operation": "sentaurus_update_assignment",
                        "section_path": ["Math"],
                        "parameter": "Iterations",
                        "value": 40,
                    },
                    {
                        "file": "device.cmd",
                        "operation": "sentaurus_upsert_assignment",
                        "section_path": ["Math"],
                        "parameter": "Digits",
                        "value": 5,
                    },
                ],
                timeout_seconds=10,
            )
        )

        metrics = state.quality_report["metrics"]
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.quality_report["status"], "passed")
        self.assertEqual(metrics["sentaurus_patches_verified"], 4)
        self.assertEqual(metrics["sentaurus_deck_ir_files"], 1)
        self.assertTrue(any(key.startswith("sentaurus_deck_ir_") for key in state.artifacts))

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
        preflight_calls: list[dict] = []

        def fake_preflight(request: dict) -> dict:
            preflight_calls.append(request)
            return {
                "tool_name": "sentaurus_preflight",
                "status": "ready",
                "ready_to_execute_real_sentaurus": True,
                "output_path": str(self.root / "preflight.json"),
                "report_path": str(self.root / "preflight.md"),
            }

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
            max_steps=4,
            sentaurus_project_path=project,
            sentaurus_profile_path=self.root / "sentaurus_profile.json",
            sentaurus_request={"flow": ["sdevice"], "deck_files": ["device.cmd"]},
            generate_report=False,
            generate_dashboard=False,
        )

        state = run_autonomous_devsim_agent(
            request,
            runner_registry={
                "sentaurus_preflight": fake_preflight,
                "sentaurus_run": fake_sentaurus,
                "physical_benchmark": fake_benchmark,
            },
        )

        self.assertEqual(state.status, DevsimAgentStatus.COMPLETED)
        self.assertEqual(state.steps[0].kind, DevsimAgentActionKind.RUN_TOOL)
        self.assertEqual(state.steps[0].action["tool_name"], "sentaurus_preflight")
        self.assertEqual(state.steps[1].action["tool_name"], "sentaurus_run")
        self.assertEqual(preflight_calls[0]["project_path"], str(project))
        self.assertEqual(preflight_calls[0]["profile_path"], str(self.root / "sentaurus_profile.json"))
        self.assertEqual(preflight_calls[0]["deck_files"], ["device.cmd"])
        self.assertEqual(calls[0]["goal_text"], request.goal_text)
        self.assertEqual(calls[0]["project_path"], str(project))
        self.assertEqual(calls[0]["profile_path"], str(self.root / "sentaurus_profile.json"))
        self.assertEqual(calls[0]["deck_files"], ["device.cmd"])
        self.assertTrue(state.checkpoint["sentaurus_preflight_ready"])
        self.assertTrue(state.checkpoint["sentaurus_initial_run_done"])


if __name__ == "__main__":
    unittest.main()
