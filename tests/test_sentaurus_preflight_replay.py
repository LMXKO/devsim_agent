from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from tcad_agent.sentaurus_preflight import SentaurusPreflightRequest, run_sentaurus_preflight
from tcad_agent.sentaurus_replay import SentaurusReplayRequest, run_sentaurus_replay


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class SentaurusPreflightReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_project(self) -> Path:
        project = self.root / "project"
        project.mkdir()
        (project / "device.cmd").write_text(
            """
set LIFETIME_SCALE 1.0
File { Grid="@tdr@" Plot="@tdrdat@" Current="@plot@" Output="@log@" }
Electrode {
  { Name="anode" Voltage=0.0 }
  { Name="cathode" Voltage=0.0 }
}
Solve {
  Quasistationary( InitialStep=1e-3 MaxStep=0.1 Goal { Name="cathode" Voltage=-100 } ) {
    Coupled { Poisson Electron Hole }
  }
}
""".lstrip(),
            encoding="utf-8",
        )
        return project

    def write_fake_ssh(self) -> Path:
        remote_root = self.root / "fake_remote"
        remote_root.mkdir()
        path = self.root / "fake_ssh.py"
        path.write_text(
            f"""
import pathlib
import subprocess
import sys

remote_root = pathlib.Path({str(remote_root)!r})
command = sys.argv[-1].replace("/remote/actsoft", str(remote_root))
completed = subprocess.run(command, shell=True, executable="/bin/bash", capture_output=True, text=True)
sys.stdout.write(completed.stdout or "")
sys.stderr.write(completed.stderr or "")
raise SystemExit(completed.returncode)
""".lstrip(),
            encoding="utf-8",
        )
        return path

    def write_sentaurus_state(self, name: str, *, leakage: float, field: float, baseline: Path | None = None) -> Path:
        run_dir = self.root / name
        curve = run_dir / "curve.csv"
        curve.parent.mkdir(parents=True, exist_ok=True)
        curve.write_text(
            "voltage_v,current_a,electric_field_v_per_cm\n"
            f"0,{leakage},{field * 0.05}\n"
            f"-50,{leakage * 10},{field * 0.5}\n"
            f"-100,1e-6,{field}\n",
            encoding="utf-8",
        )
        state_path = run_dir / "sentaurus_state.json"
        repair_context = {"baseline_state_path": str(baseline)} if baseline else {}
        write_json(
            state_path,
            {
                "tool_name": "sentaurus_run",
                "status": "completed",
                "run_id": name,
                "run_dir": str(run_dir),
                "quality_report": {
                    "status": "passed",
                    "metrics": {
                        "solver_backend": "sentaurus",
                        "tcad_solver_invoked": True,
                        "curve_path": str(curve),
                        "curve_points": 3,
                        "curve_x_key": "voltage_v",
                        "curve_y_key": "current_a",
                        "curve_field_key": "electric_field_v_per_cm",
                        "leakage_abs_current_at_target_a": leakage,
                        "breakdown_voltage_at_threshold_v": -100,
                        "max_electric_field_v_per_cm": field,
                    },
                },
                "final_summary": {
                    "artifacts": {"sentaurus_curve_csv": str(curve)},
                    "metrics": {
                        "solver_backend": "sentaurus",
                        "tcad_solver_invoked": True,
                        "curve_points": 3,
                        "leakage_abs_current_at_target_a": leakage,
                        "breakdown_voltage_at_threshold_v": -100,
                        "max_electric_field_v_per_cm": field,
                    },
                },
                "repair_context": repair_context,
            },
        )
        return state_path

    def test_preflight_blocks_without_real_profile_or_project(self) -> None:
        result = run_sentaurus_preflight(
            SentaurusPreflightRequest(
                output_path=self.root / "preflight.json",
                report_path=self.root / "preflight.md",
            )
        )

        self.assertEqual(result.status, "blocked")
        self.assertFalse(result.ready_to_execute_real_sentaurus)
        self.assertTrue(result.blocked_code)
        self.assertTrue((self.root / "preflight.json").exists())
        self.assertTrue((self.root / "preflight.md").exists())
        self.assertIn("blocked_missing_sentaurus_profile", {check.code for check in result.checks})

    def test_preflight_ready_when_profile_project_deck_and_curve_contract_exist(self) -> None:
        project = self.write_project()
        result = run_sentaurus_preflight(
            SentaurusPreflightRequest(
                project_path=project,
                profile={
                    "profile_id": "unit_preflight",
                    "commands": {"sdevice": sys.executable},
                    "allowed_project_roots": [str(self.root)],
                    "env": {"LM_LICENSE_FILE": "unit-test-only"},
                    "curve_globs": ["*.csv"],
                },
                deck_files=["device.cmd"],
                require_license_hint=True,
            )
        )

        self.assertEqual(result.status, "ready")
        self.assertTrue(result.ready_to_execute_real_sentaurus)
        codes = {check.code for check in result.checks}
        self.assertIn("sentaurus_command_resolved", codes)
        self.assertIn("sentaurus_deck_ir_parseable", codes)
        self.assertIn("sentaurus_license_env_hint_present", codes)

    def test_preflight_ready_for_remote_ssh_profile_without_leaking_env_values(self) -> None:
        project = self.write_project()
        fake_ssh = self.write_fake_ssh()
        secret_license = "unit-preflight-license-secret-value"
        result = run_sentaurus_preflight(
            SentaurusPreflightRequest(
                project_path=project,
                profile={
                    "profile_id": "unit_remote_preflight",
                    "execution_mode": "remote_ssh",
                    "commands": {"sdevice": sys.executable},
                    "allowed_project_roots": [str(self.root)],
                    "env": {"LM_LICENSE_FILE": secret_license},
                    "curve_globs": ["*.csv"],
                    "remote": {
                        "host": "fakehost",
                        "remote_run_root": "/remote/actsoft",
                        "ssh_command": f"{sys.executable} {fake_ssh}",
                        "rsync_command": sys.executable,
                    },
                },
                deck_files=["device.cmd"],
                require_license_hint=True,
            )
        )

        self.assertEqual(result.status, "ready")
        self.assertTrue(result.ready_to_execute_real_sentaurus)
        codes = {check.code for check in result.checks}
        self.assertIn("remote_transport_command_resolved", codes)
        self.assertIn("remote_run_root_writeable", codes)
        self.assertIn("sentaurus_remote_command_resolved", codes)
        payload = result.model_dump_json()
        self.assertNotIn(secret_license, payload)
        self.assertNotIn("fakehost", payload)
        self.assertTrue(result.runtime_profile["remote"]["host_configured"])

    def test_replay_consumes_existing_sentaurus_states_without_running_solver(self) -> None:
        baseline = self.write_sentaurus_state("baseline", leakage=1e-9, field=8e5)
        mutation = self.write_sentaurus_state("mutation", leakage=5e-10, field=7.5e5, baseline=baseline)

        result = run_sentaurus_replay(
            SentaurusReplayRequest(
                baseline_state_path=baseline,
                mutation_state_path=mutation,
                candidate={"candidate_id": "device.cmd:lifetime:LIFETIME_SCALE"},
                goal_text="Reduce leakage without hurting BV or field peak.",
                output_dir=self.root / "replay",
            )
        )

        self.assertEqual(result.status, "completed")
        self.assertTrue(result.final_summary["metrics"]["sentaurus_replay_only"])
        self.assertFalse(result.final_summary["metrics"]["tcad_solver_invoked"])
        self.assertTrue(result.mutation_effect)
        self.assertTrue(result.lineage_archive)
        self.assertGreaterEqual(len(result.lineage_archive["entries"]), 2)
        self.assertTrue((self.root / "replay" / "sentaurus_replay_state.json").exists())


if __name__ == "__main__":
    unittest.main()
