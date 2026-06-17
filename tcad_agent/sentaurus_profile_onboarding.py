from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.sentaurus import SentaurusRuntimeProfile, normalized_execution_mode
from tcad_agent.sentaurus_preflight import (
    SentaurusPreflightCheck,
    SentaurusPreflightRequest,
    fail_check,
    pass_check,
    run_sentaurus_preflight,
    warn_check,
)
from tcad_agent.task_spec import PROJECT_ROOT


class SentaurusProfileOnboardingRequest(BaseModel):
    goal_text: str
    project_path: Path | None = None
    profile_path: Path | None = None
    execution_mode: str | None = None
    remote_host: str | None = None
    remote_run_root: str | None = None
    sentaurus_commands: dict[str, str] = Field(default_factory=dict)
    flow: list[str] = Field(default_factory=lambda: ["sdevice"])
    deck_files: list[str] = Field(default_factory=list)
    require_license_hint: bool = True
    run_preflight: bool = True
    output_dir: Path = PROJECT_ROOT / "runs" / "sentaurus_profile_onboarding"
    profile_template_path: Path | None = None
    output_path: Path | None = None
    report_path: Path | None = None
    preflight_output_path: Path | None = None
    preflight_report_path: Path | None = None


class SentaurusProfileOnboardingResult(BaseModel):
    tool_name: str = "sentaurus_profile_onboarding"
    schema_version: str = "actsoft.tcad.sentaurus_profile_onboarding.v1"
    status: str
    goal_text: str
    inferred_execution_mode: str
    project_path: str | None = None
    profile_path: str | None = None
    profile_template_path: str | None = None
    preflight_path: str | None = None
    preflight_report_path: str | None = None
    checks: list[SentaurusPreflightCheck] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    blocked_code: str | None = None
    ready_to_execute_real_sentaurus: bool = False
    next_action: str | None = None
    output_path: str | None = None
    report_path: str | None = None
    preflight: dict[str, Any] | None = None
    profile_summary: dict[str, Any] = Field(default_factory=dict)


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_slug(text: str, *, fallback: str = "sentaurus_remote") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip().lower()).strip("._-")
    return (slug or fallback)[:80]


def text_mentions_any(text: str, tokens: list[str]) -> bool:
    lowered = text.lower()
    return any(token.lower() in lowered for token in tokens)


def infer_execution_mode(goal_text: str, explicit: str | None = None) -> str:
    raw = (explicit or "").strip().lower()
    if raw in {"remote_slurm", "slurm", "sbatch", "cluster"}:
        return "remote_slurm"
    if raw in {"remote_ssh", "ssh", "remote", "workstation"}:
        return "remote_ssh"
    if text_mentions_any(goal_text, ["slurm", "sbatch", "squeue", "scancel", "集群", "队列", "调度"]):
        return "remote_slurm"
    return "remote_ssh"


def extract_remote_host(goal_text: str, explicit: str | None = None) -> str | None:
    if explicit and explicit.strip():
        return explicit.strip()
    patterns = [
        r"(?:host|machine|server|login|node)\s*[:=]\s*([A-Za-z0-9._@-]+)",
        r"ssh\s+([A-Za-z0-9._@-]+)",
        r"\b([A-Za-z0-9._-]+@[A-Za-z0-9._-]+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, goal_text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def extract_remote_run_root(goal_text: str, explicit: str | None = None) -> str | None:
    if explicit and explicit.strip():
        return explicit.strip()
    patterns = [
        r"(?:remote_run_root|remote-root|run_root)\s*[:=]\s*(/[^\s,;]+)",
        r"(?:scratch|workdir|workspace)\s*[:=]\s*(/[^\s,;]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, goal_text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def local_command_check(label: str, raw: str) -> SentaurusPreflightCheck:
    parts = raw.split()
    executable = parts[0] if parts else raw
    resolved = shutil.which(executable)
    if resolved:
        return pass_check(
            "remote_onboarding_local_command_resolved",
            f"Local `{label}` command is available.",
            {"command": executable, "resolved": resolved},
        )
    return fail_check(
        "blocked_missing_remote_transport",
        f"Local `{label}` command is not available on PATH.",
        {"command": executable},
    )


def template_profile_dict(
    request: SentaurusProfileOnboardingRequest,
    *,
    execution_mode: str,
    remote_host: str | None,
    remote_run_root: str | None,
) -> dict[str, Any]:
    commands = dict(request.sentaurus_commands) or {"sdevice": "sdevice"}
    allowed_project_roots = []
    if request.project_path:
        allowed_project_roots.append(str(request.project_path.expanduser().resolve().parent))
    return {
        "profile_id": f"{execution_mode}_sentaurus_profile",
        "execution_mode": execution_mode,
        "commands": commands,
        "allowed_project_roots": allowed_project_roots,
        "run_root": str((request.output_dir / "sentaurus_runs").expanduser()),
        "env": {},
        "default_flow": request.flow or ["sdevice"],
        "curve_globs": ["*.csv", "*_extract.csv", "*_iv.csv"],
        "artifact_globs": ["*.log", "*.out", "*.plt", "*.tdr", "*.csv"],
        "remote": {
            "host": remote_host,
            "remote_run_root": remote_run_root,
            "ssh_command": "ssh",
            "rsync_command": "rsync",
            "slurm_submit_command": "sbatch",
            "slurm_status_command": "squeue",
            "slurm_cancel_command": "scancel",
            "slurm_poll_interval_seconds": 10.0,
            "remote_setup_commands": [],
        },
    }


def preflight_profile_dict(template: dict[str, Any], *, remote_host: str | None, remote_run_root: str | None) -> dict[str, Any]:
    profile = json.loads(json.dumps(template))
    profile["remote"]["host"] = remote_host
    profile["remote"]["remote_run_root"] = remote_run_root
    return profile


def collect_missing_inputs(
    request: SentaurusProfileOnboardingRequest,
    *,
    remote_host: str | None,
    remote_run_root: str | None,
    preflight: dict[str, Any] | None,
) -> list[str]:
    missing: list[str] = []
    if not request.project_path:
        missing.append("sentaurus_project_path")
    if not request.profile_path:
        missing.append("sentaurus_profile_path_or_accepted_generated_template")
    if not remote_host:
        missing.append("remote_host")
    if not remote_run_root:
        missing.append("remote_run_root")
    if preflight and preflight.get("blocked_code") == "blocked_missing_license_configuration":
        missing.append("license_environment_hint")
    for check in (preflight or {}).get("checks") or []:
        if not isinstance(check, dict) or check.get("status") != "failed":
            continue
        code = str(check.get("code") or "")
        if code == "blocked_missing_remote_scheduler":
            missing.append("remote_slurm_commands")
        elif code == "blocked_missing_sentaurus_installation":
            missing.append("remote_sentaurus_command")
        elif code == "blocked_missing_sentaurus_deck":
            missing.append("sentaurus_deck_files")
    return list(dict.fromkeys(missing))


def first_blocked_code(checks: list[SentaurusPreflightCheck], preflight: dict[str, Any] | None) -> str | None:
    for preferred in [
        "blocked_missing_remote_transport",
        "blocked_missing_remote_host",
        "blocked_missing_remote_run_root",
        "blocked_missing_sentaurus_profile",
    ]:
        if any(check.code == preferred and check.status == "failed" for check in checks):
            return preferred
    if preflight and preflight.get("blocked_code"):
        return str(preflight["blocked_code"])
    failed = [check for check in checks if check.status == "failed"]
    return failed[0].code if failed else None


def render_report(result: SentaurusProfileOnboardingResult) -> str:
    lines = [
        "# Sentaurus Remote Profile Onboarding",
        "",
        f"Status: `{result.status}`",
        f"Execution mode: `{result.inferred_execution_mode}`",
        f"Ready: `{result.ready_to_execute_real_sentaurus}`",
    ]
    if result.blocked_code:
        lines.append(f"Blocked code: `{result.blocked_code}`")
    if result.profile_template_path:
        lines.append(f"Profile template: `{result.profile_template_path}`")
    if result.preflight_path:
        lines.append(f"Preflight: `{result.preflight_path}`")
    if result.missing_inputs:
        lines.extend(["", "## Missing Inputs", ""])
        for item in result.missing_inputs:
            lines.append(f"- `{item}`")
    lines.extend(["", "## Checks", ""])
    for check in result.checks:
        lines.append(f"- `{check.status}` `{check.code}`: {check.message}")
    preflight_checks = (result.preflight or {}).get("checks") if isinstance(result.preflight, dict) else None
    if preflight_checks:
        lines.extend(["", "## Preflight Checks", ""])
        for check in preflight_checks:
            if isinstance(check, dict):
                lines.append(f"- `{check.get('status')}` `{check.get('code')}`: {check.get('message')}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This artifact is an onboarding plan. Keep real host names, license values, PDKs, calibrated models, commercial decks, and site wrappers outside git.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_sentaurus_profile_onboarding(request: SentaurusProfileOnboardingRequest) -> SentaurusProfileOnboardingResult:
    execution_mode = infer_execution_mode(request.goal_text, request.execution_mode)
    remote_host = extract_remote_host(request.goal_text, request.remote_host)
    remote_run_root = extract_remote_run_root(request.goal_text, request.remote_run_root)
    output_dir = request.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_template_path = (
        request.profile_template_path.expanduser().resolve()
        if request.profile_template_path
        else output_dir / f"{safe_slug(request.goal_text)}_{utc_timestamp()}_profile_template.json"
    )
    output_path = request.output_path.expanduser().resolve() if request.output_path else output_dir / "sentaurus_profile_onboarding.json"
    report_path = request.report_path.expanduser().resolve() if request.report_path else output_dir / "sentaurus_profile_onboarding.md"
    preflight_output_path = (
        request.preflight_output_path.expanduser().resolve()
        if request.preflight_output_path
        else output_dir / "sentaurus_preflight.json"
    )
    preflight_report_path = (
        request.preflight_report_path.expanduser().resolve()
        if request.preflight_report_path
        else output_dir / "sentaurus_preflight.md"
    )

    checks: list[SentaurusPreflightCheck] = [
        pass_check("remote_onboarding_execution_mode_inferred", "Remote execution mode inferred from request.", {"execution_mode": execution_mode})
    ]
    if remote_host:
        checks.append(pass_check("remote_onboarding_host_configured", "A remote host was provided or inferred."))
    else:
        checks.append(fail_check("blocked_missing_remote_host", "Provide the remote login host in the external profile or request."))
    if remote_run_root:
        checks.append(pass_check("remote_onboarding_run_root_configured", "A remote run root was provided or inferred."))
    else:
        checks.append(fail_check("blocked_missing_remote_run_root", "Provide a remote run root where patched projects can be copied."))
    checks.append(local_command_check("ssh", "ssh"))
    checks.append(local_command_check("rsync", "rsync"))
    if execution_mode == "remote_slurm":
        checks.append(
            warn_check(
                "remote_onboarding_slurm_checked_by_remote_preflight",
                "Slurm commands are checked on the remote host during preflight once SSH is reachable.",
                {"commands": ["sbatch", "squeue", "scancel"]},
            )
        )

    template = template_profile_dict(request, execution_mode=execution_mode, remote_host=remote_host, remote_run_root=remote_run_root)
    write_json(profile_template_path, template)
    profile = SentaurusRuntimeProfile.model_validate(preflight_profile_dict(template, remote_host=remote_host, remote_run_root=remote_run_root))
    preflight_result = None
    if request.run_preflight:
        preflight_request = SentaurusPreflightRequest(
            project_path=request.project_path,
            profile_path=request.profile_path,
            profile=None if request.profile_path else profile,
            flow=request.flow or ["sdevice"],
            deck_files=request.deck_files,
            require_license_hint=request.require_license_hint,
            output_path=preflight_output_path,
            report_path=preflight_report_path,
        )
        preflight_result = run_sentaurus_preflight(preflight_request).model_dump(mode="json")

    missing_inputs = collect_missing_inputs(
        request,
        remote_host=remote_host,
        remote_run_root=remote_run_root,
        preflight=preflight_result,
    )
    code = first_blocked_code(checks, preflight_result)
    ready = bool(preflight_result and preflight_result.get("ready_to_execute_real_sentaurus")) and not code
    result = SentaurusProfileOnboardingResult(
        status="ready" if ready else "blocked",
        goal_text=request.goal_text,
        inferred_execution_mode=execution_mode,
        project_path=str(request.project_path.expanduser()) if request.project_path else None,
        profile_path=str(request.profile_path.expanduser()) if request.profile_path else None,
        profile_template_path=str(profile_template_path),
        preflight_path=preflight_result.get("output_path") if preflight_result else None,
        preflight_report_path=preflight_result.get("report_path") if preflight_result else None,
        checks=checks,
        missing_inputs=missing_inputs,
        blocked_code=code,
        ready_to_execute_real_sentaurus=ready,
        next_action=(
            "use the external profile in autonomous Sentaurus execution"
            if ready
            else "fill the generated external profile template, keep secrets outside git, then rerun onboarding or sentaurus_preflight"
        ),
        output_path=str(output_path),
        report_path=str(report_path),
        preflight=preflight_result,
        profile_summary=profile.safe_summary(),
    )
    write_json(output_path, result.model_dump(mode="json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and preflight a remote Sentaurus execution profile template.")
    parser.add_argument("--goal", "--goal-text", dest="goal_text", required=True)
    parser.add_argument("--project", "--project-path", dest="project_path", type=Path, default=None)
    parser.add_argument("--profile", "--profile-path", dest="profile_path", type=Path, default=None)
    parser.add_argument("--execution-mode", default=None)
    parser.add_argument("--remote-host", default=None)
    parser.add_argument("--remote-run-root", default=None)
    parser.add_argument("--flow", action="append", default=[])
    parser.add_argument("--deck-file", dest="deck_files", action="append", default=[])
    parser.add_argument("--no-license-hint-required", action="store_true")
    parser.add_argument("--no-preflight", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "sentaurus_profile_onboarding")
    parser.add_argument("--output", "--output-path", dest="output_path", type=Path, default=None)
    parser.add_argument("--report", "--report-path", dest="report_path", type=Path, default=None)
    return parser.parse_args()


def request_from_args(args: argparse.Namespace) -> SentaurusProfileOnboardingRequest:
    return SentaurusProfileOnboardingRequest(
        goal_text=args.goal_text,
        project_path=args.project_path,
        profile_path=args.profile_path,
        execution_mode=args.execution_mode,
        remote_host=args.remote_host,
        remote_run_root=args.remote_run_root,
        flow=args.flow or ["sdevice"],
        deck_files=args.deck_files,
        require_license_hint=not args.no_license_hint_required,
        run_preflight=not args.no_preflight,
        output_dir=args.output_dir,
        output_path=args.output_path,
        report_path=args.report_path,
    )


def main() -> None:
    result = run_sentaurus_profile_onboarding(request_from_args(parse_args()))
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
