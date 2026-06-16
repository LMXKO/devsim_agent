from __future__ import annotations

import argparse
import json
import os
import subprocess
import shlex
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.sentaurus import SentaurusRuntimeProfile, is_remote_execution, normalized_execution_mode, remote_config, remote_shell_command
from tcad_agent.sentaurus_deck import parse_sentaurus_deck_file
from tcad_agent.task_spec import PROJECT_ROOT


class SentaurusPreflightCheck(BaseModel):
    code: str
    status: str
    message: str
    observed: dict[str, Any] = Field(default_factory=dict)


class SentaurusPreflightRequest(BaseModel):
    project_path: Path | None = None
    profile_path: Path | None = None
    profile: SentaurusRuntimeProfile | dict[str, Any] | None = None
    flow: list[str] = Field(default_factory=lambda: ["sdevice"])
    deck_files: list[str] = Field(default_factory=list)
    license_env_keys: list[str] = Field(default_factory=lambda: ["SNPSLMD_LICENSE_FILE", "LM_LICENSE_FILE"])
    require_license_hint: bool = True
    require_real_installation: bool = True
    output_path: Path | None = None
    report_path: Path | None = None


class SentaurusPreflightResult(BaseModel):
    tool_name: str = "sentaurus_preflight"
    schema_version: str = "actsoft.tcad.sentaurus_preflight.v1"
    status: str
    project_path: str | None = None
    profile_path: str | None = None
    runtime_profile: dict[str, Any] = Field(default_factory=dict)
    checks: list[SentaurusPreflightCheck] = Field(default_factory=list)
    ready_to_execute_real_sentaurus: bool = False
    blocked_code: str | None = None
    next_action: str | None = None
    output_path: str | None = None
    report_path: str | None = None
    failure_reason: str | None = None


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def pass_check(code: str, message: str, observed: dict[str, Any] | None = None) -> SentaurusPreflightCheck:
    return SentaurusPreflightCheck(code=code, status="passed", message=message, observed=observed or {})


def fail_check(code: str, message: str, observed: dict[str, Any] | None = None) -> SentaurusPreflightCheck:
    return SentaurusPreflightCheck(code=code, status="failed", message=message, observed=observed or {})


def warn_check(code: str, message: str, observed: dict[str, Any] | None = None) -> SentaurusPreflightCheck:
    return SentaurusPreflightCheck(code=code, status="warning", message=message, observed=observed or {})


def load_profile(request: SentaurusPreflightRequest) -> tuple[SentaurusRuntimeProfile | None, str | None]:
    if isinstance(request.profile, SentaurusRuntimeProfile):
        return request.profile, None
    if isinstance(request.profile, dict):
        return SentaurusRuntimeProfile.model_validate(request.profile), None
    if request.profile_path:
        path = request.profile_path.expanduser()
        if not path.exists():
            return None, f"Sentaurus profile does not exist: {path}"
        raw = json.loads(path.read_text(encoding="utf-8"))
        return SentaurusRuntimeProfile.model_validate(raw), None
    return None, "A real Sentaurus runtime profile is required and must stay outside the repository."


def command_display(profile: SentaurusRuntimeProfile, step: str) -> str:
    raw = profile.commands.get(step) or profile.commands.get("default")
    if raw:
        return raw
    if profile.sentaurus_home:
        return str((profile.sentaurus_home / "bin" / step).expanduser())
    return step


def resolve_command(raw: str) -> tuple[str | None, str]:
    parts = shlex.split(raw)
    if not parts:
        return None, "empty command"
    executable = parts[0]
    if Path(executable).is_absolute() or "/" in executable:
        path = Path(executable).expanduser()
        if path.exists():
            return str(path.resolve()), "path"
        return None, "path_missing"
    resolved = shutil.which(executable)
    if resolved:
        return resolved, "path_lookup"
    return None, "not_on_path"


def check_commands(profile: SentaurusRuntimeProfile, flow: list[str]) -> list[SentaurusPreflightCheck]:
    if is_remote_execution(profile):
        return check_remote_commands(profile, flow)
    checks: list[SentaurusPreflightCheck] = []
    actual_flow = flow or profile.default_flow or ["sdevice"]
    for step in actual_flow:
        raw = command_display(profile, step)
        resolved, source = resolve_command(raw)
        if resolved:
            checks.append(pass_check("sentaurus_command_resolved", f"Command for `{step}` is resolvable.", {"step": step, "command": raw, "resolved": resolved, "source": source}))
        else:
            checks.append(fail_check("blocked_missing_sentaurus_installation", f"Command for `{step}` is not resolvable.", {"step": step, "command": raw, "lookup": source}))
    return checks


def run_remote_check(profile: SentaurusRuntimeProfile, shell_command: str, *, timeout_seconds: float = 20.0) -> tuple[bool, dict[str, Any]]:
    try:
        completed = subprocess.run(
            remote_shell_command(profile, shell_command),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return (
            completed.returncode == 0,
            {
                "returncode": completed.returncode,
                "stdout_tail": (completed.stdout or "")[-1000:],
                "stderr_tail": (completed.stderr or "")[-1000:],
            },
        )
    except Exception as exc:
        return False, {"error": str(exc)}


def remote_command_check_shell(raw: str) -> str:
    parts = shlex.split(raw)
    if not parts:
        return "false"
    executable = parts[0]
    if Path(executable).is_absolute() or "/" in executable:
        return f"test -x {shlex.quote(executable)}"
    return f"command -v {shlex.quote(executable)} >/dev/null"


def check_remote_transport(profile: SentaurusRuntimeProfile) -> list[SentaurusPreflightCheck]:
    checks: list[SentaurusPreflightCheck] = []
    try:
        remote = remote_config(profile)
    except Exception as exc:
        return [fail_check("blocked_missing_remote_execution_profile", str(exc))]
    for label, raw in [("ssh", remote.ssh_command), ("rsync", remote.rsync_command)]:
        resolved, source = resolve_command(raw)
        if resolved:
            checks.append(pass_check("remote_transport_command_resolved", f"Local `{label}` transport command is resolvable.", {"transport": label, "resolved": resolved, "source": source}))
        else:
            checks.append(fail_check("blocked_missing_remote_transport", f"Local `{label}` transport command is not resolvable.", {"transport": label, "lookup": source}))
    ok, observed = run_remote_check(profile, f"mkdir -p {shlex.quote(str(remote.remote_run_root))} && test -w {shlex.quote(str(remote.remote_run_root))}")
    if ok:
        checks.append(pass_check("remote_run_root_writeable", "Remote run root exists or can be created and is writeable.", {"execution_mode": normalized_execution_mode(profile)}))
    else:
        checks.append(fail_check("blocked_remote_run_root_unavailable", "Remote run root is not reachable or writeable.", observed))
    return checks


def check_remote_scheduler(profile: SentaurusRuntimeProfile) -> list[SentaurusPreflightCheck]:
    if normalized_execution_mode(profile) != "remote_slurm":
        return []
    remote = remote_config(profile)
    checks: list[SentaurusPreflightCheck] = []
    for command in [remote.slurm_submit_command, remote.slurm_status_command, remote.slurm_cancel_command]:
        ok, observed = run_remote_check(profile, remote_command_check_shell(command))
        if ok:
            checks.append(pass_check("remote_scheduler_command_resolved", "Remote scheduler command is resolvable.", {"command": command}))
        else:
            checks.append(fail_check("blocked_missing_remote_scheduler", "Remote scheduler command is not resolvable.", {"command": command, **observed}))
    return checks


def check_remote_commands(profile: SentaurusRuntimeProfile, flow: list[str]) -> list[SentaurusPreflightCheck]:
    checks = check_remote_transport(profile)
    if any(check.status == "failed" for check in checks):
        return checks
    checks.extend(check_remote_scheduler(profile))
    actual_flow = flow or profile.default_flow or ["sdevice"]
    for step in actual_flow:
        raw = command_display(profile, step)
        ok, observed = run_remote_check(profile, remote_command_check_shell(raw))
        if ok:
            checks.append(pass_check("sentaurus_remote_command_resolved", f"Remote command for `{step}` is resolvable.", {"step": step, "command_name": shlex.split(raw)[0] if shlex.split(raw) else raw}))
        else:
            checks.append(fail_check("blocked_missing_sentaurus_installation", f"Remote command for `{step}` is not resolvable.", {"step": step, "command_name": shlex.split(raw)[0] if shlex.split(raw) else raw, **observed}))
    return checks


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def check_project(profile: SentaurusRuntimeProfile, project_path: Path | None, deck_files: list[str]) -> list[SentaurusPreflightCheck]:
    checks: list[SentaurusPreflightCheck] = []
    if project_path is None:
        return [fail_check("blocked_missing_sentaurus_project", "A user-owned Sentaurus project path is required for real execution.")]
    project = project_path.expanduser()
    if not project.exists():
        return [fail_check("blocked_missing_sentaurus_project", "Sentaurus project path does not exist.", {"project_path": str(project)})]
    project = project.resolve()
    roots = [root.expanduser().resolve() for root in profile.allowed_project_roots]
    if roots and not any(is_relative_to(project, root) for root in roots):
        checks.append(fail_check("blocked_project_outside_allowed_roots", "Project path is outside profile allowed_project_roots.", {"project_path": str(project), "allowed_roots": [str(root) for root in roots]}))
    else:
        checks.append(pass_check("sentaurus_project_allowed", "Project path exists and is allowed by the runtime profile.", {"project_path": str(project)}))
    actual_decks = deck_files or [str(path.relative_to(project)) for path in sorted(project.rglob("*.cmd"))[:12]]
    if not actual_decks:
        checks.append(fail_check("blocked_missing_sentaurus_deck", "No deck files were provided or discovered."))
        return checks
    for raw in actual_decks:
        deck_path = (project / raw).resolve() if not Path(raw).is_absolute() else Path(raw).expanduser().resolve()
        if not deck_path.exists():
            checks.append(fail_check("blocked_missing_sentaurus_deck", "Deck file does not exist.", {"deck_file": raw}))
            continue
        try:
            ir = parse_sentaurus_deck_file(deck_path)
            checks.append(pass_check("sentaurus_deck_ir_parseable", "Deck file is parseable by the conservative Sentaurus IR.", {"deck_file": raw, "sections": len(ir.sections), "variables": len(ir.set_variables), "warnings": ir.warnings}))
        except Exception as exc:
            checks.append(fail_check("blocked_unparseable_sentaurus_deck", "Deck file could not be parsed by the conservative Sentaurus IR.", {"deck_file": raw, "error": str(exc)}))
    return checks


def check_license_hint(profile: SentaurusRuntimeProfile, request: SentaurusPreflightRequest) -> list[SentaurusPreflightCheck]:
    configured = [key for key in request.license_env_keys if key in profile.env or os.environ.get(key)]
    if configured:
        return [pass_check("sentaurus_license_env_hint_present", "A license-related environment key is configured in the profile or current environment.", {"env_keys": configured})]
    if request.require_license_hint:
        return [fail_check("blocked_missing_license_configuration", "No license-related environment hint was found. Configure license access in the external profile or shell, not in git.", {"checked_env_keys": request.license_env_keys})]
    return [warn_check("sentaurus_license_env_hint_missing", "No license-related environment hint was found; execution may still work if a site wrapper handles license setup.", {"checked_env_keys": request.license_env_keys})]


def check_curve_contract(profile: SentaurusRuntimeProfile) -> SentaurusPreflightCheck:
    if profile.curve_globs:
        return pass_check("sentaurus_curve_extraction_contract_configured", "Profile declares CSV curve artifact globs for post-run extraction.", {"curve_globs": profile.curve_globs})
    return fail_check("blocked_missing_curve_extraction_contract", "Profile must declare CSV curve artifact globs so the agent can inspect real curves.")


def blocked_code(checks: list[SentaurusPreflightCheck]) -> str | None:
    for preferred in [
        "blocked_missing_remote_execution_profile",
        "blocked_missing_remote_transport",
        "blocked_remote_run_root_unavailable",
        "blocked_missing_remote_scheduler",
        "blocked_missing_sentaurus_installation",
        "blocked_missing_sentaurus_profile",
        "blocked_missing_license_configuration",
        "blocked_missing_sentaurus_project",
        "blocked_project_outside_allowed_roots",
        "blocked_missing_sentaurus_deck",
        "blocked_unparseable_sentaurus_deck",
        "blocked_missing_curve_extraction_contract",
    ]:
        if any(check.code == preferred and check.status == "failed" for check in checks):
            return preferred
    failed = [check for check in checks if check.status == "failed"]
    return failed[0].code if failed else None


def render_report(result: SentaurusPreflightResult) -> str:
    lines = [
        "# Sentaurus Preflight",
        "",
        f"Status: `{result.status}`",
        f"Ready: `{result.ready_to_execute_real_sentaurus}`",
    ]
    if result.blocked_code:
        lines.append(f"Blocked code: `{result.blocked_code}`")
    lines.extend(["", "## Checks", ""])
    for check in result.checks:
        lines.append(f"- `{check.status}` `{check.code}`: {check.message}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Commercial Sentaurus binaries, license strings, PDKs, process decks, calibrated models, and private run artifacts must remain outside the repository.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_sentaurus_preflight(request: SentaurusPreflightRequest) -> SentaurusPreflightResult:
    checks: list[SentaurusPreflightCheck] = []
    profile, profile_error = load_profile(request)
    if profile_error:
        checks.append(fail_check("blocked_missing_sentaurus_profile", profile_error))
        profile = SentaurusRuntimeProfile()
    else:
        checks.append(pass_check("sentaurus_profile_loaded", "Sentaurus runtime profile loaded.", profile.safe_summary()))
    checks.extend(check_commands(profile, request.flow))
    checks.extend(check_license_hint(profile, request))
    checks.extend(check_project(profile, request.project_path, request.deck_files))
    checks.append(check_curve_contract(profile))
    code = blocked_code(checks)
    result = SentaurusPreflightResult(
        status="blocked" if code else "ready",
        project_path=str(request.project_path.expanduser()) if request.project_path else None,
        profile_path=str(request.profile_path.expanduser()) if request.profile_path else None,
        runtime_profile=profile.safe_summary(),
        checks=checks,
        ready_to_execute_real_sentaurus=code is None,
        blocked_code=code,
        next_action=(
            "provide a real Sentaurus installation/profile/license/project outside git, then rerun preflight"
            if code
            else "run real Sentaurus baseline through the autonomous agent"
        ),
        failure_reason=code,
    )
    if request.output_path:
        output_path = request.output_path.expanduser().resolve()
        result.output_path = str(output_path)
        write_json(output_path, result.model_dump(mode="json"))
    if request.report_path:
        report_path = request.report_path.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(result), encoding="utf-8")
        result.report_path = str(report_path)
        if request.output_path:
            write_json(Path(result.output_path), result.model_dump(mode="json"))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight a user-owned real Sentaurus runtime profile and project.")
    parser.add_argument("--project", "--project-path", dest="project_path", type=Path, default=None)
    parser.add_argument("--profile", "--profile-path", dest="profile_path", type=Path, default=None)
    parser.add_argument("--flow", action="append", default=[])
    parser.add_argument("--deck-file", dest="deck_files", action="append", default=[])
    parser.add_argument("--no-license-hint-required", action="store_true")
    parser.add_argument("--output", "--output-path", dest="output_path", type=Path, default=None)
    parser.add_argument("--report", "--report-path", dest="report_path", type=Path, default=None)
    return parser.parse_args()


def request_from_args(args: argparse.Namespace) -> SentaurusPreflightRequest:
    return SentaurusPreflightRequest(
        project_path=args.project_path,
        profile_path=args.profile_path,
        flow=args.flow or ["sdevice"],
        deck_files=args.deck_files,
        require_license_hint=not args.no_license_hint_required,
        output_path=args.output_path,
        report_path=args.report_path,
    )


def main() -> None:
    result = run_sentaurus_preflight(request_from_args(parse_args()))
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
