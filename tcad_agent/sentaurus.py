from __future__ import annotations

import difflib
import json
import os
import posixpath
import re
import shlex
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.curve_diagnostics import curve_shape_diagnostic, finite_float, infer_x_y_keys, load_curve_rows
from tcad_agent.process_control import run_cancellable
from tcad_agent.sentaurus_deck import apply_sentaurus_semantic_patch_text, parse_sentaurus_deck_file
from tcad_agent.task_spec import PROJECT_ROOT


class SentaurusRemoteExecutionProfile(BaseModel):
    """Remote execution transport for user-owned Sentaurus environments."""

    host: str | None = None
    remote_run_root: str | None = None
    ssh_command: str = "ssh"
    ssh_args: list[str] = Field(default_factory=list)
    rsync_command: str = "rsync"
    rsync_args: list[str] = Field(default_factory=lambda: ["-az", "--delete"])
    slurm_submit_command: str = "sbatch"
    slurm_status_command: str = "squeue"
    slurm_cancel_command: str = "scancel"
    slurm_submit_args: list[str] = Field(default_factory=lambda: ["--parsable"])
    slurm_poll_interval_seconds: float = Field(default=10.0, gt=0)
    remote_setup_commands: list[str] = Field(default_factory=list)

    @staticmethod
    def command_name(raw: str | None) -> str | None:
        if not raw:
            return None
        parts = shlex.split(raw)
        return Path(parts[0]).name if parts else None

    def safe_summary(self) -> dict[str, Any]:
        return {
            "host_configured": bool(self.host),
            "remote_run_root_configured": bool(self.remote_run_root),
            "ssh_command": self.command_name(self.ssh_command),
            "rsync_command": self.command_name(self.rsync_command),
            "slurm_submit_command": self.command_name(self.slurm_submit_command),
            "slurm_status_command": self.command_name(self.slurm_status_command),
            "slurm_cancel_command": self.command_name(self.slurm_cancel_command),
            "slurm_poll_interval_seconds": self.slurm_poll_interval_seconds,
            "remote_setup_command_count": len(self.remote_setup_commands),
        }


class SentaurusRuntimeProfile(BaseModel):
    """Sentaurus execution profile for local or remote user-owned runtimes.

    The profile intentionally keeps commercial installation paths, license
    variables, PDK roots, and command overrides outside the repository. Tests
    can provide fake commands through this same interface without pretending to
    generate real Synopsys physics results.
    """

    profile_id: str = "local"
    execution_mode: str = "local"
    sentaurus_home: Path | None = None
    commands: dict[str, str] = Field(default_factory=dict)
    allowed_project_roots: list[Path] = Field(default_factory=list)
    run_root: Path = PROJECT_ROOT / "runs" / "sentaurus"
    env: dict[str, str] = Field(default_factory=dict)
    default_flow: list[str] = Field(default_factory=lambda: ["sdevice"])
    curve_globs: list[str] = Field(default_factory=lambda: ["*.csv", "*_extract.csv", "*_iv.csv"])
    artifact_globs: list[str] = Field(
        default_factory=lambda: ["*.log", "*_des.log", "*.plt", "*_des.plt", "*.tdr", "*_des.tdr", "*.csv"]
    )
    remote: SentaurusRemoteExecutionProfile = Field(default_factory=SentaurusRemoteExecutionProfile)

    def command_vector(self, step: str, args: list[str] | None = None) -> list[str]:
        raw = self.commands.get(step) or self.commands.get("default")
        if raw:
            base = shlex.split(raw)
        elif self.sentaurus_home:
            base = [str((self.sentaurus_home / "bin" / step).expanduser())]
        else:
            base = [step]
        return [*base, *(args or [])]

    def safe_summary(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "execution_mode": normalized_execution_mode(self),
            "sentaurus_home_configured": self.sentaurus_home is not None,
            "commands": sorted(self.commands),
            "allowed_project_roots": [str(path.expanduser()) for path in self.allowed_project_roots],
            "default_flow": self.default_flow,
            "curve_globs": self.curve_globs,
            "artifact_globs": self.artifact_globs,
            "env_keys": sorted(self.env),
            "remote": self.remote.safe_summary(),
        }


class SentaurusPatch(BaseModel):
    file: str
    operation: str = "replace_text"
    pattern: str | None = None
    replacement: str | None = None
    regex: bool = False
    json_path: str | None = None
    section_path: list[str] | str | None = None
    selector: dict[str, Any] = Field(default_factory=dict)
    parameter: str | None = None
    variable: str | None = None
    model: str | None = None
    insert_if_missing: bool = False
    value: Any = None
    reason: str | None = None
    required: bool = True


class SentaurusRunRequest(BaseModel):
    goal_text: str
    project_path: Path
    profile_path: Path | None = None
    profile: SentaurusRuntimeProfile | dict[str, Any] | None = None
    run_id: str | None = None
    run_root: Path | None = None
    flow: list[str] = Field(default_factory=list)
    command_args: dict[str, list[str]] = Field(default_factory=dict)
    deck_files: list[str] = Field(default_factory=list)
    patches: list[SentaurusPatch | dict[str, Any]] = Field(default_factory=list)
    reference_curve_path: Path | None = None
    breakdown_current_threshold_a: float = 1.0e-6
    timeout_seconds: float = Field(default=3600.0, gt=0)
    cancel_file: str | None = None
    execute: bool = True


class SentaurusStepResult(BaseModel):
    step: str
    command: list[str]
    returncode: int | None
    stdout_path: str
    stderr_path: str
    stdout_tail: str
    stderr_tail: str


class SentaurusRunState(BaseModel):
    tool_name: str = "sentaurus_run"
    status: str
    run_id: str
    run_dir: str
    project_path: str
    project_copy_path: str
    started_at: str
    completed_at: str | None = None
    request: dict[str, Any] = Field(default_factory=dict)
    runtime_profile: dict[str, Any] = Field(default_factory=dict)
    commands: list[dict[str, Any]] = Field(default_factory=list)
    patches: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    log_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    final_summary: dict[str, Any] = Field(default_factory=dict)
    quality_report: dict[str, Any] = Field(default_factory=dict)
    repair_context: dict[str, Any] = Field(default_factory=dict)
    next_action: str | None = None
    failure_reason: str | None = None
    state_path: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_run_id(prefix: str = "sentaurus") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def tail(text: str | None, limit: int = 4000) -> str:
    return (text or "")[-limit:]


def load_runtime_profile(request: SentaurusRunRequest) -> SentaurusRuntimeProfile:
    if isinstance(request.profile, SentaurusRuntimeProfile):
        profile = request.profile
    elif isinstance(request.profile, dict):
        profile = SentaurusRuntimeProfile.model_validate(request.profile)
    elif request.profile_path:
        raw = json.loads(request.profile_path.expanduser().read_text(encoding="utf-8"))
        profile = SentaurusRuntimeProfile.model_validate(raw)
    else:
        profile = SentaurusRuntimeProfile()
    if request.run_root:
        profile = profile.model_copy(update={"run_root": request.run_root})
    return profile


def normalized_execution_mode(profile: SentaurusRuntimeProfile) -> str:
    raw = (profile.execution_mode or "local").strip().lower()
    aliases = {
        "ssh": "remote_ssh",
        "remote": "remote_ssh",
        "slurm": "remote_slurm",
    }
    return aliases.get(raw, raw)


def is_remote_execution(profile: SentaurusRuntimeProfile) -> bool:
    return normalized_execution_mode(profile) in {"remote_ssh", "remote_slurm"}


def fake_interface_only(profile: SentaurusRuntimeProfile) -> bool:
    return "fake" in profile.profile_id.lower() or "contract" in profile.profile_id.lower()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def ensure_allowed_project(profile: SentaurusRuntimeProfile, project_path: Path) -> Path:
    resolved = project_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Sentaurus project path does not exist: {resolved}")
    roots = [root.expanduser().resolve() for root in profile.allowed_project_roots]
    if roots and not any(is_relative_to(resolved, root) for root in roots):
        allowed = ", ".join(str(root) for root in roots)
        raise ValueError(f"Sentaurus project path is outside allowed roots: {resolved}; allowed roots: {allowed}")
    return resolved


def clone_project(project_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    project_copy = destination / "project"
    if project_copy.exists():
        raise FileExistsError(f"Sentaurus project copy already exists: {project_copy}")
    if project_path.is_file():
        project_copy.mkdir(parents=True, exist_ok=False)
        shutil.copy2(project_path, project_copy / project_path.name)
    else:
        shutil.copytree(
            project_path,
            project_copy,
            symlinks=True,
            ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
        )
    return project_copy


def project_relative_path(project_copy: Path, raw: str) -> Path:
    if not raw or Path(raw).is_absolute() or ".." in Path(raw).parts:
        raise ValueError(f"Sentaurus patch file must be a project-relative path: {raw}")
    target = (project_copy / raw).resolve()
    root = project_copy.resolve()
    if not is_relative_to(target, root):
        raise ValueError(f"Sentaurus patch target escapes project copy: {raw}")
    return target


def set_nested_json_value(data: dict[str, Any], dotted_path: str, value: Any) -> None:
    current: Any = data
    parts = [part for part in dotted_path.split(".") if part]
    if not parts:
        raise ValueError("json_path must not be empty")
    for part in parts[:-1]:
        if not isinstance(current, dict):
            raise ValueError(f"json_path crosses a non-object node: {dotted_path}")
        current = current.setdefault(part, {})
    if not isinstance(current, dict):
        raise ValueError(f"json_path target parent is not an object: {dotted_path}")
    current[parts[-1]] = value


def apply_one_patch(project_copy: Path, patch: SentaurusPatch) -> dict[str, Any]:
    target = project_relative_path(project_copy, patch.file)
    record: dict[str, Any] = {
        "file": patch.file,
        "operation": patch.operation,
        "reason": patch.reason,
        "required": patch.required,
        "applied": False,
        "verified": False,
    }
    if not target.exists():
        record["error"] = "target file missing"
        return record
    before = target.read_text(encoding="utf-8", errors="replace")
    after = before
    try:
        if patch.operation.startswith("sentaurus_"):
            after, semantic_record, ir = apply_sentaurus_semantic_patch_text(
                before,
                patch.model_dump(mode="json"),
                source_path=patch.file,
            )
            record.update(
                {
                    key: value
                    for key, value in semantic_record.items()
                    if key not in {"applied", "verified", "diff"}
                }
            )
            record["ir_sections"] = len(ir.sections)
            record["ir_assignments"] = len(ir.assignments)
            if not semantic_record.get("applied"):
                record["error"] = semantic_record.get("error") or "semantic patch was not applied"
                return record
        elif patch.operation == "json_set" or patch.json_path:
            data = json.loads(before)
            if not isinstance(data, dict):
                raise ValueError("JSON patch target must be an object")
            if not patch.json_path:
                raise ValueError("json_set requires json_path")
            set_nested_json_value(data, patch.json_path, patch.value)
            after = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
            record["json_path"] = patch.json_path
            record["value"] = patch.value
        elif patch.operation in {"replace_text", "replace_regex"}:
            if patch.pattern is None or patch.replacement is None:
                raise ValueError("text replacement requires pattern and replacement")
            if patch.regex or patch.operation == "replace_regex":
                after, count = re.subn(patch.pattern, patch.replacement, before, count=1, flags=re.MULTILINE)
            else:
                count = before.count(patch.pattern)
                after = before.replace(patch.pattern, patch.replacement, 1)
            record["matches"] = count
            record["pattern"] = patch.pattern
            if count <= 0:
                record["error"] = "pattern not found"
                return record
        else:
            raise ValueError(f"unsupported Sentaurus patch operation: {patch.operation}")
    except Exception as exc:
        record["error"] = str(exc)
        return record
    if after == before and not record.get("already_present"):
        record["error"] = "patch produced no content change"
        return record
    target.write_text(after, encoding="utf-8")
    record["applied"] = True
    record["verified"] = True
    record["diff"] = "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{patch.file}",
            tofile=f"b/{patch.file}",
            lineterm="",
        )
    )
    return record


def apply_sentaurus_patches(project_copy: Path, raw_patches: list[SentaurusPatch | dict[str, Any]], run_dir: Path) -> list[dict[str, Any]]:
    patches = [patch if isinstance(patch, SentaurusPatch) else SentaurusPatch.model_validate(patch) for patch in raw_patches]
    records = [apply_one_patch(project_copy, patch) for patch in patches]
    diff_text = "\n\n".join(str(record.get("diff") or "") for record in records if record.get("diff"))
    if diff_text:
        (run_dir / "sentaurus_patch.diff").write_text(diff_text + "\n", encoding="utf-8")
    return records


def sentaurus_deck_artifact_candidates(
    project_copy: Path,
    deck_files: list[str],
    raw_patches: list[SentaurusPatch | dict[str, Any]],
) -> list[Path]:
    candidates: dict[Path, None] = {}
    for raw in deck_files:
        try:
            path = project_relative_path(project_copy, raw)
        except ValueError:
            continue
        if path.exists() and path.is_file():
            candidates[path.resolve()] = None
    for raw_patch in raw_patches:
        patch = raw_patch if isinstance(raw_patch, SentaurusPatch) else SentaurusPatch.model_validate(raw_patch)
        try:
            path = project_relative_path(project_copy, patch.file)
        except ValueError:
            continue
        if path.exists() and path.is_file():
            candidates[path.resolve()] = None
    if not candidates:
        for pattern in ["*.cmd", "*.des", "*.par"]:
            for path in project_copy.rglob(pattern):
                if path.is_file():
                    candidates[path.resolve()] = None
    return sorted(candidates, key=lambda path: str(path))


def write_sentaurus_deck_ir_artifacts(
    project_copy: Path,
    run_dir: Path,
    *,
    deck_files: list[str],
    patches: list[SentaurusPatch | dict[str, Any]],
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for index, path in enumerate(sentaurus_deck_artifact_candidates(project_copy, deck_files, patches), start=1):
        try:
            ir = parse_sentaurus_deck_file(path)
        except Exception:
            continue
        output_path = run_dir / f"sentaurus_deck_ir_{index:02d}_{path.stem}.json"
        write_json(output_path, ir.model_dump(mode="json"))
        artifacts[f"sentaurus_deck_ir_{path.name}".replace(" ", "_")] = str(output_path)
    return artifacts


def collect_files(root: Path, globs: list[str]) -> list[Path]:
    found: dict[Path, None] = {}
    for pattern in globs:
        for path in root.rglob(pattern):
            if path.is_file():
                found[path.resolve()] = None
    return sorted(found, key=lambda path: str(path))


def read_log_diagnostics(text: str, *, source: str, returncode: int | None = None) -> list[dict[str, Any]]:
    lowered = text.lower()
    diagnostics: list[dict[str, Any]] = []
    patterns = [
        ("sentaurus_license_issue", "error", ["license", "licensed", "checkout", "lmgrd", "snpslmd", "denied"]),
        ("sentaurus_convergence_issue", "error", ["failed to converge", "not converged", "newton failed", "step too small"]),
        ("sentaurus_mesh_issue", "warning", ["bad element", "mesh error", "grid error", "negative volume"]),
        ("sentaurus_fatal_error", "error", ["fatal", "segmentation fault", "traceback"]),
        ("sentaurus_warning", "warning", ["warning"]),
    ]
    for code, severity, tokens in patterns:
        if any(token in lowered for token in tokens):
            diagnostics.append({"code": code, "severity": severity, "source": source})
    if returncode not in {None, 0} and not any(item["severity"] == "error" for item in diagnostics):
        diagnostics.append({"code": "sentaurus_command_failed", "severity": "error", "source": source, "returncode": returncode})
    return diagnostics


def parse_log_files(project_copy: Path, step_results: list[SentaurusStepResult], artifact_globs: list[str]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for step in step_results:
        diagnostics.extend(read_log_diagnostics(step.stdout_tail, source=step.stdout_path, returncode=step.returncode))
        diagnostics.extend(read_log_diagnostics(step.stderr_tail, source=step.stderr_path, returncode=step.returncode))
    for path in collect_files(project_copy, [pattern for pattern in artifact_globs if "log" in pattern.lower()]):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        diagnostics.extend(read_log_diagnostics(tail(text, 8000), source=str(path)))
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for item in diagnostics:
        unique[(str(item.get("code")), str(item.get("source")))] = item
    return list(unique.values())


def numeric_columns(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    keys = list(rows[0])
    output: list[str] = []
    for key in keys:
        values = [finite_float(row.get(key)) for row in rows]
        if any(value is not None for value in values):
            output.append(key)
    return output


def infer_curve_keys(rows: list[dict[str, Any]]) -> tuple[str | None, str | None, str | None]:
    x_key, y_key, field_key = infer_x_y_keys(rows)
    keys = numeric_columns(rows)
    if x_key is None:
        x_candidates = [key for key in keys if any(token in key.lower() for token in ["voltage", "bias", "v("])]
        x_key = x_candidates[0] if x_candidates else (keys[0] if keys else None)
    if y_key is None:
        y_candidates = [key for key in keys if any(token in key.lower() for token in ["current", "i(", "drain", "leak"])]
        y_key = y_candidates[0] if y_candidates else (keys[1] if len(keys) > 1 else None)
    if field_key is None:
        field_candidates = [key for key in keys if "field" in key.lower() or "efield" in key.lower()]
        field_key = field_candidates[0] if field_candidates else None
    return x_key, y_key, field_key


def extract_curve_metrics(curve_path: Path, *, breakdown_threshold: float) -> dict[str, Any]:
    rows = load_curve_rows(curve_path)
    x_key, y_key, field_key = infer_curve_keys(rows)
    shape = curve_shape_diagnostic(rows, x_key=x_key, y_key=y_key, field_key=field_key, threshold_y=breakdown_threshold)
    metrics: dict[str, Any] = {
        "curve_points": len(rows),
        "curve_path": str(curve_path),
        "curve_x_key": x_key,
        "curve_y_key": y_key,
        "curve_field_key": field_key,
        "breakdown_current_threshold_a": breakdown_threshold,
        "curve_shape": shape.model_dump(mode="json"),
    }
    if not rows or not x_key or not y_key:
        return metrics
    pairs: list[tuple[float, float]] = []
    for row in rows:
        x = finite_float(row.get(x_key))
        y = finite_float(row.get(y_key))
        if x is not None and y is not None:
            pairs.append((x, y))
    if not pairs:
        return metrics
    xs = [x for x, _ in pairs]
    ys_abs = [abs(y) for _, y in pairs]
    metrics.update(
        {
            "voltage_min_v": min(xs),
            "voltage_max_v": max(xs),
            "max_abs_current_a": max(ys_abs),
            "min_abs_current_a": min(ys_abs),
            "leakage_abs_current_at_target_a": ys_abs[0],
            "tcad_solver_invoked": True,
            "solver_backend": "sentaurus",
            "fidelity": "external_tcad",
        }
    )
    crossings = [(x, abs_y) for x, abs_y in sorted(zip(xs, ys_abs), key=lambda item: abs(item[0])) if abs_y >= breakdown_threshold]
    if crossings:
        metrics["breakdown_voltage_at_threshold_v"] = crossings[0][0]
        metrics["breakdown_detected"] = True
    else:
        metrics["breakdown_detected"] = False
    if field_key:
        field_points = [
            (finite_float(row.get(x_key)), finite_float(row.get(field_key)))
            for row in rows
        ]
        field_points = [(x, value) for x, value in field_points if x is not None and value is not None]
        if field_points:
            x_peak, field_peak = max(field_points, key=lambda item: abs(item[1]))
            metrics["field_peak_x_v"] = x_peak
            metrics["max_electric_field_v_per_cm"] = abs(field_peak)
    return metrics


def choose_curve_file(project_copy: Path, profile: SentaurusRuntimeProfile) -> Path | None:
    candidates = collect_files(project_copy, profile.curve_globs)
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        priority = 0
        for token in ["extract", "iv", "curve", "des"]:
            if token in name:
                priority -= 1
        return priority, str(path)

    return sorted(candidates, key=score)[0]


def sentaurus_repair_context(goal_text: str, metrics: dict[str, Any], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    codes = {str(item.get("code")) for item in diagnostics}
    candidates: list[dict[str, Any]] = []
    if any("license" in code for code in codes):
        candidates.append(
            {
                "target": "runtime_profile",
                "reason": "license checkout failed; verify local profile env and license server outside the repository",
                "requires_user_confirmation": True,
            }
        )
    if any("convergence" in code for code in codes):
        candidates.extend(
            [
                {"target": "bias_ramp", "reason": "reduce or bracket the bias step around the failed continuation region"},
                {"target": "solver_damping", "reason": "increase damping or staged model activation for Newton convergence"},
                {"target": "mesh_refinement", "reason": "inspect field/current peak region before adding local mesh refinement"},
            ]
        )
    if metrics.get("breakdown_detected") is False and "bv" in goal_text.lower():
        candidates.append({"target": "bias_range", "reason": "extend reverse-bias bracket until threshold crossing or hard stop"})
    if finite_float(metrics.get("max_abs_current_a")) is not None:
        candidates.append(
            {
                "target": "curve_extraction",
                "reason": "compare extracted IV shape before choosing field plate, drift doping, lifetime, or trap-density patches",
            }
        )
    return {
        "schema_version": "actsoft.tcad.sentaurus_repair_context.v1",
        "goal_text": goal_text,
        "candidate_next_actions": candidates,
        "policy": "agent should inspect real logs, extracted curves, and explicit user-approved patch schemas before modifying commercial decks",
    }


def quality_from_sentaurus(returncodes: list[int | None], diagnostics: list[dict[str, Any]], metrics: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    issues = list(diagnostics)
    if any(code not in {0, None} for code in returncodes):
        issues.append({"code": "sentaurus_nonzero_returncode", "severity": "error", "returncodes": returncodes})
    if not metrics.get("curve_path"):
        issues.append({"code": "sentaurus_curve_missing", "severity": "warning"})
    if any(str(item.get("severity")) == "error" for item in issues):
        return "failed", issues
    if issues:
        return "suspicious", issues
    return "passed", issues


def status_from_issue_severity(issues: list[dict[str, Any]]) -> str:
    if any(str(item.get("severity")) == "error" for item in issues):
        return "failed"
    if issues:
        return "suspicious"
    return "passed"


def split_command(raw: str) -> list[str]:
    parts = shlex.split(raw)
    if not parts:
        raise ValueError("remote transport command must not be empty")
    return parts


def validate_env_key(key: str) -> None:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
        raise ValueError(f"unsafe environment key for remote execution: {key}")


def quoted_command(command: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


def remote_env_prefix(env: dict[str, str], *, redacted: bool = False) -> str:
    if not env:
        return ""
    assignments: list[str] = []
    for key, value in sorted(env.items()):
        validate_env_key(key)
        rendered = "<redacted>" if redacted else str(value)
        assignments.append(f"{key}={shlex.quote(rendered)}")
    return "env " + " ".join(assignments) + " "


def remote_config(profile: SentaurusRuntimeProfile) -> SentaurusRemoteExecutionProfile:
    remote = profile.remote
    if not remote.host:
        raise ValueError("remote Sentaurus execution requires remote.host in the external profile")
    if not remote.remote_run_root:
        raise ValueError("remote Sentaurus execution requires remote.remote_run_root in the external profile")
    return remote


def remote_run_paths(profile: SentaurusRuntimeProfile, run_id: str) -> tuple[str, str]:
    remote = remote_config(profile)
    root = str(remote.remote_run_root or "").rstrip("/")
    if not root.startswith("/"):
        raise ValueError("remote.remote_run_root must be an absolute path on the remote host")
    run_dir = posixpath.join(root, run_id)
    return run_dir, posixpath.join(run_dir, "project")


def remote_shell_command(profile: SentaurusRuntimeProfile, shell_command: str, *, redacted: bool = False) -> list[str]:
    remote = remote_config(profile)
    host = "<remote-host>" if redacted else str(remote.host)
    return [*split_command(remote.ssh_command), *remote.ssh_args, host, shell_command]


def remote_rsync_command(
    profile: SentaurusRuntimeProfile,
    source: str,
    destination: str,
    *,
    redacted: bool = False,
) -> list[str]:
    remote = remote_config(profile)
    src = source
    dst = destination
    if redacted:
        if source.startswith(str(remote.host) + ":"):
            src = "<remote-host>:" + source.split(":", 1)[1]
        if destination.startswith(str(remote.host) + ":"):
            dst = "<remote-host>:" + destination.split(":", 1)[1]
    return [*split_command(remote.rsync_command), *remote.rsync_args, src, dst]


def run_logged_command(
    *,
    step: str,
    actual_command: list[str],
    display_command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
    cancel_file: str | None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> SentaurusStepResult:
    try:
        completed = run_cancellable(
            actual_command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            cancel_file=cancel_file,
            env=env,
        )
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        return SentaurusStepResult(
            step=step,
            command=display_command,
            returncode=completed.returncode,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            stdout_tail=tail(completed.stdout),
            stderr_tail=tail(completed.stderr),
        )
    except Exception as exc:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(str(exc), encoding="utf-8")
        return SentaurusStepResult(
            step=step,
            command=display_command,
            returncode=-1,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            stdout_tail="",
            stderr_tail=str(exc),
        )


def sync_remote_project(
    *,
    profile: SentaurusRuntimeProfile,
    project_copy: Path,
    run_dir: Path,
    remote_project_dir: str,
    direction: str,
    timeout_seconds: float,
    cancel_file: str | None,
) -> SentaurusStepResult:
    remote = remote_config(profile)
    local_project = str(project_copy) + "/"
    remote_project = f"{remote.host}:{remote_project_dir.rstrip('/')}/"
    if direction == "push":
        source, destination = local_project, remote_project
    elif direction == "pull":
        source, destination = remote_project, local_project
    else:
        raise ValueError(f"unknown remote sync direction: {direction}")
    stdout_path = run_dir / f"remote_sync_{direction}_stdout.log"
    stderr_path = run_dir / f"remote_sync_{direction}_stderr.log"
    return run_logged_command(
        step=f"remote_sync_{direction}",
        actual_command=remote_rsync_command(profile, source, destination),
        display_command=remote_rsync_command(profile, source, destination, redacted=True),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout_seconds=timeout_seconds,
        cancel_file=cancel_file,
    )


def remote_step_args(index: int, step: str, request: SentaurusRunRequest) -> list[str]:
    args = list(request.command_args.get(step) or [])
    if not args and index == 0 and request.deck_files and step in {"sdevice", "sprocess", "svisual", "inspect"}:
        args = list(request.deck_files)
    return args


def remote_execution_command(
    *,
    profile: SentaurusRuntimeProfile,
    project_dir: str,
    command: list[str],
    redacted: bool = False,
) -> str:
    remote = remote_config(profile)
    setup = " && ".join(remote.remote_setup_commands)
    setup_prefix = f"{setup} && " if setup else ""
    return (
        f"cd {shlex.quote(project_dir)} && "
        f"{setup_prefix}"
        f"{remote_env_prefix(profile.env, redacted=redacted)}{quoted_command(command)}"
    )


def run_remote_ssh_flow(
    *,
    profile: SentaurusRuntimeProfile,
    project_copy: Path,
    run_dir: Path,
    run_id: str,
    flow: list[str],
    request: SentaurusRunRequest,
) -> tuple[list[SentaurusStepResult], str, str | None, dict[str, Any]]:
    remote_run_dir, remote_project_dir = remote_run_paths(profile, run_id)
    commands: list[SentaurusStepResult] = []
    mkdir_command = f"mkdir -p {shlex.quote(remote_project_dir)}"
    mkdir_result = run_logged_command(
        step="remote_prepare",
        actual_command=remote_shell_command(profile, mkdir_command),
        display_command=remote_shell_command(profile, mkdir_command, redacted=True),
        stdout_path=run_dir / "remote_prepare_stdout.log",
        stderr_path=run_dir / "remote_prepare_stderr.log",
        timeout_seconds=request.timeout_seconds,
        cancel_file=request.cancel_file,
    )
    commands.append(mkdir_result)
    if mkdir_result.returncode:
        return commands, "failed", "remote Sentaurus workspace preparation failed", {
            "remote_run_dir_configured": True,
            "remote_project_synced": False,
            "remote_execution_mode": normalized_execution_mode(profile),
        }

    push_result = sync_remote_project(
        profile=profile,
        project_copy=project_copy,
        run_dir=run_dir,
        remote_project_dir=remote_project_dir,
        direction="push",
        timeout_seconds=request.timeout_seconds,
        cancel_file=request.cancel_file,
    )
    commands.append(push_result)
    if push_result.returncode:
        return commands, "failed", "remote Sentaurus project upload failed", {
            "remote_run_dir_configured": True,
            "remote_project_synced": False,
            "remote_execution_mode": normalized_execution_mode(profile),
        }

    status = "completed"
    failure_reason = None
    for index, step in enumerate(flow):
        args = remote_step_args(index, step, request)
        step_command = profile.command_vector(step, args)
        actual_shell = remote_execution_command(profile=profile, project_dir=remote_project_dir, command=step_command)
        display_shell = remote_execution_command(profile=profile, project_dir=remote_project_dir, command=step_command, redacted=True)
        stdout_path = run_dir / f"{index + 1:02d}_{step}_stdout.log"
        stderr_path = run_dir / f"{index + 1:02d}_{step}_stderr.log"
        step_result = run_logged_command(
            step=step,
            actual_command=remote_shell_command(profile, actual_shell),
            display_command=remote_shell_command(profile, display_shell, redacted=True),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=request.timeout_seconds,
            cancel_file=request.cancel_file,
        )
        commands.append(step_result)
        pull_result = sync_remote_project(
            profile=profile,
            project_copy=project_copy,
            run_dir=run_dir,
            remote_project_dir=remote_project_dir,
            direction="pull",
            timeout_seconds=request.timeout_seconds,
            cancel_file=request.cancel_file,
        )
        commands.append(pull_result)
        if step_result.returncode:
            status = "failed"
            failure_reason = f"Remote Sentaurus step `{step}` exited with return code {step_result.returncode}"
            break
        if pull_result.returncode:
            status = "failed"
            failure_reason = "remote Sentaurus project download failed"
            break
    return commands, status, failure_reason, {
        "remote_run_dir_configured": True,
        "remote_project_synced": True,
        "remote_execution_mode": normalized_execution_mode(profile),
        "remote_flow_steps": len(flow),
    }


def slurm_script_name(index: int, step: str) -> str:
    safe_step = re.sub(r"[^A-Za-z0-9_.-]+", "_", step).strip("_") or "step"
    return f"{index + 1:02d}_{safe_step}.sh"


def write_slurm_script(
    *,
    profile: SentaurusRuntimeProfile,
    project_copy: Path,
    remote_project_dir: str,
    index: int,
    step: str,
    command: list[str],
) -> Path:
    remote = remote_config(profile)
    actsoft_dir = project_copy / ".actsoft"
    actsoft_dir.mkdir(parents=True, exist_ok=True)
    script_path = actsoft_dir / slurm_script_name(index, step)
    stdout_name = f"{index + 1:02d}_{step}_stdout.log"
    stderr_name = f"{index + 1:02d}_{step}_stderr.log"
    returncode_name = f"{index + 1:02d}_{step}_returncode.txt"
    setup_lines = "\n".join(remote.remote_setup_commands)
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set +e",
                f"cd {shlex.quote(remote_project_dir)}",
                "mkdir -p .actsoft",
                setup_lines,
                f"{quoted_command(command)} > .actsoft/{shlex.quote(stdout_name)} 2> .actsoft/{shlex.quote(stderr_name)}",
                "rc=$?",
                f"printf '%s\\n' \"$rc\" > .actsoft/{shlex.quote(returncode_name)}",
                "exit \"$rc\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o755)
    return script_path


def parse_slurm_job_id(stdout: str) -> str | None:
    first = (stdout or "").strip().splitlines()[0:1]
    if not first:
        return None
    token = first[0].strip().split(";", 1)[0].strip()
    return token if re.match(r"^[A-Za-z0-9_.-]+$", token) else None


def slurm_poll_until_done(
    *,
    profile: SentaurusRuntimeProfile,
    job_id: str,
    remote_project_dir: str,
    timeout_seconds: float,
    cancel_file: str | None,
    run_dir: Path,
    index: int,
    step: str,
) -> SentaurusStepResult | None:
    remote = remote_config(profile)
    start = time.monotonic()
    while True:
        if cancel_file and Path(cancel_file).exists():
            cancel_shell = f"cd {shlex.quote(remote_project_dir)} && {shlex.quote(remote.slurm_cancel_command)} {shlex.quote(job_id)}"
            run_logged_command(
                step=f"{step}_slurm_cancel",
                actual_command=remote_shell_command(profile, cancel_shell),
                display_command=remote_shell_command(profile, cancel_shell, redacted=True),
                stdout_path=run_dir / f"{index + 1:02d}_{step}_scancel_stdout.log",
                stderr_path=run_dir / f"{index + 1:02d}_{step}_scancel_stderr.log",
                timeout_seconds=min(timeout_seconds, 60),
                cancel_file=None,
            )
            return SentaurusStepResult(
                step=step,
                command=remote_shell_command(profile, cancel_shell, redacted=True),
                returncode=-15,
                stdout_path=str(run_dir / f"{index + 1:02d}_{step}_stdout.log"),
                stderr_path=str(run_dir / f"{index + 1:02d}_{step}_stderr.log"),
                stdout_tail="",
                stderr_tail=f"ACTSOFT_CANCELLED: Slurm job {job_id} cancelled after local cancel token",
            )
        if time.monotonic() - start >= timeout_seconds:
            cancel_shell = f"cd {shlex.quote(remote_project_dir)} && {shlex.quote(remote.slurm_cancel_command)} {shlex.quote(job_id)}"
            run_logged_command(
                step=f"{step}_slurm_timeout_cancel",
                actual_command=remote_shell_command(profile, cancel_shell),
                display_command=remote_shell_command(profile, cancel_shell, redacted=True),
                stdout_path=run_dir / f"{index + 1:02d}_{step}_timeout_scancel_stdout.log",
                stderr_path=run_dir / f"{index + 1:02d}_{step}_timeout_scancel_stderr.log",
                timeout_seconds=60,
                cancel_file=None,
            )
            return SentaurusStepResult(
                step=step,
                command=remote_shell_command(profile, cancel_shell, redacted=True),
                returncode=-1,
                stdout_path=str(run_dir / f"{index + 1:02d}_{step}_stdout.log"),
                stderr_path=str(run_dir / f"{index + 1:02d}_{step}_stderr.log"),
                stdout_tail="",
                stderr_tail=f"Slurm job {job_id} exceeded timeout_seconds={timeout_seconds}",
            )
        status_shell = f"cd {shlex.quote(remote_project_dir)} && {shlex.quote(remote.slurm_status_command)} -h -j {shlex.quote(job_id)}"
        status_result = run_logged_command(
            step=f"{step}_slurm_poll",
            actual_command=remote_shell_command(profile, status_shell),
            display_command=remote_shell_command(profile, status_shell, redacted=True),
            stdout_path=run_dir / f"{index + 1:02d}_{step}_squeue_stdout.log",
            stderr_path=run_dir / f"{index + 1:02d}_{step}_squeue_stderr.log",
            timeout_seconds=min(timeout_seconds, 60),
            cancel_file=None,
        )
        if status_result.returncode not in {0, None}:
            return status_result.model_copy(update={"step": step})
        if not status_result.stdout_tail.strip():
            return None
        time.sleep(remote.slurm_poll_interval_seconds)


def read_slurm_step_result(
    *,
    profile: SentaurusRuntimeProfile,
    project_copy: Path,
    run_dir: Path,
    remote_project_dir: str,
    index: int,
    step: str,
    job_id: str,
) -> SentaurusStepResult:
    stdout_name = f"{index + 1:02d}_{step}_stdout.log"
    stderr_name = f"{index + 1:02d}_{step}_stderr.log"
    returncode_name = f"{index + 1:02d}_{step}_returncode.txt"
    local_stdout = project_copy / ".actsoft" / stdout_name
    local_stderr = project_copy / ".actsoft" / stderr_name
    local_returncode = project_copy / ".actsoft" / returncode_name
    stdout_path = run_dir / stdout_name
    stderr_path = run_dir / stderr_name
    stdout_text = local_stdout.read_text(encoding="utf-8", errors="replace") if local_stdout.exists() else ""
    stderr_text = local_stderr.read_text(encoding="utf-8", errors="replace") if local_stderr.exists() else ""
    if local_returncode.exists():
        try:
            returncode = int(local_returncode.read_text(encoding="utf-8").strip())
        except ValueError:
            returncode = -1
    else:
        returncode = -1
        stderr_text = (stderr_text + "\n" if stderr_text else "") + "missing Slurm returncode artifact"
    stdout_path.write_text(stdout_text, encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")
    remote = remote_config(profile)
    submit_shell = (
        f"cd {shlex.quote(remote_project_dir)} && "
        f"{remote_env_prefix(profile.env, redacted=True)}"
        f"{quoted_command([remote.slurm_submit_command, *remote.slurm_submit_args, '.actsoft/' + slurm_script_name(index, step)])}"
    )
    return SentaurusStepResult(
        step=step,
        command=remote_shell_command(profile, submit_shell, redacted=True),
        returncode=returncode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        stdout_tail=tail(stdout_text),
        stderr_tail=tail(stderr_text),
    )


def run_remote_slurm_flow(
    *,
    profile: SentaurusRuntimeProfile,
    project_copy: Path,
    run_dir: Path,
    run_id: str,
    flow: list[str],
    request: SentaurusRunRequest,
) -> tuple[list[SentaurusStepResult], str, str | None, dict[str, Any]]:
    remote = remote_config(profile)
    remote_run_dir, remote_project_dir = remote_run_paths(profile, run_id)
    commands: list[SentaurusStepResult] = []
    for index, step in enumerate(flow):
        args = remote_step_args(index, step, request)
        write_slurm_script(
            profile=profile,
            project_copy=project_copy,
            remote_project_dir=remote_project_dir,
            index=index,
            step=step,
            command=profile.command_vector(step, args),
        )
    mkdir_command = f"mkdir -p {shlex.quote(remote_project_dir)}"
    mkdir_result = run_logged_command(
        step="remote_prepare",
        actual_command=remote_shell_command(profile, mkdir_command),
        display_command=remote_shell_command(profile, mkdir_command, redacted=True),
        stdout_path=run_dir / "remote_prepare_stdout.log",
        stderr_path=run_dir / "remote_prepare_stderr.log",
        timeout_seconds=request.timeout_seconds,
        cancel_file=request.cancel_file,
    )
    commands.append(mkdir_result)
    if mkdir_result.returncode:
        return commands, "failed", "remote Sentaurus workspace preparation failed", {
            "remote_run_dir_configured": True,
            "remote_project_synced": False,
            "remote_execution_mode": normalized_execution_mode(profile),
            "remote_scheduler": "slurm",
        }
    push_result = sync_remote_project(
        profile=profile,
        project_copy=project_copy,
        run_dir=run_dir,
        remote_project_dir=remote_project_dir,
        direction="push",
        timeout_seconds=request.timeout_seconds,
        cancel_file=request.cancel_file,
    )
    commands.append(push_result)
    if push_result.returncode:
        return commands, "failed", "remote Sentaurus project upload failed", {
            "remote_run_dir_configured": True,
            "remote_project_synced": False,
            "remote_execution_mode": normalized_execution_mode(profile),
            "remote_scheduler": "slurm",
        }

    status = "completed"
    failure_reason = None
    for index, step in enumerate(flow):
        script = ".actsoft/" + slurm_script_name(index, step)
        submit_shell = (
            f"cd {shlex.quote(remote_project_dir)} && "
            f"{remote_env_prefix(profile.env)}"
            f"{quoted_command([remote.slurm_submit_command, *remote.slurm_submit_args, script])}"
        )
        display_shell = (
            f"cd {shlex.quote(remote_project_dir)} && "
            f"{remote_env_prefix(profile.env, redacted=True)}"
            f"{quoted_command([remote.slurm_submit_command, *remote.slurm_submit_args, script])}"
        )
        submit_result = run_logged_command(
            step=f"{step}_slurm_submit",
            actual_command=remote_shell_command(profile, submit_shell),
            display_command=remote_shell_command(profile, display_shell, redacted=True),
            stdout_path=run_dir / f"{index + 1:02d}_{step}_sbatch_stdout.log",
            stderr_path=run_dir / f"{index + 1:02d}_{step}_sbatch_stderr.log",
            timeout_seconds=request.timeout_seconds,
            cancel_file=request.cancel_file,
        )
        commands.append(submit_result)
        job_id = parse_slurm_job_id(submit_result.stdout_tail)
        if submit_result.returncode or not job_id:
            status = "failed"
            failure_reason = f"Slurm submit for `{step}` failed"
            break
        poll_failure = slurm_poll_until_done(
            profile=profile,
            job_id=job_id,
            remote_project_dir=remote_project_dir,
            timeout_seconds=request.timeout_seconds,
            cancel_file=request.cancel_file,
            run_dir=run_dir,
            index=index,
            step=step,
        )
        pull_result = sync_remote_project(
            profile=profile,
            project_copy=project_copy,
            run_dir=run_dir,
            remote_project_dir=remote_project_dir,
            direction="pull",
            timeout_seconds=request.timeout_seconds,
            cancel_file=request.cancel_file,
        )
        commands.append(pull_result)
        if poll_failure:
            commands.append(poll_failure)
            status = "failed"
            failure_reason = poll_failure.stderr_tail or f"Slurm poll for `{step}` failed"
            break
        step_result = read_slurm_step_result(
            profile=profile,
            project_copy=project_copy,
            run_dir=run_dir,
            remote_project_dir=remote_project_dir,
            index=index,
            step=step,
            job_id=job_id,
        )
        commands.append(step_result)
        if pull_result.returncode:
            status = "failed"
            failure_reason = "remote Sentaurus project download failed"
            break
        if step_result.returncode:
            status = "failed"
            failure_reason = f"Remote Slurm Sentaurus step `{step}` exited with return code {step_result.returncode}"
            break
    return commands, status, failure_reason, {
        "remote_run_dir_configured": True,
        "remote_project_synced": True,
        "remote_execution_mode": normalized_execution_mode(profile),
        "remote_scheduler": "slurm",
        "remote_flow_steps": len(flow),
    }


def run_remote_sentaurus_flow(
    *,
    profile: SentaurusRuntimeProfile,
    project_copy: Path,
    run_dir: Path,
    run_id: str,
    flow: list[str],
    request: SentaurusRunRequest,
) -> tuple[list[SentaurusStepResult], str, str | None, dict[str, Any]]:
    mode = normalized_execution_mode(profile)
    if mode == "remote_ssh":
        return run_remote_ssh_flow(
            profile=profile,
            project_copy=project_copy,
            run_dir=run_dir,
            run_id=run_id,
            flow=flow,
            request=request,
        )
    if mode == "remote_slurm":
        return run_remote_slurm_flow(
            profile=profile,
            project_copy=project_copy,
            run_dir=run_dir,
            run_id=run_id,
            flow=flow,
            request=request,
        )
    raise ValueError(f"unsupported Sentaurus execution_mode: {profile.execution_mode}")


def sentaurus_flow_step_count(commands: list[SentaurusStepResult], flow: list[str]) -> int:
    flow_names = set(flow)
    return sum(1 for step in commands if step.step in flow_names)


def run_sentaurus(request: SentaurusRunRequest) -> SentaurusRunState:
    profile = load_runtime_profile(request)
    project_path = ensure_allowed_project(profile, request.project_path)
    run_id = request.run_id or safe_run_id()
    run_dir = (profile.run_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    state_path = run_dir / "sentaurus_state.json"
    started_at = utc_timestamp()
    project_copy = clone_project(project_path, run_dir)
    flow = request.flow or profile.default_flow
    patches = apply_sentaurus_patches(project_copy, request.patches, run_dir) if request.patches else []
    deck_ir_artifacts = write_sentaurus_deck_ir_artifacts(
        project_copy,
        run_dir,
        deck_files=request.deck_files,
        patches=request.patches,
    )
    commands: list[SentaurusStepResult] = []
    env = {**os.environ, **profile.env}
    status = "completed"
    failure_reason = None
    remote_metrics: dict[str, Any] = {
        "remote_execution": is_remote_execution(profile),
        "remote_execution_mode": normalized_execution_mode(profile),
    }
    if request.execute:
        if is_remote_execution(profile):
            commands, status, failure_reason, remote_metrics = run_remote_sentaurus_flow(
                profile=profile,
                project_copy=project_copy,
                run_dir=run_dir,
                run_id=run_id,
                flow=flow,
                request=request,
            )
        else:
            for index, step in enumerate(flow):
                args = list(request.command_args.get(step) or [])
                if not args and index == 0 and request.deck_files and step in {"sdevice", "sprocess", "svisual", "inspect"}:
                    args = list(request.deck_files)
                command = profile.command_vector(step, args)
                stdout_path = run_dir / f"{index + 1:02d}_{step}_stdout.log"
                stderr_path = run_dir / f"{index + 1:02d}_{step}_stderr.log"
                try:
                    completed = run_cancellable(
                        command,
                        cwd=project_copy,
                        capture_output=True,
                        text=True,
                        timeout=request.timeout_seconds,
                        check=False,
                        cancel_file=request.cancel_file,
                        env=env,
                    )
                    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
                    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
                    step_result = SentaurusStepResult(
                        step=step,
                        command=command,
                        returncode=completed.returncode,
                        stdout_path=str(stdout_path),
                        stderr_path=str(stderr_path),
                        stdout_tail=tail(completed.stdout),
                        stderr_tail=tail(completed.stderr),
                    )
                    commands.append(step_result)
                    if completed.returncode:
                        status = "failed"
                        failure_reason = f"Sentaurus step `{step}` exited with return code {completed.returncode}"
                        break
                except Exception as exc:
                    stdout_path.write_text("", encoding="utf-8")
                    stderr_path.write_text(str(exc), encoding="utf-8")
                    commands.append(
                        SentaurusStepResult(
                            step=step,
                            command=command,
                            returncode=-1,
                            stdout_path=str(stdout_path),
                            stderr_path=str(stderr_path),
                            stdout_tail="",
                            stderr_tail=str(exc),
                        )
                    )
                    status = "failed"
                    failure_reason = str(exc)
                    break
    else:
        status = "planned"
    remote_metrics.setdefault("remote_execution", is_remote_execution(profile))
    remote_metrics.setdefault("remote_execution_mode", normalized_execution_mode(profile))

    artifacts: dict[str, str] = {
        "project_copy": str(project_copy),
        **deck_ir_artifacts,
    }
    patch_diff = run_dir / "sentaurus_patch.diff"
    if patch_diff.exists():
        artifacts["sentaurus_patch_diff"] = str(patch_diff)
    for step in commands:
        artifacts[f"{step.step}_stdout"] = step.stdout_path
        artifacts[f"{step.step}_stderr"] = step.stderr_path
    for path in collect_files(project_copy, profile.artifact_globs):
        key = f"sentaurus_artifact_{path.name}".replace(" ", "_")
        artifacts.setdefault(key, str(path))
    if request.reference_curve_path:
        artifacts["reference_curve"] = str(request.reference_curve_path.expanduser())

    curve_path = choose_curve_file(project_copy, profile)
    sentaurus_steps = sentaurus_flow_step_count(commands, flow) if is_remote_execution(profile) else len(commands)
    metrics: dict[str, Any] = {
        "sentaurus_steps": sentaurus_steps,
        "sentaurus_project_copied": True,
        "sentaurus_patches_requested": len(request.patches),
        "sentaurus_patches_applied": sum(1 for patch in patches if patch.get("applied")),
        "sentaurus_patches_verified": sum(1 for patch in patches if patch.get("verified")),
        "sentaurus_deck_ir_files": len(deck_ir_artifacts),
        "tcad_solver_invoked": bool(request.execute and sentaurus_steps),
        "solver_backend": "sentaurus",
        "fidelity": "external_tcad",
        "reference_curve_provided": bool(request.reference_curve_path),
        **remote_metrics,
    }
    if curve_path:
        artifacts["sentaurus_curve_csv"] = str(curve_path)
        metrics.update(extract_curve_metrics(curve_path, breakdown_threshold=request.breakdown_current_threshold_a))
    log_diagnostics = parse_log_files(project_copy, commands, profile.artifact_globs)
    returncodes = [step.returncode for step in commands]
    quality_status, issues = quality_from_sentaurus(returncodes, log_diagnostics, metrics)
    for patch in patches:
        if patch.get("verified"):
            continue
        issues.append(
            {
                "code": "sentaurus_patch_unverified",
                "severity": "error" if patch.get("required", True) else "warning",
                "patch": {key: patch.get(key) for key in ["file", "operation", "reason", "error"]},
            }
        )
    quality_status = status_from_issue_severity(issues)
    if status == "planned":
        quality_status = "suspicious"
        issues.append({"code": "sentaurus_execution_planned_only", "severity": "warning"})
    if status == "completed" and quality_status == "failed":
        status = "failed"

    state = SentaurusRunState(
        status=status,
        run_id=run_id,
        run_dir=str(run_dir),
        project_path=str(project_path),
        project_copy_path=str(project_copy),
        started_at=started_at,
        completed_at=utc_timestamp(),
        request=request.model_dump(mode="json", exclude={"profile"}),
        runtime_profile=profile.safe_summary(),
        commands=[step.model_dump(mode="json") for step in commands],
        patches=patches,
        artifacts=artifacts,
        log_diagnostics=log_diagnostics,
        final_summary={
            "artifacts": artifacts,
            "metrics": metrics,
            "parameters": {
                "flow": flow,
                "deck_files": request.deck_files,
                "breakdown_current_threshold_a": request.breakdown_current_threshold_a,
            },
            "data_provenance": {
                "real_sentaurus_required_for_physics": True,
                "fake_commands_only_validate_agent_io": fake_interface_only(profile),
                "public_interface_notes": [
                    "Sentaurus Device runs command files and writes log/plot/device outputs.",
                    "Sentaurus Visual/Inspect-style extraction should be configured locally to export CSV for this agent.",
                ],
            },
        },
        quality_report={
            "status": quality_status,
            "issues": issues,
            "metrics": metrics,
        },
        repair_context=sentaurus_repair_context(request.goal_text, metrics, issues),
        next_action="inspect Sentaurus logs and extracted curves" if status != "completed" else "benchmark Sentaurus state and plan next patch",
        failure_reason=failure_reason,
        state_path=str(state_path),
    )
    write_json(state_path, state.model_dump(mode="json"))
    return state
