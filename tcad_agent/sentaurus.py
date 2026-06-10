from __future__ import annotations

import difflib
import json
import os
import re
import shlex
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.curve_diagnostics import curve_shape_diagnostic, finite_float, infer_x_y_keys, load_curve_rows
from tcad_agent.process_control import run_cancellable
from tcad_agent.task_spec import PROJECT_ROOT


class SentaurusRuntimeProfile(BaseModel):
    """Local-only Sentaurus execution profile.

    The profile intentionally keeps commercial installation paths, license
    variables, PDK roots, and command overrides outside the repository. Tests
    can provide fake commands through this same interface without pretending to
    generate real Synopsys physics results.
    """

    profile_id: str = "local"
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
            "sentaurus_home_configured": self.sentaurus_home is not None,
            "commands": sorted(self.commands),
            "allowed_project_roots": [str(path.expanduser()) for path in self.allowed_project_roots],
            "default_flow": self.default_flow,
            "curve_globs": self.curve_globs,
            "artifact_globs": self.artifact_globs,
            "env_keys": sorted(self.env),
        }


class SentaurusPatch(BaseModel):
    file: str
    operation: str = "replace_text"
    pattern: str | None = None
    replacement: str | None = None
    regex: bool = False
    json_path: str | None = None
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
        if patch.operation == "json_set" or patch.json_path:
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
    if after == before:
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
    commands: list[SentaurusStepResult] = []
    env = {**os.environ, **profile.env}
    status = "completed"
    failure_reason = None
    if request.execute:
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

    artifacts: dict[str, str] = {
        "project_copy": str(project_copy),
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
    metrics: dict[str, Any] = {
        "sentaurus_steps": len(commands),
        "sentaurus_project_copied": True,
        "sentaurus_patches_requested": len(request.patches),
        "sentaurus_patches_applied": sum(1 for patch in patches if patch.get("applied")),
        "sentaurus_patches_verified": sum(1 for patch in patches if patch.get("verified")),
        "tcad_solver_invoked": bool(request.execute and commands),
        "solver_backend": "sentaurus",
        "fidelity": "external_tcad",
        "reference_curve_provided": bool(request.reference_curve_path),
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
                "fake_commands_only_validate_agent_io": True,
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
