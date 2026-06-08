from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from tcad_agent.deck_writer import write_deck_artifacts
from tcad_agent.process_control import run_cancellable
from tcad_agent.physical_quality import check_mos_capacitor_physics


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ToolStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class FailureClass(str, Enum):
    NONE = "none"
    VALIDATION = "validation"
    CONVERGENCE = "convergence"
    TIMEOUT = "timeout"
    OUTPUT_MISSING = "output_missing"
    RUNNER_ERROR = "runner_error"
    UNKNOWN = "unknown"


class MOSCapacitorCVRequest(BaseModel):
    start: float = -1.0
    stop: float = 1.0
    step: float = Field(default=0.25, gt=0.0)
    min_step: float = Field(default=0.0625, gt=0.0)
    max_attempts: int = Field(default=3, ge=1)
    timeout_seconds: float = Field(default=300.0, gt=0.0)
    quality_min_points: int = Field(default=3, ge=1)
    oxide_thickness_nm: float = Field(default=5.0, gt=0.0)
    silicon_thickness_um: float = Field(default=0.2, gt=0.0)
    substrate_doping_cm3: float = Field(default=1.0e17, gt=0.0)
    temperature_k: float = Field(default=300.0, gt=0.0)
    oxide_spacing_nm: float = Field(default=0.25, gt=0.0)
    silicon_spacing_um: float = Field(default=0.002, gt=0.0)
    fixed_oxide_charge_cm2: float = Field(default=0.0, ge=0.0)
    run_id: str | None = None
    run_root: Path = PROJECT_ROOT / "runs" / "agent_tools"
    resume: bool = False
    tcad_deck_spec: dict[str, Any] | None = None
    tcad_deck_mutations: list[dict[str, Any]] = Field(default_factory=list)
    deck_patch_history: list[dict[str, Any]] = Field(default_factory=list)
    source_deck_path: str | None = None
    repair_source_state_path: str | None = None
    repair_baseline_state_path: str | None = None

    @model_validator(mode="after")
    def validate_request(self) -> "MOSCapacitorCVRequest":
        if self.min_step > self.step:
            raise ValueError("min_step must be less than or equal to step")
        return self


class AttemptRecord(BaseModel):
    index: int
    status: ToolStatus
    step_v: float
    started_at: str
    completed_at: str | None = None
    command: list[str]
    returncode: int | None = None
    run_dir: str | None = None
    summary_path: str | None = None
    failure_class: FailureClass = FailureClass.NONE
    failure_reason: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""


class RunState(BaseModel):
    tool_name: str = "mos_capacitor_cv_sweep"
    status: ToolStatus
    run_id: str
    run_dir: str
    request: dict[str, Any]
    created_at: str
    updated_at: str
    attempts: list[AttemptRecord] = Field(default_factory=list)
    next_action: str | None = None
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    final_summary: dict[str, Any] | None = None
    quality_report: dict[str, Any] | None = None
    tcad_deck_spec: dict[str, Any] | None = None
    tcad_deck_mutations: list[dict[str, Any]] = Field(default_factory=list)


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def tail(text: str, limit: int = 4000) -> str:
    return text[-limit:] if len(text) > limit else text


def write_state(state: RunState, path: Path) -> None:
    state.updated_at = utc_timestamp()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")


def load_state(path: Path) -> RunState:
    return RunState.model_validate_json(path.read_text(encoding="utf-8"))


def classify_failure(returncode: int | None, stdout: str, stderr: str) -> tuple[FailureClass, str]:
    combined = f"{stdout}\n{stderr}".lower()
    if returncode == 0:
        return FailureClass.NONE, ""
    if "converg" in combined or "maximum_iterations" in combined:
        return FailureClass.CONVERGENCE, "DEVSIM solver did not converge."
    if "timeout" in combined or "timed out" in combined:
        return FailureClass.TIMEOUT, "Runner timed out."
    if "valueerror" in combined or "validation" in combined or "argument" in combined:
        return FailureClass.VALIDATION, "Runner rejected the provided arguments."
    if "traceback" in combined or "runtimeerror" in combined or "exception" in combined:
        return FailureClass.RUNNER_ERROR, "Runner raised an exception."
    return FailureClass.UNKNOWN, "Runner failed for an unclassified reason."


def parse_runner_stdout(stdout: str) -> dict[str, Any] | None:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(stdout[start : end + 1])
    except json.JSONDecodeError:
        return None


def build_runner_command(request: MOSCapacitorCVRequest, attempt_index: int, step: float, run_dir: Path) -> list[str]:
    attempt_root = run_dir / "attempt_runs"
    attempt_id = f"attempt_{attempt_index:03d}"
    return [
        sys.executable,
        "-m",
        "tcad_agent.examples.mos_capacitor.run",
        "--start",
        str(request.start),
        "--stop",
        str(request.stop),
        "--step",
        str(step),
        "--run-id",
        attempt_id,
        "--run-root",
        str(attempt_root),
        "--oxide-thickness-nm",
        str(request.oxide_thickness_nm),
        "--silicon-thickness-um",
        str(request.silicon_thickness_um),
        "--substrate-doping-cm3",
        str(request.substrate_doping_cm3),
        "--temperature-k",
        str(request.temperature_k),
        "--oxide-spacing-nm",
        str(request.oxide_spacing_nm),
        "--silicon-spacing-um",
        str(request.silicon_spacing_um),
        "--fixed-oxide-charge-cm2",
        str(request.fixed_oxide_charge_cm2),
    ]


def create_initial_state(request: MOSCapacitorCVRequest, run_id: str, run_dir: Path) -> RunState:
    now = utc_timestamp()
    return RunState(
        status=ToolStatus.RUNNING,
        run_id=run_id,
        run_dir=str(run_dir),
        request=request.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        tcad_deck_spec=request.tcad_deck_spec,
        tcad_deck_mutations=request.tcad_deck_mutations,
        next_action="start first DEVSIM MOS capacitor attempt",
        checkpoint={"current_step_v": request.step, "completed_attempts": 0},
    )


def prepare_state(request: MOSCapacitorCVRequest) -> tuple[RunState, Path]:
    run_id = request.run_id or default_run_id()
    run_dir = request.run_root / "mos_capacitor_cv" / run_id
    state_path = run_dir / "state.json"
    if request.resume:
        if not state_path.exists():
            raise FileNotFoundError(f"Cannot resume; state file does not exist: {state_path}")
        return load_state(state_path), state_path
    run_dir.mkdir(parents=True, exist_ok=False)
    state = create_initial_state(request, run_id, run_dir)
    write_state(state, state_path)
    return state, state_path


def latest_retry_step(state: RunState, request: MOSCapacitorCVRequest) -> float:
    if not state.attempts:
        return request.step
    last = state.attempts[-1]
    if last.status == ToolStatus.FAILED and last.failure_class == FailureClass.CONVERGENCE:
        return max(last.step_v / 2.0, request.min_step)
    return last.step_v


def should_retry_with_smaller_step(attempt: AttemptRecord, request: MOSCapacitorCVRequest, next_attempt: int) -> bool:
    if attempt.failure_class != FailureClass.CONVERGENCE:
        return False
    if next_attempt > request.max_attempts:
        return False
    return attempt.step_v / 2.0 >= request.min_step


def judge_summary_quality(summary: dict[str, Any], request: MOSCapacitorCVRequest) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    artifacts = summary.get("artifacts") or {}
    for name in ["csv", "plot", "tecplot", "log"]:
        path = artifacts.get(name)
        if not path or not Path(path).exists():
            issues.append({"code": f"missing_{name}", "severity": "error" if name == "csv" else "warning", "path": path})

    metrics = {
        "points": summary.get("points"),
        "voltage_range_v": summary.get("voltage_range_v"),
        "min_gate_charge_c_per_cm2": summary.get("min_gate_charge_c_per_cm2"),
        "max_gate_charge_c_per_cm2": summary.get("max_gate_charge_c_per_cm2"),
        "min_capacitance_f_per_cm2": summary.get("min_capacitance_f_per_cm2"),
        "max_capacitance_f_per_cm2": summary.get("max_capacitance_f_per_cm2"),
        "final_capacitance_f_per_cm2": summary.get("final_capacitance_f_per_cm2"),
        "fixed_charge_voltage_shift_v": summary.get("fixed_charge_voltage_shift_v"),
    }
    if (summary.get("points") or 0) < request.quality_min_points:
        issues.append({"code": "too_few_points", "severity": "warning", "points": summary.get("points")})
    for key in ["min_capacitance_f_per_cm2", "max_capacitance_f_per_cm2", "final_capacitance_f_per_cm2"]:
        value = summary.get(key)
        if value is not None and not math.isfinite(float(value)):
            issues.append({"code": "nonfinite_capacitance", "severity": "error", "metric": key, "value": value})
    if summary.get("max_capacitance_f_per_cm2") is None:
        issues.append({"code": "missing_capacitance", "severity": "error"})
    issues.extend(check_mos_capacitor_physics(metrics, summary.get("parameters") or {}))

    status = "passed"
    if any(issue["severity"] == "error" for issue in issues):
        status = "failed"
    elif issues:
        status = "suspicious"
    return {
        "status": status,
        "issues": issues,
        "metrics": metrics,
        "recommended_next_action": "accept MOS capacitor C-V artifacts" if status == "passed" else "inspect MOS capacitor artifacts before using the result",
    }


def run_attempt(
    request: MOSCapacitorCVRequest,
    state: RunState,
    state_path: Path,
    attempt_index: int,
    step: float,
) -> AttemptRecord:
    command = build_runner_command(request, attempt_index, step, Path(state.run_dir))
    attempt = AttemptRecord(
        index=attempt_index,
        status=ToolStatus.RUNNING,
        step_v=step,
        started_at=utc_timestamp(),
        command=command,
    )
    state.attempts.append(attempt)
    state.next_action = f"run MOS capacitor C-V attempt {attempt_index} with step {step:g} V"
    state.checkpoint = {"current_step_v": step, "completed_attempts": attempt_index - 1}
    write_state(state, state_path)

    try:
        completed = run_cancellable(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=request.timeout_seconds,
            check=False,
        )
        attempt.returncode = completed.returncode
        attempt.stdout_tail = tail(completed.stdout)
        attempt.stderr_tail = tail(completed.stderr)
        runner_result = parse_runner_stdout(completed.stdout)
        if completed.returncode == 0 and runner_result:
            attempt.run_dir = runner_result.get("run_dir")
            summary_path = Path(attempt.run_dir) / "summary.json" if attempt.run_dir else None
            if summary_path and summary_path.exists():
                attempt.summary_path = str(summary_path)
                attempt.status = ToolStatus.COMPLETED
                attempt.failure_class = FailureClass.NONE
                attempt.failure_reason = None
            else:
                attempt.status = ToolStatus.FAILED
                attempt.failure_class = FailureClass.OUTPUT_MISSING
                attempt.failure_reason = "Runner completed but summary.json was not found."
        else:
            failure_class, reason = classify_failure(completed.returncode, completed.stdout, completed.stderr)
            attempt.status = ToolStatus.FAILED
            attempt.failure_class = failure_class
            attempt.failure_reason = reason
    except subprocess.TimeoutExpired as exc:
        attempt.returncode = None
        attempt.stdout_tail = tail(exc.stdout or "")
        attempt.stderr_tail = tail(exc.stderr or "")
        attempt.status = ToolStatus.FAILED
        attempt.failure_class = FailureClass.TIMEOUT
        attempt.failure_reason = f"Runner exceeded {request.timeout_seconds:g} seconds."

    attempt.completed_at = utc_timestamp()
    write_state(state, state_path)
    return attempt


def run_mos_capacitor_cv_sweep(request: MOSCapacitorCVRequest) -> dict[str, Any]:
    state, state_path = prepare_state(request)
    if state.status == ToolStatus.COMPLETED:
        return state.model_dump(mode="json")

    next_index = len(state.attempts) + 1
    step = latest_retry_step(state, request)
    while next_index <= request.max_attempts:
        attempt = run_attempt(request, state, state_path, next_index, step)
        if attempt.status == ToolStatus.COMPLETED:
            summary = json.loads(Path(attempt.summary_path).read_text(encoding="utf-8"))
            quality_report = judge_summary_quality(summary, request)
            deck_artifacts = write_deck_artifacts(
                Path(state.run_dir),
                tool_name="mos_capacitor_cv_sweep",
                request=request.model_dump(mode="json"),
                deck_spec=request.tcad_deck_spec,
                mutations=request.tcad_deck_mutations,
                source_goal_text=(request.tcad_deck_spec or {}).get("source_goal_text") if request.tcad_deck_spec else None,
            )
            summary.setdefault("artifacts", {}).update(deck_artifacts)
            summary["tcad_deck_spec"] = request.tcad_deck_spec
            summary["tcad_deck_mutations"] = request.tcad_deck_mutations
            state.status = ToolStatus.COMPLETED if quality_report["status"] != "failed" else ToolStatus.FAILED
            state.final_summary = summary
            state.quality_report = quality_report
            state.next_action = quality_report["recommended_next_action"]
            state.checkpoint = {
                "completed_attempts": next_index,
                "successful_attempt": next_index,
                "successful_step_v": step,
                "summary_path": attempt.summary_path,
                "quality_status": quality_report["status"],
            }
            write_state(state, state_path)
            return state.model_dump(mode="json")

        next_index += 1
        if should_retry_with_smaller_step(attempt, request, next_index):
            step = attempt.step_v / 2.0
            state.next_action = f"retry with smaller gate bias step {step:g} V"
            state.checkpoint = {
                "completed_attempts": next_index - 1,
                "current_step_v": step,
                "last_failure_class": attempt.failure_class,
            }
            write_state(state, state_path)
            continue

        state.status = ToolStatus.FAILED
        state.next_action = "stop and report failure"
        state.checkpoint = {
            "completed_attempts": next_index - 1,
            "last_failure_class": attempt.failure_class,
            "last_failure_reason": attempt.failure_reason,
        }
        write_state(state, state_path)
        return state.model_dump(mode="json")

    state.status = ToolStatus.FAILED
    state.next_action = "maximum attempts exhausted"
    write_state(state, state_path)
    return state.model_dump(mode="json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent-callable MOS capacitor C-V sweep tool.")
    parser.add_argument("--start", type=float, default=-1.0)
    parser.add_argument("--stop", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--min-step", type=float, default=0.0625)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--quality-min-points", type=int, default=3)
    parser.add_argument("--oxide-thickness-nm", type=float, default=5.0)
    parser.add_argument("--silicon-thickness-um", type=float, default=0.2)
    parser.add_argument("--substrate-doping-cm3", type=float, default=1.0e17)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--oxide-spacing-nm", type=float, default=0.25)
    parser.add_argument("--silicon-spacing-um", type=float, default=0.002)
    parser.add_argument("--fixed-oxide-charge-cm2", type=float, default=0.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs" / "agent_tools")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def request_from_args(args: argparse.Namespace) -> MOSCapacitorCVRequest:
    return MOSCapacitorCVRequest(
        start=args.start,
        stop=args.stop,
        step=args.step,
        min_step=args.min_step,
        max_attempts=args.max_attempts,
        timeout_seconds=args.timeout_seconds,
        quality_min_points=args.quality_min_points,
        oxide_thickness_nm=args.oxide_thickness_nm,
        silicon_thickness_um=args.silicon_thickness_um,
        substrate_doping_cm3=args.substrate_doping_cm3,
        temperature_k=args.temperature_k,
        oxide_spacing_nm=args.oxide_spacing_nm,
        silicon_spacing_um=args.silicon_spacing_um,
        fixed_oxide_charge_cm2=args.fixed_oxide_charge_cm2,
        run_id=args.run_id,
        run_root=args.run_root,
        resume=args.resume,
    )


def main() -> None:
    try:
        result = run_mos_capacitor_cv_sweep(request_from_args(parse_args()))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        raise SystemExit(0 if result.get("status") != ToolStatus.FAILED.value else 1)
    except (ValidationError, ValueError, FileExistsError, FileNotFoundError) as exc:
        print(
            json.dumps(
                {
                    "tool_name": "mos_capacitor_cv_sweep",
                    "status": ToolStatus.FAILED,
                    "failure_reason": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
