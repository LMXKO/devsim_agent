from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from tcad_agent.deck_writer import write_deck_artifacts
from tcad_agent.tools.result_judge import QualityPolicy, judge_pn_junction_iv


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


class PNJunctionIVRequest(BaseModel):
    start: float = Field(default=0.0, description="Start bias in volts.")
    stop: float = Field(default=0.5, description="Stop bias in volts.")
    step: float = Field(default=0.1, gt=0.0, description="Initial bias step in volts.")
    min_step: float = Field(
        default=0.0125,
        gt=0.0,
        description="Smallest retry bias step in volts.",
    )
    max_attempts: int = Field(default=5, ge=1)
    timeout_seconds: float = Field(default=300.0, gt=0.0)
    quality_min_points: int = Field(default=3, ge=1)
    quality_max_abs_current_a: float = Field(default=1.0, gt=0.0)
    quality_max_convergence_failures: int = Field(default=0, ge=0)
    length_um: float = Field(default=0.1, gt=0.0)
    junction_um: float = Field(default=0.05, gt=0.0)
    p_doping_cm3: float = Field(default=1.0e18, gt=0.0)
    n_doping_cm3: float = Field(default=1.0e18, gt=0.0)
    temperature_k: float = Field(default=300.0, gt=0.0)
    electron_lifetime_s: float = Field(default=1.0e-8, gt=0.0)
    hole_lifetime_s: float = Field(default=1.0e-8, gt=0.0)
    contact_spacing_um: float = Field(default=0.001, gt=0.0)
    junction_spacing_um: float = Field(default=1.0e-5, gt=0.0)
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
    def validate_sweep(self) -> "PNJunctionIVRequest":
        if self.stop == self.start:
            raise ValueError("stop must differ from start")
        if self.min_step > self.step:
            raise ValueError("min_step must be less than or equal to step")
        if self.junction_um >= self.length_um:
            raise ValueError("junction_um must be less than length_um")
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
    tool_name: str = "pn_junction_iv_sweep"
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
    path.write_text(
        json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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


def build_runner_command(request: PNJunctionIVRequest, attempt_index: int, step: float, run_dir: Path) -> list[str]:
    attempt_root = run_dir / "attempt_runs"
    attempt_id = f"attempt_{attempt_index:03d}"
    return [
        sys.executable,
        "-m",
        "tcad_agent.examples.pn_junction.run",
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
        "--length-um",
        str(request.length_um),
        "--junction-um",
        str(request.junction_um),
        "--p-doping-cm3",
        str(request.p_doping_cm3),
        "--n-doping-cm3",
        str(request.n_doping_cm3),
        "--temperature-k",
        str(request.temperature_k),
        "--electron-lifetime-s",
        str(request.electron_lifetime_s),
        "--hole-lifetime-s",
        str(request.hole_lifetime_s),
        "--contact-spacing-um",
        str(request.contact_spacing_um),
        "--junction-spacing-um",
        str(request.junction_spacing_um),
    ]


def latest_retry_step(state: RunState, request: PNJunctionIVRequest) -> float:
    if not state.attempts:
        return request.step
    last = state.attempts[-1]
    if last.status == ToolStatus.FAILED and last.failure_class == FailureClass.CONVERGENCE:
        return max(last.step_v / 2.0, request.min_step)
    return last.step_v


def should_retry_with_smaller_step(
    attempt: AttemptRecord,
    request: PNJunctionIVRequest,
    next_attempt_index: int,
) -> bool:
    if attempt.failure_class != FailureClass.CONVERGENCE:
        return False
    if next_attempt_index > request.max_attempts:
        return False
    return attempt.step_v / 2.0 >= request.min_step


def create_initial_state(request: PNJunctionIVRequest, run_id: str, run_dir: Path) -> RunState:
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
        next_action="start first DEVSIM attempt",
        checkpoint={
            "current_step_v": request.step,
            "completed_attempts": 0,
        },
    )


def judge_summary_quality(
    summary: dict[str, Any],
    state: RunState,
    request: PNJunctionIVRequest,
) -> dict[str, Any]:
    report = judge_pn_junction_iv(
        summary,
        attempts=[attempt.model_dump(mode="json") for attempt in state.attempts],
        policy=QualityPolicy(
            min_points=request.quality_min_points,
            max_abs_current_a=request.quality_max_abs_current_a,
            max_convergence_failures=request.quality_max_convergence_failures,
        ),
    )
    return report.model_dump(mode="json")


def prepare_state(request: PNJunctionIVRequest) -> tuple[RunState, Path]:
    run_id = request.run_id or default_run_id()
    run_dir = request.run_root / "pn_junction_iv" / run_id
    state_path = run_dir / "state.json"

    if request.resume:
        if not state_path.exists():
            raise FileNotFoundError(f"Cannot resume; state file does not exist: {state_path}")
        return load_state(state_path), state_path

    run_dir.mkdir(parents=True, exist_ok=False)
    state = create_initial_state(request, run_id, run_dir)
    write_state(state, state_path)
    return state, state_path


def run_attempt(
    request: PNJunctionIVRequest,
    state: RunState,
    state_path: Path,
    attempt_index: int,
    step: float,
) -> AttemptRecord:
    run_dir = Path(state.run_dir)
    command = build_runner_command(request, attempt_index, step, run_dir)
    attempt = AttemptRecord(
        index=attempt_index,
        status=ToolStatus.RUNNING,
        step_v=step,
        started_at=utc_timestamp(),
        command=command,
    )
    state.attempts.append(attempt)
    state.next_action = f"run DEVSIM attempt {attempt_index} with step {step:g} V"
    state.checkpoint = {
        "current_step_v": step,
        "completed_attempts": attempt_index - 1,
    }
    write_state(state, state_path)

    try:
        completed = subprocess.run(
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
            failure_class, reason = classify_failure(
                completed.returncode,
                completed.stdout,
                completed.stderr,
            )
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


def run_pn_junction_iv_sweep(request: PNJunctionIVRequest) -> dict[str, Any]:
    state, state_path = prepare_state(request)
    if state.status == ToolStatus.COMPLETED:
        if state.final_summary and not state.quality_report:
            state.quality_report = judge_summary_quality(state.final_summary, state, request)
            write_state(state, state_path)
        return state.model_dump(mode="json")

    next_index = len(state.attempts) + 1
    step = latest_retry_step(state, request)

    while next_index <= request.max_attempts:
        attempt = run_attempt(request, state, state_path, next_index, step)

        if attempt.status == ToolStatus.COMPLETED:
            summary = json.loads(Path(attempt.summary_path).read_text(encoding="utf-8"))
            quality_report = judge_summary_quality(summary, state, request)
            deck_artifacts = write_deck_artifacts(
                Path(state.run_dir),
                tool_name="pn_junction_iv_sweep",
                request=request.model_dump(mode="json"),
                deck_spec=request.tcad_deck_spec,
                mutations=request.tcad_deck_mutations,
                source_goal_text=(request.tcad_deck_spec or {}).get("source_goal_text") if request.tcad_deck_spec else None,
            )
            summary.setdefault("artifacts", {}).update(deck_artifacts)
            summary["tcad_deck_spec"] = request.tcad_deck_spec
            summary["tcad_deck_mutations"] = request.tcad_deck_mutations
            state.status = ToolStatus.COMPLETED
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
            state.next_action = f"retry with smaller bias step {step:g} V"
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
    state.next_action = "stop; maximum attempts reached"
    state.checkpoint = {
        "completed_attempts": len(state.attempts),
        "last_failure_class": state.attempts[-1].failure_class if state.attempts else None,
    }
    write_state(state, state_path)
    return state.model_dump(mode="json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent-callable PN junction IV sweep tool.")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--stop", type=float, default=0.5)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--min-step", type=float, default=0.0125)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--quality-min-points", type=int, default=3)
    parser.add_argument("--quality-max-abs-current-a", type=float, default=1.0)
    parser.add_argument("--quality-max-convergence-failures", type=int, default=0)
    parser.add_argument("--length-um", type=float, default=0.1)
    parser.add_argument("--junction-um", type=float, default=0.05)
    parser.add_argument("--p-doping-cm3", type=float, default=1.0e18)
    parser.add_argument("--n-doping-cm3", type=float, default=1.0e18)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--electron-lifetime-s", type=float, default=1.0e-8)
    parser.add_argument("--hole-lifetime-s", type=float, default=1.0e-8)
    parser.add_argument("--contact-spacing-um", type=float, default=0.001)
    parser.add_argument("--junction-spacing-um", type=float, default=1.0e-5)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs" / "agent_tools")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def request_from_args(args: argparse.Namespace) -> PNJunctionIVRequest:
    return PNJunctionIVRequest(
        start=args.start,
        stop=args.stop,
        step=args.step,
        min_step=args.min_step,
        max_attempts=args.max_attempts,
        timeout_seconds=args.timeout_seconds,
        quality_min_points=args.quality_min_points,
        quality_max_abs_current_a=args.quality_max_abs_current_a,
        quality_max_convergence_failures=args.quality_max_convergence_failures,
        length_um=args.length_um,
        junction_um=args.junction_um,
        p_doping_cm3=args.p_doping_cm3,
        n_doping_cm3=args.n_doping_cm3,
        temperature_k=args.temperature_k,
        electron_lifetime_s=args.electron_lifetime_s,
        hole_lifetime_s=args.hole_lifetime_s,
        contact_spacing_um=args.contact_spacing_um,
        junction_spacing_um=args.junction_spacing_um,
        run_id=args.run_id,
        run_root=args.run_root,
        resume=args.resume,
    )


def main() -> None:
    try:
        request = request_from_args(parse_args())
        result = run_pn_junction_iv_sweep(request)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        raise SystemExit(0 if result["status"] == ToolStatus.COMPLETED else 1)
    except (ValidationError, ValueError, FileNotFoundError) as exc:
        print(
            json.dumps(
                {
                    "tool_name": "pn_junction_iv_sweep",
                    "status": ToolStatus.FAILED,
                    "failure_class": FailureClass.VALIDATION,
                    "failure_reason": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
