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
from tcad_agent.metrics import extract_mosfet_metrics_from_csv, load_mosfet_points
from tcad_agent.physical_quality import check_mosfet_physics


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


class MOSFET2DIDRequest(BaseModel):
    sweep_type: str = Field(default="both", pattern="^(idvg|idvd|both)$")
    gate_start: float = 0.0
    gate_stop: float = 1.0
    gate_step: float = Field(default=0.5, gt=0.0)
    min_gate_step: float = Field(default=0.125, gt=0.0)
    drain_voltage: float = 0.05
    drain_start: float = 0.0
    drain_stop: float = 0.1
    drain_step: float = Field(default=0.05, gt=0.0)
    min_drain_step: float = Field(default=0.0125, gt=0.0)
    idvd_gate_voltage: float = 1.0
    threshold_current_a: float = Field(default=1e-6, gt=0.0)
    max_attempts: int = Field(default=3, ge=1)
    timeout_seconds: float = Field(default=600.0, gt=0.0)
    quality_min_points: int = Field(default=3, ge=1)
    quality_max_abs_current_a: float = Field(default=10.0, gt=0.0)
    quality_min_ion_ioff_ratio: float = Field(default=10.0, gt=0.0)
    length_um: float = Field(default=0.2, gt=0.0)
    oxide_thickness_nm: float = Field(default=5.0, gt=0.0)
    silicon_thickness_um: float = Field(default=0.05, gt=0.0)
    source_drain_length_um: float = Field(default=0.04, gt=0.0)
    source_drain_depth_um: float = Field(default=0.015, gt=0.0)
    substrate_doping_cm3: float = Field(default=1.0e17, gt=0.0)
    source_drain_doping_cm3: float = Field(default=1.0e20, gt=0.0)
    temperature_k: float = Field(default=300.0, gt=0.0)
    x_divisions: int = Field(default=12, ge=4)
    silicon_y_divisions: int = Field(default=4, ge=3)
    mobility_model: str = Field(default="constant", pattern="^(constant|doping_dependent)$")
    electron_mobility_cm2_v_s: float | None = Field(default=None, gt=0.0)
    hole_mobility_cm2_v_s: float | None = Field(default=None, gt=0.0)
    recombination_model: str = Field(default="srh", pattern="^(none|srh)$")
    electron_lifetime_s: float = Field(default=1.0e-5, gt=0.0)
    hole_lifetime_s: float = Field(default=1.0e-5, gt=0.0)
    interface_trap_density_cm2: float = Field(default=0.0, ge=0.0)
    fixed_oxide_charge_cm2: float = Field(default=0.0, ge=0.0)
    impact_ionization_model: str = Field(default="none", pattern="^(none|selberherr)$")
    model_strategy: str = Field(default="poisson_then_dd", pattern="^(poisson_then_dd|dd_direct)$")
    solver_initial_absolute_error: float = Field(default=1.0, gt=0.0)
    solver_absolute_error: float = Field(default=1.0e10, gt=0.0)
    solver_relative_error: float = Field(default=1.0e-10, gt=0.0)
    solver_max_iterations: int = Field(default=80, ge=1)
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
    def validate_request(self) -> "MOSFET2DIDRequest":
        if self.min_gate_step > self.gate_step:
            raise ValueError("min_gate_step must be less than or equal to gate_step")
        if self.min_drain_step > self.drain_step:
            raise ValueError("min_drain_step must be less than or equal to drain_step")
        if self.source_drain_length_um * 2.0 >= self.length_um:
            raise ValueError("source/drain length must leave a channel region")
        if self.source_drain_depth_um >= self.silicon_thickness_um:
            raise ValueError("source/drain depth must be less than silicon thickness")
        return self


class AttemptRecord(BaseModel):
    index: int
    status: ToolStatus
    gate_step_v: float
    drain_step_v: float
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
    tool_name: str = "mosfet_2d_id_sweep"
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


def build_runner_command(
    request: MOSFET2DIDRequest,
    attempt_index: int,
    gate_step: float,
    drain_step: float,
    run_dir: Path,
) -> list[str]:
    attempt_root = run_dir / "attempt_runs"
    attempt_id = f"attempt_{attempt_index:03d}"
    command = [
        sys.executable,
        "-m",
        "tcad_agent.examples.mosfet_2d.run",
        "--sweep-type",
        request.sweep_type,
        "--gate-start",
        str(request.gate_start),
        "--gate-stop",
        str(request.gate_stop),
        "--gate-step",
        str(gate_step),
        "--drain-voltage",
        str(request.drain_voltage),
        "--drain-start",
        str(request.drain_start),
        "--drain-stop",
        str(request.drain_stop),
        "--drain-step",
        str(drain_step),
        "--idvd-gate-voltage",
        str(request.idvd_gate_voltage),
        "--threshold-current-a",
        str(request.threshold_current_a),
        "--length-um",
        str(request.length_um),
        "--oxide-thickness-nm",
        str(request.oxide_thickness_nm),
        "--silicon-thickness-um",
        str(request.silicon_thickness_um),
        "--source-drain-length-um",
        str(request.source_drain_length_um),
        "--source-drain-depth-um",
        str(request.source_drain_depth_um),
        "--substrate-doping-cm3",
        str(request.substrate_doping_cm3),
        "--source-drain-doping-cm3",
        str(request.source_drain_doping_cm3),
        "--temperature-k",
        str(request.temperature_k),
        "--x-divisions",
        str(request.x_divisions),
        "--silicon-y-divisions",
        str(request.silicon_y_divisions),
        "--mobility-model",
        request.mobility_model,
        "--recombination-model",
        request.recombination_model,
        "--electron-lifetime-s",
        str(request.electron_lifetime_s),
        "--hole-lifetime-s",
        str(request.hole_lifetime_s),
        "--interface-trap-density-cm2",
        str(request.interface_trap_density_cm2),
        "--fixed-oxide-charge-cm2",
        str(request.fixed_oxide_charge_cm2),
        "--impact-ionization-model",
        request.impact_ionization_model,
        "--model-strategy",
        request.model_strategy,
        "--solver-initial-absolute-error",
        str(request.solver_initial_absolute_error),
        "--solver-absolute-error",
        str(request.solver_absolute_error),
        "--solver-relative-error",
        str(request.solver_relative_error),
        "--solver-max-iterations",
        str(request.solver_max_iterations),
        "--run-id",
        attempt_id,
        "--run-root",
        str(attempt_root),
    ]
    if request.electron_mobility_cm2_v_s is not None:
        command.extend(["--electron-mobility-cm2-v-s", str(request.electron_mobility_cm2_v_s)])
    if request.hole_mobility_cm2_v_s is not None:
        command.extend(["--hole-mobility-cm2-v-s", str(request.hole_mobility_cm2_v_s)])
    return command


def create_initial_state(request: MOSFET2DIDRequest, run_id: str, run_dir: Path) -> RunState:
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
        next_action="start first DEVSIM 2D MOSFET attempt",
        checkpoint={"gate_step_v": request.gate_step, "drain_step_v": request.drain_step, "completed_attempts": 0},
    )


def prepare_state(request: MOSFET2DIDRequest) -> tuple[RunState, Path]:
    run_id = request.run_id or default_run_id()
    run_dir = request.run_root / "mosfet_2d_id" / run_id
    state_path = run_dir / "state.json"
    if request.resume:
        if not state_path.exists():
            raise FileNotFoundError(f"Cannot resume; state file does not exist: {state_path}")
        return load_state(state_path), state_path
    run_dir.mkdir(parents=True, exist_ok=False)
    state = create_initial_state(request, run_id, run_dir)
    write_state(state, state_path)
    return state, state_path


def latest_retry_steps(state: RunState, request: MOSFET2DIDRequest) -> tuple[float, float]:
    if not state.attempts:
        return request.gate_step, request.drain_step
    last = state.attempts[-1]
    if last.status == ToolStatus.FAILED and last.failure_class == FailureClass.CONVERGENCE:
        return max(last.gate_step_v / 2.0, request.min_gate_step), max(last.drain_step_v / 2.0, request.min_drain_step)
    return last.gate_step_v, last.drain_step_v


def should_retry(attempt: AttemptRecord, request: MOSFET2DIDRequest, next_attempt: int) -> bool:
    if attempt.failure_class != FailureClass.CONVERGENCE:
        return False
    if next_attempt > request.max_attempts:
        return False
    return attempt.gate_step_v / 2.0 >= request.min_gate_step or attempt.drain_step_v / 2.0 >= request.min_drain_step


def idvg_shape_violations(csv_path: Path) -> int:
    points = sorted(
        [point for point in load_mosfet_points(csv_path) if point.sweep_type == "idvg"],
        key=lambda point: point.gate_voltage_v,
    )
    violations = 0
    for left, right in zip(points[:-1], points[1:]):
        if abs(right.drain_total_current_a) < abs(left.drain_total_current_a):
            violations += 1
    return violations


def judge_summary_quality(summary: dict[str, Any], request: MOSFET2DIDRequest) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    artifacts = summary.get("artifacts") or {}
    for name in ["csv", "plot", "tecplot", "log"]:
        path = artifacts.get(name)
        if not path or not Path(path).exists():
            issues.append({"code": f"missing_{name}", "severity": "error" if name == "csv" else "warning", "path": path})
    metrics = dict(summary.get("metrics") or {})
    csv_path = Path(artifacts["csv"]) if artifacts.get("csv") else None
    if csv_path and csv_path.exists():
        metrics.update(extract_mosfet_metrics_from_csv(csv_path, threshold_current_a=request.threshold_current_a))
        violations = idvg_shape_violations(csv_path)
        metrics["idvg_shape_violations"] = violations
        if violations:
            issues.append(
                {
                    "code": "idvg_not_monotonic",
                    "severity": "warning",
                    "message": "Absolute drain current decreases as gate voltage increases.",
                    "evidence": {"violations": violations},
                }
            )
    if (metrics.get("points") or 0) < request.quality_min_points:
        issues.append({"code": "too_few_points", "severity": "warning", "points": metrics.get("points")})
    max_current = metrics.get("max_abs_drain_current_a")
    if max_current is not None and float(max_current) > request.quality_max_abs_current_a:
        issues.append(
            {
                "code": "current_exceeds_policy",
                "severity": "warning",
                "max_abs_drain_current_a": max_current,
                "threshold_a": request.quality_max_abs_current_a,
            }
        )
    ratio = metrics.get("ion_ioff_ratio")
    if ratio is not None and float(ratio) < request.quality_min_ion_ioff_ratio:
        issues.append(
            {
                "code": "low_ion_ioff_ratio",
                "severity": "warning",
                "ion_ioff_ratio": ratio,
                "min_ion_ioff_ratio": request.quality_min_ion_ioff_ratio,
            }
        )
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            issues.append({"code": "nonfinite_metric", "severity": "error", "metric": key, "value": value})
    issues.extend(
        check_mosfet_physics(
            metrics,
            summary.get("parameters") or {},
            gate_start_v=request.gate_start,
            gate_stop_v=request.gate_stop,
            drain_start_v=request.drain_start,
            drain_stop_v=request.drain_stop,
            physics_models=summary.get("physics_models") or {},
        )
    )

    status = "passed"
    if any(issue["severity"] == "error" for issue in issues):
        status = "failed"
    elif issues:
        status = "suspicious"
    return {
        "status": status,
        "issues": issues,
        "metrics": metrics,
        "recommended_next_action": (
            "accept 2D MOSFET Id sweep artifacts"
            if status == "passed"
            else "review 2D MOSFET physical-quality warnings before using extracted parameters"
        ),
    }


def run_attempt(
    request: MOSFET2DIDRequest,
    state: RunState,
    state_path: Path,
    attempt_index: int,
    gate_step: float,
    drain_step: float,
) -> AttemptRecord:
    command = build_runner_command(request, attempt_index, gate_step, drain_step, Path(state.run_dir))
    attempt = AttemptRecord(
        index=attempt_index,
        status=ToolStatus.RUNNING,
        gate_step_v=gate_step,
        drain_step_v=drain_step,
        started_at=utc_timestamp(),
        command=command,
    )
    state.attempts.append(attempt)
    state.next_action = f"run 2D MOSFET attempt {attempt_index}"
    state.checkpoint = {"gate_step_v": gate_step, "drain_step_v": drain_step, "completed_attempts": attempt_index - 1}
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


def run_mosfet_2d_id_sweep(request: MOSFET2DIDRequest) -> dict[str, Any]:
    state, state_path = prepare_state(request)
    if state.status == ToolStatus.COMPLETED:
        return state.model_dump(mode="json")

    next_index = len(state.attempts) + 1
    gate_step, drain_step = latest_retry_steps(state, request)
    while next_index <= request.max_attempts:
        attempt = run_attempt(request, state, state_path, next_index, gate_step, drain_step)
        if attempt.status == ToolStatus.COMPLETED:
            summary = json.loads(Path(attempt.summary_path).read_text(encoding="utf-8"))
            quality_report = judge_summary_quality(summary, request)
            deck_artifacts = write_deck_artifacts(
                Path(state.run_dir),
                tool_name="mosfet_2d_id_sweep",
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
                "successful_gate_step_v": gate_step,
                "successful_drain_step_v": drain_step,
                "summary_path": attempt.summary_path,
                "quality_status": quality_report["status"],
            }
            write_state(state, state_path)
            return state.model_dump(mode="json")

        next_index += 1
        if should_retry(attempt, request, next_index):
            gate_step = max(attempt.gate_step_v / 2.0, request.min_gate_step)
            drain_step = max(attempt.drain_step_v / 2.0, request.min_drain_step)
            state.next_action = f"retry with smaller bias steps gate={gate_step:g} V drain={drain_step:g} V"
            state.checkpoint = {
                "completed_attempts": next_index - 1,
                "gate_step_v": gate_step,
                "drain_step_v": drain_step,
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
    parser = argparse.ArgumentParser(description="Agent-callable simplified 2D MOSFET Id-Vg / Id-Vd sweep tool.")
    parser.add_argument("--sweep-type", choices=["idvg", "idvd", "both"], default="both")
    parser.add_argument("--gate-start", type=float, default=0.0)
    parser.add_argument("--gate-stop", type=float, default=1.0)
    parser.add_argument("--gate-step", type=float, default=0.5)
    parser.add_argument("--min-gate-step", type=float, default=0.125)
    parser.add_argument("--drain-voltage", type=float, default=0.05)
    parser.add_argument("--drain-start", type=float, default=0.0)
    parser.add_argument("--drain-stop", type=float, default=0.1)
    parser.add_argument("--drain-step", type=float, default=0.05)
    parser.add_argument("--min-drain-step", type=float, default=0.0125)
    parser.add_argument("--idvd-gate-voltage", type=float, default=1.0)
    parser.add_argument("--threshold-current-a", type=float, default=1e-6)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--quality-min-points", type=int, default=3)
    parser.add_argument("--quality-max-abs-current-a", type=float, default=10.0)
    parser.add_argument("--quality-min-ion-ioff-ratio", type=float, default=10.0)
    parser.add_argument("--length-um", type=float, default=0.2)
    parser.add_argument("--oxide-thickness-nm", type=float, default=5.0)
    parser.add_argument("--silicon-thickness-um", type=float, default=0.05)
    parser.add_argument("--source-drain-length-um", type=float, default=0.04)
    parser.add_argument("--source-drain-depth-um", type=float, default=0.015)
    parser.add_argument("--substrate-doping-cm3", type=float, default=1.0e17)
    parser.add_argument("--source-drain-doping-cm3", type=float, default=1.0e20)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--x-divisions", type=int, default=12)
    parser.add_argument("--silicon-y-divisions", type=int, default=4)
    parser.add_argument("--mobility-model", choices=["constant", "doping_dependent"], default="constant")
    parser.add_argument("--electron-mobility-cm2-v-s", type=float, default=None)
    parser.add_argument("--hole-mobility-cm2-v-s", type=float, default=None)
    parser.add_argument("--recombination-model", choices=["none", "srh"], default="srh")
    parser.add_argument("--electron-lifetime-s", type=float, default=1.0e-5)
    parser.add_argument("--hole-lifetime-s", type=float, default=1.0e-5)
    parser.add_argument("--interface-trap-density-cm2", type=float, default=0.0)
    parser.add_argument("--fixed-oxide-charge-cm2", type=float, default=0.0)
    parser.add_argument("--impact-ionization-model", choices=["none", "selberherr"], default="none")
    parser.add_argument("--model-strategy", choices=["poisson_then_dd", "dd_direct"], default="poisson_then_dd")
    parser.add_argument("--solver-initial-absolute-error", type=float, default=1.0)
    parser.add_argument("--solver-absolute-error", type=float, default=1.0e10)
    parser.add_argument("--solver-relative-error", type=float, default=1.0e-10)
    parser.add_argument("--solver-max-iterations", type=int, default=80)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs" / "agent_tools")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def request_from_args(args: argparse.Namespace) -> MOSFET2DIDRequest:
    return MOSFET2DIDRequest(
        sweep_type=args.sweep_type,
        gate_start=args.gate_start,
        gate_stop=args.gate_stop,
        gate_step=args.gate_step,
        min_gate_step=args.min_gate_step,
        drain_voltage=args.drain_voltage,
        drain_start=args.drain_start,
        drain_stop=args.drain_stop,
        drain_step=args.drain_step,
        min_drain_step=args.min_drain_step,
        idvd_gate_voltage=args.idvd_gate_voltage,
        threshold_current_a=args.threshold_current_a,
        max_attempts=args.max_attempts,
        timeout_seconds=args.timeout_seconds,
        quality_min_points=args.quality_min_points,
        quality_max_abs_current_a=args.quality_max_abs_current_a,
        quality_min_ion_ioff_ratio=args.quality_min_ion_ioff_ratio,
        length_um=args.length_um,
        oxide_thickness_nm=args.oxide_thickness_nm,
        silicon_thickness_um=args.silicon_thickness_um,
        source_drain_length_um=args.source_drain_length_um,
        source_drain_depth_um=args.source_drain_depth_um,
        substrate_doping_cm3=args.substrate_doping_cm3,
        source_drain_doping_cm3=args.source_drain_doping_cm3,
        temperature_k=args.temperature_k,
        x_divisions=args.x_divisions,
        silicon_y_divisions=args.silicon_y_divisions,
        mobility_model=args.mobility_model,
        electron_mobility_cm2_v_s=args.electron_mobility_cm2_v_s,
        hole_mobility_cm2_v_s=args.hole_mobility_cm2_v_s,
        recombination_model=args.recombination_model,
        electron_lifetime_s=args.electron_lifetime_s,
        hole_lifetime_s=args.hole_lifetime_s,
        interface_trap_density_cm2=args.interface_trap_density_cm2,
        fixed_oxide_charge_cm2=args.fixed_oxide_charge_cm2,
        impact_ionization_model=args.impact_ionization_model,
        model_strategy=args.model_strategy,
        solver_initial_absolute_error=args.solver_initial_absolute_error,
        solver_absolute_error=args.solver_absolute_error,
        solver_relative_error=args.solver_relative_error,
        solver_max_iterations=args.solver_max_iterations,
        run_id=args.run_id,
        run_root=args.run_root,
        resume=args.resume,
    )


def main() -> None:
    try:
        result = run_mosfet_2d_id_sweep(request_from_args(parse_args()))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        raise SystemExit(0 if result.get("status") != ToolStatus.FAILED.value else 1)
    except (ValidationError, ValueError, FileExistsError, FileNotFoundError) as exc:
        print(
            json.dumps(
                {
                    "tool_name": "mosfet_2d_id_sweep",
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
