from __future__ import annotations

import argparse
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from tcad_agent.tools.llm_diagnose import diagnose_state
from tcad_agent.tools.pn_junction_iv import PNJunctionIVRequest, run_pn_junction_iv_sweep
from tcad_agent.tools.strategy_executor import StrategyStatus, build_strategy_plan


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LoopStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CycleStatus(str, Enum):
    RUNNING = "running"
    ACCEPTED = "accepted"
    NEEDS_FOLLOWUP = "needs_followup"
    FAILED = "failed"


class AutonomousLoopRequest(BaseModel):
    task: Literal["pn_junction_iv"] = "pn_junction_iv"
    start: float = 0.0
    stop: float = 0.5
    step: float = Field(default=0.1, gt=0.0)
    min_step: float = Field(default=0.0125, gt=0.0)
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
    max_cycles: int = Field(default=3, ge=1)
    use_llm: bool = False
    force_llm: bool = False
    max_log_chars: int = Field(default=4000, ge=0)
    loop_id: str | None = None
    loop_root: Path = PROJECT_ROOT / "runs" / "autonomous_loop"
    run_root: Path = PROJECT_ROOT / "runs" / "agent_tools"
    resume: bool = False

    @model_validator(mode="after")
    def validate_sweep(self) -> "AutonomousLoopRequest":
        if self.stop < self.start:
            raise ValueError("stop must be greater than or equal to start")
        if self.min_step > self.step:
            raise ValueError("min_step must be less than or equal to step")
        if self.junction_um >= self.length_um:
            raise ValueError("junction_um must be less than length_um")
        return self


class CycleRecord(BaseModel):
    index: int
    status: CycleStatus
    started_at: str
    completed_at: str | None = None
    request: dict[str, Any]
    run_id: str | None = None
    state_path: str | None = None
    tool_status: str | None = None
    quality_status: str | None = None
    diagnosis_path: str | None = None
    diagnosis_status: str | None = None
    strategy_plan_path: str | None = None
    strategy_status: str | None = None
    next_request: dict[str, Any] | None = None
    reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class LoopState(BaseModel):
    tool_name: str = "autonomous_tcad_loop"
    task: str = "pn_junction_iv"
    status: LoopStatus
    loop_id: str
    loop_dir: str
    request: dict[str, Any]
    created_at: str
    updated_at: str
    cycles: list[CycleRecord] = Field(default_factory=list)
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    next_action: str | None = None
    final_state_path: str | None = None
    final_quality_report: dict[str, Any] | None = None
    final_summary: dict[str, Any] | None = None
    failure_reason: str | None = None


PNRunner = Callable[[PNJunctionIVRequest], dict[str, Any]]
DiagnosisRunner = Callable[..., Any]
StrategyBuilder = Callable[..., Any]


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_loop_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_loop_state(state: LoopState, path: Path) -> None:
    state.updated_at = utc_timestamp()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_loop_state(path: Path) -> LoopState:
    return LoopState.model_validate_json(path.read_text(encoding="utf-8"))


def create_initial_state(
    request: AutonomousLoopRequest,
    loop_id: str,
    loop_dir: Path,
) -> LoopState:
    now = utc_timestamp()
    return LoopState(
        status=LoopStatus.RUNNING,
        loop_id=loop_id,
        loop_dir=str(loop_dir),
        request=request.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        checkpoint={"completed_cycles": 0},
        next_action="start first TCAD tool cycle",
    )


def prepare_loop_state(
    request: AutonomousLoopRequest,
) -> tuple[LoopState, Path, AutonomousLoopRequest]:
    if request.resume:
        if not request.loop_id:
            raise ValueError("--resume requires --loop-id")
        state_path = request.loop_root / request.loop_id / "loop_state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"Cannot resume; loop state does not exist: {state_path}")
        state = load_loop_state(state_path)
        stored_request = AutonomousLoopRequest.model_validate(state.request)
        return state, state_path, stored_request

    loop_id = request.loop_id or default_loop_id()
    loop_dir = request.loop_root / loop_id
    loop_dir.mkdir(parents=True, exist_ok=False)
    state_path = loop_dir / "loop_state.json"
    state = create_initial_state(request, loop_id, loop_dir)
    write_loop_state(state, state_path)
    return state, state_path, request


def initial_pn_request(request: AutonomousLoopRequest, loop_id: str) -> PNJunctionIVRequest:
    return PNJunctionIVRequest(
        start=request.start,
        stop=request.stop,
        step=request.step,
        min_step=request.min_step,
        max_attempts=request.max_attempts,
        timeout_seconds=request.timeout_seconds,
        quality_min_points=request.quality_min_points,
        quality_max_abs_current_a=request.quality_max_abs_current_a,
        quality_max_convergence_failures=request.quality_max_convergence_failures,
        length_um=request.length_um,
        junction_um=request.junction_um,
        p_doping_cm3=request.p_doping_cm3,
        n_doping_cm3=request.n_doping_cm3,
        temperature_k=request.temperature_k,
        electron_lifetime_s=request.electron_lifetime_s,
        hole_lifetime_s=request.hole_lifetime_s,
        contact_spacing_um=request.contact_spacing_um,
        junction_spacing_um=request.junction_spacing_um,
        run_id=f"{loop_id}_cycle_001",
        run_root=request.run_root,
        resume=False,
    )


def expected_tool_state_path(request: PNJunctionIVRequest) -> Path | None:
    if not request.run_id:
        return None
    return request.run_root / "pn_junction_iv" / request.run_id / "state.json"


def request_for_running_cycle(cycle: CycleRecord) -> PNJunctionIVRequest:
    request = PNJunctionIVRequest.model_validate(cycle.request)
    expected_state = expected_tool_state_path(request)
    if expected_state and expected_state.exists():
        request.resume = True
    return request


def current_request_from_checkpoint(
    state: LoopState,
    request: AutonomousLoopRequest,
) -> tuple[PNJunctionIVRequest, CycleRecord | None]:
    if state.cycles and state.cycles[-1].status == CycleStatus.RUNNING:
        return request_for_running_cycle(state.cycles[-1]), state.cycles[-1]

    pending = state.checkpoint.get("pending_request")
    if pending:
        return PNJunctionIVRequest.model_validate(pending), None

    return initial_pn_request(request, state.loop_id), None


def tool_state_path_from_result(result: dict[str, Any]) -> Path | None:
    run_dir = result.get("run_dir")
    if not run_dir:
        return None
    return Path(run_dir) / "state.json"


def quality_status_from_result(result: dict[str, Any]) -> str | None:
    quality_report = result.get("quality_report") or {}
    return quality_report.get("status")


def finish_failed_loop(
    state: LoopState,
    state_path: Path,
    cycle: CycleRecord | None,
    reason: str,
) -> dict[str, Any]:
    if cycle:
        cycle.status = CycleStatus.FAILED
        cycle.completed_at = cycle.completed_at or utc_timestamp()
        cycle.error = reason
    state.status = LoopStatus.FAILED
    state.failure_reason = reason
    state.next_action = "stop autonomous loop and inspect checkpoint"
    state.checkpoint["failure_reason"] = reason
    write_loop_state(state, state_path)
    return state.model_dump(mode="json")


def run_diagnosis_if_requested(
    request: AutonomousLoopRequest,
    tool_state_path: Path,
    cycle: CycleRecord,
    diagnosis_runner: DiagnosisRunner,
    diagnosis_client: Any | None,
) -> Path | None:
    if not request.use_llm:
        return tool_state_path.parent / "llm_diagnosis.disabled.json"

    try:
        result = diagnosis_runner(
            state_path=tool_state_path,
            force=request.force_llm,
            max_log_chars=request.max_log_chars,
            client=diagnosis_client,
        )
    except Exception as exc:
        cycle.diagnosis_status = "failed"
        cycle.warnings.append(f"LLM diagnosis failed; falling back to deterministic strategy: {exc}")
        return tool_state_path.parent / "llm_diagnosis.failed.json"

    cycle.diagnosis_status = getattr(result, "status", None)
    output_path = getattr(result, "output_path", None)
    if output_path:
        cycle.diagnosis_path = str(output_path)
        return Path(output_path)
    return None


def plan_followup(
    tool_state_path: Path,
    diagnosis_path: Path | None,
    cycle: CycleRecord,
    strategy_builder: StrategyBuilder,
) -> dict[str, Any] | None:
    plan = strategy_builder(
        state_path=tool_state_path,
        diagnosis_path=diagnosis_path,
        execute=False,
    )
    cycle.strategy_status = plan.status.value if isinstance(plan.status, StrategyStatus) else str(plan.status)
    cycle.strategy_plan_path = plan.output_path
    cycle.reason = plan.reason
    cycle.warnings.extend(plan.warnings)
    cycle.next_request = plan.next_request
    return plan.next_request


def run_autonomous_loop(
    request: AutonomousLoopRequest,
    pn_runner: PNRunner = run_pn_junction_iv_sweep,
    diagnosis_runner: DiagnosisRunner = diagnose_state,
    strategy_builder: StrategyBuilder = build_strategy_plan,
    diagnosis_client: Any | None = None,
) -> dict[str, Any]:
    state, state_path, active_request = prepare_loop_state(request)
    if state.status != LoopStatus.RUNNING:
        return state.model_dump(mode="json")

    pn_request, active_cycle = current_request_from_checkpoint(state, active_request)

    while state.status == LoopStatus.RUNNING:
        if active_cycle is None:
            if len(state.cycles) >= active_request.max_cycles:
                return finish_failed_loop(
                    state,
                    state_path,
                    None,
                    "maximum autonomous cycles reached without an accepted result",
                )
            cycle_index = len(state.cycles) + 1
            active_cycle = CycleRecord(
                index=cycle_index,
                status=CycleStatus.RUNNING,
                started_at=utc_timestamp(),
                request=pn_request.model_dump(mode="json"),
                run_id=pn_request.run_id,
                state_path=str(expected_tool_state_path(pn_request))
                if expected_tool_state_path(pn_request)
                else None,
            )
            state.cycles.append(active_cycle)
            state.checkpoint = {
                "completed_cycles": cycle_index - 1,
                "active_request": pn_request.model_dump(mode="json"),
            }
            state.next_action = f"run cycle {cycle_index} TCAD tool request"
            write_loop_state(state, state_path)

        try:
            result = pn_runner(pn_request)
        except Exception as exc:
            return finish_failed_loop(state, state_path, active_cycle, str(exc))

        tool_state_path = tool_state_path_from_result(result)
        active_cycle.tool_status = result.get("status")
        active_cycle.quality_status = quality_status_from_result(result)
        if tool_state_path:
            active_cycle.state_path = str(tool_state_path)
            state.checkpoint["last_state_path"] = str(tool_state_path)

        if active_cycle.tool_status == "completed" and active_cycle.quality_status == "passed":
            active_cycle.status = CycleStatus.ACCEPTED
            active_cycle.completed_at = utc_timestamp()
            active_cycle.reason = "quality_report.status is passed"
            state.status = LoopStatus.COMPLETED
            state.final_state_path = str(tool_state_path) if tool_state_path else None
            state.final_quality_report = result.get("quality_report")
            state.final_summary = result.get("final_summary")
            state.next_action = "accept result artifacts and proceed to the next TCAD task"
            state.checkpoint = {
                "completed_cycles": len(state.cycles),
                "accepted_cycle": active_cycle.index,
                "final_state_path": state.final_state_path,
                "quality_status": active_cycle.quality_status,
            }
            write_loop_state(state, state_path)
            return state.model_dump(mode="json")

        if len(state.cycles) >= active_request.max_cycles:
            return finish_failed_loop(
                state,
                state_path,
                active_cycle,
                "maximum autonomous cycles reached without an accepted result",
            )

        if tool_state_path is None or not tool_state_path.exists():
            return finish_failed_loop(
                state,
                state_path,
                active_cycle,
                "TCAD tool did not produce a resumable state.json",
            )

        diagnosis_path = run_diagnosis_if_requested(
            active_request,
            tool_state_path,
            active_cycle,
            diagnosis_runner,
            diagnosis_client,
        )

        try:
            next_request = plan_followup(
                tool_state_path,
                diagnosis_path,
                active_cycle,
                strategy_builder,
            )
        except Exception as exc:
            return finish_failed_loop(state, state_path, active_cycle, str(exc))

        if next_request is None:
            return finish_failed_loop(
                state,
                state_path,
                active_cycle,
                "strategy executor did not produce a follow-up request",
            )

        active_cycle.status = CycleStatus.NEEDS_FOLLOWUP
        active_cycle.completed_at = utc_timestamp()
        state.checkpoint = {
            "completed_cycles": len(state.cycles),
            "last_state_path": str(tool_state_path),
            "pending_request": next_request,
            "quality_status": active_cycle.quality_status,
        }
        state.next_action = "run constrained follow-up TCAD request"
        write_loop_state(state, state_path)

        pn_request = PNJunctionIVRequest.model_validate(next_request)
        active_cycle = None

    return state.model_dump(mode="json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpointed autonomous TCAD execution loop.")
    parser.add_argument("--task", default="pn_junction_iv")
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
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--loop-id", default=None)
    parser.add_argument("--loop-root", type=Path, default=PROJECT_ROOT / "runs" / "autonomous_loop")
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs" / "agent_tools")
    parser.add_argument("--resume", action="store_true")
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument("--use-llm", dest="use_llm", action="store_true")
    llm_group.add_argument("--no-llm", dest="use_llm", action="store_false")
    parser.set_defaults(use_llm=False)
    parser.add_argument("--force-llm", action="store_true")
    parser.add_argument("--max-log-chars", type=int, default=4000)
    return parser.parse_args()


def request_from_args(args: argparse.Namespace) -> AutonomousLoopRequest:
    return AutonomousLoopRequest(
        task=args.task,
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
        max_cycles=args.max_cycles,
        use_llm=args.use_llm,
        force_llm=args.force_llm,
        max_log_chars=args.max_log_chars,
        loop_id=args.loop_id,
        loop_root=args.loop_root,
        run_root=args.run_root,
        resume=args.resume,
    )


def main() -> None:
    try:
        result = run_autonomous_loop(request_from_args(parse_args()))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        raise SystemExit(0 if result["status"] == LoopStatus.COMPLETED else 1)
    except (ValidationError, ValueError, FileNotFoundError) as exc:
        print(
            json.dumps(
                {
                    "tool_name": "autonomous_tcad_loop",
                    "status": LoopStatus.FAILED,
                    "failure_reason": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
