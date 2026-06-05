from __future__ import annotations

import json
import hashlib
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from tcad_agent.repair_strategy import RepairAction, RepairPlan, build_repair_plan, repair_request
from tcad_agent.task_spec import PROJECT_ROOT
from tcad_agent.tools.diode_breakdown import DiodeBreakdownRequest, run_diode_breakdown_sweep
from tcad_agent.tools.extended_device_sweep import ExtendedDeviceRequest, run_extended_device_sweep
from tcad_agent.tools.mos_capacitor_cv import MOSCapacitorCVRequest, run_mos_capacitor_cv_sweep
from tcad_agent.tools.mosfet_2d_id import MOSFET2DIDRequest, run_mosfet_2d_id_sweep
from tcad_agent.tools.pn_junction_iv import PNJunctionIVRequest, run_pn_junction_iv_sweep


Runner = Callable[[dict[str, Any]], dict[str, Any]]


class RepairExecutionStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    WAITING_FOR_USER = "waiting_for_user"
    FAILED = "failed"


class RepairExecutionAttempt(BaseModel):
    index: int
    action_name: str
    target_tool: str
    source_state_path: str
    request_patch: dict[str, Any]
    next_request: dict[str, Any]
    status: RepairExecutionStatus
    started_at: str
    completed_at: str | None = None
    result_state_path: str | None = None
    quality_status: str | None = None
    result: dict[str, Any] | None = None
    failure_reason: str | None = None


class RepairExecutionState(BaseModel):
    tool_name: str = "tcad_repair_executor"
    status: RepairExecutionStatus
    execution_id: str
    source_state_path: str
    execution_dir: str
    created_at: str
    updated_at: str
    execute: bool
    max_rounds: int
    allow_user_confirmation_actions: bool = False
    current_state_path: str
    repair_plan_path: str | None = None
    attempts: list[RepairExecutionAttempt] = Field(default_factory=list)
    final_state_path: str | None = None
    final_quality_status: str | None = None
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    next_action: str | None = None
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_execution_id() -> str:
    return f"repair_exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_state(state: RepairExecutionState, path: Path) -> None:
    state.updated_at = utc_timestamp()
    write_json(path, state.model_dump(mode="json"))


def load_state(path: Path) -> RepairExecutionState:
    return RepairExecutionState.model_validate_json(path.read_text(encoding="utf-8"))


def default_execution_root(source_state_path: Path) -> Path:
    return source_state_path.parent / "repair_execution"


def sanitize_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned or "repair"


def compact_source_run_id(value: str, *, max_chars: int = 72) -> str:
    cleaned = sanitize_id(value)
    if "_repair_" in cleaned:
        cleaned = cleaned.split("_repair_", 1)[0]
    if len(cleaned) <= max_chars:
        return cleaned
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned[: max_chars - 11]}_{digest}"


def quality_status_from_state(state: dict[str, Any]) -> str | None:
    report = state.get("quality_report") or state.get("final_quality_report") or {}
    return report.get("status")


def is_accepted_state(state: dict[str, Any]) -> bool:
    return state.get("status") == "completed" and quality_status_from_state(state) == "passed"


def infer_result_state_path(result: dict[str, Any]) -> str | None:
    for key in ["state_path", "result_state_path", "source_state_path"]:
        value = result.get(key)
        if value:
            return str(value)
    run_dir = result.get("run_dir") or result.get("convergence_dir")
    if run_dir:
        candidate = Path(run_dir) / "state.json"
        if candidate.exists():
            return str(candidate.resolve())
    return None


def allocate_repair_run_id(source_state: dict[str, Any], action: RepairAction, round_index: int) -> str:
    source = source_state.get("run_id") or source_state.get("task_id") or source_state.get("convergence_id") or "run"
    compact_source = compact_source_run_id(str(source))
    return sanitize_id(f"{compact_source}_repair_{action.name}_{round_index:03d}")


def apply_patch_to_request(
    source_state: dict[str, Any],
    action: RepairAction,
    round_index: int,
) -> dict[str, Any]:
    request = repair_request(source_state)
    request.update(action.request_patch)
    request.setdefault("resume", False)
    request["run_id"] = allocate_repair_run_id(source_state, action, round_index)
    if "run_root" not in request and source_state.get("run_dir"):
        run_dir = Path(str(source_state["run_dir"]))
        # Most tool layouts are <run_root>/<tool_family>/<run_id>.
        request["run_root"] = str(run_dir.parent.parent if len(run_dir.parents) >= 2 else PROJECT_ROOT / "runs" / "agent_tools")
    elif "run_root" not in request:
        request["run_root"] = str(PROJECT_ROOT / "runs" / "agent_tools")
    return request


def default_runner_registry() -> dict[str, Runner]:
    def pn_runner(request: dict[str, Any]) -> dict[str, Any]:
        return run_pn_junction_iv_sweep(PNJunctionIVRequest.model_validate(request))

    def moscap_runner(request: dict[str, Any]) -> dict[str, Any]:
        return run_mos_capacitor_cv_sweep(MOSCapacitorCVRequest.model_validate(request))

    def diode_runner(request: dict[str, Any]) -> dict[str, Any]:
        return run_diode_breakdown_sweep(DiodeBreakdownRequest.model_validate(request))

    def mosfet_runner(request: dict[str, Any]) -> dict[str, Any]:
        return run_mosfet_2d_id_sweep(MOSFET2DIDRequest.model_validate(request))

    def extended_runner(request: dict[str, Any]) -> dict[str, Any]:
        return run_extended_device_sweep(ExtendedDeviceRequest.model_validate(request)).model_dump(mode="json")

    return {
        "pn_junction_iv_sweep": pn_runner,
        "mos_capacitor_cv_sweep": moscap_runner,
        "diode_breakdown_leakage_sweep": diode_runner,
        "mosfet_2d_id_sweep": mosfet_runner,
        "extended_device_sweep": extended_runner,
    }


def is_executable_action(action: RepairAction, *, allow_user_confirmation_actions: bool) -> bool:
    if action.user_confirmation_required and not allow_user_confirmation_actions:
        return False
    if not action.target_tool:
        return False
    if not action.request_patch and action.name.endswith("_review"):
        return False
    return True


def choose_action(
    plan: RepairPlan,
    *,
    allow_user_confirmation_actions: bool,
    tried_action_names: set[str] | None = None,
) -> RepairAction | None:
    tried = tried_action_names or set()
    first_executable: RepairAction | None = None
    for action in plan.actions:
        if not is_executable_action(action, allow_user_confirmation_actions=allow_user_confirmation_actions):
            continue
        if first_executable is None:
            first_executable = action
        if action.name in tried:
            continue
        return action
    return first_executable


def create_initial_state(
    source_state_path: Path,
    execution_id: str,
    execution_dir: Path,
    execute: bool,
    max_rounds: int,
    allow_user_confirmation_actions: bool,
) -> RepairExecutionState:
    now = utc_timestamp()
    return RepairExecutionState(
        status=RepairExecutionStatus.RUNNING if execute else RepairExecutionStatus.PLANNED,
        execution_id=execution_id,
        source_state_path=str(source_state_path),
        execution_dir=str(execution_dir),
        created_at=now,
        updated_at=now,
        execute=execute,
        max_rounds=max_rounds,
        allow_user_confirmation_actions=allow_user_confirmation_actions,
        current_state_path=str(source_state_path),
        checkpoint={"completed_rounds": 0},
        next_action="build repair plan",
    )


def execute_repair_round(
    state: RepairExecutionState,
    state_path: Path,
    *,
    registry: dict[str, Runner] | None = None,
) -> RepairExecutionAttempt | None:
    source_path = Path(state.current_state_path)
    source_state = read_json(source_path)
    if is_accepted_state(source_state):
        state.status = RepairExecutionStatus.COMPLETED
        state.final_state_path = str(source_path)
        state.final_quality_status = quality_status_from_state(source_state)
        state.next_action = "accept repaired TCAD result"
        return None

    plan = build_repair_plan(source_path)
    state.repair_plan_path = plan.output_path
    tried_action_names = {attempt.action_name for attempt in state.attempts}
    state.checkpoint["tried_repair_actions"] = sorted(tried_action_names)
    action = choose_action(
        plan,
        allow_user_confirmation_actions=state.allow_user_confirmation_actions,
        tried_action_names=tried_action_names,
    )
    if action is None:
        if any(candidate.user_confirmation_required for candidate in plan.actions):
            state.status = RepairExecutionStatus.WAITING_FOR_USER
            state.next_action = "wait for user confirmation before applying sensitive repair action"
            state.checkpoint["blocked_repair_plan_path"] = plan.output_path
        elif not plan.actions:
            state.status = RepairExecutionStatus.FAILED
            state.failure_reason = "repair plan did not contain an executable action"
            state.next_action = "inspect run manually"
        else:
            state.status = RepairExecutionStatus.FAILED
            state.failure_reason = "repair plan actions were not executable by this executor"
            state.next_action = "extend repair executor registry or inspect manually"
        return None

    target_tool = action.target_tool or str(source_state.get("tool_name"))
    runners = registry or default_runner_registry()
    runner = runners.get(target_tool)
    round_index = len(state.attempts) + 1
    next_request = apply_patch_to_request(source_state, action, round_index)
    attempt = RepairExecutionAttempt(
        index=round_index,
        action_name=action.name,
        target_tool=target_tool,
        source_state_path=str(source_path),
        request_patch=action.request_patch,
        next_request=next_request,
        status=RepairExecutionStatus.PLANNED if not state.execute else RepairExecutionStatus.RUNNING,
        started_at=utc_timestamp(),
    )
    state.attempts.append(attempt)
    state.next_action = f"execute repair action {action.name}"
    write_state(state, state_path)

    if not state.execute:
        return attempt

    if runner is None:
        attempt.status = RepairExecutionStatus.FAILED
        attempt.failure_reason = f"no runner registered for target tool {target_tool}"
        attempt.completed_at = utc_timestamp()
        state.status = RepairExecutionStatus.FAILED
        state.failure_reason = attempt.failure_reason
        state.next_action = "extend repair executor runner registry"
        return attempt

    try:
        result = runner(next_request)
        attempt.result = result
        result_state_path = infer_result_state_path(result)
        attempt.result_state_path = result_state_path
        if result_state_path:
            state.current_state_path = result_state_path
            repaired_state = read_json(Path(result_state_path))
            attempt.quality_status = quality_status_from_state(repaired_state)
            if is_accepted_state(repaired_state):
                state.status = RepairExecutionStatus.COMPLETED
                state.final_state_path = result_state_path
                state.final_quality_status = attempt.quality_status
                state.next_action = "accept repaired TCAD result"
            else:
                state.status = RepairExecutionStatus.RUNNING
                state.final_quality_status = attempt.quality_status
                state.next_action = "build next repair plan from repaired run"
        else:
            attempt.quality_status = (result.get("quality_report") or {}).get("status")
            state.status = RepairExecutionStatus.FAILED
            state.failure_reason = "repair runner did not expose a result state path"
            state.next_action = "inspect repair runner output"
        attempt.status = RepairExecutionStatus.COMPLETED
    except Exception as exc:
        attempt.status = RepairExecutionStatus.FAILED
        attempt.failure_reason = str(exc)
        state.status = RepairExecutionStatus.RUNNING
        state.failure_reason = str(exc)
        state.next_action = "classify failed repair attempt and try next repair action"
    attempt.completed_at = utc_timestamp()
    state.checkpoint = {
        "completed_rounds": len(state.attempts),
        "last_attempt": attempt.model_dump(mode="json"),
    }
    return attempt


def run_repair_executor(
    source_state_path: Path,
    *,
    execution_id: str | None = None,
    execution_root: Path | None = None,
    execute: bool = False,
    resume: bool = False,
    max_rounds: int = 3,
    allow_user_confirmation_actions: bool = False,
    registry: dict[str, Runner] | None = None,
) -> RepairExecutionState:
    actual_execution_id = execution_id or default_execution_id()
    actual_root = execution_root or default_execution_root(source_state_path)
    execution_dir = actual_root / actual_execution_id
    actual_state_path = execution_dir / "repair_execution_state.json"

    if resume and actual_state_path.exists():
        state = load_state(actual_state_path)
        state.execute = execute
        state.max_rounds = max_rounds
        state.allow_user_confirmation_actions = allow_user_confirmation_actions
        state.status = RepairExecutionStatus.RUNNING if execute else RepairExecutionStatus.PLANNED
    else:
        execution_dir.mkdir(parents=True, exist_ok=True)
        state = create_initial_state(
            source_state_path,
            actual_execution_id,
            execution_dir,
            execute,
            max_rounds,
            allow_user_confirmation_actions,
        )
    write_state(state, actual_state_path)

    while len(state.attempts) < state.max_rounds and state.status in {
        RepairExecutionStatus.RUNNING,
        RepairExecutionStatus.PLANNED,
    }:
        execute_repair_round(state, actual_state_path, registry=registry)
        write_state(state, actual_state_path)
        if not state.execute:
            state.status = RepairExecutionStatus.PLANNED
            state.checkpoint["planned_attempt"] = state.attempts[-1].model_dump(mode="json") if state.attempts else None
            write_state(state, actual_state_path)
            return state
        if state.status in {
            RepairExecutionStatus.COMPLETED,
            RepairExecutionStatus.WAITING_FOR_USER,
            RepairExecutionStatus.FAILED,
        }:
            write_state(state, actual_state_path)
            return state

    if state.status == RepairExecutionStatus.RUNNING:
        state.status = RepairExecutionStatus.FAILED
        state.failure_reason = f"maximum repair rounds reached: {state.max_rounds}"
        state.next_action = "inspect last repair attempt and decide whether to extend budget"
    write_state(state, actual_state_path)
    return state
