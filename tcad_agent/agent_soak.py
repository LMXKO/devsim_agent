from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.agent_cockpit import generate_agent_cockpit
from tcad_agent.agent_curve_guidance import build_agent_curve_guidance
from tcad_agent.agent_guidance_patch import guidance_is_actionable_patch
from tcad_agent.agent_memory import append_agent_memory_from_soak, retrieve_agent_memory
from tcad_agent.agent_recovery import build_recovery_decision
from tcad_agent.autonomous_devsim_agent import (
    AutonomousDevsimRequest,
    AutonomousDevsimAgentState,
    ChatClient,
    DevsimAgentStatus,
    Runner,
    run_autonomous_devsim_agent,
    state_path as autonomous_state_path,
)
from tcad_agent.mission_spec_compiler import apply_mission_spec_to_autonomous_request, compile_mission_spec
from tcad_agent.task_spec import PROJECT_ROOT


TERMINAL_AGENT_STATUSES = {
    DevsimAgentStatus.COMPLETED.value,
    DevsimAgentStatus.WAITING_FOR_USER.value,
    DevsimAgentStatus.CANCELLED.value,
}


class AgentSoakStatus(str):
    RUNNING = "running"
    COMPLETED = "completed"
    WAITING_FOR_USER = "waiting_for_user"
    CANCELLED = "cancelled"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    MAX_STEPS_REACHED = "max_steps_reached"


class AgentSoakRequest(BaseModel):
    goal_text: str
    soak_id: str | None = None
    soak_root: Path = PROJECT_ROOT / "runs" / "agent_soak"
    execute: bool = True
    resume: bool = False
    duration_hours: float = Field(default=1.0, ge=0.0)
    max_steps: int = Field(default=40, ge=1)
    step_slice: int = Field(default=4, ge=1)
    poll_interval_seconds: float = Field(default=0.0, ge=0.0)
    agent_id: str | None = None
    agent_root: Path | None = None
    autonomous_request: dict[str, Any] = Field(default_factory=dict)
    cancel_file: Path | None = None
    heartbeat_path: Path | None = None
    generate_cockpit: bool = True
    cockpit_interval_steps: int = Field(default=1, ge=1)
    compile_mission_spec: bool = True
    enable_recovery: bool = True
    max_recovery_attempts: int = Field(default=2, ge=0)
    enable_agent_memory: bool = True
    memory_path: Path | None = None
    enable_curve_guidance: bool = True
    auto_execute_curve_guidance: bool = True
    max_curve_guided_patches: int = Field(default=1, ge=0)


class AgentSoakCycle(BaseModel):
    index: int
    status: str
    started_at: str
    completed_at: str
    requested_max_steps: int
    agent_status: str
    agent_steps: int
    new_steps: int
    model_decisions: int
    fallback_decisions: int
    agent_state_path: str | None = None
    cockpit_path: str | None = None
    failure_reason: str | None = None
    recovery_decision: dict[str, Any] | None = None
    curve_guidance: dict[str, Any] | None = None


class AgentSoakState(BaseModel):
    tool_name: str = "agent_soak"
    schema_version: str = "actsoft.tcad.agent_soak.v1"
    status: str
    soak_id: str
    soak_dir: str
    created_at: str
    updated_at: str
    deadline_at: str
    request: dict[str, Any]
    cycles: list[AgentSoakCycle] = Field(default_factory=list)
    agent_id: str
    agent_root: str
    agent_state_path: str | None = None
    heartbeat_path: str
    cancel_file: str
    latest_cockpit_path: str | None = None
    final_agent_status: str | None = None
    final_state_path: str | None = None
    completed_steps: int = 0
    model_decisions: int = 0
    fallback_decisions: int = 0
    next_action: str | None = None
    failure_reason: str | None = None
    state_path: str | None = None
    mission_spec: dict[str, Any] | None = None
    agent_memory_context: list[dict[str, Any]] = Field(default_factory=list)
    memory_record_path: str | None = None
    memory_record: dict[str, Any] | None = None
    recovery_events: list[dict[str, Any]] = Field(default_factory=list)
    curve_guidance: dict[str, Any] | None = None
    lifecycle_events: list[dict[str, Any]] = Field(default_factory=list)
    curve_guided_patch_runs: int = 0


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_soak_id() -> str:
    return f"agent_soak_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def state_path(soak_root: Path, soak_id: str) -> Path:
    return soak_root / soak_id / "agent_soak_state.json"


def load_soak_state(path: Path) -> AgentSoakState:
    return AgentSoakState.model_validate_json(path.read_text(encoding="utf-8"))


def write_soak_state(state: AgentSoakState, path: Path) -> None:
    state.updated_at = utc_timestamp()
    state.state_path = str(path.resolve())
    write_json(path, state.model_dump(mode="json"))


def write_soak_heartbeat(state: AgentSoakState) -> None:
    payload = {
        "schema_version": "actsoft.tcad.agent_soak_heartbeat.v1",
        "soak_id": state.soak_id,
        "status": state.status,
        "updated_at": utc_timestamp(),
        "completed_steps": state.completed_steps,
        "model_decisions": state.model_decisions,
        "fallback_decisions": state.fallback_decisions,
        "agent_state_path": state.agent_state_path,
        "latest_cockpit_path": state.latest_cockpit_path,
        "next_action": state.next_action,
        "failure_reason": state.failure_reason,
        "lifecycle_tail": state.lifecycle_events[-3:],
        "curve_guidance": state.curve_guidance,
    }
    write_json(Path(state.heartbeat_path), payload)


def append_lifecycle(state: AgentSoakState, event: str, *, detail: str | None = None, data: dict[str, Any] | None = None) -> None:
    state.lifecycle_events.append(
        {
            "created_at": utc_timestamp(),
            "event": event,
            "detail": detail,
            "data": data or {},
        }
    )
    state.lifecycle_events = state.lifecycle_events[-120:]


def count_decisions(agent_state: AutonomousDevsimAgentState) -> tuple[int, int]:
    ledger = agent_state.checkpoint.get("agent_decision_ledger")
    items = ledger if isinstance(ledger, list) else []
    fallback = sum(1 for item in items if isinstance(item, dict) and item.get("fallback_used"))
    model = sum(1 for item in items if isinstance(item, dict) and not item.get("fallback_used"))
    return model, fallback


def max_steps_exhausted(agent_state: AutonomousDevsimAgentState) -> bool:
    return (
        agent_state.status == DevsimAgentStatus.FAILED
        and str(agent_state.failure_reason or "").startswith("maximum autonomous DEVSIM steps reached")
    )


def guidance_patch_runs(agent_state: AutonomousDevsimAgentState) -> int:
    try:
        return int(agent_state.checkpoint.get("guidance_patch_runs") or 0)
    except (TypeError, ValueError):
        return 0


def terminal_soak_status(agent_state: AutonomousDevsimAgentState) -> str | None:
    status = agent_state.status.value if isinstance(agent_state.status, DevsimAgentStatus) else str(agent_state.status)
    if status == DevsimAgentStatus.COMPLETED.value:
        return AgentSoakStatus.COMPLETED
    if status == DevsimAgentStatus.WAITING_FOR_USER.value:
        return AgentSoakStatus.WAITING_FOR_USER
    if status == DevsimAgentStatus.CANCELLED.value:
        return AgentSoakStatus.CANCELLED
    if status == DevsimAgentStatus.FAILED.value and not max_steps_exhausted(agent_state):
        return AgentSoakStatus.FAILED
    return None


def build_autonomous_payload(
    request: AgentSoakRequest,
    *,
    soak_dir: Path,
    agent_id: str,
    agent_root: Path,
    max_steps: int,
    resume: bool,
    cancel_file: Path,
    mission_spec: dict[str, Any] | None = None,
    agent_memory_context: list[dict[str, Any]] | None = None,
    curve_guidance: dict[str, Any] | None = None,
    recovery_context: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = dict(request.autonomous_request)
    payload.update(
        {
            "goal_text": request.goal_text,
            "agent_id": agent_id,
            "agent_root": agent_root,
            "execute": request.execute,
            "resume": resume,
            "max_steps": max_steps,
            "cancel_file": cancel_file,
            "heartbeat_path": soak_dir / "agent_heartbeat.json",
        }
    )
    if mission_spec:
        payload["mission_spec"] = mission_spec
    if agent_memory_context:
        payload["agent_memory_context"] = agent_memory_context
    if curve_guidance:
        payload["curve_guidance"] = curve_guidance
    if recovery_context:
        payload["recovery_context"] = recovery_context[-8:]
    return payload


def maybe_generate_cockpit(
    request: AgentSoakRequest,
    *,
    soak_dir: Path,
    cycle_index: int,
    agent_state_path: Path,
) -> str | None:
    if not request.generate_cockpit or not agent_state_path.exists():
        return None
    cycle_path = soak_dir / "cockpit" / f"agent_cockpit_cycle_{cycle_index:03d}.html"
    latest_path = soak_dir / "agent_cockpit.html"
    generate_agent_cockpit(agent_state_path, cycle_path)
    generate_agent_cockpit(agent_state_path, latest_path)
    return str(latest_path.resolve())


def run_agent_soak(
    request: AgentSoakRequest,
    *,
    runner_registry: dict[str, Runner] | None = None,
    llm_client: ChatClient | None = None,
) -> AgentSoakState:
    soak_id = request.soak_id or default_soak_id()
    soak_dir = (request.soak_root / soak_id).resolve()
    soak_dir.mkdir(parents=True, exist_ok=True)
    actual_state_path = state_path(request.soak_root, soak_id).resolve()
    agent_root = (request.agent_root or soak_dir / "agents").resolve()
    agent_id = request.agent_id or soak_id
    cancel_file = (request.cancel_file or soak_dir / "cancel.requested").resolve()
    heartbeat_path = (request.heartbeat_path or soak_dir / "agent_soak_heartbeat.json").resolve()
    memory_context = retrieve_agent_memory(request.goal_text, memory_path=request.memory_path) if request.enable_agent_memory else []
    mission_spec = compile_mission_spec(request.goal_text, memory_records=memory_context) if request.compile_mission_spec else None
    effective_autonomous_request = dict(request.autonomous_request)
    if mission_spec:
        effective_autonomous_request = apply_mission_spec_to_autonomous_request(effective_autonomous_request, mission_spec)
    effective_request = request.model_copy(deep=True)
    effective_request.autonomous_request = effective_autonomous_request
    deadline = datetime.utcnow() + timedelta(hours=request.duration_hours)
    if request.resume and actual_state_path.exists():
        state = load_soak_state(actual_state_path)
        state.status = AgentSoakStatus.RUNNING
        state.request = effective_request.model_dump(mode="json")
        state.deadline_at = deadline.replace(microsecond=0).isoformat() + "Z"
        state.cancel_file = str(cancel_file)
        state.heartbeat_path = str(heartbeat_path)
        state.mission_spec = mission_spec.model_dump(mode="json") if mission_spec else state.mission_spec
        state.agent_memory_context = memory_context or state.agent_memory_context
        append_lifecycle(state, "resume", detail="agent soak resumed with refreshed mission context")
    else:
        state = AgentSoakState(
            status=AgentSoakStatus.RUNNING,
            soak_id=soak_id,
            soak_dir=str(soak_dir),
            created_at=utc_timestamp(),
            updated_at=utc_timestamp(),
            deadline_at=deadline.replace(microsecond=0).isoformat() + "Z",
            request=effective_request.model_dump(mode="json"),
            agent_id=agent_id,
            agent_root=str(agent_root),
            heartbeat_path=str(heartbeat_path),
            cancel_file=str(cancel_file),
            next_action="start autonomous agent soak",
            mission_spec=mission_spec.model_dump(mode="json") if mission_spec else None,
            agent_memory_context=memory_context,
        )
        append_lifecycle(
            state,
            "start",
            detail="agent soak started",
            data={
                "mission_spec_status": mission_spec.status if mission_spec else None,
                "memory_matches": len(memory_context),
            },
        )
    write_soak_state(state, actual_state_path)
    write_soak_heartbeat(state)

    request = effective_request
    started = time.monotonic()
    duration_seconds = request.duration_hours * 3600.0
    while state.status == AgentSoakStatus.RUNNING:
        if cancel_file.exists():
            state.status = AgentSoakStatus.CANCELLED
            state.failure_reason = "cancel requested by soak control file"
            state.next_action = "cancelled before next soak cycle"
            break
        if duration_seconds and time.monotonic() - started >= duration_seconds:
            state.status = AgentSoakStatus.TIMED_OUT
            state.next_action = "resume soak or increase duration_hours"
            break
        agent_state_file = autonomous_state_path(agent_root, agent_id)
        existing_steps = 0
        if agent_state_file.exists():
            try:
                existing_steps = len(AutonomousDevsimAgentState.model_validate_json(agent_state_file.read_text(encoding="utf-8")).steps)
            except Exception:
                existing_steps = state.completed_steps
        if existing_steps >= request.max_steps:
            state.status = AgentSoakStatus.MAX_STEPS_REACHED
            state.next_action = "increase max_steps or inspect final agent state"
            break
        cycle_index = len(state.cycles) + 1
        requested_max_steps = min(request.max_steps, max(existing_steps + request.step_slice, existing_steps + 1))
        cycle_started = utc_timestamp()
        payload = build_autonomous_payload(
            request,
            soak_dir=soak_dir,
            agent_id=agent_id,
            agent_root=agent_root,
            max_steps=requested_max_steps,
            resume=agent_state_file.exists() or request.resume,
            cancel_file=cancel_file,
            mission_spec=state.mission_spec,
            agent_memory_context=state.agent_memory_context,
            curve_guidance=state.curve_guidance,
            recovery_context=state.recovery_events,
        )
        try:
            agent_state = run_autonomous_devsim_agent(
                AutonomousDevsimRequest.model_validate(payload),
                runner_registry=runner_registry,
                llm_client=llm_client,
            )
        except Exception as exc:
            recovery = build_recovery_decision(
                failure_reason=str(exc),
                agent_status="failed",
                completed_steps=state.completed_steps,
                autonomous_request=request.autonomous_request,
                recovery_events=state.recovery_events,
                max_attempts=request.max_recovery_attempts,
            ).model_dump(mode="json")
            state.recovery_events.append(recovery)
            retrying = request.enable_recovery and bool(recovery.get("should_retry"))
            state.cycles.append(
                AgentSoakCycle(
                    index=cycle_index,
                    status="recovered_retry" if retrying else "failed",
                    started_at=cycle_started,
                    completed_at=utc_timestamp(),
                    requested_max_steps=requested_max_steps,
                    agent_status="failed",
                    agent_steps=state.completed_steps,
                    new_steps=0,
                    model_decisions=state.model_decisions,
                    fallback_decisions=state.fallback_decisions,
                    agent_state_path=str(agent_state_file.resolve()) if agent_state_file.exists() else None,
                    failure_reason=str(exc),
                    recovery_decision=recovery,
                )
            )
            append_lifecycle(state, "recovery", detail=str(recovery.get("reason") or ""), data=recovery)
            if retrying:
                request.autonomous_request.update(recovery.get("request_patch") or {})
                state.request = request.model_dump(mode="json")
                state.next_action = str(recovery.get("next_action") or "retry recovered agent cycle")
                write_soak_state(state, actual_state_path)
                write_soak_heartbeat(state)
                continue
            state.status = AgentSoakStatus.WAITING_FOR_USER if recovery.get("should_pause_for_user") else AgentSoakStatus.FAILED
            state.failure_reason = str(exc)
            state.next_action = str(recovery.get("next_action") or "inspect agent soak cycle failure")
            break
        agent_steps = len(agent_state.steps)
        model_decisions, fallback_decisions = count_decisions(agent_state)
        cockpit_path = None
        if agent_steps == 0 or agent_steps % request.cockpit_interval_steps == 0 or agent_state.status != DevsimAgentStatus.RUNNING:
            cockpit_path = maybe_generate_cockpit(request, soak_dir=soak_dir, cycle_index=cycle_index, agent_state_path=agent_state_file)
        terminal = terminal_soak_status(agent_state)
        curve_guidance = None
        latest_state_for_guidance = agent_state.final_state_path or agent_state.latest_state_path
        if request.enable_curve_guidance and latest_state_for_guidance:
            curve_guidance = build_agent_curve_guidance(
                goal_text=request.goal_text,
                source_state_path=latest_state_for_guidance,
            ).model_dump(mode="json")
        curve_guided_runs = guidance_patch_runs(agent_state)
        can_continue_from_guidance = (
            request.auto_execute_curve_guidance
            and request.enable_curve_guidance
            and bool(curve_guidance)
            and guidance_is_actionable_patch(curve_guidance)
            and curve_guided_runs < request.max_curve_guided_patches
            and agent_steps < request.max_steps
        )
        cycle_status = terminal or (
            "slice_exhausted" if max_steps_exhausted(agent_state) else AgentSoakStatus.RUNNING
        )
        if terminal == AgentSoakStatus.COMPLETED and can_continue_from_guidance:
            cycle_status = "curve_guidance_continue"
            terminal = None
        cycle_failure_reason = agent_state.failure_reason
        if cycle_status == AgentSoakStatus.COMPLETED:
            cycle_failure_reason = None
        recovery_decision = None
        if terminal == AgentSoakStatus.FAILED:
            recovery_decision = build_recovery_decision(
                failure_reason=agent_state.failure_reason,
                agent_status=agent_state.status.value if isinstance(agent_state.status, DevsimAgentStatus) else str(agent_state.status),
                completed_steps=agent_steps,
                autonomous_request=request.autonomous_request,
                recovery_events=state.recovery_events,
                max_attempts=request.max_recovery_attempts,
            ).model_dump(mode="json")
            state.recovery_events.append(recovery_decision)
            if request.enable_recovery and recovery_decision.get("should_retry"):
                cycle_status = "recovered_retry"
        state.cycles.append(
            AgentSoakCycle(
                index=cycle_index,
                status=cycle_status,
                started_at=cycle_started,
                completed_at=utc_timestamp(),
                requested_max_steps=requested_max_steps,
                agent_status=agent_state.status.value if isinstance(agent_state.status, DevsimAgentStatus) else str(agent_state.status),
                agent_steps=agent_steps,
                new_steps=max(agent_steps - existing_steps, 0),
                model_decisions=model_decisions,
                fallback_decisions=fallback_decisions,
                agent_state_path=str(agent_state_file.resolve()) if agent_state_file.exists() else None,
                cockpit_path=cockpit_path,
                failure_reason=cycle_failure_reason,
                recovery_decision=recovery_decision,
                curve_guidance=curve_guidance,
            )
        )
        state.agent_state_path = str(agent_state_file.resolve()) if agent_state_file.exists() else None
        state.latest_cockpit_path = cockpit_path or state.latest_cockpit_path
        state.final_agent_status = agent_state.status.value if isinstance(agent_state.status, DevsimAgentStatus) else str(agent_state.status)
        state.final_state_path = agent_state.final_state_path or agent_state.latest_state_path
        state.curve_guidance = curve_guidance or state.curve_guidance
        state.curve_guided_patch_runs = curve_guided_runs
        state.completed_steps = agent_steps
        state.model_decisions = model_decisions
        state.fallback_decisions = fallback_decisions
        state.next_action = agent_state.next_action
        append_lifecycle(
            state,
            "cycle",
            detail=f"cycle {cycle_index} {cycle_status}",
            data={
                "agent_status": state.final_agent_status,
                "new_steps": max(agent_steps - existing_steps, 0),
                "curve_guidance": curve_guidance,
            },
        )
        if recovery_decision and request.enable_recovery and recovery_decision.get("should_retry"):
            request.autonomous_request.update(recovery_decision.get("request_patch") or {})
            state.request = request.model_dump(mode="json")
            append_lifecycle(state, "recovery", detail=str(recovery_decision.get("reason") or ""), data=recovery_decision)
            state.next_action = str(recovery_decision.get("next_action") or "retry recovered agent cycle")
            write_soak_state(state, actual_state_path)
            write_soak_heartbeat(state)
            continue
        if can_continue_from_guidance:
            append_lifecycle(
                state,
                "curve_guidance_continue",
                detail="curve guidance will drive the next autonomous patch slice",
                data={
                    "guidance_action": (curve_guidance or {}).get("recommended_action"),
                    "guidance_target": (curve_guidance or {}).get("recommended_target"),
                    "curve_guided_patch_runs": curve_guided_runs,
                },
            )
            state.next_action = "execute curve-guided patch in next soak cycle"
            write_soak_state(state, actual_state_path)
            write_soak_heartbeat(state)
            continue
        if terminal:
            state.status = terminal
            state.failure_reason = None if terminal == AgentSoakStatus.COMPLETED else agent_state.failure_reason
        elif max_steps_exhausted(agent_state) and agent_steps >= request.max_steps:
            state.status = AgentSoakStatus.MAX_STEPS_REACHED
            state.failure_reason = agent_state.failure_reason
            state.next_action = "increase max_steps or inspect final agent state"
        write_soak_state(state, actual_state_path)
        write_soak_heartbeat(state)
        if state.status != AgentSoakStatus.RUNNING:
            break
        if request.poll_interval_seconds:
            time.sleep(request.poll_interval_seconds)

    if request.enable_agent_memory and state.status != AgentSoakStatus.RUNNING and not state.memory_record:
        memory_result = append_agent_memory_from_soak(
            state.model_dump(mode="json"),
            mission_spec=state.mission_spec,
            memory_path=request.memory_path,
        )
        state.memory_record_path = memory_result.get("memory_path")
        state.memory_record = memory_result.get("record")
        append_lifecycle(state, "memory", detail="agent memory record appended", data={"memory_path": state.memory_record_path})
    write_soak_state(state, actual_state_path)
    write_soak_heartbeat(state)
    return state
