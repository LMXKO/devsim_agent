from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.agent_cockpit import generate_agent_cockpit
from tcad_agent.autonomous_devsim_agent import (
    AutonomousDevsimRequest,
    AutonomousDevsimAgentState,
    ChatClient,
    DevsimAgentStatus,
    Runner,
    run_autonomous_devsim_agent,
    state_path as autonomous_state_path,
)
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
    }
    write_json(Path(state.heartbeat_path), payload)


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
    deadline = datetime.utcnow() + timedelta(hours=request.duration_hours)
    if request.resume and actual_state_path.exists():
        state = load_soak_state(actual_state_path)
        state.status = AgentSoakStatus.RUNNING
        state.request = request.model_dump(mode="json")
        state.deadline_at = deadline.replace(microsecond=0).isoformat() + "Z"
        state.cancel_file = str(cancel_file)
        state.heartbeat_path = str(heartbeat_path)
    else:
        state = AgentSoakState(
            status=AgentSoakStatus.RUNNING,
            soak_id=soak_id,
            soak_dir=str(soak_dir),
            created_at=utc_timestamp(),
            updated_at=utc_timestamp(),
            deadline_at=deadline.replace(microsecond=0).isoformat() + "Z",
            request=request.model_dump(mode="json"),
            agent_id=agent_id,
            agent_root=str(agent_root),
            heartbeat_path=str(heartbeat_path),
            cancel_file=str(cancel_file),
            next_action="start autonomous agent soak",
        )
    write_soak_state(state, actual_state_path)
    write_soak_heartbeat(state)

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
        )
        try:
            agent_state = run_autonomous_devsim_agent(
                AutonomousDevsimRequest.model_validate(payload),
                runner_registry=runner_registry,
                llm_client=llm_client,
            )
        except Exception as exc:
            state.status = AgentSoakStatus.FAILED
            state.failure_reason = str(exc)
            state.next_action = "inspect agent soak cycle failure"
            break
        agent_steps = len(agent_state.steps)
        model_decisions, fallback_decisions = count_decisions(agent_state)
        cockpit_path = None
        if agent_steps == 0 or agent_steps % request.cockpit_interval_steps == 0 or agent_state.status != DevsimAgentStatus.RUNNING:
            cockpit_path = maybe_generate_cockpit(request, soak_dir=soak_dir, cycle_index=cycle_index, agent_state_path=agent_state_file)
        terminal = terminal_soak_status(agent_state)
        cycle_status = terminal or (
            "slice_exhausted" if max_steps_exhausted(agent_state) else AgentSoakStatus.RUNNING
        )
        cycle_failure_reason = agent_state.failure_reason
        if cycle_status == AgentSoakStatus.COMPLETED:
            cycle_failure_reason = None
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
            )
        )
        state.agent_state_path = str(agent_state_file.resolve()) if agent_state_file.exists() else None
        state.latest_cockpit_path = cockpit_path or state.latest_cockpit_path
        state.final_agent_status = agent_state.status.value if isinstance(agent_state.status, DevsimAgentStatus) else str(agent_state.status)
        state.final_state_path = agent_state.final_state_path or agent_state.latest_state_path
        state.completed_steps = agent_steps
        state.model_decisions = model_decisions
        state.fallback_decisions = fallback_decisions
        state.next_action = agent_state.next_action
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

    write_soak_state(state, actual_state_path)
    write_soak_heartbeat(state)
    return state
