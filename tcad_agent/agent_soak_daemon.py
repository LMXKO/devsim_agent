from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.run_queue import (
    QueueStatus,
    Runner,
    default_queue_db_path,
    enqueue_run,
    get_item,
    list_items,
    run_queue_daemon,
)
from tcad_agent.task_spec import PROJECT_ROOT


class AgentSoakDaemonRequest(BaseModel):
    daemon_id: str | None = None
    goal_text: str
    queue_id: str | None = None
    queue_db_path: Path = Field(default_factory=default_queue_db_path)
    daemon_root: Path = PROJECT_ROOT / "runs" / "agent_soak_daemon"
    execute: bool = True
    duration_hours: float = Field(default=1.0, ge=0.0)
    max_steps: int = Field(default=40, ge=1)
    step_slice: int = Field(default=4, ge=1)
    use_llm: bool = True
    allow_llm_fallback: bool = True
    auto_execute_curve_guidance: bool = True
    max_curve_guided_patches: int = Field(default=1, ge=0)
    priority: int = 10
    max_attempts: int = 1
    owner: str = "agent_soak_daemon"
    concurrency: int = Field(default=1, ge=1)
    lease_seconds: float = Field(default=7200.0, gt=0.0)
    poll_interval_seconds: float = Field(default=5.0, ge=0.0)
    max_loops: int | None = None
    max_idle_loops: int | None = Field(default=1, ge=0)
    stop_file: Path | None = None
    autonomous_request: dict[str, Any] = Field(default_factory=dict)


class AgentSoakDaemonState(BaseModel):
    tool_name: str = "agent_soak_daemon"
    schema_version: str = "actsoft.tcad.agent_soak_daemon.v1"
    daemon_id: str
    status: str
    created_at: str
    updated_at: str
    queue_id: str
    queue_db_path: str
    state_path: str
    request: dict[str, Any]
    daemon_result: dict[str, Any] | None = None
    queue_item: dict[str, Any] | None = None
    lifecycle_events: list[dict[str, Any]] = Field(default_factory=list)
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_daemon_id() -> str:
    return f"agent_soak_daemon_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def state_path(root: Path, daemon_id: str) -> Path:
    return root / daemon_id / "agent_soak_daemon_state.json"


def write_state(state: AgentSoakDaemonState, path: Path) -> None:
    state.updated_at = utc_timestamp()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")


def append_event(state: AgentSoakDaemonState, event: str, *, detail: str | None = None, data: dict[str, Any] | None = None) -> None:
    state.lifecycle_events.append({"created_at": utc_timestamp(), "event": event, "detail": detail, "data": data or {}})
    state.lifecycle_events = state.lifecycle_events[-100:]


def queue_item_exists(db_path: Path, queue_id: str) -> bool:
    return get_item(db_path, queue_id) is not None


def build_soak_request(request: AgentSoakDaemonRequest, queue_id: str) -> dict[str, Any]:
    autonomous_request = dict(request.autonomous_request)
    autonomous_request.setdefault("use_llm", request.use_llm)
    autonomous_request.setdefault("allow_llm_fallback", request.allow_llm_fallback)
    return {
        "goal_text": request.goal_text,
        "soak_id": queue_id,
        "execute": request.execute,
        "duration_hours": request.duration_hours,
        "max_steps": request.max_steps,
        "step_slice": request.step_slice,
        "auto_execute_curve_guidance": request.auto_execute_curve_guidance,
        "max_curve_guided_patches": request.max_curve_guided_patches,
        "autonomous_request": autonomous_request,
    }


def run_agent_soak_daemon(
    request: AgentSoakDaemonRequest,
    *,
    registry: dict[str, Runner] | None = None,
) -> AgentSoakDaemonState:
    daemon_id = request.daemon_id or default_daemon_id()
    queue_id = request.queue_id or daemon_id
    actual_state_path = state_path(request.daemon_root, daemon_id).resolve()
    state = AgentSoakDaemonState(
        daemon_id=daemon_id,
        status="running",
        created_at=utc_timestamp(),
        updated_at=utc_timestamp(),
        queue_id=queue_id,
        queue_db_path=str(request.queue_db_path),
        state_path=str(actual_state_path),
        request=request.model_dump(mode="json"),
    )
    append_event(state, "start", detail="agent soak daemon started")
    write_state(state, actual_state_path)
    try:
        if not queue_item_exists(request.queue_db_path, queue_id):
            item = enqueue_run(
                request.queue_db_path,
                queue_id=queue_id,
                tool_name="agent_soak",
                request=build_soak_request(request, queue_id),
                priority=request.priority,
                tags=["agent_soak_daemon", "agent_soak"],
                max_attempts=request.max_attempts,
            )
            append_event(state, "enqueue", detail="agent_soak queue item enqueued", data={"queue_id": item.queue_id})
        else:
            append_event(state, "reuse_queue_item", detail="existing queue item reused", data={"queue_id": queue_id})
        stop_file = request.stop_file or (request.daemon_root / daemon_id / "stop.requested")
        daemon_result = run_queue_daemon(
            request.queue_db_path,
            owner=request.owner,
            concurrency=request.concurrency,
            lease_seconds=request.lease_seconds,
            poll_interval_seconds=request.poll_interval_seconds,
            max_loops=request.max_loops,
            max_idle_loops=request.max_idle_loops,
            stop_file=stop_file,
            registry=registry,
        )
        state.daemon_result = daemon_result.model_dump(mode="json")
        item = get_item(request.queue_db_path, queue_id)
        state.queue_item = item.model_dump(mode="json") if item else None
        if item and item.status == QueueStatus.COMPLETED:
            state.status = "completed"
        elif item and item.status == QueueStatus.PAUSED:
            state.status = "waiting_for_user"
        elif item and item.status == QueueStatus.FAILED:
            state.status = "failed"
            state.failure_reason = item.failure_reason
        elif daemon_result.stopped_by == "stop_file":
            state.status = "stopped"
        else:
            state.status = "idle"
        append_event(state, "daemon_result", detail=f"daemon stopped by {daemon_result.stopped_by}", data=state.daemon_result)
    except Exception as exc:
        state.status = "failed"
        state.failure_reason = str(exc)
        append_event(state, "failed", detail=str(exc))
    write_state(state, actual_state_path)
    return state


def list_soak_daemon_items(db_path: Path, *, limit: int = 20) -> list[dict[str, Any]]:
    return list_items(db_path, tool_name="agent_soak", limit=limit)


def parse_json_object(raw: str | None, flag: str) -> dict[str, object]:
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{flag} must decode to a JSON object")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an agent_soak queue daemon for a natural-language TCAD goal.")
    parser.add_argument("--goal", "--goal-text", dest="goal_text", required=True)
    parser.add_argument("--daemon-id", default=None)
    parser.add_argument("--queue-id", default=None)
    parser.add_argument("--db", type=Path, default=default_queue_db_path())
    parser.add_argument("--daemon-root", type=Path, default=PROJECT_ROOT / "runs" / "agent_soak_daemon")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--duration-hours", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--step-slice", type=int, default=4)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-llm-fallback", action="store_true")
    parser.add_argument("--no-auto-curve-guidance", action="store_true")
    parser.add_argument("--max-curve-guided-patches", type=int, default=1)
    parser.add_argument("--priority", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--owner", default="agent_soak_daemon")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--lease-seconds", type=float, default=7200.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--max-loops", type=int, default=None)
    parser.add_argument("--max-idle-loops", type=int, default=1)
    parser.add_argument("--stop-file", type=Path, default=None)
    parser.add_argument("--autonomous-request-json", default=None)
    return parser.parse_args()


def request_from_args(args: argparse.Namespace) -> AgentSoakDaemonRequest:
    return AgentSoakDaemonRequest(
        daemon_id=args.daemon_id,
        goal_text=args.goal_text,
        queue_id=args.queue_id,
        queue_db_path=args.db,
        daemon_root=args.daemon_root,
        execute=args.execute,
        duration_hours=args.duration_hours,
        max_steps=args.max_steps,
        step_slice=args.step_slice,
        use_llm=not args.no_llm,
        allow_llm_fallback=not args.no_llm_fallback,
        auto_execute_curve_guidance=not args.no_auto_curve_guidance,
        max_curve_guided_patches=args.max_curve_guided_patches,
        priority=args.priority,
        max_attempts=args.max_attempts,
        owner=args.owner,
        concurrency=args.concurrency,
        lease_seconds=args.lease_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        max_loops=args.max_loops,
        max_idle_loops=args.max_idle_loops,
        stop_file=args.stop_file,
        autonomous_request=parse_json_object(args.autonomous_request_json, "--autonomous-request-json"),
    )


def main() -> None:
    state = run_agent_soak_daemon(request_from_args(parse_args()))
    print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if state.status in {"completed", "waiting_for_user", "idle", "stopped"} else 1)


if __name__ == "__main__":
    main()
