from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.engineering_intent import parse_engineering_intent
from tcad_agent.task_spec import PROJECT_ROOT


class AgentMemoryRecord(BaseModel):
    schema_version: str = "actsoft.tcad.agent_memory.v1"
    record_id: str
    created_at: str
    goal_text: str
    device_family: str = "unknown"
    template_id: str | None = None
    status: str
    outcome: str
    soak_id: str | None = None
    state_path: str | None = None
    final_state_path: str | None = None
    completed_steps: int = 0
    model_decisions: int = 0
    fallback_decisions: int = 0
    final_agent_status: str | None = None
    curve_guidance_summary: str | None = None
    recovery_summary: str | None = None
    next_action: str | None = None
    tags: list[str] = Field(default_factory=list)


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_agent_memory_path() -> Path:
    override = os.environ.get("ACTSOFT_AGENT_MEMORY_PATH")
    if override:
        return Path(override)
    return PROJECT_ROOT / "runs" / "agent_memory.jsonl"


def keyword_tokens(text: str) -> set[str]:
    lowered = text.lower()
    tokens = set(re.findall(r"[a-z0-9_+-]{2,}", lowered))
    cjk_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    tokens.update(cjk_chunks)
    return tokens


def read_agent_memory(*, memory_path: Path | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    path = memory_path or default_agent_memory_path()
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows[-limit:] if limit else rows


def retrieve_agent_memory(
    goal_text: str,
    *,
    memory_path: Path | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    rows = read_agent_memory(memory_path=memory_path)
    if not rows:
        return []
    intent = parse_engineering_intent(goal_text)
    goal_tokens = keyword_tokens(goal_text)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        score = 0.0
        if row.get("device_family") == intent.device_family and intent.device_family != "unknown":
            score += 4.0
        if row.get("template_id") == intent.template_id and intent.template_id:
            score += 3.0
        score += min(len(goal_tokens & keyword_tokens(str(row.get("goal_text") or ""))), 8) * 0.4
        if row.get("outcome") == "useful":
            score += 1.0
        if row.get("status") == "completed":
            score += 0.5
        if score > 0:
            ranked.append((score, row))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in ranked[:limit]]


def summarize_recovery_events(events: list[dict[str, Any]]) -> str | None:
    if not events:
        return None
    families: dict[str, int] = {}
    recovered = 0
    for event in events:
        family = str(event.get("family") or "unknown")
        families[family] = families.get(family, 0) + 1
        if event.get("recovered"):
            recovered += 1
    parts = [f"{key}:{value}" for key, value in sorted(families.items())]
    return f"recovery_events={len(events)}, recovered={recovered}, families={','.join(parts)}"


def summarize_curve_guidance(guidance: dict[str, Any] | None) -> str | None:
    if not isinstance(guidance, dict) or not guidance:
        return None
    action = guidance.get("recommended_action")
    target = guidance.get("recommended_target")
    reason = guidance.get("reason")
    return " ".join(str(item) for item in [action, target, reason] if item)


def build_agent_memory_record(
    soak_state: dict[str, Any],
    *,
    mission_spec: dict[str, Any] | None = None,
) -> AgentMemoryRecord:
    goal_text = str(soak_state.get("request", {}).get("goal_text") or soak_state.get("goal_text") or "")
    intent_data = (mission_spec or {}).get("intent") if isinstance(mission_spec, dict) else None
    if not isinstance(intent_data, dict):
        intent = parse_engineering_intent(goal_text)
        intent_data = intent.model_dump(mode="json")
    status = str(soak_state.get("status") or "unknown")
    useful = status in {"completed", "waiting_for_user", "max_steps_reached"} and bool(soak_state.get("final_state_path") or soak_state.get("agent_state_path"))
    tags = [status]
    if soak_state.get("model_decisions"):
        tags.append("model_driven")
    if soak_state.get("fallback_decisions"):
        tags.append("fallback_used")
    curve_guidance = soak_state.get("curve_guidance") if isinstance(soak_state.get("curve_guidance"), dict) else None
    recovery_events = soak_state.get("recovery_events") if isinstance(soak_state.get("recovery_events"), list) else []
    return AgentMemoryRecord(
        record_id=f"{soak_state.get('soak_id') or 'soak'}:{soak_state.get('updated_at') or utc_timestamp()}",
        created_at=utc_timestamp(),
        goal_text=goal_text,
        device_family=str(intent_data.get("device_family") or "unknown"),
        template_id=str(intent_data.get("template_id")) if intent_data.get("template_id") else None,
        status=status,
        outcome="useful" if useful else "failed_or_incomplete",
        soak_id=soak_state.get("soak_id"),
        state_path=soak_state.get("state_path"),
        final_state_path=soak_state.get("final_state_path"),
        completed_steps=int(soak_state.get("completed_steps") or 0),
        model_decisions=int(soak_state.get("model_decisions") or 0),
        fallback_decisions=int(soak_state.get("fallback_decisions") or 0),
        final_agent_status=soak_state.get("final_agent_status"),
        curve_guidance_summary=summarize_curve_guidance(curve_guidance),
        recovery_summary=summarize_recovery_events(recovery_events),
        next_action=soak_state.get("next_action"),
        tags=tags,
    )


def append_agent_memory_record(record: AgentMemoryRecord, *, memory_path: Path | None = None) -> str:
    path = memory_path or default_agent_memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")
    return str(path.resolve())


def append_agent_memory_from_soak(
    soak_state: dict[str, Any],
    *,
    mission_spec: dict[str, Any] | None = None,
    memory_path: Path | None = None,
) -> dict[str, Any]:
    record = build_agent_memory_record(soak_state, mission_spec=mission_spec)
    path = append_agent_memory_record(record, memory_path=memory_path)
    return {"memory_path": path, "record": record.model_dump(mode="json")}
