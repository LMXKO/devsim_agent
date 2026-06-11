from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AgentRecoveryDecision(BaseModel):
    schema_version: str = "actsoft.tcad.agent_recovery_decision.v1"
    created_at: str
    family: str
    transient: bool = False
    should_retry: bool = False
    should_pause_for_user: bool = False
    recovered: bool = False
    attempt_index: int = 1
    max_attempts: int = 2
    reason: str
    request_patch: dict[str, Any] = Field(default_factory=dict)
    next_action: str


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def normalize_reason(reason: str | None) -> str:
    return str(reason or "").strip().lower()


def classify_failure_family(reason: str | None, *, agent_status: str | None = None, completed_steps: int = 0) -> tuple[str, bool, str]:
    text = normalize_reason(reason)
    status = normalize_reason(agent_status)
    if "connection error" in text or "timeout" in text or "temporarily unavailable" in text or "rate limit" in text:
        return "llm_transport", True, "Model transport failed before a safe decision was recorded."
    if "did not return json" in text or "invalid or unsafe" in text or "json" in text:
        return "llm_schema", True, "Model response schema was invalid for the agent action contract."
    if "maximum autonomous devsim steps reached" in text:
        return "step_slice_exhausted", False, "The inner agent exhausted the current step slice and can continue in the next slice."
    if "convergence" in text or "newton" in text or "linear solver" in text:
        return "simulator_convergence", False, "Simulator convergence failed; retry should favor smaller steps or repair planning."
    if "summary.json was not found" in text or "output_missing" in text or "missing" in text:
        return "output_missing", False, "The tool completed without the expected artifact."
    if "quality" in text or "suspicious" in text or status == "failed" and completed_steps > 0:
        return "physical_quality", False, "The latest TCAD state exists but needs repair, benchmark, or curve review."
    if "cancel" in text:
        return "cancelled", False, "User or queue requested cancellation."
    return "unknown", False, "Failure was not recognized; retry once before pausing or failing closed."


def count_family(events: list[dict[str, Any]], family: str) -> int:
    return sum(1 for event in events if isinstance(event, dict) and event.get("family") == family)


def build_recovery_decision(
    *,
    failure_reason: str | None,
    agent_status: str | None,
    completed_steps: int,
    autonomous_request: dict[str, Any],
    recovery_events: list[dict[str, Any]] | None = None,
    max_attempts: int = 2,
) -> AgentRecoveryDecision:
    family, transient, reason = classify_failure_family(
        failure_reason,
        agent_status=agent_status,
        completed_steps=completed_steps,
    )
    prior = count_family(recovery_events or [], family)
    attempt = prior + 1
    request_patch: dict[str, Any] = {}
    allow_llm_fallback = bool(autonomous_request.get("allow_llm_fallback", True))
    if family == "llm_schema" and allow_llm_fallback and attempt >= max_attempts:
        request_patch["use_llm"] = False
        request_patch["recovery_note"] = "LLM schema failed repeatedly; deterministic guardrail is allowed by fallback policy."
    if family == "simulator_convergence":
        request_patch["enable_experiment_design"] = True
        request_patch["auto_execute_experiment_design"] = True
        request_patch["recovery_note"] = "Convergence failure; favor smaller sweep/bias/mesh repair candidates."
    if family == "output_missing":
        request_patch["generate_report"] = False
        request_patch["recovery_note"] = "Missing output; rerun the same tool once before reporting."
    should_retry = family not in {"cancelled", "step_slice_exhausted"} and attempt <= max_attempts
    should_pause = family in {"unknown", "llm_schema"} and attempt > max_attempts and not request_patch
    next_action = (
        "continue next slice"
        if family == "step_slice_exhausted"
        else "retry recovered agent cycle"
        if should_retry
        else "pause for user review"
        if should_pause
        else "fail closed"
    )
    return AgentRecoveryDecision(
        created_at=utc_timestamp(),
        family=family,
        transient=transient,
        should_retry=should_retry,
        should_pause_for_user=should_pause,
        attempt_index=attempt,
        max_attempts=max_attempts,
        reason=reason,
        request_patch=request_patch,
        next_action=next_action,
    )
