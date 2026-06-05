from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.engineering_intent import EngineeringIntent, parse_engineering_intent


class LongHorizonSnapshot(BaseModel):
    goal_text: str
    engineering_intent: dict[str, Any]
    observation: dict[str, Any] = Field(default_factory=dict)
    policy_state: dict[str, Any] = Field(default_factory=dict)
    risk_ledger: list[dict[str, Any]] = Field(default_factory=list)
    pending_goal_kinds: list[str] = Field(default_factory=list)
    soft_failure_count: int = 0
    blocked_goal_steps: list[int] = Field(default_factory=list)
    replan_attempts: int = 0
    replan_max_attempts: int = 0


class LongHorizonDecision(BaseModel):
    action: str
    reason_zh: str
    risk_level: str
    should_replan: bool = False
    should_continue: bool = True
    needs_user: bool = False
    next_action_hint: str | None = None
    required_evidence: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    risk_ledger_updates: list[dict[str, Any]] = Field(default_factory=list)
    budget: dict[str, int] = Field(default_factory=dict)


def intent_from_checkpoint(goal_text: str, checkpoint: dict[str, Any]) -> EngineeringIntent:
    raw = checkpoint.get("engineering_intent")
    if isinstance(raw, dict):
        try:
            return EngineeringIntent.model_validate(raw)
        except Exception:
            pass
    return parse_engineering_intent(goal_text)


def evidence_matrix_from_observation(observation: dict[str, Any]) -> dict[str, Any]:
    benchmark = observation.get("physical_benchmark") or {}
    if isinstance(benchmark.get("summary"), dict):
        matrix = benchmark["summary"].get("evidence_matrix")
        if isinstance(matrix, dict):
            return matrix
    return {}


def build_long_horizon_snapshot(
    goal_text: str,
    checkpoint: dict[str, Any],
    observation: dict[str, Any] | None = None,
) -> LongHorizonSnapshot:
    actual_observation = observation or {}
    intent = intent_from_checkpoint(goal_text, checkpoint)
    return LongHorizonSnapshot(
        goal_text=goal_text,
        engineering_intent=intent.model_dump(mode="json"),
        observation=actual_observation,
        policy_state=dict(checkpoint.get("long_horizon_policy") or {}),
        risk_ledger=[
            item for item in (checkpoint.get("risk_ledger") or []) if isinstance(item, dict)
        ][-20:],
        pending_goal_kinds=[str(item) for item in actual_observation.get("pending_goal_kinds") or []],
        soft_failure_count=int(actual_observation.get("soft_failure_count") or 0),
        blocked_goal_steps=[
            int(item)
            for item in actual_observation.get("blocked_goal_steps") or []
            if isinstance(item, int) or (isinstance(item, str) and item.isdigit())
        ],
        replan_attempts=int(checkpoint.get("agent_replan_attempts") or 0),
        replan_max_attempts=int(checkpoint.get("agent_replan_max_attempts") or 0),
    )


def missing_evidence_for_intent(intent: EngineeringIntent, observation: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    matrix = evidence_matrix_from_observation(observation)
    benchmark = observation.get("physical_benchmark") or {}
    if "mesh_convergence" in intent.evidence_requirements:
        convergence = observation.get("tool_convergence") or {}
        if convergence.get("status") not in {"completed", "passed"} or convergence.get("quality_status") == "failed":
            missing.append("mesh_or_tool_convergence")
    if "golden_or_measured" in intent.evidence_requirements and matrix.get("measured_curve") != "present":
        missing.append("measured_or_golden_curve_comparison")
    if "engineering_signoff" in intent.evidence_requirements:
        if benchmark.get("status") not in {"passed"}:
            missing.append("passed_physical_benchmark")
        if matrix and matrix.get("convergence_evidence") != "present":
            missing.append("signoff_convergence_evidence")
    if "unit_check" in intent.evidence_requirements and not matrix:
        missing.append("unit_and_dimension_audit")
    return missing


def risk_updates(snapshot: LongHorizonSnapshot, missing_evidence: list[str]) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    observation = snapshot.observation
    primary = observation.get("primary_tcad_record") or {}
    if primary.get("quality_status") in {"failed", "suspicious"}:
        updates.append(
            {
                "kind": "primary_quality_risk",
                "severity": primary.get("quality_status"),
                "message_zh": "主 TCAD 结果质量未完全通过，后续结论需要带风险说明。",
                "state_path": primary.get("state_path"),
            }
        )
    benchmark = observation.get("physical_benchmark") or {}
    if benchmark.get("status") in {"failed", "suspicious"}:
        updates.append(
            {
                "kind": "physical_benchmark_risk",
                "severity": benchmark.get("status"),
                "message_zh": benchmark.get("failure_reason") or "物理 benchmark 标记了可信度风险。",
                "benchmark_path": benchmark.get("benchmark_path"),
            }
        )
    repair = observation.get("repair") or {}
    if repair.get("status") == "failed":
        updates.append(
            {
                "kind": "repair_exhausted",
                "severity": "warning",
                "message_zh": repair.get("failure_reason") or "自动修复没有得到可信结果。",
                "state_path": repair.get("state_path"),
            }
        )
    for item in missing_evidence:
        updates.append(
            {
                "kind": "missing_evidence",
                "severity": "warning",
                "message_zh": f"当前任务仍缺少证据：{item}。",
            }
        )
    return updates


def max_risk(left: str, right: str) -> str:
    rank = {"low": 0, "medium": 1, "high": 2}
    return left if rank.get(left, 1) >= rank.get(right, 1) else right


def decide_long_horizon_action(snapshot: LongHorizonSnapshot) -> LongHorizonDecision:
    intent = EngineeringIntent.model_validate(snapshot.engineering_intent)
    missing = missing_evidence_for_intent(intent, snapshot.observation)
    updates = risk_updates(snapshot, missing)
    budget = {
        "replan_attempts": snapshot.replan_attempts,
        "replan_max_attempts": snapshot.replan_max_attempts,
        "remaining_replans": max(snapshot.replan_max_attempts - snapshot.replan_attempts, 0),
    }
    if snapshot.blocked_goal_steps and budget["remaining_replans"] <= 0:
        return LongHorizonDecision(
            action="ask_user",
            reason_zh="仍有阻塞步骤，且自动重编排预算已用完，需要用户确认是否扩大预算或降低证据要求。",
            risk_level="high",
            should_continue=False,
            needs_user=True,
            required_evidence=intent.evidence_requirements,
            missing_evidence=missing,
            risk_ledger_updates=updates,
            budget=budget,
        )
    if snapshot.blocked_goal_steps or (snapshot.soft_failure_count and budget["remaining_replans"] > 0):
        return LongHorizonDecision(
            action="replan",
            reason_zh="检测到失败/软失败，先让总控基于当前证据重新编排，而不是直接终止任务。",
            risk_level=max_risk(intent.risk_level, "medium"),
            should_replan=True,
            next_action_hint="agent_replan",
            required_evidence=intent.evidence_requirements,
            missing_evidence=missing,
            risk_ledger_updates=updates,
            budget=budget,
        )
    if missing and "generate_conclusion" in snapshot.pending_goal_kinds:
        return LongHorizonDecision(
            action="continue_with_risk",
            reason_zh="证据仍不完整，但已进入结论阶段；继续生成带风险和下一轮建议的工程结论。",
            risk_level="high" if intent.risk_level == "high" else "medium",
            next_action_hint="generate_risk_aware_conclusion",
            required_evidence=intent.evidence_requirements,
            missing_evidence=missing,
            risk_ledger_updates=updates,
            budget=budget,
        )
    if missing:
        return LongHorizonDecision(
            action="repair_or_verify",
            reason_zh="当前工程意图要求更多可信证据，优先补 benchmark、收敛或模型对比。",
            risk_level="medium",
            next_action_hint="collect_missing_evidence",
            required_evidence=intent.evidence_requirements,
            missing_evidence=missing,
            risk_ledger_updates=updates,
            budget=budget,
        )
    return LongHorizonDecision(
        action="continue",
        reason_zh="当前观察未发现阻塞项，按长期计划继续执行下一步。",
        risk_level=intent.risk_level,
        required_evidence=intent.evidence_requirements,
        missing_evidence=missing,
        risk_ledger_updates=updates,
        budget=budget,
    )


def merge_risk_ledger(existing: list[dict[str, Any]], updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = list(existing)
    seen = {
        (
            str(item.get("kind")),
            str(item.get("message_zh")),
            str(item.get("state_path") or item.get("benchmark_path") or ""),
        )
        for item in merged
        if isinstance(item, dict)
    }
    for update in updates:
        key = (
            str(update.get("kind")),
            str(update.get("message_zh")),
            str(update.get("state_path") or update.get("benchmark_path") or ""),
        )
        if key in seen:
            continue
        merged.append(update)
        seen.add(key)
    return merged[-50:]
