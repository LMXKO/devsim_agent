from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from tcad_agent.llm import LLMClient, LLMConfig
from tcad_agent.repair_memory import recent_repair_case_memory
from tcad_agent.repair_strategy import (
    RepairAction,
    RepairPlan,
    build_repair_plan,
    failure_classes,
    issue_codes,
    quality_status,
    repair_request,
    repair_target_tool,
)
from tcad_agent.task_planner import parse_json_object
from tcad_agent.tcad_deck import compact_tcad_deck_spec


class ChatClient(Protocol):
    config: LLMConfig

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        ...


class RepairAgentStatus(str, Enum):
    COMPLETED = "completed"
    FALLBACK = "fallback"
    FAILED = "failed"


class RepairAgentDecision(BaseModel):
    schema_version: str = "actsoft.tcad.repair_agent_decision.v1"
    status: RepairAgentStatus
    state_path: str
    model: str | None = None
    action: RepairAction | None = None
    raw_response: str | None = None
    parsed_response: dict[str, Any] | None = None
    prompt_context: dict[str, Any] = Field(default_factory=dict)
    hypothesis_zh: str | None = None
    observation_summary: str | None = None
    tool_plan: list[dict[str, Any]] = Field(default_factory=list)
    safety_review: dict[str, Any] = Field(default_factory=dict)
    fallback_used: bool = False
    fallback_action_name: str | None = None
    warnings: list[str] = Field(default_factory=list)
    failure_reason: str | None = None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_dict(data: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: data.get(key) for key in keys if key in data}


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    preferred = [
        "leakage_current_a",
        "leakage_abs_current_at_target_a",
        "breakdown_voltage_v",
        "breakdown_voltage_at_threshold_v",
        "specific_on_resistance_ohm_cm2",
        "max_electric_field_v_per_cm",
        "field_peak_location_um",
        "field_peak_voltage_v",
        "breakdown_bracket_v",
        "leakage_interval_a",
        "curve_shape_summary",
        "curve_shape_monotonic_abs_y_violations",
        "idvd_kink_slope_jumps",
        "idvd_negative_differential_segments",
        "ion_ioff_ratio",
        "ioff_current_a",
        "vth_at_threshold_current_v",
        "subthreshold_swing_mv_dec",
    ]
    output = compact_dict(metrics, preferred)
    if not output:
        for key, value in list(metrics.items())[:24]:
            if isinstance(value, (str, int, float, bool, type(None), list, dict)):
                output[key] = value
    return output


def final_metrics(state: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(((state.get("final_summary") or {}).get("metrics") or {}))
    metrics.update(((state.get("quality_report") or {}).get("metrics") or {}))
    return metrics


def repair_agent_context(state_path: Path, deterministic_plan: RepairPlan | None = None) -> dict[str, Any]:
    state = read_json(state_path)
    request = repair_request(state)
    quality = state.get("quality_report") or state.get("final_quality_report") or {}
    deck_spec = state.get("tcad_deck_spec") or request.get("tcad_deck_spec")
    artifacts = ((state.get("final_summary") or {}).get("artifacts") or {})
    deterministic_actions = [
        {
            "name": action.name,
            "reason": action.reason,
            "request_patch": action.request_patch,
            "deck_patch": action.deck_patch,
            "expected_effect": action.expected_effect,
            "requires_user_confirmation": action.user_confirmation_required,
        }
        for action in (deterministic_plan.actions if deterministic_plan else [])[:8]
    ]
    return {
        "state_path": str(state_path),
        "tool_name": state.get("tool_name"),
        "target_tool": repair_target_tool(state),
        "status": state.get("status"),
        "run_id": state.get("run_id") or state.get("task_id") or state.get("convergence_id"),
        "quality_status": quality_status(state),
        "issue_codes": sorted(issue_codes(state)),
        "failure_classes": failure_classes(state),
        "quality_issues": (quality.get("issues") or [])[:12],
        "metrics": compact_metrics(final_metrics(state)),
        "request": request,
        "tcad_deck_spec": compact_tcad_deck_spec(deck_spec),
        "tcad_deck_mutations": request.get("tcad_deck_mutations") or state.get("tcad_deck_mutations") or [],
        "deck_patch_history": request.get("deck_patch_history") or [],
        "mutation_effect_analysis": state.get("mutation_effect_analysis"),
        "repair_context": state.get("repair_context"),
        "artifact_paths": compact_dict(
            artifacts,
            [
                "csv",
                "plot",
                "log",
                "tcad_deck_ir",
                "deck_patch_history",
                "semantic_deck_diff",
                "patched_source_deck",
                "baseline_mutation_overlay",
            ],
        ),
        "deterministic_fallback_actions": deterministic_actions,
        "agent_toolbelt": [
            {
                "tool": "deck_ir.semantic_patch",
                "purpose": "parse user deck, locate geometry/model/bias/mesh/doping sections, and produce a diff before execution",
            },
            {
                "tool": "curve_diagnostics.overlay",
                "purpose": "compare baseline vs mutation curve shape, BV bracket, leakage interval, and field peak",
            },
            {
                "tool": "physical_benchmark",
                "purpose": "gate physics consistency, model coupling, and compact-vs-TCAD capability boundaries",
            },
            {
                "tool": "tool_convergence_or_mesh_convergence",
                "purpose": "separate numerical/bias/mesh artifacts from physical trends",
            },
            {
                "tool": "engineering_objectives",
                "purpose": "evaluate Pareto feasibility and hard constraints over leakage, BV, Ron, and field",
            },
            {
                "tool": "golden_curve_comparison",
                "purpose": "compare with measured/trusted curves before signoff claims",
            },
        ],
        "repair_case_memory": recent_repair_case_memory(limit=12),
    }


def build_repair_agent_messages(context: dict[str, Any]) -> tuple[str, str]:
    system = (
        "你是一个 TCAD repair inner-loop agent，不是规则表。"
        "你的任务是观察现有 state、deck IR/patch lineage、曲线形状、baseline-vs-mutation 结果和 Pareto 约束，"
        "主动提出下一次最有信息量、最小风险的实验或 deck patch。只返回 JSON。"
        "不要输出 shell 命令。不要编造工具不存在的 runner。"
        "可以利用 deterministic_fallback_actions，但不能机械照抄；必须说明你基于哪些证据做判断。"
        "如果改 geometry/process/model 有真实工程风险，把 requires_user_confirmation 设为 true。"
        "如果上一轮 mutation_effect_analysis 显示有用，就沿有效方向做更细 patch；如果 tradeoff 变坏，切换 target 或要求 Pareto review。"
        "你需要显式写出 observation、hypothesis、tool_plan、safety_review，再给 action。"
        "领域策略：BV/Ron 是天然 tradeoff；field crowding 优先看 termination/field plate/guard ring/trench radius；"
        "漏电先区分 SRH/lifetime、trap、边界条件、mesh/bias artifact；击穿要看 bracket 和 avalanche onset；"
        "曲线 kink 或非单调先局部 bias/mesh 验证，不能直接调物理参数；"
        "compact/surrogate 不能当 signoff，golden/measured 缺失不能强结论。"
    )
    user = {
        "task": "agent-driven TCAD repair decision",
        "response_schema": {
            "action": {
                "name": "短 action id，例如 agent_refine_field_plate",
                "priority": "建议 120 以上，让 agent action 优先",
                "reason": "中文，一句话说明证据链",
                "target_tool": "目标工具名；默认使用 context.target_tool",
                "request_patch": "object；只写本轮要改的请求字段",
                "deck_patch": {
                    "operation": "set/sweep/refine/reject/review",
                    "request_path": "被修改的 request 字段",
                    "deck_path": "语义 deck 路径",
                    "value": "新值",
                    "target": "field_plate/drift_doping/lifetime/...",
                    "agent_rationale": "为什么这个 patch 值值得试",
                },
                "deck_mutations": "list，可为空；若使用已有 mutation，请放入对应 schema",
                "checklist": ["执行后必须检查的曲线/指标/约束"],
                "expected_effect": "预期改善什么，以及可能牺牲什么",
                "user_confirmation_required": "boolean",
            },
            "observation_summary": "你观察到的关键证据，中文",
            "hypothesis_zh": "当前失败/退化的工程假设，中文",
            "tool_plan": [
                {
                    "tool": "agent_toolbelt 里的工具名",
                    "why": "为什么下一轮或本轮后需要它",
                    "expected_evidence": "希望它证明/否定什么",
                }
            ],
            "safety_review": {
                "risk_level": "low/medium/high",
                "requires_user_confirmation": "boolean",
                "blocked_reason": "如果不应自动 patch，说明原因",
                "constraints_checked": ["BV/Ron/field/leakage/golden/signoff 等"],
            },
            "evidence_used": ["列出你真正用到的 evidence key"],
            "reject_reason": "如果不应继续自动 patch，说明原因",
        },
        "guardrails": [
            "只能提出一个下一步 action。",
            "request_patch 不要包含 shell command、文件删除、git 操作或任意代码。",
            "deck_patch 有 request_path/value 时，request_patch 必须同步设置同一字段。",
            "不要为了降低漏电牺牲超过 10% BV，除非用户目标明确允许。",
            "不要为了降低 field peak 牺牲超过 20% Ron，除非用户目标明确允许。",
            "如果信息不足，选择最小信息增益实验，而不是泛泛要求人工查看。",
            "高风险 geometry/process/model patch 必须要求用户确认，除非只是 plan-only 或已有用户明确授权。",
            "如果 agent 决定用规则 fallback，也要说明为什么 fallback 是当前最小风险动作。",
        ],
        "context": context,
    }
    return system, json.dumps(user, ensure_ascii=False, indent=2)


def fallback_action(plan: RepairPlan) -> RepairAction | None:
    return plan.actions[0] if plan.actions else None


def normalize_agent_action(parsed: dict[str, Any], context: dict[str, Any], warnings: list[str]) -> RepairAction | None:
    raw = parsed.get("action") or parsed.get("repair_action") or parsed
    if not isinstance(raw, dict):
        warnings.append("agent response did not contain an action object")
        return None
    if raw.get("next_tool_command") or raw.get("command"):
        warnings.append("ignored agent shell command; repair agent may only return structured patches")
    request_patch = raw.get("request_patch") if isinstance(raw.get("request_patch"), dict) else {}
    deck_patch = raw.get("deck_patch") if isinstance(raw.get("deck_patch"), dict) else {}
    request_path = deck_patch.get("request_path")
    if isinstance(request_path, str) and request_path and "value" in deck_patch:
        request_patch = {**request_patch, request_path: deck_patch.get("value")}
    deck_mutations = raw.get("deck_mutations") if isinstance(raw.get("deck_mutations"), list) else []
    confirmation_required = bool(raw.get("user_confirmation_required") or raw.get("requires_user_confirmation"))
    safety = parsed.get("safety_review") if isinstance(parsed.get("safety_review"), dict) else {}
    if safety.get("requires_user_confirmation"):
        confirmation_required = True
    high_risk_targets = {"guard_ring", "junction_depth", "oxide_thickness", "implant_dose", "trench_corner_radius", "trap_density", "region_lifetime"}
    deck_target = str(deck_patch.get("target") or "")
    deck_path = str(deck_patch.get("deck_path") or "")
    if deck_target in high_risk_targets or any(token in deck_path for token in ["geometry.", "doping.", "physics_models.trap"]):
        confirmation_required = True
        warnings.append("agent action touches high-risk geometry/process/model fields; user confirmation required")
    if not request_patch and not deck_patch and not deck_mutations and not confirmation_required:
        warnings.append("agent action had no executable request/deck patch")
        return None
    checklist = raw.get("checklist") if isinstance(raw.get("checklist"), list) else []
    checklist = [str(item) for item in checklist][:8]
    tool_plan = parsed.get("tool_plan") if isinstance(parsed.get("tool_plan"), list) else []
    for item in tool_plan[:4]:
        if isinstance(item, dict):
            checklist.append(f"Agent tool plan: {item.get('tool')} -> {item.get('expected_evidence') or item.get('why')}")
    evidence_used = parsed.get("evidence_used") if isinstance(parsed.get("evidence_used"), list) else []
    if evidence_used:
        checklist.append("Agent evidence used: " + ", ".join(str(item) for item in evidence_used[:6]))
    return RepairAction(
        name=str(raw.get("name") or "agent_repair_action"),
        priority=int(raw.get("priority") or 125),
        reason=str(raw.get("reason") or parsed.get("rationale_zh") or "Agent selected next repair action from TCAD evidence."),
        target_tool=str(raw.get("target_tool") or context.get("target_tool") or context.get("tool_name")),
        request_patch=request_patch,
        deck_patch=deck_patch,
        deck_mutations=[item for item in deck_mutations if isinstance(item, dict)],
        checklist=checklist,
        expected_effect=str(raw.get("expected_effect") or "Agent-driven repair action; verify metrics and curve overlay after execution."),
        user_confirmation_required=confirmation_required,
    )


def fallback_decision(
    *,
    state_path: Path,
    plan: RepairPlan,
    context: dict[str, Any],
    warnings: list[str],
    raw_response: str | None = None,
    parsed_response: dict[str, Any] | None = None,
    model: str | None = None,
    failure_reason: str | None = None,
) -> RepairAgentDecision:
    action = fallback_action(plan)
    return RepairAgentDecision(
        status=RepairAgentStatus.FALLBACK if action else RepairAgentStatus.FAILED,
        state_path=str(state_path),
        model=model,
        action=action,
        raw_response=raw_response,
        parsed_response=parsed_response,
        prompt_context=context,
        observation_summary=(parsed_response or {}).get("observation_summary") if parsed_response else None,
        hypothesis_zh=(parsed_response or {}).get("hypothesis_zh") if parsed_response else None,
        tool_plan=(parsed_response or {}).get("tool_plan") if isinstance((parsed_response or {}).get("tool_plan"), list) else [],
        safety_review=(parsed_response or {}).get("safety_review") if isinstance((parsed_response or {}).get("safety_review"), dict) else {},
        fallback_used=True,
        fallback_action_name=action.name if action else None,
        warnings=warnings,
        failure_reason=failure_reason,
    )


def decide_repair_action_with_agent(
    state_path: Path,
    *,
    deterministic_plan: RepairPlan | None = None,
    client: ChatClient | None = None,
    allow_fallback: bool = True,
) -> RepairAgentDecision:
    plan = deterministic_plan or build_repair_plan(state_path)
    context = repair_agent_context(state_path, deterministic_plan=plan)
    chat_client = client or LLMClient()
    system, user = build_repair_agent_messages(context)
    warnings: list[str] = []
    try:
        raw_response = chat_client.chat(system=system, user=user, temperature=0.2)
    except Exception as exc:
        if allow_fallback:
            return fallback_decision(
                state_path=state_path,
                plan=plan,
                context=context,
                warnings=[f"repair agent LLM call failed: {exc}"],
                model=getattr(chat_client.config, "model", None),
                failure_reason=str(exc),
            )
        return RepairAgentDecision(
            status=RepairAgentStatus.FAILED,
            state_path=str(state_path),
            model=getattr(chat_client.config, "model", None),
            prompt_context=context,
            failure_reason=str(exc),
        )
    parsed = parse_json_object(raw_response)
    if parsed is None:
        if allow_fallback:
            return fallback_decision(
                state_path=state_path,
                plan=plan,
                context=context,
                warnings=["repair agent did not return a JSON object"],
                raw_response=raw_response,
                model=getattr(chat_client.config, "model", None),
            )
        return RepairAgentDecision(
            status=RepairAgentStatus.FAILED,
            state_path=str(state_path),
            model=getattr(chat_client.config, "model", None),
            raw_response=raw_response,
            prompt_context=context,
            failure_reason="repair agent did not return a JSON object",
        )
    action = normalize_agent_action(parsed, context, warnings)
    if action is None:
        if allow_fallback:
            return fallback_decision(
                state_path=state_path,
                plan=plan,
                context=context,
                warnings=warnings,
                raw_response=raw_response,
                parsed_response=parsed,
                model=getattr(chat_client.config, "model", None),
            )
        return RepairAgentDecision(
            status=RepairAgentStatus.FAILED,
            state_path=str(state_path),
            model=getattr(chat_client.config, "model", None),
            raw_response=raw_response,
            parsed_response=parsed,
            prompt_context=context,
            warnings=warnings,
            failure_reason="repair agent action was not executable",
        )
    return RepairAgentDecision(
        status=RepairAgentStatus.COMPLETED,
        state_path=str(state_path),
        model=getattr(chat_client.config, "model", None),
        action=action,
        raw_response=raw_response,
        parsed_response=parsed,
        prompt_context=context,
        observation_summary=parsed.get("observation_summary"),
        hypothesis_zh=parsed.get("hypothesis_zh"),
        tool_plan=parsed.get("tool_plan") if isinstance(parsed.get("tool_plan"), list) else [],
        safety_review=parsed.get("safety_review") if isinstance(parsed.get("safety_review"), dict) else {},
        warnings=warnings,
    )
