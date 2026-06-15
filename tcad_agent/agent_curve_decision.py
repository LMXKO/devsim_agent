from __future__ import annotations

import argparse
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from tcad_agent.agent_curve_guidance import AgentCurveGuidance, build_agent_curve_guidance
from tcad_agent.llm import LLMClient, LLMConfig
from tcad_agent.task_planner import parse_json_object
from tcad_agent.task_spec import PROJECT_ROOT


ALLOWED_CURVE_DECISION_ACTIONS = [
    "refine_effective_mutation",
    "switch_mutation_target",
    "pareto_review_before_next_patch",
    "repair_curve_shape",
    "collect_more_curve_evidence",
]

ALLOWED_CURVE_DECISION_TARGETS = [
    "field_plate",
    "guard_ring",
    "drift_doping",
    "implant_dose",
    "junction_depth",
    "region_specific_lifetime",
    "trap_density",
    "oxide_thickness",
    "trench_corner_radius",
    "bias_or_mesh_refinement",
]

ACTION_ALIASES = {
    "continue_same_target": "refine_effective_mutation",
    "continue_refine": "refine_effective_mutation",
    "refine": "refine_effective_mutation",
    "switch_target": "switch_mutation_target",
    "pareto_review": "pareto_review_before_next_patch",
    "review_constraints": "pareto_review_before_next_patch",
    "blocked_for_pareto_review": "pareto_review_before_next_patch",
    "repair_curve": "repair_curve_shape",
    "collect_more_evidence": "collect_more_curve_evidence",
}

TARGET_ALIASES = {
    "lifetime": "region_specific_lifetime",
    "local_lifetime": "region_specific_lifetime",
    "region_lifetime": "region_specific_lifetime",
    "mesh": "bias_or_mesh_refinement",
    "bias": "bias_or_mesh_refinement",
    "bias_refinement": "bias_or_mesh_refinement",
    "mesh_refinement": "bias_or_mesh_refinement",
    "fieldplate": "field_plate",
    "guardring": "guard_ring",
    "drift": "drift_doping",
    "doping": "drift_doping",
}


class ChatClient(Protocol):
    config: LLMConfig

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        ...


class CurveDecisionPlannerStatus(str, Enum):
    COMPLETED = "completed"
    FALLBACK = "fallback"
    FAILED = "failed"


class CurveDecisionNextAgentAction(str, Enum):
    PLAN_GUIDANCE_PATCH = "plan_guidance_patch"
    RUN_REPAIR_EXECUTOR = "run_repair_executor"
    PARETO_REVIEW = "pareto_review"
    COLLECT_MORE_EVIDENCE = "collect_more_evidence"


class CurveDecisionPlannerRequest(BaseModel):
    source_state_path: Path
    goal_text: str
    output_path: Path | None = None
    use_llm: bool = False
    allow_llm_fallback: bool = True


class CurveDecisionPlan(BaseModel):
    schema_version: str = "actsoft.tcad.agent_curve_decision.v1"
    status: CurveDecisionPlannerStatus
    source_state_path: str
    output_path: str | None = None
    use_llm: bool = False
    fallback_used: bool = False
    decision_source: str = "deterministic_guidance"
    model: str | None = None
    raw_response: str | None = None
    parsed_response: dict[str, Any] | None = None
    recommended_action: str | None = None
    recommended_target: str | None = None
    recommended_direction: str | None = None
    rationale: str = ""
    evidence_used: list[str] = Field(default_factory=list)
    next_agent_action: CurveDecisionNextAgentAction = CurveDecisionNextAgentAction.COLLECT_MORE_EVIDENCE
    curve_guidance: dict[str, Any] = Field(default_factory=dict)
    oracle_guidance: dict[str, Any] = Field(default_factory=dict)
    mutation_effect_analysis: dict[str, Any] = Field(default_factory=dict)
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalized_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def normalize_curve_action(value: Any) -> str | None:
    label = normalized_label(value)
    label = ACTION_ALIASES.get(label, label)
    return label if label in ALLOWED_CURVE_DECISION_ACTIONS else None


def normalize_curve_target(value: Any) -> str | None:
    label = normalized_label(value)
    label = TARGET_ALIASES.get(label, label)
    return label if label in ALLOWED_CURVE_DECISION_TARGETS else None


def model_name(llm_client: ChatClient | None) -> str | None:
    config = getattr(llm_client, "config", None)
    return getattr(config, "model", None)


def effect_from_guidance(guidance: AgentCurveGuidance) -> dict[str, Any]:
    effect = guidance.mutation_effect
    return dict(effect) if isinstance(effect, dict) else {}


def deterministic_decision(guidance: AgentCurveGuidance) -> dict[str, Any]:
    return {
        "recommended_action": normalize_curve_action(guidance.recommended_action) or guidance.recommended_action,
        "recommended_target": normalize_curve_target(guidance.recommended_target) or guidance.recommended_target,
        "recommended_direction": guidance.recommended_direction,
        "rationale": guidance.reason,
        "evidence_used": guidance.decision_basis,
    }


def build_curve_decision_messages(
    *,
    goal_text: str,
    source_state_path: str,
    guidance: AgentCurveGuidance,
) -> tuple[str, str]:
    effect = effect_from_guidance(guidance)
    system = (
        "You are the curve-review brain inside a long-running TCAD agent. "
        "Compare the baseline-vs-mutation evidence and choose the next agent action from allowed labels only. "
        "Do not invent labels or patches. Return JSON only."
    )
    payload = {
        "task": "choose the next autonomous TCAD action after reviewing mutation curves",
        "response_schema": {
            "recommended_action": ALLOWED_CURVE_DECISION_ACTIONS,
            "recommended_target": ALLOWED_CURVE_DECISION_TARGETS,
            "recommended_direction": "increase | decrease | smaller_step_same_direction | probe_alternate | add_points | pareto_review",
            "rationale": "short explanation tied to metric deltas, curve shape, overlay, and tradeoffs",
            "evidence_used": ["mutation_effect_analysis", "metric_deltas", "curve_shape", "overlay", "tradeoff_violations"],
        },
        "action_meanings": {
            "refine_effective_mutation": "primary metric improved and tradeoffs are acceptable; continue with a smaller patch on the same useful direction",
            "switch_mutation_target": "the tested mutation did not improve its primary metric enough; probe another physical target",
            "pareto_review_before_next_patch": "primary movement exists but BV/Ron/field/leakage tradeoffs require constraint/Pareto review first",
            "repair_curve_shape": "curve is too sparse or nonmonotonic; fix bias stepping, mesh, or numerical setup before physical edits",
            "collect_more_curve_evidence": "there is not enough comparable curve evidence to choose a physical patch",
        },
        "goal_text": goal_text,
        "source_state_path": source_state_path,
        "oracle_guidance": guidance.model_dump(mode="json"),
        "curve_evidence": {
            "mutation_effect_analysis": effect,
            "pareto_decision": guidance.pareto_decision,
            "metric_snapshot": guidance.metric_snapshot,
            "curve_shape": guidance.shape,
            "curve_csv_path": guidance.curve_csv_path,
        },
    }
    return system, json.dumps(payload, indent=2, ensure_ascii=False)


def validate_curve_decision(decision: dict[str, Any]) -> tuple[bool, str | None, str | None, str | None, str | None]:
    action = normalize_curve_action(decision.get("recommended_action") or decision.get("action"))
    target = normalize_curve_target(decision.get("recommended_target") or decision.get("target"))
    direction = normalized_label(decision.get("recommended_direction") or decision.get("direction")) or None
    if action is None:
        return False, None, target, direction, "decision did not choose an allowed action"
    if target is None:
        return False, action, None, direction, "decision did not choose an allowed target"
    rationale = str(decision.get("rationale") or decision.get("reason") or "").strip()
    if not rationale:
        return False, action, target, direction, "decision did not include a rationale"
    evidence = decision.get("evidence_used")
    if not isinstance(evidence, list) or not evidence:
        return False, action, target, direction, "decision did not cite evidence_used"
    return True, action, target, direction, None


def choose_next_agent_action(action: str | None) -> CurveDecisionNextAgentAction:
    if action in {"refine_effective_mutation", "switch_mutation_target"}:
        return CurveDecisionNextAgentAction.PLAN_GUIDANCE_PATCH
    if action == "pareto_review_before_next_patch":
        return CurveDecisionNextAgentAction.PARETO_REVIEW
    if action == "repair_curve_shape":
        return CurveDecisionNextAgentAction.RUN_REPAIR_EXECUTOR
    return CurveDecisionNextAgentAction.COLLECT_MORE_EVIDENCE


def merge_decision_into_guidance(
    *,
    guidance: AgentCurveGuidance,
    decision: dict[str, Any],
    action: str,
    target: str,
    direction: str | None,
    decision_source: str,
) -> dict[str, Any]:
    data = guidance.model_dump(mode="json")
    rationale = str(decision.get("rationale") or decision.get("reason") or guidance.reason)
    evidence = decision.get("evidence_used") if isinstance(decision.get("evidence_used"), list) else guidance.decision_basis
    hint = data.get("next_patch_hint") if isinstance(data.get("next_patch_hint"), dict) else {}
    hint.update(
        {
            "target": target,
            "direction": direction or hint.get("direction") or data.get("recommended_direction"),
            "basis": list(dict.fromkeys([*(hint.get("basis") if isinstance(hint.get("basis"), list) else []), *[str(item) for item in evidence]])),
            "decision_source": decision_source,
        }
    )
    if target in {"field_plate", "guard_ring", "trench_corner_radius", "junction_depth"}:
        hint["requires_user_confirmation"] = True
    data.update(
        {
            "schema_version": "actsoft.tcad.agent_curve_guidance.v1",
            "recommended_action": action,
            "recommended_target": target,
            "recommended_direction": direction or data.get("recommended_direction"),
            "reason": rationale,
            "decision_basis": [str(item) for item in evidence],
            "next_patch_hint": hint,
            "curve_decision_source": decision_source,
        }
    )
    return data


def llm_decision(
    *,
    request: CurveDecisionPlannerRequest,
    guidance: AgentCurveGuidance,
    llm_client: ChatClient | None,
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None, str | None, str | None]:
    chat_client = llm_client or LLMClient()
    system, user = build_curve_decision_messages(
        goal_text=request.goal_text,
        source_state_path=str(request.source_state_path),
        guidance=guidance,
    )
    try:
        raw = chat_client.chat(system=system, user=user, temperature=0.1)
    except Exception as exc:
        return None, None, None, model_name(chat_client), str(exc)
    parsed = parse_json_object(raw)
    if parsed is None:
        return None, raw, None, model_name(chat_client), "LLM response did not contain a JSON object"
    return parsed, raw, parsed, model_name(chat_client), None


def build_curve_decision_plan(
    request: CurveDecisionPlannerRequest,
    *,
    llm_client: ChatClient | None = None,
) -> CurveDecisionPlan:
    source = request.source_state_path.expanduser().resolve()
    try:
        guidance = build_agent_curve_guidance(goal_text=request.goal_text, source_state_path=str(source))
    except Exception as exc:
        plan = CurveDecisionPlan(
            status=CurveDecisionPlannerStatus.FAILED,
            source_state_path=str(source),
            use_llm=request.use_llm,
            failure_reason=str(exc),
        )
        if request.output_path:
            plan.output_path = str(request.output_path.resolve())
            write_json(request.output_path, plan.model_dump(mode="json"))
        return plan

    oracle = guidance.model_dump(mode="json")
    effect = effect_from_guidance(guidance)
    decision = deterministic_decision(guidance)
    decision_source = "deterministic_guidance"
    fallback_used = False
    raw_response = None
    parsed_response = None
    model = None
    call_failure = None

    if request.use_llm:
        llm, raw_response, parsed_response, model, call_failure = llm_decision(
            request=request,
            guidance=guidance,
            llm_client=llm_client,
        )
        if llm is not None:
            decision = llm
            decision_source = "llm"
        elif request.allow_llm_fallback:
            fallback_used = True
            decision_source = "deterministic_guidance"
        else:
            plan = CurveDecisionPlan(
                status=CurveDecisionPlannerStatus.FAILED,
                source_state_path=str(source),
                use_llm=request.use_llm,
                fallback_used=False,
                decision_source="llm",
                model=model,
                raw_response=raw_response,
                parsed_response=parsed_response,
                oracle_guidance=oracle,
                mutation_effect_analysis=effect,
                failure_reason=call_failure or "LLM did not produce a curve decision",
            )
            if request.output_path:
                plan.output_path = str(request.output_path.resolve())
                write_json(request.output_path, plan.model_dump(mode="json"))
            return plan

    valid, action, target, direction, validation_failure = validate_curve_decision(decision)
    if not valid or action is None or target is None:
        if request.use_llm and request.allow_llm_fallback and decision_source == "llm":
            fallback_used = True
            decision = deterministic_decision(guidance)
            decision_source = "deterministic_guidance"
            valid, action, target, direction, validation_failure = validate_curve_decision(decision)
        if not valid or action is None or target is None:
            plan = CurveDecisionPlan(
                status=CurveDecisionPlannerStatus.FAILED,
                source_state_path=str(source),
                use_llm=request.use_llm,
                fallback_used=fallback_used,
                decision_source=decision_source,
                model=model,
                raw_response=raw_response,
                parsed_response=parsed_response,
                oracle_guidance=oracle,
                mutation_effect_analysis=effect,
                failure_reason=validation_failure,
            )
            if request.output_path:
                plan.output_path = str(request.output_path.resolve())
                write_json(request.output_path, plan.model_dump(mode="json"))
            return plan

    curve_guidance = merge_decision_into_guidance(
        guidance=guidance,
        decision=decision,
        action=action,
        target=target,
        direction=direction,
        decision_source=decision_source,
    )
    status = CurveDecisionPlannerStatus.FALLBACK if fallback_used else CurveDecisionPlannerStatus.COMPLETED
    plan = CurveDecisionPlan(
        status=status,
        source_state_path=str(source),
        use_llm=request.use_llm,
        fallback_used=fallback_used,
        decision_source=decision_source,
        model=model,
        raw_response=raw_response,
        parsed_response=parsed_response,
        recommended_action=action,
        recommended_target=target,
        recommended_direction=direction,
        rationale=str(decision.get("rationale") or decision.get("reason") or guidance.reason),
        evidence_used=[str(item) for item in decision.get("evidence_used", [])] if isinstance(decision.get("evidence_used"), list) else guidance.decision_basis,
        next_agent_action=choose_next_agent_action(action),
        curve_guidance=curve_guidance,
        oracle_guidance=oracle,
        mutation_effect_analysis=effect,
        failure_reason=call_failure if fallback_used else None,
    )
    if request.output_path:
        plan.output_path = str(request.output_path.resolve())
        write_json(request.output_path, plan.model_dump(mode="json"))
    return plan


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan the next autonomous TCAD action from curve mutation evidence.")
    parser.add_argument("--source-state-path", type=Path, required=True)
    parser.add_argument("--goal", "--goal-text", dest="goal_text", required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--no-llm-fallback", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_path = args.output_path
    if output_path is None:
        output_path = PROJECT_ROOT / "runs" / "agent_curve_decision" / f"curve_decision_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    plan = build_curve_decision_plan(
        CurveDecisionPlannerRequest(
            source_state_path=args.source_state_path,
            goal_text=args.goal_text,
            output_path=output_path,
            use_llm=bool(args.use_llm),
            allow_llm_fallback=not bool(args.no_llm_fallback),
        )
    )
    print(json.dumps(plan.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if plan.status in {CurveDecisionPlannerStatus.COMPLETED, CurveDecisionPlannerStatus.FALLBACK} else 1)


if __name__ == "__main__":
    main()
