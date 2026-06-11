from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.curve_diagnostics import curve_shape_diagnostic, final_artifacts, final_metrics, load_curve_rows


class AgentCurveGuidance(BaseModel):
    schema_version: str = "actsoft.tcad.agent_curve_guidance.v1"
    created_at: str
    status: str
    source_state_path: str | None = None
    curve_csv_path: str | None = None
    shape: dict[str, Any] | None = None
    metric_snapshot: dict[str, Any] = Field(default_factory=dict)
    mutation_effect: dict[str, Any] | None = None
    pareto_decision: dict[str, Any] | None = None
    decision_basis: list[str] = Field(default_factory=list)
    recommended_action: str = "collect_more_curve_evidence"
    recommended_target: str | None = None
    recommended_direction: str | None = None
    reason: str = ""
    next_patch_hint: dict[str, Any] = Field(default_factory=dict)


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def read_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        return None
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def mapping_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def compact_mutation_effect(effect: dict[str, Any] | None) -> dict[str, Any] | None:
    if not effect:
        return None
    keys = [
        "mutation_target",
        "primary_metric",
        "primary_improved",
        "worth_continuing",
        "decision",
        "rationale",
        "recommended_next_target",
        "recommended_next_direction",
        "improved_metrics",
        "regressed_metrics",
        "tradeoff_violations",
    ]
    return {key: effect[key] for key in keys if key in effect}


def compact_pareto_decision(effect: dict[str, Any] | None) -> dict[str, Any] | None:
    if not effect:
        return None
    candidates = [
        mapping_or_none(effect.get("pareto_decision")),
        mapping_or_none(effect.get("objective_decision")),
        mapping_or_none(effect.get("engineering_objective_decision")),
    ]
    for candidate in candidates:
        if candidate:
            return {
                key: candidate[key]
                for key in [
                    "action",
                    "best_candidate_id",
                    "best_on_pareto_front",
                    "pareto_front_ids",
                    "required_violation_count",
                    "rationale",
                ]
                if key in candidate
            }
    return None


def mutation_effect_from_state(state: dict[str, Any]) -> dict[str, Any] | None:
    for key in [
        "mutation_effect_analysis",
        "sentaurus_mutation_effect_analysis",
        "curve_mutation_effect",
    ]:
        effect = mapping_or_none(state.get(key))
        if effect:
            return effect
    artifacts = final_artifacts(state)
    for key in [
        "mutation_effect",
        "mutation_effect_analysis",
        "sentaurus_mutation_effect",
        "sentaurus_mutation_effect_analysis",
    ]:
        effect = read_json(artifacts.get(key))
        if effect:
            return effect
    return None


def primary_curve_csv(state: dict[str, Any]) -> str | None:
    artifacts = final_artifacts(state)
    for key in ["csv", "curve_csv", "sentaurus_curve_csv", "iv_csv"]:
        if artifacts.get(key):
            return artifacts[key]
    metrics = final_metrics(state)
    if metrics.get("curve_path"):
        return str(metrics["curve_path"])
    return None


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "leakage_current_a",
        "leakage_abs_current_at_target_a",
        "breakdown_voltage_v",
        "breakdown_voltage_at_threshold_v",
        "specific_on_resistance_ohm_cm2",
        "max_electric_field_v_per_cm",
        "points",
    ]
    return {key: metrics[key] for key in keys if key in metrics}


def target_from_goal(goal_text: str, metrics: dict[str, Any], shape: dict[str, Any] | None) -> tuple[str, str, str]:
    lowered = goal_text.lower()
    if "field" in lowered or "场" in goal_text or metrics.get("max_electric_field_v_per_cm") or (shape or {}).get("field_peak_value"):
        return "reduce_field_peak", "field_plate", "adjust", "Field peak evidence is present; field plate or guard-ring edits are the first physical direction."
    if "ron" in lowered or "导通" in goal_text or metrics.get("specific_on_resistance_ohm_cm2"):
        return "improve_tradeoff", "drift_doping", "adjust", "Ron/BV tradeoff is visible; drift doping is the next controlled direction."
    if "bv" in lowered or "breakdown" in lowered or "耐压" in goal_text or metrics.get("breakdown_voltage_v"):
        return "bracket_breakdown", "guard_ring", "adjust", "Breakdown evidence should be bracketed before finer high-voltage geometry edits."
    if "leak" in lowered or "漏电" in goal_text or metrics.get("leakage_current_a") or metrics.get("leakage_abs_current_at_target_a"):
        return "reduce_leakage", "region_specific_lifetime", "decrease_leakage", "Leakage is the dominant observed metric; use localized lifetime/trap edits before global geometry changes."
    return "collect_more_curve_evidence", "bias_or_mesh_refinement", "refine", "Curve evidence is not specific enough for a physical mutation target."


def guidance_from_mutation_effect(effect: dict[str, Any] | None) -> tuple[str, str | None, str | None, str] | None:
    if not effect:
        return None
    target = str(effect.get("recommended_next_target") or effect.get("mutation_target") or "") or None
    direction = str(effect.get("recommended_next_direction") or "") or None
    tradeoffs = effect.get("tradeoff_violations")
    tradeoff_count = len(tradeoffs) if isinstance(tradeoffs, list) else 0
    if tradeoff_count:
        return (
            "pareto_review_before_next_patch",
            target,
            direction,
            "Mutation moved at least one metric but violated tradeoff constraints; review Pareto/constraints before another patch.",
        )
    if effect.get("worth_continuing"):
        return (
            "refine_effective_mutation",
            target,
            direction or "smaller_step_same_direction",
            "Baseline-vs-mutation evidence improved the primary metric without blocking tradeoffs.",
        )
    decision = str(effect.get("decision") or "")
    if decision in {"switch_target", "reject_or_collect_more_evidence"} or effect.get("primary_improved") is False:
        return (
            "switch_mutation_target",
            target,
            direction or "probe_alternate",
            str(effect.get("rationale") or "Previous mutation did not improve the primary metric enough."),
        )
    return None


def build_agent_curve_guidance(
    *,
    goal_text: str,
    source_state_path: str | None,
) -> AgentCurveGuidance:
    state = read_json(source_state_path)
    if not state:
        return AgentCurveGuidance(
            created_at=utc_timestamp(),
            status="missing_state",
            source_state_path=source_state_path,
            reason="No readable state path is available for curve guidance.",
        )
    metrics = compact_metrics(final_metrics(state))
    csv_path = primary_curve_csv(state)
    rows = load_curve_rows(csv_path)
    shape = curve_shape_diagnostic(rows).model_dump(mode="json") if rows else None
    raw_mutation_effect = mutation_effect_from_state(state)
    mutation_effect = compact_mutation_effect(raw_mutation_effect)
    pareto_decision = compact_pareto_decision(raw_mutation_effect)
    effect_guidance = guidance_from_mutation_effect(raw_mutation_effect)
    decision_basis: list[str] = []
    if effect_guidance:
        action, target, direction, reason = effect_guidance
        decision_basis.append("baseline_vs_mutation_effect")
    else:
        action, target, direction, reason = target_from_goal(goal_text, metrics, shape)
        decision_basis.append("goal_metric_curve_shape")
    if pareto_decision:
        decision_basis.append("pareto_decision")
        pareto_action = str(pareto_decision.get("action") or "")
        if pareto_action in {"review_constraints", "reject_or_collect_more_evidence"}:
            action = "pareto_review_before_next_patch"
            reason = str(pareto_decision.get("rationale") or reason)
    if rows and shape:
        if shape.get("points", 0) < 3:
            action = "collect_more_curve_evidence"
            target = "bias_or_mesh_refinement"
            direction = "add_points"
            reason = "Curve is too sparse for a safe physical mutation."
        elif shape.get("monotonic_abs_y_violations", 0) > 0:
            action = "repair_curve_shape"
            target = "bias_or_mesh_refinement"
            direction = "smaller_step"
            reason = "Curve has monotonicity breaks; repair numerical/bias stepping before physical edits."
    return AgentCurveGuidance(
        created_at=utc_timestamp(),
        status="completed" if rows or metrics else "metric_only",
        source_state_path=source_state_path,
        curve_csv_path=csv_path,
        shape=shape,
        metric_snapshot=metrics,
        mutation_effect=mutation_effect,
        pareto_decision=pareto_decision,
        decision_basis=decision_basis,
        recommended_action=action,
        recommended_target=target,
        recommended_direction=direction,
        reason=reason,
        next_patch_hint={
            "target": target,
            "direction": direction,
            "requires_user_confirmation": target in {"field_plate", "guard_ring", "trench_corner_radius", "junction_depth"},
            "basis": decision_basis,
            "evidence_to_check_next": ["overlay", "curve_shape", "pareto_constraints"],
        },
    )
