from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.curve_diagnostics import (
    CurveShapeDiagnostic,
    compare_metrics,
    curve_shape_diagnostic,
    finite_float,
    load_curve_rows,
    metric_improved,
    metric_regressed,
    write_curve_overlay_svg,
)
from tcad_agent.mutation_vocabulary import mutation_class_ids
from tcad_agent.engineering_objectives import (
    EngineeringObjective,
    ObjectiveCandidate,
    ObjectiveDirection,
    assign_pareto_front,
    evaluate_candidate,
)
from tcad_agent.reporting import final_artifacts, final_metrics, load_final_state


LOWER_BETTER_ALIASES = [
    "leakage_abs_current_at_target_a",
    "leakage_current_a",
    "reverse_leakage_current_a",
    "ioff_current_a",
    "max_abs_current_a",
]
BV_ALIASES = ["breakdown_voltage_v", "breakdown_voltage_at_threshold_v", "breakdown_voltage_at_1ua_v"]
FIELD_ALIASES = ["max_electric_field_v_per_cm"]
RON_ALIASES = ["specific_on_resistance_ohm_cm2", "ron_ohm_cm2", "ron_ohm"]


class SentaurusMutationEffectResult(BaseModel):
    tool_name: str = "sentaurus_mutation_effect_analyzer"
    schema_version: str = "actsoft.tcad.sentaurus_mutation_effect.v1"
    status: str
    baseline_state_path: str
    mutation_state_path: str
    candidate_id: str | None = None
    candidate: dict[str, Any] = Field(default_factory=dict)
    primary_metric: str | None = None
    primary_improved: bool = False
    decision: str = "insufficient_evidence"
    worth_continuing: bool = False
    metric_deltas: dict[str, dict[str, Any]] = Field(default_factory=dict)
    improved_metrics: list[str] = Field(default_factory=list)
    regressed_metrics: list[str] = Field(default_factory=list)
    tradeoff_violations: list[dict[str, Any]] = Field(default_factory=list)
    pareto_summary: dict[str, Any] = Field(default_factory=dict)
    baseline_shape: CurveShapeDiagnostic | None = None
    mutation_shape: CurveShapeDiagnostic | None = None
    curve_comparison: dict[str, Any] = Field(default_factory=dict)
    curve_engineering_review: dict[str, Any] = Field(default_factory=dict)
    overlay_svg_path: str | None = None
    pareto_decision: dict[str, Any] = Field(default_factory=dict)
    recommended_next_action: str = "inspect_evidence"
    recommended_next_target: str | None = None
    rationale: str = ""
    output_path: str | None = None
    failure_reason: str | None = None


class SentaurusMutationEffectRequest(BaseModel):
    baseline_state_path: Path
    mutation_state_path: Path
    candidate: dict[str, Any] = Field(default_factory=dict)
    goal_text: str = ""
    output_path: Path | None = None
    overlay_output_path: Path | None = None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def merge_artifacts(state: dict[str, Any]) -> dict[str, str]:
    artifacts = final_artifacts(state)
    raw = state.get("artifacts")
    if isinstance(raw, dict):
        artifacts.update({str(key): str(value) for key, value in raw.items() if value})
    return artifacts


def sentaurus_curve_path(state: dict[str, Any], metrics: dict[str, Any]) -> str | None:
    artifacts = merge_artifacts(state)
    for key in ["sentaurus_curve_csv", "csv", "curve_csv", "curve"]:
        value = artifacts.get(key)
        if value:
            return value
    value = metrics.get("curve_path")
    return str(value) if value else None


def first_present(metrics: dict[str, Any], aliases: list[str]) -> str | None:
    for alias in aliases:
        if finite_float(metrics.get(alias)) is not None:
            return alias
    return None


def candidate_text(candidate: dict[str, Any]) -> str:
    return json.dumps(candidate, ensure_ascii=False, sort_keys=True).lower()


def candidate_focus_text(candidate: dict[str, Any]) -> str:
    focused = {
        "candidate_id": candidate.get("candidate_id"),
        "title": candidate.get("title"),
        "hypothesis": candidate.get("hypothesis"),
        "patches": candidate.get("patches"),
    }
    return json.dumps(focused, ensure_ascii=False, sort_keys=True).lower()


def goal_tags(goal_text: str) -> dict[str, bool]:
    text = goal_text.lower()
    return {
        "leakage": any(token in text for token in ["leakage", "ioff", "off current", "漏电", "关态"]),
        "bv": any(token in text for token in ["breakdown", "bv", "击穿", "耐压", "反偏"]),
        "field": any(token in text for token in ["field", "电场", "场峰"]),
        "ron": any(token in text for token in ["ron", "on resistance", "导通"]),
        "convergence": any(token in text for token in ["convergence", "converge", "newton", "收敛", "步长"]),
        "tradeoff_guard": any(token in text for token in ["not worse", "without hurting", "不能变差", "不能恶化", "不牺牲"]),
    }


def infer_primary_metric(
    *,
    candidate: dict[str, Any],
    goal_text: str,
    baseline_metrics: dict[str, Any],
    mutation_metrics: dict[str, Any],
) -> str | None:
    merged_keys = {key for key in baseline_metrics if finite_float(baseline_metrics.get(key)) is not None}
    merged_keys.update(key for key in mutation_metrics if finite_float(mutation_metrics.get(key)) is not None)
    text = candidate_focus_text(candidate)
    tags = goal_tags(goal_text)

    if "convergence" in text or tags["convergence"]:
        if "curve_points" in merged_keys:
            return "curve_points"
    if "lifetime" in text or "trap" in text or tags["leakage"]:
        return first_present({key: baseline_metrics.get(key, mutation_metrics.get(key)) for key in merged_keys}, LOWER_BETTER_ALIASES)
    if "field_plate" in text or "guard_ring" in text or "trench" in text or tags["field"]:
        return first_present({key: baseline_metrics.get(key, mutation_metrics.get(key)) for key in merged_keys}, FIELD_ALIASES)
    if "drift_doping" in text and tags["ron"]:
        return first_present({key: baseline_metrics.get(key, mutation_metrics.get(key)) for key in merged_keys}, RON_ALIASES)
    if "bv_goal" in text or "drift_doping" in text or "junction" in text or tags["bv"]:
        return first_present({key: baseline_metrics.get(key, mutation_metrics.get(key)) for key in merged_keys}, BV_ALIASES)
    return first_present({key: baseline_metrics.get(key, mutation_metrics.get(key)) for key in merged_keys}, LOWER_BETTER_ALIASES + BV_ALIASES + FIELD_ALIASES + RON_ALIASES)


def status_improved(baseline: dict[str, Any], mutation: dict[str, Any]) -> bool:
    base_quality = str(((baseline.get("quality_report") or {}).get("status") or "")).lower()
    mut_quality = str(((mutation.get("quality_report") or {}).get("status") or "")).lower()
    if base_quality in {"failed", "suspicious"} and mut_quality == "passed":
        return True
    base_status = str(baseline.get("status") or "").lower()
    mut_status = str(mutation.get("status") or "").lower()
    return base_status != "completed" and mut_status == "completed"


def status_regressed(baseline: dict[str, Any], mutation: dict[str, Any]) -> bool:
    base_quality = str(((baseline.get("quality_report") or {}).get("status") or "")).lower()
    mut_quality = str(((mutation.get("quality_report") or {}).get("status") or "")).lower()
    return base_quality == "passed" and mut_quality in {"failed", "suspicious"}


def delta_for_metric(metric: str, baseline_metrics: dict[str, Any], mutation_metrics: dict[str, Any]) -> dict[str, Any] | None:
    base = finite_float(baseline_metrics.get(metric))
    new = finite_float(mutation_metrics.get(metric))
    if base is None or new is None:
        return None
    relative = (new - base) / max(abs(base), 1.0e-300)
    return {
        "baseline": base,
        "mutation": new,
        "delta": new - base,
        "relative_delta": relative,
        "improved": sentaurus_metric_improved(metric, base, new),
        "regressed": sentaurus_metric_regressed(metric, base, new),
    }


def sentaurus_metric_improved(metric: str, baseline: float, mutation: float) -> bool:
    if metric in LOWER_BETTER_ALIASES + FIELD_ALIASES + RON_ALIASES:
        return mutation < baseline
    if metric in BV_ALIASES:
        return abs(mutation) > abs(baseline)
    if metric == "curve_points":
        return mutation >= baseline
    return metric_improved(metric, baseline, mutation)


def sentaurus_metric_regressed(metric: str, baseline: float, mutation: float) -> bool:
    if metric in LOWER_BETTER_ALIASES + FIELD_ALIASES + RON_ALIASES:
        return mutation > baseline
    if metric in BV_ALIASES:
        return abs(mutation) < abs(baseline)
    if metric == "curve_points":
        return mutation < baseline
    return metric_regressed(metric, baseline, mutation)


def normalized_metric_deltas(baseline_metrics: dict[str, Any], mutation_metrics: dict[str, Any], primary_metric: str | None) -> dict[str, dict[str, Any]]:
    deltas = compare_metrics(baseline_metrics, mutation_metrics)
    for metric, delta in deltas.items():
        base = finite_float(delta.get("baseline"))
        new = finite_float(delta.get("mutation"))
        if base is not None and new is not None:
            delta["improved"] = sentaurus_metric_improved(metric, base, new)
            delta["regressed"] = sentaurus_metric_regressed(metric, base, new)
    if primary_metric and primary_metric not in deltas:
        primary_delta = delta_for_metric(primary_metric, baseline_metrics, mutation_metrics)
        if primary_delta:
            deltas[primary_metric] = primary_delta
    for metric in ["curve_points"]:
        if metric not in deltas:
            delta = delta_for_metric(metric, baseline_metrics, mutation_metrics)
            if delta:
                delta["improved"] = bool(delta["mutation"] >= delta["baseline"])
                delta["regressed"] = bool(delta["mutation"] < delta["baseline"])
                deltas[metric] = delta
    return deltas


def tradeoff_tolerance(metric: str) -> float:
    if metric in RON_ALIASES:
        return 0.2
    if metric in BV_ALIASES:
        return 0.02
    return 0.1


def infer_tradeoff_violations(deltas: dict[str, dict[str, Any]], primary_metric: str | None) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    watched = set(LOWER_BETTER_ALIASES + BV_ALIASES + FIELD_ALIASES + RON_ALIASES)
    for metric, delta in deltas.items():
        if metric == primary_metric or metric not in watched:
            continue
        if not delta.get("regressed"):
            continue
        rel = abs(float(delta.get("relative_delta") or 0.0))
        tolerance = tradeoff_tolerance(metric)
        if rel > tolerance:
            violations.append(
                {
                    "metric": metric,
                    "baseline": delta.get("baseline"),
                    "mutation": delta.get("mutation"),
                    "relative_delta": delta.get("relative_delta"),
                    "tolerance": tolerance,
                }
            )
    return violations


def objective_for_metric(metric: str) -> EngineeringObjective:
    if metric in LOWER_BETTER_ALIASES + FIELD_ALIASES + RON_ALIASES:
        return EngineeringObjective(metric_path=metric, direction=ObjectiveDirection.MINIMIZE)
    if metric in BV_ALIASES:
        return EngineeringObjective(metric_path=metric, direction=ObjectiveDirection.MAXIMIZE_ABS)
    return EngineeringObjective(metric_path=metric, direction=ObjectiveDirection.MAXIMIZE)


def pareto_summary(
    baseline_metrics: dict[str, Any],
    mutation_metrics: dict[str, Any],
    deltas: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metrics = [metric for metric in deltas if metric in set(LOWER_BETTER_ALIASES + BV_ALIASES + FIELD_ALIASES + RON_ALIASES)]
    objectives = [objective_for_metric(metric) for metric in metrics]
    if not objectives:
        return {"status": "not_evaluated", "reason": "no comparable objective metrics"}
    candidates = [
        ObjectiveCandidate(candidate_id="baseline", metrics=baseline_metrics),
        ObjectiveCandidate(candidate_id="mutation", metrics=mutation_metrics),
    ]
    evaluated = [evaluate_candidate(candidate, objectives, []) for candidate in candidates]
    evaluated = assign_pareto_front(evaluated, objectives)
    mutation = next(candidate for candidate in evaluated if candidate.candidate_id == "mutation")
    baseline = next(candidate for candidate in evaluated if candidate.candidate_id == "baseline")
    return {
        "status": "completed",
        "objectives": [objective.model_dump(mode="json") for objective in objectives],
        "baseline_score": baseline.score,
        "mutation_score": mutation.score,
        "mutation_on_pareto_front": mutation.pareto_front,
        "baseline_on_pareto_front": baseline.pareto_front,
        "mutation_dominates_baseline": bool(mutation.pareto_front and not baseline.pareto_front),
    }


def recommended_target(candidate: dict[str, Any], primary_metric: str | None, worth_continuing: bool) -> str | None:
    if worth_continuing:
        text = candidate_text(candidate)
        for token in mutation_class_ids():
            if token in text:
                return token
    if primary_metric in FIELD_ALIASES:
        return "field_plate"
    if primary_metric in RON_ALIASES:
        return "drift_doping"
    if primary_metric in BV_ALIASES:
        return "drift_doping"
    return "lifetime"


def compare_curve_shapes(
    baseline_rows: list[dict[str, Any]],
    mutation_rows: list[dict[str, Any]],
    baseline_metrics: dict[str, Any],
    mutation_metrics: dict[str, Any],
) -> tuple[CurveShapeDiagnostic, CurveShapeDiagnostic, dict[str, Any]]:
    threshold = finite_float(mutation_metrics.get("breakdown_current_threshold_a")) or finite_float(
        baseline_metrics.get("breakdown_current_threshold_a")
    )
    x_key = str(mutation_metrics.get("curve_x_key") or baseline_metrics.get("curve_x_key") or "") or None
    y_key = str(mutation_metrics.get("curve_y_key") or baseline_metrics.get("curve_y_key") or "") or None
    field_key = str(mutation_metrics.get("curve_field_key") or baseline_metrics.get("curve_field_key") or "") or None
    baseline_shape = curve_shape_diagnostic(baseline_rows, x_key=x_key, y_key=y_key, field_key=field_key, threshold_y=threshold)
    mutation_shape = curve_shape_diagnostic(mutation_rows, x_key=x_key, y_key=y_key, field_key=field_key, threshold_y=threshold)
    comparison = {
        "baseline_points": len(baseline_rows),
        "mutation_points": len(mutation_rows),
        "baseline_shape_summary": baseline_shape.summary,
        "mutation_shape_summary": mutation_shape.summary,
        "baseline_threshold_bracket_x": baseline_shape.threshold_bracket_x,
        "mutation_threshold_bracket_x": mutation_shape.threshold_bracket_x,
        "baseline_leakage_interval_y_abs": baseline_shape.leakage_interval_y_abs,
        "mutation_leakage_interval_y_abs": mutation_shape.leakage_interval_y_abs,
        "baseline_field_peak": {"x": baseline_shape.field_peak_x, "value": baseline_shape.field_peak_value},
        "mutation_field_peak": {"x": mutation_shape.field_peak_x, "value": mutation_shape.field_peak_value},
        "baseline_knee_x": baseline_shape.knee_x,
        "mutation_knee_x": mutation_shape.knee_x,
    }
    return baseline_shape, mutation_shape, comparison


def relative_change(baseline: float | None, mutation: float | None) -> float | None:
    if baseline is None or mutation is None:
        return None
    return (mutation - baseline) / max(abs(baseline), 1.0e-300)


def interval_midpoint(values: list[float] | None) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def direction_label(delta: float | None, *, lower_is_better: bool = False, tolerance: float = 0.02) -> str:
    if delta is None:
        return "unknown"
    if abs(delta) <= tolerance:
        return "flat"
    improved = delta < 0 if lower_is_better else delta > 0
    return "improved" if improved else "regressed"


def engineering_curve_review(
    baseline_shape: CurveShapeDiagnostic,
    mutation_shape: CurveShapeDiagnostic,
    curve_comparison: dict[str, Any],
) -> dict[str, Any]:
    base_leak_low = min(baseline_shape.leakage_interval_y_abs or []) if baseline_shape.leakage_interval_y_abs else None
    mut_leak_low = min(mutation_shape.leakage_interval_y_abs or []) if mutation_shape.leakage_interval_y_abs else None
    base_leak_high = max(baseline_shape.leakage_interval_y_abs or []) if baseline_shape.leakage_interval_y_abs else None
    mut_leak_high = max(mutation_shape.leakage_interval_y_abs or []) if mutation_shape.leakage_interval_y_abs else None
    leakage_delta = relative_change(base_leak_low, mut_leak_low)
    base_bracket_mid = interval_midpoint(baseline_shape.threshold_bracket_x)
    mut_bracket_mid = interval_midpoint(mutation_shape.threshold_bracket_x)
    bracket_shift = None if base_bracket_mid is None or mut_bracket_mid is None else mut_bracket_mid - base_bracket_mid
    field_delta = relative_change(baseline_shape.field_peak_value, mutation_shape.field_peak_value)
    field_x_shift = None
    if baseline_shape.field_peak_x is not None and mutation_shape.field_peak_x is not None:
        field_x_shift = mutation_shape.field_peak_x - baseline_shape.field_peak_x
    knee_shift = None
    if baseline_shape.knee_x is not None and mutation_shape.knee_x is not None:
        knee_shift = mutation_shape.knee_x - baseline_shape.knee_x
    flags: list[str] = []
    if mutation_shape.monotonic_abs_y_violations > baseline_shape.monotonic_abs_y_violations:
        flags.append("monotonicity_worsened")
    if baseline_shape.threshold_bracket_x and not mutation_shape.threshold_bracket_x:
        flags.append("lost_threshold_bracket")
    if baseline_shape.field_peak_value is not None and mutation_shape.field_peak_value is None:
        flags.append("lost_field_peak_extraction")
    if mutation_shape.points < max(2, baseline_shape.points // 2):
        flags.append("mutation_curve_sparse")
    summary_bits = [
        f"leakage window {direction_label(leakage_delta, lower_is_better=True)}",
        f"field peak {direction_label(field_delta, lower_is_better=True)}",
    ]
    if bracket_shift is not None:
        summary_bits.append(f"BV bracket shift {bracket_shift:.6g} V")
    if knee_shift is not None:
        summary_bits.append(f"knee shift {knee_shift:.6g}")
    if flags:
        summary_bits.append("flags: " + ", ".join(flags))
    return {
        "schema_version": "actsoft.tcad.sentaurus_curve_engineering_review.v1",
        "leakage_interval_low": {"baseline": base_leak_low, "mutation": mut_leak_low, "relative_delta": leakage_delta},
        "leakage_interval_high": {"baseline": base_leak_high, "mutation": mut_leak_high},
        "bv_bracket_midpoint": {"baseline": base_bracket_mid, "mutation": mut_bracket_mid, "shift_v": bracket_shift},
        "field_peak": {
            "baseline_value": baseline_shape.field_peak_value,
            "mutation_value": mutation_shape.field_peak_value,
            "relative_delta": field_delta,
            "baseline_x": baseline_shape.field_peak_x,
            "mutation_x": mutation_shape.field_peak_x,
            "x_shift": field_x_shift,
        },
        "knee_shift": knee_shift,
        "shape_flags": flags,
        "summary": "; ".join(summary_bits),
        "raw_curve_comparison": curve_comparison,
    }


def decision_from_evidence(
    *,
    primary_delta: dict[str, Any] | None,
    primary_improved: bool,
    status_better: bool,
    status_worse: bool,
    tradeoffs: list[dict[str, Any]],
    pareto: dict[str, Any],
) -> tuple[str, bool, str, str]:
    if status_worse:
        return "reject_candidate", False, "do_not_continue", "Patched Sentaurus run regressed quality/status."
    if tradeoffs:
        return "blocked_for_pareto_review", False, "pareto_or_constraint_review", "Primary movement is not enough to ignore the observed tradeoff regressions."
    if primary_improved and primary_delta and not pareto.get("mutation_on_pareto_front"):
        return "blocked_for_pareto_review", False, "pareto_or_constraint_review", "Mutation is not on the Pareto front across comparable metrics."
    if primary_improved or status_better:
        action = "continue_same_direction" if primary_delta else "rerun_with_curve_extraction"
        rationale = "Primary Sentaurus metric improved without blocking tradeoffs." if primary_delta else "Run status improved; collect/compare curves before finer physical edits."
        return "continue_refine", True, action, rationale
    if primary_delta:
        if pareto.get("mutation_dominates_baseline"):
            return "continue_refine", True, "continue_same_direction", "Mutation dominates baseline across comparable objective metrics."
        return "switch_target", False, "switch_patch_direction", "Primary Sentaurus metric did not improve."
    return "insufficient_evidence", False, "collect_curve_or_metrics", "Missing comparable Sentaurus curve/metric evidence."


def build_pareto_decision(
    *,
    decision: str,
    worth_continuing: bool,
    recommended_next_action: str,
    pareto: dict[str, Any],
    tradeoffs: list[dict[str, Any]],
    curve_review: dict[str, Any],
) -> dict[str, Any]:
    review_required = decision in {"blocked_for_pareto_review", "reject_candidate"} or bool(tradeoffs)
    if decision == "continue_refine":
        action = "continue_refine"
    elif decision == "switch_target":
        action = "switch_target"
    elif decision == "reject_candidate":
        action = "reject_or_rollback"
    elif decision == "blocked_for_pareto_review":
        action = "review_constraints_before_next_patch"
    else:
        action = "collect_more_evidence"
    return {
        "schema_version": "actsoft.tcad.sentaurus_pareto_decision.v1",
        "action": action,
        "worth_continuing": worth_continuing,
        "review_required": review_required,
        "recommended_next_action": recommended_next_action,
        "mutation_on_pareto_front": pareto.get("mutation_on_pareto_front"),
        "mutation_dominates_baseline": pareto.get("mutation_dominates_baseline"),
        "baseline_on_pareto_front": pareto.get("baseline_on_pareto_front"),
        "tradeoff_count": len(tradeoffs),
        "curve_flags": curve_review.get("shape_flags") or [],
        "reason": curve_review.get("summary") or "metric-only Pareto decision",
    }


def analyze_sentaurus_mutation_effect(request: SentaurusMutationEffectRequest) -> SentaurusMutationEffectResult:
    baseline_path = request.baseline_state_path.expanduser().resolve()
    mutation_path = request.mutation_state_path.expanduser().resolve()
    try:
        baseline_state = read_json(baseline_path)
        mutation_state = read_json(mutation_path)
        baseline_metrics = final_metrics(baseline_state)
        mutation_metrics = final_metrics(mutation_state)
        primary = infer_primary_metric(
            candidate=request.candidate,
            goal_text=request.goal_text,
            baseline_metrics=baseline_metrics,
            mutation_metrics=mutation_metrics,
        )
        deltas = normalized_metric_deltas(baseline_metrics, mutation_metrics, primary)
        baseline_curve = sentaurus_curve_path(baseline_state, baseline_metrics)
        mutation_curve = sentaurus_curve_path(mutation_state, mutation_metrics)
        baseline_rows = load_curve_rows(baseline_curve)
        mutation_rows = load_curve_rows(mutation_curve)
        baseline_shape, mutation_shape, curve_comparison = compare_curve_shapes(
            baseline_rows,
            mutation_rows,
            baseline_metrics,
            mutation_metrics,
        )
        overlay_path = request.overlay_output_path or mutation_path.parent / "sentaurus_baseline_mutation_overlay.svg"
        overlay_svg = write_curve_overlay_svg(baseline_rows, mutation_rows, overlay_path) if baseline_rows and mutation_rows else None
        primary_delta = deltas.get(primary or "")
        primary_improved = bool(primary_delta and primary_delta.get("improved"))
        status_better = status_improved(baseline_state, mutation_state)
        status_worse = status_regressed(baseline_state, mutation_state)
        tradeoffs = infer_tradeoff_violations(deltas, primary)
        pareto = pareto_summary(baseline_metrics, mutation_metrics, deltas)
        curve_review = engineering_curve_review(baseline_shape, mutation_shape, curve_comparison)
        decision, worth, next_action, rationale = decision_from_evidence(
            primary_delta=primary_delta,
            primary_improved=primary_improved,
            status_better=status_better,
            status_worse=status_worse,
            tradeoffs=tradeoffs,
            pareto=pareto,
        )
        pareto_decision = build_pareto_decision(
            decision=decision,
            worth_continuing=worth,
            recommended_next_action=next_action,
            pareto=pareto,
            tradeoffs=tradeoffs,
            curve_review=curve_review,
        )
        result = SentaurusMutationEffectResult(
            status="completed",
            baseline_state_path=str(baseline_path),
            mutation_state_path=str(mutation_path),
            candidate_id=str(request.candidate.get("candidate_id") or "") or None,
            candidate=request.candidate,
            primary_metric=primary,
            primary_improved=primary_improved or status_better,
            decision=decision,
            worth_continuing=worth,
            metric_deltas=deltas,
            improved_metrics=[metric for metric, delta in deltas.items() if delta.get("improved")],
            regressed_metrics=[metric for metric, delta in deltas.items() if delta.get("regressed")],
            tradeoff_violations=tradeoffs,
            pareto_summary=pareto,
            baseline_shape=baseline_shape,
            mutation_shape=mutation_shape,
            curve_comparison={
                **curve_comparison,
                "baseline_csv": baseline_curve,
                "mutation_csv": mutation_curve,
                "overlay_svg": overlay_svg,
            },
            curve_engineering_review=curve_review,
            overlay_svg_path=overlay_svg,
            pareto_decision=pareto_decision,
            recommended_next_action=next_action,
            recommended_next_target=recommended_target(request.candidate, primary, worth),
            rationale=rationale,
        )
    except Exception as exc:
        result = SentaurusMutationEffectResult(
            status="failed",
            baseline_state_path=str(baseline_path),
            mutation_state_path=str(mutation_path),
            candidate=request.candidate,
            candidate_id=str(request.candidate.get("candidate_id") or "") or None,
            failure_reason=str(exc),
        )
    if request.output_path is not None:
        output_path = request.output_path.expanduser().resolve()
        result.output_path = str(output_path)
        write_json(output_path, result.model_dump(mode="json"))
    return result
