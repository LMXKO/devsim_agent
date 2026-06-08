from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


LOWER_IS_BETTER = {
    "leakage_current_a",
    "leakage_abs_current_at_target_a",
    "max_electric_field_v_per_cm",
    "specific_on_resistance_ohm_cm2",
    "ioff_current_a",
    "reverse_leakage_current_a",
}

ABS_HIGHER_IS_BETTER = {
    "breakdown_voltage_v",
    "breakdown_voltage_at_threshold_v",
    "breakdown_voltage_at_1ua_v",
}


class CurveShapeDiagnostic(BaseModel):
    schema_version: str = "actsoft.tcad.curve_shape.v1"
    points: int
    x_key: str | None = None
    y_key: str | None = None
    y_abs_min: float | None = None
    y_abs_max: float | None = None
    monotonic_abs_y_violations: int = 0
    log_slope_peak_x: float | None = None
    log_slope_peak_per_x: float | None = None
    knee_x: float | None = None
    threshold_bracket_x: list[float] | None = None
    leakage_interval_y_abs: list[float] | None = None
    field_peak_x: float | None = None
    field_peak_value: float | None = None
    summary: str = ""


class MutationEffectDiagnostic(BaseModel):
    schema_version: str = "actsoft.tcad.mutation_effect.v1"
    baseline_state_path: str
    mutation_state_path: str
    mutation_target: str | None = None
    request_path: str | None = None
    baseline_value: float | None = None
    mutation_value: float | None = None
    metric_deltas: dict[str, dict[str, Any]] = Field(default_factory=dict)
    improved_metrics: list[str] = Field(default_factory=list)
    regressed_metrics: list[str] = Field(default_factory=list)
    tradeoff_violations: list[dict[str, Any]] = Field(default_factory=list)
    curve_overlay: dict[str, Any] = Field(default_factory=dict)
    overlay_svg_path: str | None = None
    baseline_shape: CurveShapeDiagnostic | None = None
    mutation_shape: CurveShapeDiagnostic | None = None
    primary_metric: str | None = None
    primary_improved: bool = False
    worth_continuing: bool = False
    recommended_next_target: str | None = None
    recommended_next_direction: str | None = None
    decision: str = "insufficient_evidence"
    rationale: str = ""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def nested_get(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def load_curve_rows(path: Path | str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    curve_path = Path(path)
    if not curve_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with curve_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                numeric = finite_float(value)
                parsed[key] = numeric if numeric is not None else value
            rows.append(parsed)
    return rows


def write_curve_overlay_svg(
    baseline_rows: list[dict[str, Any]],
    mutation_rows: list[dict[str, Any]],
    output_path: Path,
    *,
    x_key: str | None = None,
    y_key: str | None = None,
) -> str | None:
    inferred_x, inferred_y, _ = infer_x_y_keys(baseline_rows or mutation_rows)
    actual_x = x_key or inferred_x
    actual_y = y_key or inferred_y
    if not actual_x or not actual_y:
        return None

    def pairs(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
        output: list[tuple[float, float]] = []
        for row in rows:
            x = finite_float(row.get(actual_x))
            y = finite_float(row.get(actual_y))
            if x is None or y is None:
                continue
            output.append((x, abs(y)))
        return sorted(output, key=lambda item: item[0])

    baseline = pairs(baseline_rows)
    mutation = pairs(mutation_rows)
    if len(baseline) < 2 or len(mutation) < 2:
        return None
    xs = [x for x, _ in baseline + mutation]
    ys = [max(y, 1.0e-300) for _, y in baseline + mutation]
    min_x, max_x = min(xs), max(xs)
    min_log_y, max_log_y = min(math.log10(y) for y in ys), max(math.log10(y) for y in ys)
    if math.isclose(min_x, max_x):
        min_x -= 1.0
        max_x += 1.0
    if math.isclose(min_log_y, max_log_y):
        min_log_y -= 1.0
        max_log_y += 1.0
    width, height = 720, 360
    left, right, top, bottom = 68, 22, 24, 52
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        log_y = math.log10(max(y, 1.0e-300))
        return top + (max_log_y - log_y) / (max_log_y - min_log_y) * plot_h

    def polyline(points: list[tuple[float, float]]) -> str:
        return " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="TCAD baseline mutation overlay">
  <rect width="{width}" height="{height}" fill="white"/>
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#64748b"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#64748b"/>
  <line x1="{left}" y1="{top}" x2="{width-right}" y2="{top}" stroke="#e2e8f0"/>
  <line x1="{left}" y1="{top + plot_h / 2:.2f}" x2="{width-right}" y2="{top + plot_h / 2:.2f}" stroke="#e2e8f0"/>
  <polyline points="{polyline(baseline)}" fill="none" stroke="#2563eb" stroke-width="2.4"/>
  <polyline points="{polyline(mutation)}" fill="none" stroke="#dc2626" stroke-width="2.4"/>
  <text x="{left}" y="18" fill="#2563eb" font-family="sans-serif" font-size="13">baseline</text>
  <text x="{left + 92}" y="18" fill="#dc2626" font-family="sans-serif" font-size="13">mutation</text>
  <text x="{left + plot_w / 2:.2f}" y="{height - 12}" fill="#475569" font-family="sans-serif" font-size="12" text-anchor="middle">{actual_x}</text>
  <text x="18" y="{top + plot_h / 2:.2f}" fill="#475569" font-family="sans-serif" font-size="12" text-anchor="middle" transform="rotate(-90 18 {top + plot_h / 2:.2f})">log abs {actual_y}</text>
</svg>
""",
        encoding="utf-8",
    )
    return str(output_path.resolve())


def final_artifacts(state: dict[str, Any]) -> dict[str, str]:
    summary = state.get("final_summary") or {}
    artifacts = summary.get("artifacts") or {}
    return {str(key): str(value) for key, value in artifacts.items() if value}


def final_metrics(state: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(((state.get("final_summary") or {}).get("metrics") or {}))
    quality_metrics = ((state.get("quality_report") or {}).get("metrics") or {})
    metrics.update(quality_metrics)
    summary = state.get("final_summary") or {}
    for key, value in summary.items():
        if isinstance(value, (int, float)) and key not in metrics:
            metrics[key] = value
    return metrics


def infer_x_y_keys(rows: list[dict[str, Any]]) -> tuple[str | None, str | None, str | None]:
    if not rows:
        return None, None, None
    keys = list(rows[0].keys())
    x_candidates = [
        key
        for key in keys
        if key.endswith("_voltage_v")
        or key.endswith("voltage_v")
        or key in {"voltage_v", "drain_voltage_v", "reverse_voltage_v", "gate_voltage_v"}
    ]
    y_candidates = [
        key
        for key in keys
        if ("current" in key and key.endswith("_a"))
        or key in {"current_a", "total_current_a", "off_current_a", "leakage_current_a"}
    ]
    field_candidates = [key for key in keys if "field" in key and isinstance(rows[0].get(key), (int, float))]
    return (
        x_candidates[0] if x_candidates else None,
        y_candidates[0] if y_candidates else None,
        field_candidates[0] if field_candidates else None,
    )


def interpolate_bracket(points: list[tuple[float, float]], threshold: float) -> list[float] | None:
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        if min(y0, y1) <= threshold <= max(y0, y1):
            return [x0, x1]
    return None


def curve_shape_diagnostic(
    rows: list[dict[str, Any]],
    *,
    x_key: str | None = None,
    y_key: str | None = None,
    threshold_y: float | None = None,
    field_key: str | None = None,
) -> CurveShapeDiagnostic:
    inferred_x, inferred_y, inferred_field = infer_x_y_keys(rows)
    actual_x = x_key or inferred_x
    actual_y = y_key or inferred_y
    actual_field = field_key or inferred_field
    if not rows or actual_x is None or actual_y is None:
        return CurveShapeDiagnostic(points=len(rows), x_key=actual_x, y_key=actual_y, summary="curve data is missing or lacks numeric x/y columns")

    pairs = [
        (float(row[actual_x]), abs(float(row[actual_y])))
        for row in rows
        if finite_float(row.get(actual_x)) is not None and finite_float(row.get(actual_y)) is not None
    ]
    pairs.sort(key=lambda item: abs(item[0]))
    violations = 0
    for (_, left_y), (_, right_y) in zip(pairs[:-1], pairs[1:]):
        if right_y + 1e-300 < left_y:
            violations += 1

    slopes: list[tuple[float, float]] = []
    for (x0, y0), (x1, y1) in zip(pairs[:-1], pairs[1:]):
        dx = abs(x1 - x0)
        if dx == 0 or y0 <= 0 or y1 <= 0:
            continue
        slopes.append(((x0 + x1) / 2.0, abs(math.log10(y1) - math.log10(y0)) / dx))
    peak_slope = max(slopes, key=lambda item: item[1]) if slopes else None
    knee_x = peak_slope[0] if peak_slope else None
    bracket = interpolate_bracket(pairs, threshold_y) if threshold_y is not None else None
    reverse_or_low_bias = [y for x, y in pairs if x <= 0 or abs(x) <= 1.0]
    leakage_interval = [min(reverse_or_low_bias), max(reverse_or_low_bias)] if reverse_or_low_bias else None

    field_peak_x = None
    field_peak_value = None
    if actual_field:
        field_points = [
            (finite_float(row.get(actual_x)), finite_float(row.get(actual_field)))
            for row in rows
        ]
        field_points = [(x, y) for x, y in field_points if x is not None and y is not None]
        if field_points:
            field_peak_x, field_peak_value = max(field_points, key=lambda item: abs(item[1]))

    summary_bits = []
    if violations:
        summary_bits.append(f"{violations} monotonicity breaks")
    if bracket:
        summary_bits.append(f"threshold bracket {bracket[0]:.6g}..{bracket[1]:.6g} V")
    if knee_x is not None:
        summary_bits.append(f"knee near {knee_x:.6g}")
    if field_peak_x is not None:
        summary_bits.append(f"field peak near {field_peak_x:.6g}")

    y_values = [item[1] for item in pairs]
    return CurveShapeDiagnostic(
        points=len(pairs),
        x_key=actual_x,
        y_key=actual_y,
        y_abs_min=min(y_values) if y_values else None,
        y_abs_max=max(y_values) if y_values else None,
        monotonic_abs_y_violations=violations,
        log_slope_peak_x=peak_slope[0] if peak_slope else None,
        log_slope_peak_per_x=peak_slope[1] if peak_slope else None,
        knee_x=knee_x,
        threshold_bracket_x=bracket,
        leakage_interval_y_abs=leakage_interval,
        field_peak_x=field_peak_x,
        field_peak_value=field_peak_value,
        summary=", ".join(summary_bits) or "smooth curve shape",
    )


def metric_improved(metric: str, baseline: float, mutated: float) -> bool:
    if metric in LOWER_IS_BETTER:
        return mutated < baseline
    if metric in ABS_HIGHER_IS_BETTER:
        return abs(mutated) > abs(baseline)
    return mutated > baseline


def metric_regressed(metric: str, baseline: float, mutated: float) -> bool:
    if metric in LOWER_IS_BETTER:
        return mutated > baseline
    if metric in ABS_HIGHER_IS_BETTER:
        return abs(mutated) < abs(baseline)
    return mutated < baseline


def compare_metrics(baseline_metrics: dict[str, Any], mutated_metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    watched = sorted((set(LOWER_IS_BETTER) | set(ABS_HIGHER_IS_BETTER)) & set(baseline_metrics) & set(mutated_metrics))
    deltas: dict[str, dict[str, Any]] = {}
    for metric in watched:
        base = finite_float(baseline_metrics.get(metric))
        new = finite_float(mutated_metrics.get(metric))
        if base is None or new is None:
            continue
        rel = (new - base) / max(abs(base), 1.0e-300)
        deltas[metric] = {
            "baseline": base,
            "mutation": new,
            "delta": new - base,
            "relative_delta": rel,
            "improved": metric_improved(metric, base, new),
            "regressed": metric_regressed(metric, base, new),
        }
    return deltas


def primary_metric_for_target(target: str | None, issue_codes: list[str] | None = None) -> str | None:
    codes = " ".join(issue_codes or [])
    if "field" in codes:
        return "max_electric_field_v_per_cm"
    if "ron" in codes:
        return "specific_on_resistance_ohm_cm2"
    if "breakdown" in codes or "bv" in codes:
        return "breakdown_voltage_v"
    if target in {"drift_doping", "implant_dose"}:
        return "specific_on_resistance_ohm_cm2"
    if target in {"field_plate", "guard_ring", "trench_corner_radius"}:
        return "max_electric_field_v_per_cm"
    return "leakage_current_a"


def infer_tradeoff_violations(deltas: dict[str, dict[str, Any]], primary_metric: str | None) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for metric, delta in deltas.items():
        if metric == primary_metric:
            continue
        rel = abs(float(delta.get("relative_delta") or 0.0))
        regressed = bool(delta.get("regressed"))
        tolerance = 0.2 if metric == "specific_on_resistance_ohm_cm2" else 0.1
        if regressed and rel > tolerance:
            violations.append(
                {
                    "metric": metric,
                    "relative_delta": delta.get("relative_delta"),
                    "baseline": delta.get("baseline"),
                    "mutation": delta.get("mutation"),
                    "tolerance": tolerance,
                }
            )
    return violations


def recommended_alternate_target(primary_metric: str | None, failed_target: str | None) -> str:
    if primary_metric == "max_electric_field_v_per_cm":
        for target in ["field_plate", "guard_ring", "trench_corner_radius", "drift_doping"]:
            if target != failed_target:
                return target
    if primary_metric == "specific_on_resistance_ohm_cm2":
        for target in ["drift_doping", "implant_dose", "junction_depth"]:
            if target != failed_target:
                return target
    if primary_metric == "breakdown_voltage_v":
        for target in ["drift_doping", "field_plate", "guard_ring"]:
            if target != failed_target:
                return target
    for target in ["lifetime", "trap_density", "field_plate", "guard_ring"]:
        if target != failed_target:
            return target
    return "lifetime"


def request_value_for_patch(state: dict[str, Any], path: str | None) -> float | None:
    if not path:
        return None
    request = state.get("request") or {}
    value = request.get(path)
    if value is None:
        value = nested_get(request, path)
    return finite_float(value)


def compare_state_mutation_effect(
    baseline_state_path: Path,
    mutation_state_path: Path,
    *,
    deck_patch: dict[str, Any] | None = None,
    issue_codes: list[str] | None = None,
    overlay_output_path: Path | None = None,
) -> MutationEffectDiagnostic:
    baseline_state = read_json(baseline_state_path)
    mutation_state = read_json(mutation_state_path)
    baseline_metrics = final_metrics(baseline_state)
    mutation_metrics = final_metrics(mutation_state)
    deltas = compare_metrics(baseline_metrics, mutation_metrics)
    artifacts_base = final_artifacts(baseline_state)
    artifacts_mut = final_artifacts(mutation_state)
    baseline_rows = load_curve_rows(artifacts_base.get("csv") or artifacts_base.get("curve_csv"))
    mutation_rows = load_curve_rows(artifacts_mut.get("csv") or artifacts_mut.get("curve_csv"))
    threshold = finite_float(mutation_metrics.get("breakdown_current_threshold_a")) or finite_float(
        baseline_metrics.get("breakdown_current_threshold_a")
    )
    base_shape = curve_shape_diagnostic(baseline_rows, threshold_y=threshold)
    mut_shape = curve_shape_diagnostic(mutation_rows, threshold_y=threshold)
    overlay_svg_path = None
    if overlay_output_path:
        overlay_svg_path = write_curve_overlay_svg(baseline_rows, mutation_rows, overlay_output_path)

    target = str((deck_patch or {}).get("target") or (deck_patch or {}).get("source_mutation") or "")
    request_path = str((deck_patch or {}).get("request_path") or "") or None
    target = target or None
    primary = primary_metric_for_target(target, issue_codes)
    primary_delta = deltas.get(primary or "")
    improved = bool(primary_delta and primary_delta.get("improved"))
    violations = infer_tradeoff_violations(deltas, primary)
    worth_continuing = improved and not violations
    decision = "continue_same_target" if worth_continuing else "switch_target" if primary_delta else "insufficient_evidence"
    recommendation = target if worth_continuing else recommended_alternate_target(primary, target)

    baseline_value = finite_float((deck_patch or {}).get("baseline_value"))
    if baseline_value is None:
        baseline_value = request_value_for_patch(baseline_state, request_path)
    mutation_value = finite_float((deck_patch or {}).get("value"))
    if mutation_value is None:
        mutation_value = request_value_for_patch(mutation_state, request_path)
    direction = None
    if baseline_value is not None and mutation_value is not None:
        direction = "increase" if mutation_value > baseline_value else "decrease" if mutation_value < baseline_value else "hold"

    overlay = {
        "baseline_points": len(baseline_rows),
        "mutation_points": len(mutation_rows),
        "baseline_csv": artifacts_base.get("csv") or artifacts_base.get("curve_csv"),
        "mutation_csv": artifacts_mut.get("csv") or artifacts_mut.get("curve_csv"),
        "baseline_shape_summary": base_shape.summary,
        "mutation_shape_summary": mut_shape.summary,
        "overlay_svg": overlay_svg_path,
    }
    improved_metrics = [metric for metric, delta in deltas.items() if delta.get("improved")]
    regressed_metrics = [metric for metric, delta in deltas.items() if delta.get("regressed")]
    rationale = (
        f"{primary} improved without blocking tradeoffs"
        if worth_continuing
        else f"{primary} did not improve enough or violated tradeoffs"
        if primary_delta
        else "missing comparable primary metric"
    )
    return MutationEffectDiagnostic(
        baseline_state_path=str(baseline_state_path),
        mutation_state_path=str(mutation_state_path),
        mutation_target=target,
        request_path=request_path,
        baseline_value=baseline_value,
        mutation_value=mutation_value,
        metric_deltas=deltas,
        improved_metrics=improved_metrics,
        regressed_metrics=regressed_metrics,
        tradeoff_violations=violations,
        curve_overlay=overlay,
        overlay_svg_path=overlay_svg_path,
        baseline_shape=base_shape,
        mutation_shape=mut_shape,
        primary_metric=primary,
        primary_improved=improved,
        worth_continuing=worth_continuing,
        recommended_next_target=recommendation,
        recommended_next_direction=direction,
        decision=decision,
        rationale=rationale,
    )
