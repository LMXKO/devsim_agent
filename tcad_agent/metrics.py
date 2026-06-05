from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class IVPoint(BaseModel):
    voltage_v: float
    electron_current_a: float
    hole_current_a: float
    total_current_a: float


class MOSFETPoint(BaseModel):
    sweep_type: str
    gate_voltage_v: float
    drain_voltage_v: float
    drain_electron_current_a: float
    drain_hole_current_a: float
    drain_total_current_a: float


def load_iv_points(path: Path) -> list[IVPoint]:
    points: list[IVPoint] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            points.append(
                IVPoint(
                    voltage_v=float(row["voltage_v"]),
                    electron_current_a=float(row["electron_current_a"]),
                    hole_current_a=float(row["hole_current_a"]),
                    total_current_a=float(row["total_current_a"]),
                )
            )
    return points


def finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def interpolate_crossing(points: list[tuple[float, float]], target_y: float) -> float | None:
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        if not (finite(x0) and finite(y0) and finite(x1) and finite(y1)):
            continue
        if y0 == y1:
            continue
        if min(y0, y1) <= target_y <= max(y0, y1):
            fraction = (target_y - y0) / (y1 - y0)
            return x0 + fraction * (x1 - x0)
    return None


def linear_regression_slope(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def nearest_point(points: list[IVPoint], voltage: float) -> IVPoint | None:
    if not points:
        return None
    return min(points, key=lambda point: abs(point.voltage_v - voltage))


def current_at_voltage(points: list[IVPoint], voltage: float) -> float | None:
    point = nearest_point(points, voltage)
    return point.total_current_a if point else None


def nearest_reverse_point(points: list[IVPoint], voltage: float) -> IVPoint | None:
    reverse_points = [point for point in points if point.voltage_v <= 0]
    if not reverse_points:
        return None
    return min(reverse_points, key=lambda point: abs(point.voltage_v - voltage))


def differential_resistance_last_ohm(points: list[IVPoint]) -> float | None:
    if len(points) < 2:
        return None
    left, right = points[-2], points[-1]
    delta_i = right.total_current_a - left.total_current_a
    if delta_i == 0:
        return None
    return (right.voltage_v - left.voltage_v) / delta_i


def ideality_factor(points: list[IVPoint], temperature_k: float) -> float | None:
    thermal_voltage = 8.617333262145e-5 * temperature_k
    fit_points: list[tuple[float, float]] = []
    positive_currents = [abs(point.total_current_a) for point in points if point.voltage_v > 0 and point.total_current_a > 0]
    if not positive_currents:
        return None
    floor = max(min(positive_currents) * 0.99, 1e-30)
    ceiling = max(positive_currents)
    for point in points:
        current = point.total_current_a
        if point.voltage_v <= 0 or current <= 0:
            continue
        if abs(current) < floor or abs(current) > ceiling:
            continue
        fit_points.append((point.voltage_v, math.log(abs(current))))
    if len(fit_points) < 2:
        return None
    slope = linear_regression_slope(fit_points)
    if slope is None or slope <= 0:
        return None
    return 1.0 / (slope * thermal_voltage)


def estimate_breakdown_voltage(points: list[IVPoint], threshold_a: float) -> float | None:
    reverse = sorted(
        [(point.voltage_v, abs(point.total_current_a)) for point in points if point.voltage_v < 0],
        key=lambda item: item[0],
        reverse=True,
    )
    crossed = interpolate_crossing(reverse, threshold_a)
    return crossed


def reverse_current_shape_violations(points: list[IVPoint], tolerance_a: float = 1e-18) -> int:
    reverse = sorted(
        [point for point in points if point.voltage_v <= 0 and math.isfinite(point.total_current_a)],
        key=lambda point: abs(point.voltage_v),
    )
    violations = 0
    for left, right in zip(reverse[:-1], reverse[1:]):
        if abs(right.total_current_a) + tolerance_a < abs(left.total_current_a):
            violations += 1
    return violations


def extract_diode_reverse_metrics(
    points: list[IVPoint],
    *,
    leakage_voltage_v: float = -1.0,
    breakdown_current_a: float = 1e-6,
) -> dict[str, Any]:
    reverse_points = [point for point in points if point.voltage_v < 0]
    leakage_point = nearest_reverse_point(points, leakage_voltage_v)
    reverse_abs_currents = [abs(point.total_current_a) for point in reverse_points]
    min_positive_reverse_current = min(
        (current for current in reverse_abs_currents if current > 0),
        default=None,
    )
    max_reverse_abs_current = max(reverse_abs_currents) if reverse_abs_currents else None
    reverse_gain = None
    if min_positive_reverse_current and max_reverse_abs_current is not None:
        reverse_gain = max_reverse_abs_current / min_positive_reverse_current
    breakdown_voltage = estimate_breakdown_voltage(points, breakdown_current_a)
    return {
        "reverse_points": len(reverse_points),
        "leakage_voltage_target_v": leakage_voltage_v,
        "leakage_voltage_used_v": leakage_point.voltage_v if leakage_point else None,
        "leakage_current_at_target_a": leakage_point.total_current_a if leakage_point else None,
        "leakage_abs_current_at_target_a": abs(leakage_point.total_current_a) if leakage_point else None,
        "max_reverse_abs_current_a": max_reverse_abs_current,
        "min_reverse_voltage_v": min((point.voltage_v for point in reverse_points), default=None),
        "breakdown_current_threshold_a": breakdown_current_a,
        "breakdown_voltage_at_threshold_v": breakdown_voltage,
        "breakdown_detected": breakdown_voltage is not None,
        "reverse_abs_current_gain": reverse_gain,
        "reverse_current_shape_violations": reverse_current_shape_violations(points),
    }


def extract_pn_iv_metrics(
    points: list[IVPoint],
    *,
    temperature_k: float = 300.0,
    turn_on_current_a: float = 1e-6,
    breakdown_current_a: float = 1e-6,
) -> dict[str, Any]:
    if not points:
        return {
            "points": 0,
        }
    sorted_points = sorted(points, key=lambda point: point.voltage_v)
    currents = [point.total_current_a for point in sorted_points]
    abs_currents = [abs(current) for current in currents]
    final = sorted_points[-1]
    zero_current = current_at_voltage(sorted_points, 0.0)
    reverse_points = [point for point in sorted_points if point.voltage_v < 0]
    positive_forward_points = [point for point in sorted_points if point.voltage_v > 0]
    leakage_point = reverse_points[-1] if reverse_points else nearest_point(sorted_points, 0.0)
    leakage_current = leakage_point.total_current_a if leakage_point else None
    forward_pairs = [
        (point.voltage_v, abs(point.total_current_a))
        for point in sorted_points
        if point.voltage_v >= 0 and math.isfinite(point.total_current_a)
    ]
    turn_on_voltage = interpolate_crossing(forward_pairs, turn_on_current_a)
    leakage_abs = abs(leakage_current) if leakage_current is not None else None
    rectification_ratio = None
    if leakage_abs and leakage_abs > 0:
        rectification_ratio = abs(final.total_current_a) / leakage_abs

    metrics = {
        "points": len(sorted_points),
        "forward_points": len(positive_forward_points),
        "voltage_range_v": [sorted_points[0].voltage_v, sorted_points[-1].voltage_v],
        "min_total_current_a": min(currents),
        "max_total_current_a": max(currents),
        "max_abs_current_a": max(abs_currents),
        "final_total_current_a": final.total_current_a,
        "current_at_0v_a": zero_current,
        "leakage_current_a": leakage_current,
        "leakage_voltage_v": leakage_point.voltage_v if leakage_point else None,
        "turn_on_voltage_at_1ua_v": turn_on_voltage,
        "ideality_factor_estimate": ideality_factor(sorted_points, temperature_k),
        "differential_resistance_last_ohm": differential_resistance_last_ohm(sorted_points),
        "rectification_ratio_final_to_leakage": rectification_ratio,
        "breakdown_voltage_at_1ua_v": estimate_breakdown_voltage(sorted_points, breakdown_current_a),
    }
    metrics.update(
        extract_diode_reverse_metrics(
            sorted_points,
            leakage_voltage_v=-1.0,
            breakdown_current_a=breakdown_current_a,
        )
    )
    return metrics


def extract_pn_iv_metrics_from_csv(
    path: Path,
    *,
    temperature_k: float = 300.0,
    turn_on_current_a: float = 1e-6,
    breakdown_current_a: float = 1e-6,
) -> dict[str, Any]:
    return extract_pn_iv_metrics(
        load_iv_points(path),
        temperature_k=temperature_k,
        turn_on_current_a=turn_on_current_a,
        breakdown_current_a=breakdown_current_a,
    )


def load_mosfet_points(path: Path) -> list[MOSFETPoint]:
    points: list[MOSFETPoint] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            points.append(
                MOSFETPoint(
                    sweep_type=str(row["sweep_type"]),
                    gate_voltage_v=float(row["gate_voltage_v"]),
                    drain_voltage_v=float(row["drain_voltage_v"]),
                    drain_electron_current_a=float(row["drain_electron_current_a"]),
                    drain_hole_current_a=float(row["drain_hole_current_a"]),
                    drain_total_current_a=float(row["drain_total_current_a"]),
                )
            )
    return points


def mosfet_threshold_voltage(
    points: list[MOSFETPoint],
    threshold_current_a: float,
) -> float | None:
    idvg = sorted(
        [point for point in points if point.sweep_type == "idvg"],
        key=lambda point: point.gate_voltage_v,
    )
    pairs = [(point.gate_voltage_v, abs(point.drain_total_current_a)) for point in idvg]
    return interpolate_crossing(pairs, threshold_current_a)


def mosfet_subthreshold_swing_mv_dec(points: list[MOSFETPoint]) -> float | None:
    idvg = sorted(
        [point for point in points if point.sweep_type == "idvg" and abs(point.drain_total_current_a) > 0],
        key=lambda point: point.gate_voltage_v,
    )
    if len(idvg) < 2:
        return None
    fit_points = [
        (point.gate_voltage_v, math.log10(abs(point.drain_total_current_a)))
        for point in idvg
        if math.isfinite(point.drain_total_current_a)
    ]
    slope = linear_regression_slope(fit_points)
    if slope is None or slope <= 0:
        return None
    return 1000.0 / slope


def max_transconductance_s(points: list[MOSFETPoint]) -> float | None:
    idvg = sorted(
        [point for point in points if point.sweep_type == "idvg"],
        key=lambda point: point.gate_voltage_v,
    )
    values: list[float] = []
    for left, right in zip(idvg[:-1], idvg[1:]):
        delta_v = right.gate_voltage_v - left.gate_voltage_v
        if delta_v == 0:
            continue
        values.append(abs((right.drain_total_current_a - left.drain_total_current_a) / delta_v))
    return max(values) if values else None


def output_conductance_last_s(points: list[MOSFETPoint]) -> float | None:
    idvd = sorted(
        [point for point in points if point.sweep_type == "idvd"],
        key=lambda point: point.drain_voltage_v,
    )
    if len(idvd) < 2:
        return None
    left, right = idvd[-2], idvd[-1]
    delta_v = right.drain_voltage_v - left.drain_voltage_v
    if delta_v == 0:
        return None
    return abs((right.drain_total_current_a - left.drain_total_current_a) / delta_v)


def idvd_grouped_by_gate(points: list[MOSFETPoint]) -> dict[float, list[MOSFETPoint]]:
    groups: dict[float, list[MOSFETPoint]] = {}
    for point in points:
        if point.sweep_type != "idvd":
            continue
        gate_key = round(point.gate_voltage_v, 12)
        groups.setdefault(gate_key, []).append(point)
    return {
        gate: sorted(group, key=lambda point: point.drain_voltage_v)
        for gate, group in groups.items()
    }


def idvd_shape_metrics(points: list[MOSFETPoint]) -> dict[str, Any]:
    groups = idvd_grouped_by_gate(points)
    negative_segments = 0
    kink_slope_jumps = 0
    max_slope: float | None = None
    last_slope: float | None = None
    max_drain_span = 0.0
    total_segments = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        max_drain_span = max(max_drain_span, abs(group[-1].drain_voltage_v - group[0].drain_voltage_v))
        slopes: list[float] = []
        for left, right in zip(group[:-1], group[1:]):
            delta_v = right.drain_voltage_v - left.drain_voltage_v
            if delta_v == 0:
                continue
            delta_i_abs = abs(right.drain_total_current_a) - abs(left.drain_total_current_a)
            if delta_i_abs < -1e-18:
                negative_segments += 1
            slope = delta_i_abs / delta_v
            slopes.append(slope)
            total_segments += 1
        positive_slopes = [slope for slope in slopes if math.isfinite(slope) and slope > 0]
        if positive_slopes:
            group_max = max(positive_slopes)
            group_last = positive_slopes[-1]
            max_slope = group_max if max_slope is None else max(max_slope, group_max)
            last_slope = group_last if last_slope is None else max(last_slope, group_last)
        for previous, current in zip(positive_slopes[:-1], positive_slopes[1:]):
            if previous > 0 and current / previous > 3.0:
                kink_slope_jumps += 1
    saturation_ratio = None
    if max_slope and max_slope > 0 and last_slope is not None:
        saturation_ratio = last_slope / max_slope
    return {
        "idvd_gate_curves": len(groups),
        "idvd_total_segments": total_segments,
        "idvd_negative_differential_segments": negative_segments,
        "idvd_kink_slope_jumps": kink_slope_jumps,
        "idvd_max_slope_s": max_slope,
        "idvd_last_slope_s": last_slope,
        "idvd_saturation_ratio": saturation_ratio,
        "idvd_max_drain_span_v": max_drain_span,
    }


def extract_mosfet_metrics(
    points: list[MOSFETPoint],
    *,
    threshold_current_a: float = 1e-6,
) -> dict[str, Any]:
    if not points:
        return {"points": 0}
    abs_currents = [abs(point.drain_total_current_a) for point in points]
    idvg = [point for point in points if point.sweep_type == "idvg"]
    idvd = [point for point in points if point.sweep_type == "idvd"]
    idvg_abs = [abs(point.drain_total_current_a) for point in idvg]
    idvd_abs = [abs(point.drain_total_current_a) for point in idvd]
    ioff = min(idvg_abs) if idvg_abs else None
    ion = max(idvg_abs) if idvg_abs else None
    ion_ioff = None
    if ioff and ioff > 0 and ion is not None:
        ion_ioff = ion / ioff
    metrics = {
        "points": len(points),
        "idvg_points": len(idvg),
        "idvd_points": len(idvd),
        "max_abs_drain_current_a": max(abs_currents),
        "min_abs_drain_current_a": min(abs_currents),
        "final_abs_drain_current_a": abs(points[-1].drain_total_current_a),
        "ion_current_a": ion,
        "ioff_current_a": ioff,
        "ion_ioff_ratio": ion_ioff,
        "threshold_current_a": threshold_current_a,
        "vth_at_threshold_current_v": mosfet_threshold_voltage(points, threshold_current_a),
        "subthreshold_swing_mv_dec": mosfet_subthreshold_swing_mv_dec(points),
        "max_transconductance_s": max_transconductance_s(points),
        "idvd_final_current_a": idvd_abs[-1] if idvd_abs else None,
        "output_conductance_last_s": output_conductance_last_s(points),
    }
    metrics.update(idvd_shape_metrics(points))
    return metrics


def extract_mosfet_metrics_from_csv(
    path: Path,
    *,
    threshold_current_a: float = 1e-6,
) -> dict[str, Any]:
    return extract_mosfet_metrics(load_mosfet_points(path), threshold_current_a=threshold_current_a)
