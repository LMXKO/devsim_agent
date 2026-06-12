from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.task_spec import PROJECT_ROOT


class GoldenCurveComparisonRequest(BaseModel):
    comparison_id: str | None = None
    source_state_path: Path
    reference_curve_path: Path
    run_root: Path = PROJECT_ROOT / "runs" / "golden_curve_comparison"
    source_x_column: str | None = None
    source_y_column: str | None = None
    reference_x_column: str | None = None
    reference_y_column: str | None = None
    source_x_scale: float | None = None
    source_y_scale: float | None = None
    reference_x_scale: float | None = None
    reference_y_scale: float | None = None
    match_mode: str = "interpolate"
    error_floor: float = Field(default=1.0e-20, gt=0.0)
    max_pass_rmse_log_dec: float = Field(default=0.2, gt=0.0)
    max_warn_rmse_log_dec: float = Field(default=0.5, gt=0.0)


class GoldenCurveComparisonState(BaseModel):
    tool_name: str = "golden_curve_comparison"
    status: str
    comparison_id: str
    comparison_dir: str
    request: dict[str, Any]
    created_at: str
    updated_at: str
    source_curve_path: str | None = None
    reference_curve_path: str
    aligned_curve_path: str | None = None
    calibration_path: str | None = None
    matched_points: list[dict[str, float]] = Field(default_factory=list)
    final_summary: dict[str, Any] | None = None
    quality_report: dict[str, Any] | None = None
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_comparison_id() -> str:
    return f"golden_cmp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw.get("points") if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            raise ValueError("curve JSON must be a list or an object with a points list")
        return [dict(row) for row in rows if isinstance(row, dict)]
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def infer_column(rows: list[dict[str, Any]], candidates: list[str], *, contains: str | None = None) -> str:
    if not rows:
        raise ValueError("curve has no rows")
    keys = list(rows[0])
    for candidate in candidates:
        if candidate in keys:
            return candidate
    if contains:
        for key in keys:
            if contains in key:
                return key
    for key in keys:
        if float_or_none(rows[0].get(key)) is not None:
            return key
    raise ValueError("curve has no numeric columns")


def source_curve_path_from_state(state: dict[str, Any]) -> Path:
    final_summary = state.get("final_summary") or {}
    artifacts = final_summary.get("artifacts") or {}
    csv_path = artifacts.get("csv") or artifacts.get("curve_csv")
    if csv_path:
        return Path(csv_path)
    run_dir = state.get("run_dir") or state.get("calibration_dir")
    if run_dir and (Path(run_dir) / "sweep.csv").exists():
        return Path(run_dir) / "sweep.csv"
    raise ValueError("source state does not expose a curve CSV artifact")


def log_value(value: float, floor: float) -> float:
    return math.log10(max(abs(value), floor))


def inferred_unit_scale(column: str) -> float:
    normalized = column.strip().lower().replace("[", "_").replace("]", "").replace("(", "_").replace(")", "")
    if re.search(r"(^|_)(pa|pamp|picoamp|picoamps)($|_)", normalized):
        return 1.0e-12
    if re.search(r"(^|_)(na|namp|nanoamp|nanoamps)($|_)", normalized):
        return 1.0e-9
    if re.search(r"(^|_)(ua|uamp|microamp|microamps)($|_)", normalized):
        return 1.0e-6
    if re.search(r"(^|_)(ma|mamp|milliamp|milliamps)($|_)", normalized):
        return 1.0e-3
    if re.search(r"(^|_)(mv|milliv|millivolt|millivolts)($|_)", normalized):
        return 1.0e-3
    if re.search(r"(^|_)(kv|kilov|kilovolt|kilovolts)($|_)", normalized):
        return 1.0e3
    if re.search(r"(^|_)(nm|nanometer|nanometers)($|_)", normalized):
        return 1.0e-3
    return 1.0


def curve_points(
    rows: list[dict[str, Any]],
    *,
    x_column: str,
    y_column: str,
    x_scale: float,
    y_scale: float,
) -> list[tuple[float, float]]:
    points = []
    for row in rows:
        x = float_or_none(row.get(x_column))
        y = float_or_none(row.get(y_column))
        if x is not None and y is not None:
            points.append((x * x_scale, y * y_scale))
    points.sort(key=lambda item: item[0])
    merged: dict[float, float] = {}
    for x, y in points:
        merged[round(x, 12)] = y
    return sorted(merged.items(), key=lambda item: item[0])


def interpolated_y(points: list[tuple[float, float]], x: float) -> float | None:
    if not points:
        return None
    rounded = round(x, 12)
    by_x = {round(point_x, 12): point_y for point_x, point_y in points}
    if rounded in by_x:
        return by_x[rounded]
    if x < points[0][0] or x > points[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1 and x1 != x0:
            ratio = (x - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    return None


def compare_curves(
    source_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    source_x: str,
    source_y: str,
    reference_x: str,
    reference_y: str,
    source_x_scale: float = 1.0,
    source_y_scale: float = 1.0,
    reference_x_scale: float = 1.0,
    reference_y_scale: float = 1.0,
    match_mode: str = "interpolate",
    floor: float,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    source_points = curve_points(source_rows, x_column=source_x, y_column=source_y, x_scale=source_x_scale, y_scale=source_y_scale)
    reference_points = curve_points(reference_rows, x_column=reference_x, y_column=reference_y, x_scale=reference_x_scale, y_scale=reference_y_scale)
    if len(source_points) < 2 or len(reference_points) < 2:
        raise ValueError("both source and reference curves need at least two numeric points")
    matched = []
    for x, ref_y in reference_points:
        src_y = interpolated_y(source_points, x) if match_mode != "exact" else {round(sx, 12): sy for sx, sy in source_points}.get(round(x, 12))
        if src_y is None:
            continue
        log_error = log_value(src_y, floor) - log_value(ref_y, floor)
        matched.append(
            {
                "x": x,
                "source_y": src_y,
                "reference_y": ref_y,
                "log_error_dec": log_error,
                "abs_log_error_dec": abs(log_error),
            }
        )
    if len(matched) < 2:
        raise ValueError("source and reference curves have fewer than two shared x values")
    mse = sum(point["log_error_dec"] ** 2 for point in matched) / len(matched)
    mae = sum(point["abs_log_error_dec"] for point in matched) / len(matched)
    mean_log_error = sum(point["log_error_dec"] for point in matched) / len(matched)
    source_to_reference_y_scale = 10 ** (-mean_log_error)
    fitted_errors = [
        log_value(point["source_y"] * source_to_reference_y_scale, floor) - log_value(point["reference_y"], floor)
        for point in matched
    ]
    fitted_mse = sum(error ** 2 for error in fitted_errors) / len(fitted_errors)
    sign_mismatches = sum(1 for point in matched if point["source_y"] * point["reference_y"] < 0)
    metrics = {
        "matched_points": float(len(matched)),
        "golden_curve_rmse_log_dec": math.sqrt(mse),
        "golden_curve_mae_log_dec": mae,
        "golden_curve_max_abs_log_error_dec": max(point["abs_log_error_dec"] for point in matched),
        "golden_curve_mean_log_error_dec": mean_log_error,
        "golden_curve_source_to_reference_y_scale": source_to_reference_y_scale,
        "golden_curve_rmse_after_y_scale_fit_log_dec": math.sqrt(fitted_mse),
        "golden_curve_sign_mismatch_count": float(sign_mismatches),
        "golden_curve_source_x_scale": source_x_scale,
        "golden_curve_source_y_scale": source_y_scale,
        "golden_curve_reference_x_scale": reference_x_scale,
        "golden_curve_reference_y_scale": reference_y_scale,
        "golden_curve_match_mode": 0.0 if match_mode == "exact" else 1.0,
        "golden_curve_overlap_min_x": min(point["x"] for point in matched),
        "golden_curve_overlap_max_x": max(point["x"] for point in matched),
    }
    for point, fitted_error in zip(matched, fitted_errors):
        point["source_y_scaled_fit"] = point["source_y"] * source_to_reference_y_scale
        point["fit_log_error_dec"] = fitted_error
    return matched, metrics


def calibration_recommendation(metrics: dict[str, Any], request: GoldenCurveComparisonRequest) -> dict[str, Any]:
    scale = float(metrics.get("golden_curve_source_to_reference_y_scale") or 1.0)
    rmse = float(metrics.get("golden_curve_rmse_log_dec") or 0.0)
    fitted = float(metrics.get("golden_curve_rmse_after_y_scale_fit_log_dec") or rmse)
    recommendations: list[str] = []
    if abs(math.log10(max(scale, 1.0e-300))) > 0.05:
        recommendations.append("apply_source_current_scale_or_contact_area_fit")
    if fitted < rmse * 0.75:
        recommendations.append("first_fit_current_scale_then_rerun_model_parameters")
    if rmse > request.max_pass_rmse_log_dec:
        recommendations.append("run_parameter_calibration_or_repair_before_signoff")
    if not recommendations:
        recommendations.append("accept_curve_alignment_for_current_signoff_gate")
    return {
        "schema_version": "actsoft.tcad.golden_curve_calibration.v1",
        "source_to_reference_y_scale": scale,
        "rmse_log_dec": rmse,
        "rmse_after_y_scale_fit_log_dec": fitted,
        "matched_points": metrics.get("matched_points"),
        "recommendations": recommendations,
    }


def run_golden_curve_comparison(request: GoldenCurveComparisonRequest) -> GoldenCurveComparisonState:
    comparison_id = request.comparison_id or default_comparison_id()
    run_dir = request.run_root / comparison_id
    state_path = run_dir / "state.json"
    now = utc_timestamp()
    state = GoldenCurveComparisonState(
        status="completed",
        comparison_id=comparison_id,
        comparison_dir=str(run_dir),
        request=request.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        reference_curve_path=str(request.reference_curve_path),
    )
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        source_state = read_json(request.source_state_path)
        source_curve = source_curve_path_from_state(source_state)
        source_rows = load_rows(source_curve)
        reference_rows = load_rows(request.reference_curve_path)
        source_x = request.source_x_column or infer_column(source_rows, ["voltage_v", "gate_voltage_v", "drain_voltage_v"], contains="voltage")
        source_y = request.source_y_column or infer_column(source_rows, ["current_a", "drain_current_a", "collector_current_a"], contains="current")
        reference_x = request.reference_x_column or infer_column(reference_rows, [source_x, "voltage_v", "gate_voltage_v", "drain_voltage_v"], contains="voltage")
        reference_y = request.reference_y_column or infer_column(reference_rows, [source_y, "current_a", "drain_current_a", "collector_current_a"], contains="current")
        source_x_scale = request.source_x_scale if request.source_x_scale is not None else inferred_unit_scale(source_x)
        source_y_scale = request.source_y_scale if request.source_y_scale is not None else inferred_unit_scale(source_y)
        reference_x_scale = request.reference_x_scale if request.reference_x_scale is not None else inferred_unit_scale(reference_x)
        reference_y_scale = request.reference_y_scale if request.reference_y_scale is not None else inferred_unit_scale(reference_y)
        matched, metrics = compare_curves(
            source_rows,
            reference_rows,
            source_x=source_x,
            source_y=source_y,
            reference_x=reference_x,
            reference_y=reference_y,
            source_x_scale=source_x_scale,
            source_y_scale=source_y_scale,
            reference_x_scale=reference_x_scale,
            reference_y_scale=reference_y_scale,
            match_mode=request.match_mode,
            floor=request.error_floor,
        )
        calibration = calibration_recommendation(metrics, request)
        aligned_csv_path = run_dir / "aligned_points.csv"
        with aligned_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "x",
                    "source_y",
                    "reference_y",
                    "source_y_scaled_fit",
                    "log_error_dec",
                    "abs_log_error_dec",
                    "fit_log_error_dec",
                ],
            )
            writer.writeheader()
            writer.writerows(matched)
        calibration_path = run_dir / "calibration.json"
        calibration_path.write_text(json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8")
        metrics.update(
            {
                "golden_or_measured_comparison": True,
                "source_state_path": str(request.source_state_path),
                "source_curve_path": str(source_curve),
                "reference_curve_path": str(request.reference_curve_path),
                "source_x_column": source_x,
                "source_y_column": source_y,
                "reference_x_column": reference_x,
                "reference_y_column": reference_y,
                "calibration_source_to_reference_y_scale": calibration["source_to_reference_y_scale"],
            }
        )
        issues = []
        if metrics["golden_curve_sign_mismatch_count"] > 0:
            issues.append({"code": "golden_curve_sign_mismatch", "severity": "error", "count": metrics["golden_curve_sign_mismatch_count"]})
        if metrics["golden_curve_rmse_log_dec"] > request.max_warn_rmse_log_dec:
            issues.append({"code": "golden_curve_rmse_far_above_threshold", "severity": "error", "rmse": metrics["golden_curve_rmse_log_dec"]})
        elif metrics["golden_curve_rmse_log_dec"] > request.max_pass_rmse_log_dec:
            issues.append({"code": "golden_curve_rmse_above_pass_threshold", "severity": "warning", "rmse": metrics["golden_curve_rmse_log_dec"]})
        quality_status = "failed" if any(issue["severity"] == "error" for issue in issues) else "suspicious" if issues else "passed"
        state.status = "failed" if quality_status == "failed" else "completed"
        state.source_curve_path = str(source_curve)
        state.aligned_curve_path = str(aligned_csv_path.resolve())
        state.calibration_path = str(calibration_path.resolve())
        state.matched_points = matched
        state.final_summary = {
            "task": "golden_curve_comparison",
            "status": state.status,
            "metrics": metrics,
            "artifacts": {
                "summary": str((run_dir / "summary.json").resolve()),
                "source_curve": str(source_curve.resolve()),
                "reference_curve": str(request.reference_curve_path.resolve()),
                "aligned_points": str(aligned_csv_path.resolve()),
                "calibration": str(calibration_path.resolve()),
            },
            "calibration": calibration,
        }
        state.quality_report = {
            "status": quality_status,
            "issues": issues,
            "metrics": metrics,
            "calibration": calibration,
            "recommended_next_action": (
                "accept golden/measured curve comparison"
                if quality_status == "passed"
                else "; ".join(calibration["recommendations"][:2])
            ),
        }
        (run_dir / "matched_points.json").write_text(json.dumps(matched, indent=2), encoding="utf-8")
        (run_dir / "summary.json").write_text(json.dumps(state.final_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        state.status = "failed"
        state.failure_reason = str(exc)
        state.quality_report = {"status": "failed", "issues": [{"code": "golden_curve_comparison_failed", "severity": "error", "message": str(exc)}], "metrics": {}}
    state.updated_at = utc_timestamp()
    state_path.write_text(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a TCAD curve against a golden/measured reference curve.")
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--reference-curve", required=True, type=Path)
    parser.add_argument("--comparison-id", default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--source-x-column", default=None)
    parser.add_argument("--source-y-column", default=None)
    parser.add_argument("--reference-x-column", default=None)
    parser.add_argument("--reference-y-column", default=None)
    parser.add_argument("--source-x-scale", type=float, default=None)
    parser.add_argument("--source-y-scale", type=float, default=None)
    parser.add_argument("--reference-x-scale", type=float, default=None)
    parser.add_argument("--reference-y-scale", type=float, default=None)
    parser.add_argument("--match-mode", choices=["interpolate", "exact"], default="interpolate")
    parser.add_argument("--max-pass-rmse-log-dec", type=float, default=0.2)
    parser.add_argument("--max-warn-rmse-log-dec", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = {
        "comparison_id": args.comparison_id,
        "source_state_path": args.source_state,
        "reference_curve_path": args.reference_curve,
        "source_x_column": args.source_x_column,
        "source_y_column": args.source_y_column,
        "reference_x_column": args.reference_x_column,
        "reference_y_column": args.reference_y_column,
        "source_x_scale": args.source_x_scale,
        "source_y_scale": args.source_y_scale,
        "reference_x_scale": args.reference_x_scale,
        "reference_y_scale": args.reference_y_scale,
        "match_mode": args.match_mode,
        "max_pass_rmse_log_dec": args.max_pass_rmse_log_dec,
        "max_warn_rmse_log_dec": args.max_warn_rmse_log_dec,
    }
    if args.run_root is not None:
        data["run_root"] = args.run_root
    state = run_golden_curve_comparison(GoldenCurveComparisonRequest.model_validate(data))
    print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if state.status == "completed" else 2)


if __name__ == "__main__":
    main()
