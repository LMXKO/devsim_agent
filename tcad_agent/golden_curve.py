from __future__ import annotations

import csv
import json
import math
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


def compare_curves(
    source_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    source_x: str,
    source_y: str,
    reference_x: str,
    reference_y: str,
    floor: float,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    source_points = []
    for row in source_rows:
        x = float_or_none(row.get(source_x))
        y = float_or_none(row.get(source_y))
        if x is not None and y is not None:
            source_points.append((x, y))
    reference_points = []
    for row in reference_rows:
        x = float_or_none(row.get(reference_x))
        y = float_or_none(row.get(reference_y))
        if x is not None and y is not None:
            reference_points.append((x, y))
    if len(source_points) < 2 or len(reference_points) < 2:
        raise ValueError("both source and reference curves need at least two numeric points")
    source_by_x = {round(x, 12): y for x, y in source_points}
    matched = []
    for x, ref_y in reference_points:
        key = round(x, 12)
        if key not in source_by_x:
            continue
        src_y = source_by_x[key]
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
    sign_mismatches = sum(1 for point in matched if point["source_y"] * point["reference_y"] < 0)
    metrics = {
        "matched_points": float(len(matched)),
        "golden_curve_rmse_log_dec": math.sqrt(mse),
        "golden_curve_mae_log_dec": mae,
        "golden_curve_max_abs_log_error_dec": max(point["abs_log_error_dec"] for point in matched),
        "golden_curve_sign_mismatch_count": float(sign_mismatches),
    }
    return matched, metrics


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
        matched, metrics = compare_curves(
            source_rows,
            reference_rows,
            source_x=source_x,
            source_y=source_y,
            reference_x=reference_x,
            reference_y=reference_y,
            floor=request.error_floor,
        )
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
        state.matched_points = matched
        state.final_summary = {
            "task": "golden_curve_comparison",
            "status": state.status,
            "metrics": metrics,
            "artifacts": {
                "summary": str((run_dir / "summary.json").resolve()),
                "source_curve": str(source_curve.resolve()),
                "reference_curve": str(request.reference_curve_path.resolve()),
            },
        }
        state.quality_report = {
            "status": quality_status,
            "issues": issues,
            "metrics": metrics,
            "recommended_next_action": "accept golden/measured curve comparison" if quality_status == "passed" else "inspect model parameters and rerun calibration/repair",
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
