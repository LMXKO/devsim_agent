from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.task_spec import PROJECT_ROOT
from tcad_agent.tools.extended_device_sweep import (
    ExtendedDeviceFidelity,
    ExtendedDeviceRequest,
    ExtendedDeviceType,
    run_extended_device_sweep,
)


Q_OVER_K_BOLTZMANN = 11604.518121550082


class CurvePoint(BaseModel):
    voltage_v: float
    current_a: float


class SchottkyCalibrationCandidate(BaseModel):
    candidate_id: str
    barrier_height_ev: float
    ideality_factor: float
    series_resistance_ohm: float
    image_force_lowering_ev: float
    rmse_log_current_dec: float
    mae_log_current_dec: float
    max_abs_log_error_dec: float
    sign_mismatch_count: int
    score: float


class SchottkyCalibrationRequest(BaseModel):
    calibration_id: str | None = None
    run_root: Path = PROJECT_ROOT / "runs" / "agent_tools"
    target_curve_path: Path | None = None
    voltage_column: str = "voltage_v"
    current_column: str = "current_a"
    start: float = -0.2
    stop: float = 0.4
    step: float = Field(default=0.1, gt=0.0)
    temperature_k: float = Field(default=300.0, gt=0.0)
    area_cm2: float = Field(default=1.0e-8, gt=0.0)
    richardson_a_per_cm2_k2: float = Field(default=112.0, gt=0.0)
    barrier_values_ev: list[float] = Field(default_factory=lambda: [0.68, 0.70, 0.72, 0.74, 0.76])
    ideality_values: list[float] = Field(default_factory=lambda: [1.0, 1.08, 1.15])
    series_resistance_values_ohm: list[float] = Field(default_factory=lambda: [0.0, 5.0, 20.0])
    image_force_lowering_values_ev: list[float] = Field(default_factory=lambda: [0.0, 0.01, 0.02])
    trusted_barrier_height_ev: float = 0.72
    trusted_ideality_factor: float = 1.08
    trusted_series_resistance_ohm: float = 5.0
    trusted_image_force_lowering_ev: float = 0.01
    error_floor_current_a: float = Field(default=1.0e-20, gt=0.0)
    max_pass_rmse_log_current_dec: float = Field(default=0.15, gt=0.0)
    verify_with_devsim: bool = False
    devsim_timeout_seconds: float = Field(default=300.0, gt=0.0)


class SchottkyCalibrationState(BaseModel):
    tool_name: str = "schottky_iv_calibration"
    status: str
    calibration_id: str
    run_dir: str
    request: dict[str, Any]
    created_at: str
    updated_at: str
    target_curve: list[dict[str, float]] = Field(default_factory=list)
    candidates: list[SchottkyCalibrationCandidate] = Field(default_factory=list)
    best_candidate: SchottkyCalibrationCandidate | None = None
    verified_state_path: str | None = None
    final_summary: dict[str, Any] | None = None
    quality_report: dict[str, Any] | None = None
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_calibration_id() -> str:
    return datetime.now().strftime("schottky_cal_%Y%m%d_%H%M%S_%f")


def voltage_targets(start: float, stop: float, step: float) -> list[float]:
    direction = 1.0 if stop >= start else -1.0
    signed_step = abs(step) * direction
    value = start
    targets: list[float] = []
    while (value <= stop + abs(step) * 1e-9) if direction > 0 else (value >= stop - abs(step) * 1e-9):
        targets.append(round(value, 12))
        value += signed_step
    return targets


def thermal_voltage_v(temperature_k: float) -> float:
    return temperature_k / Q_OVER_K_BOLTZMANN


def thermionic_current_a(
    *,
    voltage_v: float,
    barrier_height_ev: float,
    ideality_factor: float,
    series_resistance_ohm: float,
    image_force_lowering_ev: float,
    area_cm2: float,
    richardson_a_per_cm2_k2: float,
    temperature_k: float,
) -> float:
    vt = thermal_voltage_v(temperature_k)
    effective_barrier_ev = max(barrier_height_ev - image_force_lowering_ev, 1.0e-6)
    saturation_current = (
        area_cm2
        * richardson_a_per_cm2_k2
        * temperature_k**2
        * math.exp(-effective_barrier_ev / vt)
    )
    if voltage_v < 0:
        return -saturation_current * (1.0 + abs(voltage_v) / 5.0)
    current = saturation_current * (math.exp(min(voltage_v / (ideality_factor * vt), 80.0)) - 1.0)
    if series_resistance_ohm <= 0:
        return current
    for _ in range(30):
        effective_voltage = voltage_v - current * series_resistance_ohm
        next_current = saturation_current * (math.exp(min(effective_voltage / (ideality_factor * vt), 80.0)) - 1.0)
        if abs(next_current - current) <= max(abs(current), 1.0e-30) * 1.0e-9:
            return next_current
        current = 0.5 * current + 0.5 * next_current
    return current


def simulate_curve(
    voltages: list[float],
    *,
    barrier_height_ev: float,
    ideality_factor: float,
    series_resistance_ohm: float,
    image_force_lowering_ev: float,
    request: SchottkyCalibrationRequest,
) -> list[CurvePoint]:
    return [
        CurvePoint(
            voltage_v=voltage,
            current_a=thermionic_current_a(
                voltage_v=voltage,
                barrier_height_ev=barrier_height_ev,
                ideality_factor=ideality_factor,
                series_resistance_ohm=series_resistance_ohm,
                image_force_lowering_ev=image_force_lowering_ev,
                area_cm2=request.area_cm2,
                richardson_a_per_cm2_k2=request.richardson_a_per_cm2_k2,
                temperature_k=request.temperature_k,
            ),
        )
        for voltage in voltages
    ]


def builtin_trusted_curve(request: SchottkyCalibrationRequest) -> list[CurvePoint]:
    voltages = voltage_targets(request.start, request.stop, request.step)
    return simulate_curve(
        voltages,
        barrier_height_ev=request.trusted_barrier_height_ev,
        ideality_factor=request.trusted_ideality_factor,
        series_resistance_ohm=request.trusted_series_resistance_ohm,
        image_force_lowering_ev=request.trusted_image_force_lowering_ev,
        request=request,
    )


def float_or_none(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def load_target_curve(request: SchottkyCalibrationRequest) -> list[CurvePoint]:
    if request.target_curve_path is None:
        return builtin_trusted_curve(request)
    path = request.target_curve_path
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw.get("points") if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            raise ValueError("JSON target curve must be a list or an object with a points list")
        points = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("JSON target curve points must be objects")
            voltage = float_or_none(row.get(request.voltage_column))
            current = float_or_none(row.get(request.current_column))
            if voltage is None or current is None:
                raise ValueError("Target curve contains a nonnumeric voltage/current value")
            points.append(CurvePoint(voltage_v=voltage, current_a=current))
        return sorted(points, key=lambda point: point.voltage_v)

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        points = []
        for row in reader:
            voltage = float_or_none(row.get(request.voltage_column))
            current = float_or_none(row.get(request.current_column))
            if voltage is None or current is None:
                raise ValueError("Target curve contains a nonnumeric voltage/current value")
            points.append(CurvePoint(voltage_v=voltage, current_a=current))
    return sorted(points, key=lambda point: point.voltage_v)


def log_current(value: float, floor: float) -> float:
    return math.log10(max(abs(value), floor))


def score_curve(
    target: list[CurvePoint],
    simulated: list[CurvePoint],
    *,
    floor: float,
) -> tuple[float, float, float, int]:
    if len(target) != len(simulated):
        raise ValueError("Target and simulated curves must have the same number of points")
    errors = []
    sign_mismatches = 0
    for left, right in zip(target, simulated):
        if abs(left.voltage_v - right.voltage_v) > 1e-9:
            raise ValueError("Target and simulated voltage grids do not match")
        if left.current_a * right.current_a < 0:
            sign_mismatches += 1
        errors.append(log_current(right.current_a, floor) - log_current(left.current_a, floor))
    mse = sum(error * error for error in errors) / max(len(errors), 1)
    mae = sum(abs(error) for error in errors) / max(len(errors), 1)
    max_abs = max((abs(error) for error in errors), default=0.0)
    return math.sqrt(mse), mae, max_abs, sign_mismatches


def evaluate_candidates(
    request: SchottkyCalibrationRequest,
    target: list[CurvePoint],
) -> list[SchottkyCalibrationCandidate]:
    voltages = [point.voltage_v for point in target]
    candidates = []
    index = 1
    for barrier in request.barrier_values_ev:
        for ideality in request.ideality_values:
            for series_resistance in request.series_resistance_values_ohm:
                for image_force in request.image_force_lowering_values_ev:
                    simulated = simulate_curve(
                        voltages,
                        barrier_height_ev=barrier,
                        ideality_factor=ideality,
                        series_resistance_ohm=series_resistance,
                        image_force_lowering_ev=image_force,
                        request=request,
                    )
                    rmse, mae, max_abs, sign_mismatches = score_curve(
                        target,
                        simulated,
                        floor=request.error_floor_current_a,
                    )
                    score = rmse + sign_mismatches
                    candidates.append(
                        SchottkyCalibrationCandidate(
                            candidate_id=f"schottky_candidate_{index:04d}",
                            barrier_height_ev=barrier,
                            ideality_factor=ideality,
                            series_resistance_ohm=series_resistance,
                            image_force_lowering_ev=image_force,
                            rmse_log_current_dec=rmse,
                            mae_log_current_dec=mae,
                            max_abs_log_error_dec=max_abs,
                            sign_mismatch_count=sign_mismatches,
                            score=score,
                        )
                    )
                    index += 1
    return sorted(candidates, key=lambda item: (item.score, item.max_abs_log_error_dec))


def write_curve_csv(path: Path, points: list[CurvePoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["voltage_v", "current_a", "abs_current_a"])
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "voltage_v": point.voltage_v,
                    "current_a": point.current_a,
                    "abs_current_a": abs(point.current_a),
                }
            )


def write_candidates_csv(path: Path, candidates: list[SchottkyCalibrationCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0].model_dump(mode="json").keys()))
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(candidate.model_dump(mode="json"))


def quality_report(
    request: SchottkyCalibrationRequest,
    best: SchottkyCalibrationCandidate | None,
    candidates: list[SchottkyCalibrationCandidate],
    verified_state_path: str | None,
) -> dict[str, Any]:
    issues = []
    if not candidates:
        issues.append({"code": "no_candidates", "severity": "error"})
    if best and best.rmse_log_current_dec > request.max_pass_rmse_log_current_dec:
        issues.append(
            {
                "code": "calibration_rmse_above_threshold",
                "severity": "warning",
                "rmse_log_current_dec": best.rmse_log_current_dec,
                "threshold": request.max_pass_rmse_log_current_dec,
            }
        )
    status = "failed" if any(issue["severity"] == "error" for issue in issues) else "suspicious" if issues else "passed"
    metrics = {
        "candidate_count": len(candidates),
        "best_rmse_log_current_dec": best.rmse_log_current_dec if best else None,
        "best_barrier_height_ev": best.barrier_height_ev if best else None,
        "best_ideality_factor": best.ideality_factor if best else None,
        "best_series_resistance_ohm": best.series_resistance_ohm if best else None,
        "best_image_force_lowering_ev": best.image_force_lowering_ev if best else None,
        "verified_with_devsim": verified_state_path is not None,
    }
    return {
        "status": status,
        "issues": issues,
        "metrics": metrics,
        "recommended_next_action": (
            "use calibrated Schottky parameters in residual-coupled DEVSIM sweeps"
            if status == "passed"
            else "expand the parameter grid or inspect the trusted target curve"
        ),
    }


def verify_best_with_devsim(
    request: SchottkyCalibrationRequest,
    best: SchottkyCalibrationCandidate,
    run_dir: Path,
) -> str | None:
    if not request.verify_with_devsim:
        return None
    target = load_target_curve(request)
    start = target[0].voltage_v
    stop = target[-1].voltage_v
    step = abs(target[1].voltage_v - target[0].voltage_v) if len(target) > 1 else request.step
    state = run_extended_device_sweep(
        ExtendedDeviceRequest(
            device_type=ExtendedDeviceType.SCHOTTKY_DIODE,
            fidelity=ExtendedDeviceFidelity.DEVSIM_1D,
            start=start,
            stop=stop,
            step=step,
            temperature_k=request.temperature_k,
            area_cm2=request.area_cm2,
            richardson_a_per_cm2_k2=request.richardson_a_per_cm2_k2,
            schottky_barrier_height_ev=best.barrier_height_ev,
            schottky_ideality_factor=best.ideality_factor,
            schottky_series_resistance_ohm=best.series_resistance_ohm,
            schottky_image_force_lowering_ev=best.image_force_lowering_ev,
            run_id=f"{request.calibration_id or 'schottky_calibration'}_devsim_best",
            run_root=run_dir / "verification_agent_tools",
            timeout_seconds=request.devsim_timeout_seconds,
        )
    )
    return str(Path(state.run_dir) / "state.json")


def run_schottky_calibration(request: SchottkyCalibrationRequest) -> SchottkyCalibrationState:
    calibration_id = request.calibration_id or default_calibration_id()
    actual_request = request.model_copy(update={"calibration_id": calibration_id})
    run_dir = actual_request.run_root / "schottky_calibration" / calibration_id
    state_path = run_dir / "state.json"
    now = utc_timestamp()
    state = SchottkyCalibrationState(
        status="running",
        calibration_id=calibration_id,
        run_dir=str(run_dir),
        request=actual_request.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
    )
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        target = load_target_curve(actual_request)
        candidates = evaluate_candidates(actual_request, target)
        best = candidates[0] if candidates else None
        verified_path = verify_best_with_devsim(actual_request, best, run_dir) if best else None
        target_path = run_dir / "target_curve.csv"
        candidates_path = run_dir / "candidates.csv"
        write_curve_csv(target_path, target)
        if candidates:
            write_candidates_csv(candidates_path, candidates)
        report = quality_report(actual_request, best, candidates, verified_path)
        summary = {
            "task": "schottky_iv_calibration",
            "status": "completed",
            "calibration_id": calibration_id,
            "best_candidate": best.model_dump(mode="json") if best else None,
            "metrics": report["metrics"],
            "artifacts": {
                "target_curve": str(target_path.resolve()),
                "candidates": str(candidates_path.resolve()),
                "state": str(state_path.resolve()),
                "verified_state": verified_path,
            },
        }
        state.status = "completed"
        state.target_curve = [point.model_dump(mode="json") for point in target]
        state.candidates = candidates
        state.best_candidate = best
        state.verified_state_path = verified_path
        state.final_summary = summary
        state.quality_report = report
        state.updated_at = utc_timestamp()
        state_path.write_text(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return state
    except Exception as exc:
        state.status = "failed"
        state.failure_reason = str(exc)
        state.updated_at = utc_timestamp()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
        return state
