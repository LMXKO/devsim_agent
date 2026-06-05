from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from tcad_agent.task_spec import PROJECT_ROOT


Q_OVER_K_BOLTZMANN = 11604.518121550082


class ExtendedDeviceType(str, Enum):
    SCHOTTKY_DIODE = "schottky_diode"
    BJT_GUMMEL_OUTPUT = "bjt_gummel_output"
    JFET_TRANSFER_OUTPUT = "jfet_transfer_output"
    POWER_MOSFET_BV_RON = "power_mosfet_bv_ron"
    PHOTODIODE_IV = "photodiode_iv"


class ExtendedDeviceFidelity(str, Enum):
    COMPACT = "compact"
    DEVSIM_1D = "devsim_1d"


class ExtendedDeviceStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class ExtendedDeviceRequest(BaseModel):
    device_type: ExtendedDeviceType
    start: float | None = None
    stop: float | None = None
    step: float | None = Field(default=None, gt=0.0)
    temperature_k: float = Field(default=300.0, gt=0.0)
    area_cm2: float = Field(default=1.0e-8, gt=0.0)
    quality_min_points: int = Field(default=3, ge=1)
    run_id: str | None = None
    run_root: Path = PROJECT_ROOT / "runs" / "agent_tools"
    resume: bool = False
    fidelity: ExtendedDeviceFidelity = ExtendedDeviceFidelity.COMPACT
    timeout_seconds: float = Field(default=300.0, gt=0.0)

    schottky_barrier_height_ev: float = Field(default=0.72, gt=0.0)
    schottky_ideality_factor: float = Field(default=1.08, gt=0.0)
    richardson_a_per_cm2_k2: float = Field(default=112.0, gt=0.0)
    schottky_length_um: float = Field(default=0.2, gt=0.0)
    schottky_n_doping_cm3: float = Field(default=1.0e16, gt=0.0)
    schottky_contact_spacing_um: float = Field(default=0.002, gt=0.0)
    schottky_bulk_spacing_um: float = Field(default=0.01, gt=0.0)
    schottky_contact_model: str = "thermionic_emission"
    schottky_contact_coupling_mode: str = "residual"
    schottky_series_resistance_ohm: float = Field(default=0.0, ge=0.0)
    schottky_image_force_lowering_ev: float = Field(default=0.0, ge=0.0)
    schottky_auto_image_force_lowering: bool = False
    schottky_max_image_force_lowering_ev: float = Field(default=0.2, ge=0.0)

    bjt_beta: float = Field(default=100.0, gt=0.0)
    bjt_saturation_current_a: float = Field(default=1.0e-16, gt=0.0)
    bjt_early_voltage_v: float = Field(default=80.0, gt=0.0)

    jfet_idss_a: float = Field(default=1.0e-3, gt=0.0)
    jfet_pinch_off_voltage_v: float = Field(default=-2.0)

    power_mos_breakdown_voltage_v: float = Field(default=-60.0)
    power_mos_specific_ron_ohm_cm2: float = Field(default=5.0e-2, gt=0.0)
    power_mos_leakage_floor_a: float = Field(default=1.0e-10, gt=0.0)

    photodiode_dark_saturation_current_a: float = Field(default=1.0e-12, gt=0.0)
    optical_power_w: float = Field(default=1.0e-6, ge=0.0)
    responsivity_a_per_w: float = Field(default=0.5, gt=0.0)

    @model_validator(mode="after")
    def validate_request(self) -> "ExtendedDeviceRequest":
        if self.fidelity == ExtendedDeviceFidelity.DEVSIM_1D and self.device_type != ExtendedDeviceType.SCHOTTKY_DIODE:
            raise ValueError("fidelity=devsim_1d is currently supported for schottky_diode only")
        if self.schottky_contact_model not in {"equivalent_density", "thermionic_emission"}:
            raise ValueError("schottky_contact_model must be equivalent_density or thermionic_emission")
        if self.schottky_contact_coupling_mode not in {"reported", "residual"}:
            raise ValueError("schottky_contact_coupling_mode must be reported or residual")
        if self.device_type == ExtendedDeviceType.JFET_TRANSFER_OUTPUT and self.jfet_pinch_off_voltage_v >= 0:
            raise ValueError("jfet_pinch_off_voltage_v must be negative for the default n-channel convention")
        if self.device_type == ExtendedDeviceType.POWER_MOSFET_BV_RON and self.power_mos_breakdown_voltage_v >= 0:
            raise ValueError("power_mos_breakdown_voltage_v must be negative for reverse-bias BV extraction")
        return self


class ExtendedDeviceRunState(BaseModel):
    tool_name: str = "extended_device_sweep"
    status: ExtendedDeviceStatus
    run_id: str
    run_dir: str
    request: dict[str, Any]
    created_at: str
    updated_at: str
    final_summary: dict[str, Any] | None = None
    quality_report: dict[str, Any] | None = None
    next_action: str | None = None
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_run_id(device_type: ExtendedDeviceType) -> str:
    return f"{device_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def thermal_voltage_v(temperature_k: float) -> float:
    return temperature_k / Q_OVER_K_BOLTZMANN


def sweep_defaults(device_type: ExtendedDeviceType) -> tuple[float, float, float]:
    if device_type == ExtendedDeviceType.SCHOTTKY_DIODE:
        return -0.5, 0.8, 0.1
    if device_type == ExtendedDeviceType.BJT_GUMMEL_OUTPUT:
        return 0.55, 0.8, 0.025
    if device_type == ExtendedDeviceType.JFET_TRANSFER_OUTPUT:
        return -3.0, 0.0, 0.25
    if device_type == ExtendedDeviceType.POWER_MOSFET_BV_RON:
        return 0.0, -90.0, 5.0
    if device_type == ExtendedDeviceType.PHOTODIODE_IV:
        return -1.0, 0.8, 0.1
    raise ValueError(f"unsupported device_type: {device_type}")


def voltage_targets(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    direction = 1.0 if stop >= start else -1.0
    signed_step = abs(step) * direction
    value = start
    targets: list[float] = []
    while (value <= stop + abs(step) * 1e-9) if direction > 0 else (value >= stop - abs(step) * 1e-9):
        targets.append(round(value, 12))
        value += signed_step
    return targets


def request_sweep(request: ExtendedDeviceRequest) -> list[float]:
    default_start, default_stop, default_step = sweep_defaults(request.device_type)
    start = request.start if request.start is not None else default_start
    stop = request.stop if request.stop is not None else default_stop
    step = request.step if request.step is not None else default_step
    return voltage_targets(start, stop, step)


def interpolate_threshold(points: list[dict[str, Any]], x_key: str, y_key: str, threshold: float) -> float | None:
    ordered = sorted(points, key=lambda item: float(item[x_key]))
    previous = None
    for point in ordered:
        value = abs(float(point[y_key]))
        if previous is not None:
            previous_value = abs(float(previous[y_key]))
            if min(previous_value, value) <= threshold <= max(previous_value, value) and value != previous_value:
                x0 = float(previous[x_key])
                x1 = float(point[x_key])
                fraction = (threshold - previous_value) / (value - previous_value)
                return x0 + fraction * (x1 - x0)
        previous = point
    return None


def simulate_schottky(request: ExtendedDeviceRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vt = thermal_voltage_v(request.temperature_k)
    saturation_current = (
        request.area_cm2
        * request.richardson_a_per_cm2_k2
        * request.temperature_k**2
        * math.exp(-request.schottky_barrier_height_ev / vt)
    )
    points = []
    for voltage in request_sweep(request):
        if voltage >= 0:
            current = saturation_current * (math.exp(min(voltage / (request.schottky_ideality_factor * vt), 80.0)) - 1.0)
        else:
            current = -saturation_current * (1.0 + abs(voltage) / 5.0)
        points.append({"voltage_v": voltage, "current_a": current, "abs_current_a": abs(current)})
    extracted_barrier = -vt * math.log(
        max(saturation_current, 1e-300)
        / (request.area_cm2 * request.richardson_a_per_cm2_k2 * request.temperature_k**2)
    )
    metrics = {
        "device_type": request.device_type.value,
        "points": len(points),
        "saturation_current_a": saturation_current,
        "barrier_height_ev": extracted_barrier,
        "ideality_factor_estimate": request.schottky_ideality_factor,
        "reverse_leakage_current_a": abs(points[0]["current_a"]),
        "turn_on_voltage_at_1ua_v": interpolate_threshold(points, "voltage_v", "current_a", 1e-6),
    }
    return points, metrics


def simulate_bjt(request: ExtendedDeviceRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vt = thermal_voltage_v(request.temperature_k)
    points = []
    for vbe in request_sweep(request):
        collector = request.bjt_saturation_current_a * math.exp(min(vbe / vt, 80.0))
        base = collector / request.bjt_beta + 1e-14 * math.exp(min(vbe / (2.0 * vt), 80.0))
        emitter = collector + base
        beta = collector / base if base > 0 else None
        points.append(
            {
                "base_emitter_voltage_v": vbe,
                "collector_current_a": collector,
                "base_current_a": base,
                "emitter_current_a": emitter,
                "current_gain_beta": beta,
            }
        )
    beta_values = [float(point["current_gain_beta"]) for point in points if point.get("current_gain_beta")]
    metrics = {
        "device_type": request.device_type.value,
        "points": len(points),
        "current_gain_beta": sum(beta_values) / len(beta_values) if beta_values else None,
        "max_collector_current_a": max(point["collector_current_a"] for point in points),
        "max_base_current_a": max(point["base_current_a"] for point in points),
        "early_voltage_v": request.bjt_early_voltage_v,
        "gummel_slope_v_per_dec": math.log(10.0) * vt,
    }
    return points, metrics


def simulate_jfet(request: ExtendedDeviceRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vp = request.jfet_pinch_off_voltage_v
    points = []
    for vgs in request_sweep(request):
        if vgs <= vp:
            drain_current = 0.0
        elif vgs >= 0:
            drain_current = request.jfet_idss_a
        else:
            drain_current = request.jfet_idss_a * (1.0 - vgs / vp) ** 2
        gm = 2.0 * request.jfet_idss_a / abs(vp) * max(0.0, 1.0 - vgs / vp) if vp else 0.0
        points.append(
            {
                "gate_source_voltage_v": vgs,
                "drain_current_a": drain_current,
                "abs_drain_current_a": abs(drain_current),
                "transconductance_s": gm,
            }
        )
    metrics = {
        "device_type": request.device_type.value,
        "points": len(points),
        "idss_a": request.jfet_idss_a,
        "pinch_off_voltage_v": vp,
        "max_transconductance_s": max(point["transconductance_s"] for point in points),
        "min_drain_current_a": min(point["drain_current_a"] for point in points),
        "max_drain_current_a": max(point["drain_current_a"] for point in points),
    }
    return points, metrics


def simulate_power_mosfet(request: ExtendedDeviceRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bv = abs(request.power_mos_breakdown_voltage_v)
    threshold = 1e-6
    points = []
    for voltage in request_sweep(request):
        reverse = abs(min(voltage, 0.0))
        field = reverse / max(bv, 1e-9) * 3.0e5
        if reverse < bv:
            current = request.power_mos_leakage_floor_a / max((1.0 - reverse / bv) ** 2, 1e-6)
        else:
            current = threshold * math.exp(min((reverse - bv) / max(0.05 * bv, 1e-9), 40.0))
        points.append(
            {
                "drain_voltage_v": voltage,
                "off_current_a": current,
                "abs_off_current_a": abs(current),
                "electric_field_v_per_cm": field,
            }
        )
    breakdown = interpolate_threshold(points, "drain_voltage_v", "off_current_a", threshold)
    metrics = {
        "device_type": request.device_type.value,
        "points": len(points),
        "breakdown_voltage_v": breakdown if breakdown is not None else request.power_mos_breakdown_voltage_v,
        "specific_on_resistance_ohm_cm2": request.power_mos_specific_ron_ohm_cm2,
        "leakage_current_a": max(point["abs_off_current_a"] for point in points if point["drain_voltage_v"] <= 0),
        "max_electric_field_v_per_cm": max(point["electric_field_v_per_cm"] for point in points),
    }
    return points, metrics


def simulate_photodiode(request: ExtendedDeviceRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vt = thermal_voltage_v(request.temperature_k)
    photocurrent = request.responsivity_a_per_w * request.optical_power_w
    points = []
    for voltage in request_sweep(request):
        dark = request.photodiode_dark_saturation_current_a * (math.exp(min(voltage / vt, 80.0)) - 1.0)
        illuminated = dark - photocurrent
        points.append(
            {
                "voltage_v": voltage,
                "dark_current_a": dark,
                "illuminated_current_a": illuminated,
                "photocurrent_a": illuminated - dark,
            }
        )
    reverse_points = [point for point in points if point["voltage_v"] <= 0]
    metrics = {
        "device_type": request.device_type.value,
        "points": len(points),
        "dark_current_a": abs(reverse_points[0]["dark_current_a"]) if reverse_points else None,
        "photocurrent_a": abs(photocurrent),
        "responsivity_a_per_w": request.responsivity_a_per_w,
        "optical_power_w": request.optical_power_w,
        "open_circuit_voltage_v": vt * math.log(photocurrent / request.photodiode_dark_saturation_current_a + 1.0)
        if photocurrent > 0
        else 0.0,
    }
    return points, metrics


def simulate_device(request: ExtendedDeviceRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if request.device_type == ExtendedDeviceType.SCHOTTKY_DIODE:
        return simulate_schottky(request)
    if request.device_type == ExtendedDeviceType.BJT_GUMMEL_OUTPUT:
        return simulate_bjt(request)
    if request.device_type == ExtendedDeviceType.JFET_TRANSFER_OUTPUT:
        return simulate_jfet(request)
    if request.device_type == ExtendedDeviceType.POWER_MOSFET_BV_RON:
        return simulate_power_mosfet(request)
    if request.device_type == ExtendedDeviceType.PHOTODIODE_IV:
        return simulate_photodiode(request)
    raise ValueError(f"unsupported device_type: {request.device_type}")


def numeric_csv_value(value: str) -> Any:
    try:
        converted = float(value)
    except ValueError:
        return value
    return converted if math.isfinite(converted) else value


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: numeric_csv_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def parse_runner_stdout(stdout: str) -> dict[str, Any] | None:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(stdout[start : end + 1])
    except json.JSONDecodeError:
        return None


def tail(text: str, limit: int = 4000) -> str:
    return text[-limit:] if len(text) > limit else text


def run_schottky_devsim_1d(
    request: ExtendedDeviceRequest,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str], str]:
    default_start, default_stop, default_step = sweep_defaults(request.device_type)
    start = request.start if request.start is not None else default_start
    stop = request.stop if request.stop is not None else default_stop
    step = request.step if request.step is not None else default_step
    inner_root = run_dir / "devsim_runs"
    inner_run_id = "schottky_1d"
    command = [
        sys.executable,
        "-m",
        "tcad_agent.examples.schottky_1d.run",
        "--start",
        str(start),
        "--stop",
        str(stop),
        "--step",
        str(step),
        "--barrier-height-ev",
        str(request.schottky_barrier_height_ev),
        "--ideality-factor",
        str(request.schottky_ideality_factor),
        "--richardson-a-per-cm2-k2",
        str(request.richardson_a_per_cm2_k2),
        "--area-cm2",
        str(request.area_cm2),
        "--temperature-k",
        str(request.temperature_k),
        "--length-um",
        str(request.schottky_length_um),
        "--n-doping-cm3",
        str(request.schottky_n_doping_cm3),
        "--contact-spacing-um",
        str(request.schottky_contact_spacing_um),
        "--bulk-spacing-um",
        str(request.schottky_bulk_spacing_um),
        "--contact-model",
        request.schottky_contact_model,
        "--contact-coupling-mode",
        request.schottky_contact_coupling_mode,
        "--series-resistance-ohm",
        str(request.schottky_series_resistance_ohm),
        "--image-force-lowering-ev",
        str(request.schottky_image_force_lowering_ev),
        "--max-image-force-lowering-ev",
        str(request.schottky_max_image_force_lowering_ev),
        "--run-id",
        inner_run_id,
        "--run-root",
        str(inner_root),
    ]
    if request.schottky_auto_image_force_lowering:
        command.append("--auto-image-force-lowering")
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=request.timeout_seconds,
        check=False,
    )
    runner_result = parse_runner_stdout(completed.stdout)
    runner_dir = Path(runner_result["run_dir"]) if runner_result and runner_result.get("run_dir") else inner_root / "schottky_1d" / inner_run_id
    if completed.returncode != 0:
        raise RuntimeError(
            "Schottky DEVSIM runner failed: "
            f"returncode={completed.returncode}; stdout={tail(completed.stdout)}; stderr={tail(completed.stderr)}"
        )
    summary_path = runner_dir / "summary.json"
    csv_path = runner_dir / "sweep.csv"
    if not summary_path.exists() or not csv_path.exists():
        raise FileNotFoundError(f"Schottky DEVSIM runner did not produce expected artifacts under {runner_dir}")
    runner_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = read_csv_rows(csv_path)
    metrics = dict(runner_summary.get("metrics") or {})
    metrics.setdefault("device_type", request.device_type.value)
    metrics.setdefault("fidelity", ExtendedDeviceFidelity.DEVSIM_1D.value)
    metrics["tcad_solver_invoked"] = True
    metrics["tcad_runner"] = "tcad_agent.examples.schottky_1d.run"
    artifacts = {
        "devsim_csv": str(csv_path.resolve()),
        "tecplot": str((runner_dir / "device_tecplot.dat").resolve()),
        "devsim_log": str((runner_dir / "devsim.log").resolve()),
        "devsim_summary": str(summary_path.resolve()),
    }
    log_text = "\n".join(
        [
            "extended_device_sweep launched Schottky DEVSIM 1D runner",
            "command=" + json.dumps(command),
            "stdout_tail=" + tail(completed.stdout),
            "stderr_tail=" + tail(completed.stderr),
        ]
    )
    return rows, metrics, artifacts, log_text


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numeric_keys = [key for key in rows[0] if key.endswith("_v") or key.endswith("voltage_v")]
    current_keys = [key for key in rows[0] if "current" in key and key.endswith("_a")]
    x_key = numeric_keys[0] if numeric_keys else next(iter(rows[0]))
    y_key = current_keys[0] if current_keys else next(key for key in rows[0] if isinstance(rows[0][key], (int, float)))
    xs = [float(row[x_key]) for row in rows]
    ys = [float(row[y_key]) for row in rows]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-30)
    span_y = max(max_y - min_y, 1e-30)
    points = []
    for x, y in zip(xs, ys):
        px = 20.0 + 260.0 * (x - min_x) / span_x
        py = 180.0 - 160.0 * (y - min_y) / span_y
        points.append(f"{px:.2f},{py:.2f}")
    path.write_text(
        "\n".join(
            [
                '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="200" viewBox="0 0 300 200">',
                '<rect x="0" y="0" width="300" height="200" fill="white"/>',
                '<line x1="20" y1="180" x2="280" y2="180" stroke="black" stroke-width="1"/>',
                '<line x1="20" y1="20" x2="20" y2="180" stroke="black" stroke-width="1"/>',
                f'<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{" ".join(points)}"/>',
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def quality_report(request: ExtendedDeviceRequest, metrics: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if len(rows) < request.quality_min_points:
        issues.append({"code": "too_few_points", "severity": "warning", "points": len(rows)})
    for row_index, row in enumerate(rows):
        for key, value in row.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                issues.append({"code": "nonfinite_curve_value", "severity": "error", "row": row_index, "field": key})
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            issues.append({"code": "nonfinite_metric", "severity": "error", "metric": key})

    if request.device_type == ExtendedDeviceType.SCHOTTKY_DIODE:
        if request.fidelity == ExtendedDeviceFidelity.DEVSIM_1D:
            if not metrics.get("tcad_solver_invoked"):
                issues.append({"code": "devsim_solver_not_invoked", "severity": "error"})
            if not metrics.get("solver_backend"):
                issues.append({"code": "missing_solver_backend", "severity": "warning"})
            elif "thermionic" not in str(metrics.get("solver_backend")):
                issues.append({"code": "schottky_thermionic_backend_missing", "severity": "warning"})
            if "devsim_thermionic_contact_current_max_abs_a" not in metrics:
                issues.append({"code": "schottky_contact_current_metric_missing", "severity": "warning"})
            if not metrics.get("thermionic_residual_coupled"):
                issues.append({"code": "schottky_thermionic_not_residual_coupled", "severity": "warning"})
        barrier = metrics.get("barrier_height_ev")
        ideality = metrics.get("ideality_factor_estimate")
        if barrier is not None and not 0.2 <= float(barrier) <= 1.2:
            issues.append({"code": "schottky_barrier_height_out_of_range", "severity": "warning", "barrier_height_ev": barrier})
        if ideality is not None and not 0.8 <= float(ideality) <= 2.0:
            issues.append({"code": "schottky_ideality_factor_out_of_range", "severity": "warning", "ideality_factor": ideality})
    elif request.device_type == ExtendedDeviceType.BJT_GUMMEL_OUTPUT:
        beta = metrics.get("current_gain_beta")
        if beta is not None and float(beta) < 1.0:
            issues.append({"code": "bjt_current_gain_too_low", "severity": "warning", "current_gain_beta": beta})
    elif request.device_type == ExtendedDeviceType.JFET_TRANSFER_OUTPUT:
        if float(metrics.get("pinch_off_voltage_v") or 0.0) >= 0:
            issues.append({"code": "jfet_pinch_off_wrong_sign", "severity": "error"})
    elif request.device_type == ExtendedDeviceType.POWER_MOSFET_BV_RON:
        if float(metrics.get("breakdown_voltage_v") or 0.0) >= 0:
            issues.append({"code": "power_mos_breakdown_wrong_sign", "severity": "error"})
    elif request.device_type == ExtendedDeviceType.PHOTODIODE_IV:
        if float(metrics.get("photocurrent_a") or 0.0) <= 0:
            issues.append({"code": "photodiode_missing_photocurrent", "severity": "warning"})

    status = "failed" if any(issue["severity"] == "error" for issue in issues) else "suspicious" if issues else "passed"
    return {
        "status": status,
        "issues": issues,
        "metrics": metrics,
        "recommended_next_action": (
            "accept DEVSIM-backed Schottky thermionic-emission contact result as a higher-fidelity baseline"
            if status == "passed" and request.fidelity == ExtendedDeviceFidelity.DEVSIM_1D
            else "accept compact extended-device result as a planning baseline"
            if status == "passed"
            else "review compact extended-device warnings before using this result as evidence"
        ),
    }


def run_extended_device_sweep(request: ExtendedDeviceRequest) -> ExtendedDeviceRunState:
    run_id = request.run_id or default_run_id(request.device_type)
    run_dir = request.run_root / "extended_devices" / request.device_type.value / run_id
    state_path = run_dir / "state.json"
    now = utc_timestamp()
    state = ExtendedDeviceRunState(
        status=ExtendedDeviceStatus.COMPLETED,
        run_id=run_id,
        run_dir=str(run_dir),
        request=request.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
        next_action=f"{request.fidelity.value} extended-device sweep completed",
    )
    try:
        run_dir.mkdir(parents=True, exist_ok=not request.resume)
        extra_artifacts: dict[str, str] = {}
        if request.fidelity == ExtendedDeviceFidelity.DEVSIM_1D:
            rows, metrics, extra_artifacts, log_text = run_schottky_devsim_1d(request, run_dir)
        else:
            rows, metrics = simulate_device(request)
            log_text = f"extended_device_sweep device_type={request.device_type.value} fidelity=compact points={len(rows)}\n"
        csv_path = run_dir / "sweep.csv"
        plot_path = run_dir / "curve.svg"
        summary_path = run_dir / "summary.json"
        log_path = run_dir / "extended_device.log"
        write_csv(csv_path, rows)
        write_svg(plot_path, rows)
        log_path.write_text(log_text, encoding="utf-8")
        artifacts = {
            "csv": str(csv_path.resolve()),
            "plot": str(plot_path.resolve()),
            "log": str(log_path.resolve()),
            "summary": str(summary_path.resolve()),
        }
        artifacts.update(extra_artifacts)
        summary = {
            "task": "extended_device_sweep",
            "status": "completed",
            "device_type": request.device_type.value,
            "fidelity": request.fidelity.value,
            "parameters": request.model_dump(mode="json"),
            "metrics": metrics,
            "artifacts": artifacts,
        }
        report = quality_report(request, metrics, rows)
        summary.update({key: value for key, value in metrics.items() if key not in summary})
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        state.final_summary = summary
        state.quality_report = report
        state.updated_at = utc_timestamp()
        state_path.write_text(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
        return state
    except Exception as exc:
        state.status = ExtendedDeviceStatus.FAILED
        state.failure_reason = str(exc)
        state.next_action = "inspect extended-device failure"
        state.updated_at = utc_timestamp()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
        return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run executable sweeps for extended TCAD device templates.")
    parser.add_argument("--device-type", choices=[item.value for item in ExtendedDeviceType], required=True)
    parser.add_argument("--fidelity", choices=[item.value for item in ExtendedDeviceFidelity], default=None)
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--stop", type=float, default=None)
    parser.add_argument("--step", type=float, default=None)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--area-cm2", type=float, default=1.0e-8)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--schottky-contact-model", choices=["equivalent_density", "thermionic_emission"], default=None)
    parser.add_argument("--schottky-contact-coupling-mode", choices=["reported", "residual"], default=None)
    parser.add_argument("--schottky-series-resistance-ohm", type=float, default=None)
    parser.add_argument("--schottky-image-force-lowering-ev", type=float, default=None)
    parser.add_argument("--schottky-auto-image-force-lowering", action="store_true")
    parser.add_argument("--schottky-max-image-force-lowering-ev", type=float, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs" / "agent_tools")
    parser.add_argument("--request-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data: dict[str, Any] = {}
    if args.request_json:
        parsed = json.loads(args.request_json)
        if not isinstance(parsed, dict):
            raise ValueError("--request-json must decode to an object")
        data.update(parsed)
    data.update(
        {
            "device_type": args.device_type,
            "start": args.start if args.start is not None else data.get("start"),
            "stop": args.stop if args.stop is not None else data.get("stop"),
            "step": args.step if args.step is not None else data.get("step"),
            "temperature_k": args.temperature_k,
            "area_cm2": args.area_cm2,
            "timeout_seconds": args.timeout_seconds,
            "run_id": args.run_id if args.run_id is not None else data.get("run_id"),
            "run_root": args.run_root,
        }
    )
    if args.fidelity is not None:
        data["fidelity"] = args.fidelity
    if args.schottky_contact_model is not None:
        data["schottky_contact_model"] = args.schottky_contact_model
    if args.schottky_contact_coupling_mode is not None:
        data["schottky_contact_coupling_mode"] = args.schottky_contact_coupling_mode
    if args.schottky_series_resistance_ohm is not None:
        data["schottky_series_resistance_ohm"] = args.schottky_series_resistance_ohm
    if args.schottky_image_force_lowering_ev is not None:
        data["schottky_image_force_lowering_ev"] = args.schottky_image_force_lowering_ev
    if args.schottky_auto_image_force_lowering:
        data["schottky_auto_image_force_lowering"] = True
    if args.schottky_max_image_force_lowering_ev is not None:
        data["schottky_max_image_force_lowering_ev"] = args.schottky_max_image_force_lowering_ev
    request = ExtendedDeviceRequest.model_validate(data)
    state = run_extended_device_sweep(request)
    print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if state.status == ExtendedDeviceStatus.COMPLETED else 2)


if __name__ == "__main__":
    main()
