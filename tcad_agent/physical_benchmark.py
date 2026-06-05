from __future__ import annotations

import json
import math
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.physical_quality import oxide_capacitance_f_per_cm2


Q_OVER_K_BOLTZMANN = 11604.518121550082
DEFAULT_GOLDEN_PROFILES: dict[str, dict[str, dict[str, float]]] = {
    "extended_device_sweep:schottky_diode": {
        "barrier_height_ev": {"expected": 0.72, "relative_tolerance": 0.15},
        "ideality_factor_estimate": {"expected": 1.08, "relative_tolerance": 0.25},
    },
    "extended_device_sweep:bjt_gummel_output": {
        "current_gain_beta": {"expected": 100.0, "relative_tolerance": 0.35},
        "early_voltage_v": {"expected": 80.0, "relative_tolerance": 0.35},
    },
    "extended_device_sweep:jfet_transfer_output": {
        "pinch_off_voltage_v": {"expected": -2.0, "relative_tolerance": 0.25},
        "idss_a": {"expected": 1.0e-3, "relative_tolerance": 0.35},
    },
    "extended_device_sweep:power_mosfet_bv_ron": {
        "breakdown_voltage_v": {"expected": -60.0, "relative_tolerance": 0.35},
        "specific_on_resistance_ohm_cm2": {"expected": 5.0e-2, "relative_tolerance": 0.35},
    },
    "extended_device_sweep:photodiode_iv": {
        "responsivity_a_per_w": {"expected": 0.5, "relative_tolerance": 0.25},
        "photocurrent_a": {"expected": 5.0e-7, "relative_tolerance": 0.35},
    },
}


class BenchmarkStatus(str, Enum):
    PASSED = "passed"
    SUSPICIOUS = "suspicious"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class BenchmarkSeverity(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    ERROR = "error"


class BenchmarkCheck(BaseModel):
    code: str
    severity: BenchmarkSeverity
    message: str
    observed: dict[str, Any] = Field(default_factory=dict)
    expected: dict[str, Any] = Field(default_factory=dict)


class PhysicalBenchmarkResult(BaseModel):
    tool_name: str = "physical_benchmark"
    status: BenchmarkStatus
    source_state_path: str
    source_tool_name: str | None = None
    benchmark_path: str | None = None
    checks: list[BenchmarkCheck] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def resolve_state_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"State path does not exist: {path}")
    for name in [
        "state.json",
        "mission_state.json",
        "supervisor_state.json",
        "optimization_state.json",
        "sweep_state.json",
        "repair_execution_state.json",
    ]:
        candidate = path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No supported TCAD state file found under: {path}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def nested_get(data: dict[str, Any] | None, path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def merged_metrics(state: dict[str, Any]) -> dict[str, Any]:
    summary = state.get("final_summary") or {}
    metrics = dict(summary.get("metrics") or {})
    for key, value in summary.items():
        if key not in {"artifacts", "metrics"}:
            metrics.setdefault(key, value)
    quality_metrics = ((state.get("quality_report") or {}).get("metrics") or {})
    metrics.update(quality_metrics)
    return metrics


def merged_parameters(state: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    final_summary = state.get("final_summary") or {}
    for source in [state.get("request"), final_summary.get("parameters"), state.get("parameters")]:
        if isinstance(source, dict):
            params.update(source)
    return params


def state_deck_spec(state: dict[str, Any]) -> dict[str, Any]:
    deck = state.get("tcad_deck_spec")
    if isinstance(deck, dict):
        return deck
    request = state.get("request") or {}
    if isinstance(request, dict) and isinstance(request.get("tcad_deck_spec"), dict):
        return request["tcad_deck_spec"]
    return {}


def relative_error(observed: float, expected: float) -> float:
    return abs(observed - expected) / max(abs(expected), 1e-300)


def pass_check(code: str, message: str, observed: dict[str, Any] | None = None, expected: dict[str, Any] | None = None) -> BenchmarkCheck:
    return BenchmarkCheck(
        code=code,
        severity=BenchmarkSeverity.PASS,
        message=message,
        observed=observed or {},
        expected=expected or {},
    )


def warn_check(code: str, message: str, observed: dict[str, Any] | None = None, expected: dict[str, Any] | None = None) -> BenchmarkCheck:
    return BenchmarkCheck(
        code=code,
        severity=BenchmarkSeverity.WARNING,
        message=message,
        observed=observed or {},
        expected=expected or {},
    )


def error_check(code: str, message: str, observed: dict[str, Any] | None = None, expected: dict[str, Any] | None = None) -> BenchmarkCheck:
    return BenchmarkCheck(
        code=code,
        severity=BenchmarkSeverity.ERROR,
        message=message,
        observed=observed or {},
        expected=expected or {},
    )


def range_check(
    code: str,
    value: Any,
    *,
    low: float | None,
    high: float | None,
    units: str,
    pass_message: str,
    warning_message: str,
    missing_message: str | None = None,
) -> BenchmarkCheck | None:
    numeric = float_or_none(value)
    expected = {"low": low, "high": high, "units": units}
    if numeric is None:
        if missing_message is None:
            return None
        return warn_check(f"{code}_missing", missing_message, {"value": value}, expected)
    if low is not None and numeric < low:
        return warn_check(f"{code}_below_range", warning_message, {"value": numeric, "units": units}, expected)
    if high is not None and numeric > high:
        return warn_check(f"{code}_above_range", warning_message, {"value": numeric, "units": units}, expected)
    return pass_check(code, pass_message, {"value": numeric, "units": units}, expected)


def thermal_voltage_v(temperature_k: float) -> float:
    return temperature_k / Q_OVER_K_BOLTZMANN


def thermal_subthreshold_swing_mv_dec(temperature_k: float) -> float:
    return math.log(10.0) * thermal_voltage_v(temperature_k) * 1000.0


def generic_quality_checks(state: dict[str, Any]) -> list[BenchmarkCheck]:
    quality = state.get("quality_report") or {}
    quality_status = quality.get("status")
    if not quality_status:
        return []
    if quality_status == "passed":
        return [pass_check("quality_report_passed", "Existing quality report passed.")]
    if quality_status == "failed":
        return [
            error_check(
                "quality_report_failed",
                "Existing quality report failed; physical benchmark cannot overrule failed numeric/artifact quality.",
                {"quality_status": quality_status, "issues": quality.get("issues") or []},
            )
        ]
    return [
        warn_check(
            "quality_report_suspicious",
            "Existing quality report is suspicious; use benchmark results as supporting evidence only.",
            {"quality_status": quality_status, "issues": quality.get("issues") or []},
        )
    ]


def benchmark_pn(metrics: dict[str, Any], params: dict[str, Any]) -> list[BenchmarkCheck]:
    checks: list[BenchmarkCheck] = []
    ideality = float_or_none(metrics.get("ideality_factor_estimate"))
    if ideality is not None:
        if 0.8 <= ideality <= 2.5:
            checks.append(
                pass_check(
                    "pn_ideality_factor_broad_silicon_range",
                    "Extracted ideality factor is inside a broad silicon diode engineering range.",
                    {"ideality_factor": ideality},
                    {"typical_range": [0.8, 2.5]},
                )
            )
        elif 0.5 <= ideality <= 5.0:
            checks.append(
                warn_check(
                    "pn_ideality_factor_marginal",
                    "Extracted ideality factor is marginal; inspect the exponential-fit window and series resistance.",
                    {"ideality_factor": ideality},
                    {"broad_range": [0.5, 5.0], "preferred_range": [0.8, 2.5]},
                )
            )
        else:
            checks.append(
                error_check(
                    "pn_ideality_factor_unphysical",
                    "Extracted ideality factor is outside a broad diode sanity range.",
                    {"ideality_factor": ideality},
                    {"broad_range": [0.5, 5.0]},
                )
            )

    rectification = float_or_none(metrics.get("rectification_ratio_final_to_leakage"))
    if rectification is not None:
        if rectification >= 10.0:
            checks.append(
                pass_check(
                    "pn_rectification_ratio_ok",
                    "Forward/reverse current ratio is large enough for a basic diode sanity check.",
                    {"rectification_ratio": rectification},
                    {"minimum": 10.0},
                )
            )
        else:
            checks.append(
                warn_check(
                    "pn_rectification_ratio_low",
                    "Forward/reverse current ratio is low; check leakage, contacts, doping, or bias polarity.",
                    {"rectification_ratio": rectification},
                    {"minimum": 10.0},
                )
            )

    turn_on = range_check(
        "pn_turn_on_voltage",
        metrics.get("turn_on_voltage_at_1ua_v"),
        low=0.05,
        high=1.5,
        units="V",
        pass_message="Turn-on voltage is in a broad silicon-diode sanity range.",
        warning_message="Turn-on voltage is outside a broad silicon-diode sanity range.",
    )
    if turn_on:
        checks.append(turn_on)

    temperature_k = float_or_none(params.get("temperature_k")) or 300.0
    checks.append(
        pass_check(
            "pn_thermal_voltage_reference",
            "Thermal voltage reference computed for interpreting exponential IV slope.",
            {"temperature_k": temperature_k, "thermal_voltage_v": thermal_voltage_v(temperature_k)},
            {"formula": "Vt = kT/q"},
        )
    )
    return checks


def benchmark_mos_capacitor(metrics: dict[str, Any], params: dict[str, Any]) -> list[BenchmarkCheck]:
    checks: list[BenchmarkCheck] = []
    oxide_thickness_nm = float_or_none(params.get("oxide_thickness_nm"))
    max_cap = float_or_none(metrics.get("max_capacitance_f_per_cm2"))
    min_cap = float_or_none(metrics.get("min_capacitance_f_per_cm2"))
    final_cap = float_or_none(metrics.get("final_capacitance_f_per_cm2"))
    if oxide_thickness_nm is not None and oxide_thickness_nm > 0 and max_cap is not None:
        cox = oxide_capacitance_f_per_cm2(oxide_thickness_nm)
        ratio = max_cap / cox if cox else None
        if ratio is not None and 0.02 <= ratio <= 1.2:
            checks.append(
                pass_check(
                    "moscap_capacitance_below_cox",
                    "Maximum simulated capacitance is consistent with the oxide capacitance upper bound.",
                    {"max_capacitance_f_per_cm2": max_cap, "cox_f_per_cm2": cox, "ratio_to_cox": ratio},
                    {"ratio_range": [0.02, 1.2]},
                )
            )
        elif ratio is not None and ratio <= 1.5:
            checks.append(
                warn_check(
                    "moscap_capacitance_near_cox_limit",
                    "Maximum capacitance is near the oxide-capacitance limit; verify area normalization and oxide thickness.",
                    {"max_capacitance_f_per_cm2": max_cap, "cox_f_per_cm2": cox, "ratio_to_cox": ratio},
                    {"soft_upper_ratio": 1.2, "hard_upper_ratio": 1.5},
                )
            )
        else:
            checks.append(
                error_check(
                    "moscap_capacitance_exceeds_cox",
                    "Maximum capacitance exceeds the oxide-capacitance benchmark by a large margin.",
                    {"max_capacitance_f_per_cm2": max_cap, "cox_f_per_cm2": cox, "ratio_to_cox": ratio},
                    {"hard_upper_ratio": 1.5},
                )
            )
    elif max_cap is not None:
        checks.append(
            warn_check(
                "moscap_missing_oxide_thickness_for_cox",
                "Oxide thickness was unavailable, so C_ox benchmark could not be computed.",
                {"max_capacitance_f_per_cm2": max_cap, "oxide_thickness_nm": oxide_thickness_nm},
            )
        )

    for key, value in [
        ("min_capacitance_f_per_cm2", min_cap),
        ("max_capacitance_f_per_cm2", max_cap),
        ("final_capacitance_f_per_cm2", final_cap),
    ]:
        if value is not None and value <= 0:
            checks.append(
                error_check(
                    "moscap_nonpositive_capacitance",
                    "MOS capacitance benchmark requires positive capacitance values.",
                    {"metric": key, "value_f_per_cm2": value},
                )
            )
    if min_cap is not None and max_cap is not None:
        if min_cap <= max_cap:
            checks.append(
                pass_check(
                    "moscap_capacitance_ordering_ok",
                    "Minimum capacitance does not exceed maximum capacitance.",
                    {"min_capacitance_f_per_cm2": min_cap, "max_capacitance_f_per_cm2": max_cap},
                )
            )
        else:
            checks.append(
                error_check(
                    "moscap_capacitance_ordering_invalid",
                    "Minimum capacitance exceeds maximum capacitance.",
                    {"min_capacitance_f_per_cm2": min_cap, "max_capacitance_f_per_cm2": max_cap},
                )
            )
        if min_cap > 0 and max_cap > 0:
            ratio = max_cap / min_cap
            voltage_range = metrics.get("voltage_range_v")
            span = None
            if isinstance(voltage_range, (list, tuple)) and len(voltage_range) == 2:
                left = float_or_none(voltage_range[0])
                right = float_or_none(voltage_range[1])
                if left is not None and right is not None:
                    span = abs(right - left)
            if span is not None and span >= 1.0 and ratio < 1.02:
                checks.append(
                    warn_check(
                        "moscap_cv_dynamic_range_too_low",
                        "C-V curve is nearly flat across the requested voltage span; inspect bias window, derivative noise, or fixed charge.",
                        {"capacitance_dynamic_range": ratio, "voltage_span_v": span},
                    )
                )
            elif ratio >= 1.02:
                checks.append(
                    pass_check(
                        "moscap_cv_dynamic_range_present",
                        "C-V curve has measurable capacitance variation.",
                        {"capacitance_dynamic_range": ratio},
                    )
                )
    fixed_shift = float_or_none(metrics.get("fixed_charge_voltage_shift_v"))
    if fixed_shift is not None and fixed_shift != 0:
        voltage_range = metrics.get("voltage_range_v")
        span = None
        if isinstance(voltage_range, (list, tuple)) and len(voltage_range) == 2:
            left = float_or_none(voltage_range[0])
            right = float_or_none(voltage_range[1])
            if left is not None and right is not None:
                span = abs(right - left)
        if span is not None and abs(fixed_shift) > span:
            checks.append(
                warn_check(
                    "moscap_fixed_charge_shift_exceeds_sweep",
                    "Fixed-charge voltage shift is larger than the C-V sweep span.",
                    {"fixed_charge_voltage_shift_v": fixed_shift, "voltage_span_v": span},
                )
            )
        else:
            checks.append(
                pass_check(
                    "moscap_fixed_charge_shift_recorded",
                    "Fixed oxide charge was translated into an equivalent voltage-shift metric.",
                    {"fixed_charge_voltage_shift_v": fixed_shift},
                )
            )
    return checks


def benchmark_diode_breakdown(metrics: dict[str, Any], params: dict[str, Any]) -> list[BenchmarkCheck]:
    checks = benchmark_pn(metrics, params)
    leakage = float_or_none(metrics.get("leakage_abs_current_at_target_a"))
    leakage_limit = float_or_none(params.get("quality_max_leakage_abs_current_a")) or 1e-3
    if leakage is not None:
        if leakage <= leakage_limit:
            checks.append(
                pass_check(
                    "diode_leakage_below_policy",
                    "Reverse leakage is below the configured or default leakage benchmark.",
                    {"leakage_abs_current_at_target_a": leakage},
                    {"maximum_a": leakage_limit},
                )
            )
        else:
            checks.append(
                warn_check(
                    "diode_leakage_above_policy",
                    "Reverse leakage exceeds the benchmark; verify lifetime, doping, contacts, and reverse-bias range.",
                    {"leakage_abs_current_at_target_a": leakage},
                    {"maximum_a": leakage_limit},
                )
            )

    breakdown_voltage = float_or_none(metrics.get("breakdown_voltage_at_threshold_v"))
    if breakdown_voltage is not None:
        if breakdown_voltage < 0:
            checks.append(
                pass_check(
                    "diode_breakdown_voltage_reverse_polarity",
                    "Extracted breakdown voltage has reverse-bias polarity.",
                    {"breakdown_voltage_at_threshold_v": breakdown_voltage},
                    {"expected_sign": "negative"},
                )
            )
        else:
            checks.append(
                error_check(
                    "diode_breakdown_voltage_wrong_polarity",
                    "Breakdown voltage should be negative for this reverse-bias sweep convention.",
                    {"breakdown_voltage_at_threshold_v": breakdown_voltage},
                    {"expected_sign": "negative"},
                )
            )
    elif params.get("require_breakdown"):
        checks.append(
            warn_check(
                "diode_breakdown_not_reached",
                "Breakdown was required but no threshold crossing was extracted.",
                {"breakdown_detected": metrics.get("breakdown_detected")},
            )
        )

    violations = int(float_or_none(metrics.get("reverse_current_shape_violations")) or 0)
    if violations == 0 and metrics.get("reverse_current_shape_violations") is not None:
        checks.append(
            pass_check(
                "diode_reverse_current_shape_ok",
                "Reverse current magnitude is monotonic over the sampled reverse-bias curve.",
                {"reverse_current_shape_violations": violations},
            )
        )
    elif violations > 0:
        checks.append(
            warn_check(
                "diode_reverse_current_shape_nonmonotonic",
                "Reverse current magnitude is not monotonic; refine bias steps around the suspicious segment.",
                {"reverse_current_shape_violations": violations},
            )
        )
    return checks


def benchmark_mosfet(metrics: dict[str, Any], params: dict[str, Any]) -> list[BenchmarkCheck]:
    checks: list[BenchmarkCheck] = []
    temperature_k = float_or_none(params.get("temperature_k")) or 300.0
    thermal_ss = thermal_subthreshold_swing_mv_dec(temperature_k)
    ss = float_or_none(metrics.get("subthreshold_swing_mv_dec"))
    if ss is not None:
        if thermal_ss <= ss <= 500.0:
            checks.append(
                pass_check(
                    "mosfet_subthreshold_swing_engineering_range",
                    "Subthreshold swing is above the thermal limit and inside a broad usable range.",
                    {"subthreshold_swing_mv_dec": ss, "thermal_limit_mv_dec": thermal_ss},
                    {"range_mv_dec": [thermal_ss, 500.0]},
                )
            )
        elif ss < thermal_ss:
            checks.append(
                warn_check(
                    "mosfet_subthreshold_swing_below_thermal_limit",
                    "Subthreshold swing is below the thermal limit; inspect extraction, units, and current floor.",
                    {"subthreshold_swing_mv_dec": ss, "thermal_limit_mv_dec": thermal_ss},
                    {"formula": "ln(10) kT/q"},
                )
            )
        else:
            checks.append(
                warn_check(
                    "mosfet_subthreshold_swing_large",
                    "Subthreshold swing is large for a usable transfer curve; inspect short-channel/mesh/model settings.",
                    {"subthreshold_swing_mv_dec": ss},
                    {"soft_upper_mv_dec": 500.0},
                )
            )

    ratio = float_or_none(metrics.get("ion_ioff_ratio"))
    min_ratio = float_or_none(params.get("quality_min_ion_ioff_ratio")) or 10.0
    if ratio is not None:
        if ratio >= min_ratio:
            checks.append(
                pass_check(
                    "mosfet_ion_ioff_ratio_ok",
                    "Ion/Ioff ratio meets the configured or default benchmark.",
                    {"ion_ioff_ratio": ratio},
                    {"minimum": min_ratio},
                )
            )
        else:
            checks.append(
                warn_check(
                    "mosfet_ion_ioff_ratio_low",
                    "Ion/Ioff ratio is below benchmark; inspect leakage, gate sweep, and threshold definition.",
                    {"ion_ioff_ratio": ratio},
                    {"minimum": min_ratio},
                )
            )

    vth = float_or_none(metrics.get("vth_at_threshold_current_v"))
    gate_start = float_or_none(params.get("gate_start"))
    gate_stop = float_or_none(params.get("gate_stop"))
    if vth is not None and gate_start is not None and gate_stop is not None:
        low, high = sorted([gate_start, gate_stop])
        if low <= vth <= high:
            checks.append(
                pass_check(
                    "mosfet_vth_inside_gate_sweep",
                    "Extracted threshold voltage lies inside the requested gate sweep.",
                    {"vth_at_threshold_current_v": vth, "gate_range_v": [low, high]},
                )
            )
        else:
            checks.append(
                warn_check(
                    "mosfet_vth_outside_gate_sweep",
                    "Extracted threshold voltage lies outside the requested gate sweep.",
                    {"vth_at_threshold_current_v": vth, "gate_range_v": [low, high]},
                )
            )

    for key in ["ion_current_a", "ioff_current_a", "max_transconductance_s"]:
        value = float_or_none(metrics.get(key))
        if value is not None and value < 0:
            checks.append(
                warn_check(
                    "mosfet_negative_extracted_metric",
                    "Extracted MOSFET magnitude metric is negative; check sign convention and absolute-value extraction.",
                    {"metric": key, "value": value},
                )
            )
    negative_idvd = int(float_or_none(metrics.get("idvd_negative_differential_segments")) or 0)
    if negative_idvd > 0:
        checks.append(
            warn_check(
                "mosfet_idvd_negative_differential_segments",
                "Id-Vd output curve has decreasing-current segments; inspect bias continuation, mesh, or sign convention.",
                {"idvd_negative_differential_segments": negative_idvd},
            )
        )
    kink_jumps = int(float_or_none(metrics.get("idvd_kink_slope_jumps")) or 0)
    if kink_jumps > 0:
        checks.append(
            warn_check(
                "mosfet_idvd_kink_suspected",
                "Id-Vd slope has abrupt jumps consistent with kink behavior or numerical artifacts.",
                {"idvd_kink_slope_jumps": kink_jumps},
            )
        )
    saturation_ratio = float_or_none(metrics.get("idvd_saturation_ratio"))
    drain_span = float_or_none(metrics.get("idvd_max_drain_span_v"))
    if saturation_ratio is not None and drain_span is not None and drain_span >= 0.8:
        if saturation_ratio <= 0.75:
            checks.append(
                pass_check(
                    "mosfet_idvd_saturation_shape_ok",
                    "High-drain Id-Vd slope is lower than the maximum slope, consistent with saturation trend.",
                    {"idvd_saturation_ratio": saturation_ratio, "drain_span_v": drain_span},
                )
            )
        else:
            checks.append(
                warn_check(
                    "mosfet_idvd_saturation_not_observed",
                    "Id-Vd output curve does not show a clear saturation trend over the requested drain range.",
                    {"idvd_saturation_ratio": saturation_ratio, "drain_span_v": drain_span},
                )
            )
    return checks


def benchmark_extended_device(metrics: dict[str, Any], params: dict[str, Any]) -> list[BenchmarkCheck]:
    device_type = str(metrics.get("device_type") or params.get("device_type") or "")
    checks: list[BenchmarkCheck] = []
    if device_type == "schottky_diode":
        barrier = range_check(
            "schottky_barrier_height",
            metrics.get("barrier_height_ev"),
            low=0.2,
            high=1.2,
            units="eV",
            pass_message="Schottky barrier height is inside a broad silicon-metal sanity range.",
            warning_message="Schottky barrier height is outside a broad silicon-metal sanity range.",
            missing_message="Schottky barrier height was not extracted.",
        )
        ideality = range_check(
            "schottky_ideality_factor",
            metrics.get("ideality_factor_estimate"),
            low=0.8,
            high=2.0,
            units="",
            pass_message="Schottky ideality factor is in a broad compact-model sanity range.",
            warning_message="Schottky ideality factor is outside a broad compact-model sanity range.",
        )
        checks.extend([check for check in [barrier, ideality] if check is not None])
        if metrics.get("fidelity") == "devsim_1d" or params.get("fidelity") == "devsim_1d":
            if metrics.get("tcad_solver_invoked") and metrics.get("solver_backend"):
                checks.append(
                    pass_check(
                        "schottky_devsim_solver_invoked",
                        "Schottky sweep records a DEVSIM-backed thermionic-emission contact solve.",
                        {
                            "solver_backend": metrics.get("solver_backend"),
                            "tcad_runner": metrics.get("tcad_runner"),
                        },
                    )
                )
            else:
                checks.append(
                    error_check(
                        "schottky_devsim_solver_missing",
                        "Schottky sweep requested DEVSIM fidelity but solver invocation metadata is missing.",
                        {
                            "fidelity": metrics.get("fidelity") or params.get("fidelity"),
                            "solver_backend": metrics.get("solver_backend"),
                            "tcad_solver_invoked": metrics.get("tcad_solver_invoked"),
                        },
                    )
                )
            if metrics.get("schottky_contact_model") == "thermionic_emission" and (
                "devsim_thermionic_contact_current_max_abs_a" in metrics
            ):
                checks.append(
                    pass_check(
                        "schottky_thermionic_contact_model_registered",
                        "Schottky run exposes thermionic-emission contact-model metrics.",
                        {
                            "schottky_contact_model": metrics.get("schottky_contact_model"),
                            "devsim_thermionic_contact_current_max_abs_a": metrics.get(
                                "devsim_thermionic_contact_current_max_abs_a"
                            ),
                        },
                    )
                )
            else:
                checks.append(
                    warn_check(
                        "schottky_thermionic_contact_model_not_confirmed",
                        "DEVSIM Schottky run did not expose thermionic-emission contact-model metrics.",
                        {
                            "schottky_contact_model": metrics.get("schottky_contact_model"),
                            "has_contact_current_metric": "devsim_thermionic_contact_current_max_abs_a" in metrics,
                        },
                    )
                )
            if metrics.get("thermionic_residual_coupled"):
                checks.append(
                    pass_check(
                        "schottky_thermionic_residual_coupled",
                        "Thermionic-emission contact current is coupled into the electron continuity contact residual.",
                        {"schottky_contact_coupling_mode": metrics.get("schottky_contact_coupling_mode")},
                    )
                )
            else:
                checks.append(
                    warn_check(
                        "schottky_thermionic_residual_not_coupled",
                        "Thermionic-emission contact current is reported but not coupled into the electron continuity residual.",
                        {"schottky_contact_coupling_mode": metrics.get("schottky_contact_coupling_mode")},
                    )
                )
    elif device_type == "bjt_gummel_output":
        beta = range_check(
            "bjt_current_gain",
            metrics.get("current_gain_beta"),
            low=1.0,
            high=1000.0,
            units="",
            pass_message="BJT current gain is inside a broad compact-model sanity range.",
            warning_message="BJT current gain is outside a broad compact-model sanity range.",
            missing_message="BJT current gain was not extracted.",
        )
        if beta:
            checks.append(beta)
    elif device_type == "jfet_transfer_output":
        pinch = float_or_none(metrics.get("pinch_off_voltage_v"))
        if pinch is None:
            checks.append(warn_check("jfet_pinch_off_missing", "JFET pinch-off voltage was not extracted."))
        elif pinch < 0:
            checks.append(
                pass_check(
                    "jfet_pinch_off_voltage_sign_ok",
                    "JFET pinch-off voltage is negative for the default n-channel convention.",
                    {"pinch_off_voltage_v": pinch},
                )
            )
        else:
            checks.append(
                error_check(
                    "jfet_pinch_off_voltage_wrong_sign",
                    "JFET pinch-off voltage should be negative for the default n-channel convention.",
                    {"pinch_off_voltage_v": pinch},
                )
            )
    elif device_type == "power_mosfet_bv_ron":
        bv = float_or_none(metrics.get("breakdown_voltage_v"))
        ron = float_or_none(metrics.get("specific_on_resistance_ohm_cm2"))
        if bv is None:
            checks.append(warn_check("power_mos_breakdown_missing", "Power MOSFET breakdown voltage was not extracted."))
        elif bv < 0:
            checks.append(
                pass_check(
                    "power_mos_breakdown_voltage_sign_ok",
                    "Power MOSFET breakdown voltage has reverse-bias polarity.",
                    {"breakdown_voltage_v": bv},
                )
            )
        else:
            checks.append(
                error_check(
                    "power_mos_breakdown_voltage_wrong_sign",
                    "Power MOSFET breakdown voltage should be negative in the reverse-bias convention.",
                    {"breakdown_voltage_v": bv},
                )
            )
        if ron is not None and ron > 0:
            checks.append(
                pass_check(
                    "power_mos_specific_ron_positive",
                    "Specific on-resistance is positive.",
                    {"specific_on_resistance_ohm_cm2": ron},
                )
            )
    elif device_type == "photodiode_iv":
        photocurrent = float_or_none(metrics.get("photocurrent_a"))
        responsivity = float_or_none(metrics.get("responsivity_a_per_w"))
        if photocurrent is not None and photocurrent > 0:
            checks.append(pass_check("photodiode_photocurrent_positive", "Photodiode photocurrent is positive.", {"photocurrent_a": photocurrent}))
        else:
            checks.append(warn_check("photodiode_photocurrent_missing", "Photodiode photocurrent was not positive.", {"photocurrent_a": photocurrent}))
        if responsivity is not None and 0 < responsivity <= 1.5:
            checks.append(
                pass_check(
                    "photodiode_responsivity_range_ok",
                    "Photodiode responsivity is inside a broad silicon detector sanity range.",
                    {"responsivity_a_per_w": responsivity},
                    {"range_a_per_w": [0.0, 1.5]},
                )
            )
    return checks


def benchmark_schottky_calibration(metrics: dict[str, Any], params: dict[str, Any]) -> list[BenchmarkCheck]:
    checks: list[BenchmarkCheck] = []
    barrier = range_check(
        "schottky_calibration_barrier_height",
        metrics.get("best_barrier_height_ev"),
        low=0.2,
        high=1.2,
        units="eV",
        pass_message="Calibrated Schottky barrier height is inside a broad silicon-metal sanity range.",
        warning_message="Calibrated Schottky barrier height is outside a broad silicon-metal sanity range.",
        missing_message="Calibrated Schottky barrier height was not extracted.",
    )
    ideality = range_check(
        "schottky_calibration_ideality_factor",
        metrics.get("best_ideality_factor"),
        low=0.8,
        high=2.5,
        units="",
        pass_message="Calibrated Schottky ideality factor is inside a broad engineering range.",
        warning_message="Calibrated Schottky ideality factor is outside a broad engineering range.",
        missing_message="Calibrated Schottky ideality factor was not extracted.",
    )
    checks.extend([check for check in [barrier, ideality] if check is not None])
    series = float_or_none(metrics.get("best_series_resistance_ohm"))
    if series is not None:
        if series >= 0.0:
            checks.append(
                pass_check(
                    "schottky_calibration_series_resistance_nonnegative",
                    "Calibrated series resistance is non-negative.",
                    {"best_series_resistance_ohm": series},
                )
            )
        else:
            checks.append(
                error_check(
                    "schottky_calibration_series_resistance_negative",
                    "Calibrated series resistance must not be negative.",
                    {"best_series_resistance_ohm": series},
                )
            )
    rmse = float_or_none(metrics.get("best_rmse_log_current_dec"))
    threshold = float_or_none(params.get("max_pass_rmse_log_current_dec")) or 0.15
    if rmse is not None:
        if rmse <= threshold:
            checks.append(
                pass_check(
                    "schottky_calibration_rmse_within_threshold",
                    "Log-current RMSE is within the configured calibration threshold.",
                    {"best_rmse_log_current_dec": rmse},
                    {"threshold": threshold},
                )
            )
        elif rmse <= threshold * 2.0:
            checks.append(
                warn_check(
                    "schottky_calibration_rmse_marginal",
                    "Log-current RMSE is above threshold; inspect residual trend and bias window.",
                    {"best_rmse_log_current_dec": rmse},
                    {"threshold": threshold},
                )
            )
        else:
            checks.append(
                error_check(
                    "schottky_calibration_rmse_far_above_threshold",
                    "Log-current RMSE is far above threshold and should block trust in the calibration.",
                    {"best_rmse_log_current_dec": rmse},
                    {"threshold": threshold},
                )
            )
    return checks


def golden_profile_key(tool_name: str, metrics: dict[str, Any], params: dict[str, Any]) -> str:
    device_type = str(metrics.get("device_type") or params.get("device_type") or "")
    return f"{tool_name}:{device_type}" if device_type else tool_name


def normalized_golden_metrics(raw: Any) -> dict[str, dict[str, float]]:
    if not isinstance(raw, dict):
        return {}
    result: dict[str, dict[str, float]] = {}
    for metric, spec in raw.items():
        if isinstance(spec, dict):
            expected = float_or_none(spec.get("expected"))
            tolerance = float_or_none(spec.get("relative_tolerance"))
        else:
            expected = float_or_none(spec)
            tolerance = None
        if expected is None:
            continue
        result[str(metric)] = {"expected": expected, "relative_tolerance": tolerance if tolerance is not None else 0.2}
    return result


def benchmark_golden_metrics(tool_name: str, metrics: dict[str, Any], params: dict[str, Any]) -> list[BenchmarkCheck]:
    profile = dict(DEFAULT_GOLDEN_PROFILES.get(golden_profile_key(tool_name, metrics, params), {}))
    profile.update(normalized_golden_metrics(params.get("golden_metrics")))
    profile.update(normalized_golden_metrics(metrics.get("golden_metrics")))
    checks: list[BenchmarkCheck] = []
    for metric, spec in profile.items():
        observed = float_or_none(metrics.get(metric))
        expected = float_or_none(spec.get("expected"))
        tolerance = float_or_none(spec.get("relative_tolerance")) or 0.2
        if expected is None:
            continue
        if observed is None:
            checks.append(
                warn_check(
                    f"golden_metric_{metric}_missing",
                    "Golden benchmark metric is missing from the run output.",
                    {"metric": metric},
                    {"expected": expected, "relative_tolerance": tolerance},
                )
            )
            continue
        error_value = relative_error(observed, expected)
        observed_data = {"metric": metric, "observed": observed, "relative_error": error_value}
        expected_data = {"expected": expected, "relative_tolerance": tolerance}
        if error_value <= tolerance:
            checks.append(
                pass_check(
                    f"golden_metric_{metric}_within_tolerance",
                    "Metric matches the golden profile within relative tolerance.",
                    observed_data,
                    expected_data,
                )
            )
        elif error_value <= 3.0 * tolerance:
            checks.append(
                warn_check(
                    f"golden_metric_{metric}_outside_tolerance",
                    "Metric differs from the golden profile; inspect model, units, or extraction settings.",
                    observed_data,
                    expected_data,
                )
            )
        else:
            checks.append(
                error_check(
                    f"golden_metric_{metric}_far_outside_tolerance",
                    "Metric is far from the golden profile and should block trust in this result.",
                    observed_data,
                    expected_data,
                )
            )
    return checks


def benchmark_deck_signoff(state: dict[str, Any], metrics: dict[str, Any], params: dict[str, Any]) -> list[BenchmarkCheck]:
    deck = state_deck_spec(state)
    if not deck:
        return []
    checks: list[BenchmarkCheck] = []
    signoff = deck.get("signoff_requirements") or {}
    physics = deck.get("physics_models") or {}
    required_level = signoff.get("required_level")
    if required_level == "engineering_signoff":
        if signoff.get("require_convergence_evidence"):
            has_convergence = any(key in metrics for key in ["relative_delta", "relative_tolerance"]) or state.get("tool_name") in {
                "tool_convergence",
                "mesh_convergence",
            }
            if has_convergence:
                checks.append(
                    pass_check(
                        "deck_signoff_convergence_evidence_present",
                        "Deck/spec 要求工程签核，当前结果包含收敛证据。",
                        {"required_level": required_level},
                    )
                )
            else:
                checks.append(
                    warn_check(
                        "deck_signoff_convergence_evidence_missing",
                        "Deck/spec 要求工程签核，但当前单次结果缺少 mesh/tool convergence 证据。",
                        {"required_level": required_level, "device_family": deck.get("device_family")},
                    )
                )
        if signoff.get("measured_curve_path") and not params.get("measured_curve_comparison"):
            checks.append(
                warn_check(
                    "deck_measured_curve_comparison_missing",
                    "任务提到了实测/可信曲线，但当前 benchmark 还没有完成曲线对比。",
                    {"measured_curve_path": signoff.get("measured_curve_path")},
                )
            )
    if physics.get("coupling_status") == "needs_benchmark_confirmation":
        requested_models = {
            key: physics.get(key)
            for key in ["interface_trap_density_cm2", "fixed_oxide_charge_cm2", "impact_ionization_model"]
            if physics.get(key) not in {None, 0, 0.0, "none"}
        }
        if requested_models:
            checks.append(
                warn_check(
                    "deck_physics_model_coupling_needs_confirmation",
                    "Deck/spec 中包含高级物理模型，但当前 runner 的方程耦合状态需要 benchmark 明确确认。",
                    {"requested_models": requested_models},
                )
            )
    deck_warnings = deck.get("warnings") or []
    if deck_warnings:
        checks.append(
            warn_check(
                "deck_spec_contains_model_warnings",
                "Deck/spec 自身包含模型或签核风险提示，结论需要保留这些风险。",
                {"warnings": deck_warnings[:4]},
            )
        )
    return checks


def benchmark_convergence(metrics: dict[str, Any]) -> list[BenchmarkCheck]:
    relative_delta = float_or_none(metrics.get("relative_delta"))
    tolerance = float_or_none(metrics.get("relative_tolerance"))
    if relative_delta is None or tolerance is None:
        return []
    if relative_delta <= tolerance:
        return [
            pass_check(
                "convergence_relative_delta_within_tolerance",
                "Last two convergence cases are within the configured relative tolerance.",
                {"relative_delta": relative_delta},
                {"relative_tolerance": tolerance},
            )
        ]
    return [
        warn_check(
            "convergence_relative_delta_above_tolerance",
            "Last two convergence cases differ more than the configured tolerance.",
            {"relative_delta": relative_delta},
            {"relative_tolerance": tolerance},
        )
    ]


def best_aggregate_item(state: dict[str, Any]) -> dict[str, Any] | None:
    if state.get("best_observation"):
        return state["best_observation"]
    if state.get("best_case"):
        return state["best_case"]
    items = state.get("observations") or state.get("cases") or []
    eligible = [item for item in items if item.get("objective_value") is not None]
    if not eligible:
        return None
    reverse = ((state.get("objective") or {}).get("direction") == "maximize")
    return sorted(eligible, key=lambda item: float(item.get("objective_value") or 0.0), reverse=reverse)[0]


def benchmark_aggregate(state: dict[str, Any], source_path: Path) -> list[BenchmarkCheck]:
    checks: list[BenchmarkCheck] = []
    items = state.get("observations") or state.get("cases") or []
    completed = [item for item in items if item.get("status") == "completed"]
    suspicious = [item for item in items if item.get("quality_status") == "suspicious"]
    failed = [item for item in items if item.get("status") == "failed" or item.get("quality_status") == "failed"]
    if completed:
        checks.append(
            pass_check(
                "aggregate_has_completed_cases",
                "Aggregate state contains completed observations or sweep cases.",
                {"completed_cases": len(completed), "total_cases": len(items)},
            )
        )
    else:
        checks.append(
            warn_check(
                "aggregate_has_no_completed_cases",
                "Aggregate state has no completed cases to benchmark.",
                {"total_cases": len(items)},
            )
        )
    if suspicious:
        checks.append(
            warn_check(
                "aggregate_contains_suspicious_cases",
                "Some aggregate cases were already flagged suspicious.",
                {"suspicious_cases": len(suspicious)},
            )
        )
    if failed:
        checks.append(
            warn_check(
                "aggregate_contains_failed_cases",
                "Some aggregate cases failed and should not be used as optimization evidence.",
                {"failed_cases": len(failed)},
            )
        )
    best = best_aggregate_item(state)
    best_path = best.get("final_state_path") if best else None
    if best_path:
        child_path = Path(best_path)
        if not child_path.is_absolute():
            child_path = source_path.parent / child_path
        if child_path.exists():
            child_state = read_json(child_path)
            child_checks = benchmark_state(child_state, child_path)
            checks.extend(
                [
                    check.model_copy(update={"code": f"best_case_{check.code}"})
                    for check in child_checks
                    if check.severity != BenchmarkSeverity.PASS or check.code.endswith("_ok")
                ]
            )
        else:
            checks.append(
                warn_check(
                    "aggregate_best_state_missing",
                    "Best aggregate item references a final state path that does not exist.",
                    {"final_state_path": str(child_path)},
                )
            )
    return checks


def benchmark_state(state: dict[str, Any], source_path: Path) -> list[BenchmarkCheck]:
    tool_name = str(state.get("tool_name") or "")
    metrics = merged_metrics(state)
    params = merged_parameters(state)
    checks = generic_quality_checks(state)
    checks.extend(benchmark_golden_metrics(tool_name, metrics, params))
    checks.extend(benchmark_deck_signoff(state, metrics, params))
    if tool_name == "pn_junction_iv_sweep":
        checks.extend(benchmark_pn(metrics, params))
    elif tool_name == "mos_capacitor_cv_sweep":
        checks.extend(benchmark_mos_capacitor(metrics, params))
    elif tool_name == "diode_breakdown_leakage_sweep":
        checks.extend(benchmark_diode_breakdown(metrics, params))
    elif tool_name == "mosfet_2d_id_sweep":
        checks.extend(benchmark_mosfet(metrics, params))
    elif tool_name == "extended_device_sweep":
        checks.extend(benchmark_extended_device(metrics, params))
    elif tool_name == "schottky_iv_calibration":
        checks.extend(benchmark_schottky_calibration(metrics, params))
    elif tool_name in {"mesh_convergence", "tool_convergence"}:
        checks.extend(benchmark_convergence(metrics))
    elif tool_name in {"adaptive_optimizer", "multidim_optimizer", "parameter_sweep"}:
        checks.extend(benchmark_aggregate(state, source_path))
    return checks


def status_from_checks(checks: list[BenchmarkCheck], *, supported: bool) -> BenchmarkStatus:
    if not supported and not checks:
        return BenchmarkStatus.UNSUPPORTED
    if any(check.severity == BenchmarkSeverity.ERROR for check in checks):
        return BenchmarkStatus.FAILED
    if any(check.severity == BenchmarkSeverity.WARNING for check in checks):
        return BenchmarkStatus.SUSPICIOUS
    return BenchmarkStatus.PASSED


def evidence_matrix(state: dict[str, Any], checks: list[BenchmarkCheck]) -> dict[str, Any]:
    metrics = merged_metrics(state)
    deck = state_deck_spec(state)
    final_summary = state.get("final_summary") or {}
    artifacts = final_summary.get("artifacts") or {}
    quality = state.get("quality_report") or {}
    return {
        "quality_report": "present" if quality else "missing",
        "curve_artifacts": "present" if artifacts.get("plot") or artifacts.get("csv") else "unknown_or_missing",
        "deck_spec": "present" if deck else "missing",
        "physical_benchmark": "present",
        "convergence_evidence": "present"
        if state.get("tool_name") in {"tool_convergence", "mesh_convergence"} or "relative_delta" in metrics
        else "missing",
        "golden_or_measured_comparison": "present"
        if any(check.code.startswith("golden_metric_") for check in checks)
        else "missing",
        "model_coupling_risk": "present"
        if any("coupling" in check.code or "metadata_only" in check.code for check in checks)
        else "not_detected",
    }


def credibility_assessment(checks: list[BenchmarkCheck], state: dict[str, Any]) -> dict[str, Any]:
    codes = {check.code for check in checks}
    warning_codes = [check.code for check in checks if check.severity == BenchmarkSeverity.WARNING]
    error_codes = [check.code for check in checks if check.severity == BenchmarkSeverity.ERROR]
    matrix = evidence_matrix(state, checks)
    quality = state.get("quality_report") or {}
    risks: list[str] = []
    gaps: list[str] = []
    must_fix: list[str] = []

    if quality.get("status") in {"failed", "suspicious"}:
        risks.append("基础 quality_report 未完全通过。")
    if any("unit" in code or "voltage_span" in code or "capacitance_exceeds" in code for code in codes):
        risks.append("存在单位、量纲或解析上界相关风险。")
        must_fix.append("复核单位、面积归一化、偏置单位和解析上界。")
    if any("kink" in code or "monotonic" in code or "shape" in code or "negative_differential" in code for code in codes):
        risks.append("曲线形状存在异常或需要局部细化。")
        must_fix.append("在异常 bias 区间缩小步长并复核曲线单调性/斜率。")
    if matrix.get("convergence_evidence") != "present":
        gaps.append("mesh/model/bias convergence evidence")
    if matrix.get("golden_or_measured_comparison") != "present":
        gaps.append("golden or measured comparison")
    if matrix.get("deck_spec") != "present":
        gaps.append("structured TCAD deck spec")
    if matrix.get("model_coupling_risk") == "present":
        risks.append("物理模型可能只是 metadata 或耦合状态需要确认。")
        must_fix.append("确认 traps、fixed charge、impact ionization 等模型是否真的耦合进方程。")
    if any("golden_metric_" in code for code in warning_codes + error_codes):
        risks.append("与 golden/经验指标的偏差需要解释。")
    if error_codes:
        level = "blocked"
        acceptance = "不可作为工程结论依据，必须先修复错误项。"
    elif warning_codes or risks:
        level = "conditional"
        acceptance = "可作为下一步规划线索，但结论必须带风险说明。"
    elif len(checks) >= 2 and not gaps[:1]:
        level = "ready"
        acceptance = "可作为本轮工程证据；若用于签核仍需补 corner/实测/更完整 deck。"
    else:
        level = "limited"
        acceptance = "证据偏少，只能作为初步 smoke/探索结果。"
    score = {
        "ready": 0.9,
        "conditional": 0.65,
        "limited": 0.45,
        "blocked": 0.1,
    }[level]
    score -= min(len(gaps) * 0.05, 0.2)
    return {
        "level": level,
        "score": round(max(score, 0.0), 3),
        "acceptance_zh": acceptance,
        "risk_factors_zh": risks,
        "evidence_gaps": gaps,
        "must_fix_before_signoff": must_fix,
        "matrix": matrix,
    }


def summarize_checks(checks: list[BenchmarkCheck], state: dict[str, Any] | None = None) -> dict[str, Any]:
    counts = {"pass": 0, "warning": 0, "error": 0}
    for check in checks:
        counts[check.severity.value] += 1
    total = sum(counts.values())
    score = 0.0 if total == 0 else max(0.0, min(1.0, 1.0 - counts["error"] * 0.45 - counts["warning"] * 0.15))
    if counts["error"]:
        signoff_status = "blocked"
        label = "不可签核"
        next_action = "先修复错误级物理/质量检查，再重新运行 benchmark。"
    elif counts["warning"]:
        signoff_status = "conditional"
        label = "有条件可用"
        next_action = "把告警项作为风险写入结论，并补做局部收敛/单位/曲线形状复核。"
    elif counts["pass"] >= 2:
        signoff_status = "ready"
        label = "可作为本轮工程证据"
        next_action = "可进入工程结论；若用于 tapeout/signoff 级判断，仍需补充更完整的 deck 和 corner。"
    elif counts["pass"] == 1:
        signoff_status = "limited"
        label = "证据偏少"
        next_action = "增加收敛点或 golden/解析 benchmark 后再做强结论。"
    else:
        signoff_status = "unsupported"
        label = "暂无可用 benchmark"
        next_action = "补充该工具类型的物理 benchmark 规则或换用已支持的 TCAD 结果。"
    credibility = credibility_assessment(checks, state or {}) if state is not None else {}
    return {
        "generated_at": utc_timestamp(),
        "counts": counts,
        "check_count": total,
        "confidence_score": round(score, 3),
        "credibility": credibility,
        "signoff_status": signoff_status,
        "signoff_label_zh": label,
        "blocking_codes": [check.code for check in checks if check.severity == BenchmarkSeverity.ERROR],
        "warning_codes": [check.code for check in checks if check.severity == BenchmarkSeverity.WARNING],
        "recommended_next_action_zh": next_action,
        "evidence_matrix": evidence_matrix(state or {}, checks) if state is not None else {},
    }


def run_physical_benchmark(source: Path, output_path: Path | None = None) -> PhysicalBenchmarkResult:
    try:
        state_path = resolve_state_path(source).resolve()
        state = read_json(state_path)
        checks = benchmark_state(state, state_path)
        supported = bool(checks) or bool(state.get("quality_report"))
        status = status_from_checks(checks, supported=supported)
        result = PhysicalBenchmarkResult(
            status=status,
            source_state_path=str(state_path),
            source_tool_name=state.get("tool_name"),
            checks=checks,
            summary=summarize_checks(checks, state),
        )
        benchmark_path = (output_path or state_path.with_name("benchmark.json")).resolve()
        result.benchmark_path = str(benchmark_path)
        write_json(benchmark_path, result.model_dump(mode="json"))
        return result
    except Exception as exc:
        return PhysicalBenchmarkResult(
            status=BenchmarkStatus.FAILED,
            source_state_path=str(source),
            failure_reason=str(exc),
        )
