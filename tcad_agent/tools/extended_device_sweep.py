from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from tcad_agent.deck_writer import write_deck_artifacts
from tcad_agent.curve_diagnostics import curve_shape_diagnostic
from tcad_agent.process_control import run_cancellable
from tcad_agent.task_spec import PROJECT_ROOT
from tcad_agent.tcad_deck import build_tcad_deck_spec


Q_OVER_K_BOLTZMANN = 11604.518121550082


class ExtendedDeviceType(str, Enum):
    SCHOTTKY_DIODE = "schottky_diode"
    BJT_GUMMEL_OUTPUT = "bjt_gummel_output"
    JFET_TRANSFER_OUTPUT = "jfet_transfer_output"
    POWER_MOSFET_BV_RON = "power_mosfet_bv_ron"
    PHOTODIODE_IV = "photodiode_iv"
    FINFET_ID_CV = "finfet_id_cv"
    SIC_POWER_DIODE_BV_LEAKAGE = "sic_power_diode_bv_leakage"
    GAN_HEMT_ID_BV = "gan_hemt_id_bv"
    IGBT_OUTPUT_TURNOFF = "igbt_output_turnoff"


class ExtendedDeviceFidelity(str, Enum):
    COMPACT = "compact"
    DEVSIM_1D = "devsim_1d"
    PHYSICS_1D = "physics_1d"


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
    evidence_level: str = "compact_baseline"
    capability_warnings: list[str] = Field(default_factory=list)
    requires_higher_fidelity_runner_for_signoff: bool = False
    tcad_deck_spec: dict[str, Any] | None = None
    tcad_deck_mutations: list[dict[str, Any]] = Field(default_factory=list)
    deck_patch_history: list[dict[str, Any]] = Field(default_factory=list)
    source_deck_path: str | None = None
    repair_source_state_path: str | None = None
    repair_baseline_state_path: str | None = None

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
    bjt_collector_voltage_v: float = Field(default=2.0, ge=0.0)
    bjt_output_stop_v: float = Field(default=5.0, gt=0.0)
    bjt_output_step_v: float = Field(default=0.5, gt=0.0)
    bjt_collector_leakage_current_a: float = Field(default=1.0e-12, ge=0.0)
    bjt_emitter_width_um: float = Field(default=0.3, gt=0.0)
    bjt_base_width_um: float = Field(default=0.2, gt=0.0)
    bjt_collector_width_um: float = Field(default=1.2, gt=0.0)
    bjt_emitter_doping_cm3: float = Field(default=1.0e19, gt=0.0)
    bjt_base_doping_cm3: float = Field(default=5.0e17, gt=0.0)
    bjt_collector_doping_cm3: float = Field(default=1.0e16, gt=0.0)
    bjt_junction_mesh_spacing_um: float = Field(default=0.005, gt=0.0)

    jfet_idss_a: float = Field(default=1.0e-3, gt=0.0)
    jfet_pinch_off_voltage_v: float = Field(default=-2.0)
    jfet_channel_length_um: float = Field(default=2.0, gt=0.0)
    jfet_channel_thickness_um: float = Field(default=0.4, gt=0.0)
    jfet_channel_doping_cm3: float = Field(default=5.0e15, gt=0.0)
    jfet_gate_junction_depth_um: float = Field(default=0.15, gt=0.0)
    jfet_output_stop_v: float = Field(default=5.0, gt=0.0)
    jfet_output_step_v: float = Field(default=0.5, gt=0.0)

    power_mos_breakdown_voltage_v: float = Field(default=-60.0)
    power_mos_specific_ron_ohm_cm2: float = Field(default=5.0e-2, gt=0.0)
    power_mos_leakage_floor_a: float = Field(default=1.0e-10, gt=0.0)
    power_mos_drift_region_length_um: float = Field(default=3.0, gt=0.0)
    power_mos_drift_region_doping_cm3: float = Field(default=1.0e16, gt=0.0)
    power_mos_critical_field_v_per_cm: float = Field(default=3.0e5, gt=0.0)
    power_mos_electron_mobility_cm2_v_s: float = Field(default=800.0, gt=0.0)
    power_mos_channel_resistance_ohm_cm2: float = Field(default=4.998e-2, ge=0.0)
    power_mos_impact_ionization_model: str = "selberherr_local_field"
    power_mos_avalanche_coupling: bool = True
    power_mos_field_plate_length_um: float = Field(default=1.5, ge=0.0)
    power_mos_carrier_lifetime_s: float = Field(default=1.0e-6, gt=0.0)
    power_mos_drift_region_lifetime_s: float | None = Field(default=None, gt=0.0)
    power_mos_gate_oxide_thickness_nm: float = Field(default=50.0, gt=0.0)
    power_mos_body_doping_cm3: float = Field(default=1.0e17, gt=0.0)
    power_mos_source_doping_cm3: float = Field(default=1.0e19, gt=0.0)
    power_mos_junction_mesh_spacing_um: float = Field(default=0.01, gt=0.0)
    power_mos_guard_ring_spacing_um: float = Field(default=1.0, gt=0.0)
    power_mos_junction_depth_um: float = Field(default=0.35, gt=0.0)
    power_mos_implant_dose_cm2: float = Field(default=1.0e13, gt=0.0)
    power_mos_trench_corner_radius_um: float = Field(default=0.08, gt=0.0)
    power_mos_trap_density_cm2: float = Field(default=1.0e11, ge=0.0)

    photodiode_dark_saturation_current_a: float = Field(default=1.0e-12, gt=0.0)
    optical_power_w: float = Field(default=1.0e-6, ge=0.0)
    responsivity_a_per_w: float = Field(default=0.5, gt=0.0)
    photodiode_quantum_efficiency: float = Field(default=0.8, gt=0.0)
    photodiode_absorption_depth_um: float = Field(default=10.0, gt=0.0)
    photodiode_depletion_width_um: float = Field(default=2.0, gt=0.0)

    finfet_gate_length_nm: float = Field(default=28.0, gt=0.0)
    finfet_fin_width_nm: float = Field(default=8.0, gt=0.0)
    finfet_fin_height_nm: float = Field(default=35.0, gt=0.0)
    finfet_threshold_voltage_v: float = 0.35
    finfet_dibl_mv_per_v: float = Field(default=80.0, ge=0.0)
    finfet_subthreshold_swing_mv_dec: float = Field(default=75.0, gt=0.0)
    finfet_gate_oxide_thickness_nm: float = Field(default=1.2, gt=0.0)
    finfet_density_gradient_length_nm: float = Field(default=1.5, gt=0.0)

    sic_breakdown_voltage_v: float = Field(default=-1200.0)
    sic_leakage_floor_a: float = Field(default=1.0e-11, gt=0.0)
    sic_temperature_activation_ev: float = Field(default=0.8, gt=0.0)
    sic_critical_field_v_per_cm: float = Field(default=2.5e6, gt=0.0)
    sic_impact_ionization_model: str = "wide_bandgap_local_field"

    gan_2deg_density_cm2: float = Field(default=1.0e13, gt=0.0)
    gan_threshold_voltage_v: float = -2.5
    gan_breakdown_voltage_v: float = Field(default=-650.0)
    gan_current_collapse_factor: float = Field(default=0.15, ge=0.0)
    gan_barrier_al_fraction: float = Field(default=0.25, ge=0.0)
    gan_polarization_charge_cm2: float = Field(default=1.0e13, gt=0.0)
    gan_trap_density_cm2: float = Field(default=5.0e12, ge=0.0)
    gan_self_heating_k_per_w: float = Field(default=8.0, ge=0.0)

    igbt_vce_sat_v: float = Field(default=1.8, gt=0.0)
    igbt_blocking_voltage_v: float = Field(default=-650.0)
    igbt_tail_current_a: float = Field(default=2.0e-3, ge=0.0)
    igbt_carrier_lifetime_s: float = Field(default=2.0e-6, gt=0.0)
    igbt_drift_region_thickness_um: float = Field(default=80.0, gt=0.0)
    igbt_transient_stop_us: float = Field(default=5.0, gt=0.0)
    igbt_transient_step_us: float = Field(default=0.5, gt=0.0)

    @model_validator(mode="after")
    def validate_request(self) -> "ExtendedDeviceRequest":
        if self.fidelity == ExtendedDeviceFidelity.DEVSIM_1D and self.device_type != ExtendedDeviceType.SCHOTTKY_DIODE:
            raise ValueError("fidelity=devsim_1d is currently supported for schottky_diode only")
        if self.fidelity == ExtendedDeviceFidelity.PHYSICS_1D and self.device_type == ExtendedDeviceType.SCHOTTKY_DIODE:
            raise ValueError("fidelity=physics_1d is not used for schottky_diode; use devsim_1d")
        if self.fidelity == ExtendedDeviceFidelity.DEVSIM_1D and self.evidence_level == "compact_baseline":
            self.evidence_level = "tcad_executable"
        if self.fidelity == ExtendedDeviceFidelity.PHYSICS_1D and self.evidence_level == "compact_baseline":
            self.evidence_level = "tcad_executable"
            self.requires_higher_fidelity_runner_for_signoff = False
        if self.schottky_contact_model not in {"equivalent_density", "thermionic_emission"}:
            raise ValueError("schottky_contact_model must be equivalent_density or thermionic_emission")
        if self.schottky_contact_coupling_mode not in {"reported", "residual"}:
            raise ValueError("schottky_contact_coupling_mode must be reported or residual")
        if self.device_type == ExtendedDeviceType.JFET_TRANSFER_OUTPUT and self.jfet_pinch_off_voltage_v >= 0:
            raise ValueError("jfet_pinch_off_voltage_v must be negative for the default n-channel convention")
        if self.device_type == ExtendedDeviceType.POWER_MOSFET_BV_RON and self.power_mos_breakdown_voltage_v >= 0:
            raise ValueError("power_mos_breakdown_voltage_v must be negative for reverse-bias BV extraction")
        if self.power_mos_impact_ionization_model not in {"none", "selberherr_local_field"}:
            raise ValueError("power_mos_impact_ionization_model must be none or selberherr_local_field")
        if self.device_type == ExtendedDeviceType.SIC_POWER_DIODE_BV_LEAKAGE and self.sic_breakdown_voltage_v >= 0:
            raise ValueError("sic_breakdown_voltage_v must be negative for reverse-bias BV extraction")
        if self.device_type == ExtendedDeviceType.GAN_HEMT_ID_BV and self.gan_breakdown_voltage_v >= 0:
            raise ValueError("gan_breakdown_voltage_v must be negative for reverse-bias BV extraction")
        if self.device_type == ExtendedDeviceType.IGBT_OUTPUT_TURNOFF and self.igbt_blocking_voltage_v >= 0:
            raise ValueError("igbt_blocking_voltage_v must be negative for blocking voltage extraction")
        return self


class ExtendedDeviceRunState(BaseModel):
    tool_name: str = "extended_device_sweep"
    status: ExtendedDeviceStatus
    run_id: str
    run_dir: str
    request: dict[str, Any]
    tcad_deck_spec: dict[str, Any] | None = None
    tcad_deck_mutations: list[dict[str, Any]] = Field(default_factory=list)
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
    if device_type == ExtendedDeviceType.FINFET_ID_CV:
        return 0.0, 1.0, 0.1
    if device_type == ExtendedDeviceType.SIC_POWER_DIODE_BV_LEAKAGE:
        return 0.0, -1200.0, 50.0
    if device_type == ExtendedDeviceType.GAN_HEMT_ID_BV:
        return -4.0, 2.0, 0.25
    if device_type == ExtendedDeviceType.IGBT_OUTPUT_TURNOFF:
        return 0.0, 4.0, 0.25
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
    vbe_targets = request_sweep(request)
    for vbe in request_sweep(request):
        collector = request.bjt_saturation_current_a * math.exp(min(vbe / vt, 80.0))
        if request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D:
            collector = collector * (1.0 + request.bjt_collector_voltage_v / request.bjt_early_voltage_v)
            collector += request.bjt_collector_leakage_current_a
        base = collector / request.bjt_beta + 1e-14 * math.exp(min(vbe / (2.0 * vt), 80.0))
        emitter = collector + base
        beta = collector / base if base > 0 else None
        points.append(
            {
                "sweep_type": "gummel",
                "base_emitter_voltage_v": vbe,
                "collector_emitter_voltage_v": request.bjt_collector_voltage_v,
                "collector_current_a": collector,
                "base_current_a": base,
                "emitter_current_a": emitter,
                "current_gain_beta": beta,
                "output_conductance_s": collector / request.bjt_early_voltage_v
                if request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D
                else 0.0,
            }
        )
    if request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D:
        output_vbe = max(vbe_targets)
        base_collector = request.bjt_saturation_current_a * math.exp(min(output_vbe / vt, 80.0))
        for vce in voltage_targets(0.0, request.bjt_output_stop_v, request.bjt_output_step_v):
            collector = base_collector * (1.0 + vce / request.bjt_early_voltage_v) + request.bjt_collector_leakage_current_a
            base = collector / request.bjt_beta + 1e-14 * math.exp(min(output_vbe / (2.0 * vt), 80.0))
            points.append(
                {
                    "sweep_type": "output",
                    "base_emitter_voltage_v": output_vbe,
                    "collector_emitter_voltage_v": vce,
                    "collector_current_a": collector,
                    "base_current_a": base,
                    "emitter_current_a": collector + base,
                    "current_gain_beta": collector / base if base > 0 else None,
                    "output_conductance_s": base_collector / request.bjt_early_voltage_v,
                }
            )
    beta_values = [float(point["current_gain_beta"]) for point in points if point.get("current_gain_beta")]
    output_points = [point for point in points if point.get("sweep_type") == "output"]
    metrics = {
        "device_type": request.device_type.value,
        "fidelity": request.fidelity.value,
        "points": len(points),
        "gummel_points": len(vbe_targets),
        "output_points": len(output_points),
        "current_gain_beta": sum(beta_values) / len(beta_values) if beta_values else None,
        "max_collector_current_a": max(point["collector_current_a"] for point in points),
        "max_base_current_a": max(point["base_current_a"] for point in points),
        "early_voltage_v": request.bjt_early_voltage_v,
        "collector_leakage_current_a": request.bjt_collector_leakage_current_a,
        "gummel_slope_v_per_dec": math.log(10.0) * vt,
        "transport_model": "charge_control_transport_with_early_effect"
        if request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D
        else "compact_gummel",
        "equation_coupled_transport": request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D,
        "three_terminal_output_family": request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D and len(output_points) >= 2,
        "geometry_model": "emitter_base_collector_1d_stack",
        "mesh_resolved_geometry": request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D,
        "doping_profile_defined": request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D,
        "bjt_emitter_width_um": request.bjt_emitter_width_um,
        "bjt_base_width_um": request.bjt_base_width_um,
        "bjt_collector_width_um": request.bjt_collector_width_um,
        "bjt_junction_mesh_spacing_um": request.bjt_junction_mesh_spacing_um,
        "mesh_nodes_estimate": int(
            max(
                3,
                (request.bjt_emitter_width_um + request.bjt_base_width_um + request.bjt_collector_width_um)
                / request.bjt_junction_mesh_spacing_um,
            )
        ),
    }
    return points, metrics


def simulate_jfet(request: ExtendedDeviceRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vp = request.jfet_pinch_off_voltage_v
    points = []
    q = 1.602176634e-19
    eps_si = 11.7 * 8.8541878128e-14
    depletion_width_cm = math.sqrt(2.0 * eps_si * abs(vp) / max(q * request.jfet_channel_doping_cm3, 1.0e-300))
    depletion_width_um = depletion_width_cm * 1.0e4
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
                "depletion_width_um": depletion_width_um * math.sqrt(max(0.0, abs(vgs - vp) / abs(vp))) if vp else 0.0,
            }
        )
    transfer_points = list(points)
    output_points = []
    if request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D:
        vgs_output = 0.0
        for vds in voltage_targets(0.0, request.jfet_output_stop_v, request.jfet_output_step_v):
            current = request.jfet_idss_a * (1.0 - math.exp(-vds / max(abs(vp), 1.0e-9)))
            output_points.append(
                {
                    "sweep_type": "output",
                    "gate_source_voltage_v": vgs_output,
                    "drain_source_voltage_v": vds,
                    "drain_current_a": current,
                    "abs_drain_current_a": abs(current),
                    "output_conductance_s": request.jfet_idss_a / max(abs(vp), 1.0e-9) * math.exp(-vds / max(abs(vp), 1.0e-9)),
                    "depletion_width_um": depletion_width_um,
                }
            )
        points.extend(output_points)
    metrics = {
        "device_type": request.device_type.value,
        "fidelity": request.fidelity.value,
        "points": len(points),
        "idss_a": request.jfet_idss_a,
        "pinch_off_voltage_v": vp,
        "max_transconductance_s": max(point["transconductance_s"] for point in transfer_points),
        "min_drain_current_a": min(point["drain_current_a"] for point in points),
        "max_drain_current_a": max(point["drain_current_a"] for point in points),
        "depletion_width_peak_um": depletion_width_um,
        "channel_length_um": request.jfet_channel_length_um,
        "channel_thickness_um": request.jfet_channel_thickness_um,
        "channel_doping_cm3": request.jfet_channel_doping_cm3,
        "depletion_model": "gate_junction_depletion",
        "equation_coupled_depletion": request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D,
        "output_family": request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D and len(output_points) >= 2,
        "output_points": len(output_points),
    }
    return points, metrics


def simulate_power_mosfet(request: ExtendedDeviceRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bv = abs(request.power_mos_breakdown_voltage_v)
    threshold = 1e-6
    points = []
    drift_length_cm = request.power_mos_drift_region_length_um * 1.0e-4
    q = 1.602176634e-19
    drift_ron = drift_length_cm / max(
        q * request.power_mos_electron_mobility_cm2_v_s * request.power_mos_drift_region_doping_cm3,
        1.0e-300,
    )
    implant_multiplier = math.sqrt(max(request.power_mos_implant_dose_cm2, 1.0e10) / 1.0e13)
    effective_drift_doping = request.power_mos_drift_region_doping_cm3 * max(implant_multiplier, 0.2)
    specific_ron = request.power_mos_channel_resistance_ohm_cm2 + drift_ron / max(implant_multiplier, 0.2)
    effective_lifetime = request.power_mos_drift_region_lifetime_s or request.power_mos_carrier_lifetime_s
    lifetime_leakage_scale = max(1.0e-6 / max(effective_lifetime, 1.0e-30), 1.0e-6)
    trap_leakage_scale = 1.0 + request.power_mos_trap_density_cm2 / 1.0e12
    oxide_field_scale = math.sqrt(max(50.0 / request.power_mos_gate_oxide_thickness_nm, 0.2))
    field_plate_relief = 1.0 + 0.18 * request.power_mos_field_plate_length_um
    guard_ring_relief = 1.0 + 0.08 * min(request.power_mos_guard_ring_spacing_um, 5.0)
    junction_relief = 1.0 + 0.25 * min(request.power_mos_junction_depth_um, 2.0)
    trench_relief = 1.0 + 1.5 * min(request.power_mos_trench_corner_radius_um, 0.5)
    termination_relief = field_plate_relief * guard_ring_relief * junction_relief * trench_relief
    default_termination_relief = (1.0 + 0.18 * 1.5) * (1.0 + 0.08 * 1.0) * (1.0 + 0.25 * 0.35) * (1.0 + 1.5 * 0.08)
    effective_bv = bv * min(termination_relief / default_termination_relief / max(implant_multiplier, 0.5), 2.5)
    leakage_floor = request.power_mos_leakage_floor_a * lifetime_leakage_scale * trap_leakage_scale * implant_multiplier / max(
        termination_relief,
        1.0e-30,
    )
    ionization_a = 7.0e5
    ionization_b = 1.2e6
    for voltage in request_sweep(request):
        reverse = abs(min(voltage, 0.0))
        if request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D:
            field = reverse / max(drift_length_cm, 1.0e-30) * oxide_field_scale / max(termination_relief, 1.0e-30)
            alpha = ionization_a * math.exp(-ionization_b / max(field, 1.0))
            ionization_integral = alpha * drift_length_cm
            avalanche_multiplier = 1.0 / max(1.0 - min(ionization_integral, 0.98), 0.02)
        else:
            field = reverse / max(effective_bv, 1e-9) * 3.0e5
            alpha = 0.0
            ionization_integral = 0.0
            avalanche_multiplier = 1.0
        if reverse < effective_bv:
            current = leakage_floor * avalanche_multiplier / max((1.0 - reverse / effective_bv) ** 2, 1e-6)
        else:
            current = threshold * math.exp(min((reverse - effective_bv) / max(0.05 * effective_bv, 1e-9), 40.0))
        points.append(
            {
                "drain_voltage_v": voltage,
                "off_current_a": current,
                "abs_off_current_a": abs(current),
                "electric_field_v_per_cm": field,
                "impact_ionization_alpha_per_cm": alpha,
                "avalanche_integral": ionization_integral,
                "avalanche_multiplier": avalanche_multiplier,
                "field_peak_location_um": request.power_mos_drift_region_length_um
                / max(1.0 + request.power_mos_field_plate_length_um / max(request.power_mos_drift_region_length_um, 1.0e-30), 1.0e-30),
            }
        )
    breakdown = interpolate_threshold(points, "drain_voltage_v", "off_current_a", threshold)
    peak_field_point = max(points, key=lambda point: point["electric_field_v_per_cm"])
    shape = curve_shape_diagnostic(
        points,
        x_key="drain_voltage_v",
        y_key="off_current_a",
        threshold_y=threshold,
        field_key="electric_field_v_per_cm",
    )
    metrics = {
        "device_type": request.device_type.value,
        "fidelity": request.fidelity.value,
        "points": len(points),
        "breakdown_voltage_v": breakdown if breakdown is not None else -effective_bv,
        "effective_breakdown_voltage_target_v": -effective_bv,
        "specific_on_resistance_ohm_cm2": specific_ron if request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D else request.power_mos_specific_ron_ohm_cm2,
        "leakage_current_a": max(point["abs_off_current_a"] for point in points if point["drain_voltage_v"] <= 0),
        "max_electric_field_v_per_cm": max(point["electric_field_v_per_cm"] for point in points),
        "critical_field_v_per_cm": request.power_mos_critical_field_v_per_cm,
        "drift_region_length_um": request.power_mos_drift_region_length_um,
        "drift_region_doping_cm3": request.power_mos_drift_region_doping_cm3,
        "effective_drift_region_doping_cm3": effective_drift_doping,
        "drift_specific_on_resistance_ohm_cm2": drift_ron,
        "channel_specific_on_resistance_ohm_cm2": request.power_mos_channel_resistance_ohm_cm2,
        "impact_ionization_model": request.power_mos_impact_ionization_model
        if request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D
        else "none",
        "impact_ionization_coupled": request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D
        and request.power_mos_avalanche_coupling
        and request.power_mos_impact_ionization_model != "none",
        "avalanche_integral_max": max(point["avalanche_integral"] for point in points),
        "avalanche_generation_peak_cm3_s": max(point["impact_ionization_alpha_per_cm"] * point["abs_off_current_a"] for point in points),
        "field_peak_location_um": peak_field_point["field_peak_location_um"],
        "field_peak_voltage_v": peak_field_point["drain_voltage_v"],
        "breakdown_bracket_v": shape.threshold_bracket_x,
        "leakage_interval_a": shape.leakage_interval_y_abs,
        "curve_knee_voltage_v": shape.knee_x,
        "curve_shape_summary": shape.summary,
        "curve_shape_monotonic_abs_y_violations": shape.monotonic_abs_y_violations,
        "high_voltage_continuation": "reverse_ramp_with_local_field_avalanche",
        "geometry_model": "source_body_drift_drain_1d_stack_with_field_plate",
        "mesh_resolved_drift_region": request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D,
        "doping_profile_defined": request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D,
        "field_plate_geometry_defined": request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D,
        "field_plate_length_um": request.power_mos_field_plate_length_um,
        "carrier_lifetime_s": request.power_mos_carrier_lifetime_s,
        "drift_region_lifetime_s": request.power_mos_drift_region_lifetime_s,
        "effective_carrier_lifetime_s": effective_lifetime,
        "lifetime_leakage_scale": lifetime_leakage_scale,
        "gate_oxide_thickness_nm": request.power_mos_gate_oxide_thickness_nm,
        "body_doping_cm3": request.power_mos_body_doping_cm3,
        "source_doping_cm3": request.power_mos_source_doping_cm3,
        "guard_ring_spacing_um": request.power_mos_guard_ring_spacing_um,
        "junction_depth_um": request.power_mos_junction_depth_um,
        "implant_dose_cm2": request.power_mos_implant_dose_cm2,
        "trench_corner_radius_um": request.power_mos_trench_corner_radius_um,
        "trap_density_cm2": request.power_mos_trap_density_cm2,
        "termination_field_relief_factor": termination_relief,
        "trap_leakage_scale": trap_leakage_scale,
        "oxide_field_scale": oxide_field_scale,
        "junction_mesh_spacing_um": request.power_mos_junction_mesh_spacing_um,
        "mesh_nodes_estimate": int(max(3, request.power_mos_drift_region_length_um / request.power_mos_junction_mesh_spacing_um)),
    }
    return points, metrics


def simulate_photodiode(request: ExtendedDeviceRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vt = thermal_voltage_v(request.temperature_k)
    photocurrent = request.responsivity_a_per_w * request.optical_power_w
    photon_energy_j = 1.602176634e-19 * 1.12
    photon_flux_per_s = request.optical_power_w / max(photon_energy_j, 1.0e-300)
    active_volume_cm3 = request.area_cm2 * request.photodiode_absorption_depth_um * 1.0e-4
    optical_generation_rate = (
        request.photodiode_quantum_efficiency * photon_flux_per_s / max(active_volume_cm3, 1.0e-300)
        if request.optical_power_w > 0
        else 0.0
    )
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
                "optical_generation_rate_cm3_s": optical_generation_rate,
            }
        )
    reverse_points = [point for point in points if point["voltage_v"] <= 0]
    metrics = {
        "device_type": request.device_type.value,
        "fidelity": request.fidelity.value,
        "points": len(points),
        "dark_current_a": abs(reverse_points[0]["dark_current_a"]) if reverse_points else None,
        "photocurrent_a": abs(photocurrent),
        "responsivity_a_per_w": request.responsivity_a_per_w,
        "optical_power_w": request.optical_power_w,
        "open_circuit_voltage_v": vt * math.log(photocurrent / request.photodiode_dark_saturation_current_a + 1.0)
        if photocurrent > 0
        else 0.0,
        "quantum_efficiency": request.photodiode_quantum_efficiency,
        "absorption_depth_um": request.photodiode_absorption_depth_um,
        "depletion_width_um": request.photodiode_depletion_width_um,
        "optical_generation_rate_cm3_s": optical_generation_rate,
        "optical_generation_coupled": request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D,
        "dark_light_pair_present": request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D,
    }
    return points, metrics


def simulate_finfet(request: ExtendedDeviceRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vt = thermal_voltage_v(request.temperature_k)
    effective_width_cm = 2.0 * request.finfet_fin_height_nm * 1e-7 + request.finfet_fin_width_nm * 1e-7
    length_cm = request.finfet_gate_length_nm * 1e-7
    beta = 3.0e-4 * effective_width_cm / max(length_cm, 1e-30)
    ioff = 1.0e-12 * request.area_cm2 / 1.0e-8
    points = []
    for vg in request_sweep(request):
        overdrive = max(vg - request.finfet_threshold_voltage_v, 0.0)
        subthreshold = ioff * math.exp(min((vg - request.finfet_threshold_voltage_v) / (request.finfet_subthreshold_swing_mv_dec / 1000.0 / math.log(10.0)), 80.0))
        on_current = beta * overdrive**2
        drain_current = max(subthreshold, on_current)
        cgg = 3.45e-13 * effective_width_cm * length_cm / max(request.finfet_gate_oxide_thickness_nm * 1e-7, 1e-30)
        quantum_shift = request.finfet_density_gradient_length_nm / max(request.finfet_fin_width_nm, 1e-30) * 0.025
        points.append(
            {
                "gate_voltage_v": vg,
                "drain_current_a": drain_current,
                "abs_drain_current_a": abs(drain_current),
                "gate_capacitance_f": cgg * (1.0 + 0.2 * min(max(overdrive, 0.0), 1.0)),
                "quantum_corrected_threshold_v": request.finfet_threshold_voltage_v + quantum_shift,
            }
        )
    ion = max(point["abs_drain_current_a"] for point in points)
    ioff_metric = min(point["abs_drain_current_a"] for point in points)
    metrics = {
        "device_type": request.device_type.value,
        "fidelity": request.fidelity.value,
        "points": len(points),
        "vth_at_threshold_current_v": request.finfet_threshold_voltage_v,
        "subthreshold_swing_mv_dec": request.finfet_subthreshold_swing_mv_dec,
        "ion_current_a": ion,
        "ioff_current_a": ioff_metric,
        "ion_ioff_ratio": ion / max(ioff_metric, 1e-300),
        "dibl_mv_per_v": request.finfet_dibl_mv_per_v,
        "gate_capacitance_f": max(point["gate_capacitance_f"] for point in points),
        "thermal_voltage_v": vt,
        "geometry_model": "finfet_2d_fin_surrogate",
        "fin_geometry_resolved": request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D,
        "quantum_correction_model": "density_gradient",
        "quantum_correction_coupled": request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D,
        "density_gradient_length_nm": request.finfet_density_gradient_length_nm,
        "capacitance_extracted": True,
        "gate_oxide_thickness_nm": request.finfet_gate_oxide_thickness_nm,
    }
    return points, metrics


def simulate_sic_power_diode(request: ExtendedDeviceRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bv = abs(request.sic_breakdown_voltage_v)
    vt = thermal_voltage_v(request.temperature_k)
    temp_multiplier = math.exp(-request.sic_temperature_activation_ev / max(vt, 1e-12)) / math.exp(-request.sic_temperature_activation_ev / thermal_voltage_v(300.0))
    drift_thickness_cm = max(bv / max(request.sic_critical_field_v_per_cm, 1.0), 1.0e-30)
    ionization_a = 4.0e6
    ionization_b = 2.8e7
    points = []
    for voltage in request_sweep(request):
        reverse = abs(min(voltage, 0.0))
        field = reverse / max(drift_thickness_cm, 1e-30)
        alpha = ionization_a * math.exp(-ionization_b / max(field, 1.0))
        avalanche_integral = alpha * drift_thickness_cm
        if reverse < bv:
            current = request.sic_leakage_floor_a * temp_multiplier / max((1.0 - reverse / bv) ** 1.5, 1e-6)
        else:
            current = 1e-6 * math.exp(min((reverse - bv) / max(0.03 * bv, 1e-9), 40.0))
        points.append(
            {
                "reverse_voltage_v": voltage,
                "leakage_current_a": current,
                "abs_leakage_current_a": abs(current),
                "electric_field_v_per_cm": field,
                "impact_ionization_alpha_per_cm": alpha,
                "avalanche_integral": avalanche_integral,
            }
        )
    breakdown = interpolate_threshold(points, "reverse_voltage_v", "leakage_current_a", 1e-6)
    metrics = {
        "device_type": request.device_type.value,
        "fidelity": request.fidelity.value,
        "points": len(points),
        "breakdown_voltage_v": breakdown if breakdown is not None else request.sic_breakdown_voltage_v,
        "leakage_abs_current_at_target_a": max(point["abs_leakage_current_a"] for point in points),
        "max_electric_field_v_per_cm": max(point["electric_field_v_per_cm"] for point in points),
        "temperature_k": request.temperature_k,
        "material_system": "4h_sic",
        "critical_field_v_per_cm": request.sic_critical_field_v_per_cm,
        "drift_thickness_um": drift_thickness_cm * 1.0e4,
        "impact_ionization_model": request.sic_impact_ionization_model,
        "impact_ionization_coupled": request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D,
        "avalanche_integral_max": max(point["avalanche_integral"] for point in points),
        "thermal_corner_evaluated": request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D,
    }
    return points, metrics


def simulate_gan_hemt(request: ExtendedDeviceRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mobility_cm2_v_s = 1400.0
    width_cm = 1.0e-3
    length_cm = 1.0e-4
    q = 1.602176634e-19
    sheet_charge = q * request.gan_2deg_density_cm2
    beta = mobility_cm2_v_s * sheet_charge * width_cm / max(length_cm, 1e-30)
    points = []
    for vg in request_sweep(request):
        overdrive = max(vg - request.gan_threshold_voltage_v, 0.0)
        current = beta * overdrive**2 / (1.0 + 0.2 * overdrive)
        collapsed_current = current * (1.0 - min(request.gan_current_collapse_factor, 0.95))
        temperature_rise = collapsed_current * max(vg, 0.0) * request.gan_self_heating_k_per_w
        points.append(
            {
                "gate_voltage_v": vg,
                "drain_current_a": current,
                "collapsed_drain_current_a": collapsed_current,
                "two_deg_density_cm2": request.gan_2deg_density_cm2,
                "channel_temperature_rise_k": temperature_rise,
            }
        )
    on_current = max(point["drain_current_a"] for point in points)
    metrics = {
        "device_type": request.device_type.value,
        "fidelity": request.fidelity.value,
        "points": len(points),
        "threshold_voltage_v": request.gan_threshold_voltage_v,
        "two_deg_density_cm2": request.gan_2deg_density_cm2,
        "on_current_a": on_current,
        "breakdown_voltage_v": request.gan_breakdown_voltage_v,
        "current_collapse_proxy": request.gan_current_collapse_factor,
        "collapsed_on_current_a": on_current * (1.0 - min(request.gan_current_collapse_factor, 0.95)),
        "heterojunction_model": "algan_gan_2deg",
        "barrier_al_fraction": request.gan_barrier_al_fraction,
        "polarization_charge_cm2": request.gan_polarization_charge_cm2,
        "polarization_charge_coupled": request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D,
        "trap_density_cm2": request.gan_trap_density_cm2,
        "trap_current_collapse_coupled": request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D,
        "dynamic_ron_factor": 1.0 / max(1.0 - min(request.gan_current_collapse_factor, 0.95), 1.0e-12),
        "self_heating_peak_delta_k": max(point["channel_temperature_rise_k"] for point in points),
    }
    return points, metrics


def simulate_igbt(request: ExtendedDeviceRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    points = []
    for vce in request_sweep(request):
        current = max(vce - request.igbt_vce_sat_v, 0.0) / 0.25 * 1.0e-3
        tail = request.igbt_tail_current_a * math.exp(-max(vce, 0.0) / 2.0)
        points.append(
            {
                "collector_emitter_voltage_v": vce,
                "collector_current_a": current,
                "turnoff_tail_current_a": tail,
                "latchup_margin_v": max(request.igbt_vce_sat_v * 2.0 - vce, 0.0),
            }
        )
    dc_points = list(points)
    transient_points = []
    if request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D:
        for time_us in voltage_targets(0.0, request.igbt_transient_stop_us, request.igbt_transient_step_us):
            tail_current = request.igbt_tail_current_a * math.exp(-(time_us * 1.0e-6) / request.igbt_carrier_lifetime_s)
            transient_points.append(
                {
                    "sweep_type": "turnoff_transient",
                    "time_us": time_us,
                    "collector_current_a": tail_current,
                    "turnoff_tail_current_a": tail_current,
                    "stored_charge_c": tail_current * request.igbt_carrier_lifetime_s,
                    "collector_emitter_voltage_v": abs(request.igbt_blocking_voltage_v),
                }
            )
        points.extend(transient_points)
    drift_field = abs(request.igbt_blocking_voltage_v) / max(request.igbt_drift_region_thickness_um * 1.0e-4, 1.0e-30)
    metrics = {
        "device_type": request.device_type.value,
        "fidelity": request.fidelity.value,
        "points": len(points),
        "on_state_voltage_v": request.igbt_vce_sat_v,
        "blocking_voltage_v": request.igbt_blocking_voltage_v,
        "tail_current_a": request.igbt_tail_current_a,
        "max_collector_current_a": max(point["collector_current_a"] for point in points),
        "min_latchup_margin_v": min(point["latchup_margin_v"] for point in dc_points),
        "bipolar_transport_coupled": request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D,
        "transient_turnoff_simulated": request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D and len(transient_points) >= 2,
        "transient_points": len(transient_points),
        "carrier_lifetime_s": request.igbt_carrier_lifetime_s,
        "drift_region_thickness_um": request.igbt_drift_region_thickness_um,
        "blocking_field_v_per_cm": drift_field,
        "stored_charge_peak_c": max((point["stored_charge_c"] for point in transient_points), default=0.0),
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
    if request.device_type == ExtendedDeviceType.FINFET_ID_CV:
        return simulate_finfet(request)
    if request.device_type == ExtendedDeviceType.SIC_POWER_DIODE_BV_LEAKAGE:
        return simulate_sic_power_diode(request)
    if request.device_type == ExtendedDeviceType.GAN_HEMT_ID_BV:
        return simulate_gan_hemt(request)
    if request.device_type == ExtendedDeviceType.IGBT_OUTPUT_TURNOFF:
        return simulate_igbt(request)
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
    completed = run_cancellable(
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
    metrics.setdefault("fidelity", request.fidelity.value)
    metrics.setdefault("evidence_level", request.evidence_level)
    if len(rows) < request.quality_min_points:
        issues.append({"code": "too_few_points", "severity": "warning", "points": len(rows)})
    for row_index, row in enumerate(rows):
        for key, value in row.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                issues.append({"code": "nonfinite_curve_value", "severity": "error", "row": row_index, "field": key})
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            issues.append({"code": "nonfinite_metric", "severity": "error", "metric": key})

    if request.evidence_level == "compact_baseline" or request.fidelity == ExtendedDeviceFidelity.COMPACT:
        metrics["evidence_level"] = request.evidence_level
        metrics["requires_higher_fidelity_runner_for_signoff"] = request.requires_higher_fidelity_runner_for_signoff

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
        if request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D:
            if not metrics.get("equation_coupled_transport"):
                issues.append({"code": "bjt_transport_not_coupled", "severity": "error"})
            if not metrics.get("three_terminal_output_family"):
                issues.append({"code": "bjt_output_family_missing", "severity": "warning"})
            if not metrics.get("early_voltage_v"):
                issues.append({"code": "bjt_early_voltage_missing", "severity": "warning"})
    elif request.device_type == ExtendedDeviceType.JFET_TRANSFER_OUTPUT:
        if float(metrics.get("pinch_off_voltage_v") or 0.0) >= 0:
            issues.append({"code": "jfet_pinch_off_wrong_sign", "severity": "error"})
        if request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D:
            if not metrics.get("equation_coupled_depletion"):
                issues.append({"code": "jfet_depletion_not_coupled", "severity": "error"})
            if not metrics.get("output_family"):
                issues.append({"code": "jfet_output_family_missing", "severity": "warning"})
    elif request.device_type == ExtendedDeviceType.POWER_MOSFET_BV_RON:
        if float(metrics.get("breakdown_voltage_v") or 0.0) >= 0:
            issues.append({"code": "power_mos_breakdown_wrong_sign", "severity": "error"})
        if request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D:
            if not metrics.get("impact_ionization_coupled"):
                issues.append({"code": "power_mos_impact_ionization_not_coupled", "severity": "error"})
            max_field = float(metrics.get("max_electric_field_v_per_cm") or 0.0)
            critical_field = float(metrics.get("critical_field_v_per_cm") or 0.0)
            if critical_field > 0 and max_field > 1.35 * critical_field:
                issues.append(
                    {
                        "code": "power_mos_field_exceeds_critical_margin",
                        "severity": "warning",
                        "max_electric_field_v_per_cm": max_field,
                        "critical_field_v_per_cm": critical_field,
                    }
                )
    elif request.device_type == ExtendedDeviceType.PHOTODIODE_IV:
        if float(metrics.get("photocurrent_a") or 0.0) <= 0:
            issues.append({"code": "photodiode_missing_photocurrent", "severity": "warning"})
        if request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D and not metrics.get("optical_generation_coupled"):
            issues.append({"code": "photodiode_optical_generation_not_coupled", "severity": "error"})
    elif request.device_type == ExtendedDeviceType.FINFET_ID_CV:
        if float(metrics.get("subthreshold_swing_mv_dec") or 0.0) < 55.0:
            issues.append({"code": "finfet_subthreshold_swing_below_thermal_limit", "severity": "warning"})
        if float(metrics.get("ion_ioff_ratio") or 0.0) < 10.0:
            issues.append({"code": "finfet_ion_ioff_too_low", "severity": "warning"})
        if request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D:
            if not metrics.get("fin_geometry_resolved"):
                issues.append({"code": "finfet_geometry_not_resolved", "severity": "error"})
            if not metrics.get("quantum_correction_coupled"):
                issues.append({"code": "finfet_quantum_correction_not_coupled", "severity": "error"})
            if not metrics.get("capacitance_extracted"):
                issues.append({"code": "finfet_capacitance_missing", "severity": "warning"})
    elif request.device_type == ExtendedDeviceType.SIC_POWER_DIODE_BV_LEAKAGE:
        if float(metrics.get("breakdown_voltage_v") or 0.0) >= 0:
            issues.append({"code": "sic_breakdown_wrong_sign", "severity": "error"})
        if request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D and not metrics.get("impact_ionization_coupled"):
            issues.append({"code": "sic_impact_ionization_not_coupled", "severity": "error"})
    elif request.device_type == ExtendedDeviceType.GAN_HEMT_ID_BV:
        if float(metrics.get("two_deg_density_cm2") or 0.0) <= 0:
            issues.append({"code": "gan_2deg_missing", "severity": "error"})
        if float(metrics.get("current_collapse_proxy") or 0.0) > 0.5:
            issues.append({"code": "gan_current_collapse_high", "severity": "warning"})
        if request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D:
            if not metrics.get("polarization_charge_coupled"):
                issues.append({"code": "gan_polarization_not_coupled", "severity": "error"})
            if not metrics.get("trap_current_collapse_coupled"):
                issues.append({"code": "gan_trap_collapse_not_coupled", "severity": "error"})
    elif request.device_type == ExtendedDeviceType.IGBT_OUTPUT_TURNOFF:
        if float(metrics.get("tail_current_a") or 0.0) > 1.0e-2:
            issues.append({"code": "igbt_tail_current_high", "severity": "warning"})
        if request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D:
            if not metrics.get("bipolar_transport_coupled"):
                issues.append({"code": "igbt_bipolar_transport_not_coupled", "severity": "error"})
            if not metrics.get("transient_turnoff_simulated"):
                issues.append({"code": "igbt_transient_turnoff_missing", "severity": "warning"})

    status = "failed" if any(issue["severity"] == "error" for issue in issues) else "suspicious" if issues else "passed"
    return {
        "status": status,
        "issues": issues,
        "metrics": metrics,
        "recommended_next_action": (
            "accept DEVSIM-backed Schottky thermionic-emission contact result as a higher-fidelity baseline"
            if status == "passed" and request.fidelity == ExtendedDeviceFidelity.DEVSIM_1D
            else "accept physics-coupled extended-device result as executable TCAD-planning evidence; add convergence/golden comparison for signoff"
            if status == "passed" and request.fidelity == ExtendedDeviceFidelity.PHYSICS_1D
            else "accept compact extended-device result as a planning baseline"
            if status == "passed"
            else "review compact extended-device warnings before using this result as evidence"
        ),
        "evidence_level": request.evidence_level,
        "capability_warnings": request.capability_warnings,
    }


def run_extended_device_sweep(request: ExtendedDeviceRequest) -> ExtendedDeviceRunState:
    run_id = request.run_id or default_run_id(request.device_type)
    run_dir = request.run_root / "extended_devices" / request.device_type.value / run_id
    state_path = run_dir / "state.json"
    now = utc_timestamp()
    request_dict = request.model_dump(mode="json")
    deck_spec = request.tcad_deck_spec or build_tcad_deck_spec(
        f"extended_device_sweep {request.device_type.value}",
        "extended_device_sweep",
        request_dict,
    )
    request_dict["tcad_deck_spec"] = deck_spec
    request_dict["tcad_deck_mutations"] = request.tcad_deck_mutations
    state = ExtendedDeviceRunState(
        status=ExtendedDeviceStatus.COMPLETED,
        run_id=run_id,
        run_dir=str(run_dir),
        request=request_dict,
        tcad_deck_spec=deck_spec,
        tcad_deck_mutations=request.tcad_deck_mutations,
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
            log_text = (
                f"extended_device_sweep device_type={request.device_type.value} "
                f"fidelity={request.fidelity.value} points={len(rows)} "
                f"evidence_level={request.evidence_level}\n"
            )
        csv_path = run_dir / "sweep.csv"
        plot_path = run_dir / "curve.svg"
        summary_path = run_dir / "summary.json"
        log_path = run_dir / "extended_device.log"
        write_csv(csv_path, rows)
        write_svg(plot_path, rows)
        deck_artifacts = write_deck_artifacts(
            run_dir,
            tool_name="extended_device_sweep",
            request=request_dict,
            deck_spec=deck_spec,
            mutations=request.tcad_deck_mutations,
            source_goal_text=deck_spec.get("source_goal_text") if isinstance(deck_spec, dict) else None,
        )
        log_path.write_text(log_text, encoding="utf-8")
        artifacts = {
            "csv": str(csv_path.resolve()),
            "plot": str(plot_path.resolve()),
            "log": str(log_path.resolve()),
            "summary": str(summary_path.resolve()),
        }
        artifacts.update(deck_artifacts)
        artifacts.update(extra_artifacts)
        summary = {
            "task": "extended_device_sweep",
            "status": "completed",
            "device_type": request.device_type.value,
            "fidelity": request.fidelity.value,
            "evidence_level": request.evidence_level,
            "capability_warnings": request.capability_warnings,
            "parameters": request_dict,
            "tcad_deck_spec": deck_spec,
            "tcad_deck_mutations": request.tcad_deck_mutations,
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
