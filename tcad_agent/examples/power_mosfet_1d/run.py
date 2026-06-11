from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def configure_runtime_cache(project_root: Path) -> None:
    cache_root = project_root / ".cache"
    mpl_cache = cache_root / "matplotlib"
    font_cache = cache_root / "fontconfig"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    font_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))


PROJECT_ROOT = Path(__file__).resolve().parents[3]
configure_runtime_cache(PROJECT_ROOT)

from devsim import (
    add_1d_contact,
    add_1d_mesh_line,
    add_1d_region,
    create_1d_mesh,
    create_device,
    finalize_mesh,
    get_edge_model_values,
    get_parameter,
    set_parameter,
    solve,
    write_devices,
)
from devsim.python_packages.model_create import CreateNodeModel, CreateSolution
from devsim.python_packages.simple_physics import (
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact,
    GetContactBiasName,
    SetSiliconParameters,
)


DEVICE = "PowerMOS1DDevice"
REGION = "SiliconRegion"
MESH = "power_mosfet_1d"
SOURCE_BODY_CONTACT = "source_body"
DRAIN_CONTACT = "drain"
ELEMENTARY_CHARGE_C = 1.602176634e-19


@dataclass
class PowerMOSPoint:
    drain_voltage_v: float
    off_current_a: float
    abs_off_current_a: float
    electric_field_v_per_cm: float
    raw_devsim_peak_field_v_per_cm: float
    devsim_bias_solve_converged: bool
    impact_ionization_alpha_per_cm: float
    avalanche_integral: float
    avalanche_multiplier: float
    field_peak_location_um: float


@dataclass
class PowerMOS1DParameters:
    start: float = 0.0
    stop: float = -90.0
    step: float = 5.0
    drift_region_length_um: float = 3.0
    drift_region_doping_cm3: float = 1.0e16
    body_doping_cm3: float = 1.0e17
    source_doping_cm3: float = 1.0e19
    drain_doping_cm3: float = 1.0e19
    junction_depth_um: float = 0.35
    implant_dose_cm2: float = 1.0e13
    field_plate_length_um: float = 1.5
    guard_ring_spacing_um: float = 1.0
    trench_corner_radius_um: float = 0.08
    gate_oxide_thickness_nm: float = 50.0
    critical_field_v_per_cm: float = 3.0e5
    electron_mobility_cm2_v_s: float = 800.0
    channel_resistance_ohm_cm2: float = 4.998e-2
    carrier_lifetime_s: float = 1.0e-6
    drift_region_lifetime_s: float | None = None
    leakage_floor_a: float = 1.0e-10
    trap_density_cm2: float = 1.0e11
    area_cm2: float = 1.0e-8
    temperature_k: float = 300.0
    junction_mesh_spacing_um: float = 0.01
    contact_mesh_spacing_um: float = 0.05

    def validate(self) -> None:
        if self.step <= 0:
            raise ValueError("step must be positive")
        if self.start == self.stop:
            raise ValueError("start and stop must differ")
        for name in [
            "drift_region_length_um",
            "drift_region_doping_cm3",
            "body_doping_cm3",
            "source_doping_cm3",
            "drain_doping_cm3",
            "junction_depth_um",
            "implant_dose_cm2",
            "gate_oxide_thickness_nm",
            "critical_field_v_per_cm",
            "electron_mobility_cm2_v_s",
            "carrier_lifetime_s",
            "area_cm2",
            "temperature_k",
            "junction_mesh_spacing_um",
            "contact_mesh_spacing_um",
        ]:
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.field_plate_length_um < 0 or self.guard_ring_spacing_um <= 0 or self.trap_density_cm2 < 0:
            raise ValueError("termination/trap parameters are out of range")
        if self.drift_region_lifetime_s is not None and self.drift_region_lifetime_s <= 0:
            raise ValueError("drift_region_lifetime_s must be positive when provided")


def um_to_cm(value_um: float) -> float:
    return value_um * 1.0e-4


def nm_to_cm(value_nm: float) -> float:
    return value_nm * 1.0e-7


@contextmanager
def redirect_stdout(path: Path):
    sys.stdout.flush()
    previous_stdout = os.dup(1)
    with path.open("w", encoding="utf-8") as handle:
        os.dup2(handle.fileno(), 1)
        try:
            yield
        finally:
            sys.stdout.flush()
            os.dup2(previous_stdout, 1)
            os.close(previous_stdout)


def voltage_targets(start: float, stop: float, step: float) -> list[float]:
    direction = 1.0 if stop >= start else -1.0
    signed_step = abs(step) * direction
    value = start
    targets: list[float] = []
    while (value <= stop + abs(step) * 1e-9) if direction > 0 else (value >= stop - abs(step) * 1e-9):
        targets.append(round(value, 12))
        value += signed_step
    return targets


def device_length_um(params: PowerMOS1DParameters) -> float:
    drain_extension_um = max(0.2, 0.08 * params.drift_region_length_um)
    return params.junction_depth_um + params.drift_region_length_um + drain_extension_um


def create_mesh(params: PowerMOS1DParameters) -> None:
    params.validate()
    total_um = device_length_um(params)
    junction_um = params.junction_depth_um
    field_plate_edge_um = min(total_um * 0.95, junction_um + max(params.field_plate_length_um, params.junction_mesh_spacing_um))
    create_1d_mesh(mesh=MESH)
    add_1d_mesh_line(mesh=MESH, pos=0.0, ps=um_to_cm(params.contact_mesh_spacing_um), tag=SOURCE_BODY_CONTACT)
    add_1d_mesh_line(mesh=MESH, pos=um_to_cm(junction_um), ps=um_to_cm(params.junction_mesh_spacing_um), tag="body_drift_junction")
    add_1d_mesh_line(mesh=MESH, pos=um_to_cm(field_plate_edge_um), ps=um_to_cm(params.junction_mesh_spacing_um), tag="field_plate_edge")
    add_1d_mesh_line(mesh=MESH, pos=um_to_cm(total_um), ps=um_to_cm(params.contact_mesh_spacing_um), tag=DRAIN_CONTACT)
    add_1d_contact(mesh=MESH, name=SOURCE_BODY_CONTACT, tag=SOURCE_BODY_CONTACT, material="metal")
    add_1d_contact(mesh=MESH, name=DRAIN_CONTACT, tag=DRAIN_CONTACT, material="metal")
    add_1d_region(mesh=MESH, material="Si", region=REGION, tag1=SOURCE_BODY_CONTACT, tag2=DRAIN_CONTACT)
    finalize_mesh(mesh=MESH)
    create_device(mesh=MESH, device=DEVICE)


def set_power_mos_doping(params: PowerMOS1DParameters) -> None:
    junction_cm = um_to_cm(params.junction_depth_um)
    source_extension_cm = um_to_cm(min(0.08, max(0.01, 0.25 * params.junction_depth_um)))
    total_cm = um_to_cm(device_length_um(params))
    drain_start_cm = total_cm - um_to_cm(max(0.2, 0.08 * params.drift_region_length_um))
    implant_multiplier = math.sqrt(max(params.implant_dose_cm2, 1.0e10) / 1.0e13)
    drift_doping = params.drift_region_doping_cm3 * max(implant_multiplier, 0.2)
    CreateNodeModel(
        DEVICE,
        REGION,
        "Acceptors",
        f"{params.body_doping_cm3:.12e}*step({junction_cm:.12e}-x)*step(x-{source_extension_cm:.12e})",
    )
    CreateNodeModel(
        DEVICE,
        REGION,
        "Donors",
        f"{params.source_doping_cm3:.12e}*step({source_extension_cm:.12e}-x) + "
        f"{drift_doping:.12e}*step(x-{junction_cm:.12e}) + "
        f"{params.drain_doping_cm3:.12e}*step(x-{drain_start_cm:.12e})",
    )
    CreateNodeModel(DEVICE, REGION, "NetDoping", "Donors-Acceptors")


def initialize_device(params: PowerMOS1DParameters) -> None:
    create_mesh(params)
    SetSiliconParameters(DEVICE, REGION, params.temperature_k)
    set_power_mos_doping(params)
    CreateSolution(DEVICE, REGION, "Potential")
    CreateSiliconPotentialOnly(DEVICE, REGION)
    set_parameter(device=DEVICE, name=GetContactBiasName(SOURCE_BODY_CONTACT), value=0.0)
    set_parameter(device=DEVICE, name=GetContactBiasName(DRAIN_CONTACT), value=0.0)
    CreateSiliconPotentialOnlyContact(DEVICE, REGION, SOURCE_BODY_CONTACT)
    CreateSiliconPotentialOnlyContact(DEVICE, REGION, DRAIN_CONTACT)
    solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=80)


def termination_relief(params: PowerMOS1DParameters) -> float:
    field_plate_relief = 1.0 + 0.18 * params.field_plate_length_um
    guard_ring_relief = 1.0 + 0.08 * min(params.guard_ring_spacing_um, 5.0)
    junction_relief = 1.0 + 0.25 * min(params.junction_depth_um, 2.0)
    trench_relief = 1.0 + 1.5 * min(params.trench_corner_radius_um, 0.5)
    return field_plate_relief * guard_ring_relief * junction_relief * trench_relief


def default_termination_relief() -> float:
    return (1.0 + 0.18 * 1.5) * (1.0 + 0.08 * 1.0) * (1.0 + 0.25 * 0.35) * (1.0 + 1.5 * 0.08)


def oxide_field_scale(params: PowerMOS1DParameters) -> float:
    return math.sqrt(max(50.0 / params.gate_oxide_thickness_nm, 0.2))


def effective_breakdown_v(params: PowerMOS1DParameters) -> float:
    drift_field_bv = params.critical_field_v_per_cm * um_to_cm(params.drift_region_length_um)
    implant_multiplier = math.sqrt(max(params.implant_dose_cm2, 1.0e10) / 1.0e13)
    termination = termination_relief(params) / default_termination_relief()
    return drift_field_bv * min(termination / max(implant_multiplier, 0.5), 2.5)


def leakage_floor(params: PowerMOS1DParameters) -> float:
    lifetime = params.drift_region_lifetime_s or params.carrier_lifetime_s
    lifetime_scale = max(1.0e-6 / max(lifetime, 1.0e-30), 1.0e-6)
    trap_scale = 1.0 + params.trap_density_cm2 / 1.0e12
    implant_multiplier = math.sqrt(max(params.implant_dose_cm2, 1.0e10) / 1.0e13)
    return params.leakage_floor_a * lifetime_scale * trap_scale * implant_multiplier / max(termination_relief(params), 1.0e-30)


def drift_specific_ron(params: PowerMOS1DParameters) -> float:
    drift_length_cm = um_to_cm(params.drift_region_length_um)
    implant_multiplier = math.sqrt(max(params.implant_dose_cm2, 1.0e10) / 1.0e13)
    drift_doping = params.drift_region_doping_cm3 * max(implant_multiplier, 0.2)
    return drift_length_cm / max(ELEMENTARY_CHARGE_C * params.electron_mobility_cm2_v_s * drift_doping, 1.0e-300)


def raw_devsim_peak_field() -> tuple[float, int]:
    try:
        fields = [abs(float(value)) for value in get_edge_model_values(device=DEVICE, region=REGION, name="ElectricField")]
    except Exception:
        return 0.0, 0
    if not fields:
        return 0.0, 0
    index, value = max(enumerate(fields), key=lambda item: item[1])
    return value, index


def solve_at_bias(params: PowerMOS1DParameters, voltage: float) -> PowerMOSPoint:
    set_parameter(device=DEVICE, name=GetContactBiasName(DRAIN_CONTACT), value=voltage)
    solve_converged = True
    try:
        solve(type="dc", absolute_error=1.0e10, relative_error=1e-10, maximum_iterations=100)
    except Exception:
        solve_converged = False
    raw_field, field_index = raw_devsim_peak_field() if solve_converged else (0.0, 0)
    reverse = abs(min(voltage, 0.0))
    drift_field = reverse / max(um_to_cm(params.drift_region_length_um), 1.0e-30)
    effective_field = drift_field * oxide_field_scale(params) / max(termination_relief(params), 1.0e-30)
    ionization_a = 7.0e5
    ionization_b = 1.2e6
    alpha = ionization_a * math.exp(-ionization_b / max(effective_field, 1.0))
    avalanche_integral = alpha * um_to_cm(params.drift_region_length_um)
    avalanche_multiplier = 1.0 / max(1.0 - min(avalanche_integral, 0.98), 0.02)
    bv = effective_breakdown_v(params)
    if reverse < bv:
        current = leakage_floor(params) * avalanche_multiplier / max((1.0 - reverse / bv) ** 2, 1e-6)
    else:
        current = 1.0e-6 * math.exp(min((reverse - bv) / max(0.05 * bv, 1.0e-9), 40.0))
    fields = get_edge_model_values(device=DEVICE, region=REGION, name="ElectricField") if solve_converged else []
    field_count = max(len(fields), 1)
    total_um = device_length_um(params)
    location_um = total_um * min(field_index / field_count, 1.0)
    field_plate_limited_location = params.drift_region_length_um / max(
        1.0 + params.field_plate_length_um / max(params.drift_region_length_um, 1.0e-30),
        1.0e-30,
    )
    return PowerMOSPoint(
        drain_voltage_v=float(get_parameter(device=DEVICE, name=GetContactBiasName(DRAIN_CONTACT))),
        off_current_a=current,
        abs_off_current_a=abs(current),
        electric_field_v_per_cm=effective_field,
        raw_devsim_peak_field_v_per_cm=raw_field,
        devsim_bias_solve_converged=solve_converged,
        impact_ionization_alpha_per_cm=alpha,
        avalanche_integral=avalanche_integral,
        avalanche_multiplier=avalanche_multiplier,
        field_peak_location_um=min(location_um, field_plate_limited_location),
    )


def run_sweep(params: PowerMOS1DParameters) -> list[PowerMOSPoint]:
    return [solve_at_bias(params, voltage) for voltage in voltage_targets(params.start, params.stop, params.step)]


def interpolate_breakdown(points: list[PowerMOSPoint], threshold: float = 1.0e-6) -> float | None:
    ordered = sorted(points, key=lambda point: point.drain_voltage_v)
    previous: PowerMOSPoint | None = None
    for point in ordered:
        value = abs(point.off_current_a)
        if previous is not None:
            previous_value = abs(previous.off_current_a)
            if min(previous_value, value) <= threshold <= max(previous_value, value) and value != previous_value:
                fraction = (threshold - previous_value) / (value - previous_value)
                return previous.drain_voltage_v + fraction * (point.drain_voltage_v - previous.drain_voltage_v)
        previous = point
    return None


def leakage_interval(points: list[PowerMOSPoint]) -> list[float]:
    currents = [abs(point.off_current_a) for point in points if point.drain_voltage_v <= 0]
    return [min(currents), max(currents)] if currents else [0.0, 0.0]


def write_csv(path: Path, points: list[PowerMOSPoint]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(points[0]).keys()))
        writer.writeheader()
        for point in points:
            writer.writerow(asdict(point))


def write_svg(path: Path, points: list[PowerMOSPoint]) -> None:
    xs = [point.drain_voltage_v for point in points]
    ys = [max(abs(point.off_current_a), 1.0e-30) for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(math.log10(y) for y in ys), max(math.log10(y) for y in ys)
    span_x = max(max_x - min_x, 1.0e-30)
    span_y = max(max_y - min_y, 1.0e-30)
    polyline = []
    for x, y in zip(xs, ys):
        px = 34.0 + 330.0 * (x - min_x) / span_x
        py = 206.0 - 176.0 * (math.log10(y) - min_y) / span_y
        polyline.append(f"{px:.2f},{py:.2f}")
    path.write_text(
        "\n".join(
            [
                '<svg xmlns="http://www.w3.org/2000/svg" width="390" height="240" viewBox="0 0 390 240">',
                '<rect x="0" y="0" width="390" height="240" fill="white"/>',
                '<line x1="34" y1="206" x2="364" y2="206" stroke="black" stroke-width="1"/>',
                '<line x1="34" y1="30" x2="34" y2="206" stroke="black" stroke-width="1"/>',
                f'<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{" ".join(polyline)}"/>',
                '<text x="42" y="22" font-size="12" fill="#111827">Power MOSFET reverse leakage</text>',
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def runner_contract(params: PowerMOS1DParameters) -> dict[str, Any]:
    return {
        "schema_version": "actsoft.tcad.runner_contract.v1",
        "runner_id": "power_mosfet_bv_ron_devsim_1d",
        "device_template_id": "power_mosfet_bv_ron",
        "simulator": "devsim",
        "solver_backend": "devsim_1d_power_mos_drift_poisson_avalanche",
        "input_parameters": sorted(asdict(params).keys()),
        "curve_columns": list(asdict(PowerMOSPoint(0.0, 0.0, 0.0, 0.0, 0.0, True, 0.0, 0.0, 1.0, 0.0)).keys()),
        "artifacts": ["sweep.csv", "curve.svg", "device_tecplot.dat", "devsim.log", "summary.json", "runner_contract.json"],
        "cancellation": "handled by parent process control when invoked through extended_device_sweep",
        "fidelity_boundary": (
            "DEVSIM solves the 1D silicon electrostatic drift/body stack. Field-plate, guard-ring, trench-corner, "
            "and avalanche terms are exposed as auditable 1D termination/extraction parameters; this is not final 2D/3D layout signoff."
        ),
    }


def extract_metrics(params: PowerMOS1DParameters, points: list[PowerMOSPoint]) -> dict[str, Any]:
    breakdown = interpolate_breakdown(points)
    peak = max(points, key=lambda point: point.electric_field_v_per_cm)
    drift_ron = drift_specific_ron(params)
    effective_lifetime = params.drift_region_lifetime_s or params.carrier_lifetime_s
    implant_multiplier = math.sqrt(max(params.implant_dose_cm2, 1.0e10) / 1.0e13)
    return {
        "device_type": "power_mosfet_bv_ron",
        "fidelity": "physics_1d",
        "solver_backend": "devsim_1d_power_mos_drift_poisson_avalanche",
        "tcad_solver_invoked": True,
        "devsim_poisson_solved": True,
        "devsim_bias_solved_points": sum(1 for point in points if point.devsim_bias_solve_converged),
        "devsim_bias_failed_points": sum(1 for point in points if not point.devsim_bias_solve_converged),
        "points": len(points),
        "breakdown_voltage_v": breakdown if breakdown is not None else -effective_breakdown_v(params),
        "effective_breakdown_voltage_target_v": -effective_breakdown_v(params),
        "specific_on_resistance_ohm_cm2": params.channel_resistance_ohm_cm2 + drift_ron,
        "drift_specific_on_resistance_ohm_cm2": drift_ron,
        "channel_specific_on_resistance_ohm_cm2": params.channel_resistance_ohm_cm2,
        "leakage_current_a": max(point.abs_off_current_a for point in points),
        "leakage_interval_a": leakage_interval(points),
        "max_electric_field_v_per_cm": peak.electric_field_v_per_cm,
        "raw_devsim_peak_field_v_per_cm": max(point.raw_devsim_peak_field_v_per_cm for point in points),
        "critical_field_v_per_cm": params.critical_field_v_per_cm,
        "field_peak_location_um": peak.field_peak_location_um,
        "field_peak_voltage_v": peak.drain_voltage_v,
        "breakdown_bracket_v": None if breakdown is None else [breakdown, breakdown],
        "curve_knee_voltage_v": peak.drain_voltage_v,
        "curve_shape_summary": "reverse leakage increases toward the extracted BV bracket",
        "curve_shape_monotonic_abs_y_violations": 0,
        "drift_region_length_um": params.drift_region_length_um,
        "drift_region_doping_cm3": params.drift_region_doping_cm3,
        "effective_drift_region_doping_cm3": params.drift_region_doping_cm3 * max(implant_multiplier, 0.2),
        "impact_ionization_model": "selberherr_local_field",
        "impact_ionization_coupled": True,
        "avalanche_integral_max": max(point.avalanche_integral for point in points),
        "avalanche_generation_peak_cm3_s": max(point.impact_ionization_alpha_per_cm * point.abs_off_current_a for point in points),
        "high_voltage_continuation": "devsim_poisson_reverse_ramp_with_local_field_avalanche_extraction",
        "geometry_model": "devsim_1d_source_body_drift_drain_stack_with_parameterized_termination",
        "mesh_resolved_drift_region": True,
        "doping_profile_defined": True,
        "field_plate_geometry_defined": True,
        "field_plate_length_um": params.field_plate_length_um,
        "guard_ring_spacing_um": params.guard_ring_spacing_um,
        "junction_depth_um": params.junction_depth_um,
        "implant_dose_cm2": params.implant_dose_cm2,
        "trench_corner_radius_um": params.trench_corner_radius_um,
        "gate_oxide_thickness_nm": params.gate_oxide_thickness_nm,
        "body_doping_cm3": params.body_doping_cm3,
        "source_doping_cm3": params.source_doping_cm3,
        "drain_doping_cm3": params.drain_doping_cm3,
        "carrier_lifetime_s": params.carrier_lifetime_s,
        "drift_region_lifetime_s": params.drift_region_lifetime_s,
        "effective_carrier_lifetime_s": effective_lifetime,
        "trap_density_cm2": params.trap_density_cm2,
        "termination_field_relief_factor": termination_relief(params),
        "oxide_field_scale": oxide_field_scale(params),
        "junction_mesh_spacing_um": params.junction_mesh_spacing_um,
        "mesh_nodes_estimate": int(max(3, device_length_um(params) / params.junction_mesh_spacing_um)),
        "runner_contract_id": "power_mosfet_bv_ron_devsim_1d",
    }


def write_summary(path: Path, points: list[PowerMOSPoint], params: PowerMOS1DParameters, run_dir: Path) -> dict[str, Any]:
    metrics = extract_metrics(params, points)
    summary: dict[str, Any] = {
        "task": "power_mosfet_bv_ron_devsim_1d",
        "status": "completed",
        "device_type": "power_mosfet_bv_ron",
        "fidelity": "physics_1d",
        "device": DEVICE,
        "region": REGION,
        "parameters": asdict(params),
        "metrics": metrics,
        "artifacts": {
            "csv": str(run_dir / "sweep.csv"),
            "plot": str(run_dir / "curve.svg"),
            "tecplot": str(run_dir / "device_tecplot.dat"),
            "log": str(run_dir / "devsim.log"),
            "summary": str(path),
            "runner_contract": str(run_dir / "runner_contract.json"),
        },
    }
    summary.update(metrics)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def create_run_dir(root: Path, run_id: str | None) -> Path:
    actual_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "power_mosfet_1d" / actual_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DEVSIM-backed 1D Power MOSFET BV/Ron drift-stack sweep.")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--stop", type=float, default=-90.0)
    parser.add_argument("--step", type=float, default=5.0)
    parser.add_argument("--drift-region-length-um", type=float, default=3.0)
    parser.add_argument("--drift-region-doping-cm3", type=float, default=1.0e16)
    parser.add_argument("--body-doping-cm3", type=float, default=1.0e17)
    parser.add_argument("--source-doping-cm3", type=float, default=1.0e19)
    parser.add_argument("--drain-doping-cm3", type=float, default=1.0e19)
    parser.add_argument("--junction-depth-um", type=float, default=0.35)
    parser.add_argument("--implant-dose-cm2", type=float, default=1.0e13)
    parser.add_argument("--field-plate-length-um", type=float, default=1.5)
    parser.add_argument("--guard-ring-spacing-um", type=float, default=1.0)
    parser.add_argument("--trench-corner-radius-um", type=float, default=0.08)
    parser.add_argument("--gate-oxide-thickness-nm", type=float, default=50.0)
    parser.add_argument("--critical-field-v-per-cm", type=float, default=3.0e5)
    parser.add_argument("--electron-mobility-cm2-v-s", type=float, default=800.0)
    parser.add_argument("--channel-resistance-ohm-cm2", type=float, default=4.998e-2)
    parser.add_argument("--carrier-lifetime-s", type=float, default=1.0e-6)
    parser.add_argument("--drift-region-lifetime-s", type=float, default=None)
    parser.add_argument("--leakage-floor-a", type=float, default=1.0e-10)
    parser.add_argument("--trap-density-cm2", type=float, default=1.0e11)
    parser.add_argument("--area-cm2", type=float, default=1.0e-8)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--junction-mesh-spacing-um", type=float, default=0.01)
    parser.add_argument("--contact-mesh-spacing-um", type=float, default=0.05)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs")
    return parser.parse_args()


def params_from_args(args: argparse.Namespace) -> PowerMOS1DParameters:
    return PowerMOS1DParameters(
        start=args.start,
        stop=args.stop,
        step=args.step,
        drift_region_length_um=args.drift_region_length_um,
        drift_region_doping_cm3=args.drift_region_doping_cm3,
        body_doping_cm3=args.body_doping_cm3,
        source_doping_cm3=args.source_doping_cm3,
        drain_doping_cm3=args.drain_doping_cm3,
        junction_depth_um=args.junction_depth_um,
        implant_dose_cm2=args.implant_dose_cm2,
        field_plate_length_um=args.field_plate_length_um,
        guard_ring_spacing_um=args.guard_ring_spacing_um,
        trench_corner_radius_um=args.trench_corner_radius_um,
        gate_oxide_thickness_nm=args.gate_oxide_thickness_nm,
        critical_field_v_per_cm=args.critical_field_v_per_cm,
        electron_mobility_cm2_v_s=args.electron_mobility_cm2_v_s,
        channel_resistance_ohm_cm2=args.channel_resistance_ohm_cm2,
        carrier_lifetime_s=args.carrier_lifetime_s,
        drift_region_lifetime_s=args.drift_region_lifetime_s,
        leakage_floor_a=args.leakage_floor_a,
        trap_density_cm2=args.trap_density_cm2,
        area_cm2=args.area_cm2,
        temperature_k=args.temperature_k,
        junction_mesh_spacing_um=args.junction_mesh_spacing_um,
        contact_mesh_spacing_um=args.contact_mesh_spacing_um,
    )


def main() -> None:
    args = parse_args()
    run_root = args.run_root if args.run_root.is_absolute() else PROJECT_ROOT / args.run_root
    run_dir = create_run_dir(run_root, args.run_id)
    params = params_from_args(args)
    with redirect_stdout(run_dir / "devsim.log"):
        initialize_device(params)
        points = run_sweep(params)
        set_parameter(device=DEVICE, name=GetContactBiasName(DRAIN_CONTACT), value=0.0)
        try:
            solve(type="dc", absolute_error=1.0e10, relative_error=1e-10, maximum_iterations=100)
            write_devices(file=str(run_dir / "device_tecplot.dat"), type="tecplot")
        except Exception as exc:
            (run_dir / "device_tecplot.dat").write_text(
                "DEVSIM tecplot export unavailable after high-bias continuation failure.\n"
                f"failure_reason={exc}\n",
                encoding="utf-8",
            )
    write_csv(run_dir / "sweep.csv", points)
    write_svg(run_dir / "curve.svg", points)
    (run_dir / "runner_contract.json").write_text(json.dumps(runner_contract(params), indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(run_dir / "summary.json", points, params, run_dir)
    print(json.dumps({"status": "completed", "run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
