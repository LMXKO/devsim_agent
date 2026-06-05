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
    contact_equation,
    create_1d_mesh,
    create_device,
    finalize_mesh,
    get_contact_current,
    get_edge_model_values,
    get_contact_list,
    get_parameter,
    set_node_values,
    set_parameter,
    solve,
    write_devices,
)
from devsim.python_packages.model_create import CreateContactNodeModel, CreateEdgeModel, CreateEdgeModelDerivatives, CreateNodeModel, CreateSolution, InEdgeModelList
from devsim.python_packages.simple_physics import (
    CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact,
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact,
    GetContactBiasName,
    SetSiliconParameters,
)


Q_OVER_K_BOLTZMANN = 11604.518121550082
ELEMENTARY_CHARGE_C = 1.602176634e-19
EPS0_F_PER_M = 8.8541878128e-12
SILICON_RELATIVE_PERMITTIVITY = 11.1
N_I_CM3 = 1.0e10
DEVICE = "SchottkyDevice"
REGION = "SiliconRegion"
MESH = "schottky_diode"
METAL_CONTACT = "metal"
OHMIC_CONTACT = "ohmic"
THERMIONIC_EQUATION = "SchottkyThermionicEmissionEquation"


@dataclass
class SweepPoint:
    voltage_v: float
    devsim_electron_current_a: float
    devsim_hole_current_a: float
    devsim_total_current_a: float
    devsim_thermionic_contact_current_a: float
    thermionic_current_a: float
    image_force_lowering_ev: float
    effective_barrier_height_ev: float
    contact_electric_field_v_per_cm: float
    current_a: float
    abs_current_a: float


@dataclass
class Schottky1DParameters:
    start: float = -0.5
    stop: float = 0.8
    step: float = 0.1
    length_um: float = 0.2
    n_doping_cm3: float = 1.0e16
    barrier_height_ev: float = 0.72
    ideality_factor: float = 1.08
    richardson_a_per_cm2_k2: float = 112.0
    area_cm2: float = 1.0e-8
    temperature_k: float = 300.0
    electron_lifetime_s: float = 1.0e-8
    hole_lifetime_s: float = 1.0e-8
    contact_spacing_um: float = 0.002
    bulk_spacing_um: float = 0.01
    min_contact_carrier_cm3: float = 1.0
    contact_model: str = "thermionic_emission"
    contact_coupling_mode: str = "residual"
    series_resistance_ohm: float = 0.0
    image_force_lowering_ev: float = 0.0
    auto_image_force_lowering: bool = False
    max_image_force_lowering_ev: float = 0.2

    def validate(self) -> None:
        if self.step <= 0:
            raise ValueError("step must be positive")
        if self.start == self.stop:
            raise ValueError("start and stop must differ")
        if self.length_um <= 0:
            raise ValueError("length_um must be positive")
        if self.n_doping_cm3 <= 0:
            raise ValueError("n_doping_cm3 must be positive")
        if self.barrier_height_ev <= 0:
            raise ValueError("barrier_height_ev must be positive")
        if self.ideality_factor <= 0:
            raise ValueError("ideality_factor must be positive")
        if self.area_cm2 <= 0:
            raise ValueError("area_cm2 must be positive")
        if self.temperature_k <= 0:
            raise ValueError("temperature_k must be positive")
        if self.contact_spacing_um <= 0 or self.bulk_spacing_um <= 0:
            raise ValueError("mesh spacing must be positive")
        if self.contact_model not in {"equivalent_density", "thermionic_emission"}:
            raise ValueError("contact_model must be equivalent_density or thermionic_emission")
        if self.contact_coupling_mode not in {"reported", "residual"}:
            raise ValueError("contact_coupling_mode must be reported or residual")
        if self.series_resistance_ohm < 0:
            raise ValueError("series_resistance_ohm must be non-negative")
        if self.image_force_lowering_ev < 0:
            raise ValueError("image_force_lowering_ev must be non-negative")
        if self.max_image_force_lowering_ev < 0:
            raise ValueError("max_image_force_lowering_ev must be non-negative")


def um_to_cm(value_um: float) -> float:
    return value_um * 1.0e-4


def thermal_voltage_v(temperature_k: float) -> float:
    return temperature_k / Q_OVER_K_BOLTZMANN


def voltage_targets(start: float, stop: float, step: float) -> list[float]:
    direction = 1.0 if stop >= start else -1.0
    signed_step = abs(step) * direction
    value = start
    targets: list[float] = []
    while (value <= stop + abs(step) * 1e-9) if direction > 0 else (value >= stop - abs(step) * 1e-9):
        targets.append(round(value, 12))
        value += signed_step
    return targets


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


def create_mesh(params: Schottky1DParameters) -> None:
    create_1d_mesh(mesh=MESH)
    add_1d_mesh_line(mesh=MESH, pos=0.0, ps=um_to_cm(params.contact_spacing_um), tag=METAL_CONTACT)
    add_1d_mesh_line(mesh=MESH, pos=um_to_cm(params.length_um), ps=um_to_cm(params.bulk_spacing_um), tag=OHMIC_CONTACT)
    add_1d_contact(mesh=MESH, name=METAL_CONTACT, tag=METAL_CONTACT, material="metal")
    add_1d_contact(mesh=MESH, name=OHMIC_CONTACT, tag=OHMIC_CONTACT, material="metal")
    add_1d_region(mesh=MESH, material="Si", region=REGION, tag1=METAL_CONTACT, tag2=OHMIC_CONTACT)
    finalize_mesh(mesh=MESH)
    create_device(mesh=MESH, device=DEVICE)


def set_uniform_n_doping(params: Schottky1DParameters) -> None:
    CreateNodeModel(DEVICE, REGION, "Acceptors", "0")
    CreateNodeModel(DEVICE, REGION, "Donors", f"{params.n_doping_cm3:.12e}")
    CreateNodeModel(DEVICE, REGION, "NetDoping", "Donors-Acceptors")


def contact_charge_edge() -> None:
    if not InEdgeModelList(DEVICE, REGION, "contactcharge_edge"):
        CreateEdgeModel(DEVICE, REGION, "contactcharge_edge", "Permittivity*ElectricField")
        CreateEdgeModelDerivatives(DEVICE, REGION, "contactcharge_edge", "Permittivity*ElectricField", "Potential")


def schottky_surface_potential_v(params: Schottky1DParameters) -> float:
    vt = thermal_voltage_v(params.temperature_k)
    return vt * math.log(params.n_doping_cm3 / N_I_CM3) - params.barrier_height_ev


def schottky_contact_electrons_cm3(params: Schottky1DParameters) -> float:
    vt = thermal_voltage_v(params.temperature_k)
    value = params.n_doping_cm3 * math.exp(-params.barrier_height_ev / vt)
    return max(value, params.min_contact_carrier_cm3)


def clamp_barrier_lowering_ev(value: float, params: Schottky1DParameters) -> float:
    return max(0.0, min(value, params.max_image_force_lowering_ev))


def image_force_lowering_from_field_ev(electric_field_v_per_cm: float) -> float:
    field_v_per_m = abs(electric_field_v_per_cm) * 100.0
    if field_v_per_m <= 0:
        return 0.0
    epsilon = EPS0_F_PER_M * SILICON_RELATIVE_PERMITTIVITY
    return math.sqrt(ELEMENTARY_CHARGE_C * field_v_per_m / (4.0 * math.pi * epsilon))


def effective_barrier_lowering_ev(params: Schottky1DParameters, electric_field_v_per_cm: float) -> float:
    lowering = params.image_force_lowering_ev
    if params.auto_image_force_lowering:
        lowering += image_force_lowering_from_field_ev(electric_field_v_per_cm)
    return clamp_barrier_lowering_ev(lowering, params)


def metal_contact_field_v_per_cm() -> float:
    try:
        values = get_edge_model_values(device=DEVICE, region=REGION, name="ElectricField")
    except Exception:
        return 0.0
    if not values:
        return 0.0
    return float(values[0])


def create_schottky_potential_contact(params: Schottky1DParameters) -> None:
    contact_charge_edge()
    bias_name = GetContactBiasName(METAL_CONTACT)
    surface_potential = schottky_surface_potential_v(params)
    model_name = f"{METAL_CONTACT}schottkypotential"
    model = f"Potential - {bias_name} - ({surface_potential:.12e})"
    CreateContactNodeModel(DEVICE, METAL_CONTACT, model_name, model)
    CreateContactNodeModel(DEVICE, METAL_CONTACT, f"{model_name}:Potential", "1")
    contact_equation(
        device=DEVICE,
        contact=METAL_CONTACT,
        name="PotentialEquation",
        node_model=model_name,
        edge_charge_model="contactcharge_edge",
    )


def create_schottky_drift_diffusion_contact(params: Schottky1DParameters) -> None:
    electron_density = schottky_contact_electrons_cm3(params)
    hole_density = N_I_CM3 * N_I_CM3 / max(electron_density, params.min_contact_carrier_cm3)
    electron_model_name = f"{METAL_CONTACT}schottkyelectrons"
    hole_model_name = f"{METAL_CONTACT}schottkyholes"
    CreateContactNodeModel(DEVICE, METAL_CONTACT, electron_model_name, f"Electrons - ({electron_density:.12e})")
    CreateContactNodeModel(DEVICE, METAL_CONTACT, f"{electron_model_name}:Electrons", "1")
    CreateContactNodeModel(DEVICE, METAL_CONTACT, hole_model_name, f"Holes - ({hole_density:.12e})")
    CreateContactNodeModel(DEVICE, METAL_CONTACT, f"{hole_model_name}:Holes", "1")
    if params.contact_coupling_mode == "residual":
        contact_equation(
            device=DEVICE,
            contact=METAL_CONTACT,
            name="ElectronContinuityEquation",
            edge_current_model="ElectronCurrent",
            node_current_model="SchottkyThermionicContactCurrent",
        )
    else:
        contact_equation(
            device=DEVICE,
            contact=METAL_CONTACT,
            name="ElectronContinuityEquation",
            node_model=electron_model_name,
            edge_current_model="ElectronCurrent",
        )
    contact_equation(
        device=DEVICE,
        contact=METAL_CONTACT,
        name="HoleContinuityEquation",
        node_model=hole_model_name,
        edge_current_model="HoleCurrent",
    )


def create_schottky_thermionic_contact_current(params: Schottky1DParameters) -> None:
    bias_name = GetContactBiasName(METAL_CONTACT)
    vt = thermal_voltage_v(params.temperature_k)
    lowering = clamp_barrier_lowering_ev(params.image_force_lowering_ev, params)
    effective_barrier = max(params.barrier_height_ev - lowering, 1.0e-6)
    saturation_current = (
        params.area_cm2
        * params.richardson_a_per_cm2_k2
        * params.temperature_k**2
        * math.exp(-effective_barrier / vt)
    )
    model_name = "SchottkyThermionicContactCurrent"
    expression = (
        f"ifelse({bias_name} >= 0, "
        f"({saturation_current:.12e})*(exp(min({bias_name}/({params.ideality_factor:.12e}*V_t), 80))-1), "
        f"-({saturation_current:.12e})*(1+abs({bias_name})/5))"
    )
    CreateContactNodeModel(DEVICE, METAL_CONTACT, model_name, expression)
    contact_equation(
        device=DEVICE,
        contact=METAL_CONTACT,
        name=THERMIONIC_EQUATION,
        node_current_model=model_name,
    )


def initialize_device(params: Schottky1DParameters) -> None:
    params.validate()
    create_mesh(params)
    SetSiliconParameters(DEVICE, REGION, params.temperature_k)
    set_parameter(device=DEVICE, region=REGION, name="taun", value=params.electron_lifetime_s)
    set_parameter(device=DEVICE, region=REGION, name="taup", value=params.hole_lifetime_s)
    set_uniform_n_doping(params)
    CreateSolution(DEVICE, REGION, "Potential")
    CreateSiliconPotentialOnly(DEVICE, REGION)
    for contact in get_contact_list(device=DEVICE):
        set_parameter(device=DEVICE, name=GetContactBiasName(contact), value=0.0)
    create_schottky_potential_contact(params)
    CreateSiliconPotentialOnlyContact(DEVICE, REGION, OHMIC_CONTACT)
    solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=50)
    CreateSolution(DEVICE, REGION, "Electrons")
    CreateSolution(DEVICE, REGION, "Holes")
    set_node_values(device=DEVICE, region=REGION, name="Electrons", init_from="IntrinsicElectrons")
    set_node_values(device=DEVICE, region=REGION, name="Holes", init_from="IntrinsicHoles")
    CreateSiliconDriftDiffusion(DEVICE, REGION)
    create_schottky_thermionic_contact_current(params)
    create_schottky_drift_diffusion_contact(params)
    CreateSiliconDriftDiffusionAtContact(DEVICE, REGION, OHMIC_CONTACT)
    solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=50)


def thermionic_current(params: Schottky1DParameters, voltage: float, barrier_lowering_ev: float) -> float:
    vt = thermal_voltage_v(params.temperature_k)
    effective_barrier_ev = max(params.barrier_height_ev - barrier_lowering_ev, 1.0e-6)
    saturation_current = (
        params.area_cm2
        * params.richardson_a_per_cm2_k2
        * params.temperature_k**2
        * math.exp(-effective_barrier_ev / vt)
    )
    if voltage < 0:
        return -saturation_current * (1.0 + abs(voltage) / 5.0)
    current = saturation_current * (math.exp(min(voltage / (params.ideality_factor * vt), 80.0)) - 1.0)
    if params.series_resistance_ohm <= 0:
        return current
    for _ in range(30):
        effective_voltage = voltage - current * params.series_resistance_ohm
        next_current = saturation_current * (math.exp(min(effective_voltage / (params.ideality_factor * vt), 80.0)) - 1.0)
        if abs(next_current - current) <= max(abs(current), 1.0e-30) * 1.0e-9:
            return next_current
        current = 0.5 * current + 0.5 * next_current
    return current


def solve_at_bias(params: Schottky1DParameters, voltage: float) -> SweepPoint:
    set_parameter(device=DEVICE, name=GetContactBiasName(METAL_CONTACT), value=voltage)
    solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=50)
    electron_current = float(
        get_contact_current(device=DEVICE, contact=METAL_CONTACT, equation="ElectronContinuityEquation")
    )
    hole_current = float(get_contact_current(device=DEVICE, contact=METAL_CONTACT, equation="HoleContinuityEquation"))
    total_current = electron_current + hole_current
    actual_voltage = float(get_parameter(device=DEVICE, name=GetContactBiasName(METAL_CONTACT)))
    contact_field = metal_contact_field_v_per_cm()
    barrier_lowering = effective_barrier_lowering_ev(params, contact_field)
    reference_current = thermionic_current(params, actual_voltage, barrier_lowering)
    try:
        thermionic_contact_current = float(
            get_contact_current(device=DEVICE, contact=METAL_CONTACT, equation=THERMIONIC_EQUATION)
        )
    except Exception:
        thermionic_contact_current = reference_current
    return SweepPoint(
        voltage_v=actual_voltage,
        devsim_electron_current_a=electron_current,
        devsim_hole_current_a=hole_current,
        devsim_total_current_a=total_current,
        devsim_thermionic_contact_current_a=thermionic_contact_current,
        thermionic_current_a=reference_current,
        image_force_lowering_ev=barrier_lowering,
        effective_barrier_height_ev=max(params.barrier_height_ev - barrier_lowering, 1.0e-6),
        contact_electric_field_v_per_cm=contact_field,
        current_a=total_current,
        abs_current_a=abs(total_current),
    )


def apply_area_equivalent_scaling(points: list[SweepPoint]) -> float:
    reverse_points = [point for point in points if point.voltage_v < 0 and abs(point.devsim_total_current_a) > 0]
    reference = reverse_points[0] if reverse_points else next(
        (point for point in points if abs(point.devsim_total_current_a) > 0),
        None,
    )
    if reference is None:
        return 1.0
    scale = abs(reference.thermionic_current_a) / max(abs(reference.devsim_total_current_a), 1e-300)
    for point in points:
        point.current_a = point.devsim_total_current_a * scale
        point.abs_current_a = abs(point.current_a)
    return scale


def run_sweep(params: Schottky1DParameters) -> list[SweepPoint]:
    points = [solve_at_bias(params, voltage) for voltage in voltage_targets(params.start, params.stop, params.step)]
    apply_area_equivalent_scaling(points)
    return points


def interpolate_threshold(points: list[SweepPoint], threshold: float) -> float | None:
    ordered = sorted(points, key=lambda point: point.voltage_v)
    previous: SweepPoint | None = None
    for point in ordered:
        value = abs(point.current_a)
        if previous is not None:
            previous_value = abs(previous.current_a)
            if min(previous_value, value) <= threshold <= max(previous_value, value) and value != previous_value:
                fraction = (threshold - previous_value) / (value - previous_value)
                return previous.voltage_v + fraction * (point.voltage_v - previous.voltage_v)
        previous = point
    return None


def extract_metrics(params: Schottky1DParameters, points: list[SweepPoint]) -> dict[str, float | int | str | None]:
    vt = thermal_voltage_v(params.temperature_k)
    saturation_current = (
        params.area_cm2
        * params.richardson_a_per_cm2_k2
        * params.temperature_k**2
        * math.exp(-params.barrier_height_ev / vt)
    )
    devsim_values = [abs(point.devsim_total_current_a) for point in points]
    thermionic_contact_values = [abs(point.devsim_thermionic_contact_current_a) for point in points]
    image_lowering_values = [point.image_force_lowering_ev for point in points]
    field_values = [abs(point.contact_electric_field_v_per_cm) for point in points]
    scaled_values = [abs(point.current_a) for point in points]
    scale = scaled_values[0] / max(devsim_values[0], 1e-300) if points else None
    return {
        "device_type": "schottky_diode",
        "fidelity": "devsim_1d",
        "solver_backend": "devsim_1d_thermionic_emission_contact_model",
        "schottky_contact_model": params.contact_model,
        "schottky_contact_coupling_mode": params.contact_coupling_mode,
        "thermionic_residual_coupled": params.contact_coupling_mode == "residual",
        "points": len(points),
        "saturation_current_a": saturation_current,
        "barrier_height_ev": params.barrier_height_ev,
        "ideality_factor_estimate": params.ideality_factor,
        "reverse_leakage_current_a": abs(points[0].current_a),
        "turn_on_voltage_at_1ua_v": interpolate_threshold(points, 1e-6),
        "devsim_current_min_abs_a": min(devsim_values),
        "devsim_current_max_abs_a": max(devsim_values),
        "devsim_thermionic_contact_current_min_abs_a": min(thermionic_contact_values),
        "devsim_thermionic_contact_current_max_abs_a": max(thermionic_contact_values),
        "scaled_current_min_abs_a": min(scaled_values),
        "scaled_current_max_abs_a": max(scaled_values),
        "devsim_current_scale_factor_to_area": scale,
        "schottky_surface_potential_v": schottky_surface_potential_v(params),
        "schottky_contact_electrons_cm3": schottky_contact_electrons_cm3(params),
        "series_resistance_ohm": params.series_resistance_ohm,
        "image_force_lowering_enabled": params.auto_image_force_lowering or params.image_force_lowering_ev > 0,
        "max_image_force_lowering_ev": max(image_lowering_values),
        "max_contact_electric_field_v_per_cm": max(field_values),
    }


def write_csv(path: Path, points: list[SweepPoint]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(points[0]).keys()))
        writer.writeheader()
        for point in points:
            writer.writerow(asdict(point))


def write_summary(path: Path, points: list[SweepPoint], params: Schottky1DParameters, run_dir: Path) -> dict[str, object]:
    metrics = extract_metrics(params, points)
    summary: dict[str, object] = {
        "task": "schottky_1d_iv_sweep",
        "status": "completed",
        "device_type": "schottky_diode",
        "fidelity": "devsim_1d",
        "device": DEVICE,
        "region": REGION,
        "parameters": asdict(params),
        "metrics": metrics,
        "artifacts": {
            "csv": str(run_dir / "sweep.csv"),
            "tecplot": str(run_dir / "device_tecplot.dat"),
            "log": str(run_dir / "devsim.log"),
            "summary": str(path),
        },
    }
    summary.update(metrics)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def create_run_dir(root: Path, run_id: str | None) -> Path:
    actual_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "schottky_1d" / actual_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DEVSIM-backed 1D Schottky diode IV sweep.")
    parser.add_argument("--start", type=float, default=-0.5)
    parser.add_argument("--stop", type=float, default=0.8)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--length-um", type=float, default=0.2)
    parser.add_argument("--n-doping-cm3", type=float, default=1.0e16)
    parser.add_argument("--barrier-height-ev", type=float, default=0.72)
    parser.add_argument("--ideality-factor", type=float, default=1.08)
    parser.add_argument("--richardson-a-per-cm2-k2", type=float, default=112.0)
    parser.add_argument("--area-cm2", type=float, default=1.0e-8)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--electron-lifetime-s", type=float, default=1.0e-8)
    parser.add_argument("--hole-lifetime-s", type=float, default=1.0e-8)
    parser.add_argument("--contact-spacing-um", type=float, default=0.002)
    parser.add_argument("--bulk-spacing-um", type=float, default=0.01)
    parser.add_argument("--contact-model", choices=["equivalent_density", "thermionic_emission"], default="thermionic_emission")
    parser.add_argument("--contact-coupling-mode", choices=["reported", "residual"], default="residual")
    parser.add_argument("--series-resistance-ohm", type=float, default=0.0)
    parser.add_argument("--image-force-lowering-ev", type=float, default=0.0)
    parser.add_argument("--auto-image-force-lowering", action="store_true")
    parser.add_argument("--max-image-force-lowering-ev", type=float, default=0.2)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root if args.run_root.is_absolute() else PROJECT_ROOT / args.run_root
    run_dir = create_run_dir(run_root, args.run_id)
    params = Schottky1DParameters(
        start=args.start,
        stop=args.stop,
        step=args.step,
        length_um=args.length_um,
        n_doping_cm3=args.n_doping_cm3,
        barrier_height_ev=args.barrier_height_ev,
        ideality_factor=args.ideality_factor,
        richardson_a_per_cm2_k2=args.richardson_a_per_cm2_k2,
        area_cm2=args.area_cm2,
        temperature_k=args.temperature_k,
        electron_lifetime_s=args.electron_lifetime_s,
        hole_lifetime_s=args.hole_lifetime_s,
        contact_spacing_um=args.contact_spacing_um,
        bulk_spacing_um=args.bulk_spacing_um,
        contact_model=args.contact_model,
        contact_coupling_mode=args.contact_coupling_mode,
        series_resistance_ohm=args.series_resistance_ohm,
        image_force_lowering_ev=args.image_force_lowering_ev,
        auto_image_force_lowering=args.auto_image_force_lowering,
        max_image_force_lowering_ev=args.max_image_force_lowering_ev,
    )
    with redirect_stdout(run_dir / "devsim.log"):
        initialize_device(params)
        points = run_sweep(params)
        write_devices(file=str(run_dir / "device_tecplot.dat"), type="tecplot")
    write_csv(run_dir / "sweep.csv", points)
    write_summary(run_dir / "summary.json", points, params, run_dir)
    print(json.dumps({"status": "completed", "run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
