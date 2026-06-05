from __future__ import annotations

import argparse
import csv
import json
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from devsim import (
    add_gmsh_contact,
    add_gmsh_interface,
    add_gmsh_region,
    create_device,
    create_gmsh_mesh,
    finalize_mesh,
    get_contact_current,
    set_node_values,
    set_parameter,
    solve,
    write_devices,
)
from devsim.python_packages.model_create import CreateNodeModel, CreateSolution
from devsim.python_packages.simple_physics import (
    CreateOxideContact,
    CreateOxidePotentialOnly,
    CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact,
    CreateSiliconOxideInterface,
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact,
    GetContactBiasName,
    SetOxideParameters,
    SetSiliconParameters,
)

from tcad_agent.metrics import MOSFETPoint, extract_mosfet_metrics


DEVICE = "MOSFET2DDevice"
OXIDE_REGION = "OxideRegion"
SILICON_REGION = "SiliconRegion"
INTERFACE = "oxide_silicon"
GATE_CONTACT = "gate"
SOURCE_CONTACT = "source"
DRAIN_CONTACT = "drain"
BODY_CONTACT = "body"
ELEMENTARY_CHARGE_C = 1.602176634e-19
EPS0_F_PER_CM = 8.8541878128e-14
SIO2_RELATIVE_PERMITTIVITY = 3.9


@dataclass
class MOSFET2DParameters:
    length_um: float = 0.2
    oxide_thickness_nm: float = 5.0
    silicon_thickness_um: float = 0.05
    source_drain_length_um: float = 0.04
    source_drain_depth_um: float = 0.015
    substrate_doping_cm3: float = 1.0e17
    source_drain_doping_cm3: float = 1.0e20
    temperature_k: float = 300.0
    x_divisions: int = 12
    silicon_y_divisions: int = 4
    mobility_model: str = "constant"
    electron_mobility_cm2_v_s: float | None = None
    hole_mobility_cm2_v_s: float | None = None
    recombination_model: str = "srh"
    electron_lifetime_s: float = 1.0e-5
    hole_lifetime_s: float = 1.0e-5
    interface_trap_density_cm2: float = 0.0
    fixed_oxide_charge_cm2: float = 0.0
    impact_ionization_model: str = "none"
    model_strategy: str = "poisson_then_dd"
    solver_initial_absolute_error: float = 1.0
    solver_absolute_error: float = 1.0e10
    solver_relative_error: float = 1.0e-10
    solver_max_iterations: int = 80

    def validate(self) -> None:
        if self.length_um <= 0:
            raise ValueError("length_um must be positive")
        if self.oxide_thickness_nm <= 0:
            raise ValueError("oxide_thickness_nm must be positive")
        if self.silicon_thickness_um <= 0:
            raise ValueError("silicon_thickness_um must be positive")
        if self.source_drain_length_um <= 0 or self.source_drain_depth_um <= 0:
            raise ValueError("source/drain dimensions must be positive")
        if self.source_drain_length_um * 2.0 >= self.length_um:
            raise ValueError("source/drain length must leave a channel region")
        if self.source_drain_depth_um >= self.silicon_thickness_um:
            raise ValueError("source/drain depth must be less than silicon thickness")
        if self.substrate_doping_cm3 <= 0 or self.source_drain_doping_cm3 <= 0:
            raise ValueError("doping concentrations must be positive")
        if self.temperature_k <= 0:
            raise ValueError("temperature_k must be positive")
        if self.x_divisions < 4 or self.silicon_y_divisions < 3:
            raise ValueError("mesh divisions are too small for a 2D MOSFET")
        if self.mobility_model not in {"constant", "doping_dependent"}:
            raise ValueError("mobility_model must be constant or doping_dependent")
        if self.recombination_model not in {"none", "srh"}:
            raise ValueError("recombination_model must be none or srh")
        if self.impact_ionization_model not in {"none", "selberherr"}:
            raise ValueError("impact_ionization_model must be none or selberherr")
        if self.model_strategy not in {"poisson_then_dd", "dd_direct"}:
            raise ValueError("model_strategy must be poisson_then_dd or dd_direct")
        for name, value in [
            ("electron_mobility_cm2_v_s", self.electron_mobility_cm2_v_s),
            ("hole_mobility_cm2_v_s", self.hole_mobility_cm2_v_s),
        ]:
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")
        if self.electron_lifetime_s <= 0 or self.hole_lifetime_s <= 0:
            raise ValueError("carrier lifetimes must be positive")
        if self.interface_trap_density_cm2 < 0 or self.fixed_oxide_charge_cm2 < 0:
            raise ValueError("interface/fixed charge densities must be non-negative")
        if self.solver_initial_absolute_error <= 0 or self.solver_absolute_error <= 0 or self.solver_relative_error <= 0:
            raise ValueError("solver errors must be positive")
        if self.solver_max_iterations < 1:
            raise ValueError("solver_max_iterations must be positive")


@dataclass
class MOSFET2DPoint:
    sweep_type: str
    gate_voltage_v: float
    drain_voltage_v: float
    drain_electron_current_a: float
    drain_hole_current_a: float
    drain_total_current_a: float
    abs_drain_current_a: float


def nm_to_cm(value_nm: float) -> float:
    return value_nm * 1.0e-7


def um_to_cm(value_um: float) -> float:
    return value_um * 1.0e-4


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
    if step <= 0:
        raise ValueError("--step must be positive")
    direction = 1.0 if stop >= start else -1.0
    signed_step = abs(step) * direction
    value = start
    targets: list[float] = []
    while (value <= stop + abs(step) * 1e-9) if direction > 0 else (value >= stop - abs(step) * 1e-9):
        targets.append(round(value, 12))
        value += signed_step
    return targets


def unique_sorted(values: list[float]) -> list[float]:
    return sorted(set(round(value, 15) for value in values))


def nearest_index(values: list[float], target: float) -> int:
    return min(range(len(values)), key=lambda index: abs(values[index] - target))


def mesh_positions(params: MOSFET2DParameters) -> tuple[list[float], list[float]]:
    length = um_to_cm(params.length_um)
    oxide = nm_to_cm(params.oxide_thickness_nm)
    silicon = um_to_cm(params.silicon_thickness_um)
    sd_length = um_to_cm(params.source_drain_length_um)
    sd_depth = um_to_cm(params.source_drain_depth_um)

    xs = [index * length / params.x_divisions for index in range(params.x_divisions + 1)]
    xs.extend([sd_length, length - sd_length, length / 2.0])
    ys = [0.0, oxide]
    ys.extend([oxide + index * silicon / params.silicon_y_divisions for index in range(1, params.silicon_y_divisions + 1)])
    ys.extend([oxide + sd_depth, oxide + sd_depth / 2.0])
    return unique_sorted(xs), unique_sorted(ys)


def build_gmsh_like_mesh(params: MOSFET2DParameters) -> tuple[list[float], list[int], list[str]]:
    params.validate()
    length = um_to_cm(params.length_um)
    oxide = nm_to_cm(params.oxide_thickness_nm)
    silicon = um_to_cm(params.silicon_thickness_um)
    sd_depth = um_to_cm(params.source_drain_depth_um)
    xs, ys = mesh_positions(params)
    coordinates: list[float] = []
    node_index: dict[tuple[int, int], int] = {}
    for y_index, y in enumerate(ys):
        for x_index, x in enumerate(xs):
            node_index[(x_index, y_index)] = len(coordinates) // 3
            coordinates.extend([x, y, 0.0])

    physical_names = ["oxide", "silicon", "gate", "source", "drain", "body", "oxide_silicon"]
    oxide_id, silicon_id, gate_id, source_id, drain_id, body_id, interface_id = range(len(physical_names))
    elements: list[int] = []

    def triangle(physical: int, a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> None:
        elements.extend([2, physical, node_index[a], node_index[b], node_index[c]])

    def edge(physical: int, a: tuple[int, int], b: tuple[int, int]) -> None:
        elements.extend([1, physical, node_index[a], node_index[b]])

    oxide_top_index = nearest_index(ys, 0.0)
    interface_index = nearest_index(ys, oxide)
    bottom_index = nearest_index(ys, oxide + silicon)

    for y_index in range(len(ys) - 1):
        y_mid = (ys[y_index] + ys[y_index + 1]) / 2.0
        physical = oxide_id if y_mid < oxide else silicon_id
        for x_index in range(len(xs) - 1):
            triangle(physical, (x_index, y_index), (x_index + 1, y_index), (x_index + 1, y_index + 1))
            triangle(physical, (x_index, y_index), (x_index + 1, y_index + 1), (x_index, y_index + 1))

    for x_index in range(len(xs) - 1):
        edge(gate_id, (x_index, oxide_top_index), (x_index + 1, oxide_top_index))
        edge(interface_id, (x_index, interface_index), (x_index + 1, interface_index))
        edge(body_id, (x_index, bottom_index), (x_index + 1, bottom_index))

    source_depth_limit = oxide + sd_depth + 1e-30
    for y_index in range(interface_index, len(ys) - 1):
        if ys[y_index + 1] <= source_depth_limit:
            edge(source_id, (0, y_index), (0, y_index + 1))
            edge(drain_id, (len(xs) - 1, y_index), (len(xs) - 1, y_index + 1))

    return coordinates, elements, physical_names


def create_mesh(params: MOSFET2DParameters, device: str = DEVICE) -> None:
    coordinates, elements, physical_names = build_gmsh_like_mesh(params)
    mesh = "mosfet_2d"
    create_gmsh_mesh(mesh=mesh, coordinates=coordinates, elements=elements, physical_names=physical_names)
    add_gmsh_region(mesh=mesh, gmsh_name="oxide", region=OXIDE_REGION, material="Ox")
    add_gmsh_region(mesh=mesh, gmsh_name="silicon", region=SILICON_REGION, material="Si")
    add_gmsh_contact(mesh=mesh, gmsh_name="gate", name=GATE_CONTACT, region=OXIDE_REGION, material="metal")
    add_gmsh_contact(mesh=mesh, gmsh_name="source", name=SOURCE_CONTACT, region=SILICON_REGION, material="metal")
    add_gmsh_contact(mesh=mesh, gmsh_name="drain", name=DRAIN_CONTACT, region=SILICON_REGION, material="metal")
    add_gmsh_contact(mesh=mesh, gmsh_name="body", name=BODY_CONTACT, region=SILICON_REGION, material="metal")
    add_gmsh_interface(
        mesh=mesh,
        gmsh_name="oxide_silicon",
        name=INTERFACE,
        region0=OXIDE_REGION,
        region1=SILICON_REGION,
    )
    finalize_mesh(mesh=mesh)
    create_device(mesh=mesh, device=device)


def set_doping(params: MOSFET2DParameters, device: str = DEVICE) -> None:
    length = um_to_cm(params.length_um)
    oxide = nm_to_cm(params.oxide_thickness_nm)
    sd_length = um_to_cm(params.source_drain_length_um)
    sd_depth = um_to_cm(params.source_drain_depth_um)
    donors = (
        f"{params.source_drain_doping_cm3:.12e}"
        f"*ifelse(y<{oxide + sd_depth:.12e}, "
        f"ifelse(x<{sd_length:.12e}, 1, ifelse(x>{length - sd_length:.12e}, 1, 0)), 0)"
    )
    CreateNodeModel(device, SILICON_REGION, "Acceptors", f"{params.substrate_doping_cm3:.12e}")
    CreateNodeModel(device, SILICON_REGION, "Donors", donors)
    CreateNodeModel(device, SILICON_REGION, "NetDoping", "Donors-Acceptors")


def doping_dependent_mobility(
    doping_cm3: float,
    *,
    mu_min: float,
    mu_0: float,
    n_ref: float,
    alpha: float,
) -> float:
    ratio = max(doping_cm3, 1.0) / n_ref
    return mu_min + (mu_0 - mu_min) / (1.0 + ratio**alpha)


def effective_mobility_values(params: MOSFET2DParameters) -> tuple[float, float, str]:
    if params.mobility_model == "constant":
        return (
            params.electron_mobility_cm2_v_s or 400.0,
            params.hole_mobility_cm2_v_s or 200.0,
            "constant",
        )
    channel_doping = params.substrate_doping_cm3
    electron = params.electron_mobility_cm2_v_s or doping_dependent_mobility(
        channel_doping,
        mu_min=65.0,
        mu_0=1350.0,
        n_ref=1.0e17,
        alpha=0.72,
    )
    hole = params.hole_mobility_cm2_v_s or doping_dependent_mobility(
        channel_doping,
        mu_min=49.7,
        mu_0=480.0,
        n_ref=1.0e17,
        alpha=0.70,
    )
    return electron, hole, "doping_dependent_effective"


def oxide_capacitance_f_per_cm2(params: MOSFET2DParameters) -> float:
    return EPS0_F_PER_CM * SIO2_RELATIVE_PERMITTIVITY / nm_to_cm(params.oxide_thickness_nm)


def interface_trap_equivalent_charge_cm2(params: MOSFET2DParameters) -> float:
    # Compact first-order coupling: assume a modest occupied fraction near threshold.
    return 0.2 * params.interface_trap_density_cm2


def charge_coupled_gate_shift_v(params: MOSFET2DParameters) -> float:
    cox = oxide_capacitance_f_per_cm2(params)
    total_charge_cm2 = params.fixed_oxide_charge_cm2 + interface_trap_equivalent_charge_cm2(params)
    return ELEMENTARY_CHARGE_C * total_charge_cm2 / cox if cox > 0 else 0.0


def impact_ionization_multiplier(params: MOSFET2DParameters, drain_voltage: float) -> float:
    if params.impact_ionization_model == "none":
        return 1.0
    channel_cm = max(um_to_cm(params.length_um - 2.0 * params.source_drain_length_um), 1.0e-8)
    electric_field = abs(drain_voltage) / channel_cm
    critical_field = 3.0e5
    if electric_field <= 0.35 * critical_field:
        return 1.0
    excess = min(electric_field / critical_field, 0.95)
    return min(1.0 / max((1.0 - excess) ** 2, 1.0e-3), 50.0)


def physics_model_summary(params: MOSFET2DParameters) -> dict[str, float | str]:
    electron_mobility, hole_mobility, mobility_model_used = effective_mobility_values(params)
    electron_lifetime = params.electron_lifetime_s
    hole_lifetime = params.hole_lifetime_s
    if params.recombination_model == "none":
        electron_lifetime = 1.0e30
        hole_lifetime = 1.0e30
    gate_shift = charge_coupled_gate_shift_v(params)
    return {
        "mobility_model_used": mobility_model_used,
        "electron_mobility_cm2_v_s": electron_mobility,
        "hole_mobility_cm2_v_s": hole_mobility,
        "electron_lifetime_s": electron_lifetime,
        "hole_lifetime_s": hole_lifetime,
        "interface_trap_density_cm2": params.interface_trap_density_cm2,
        "interface_trap_equivalent_charge_cm2": interface_trap_equivalent_charge_cm2(params),
        "fixed_oxide_charge_cm2": params.fixed_oxide_charge_cm2,
        "charge_coupled_gate_shift_v": gate_shift,
        "impact_ionization_model": params.impact_ionization_model,
        "impact_ionization_coupling": "compact_drain_field_multiplier"
        if params.impact_ionization_model != "none"
        else "disabled",
        "model_strategy": params.model_strategy,
        "advanced_model_coupling": "compact_equivalent_bias_and_avalanche",
        "solver_max_iterations": params.solver_max_iterations,
        "solver_relative_error": params.solver_relative_error,
    }


def apply_physics_parameters(params: MOSFET2DParameters, device: str = DEVICE) -> dict[str, float | str]:
    physics_models = physics_model_summary(params)
    electron_mobility = float(physics_models["electron_mobility_cm2_v_s"])
    hole_mobility = float(physics_models["hole_mobility_cm2_v_s"])
    electron_lifetime = float(physics_models["electron_lifetime_s"])
    hole_lifetime = float(physics_models["hole_lifetime_s"])
    set_parameter(device=device, region=SILICON_REGION, name="mu_n", value=electron_mobility)
    set_parameter(device=device, region=SILICON_REGION, name="mu_p", value=hole_mobility)
    set_parameter(device=device, region=SILICON_REGION, name="taun", value=electron_lifetime)
    set_parameter(device=device, region=SILICON_REGION, name="taup", value=hole_lifetime)
    set_parameter(device=device, region=SILICON_REGION, name="InterfaceTrapDensity", value=params.interface_trap_density_cm2)
    set_parameter(device=device, region=OXIDE_REGION, name="FixedOxideCharge", value=params.fixed_oxide_charge_cm2)
    set_parameter(device=device, name="ChargeCoupledGateShift", value=float(physics_models["charge_coupled_gate_shift_v"]))
    return physics_models


def build_mosfet(params: MOSFET2DParameters, device: str = DEVICE) -> None:
    params.validate()
    create_mesh(params, device=device)
    SetOxideParameters(device, OXIDE_REGION, params.temperature_k)
    SetSiliconParameters(device, SILICON_REGION, params.temperature_k)
    apply_physics_parameters(params, device=device)
    set_doping(params, device=device)
    for contact in [GATE_CONTACT, SOURCE_CONTACT, DRAIN_CONTACT, BODY_CONTACT]:
        set_parameter(device=device, name=GetContactBiasName(contact), value=0.0)

    CreateOxidePotentialOnly(device, OXIDE_REGION)
    CreateSiliconPotentialOnly(device, SILICON_REGION)
    CreateOxideContact(device, OXIDE_REGION, GATE_CONTACT)
    for contact in [SOURCE_CONTACT, DRAIN_CONTACT, BODY_CONTACT]:
        CreateSiliconPotentialOnlyContact(device, SILICON_REGION, contact)
    CreateSiliconOxideInterface(device, INTERFACE)
    solve(
        type="dc",
        absolute_error=params.solver_initial_absolute_error,
        relative_error=params.solver_relative_error,
        maximum_iterations=params.solver_max_iterations,
    )

    CreateSolution(device, SILICON_REGION, "Electrons")
    CreateSolution(device, SILICON_REGION, "Holes")
    set_node_values(device=device, region=SILICON_REGION, name="Electrons", init_from="IntrinsicElectrons")
    set_node_values(device=device, region=SILICON_REGION, name="Holes", init_from="IntrinsicHoles")
    CreateSiliconDriftDiffusion(device, SILICON_REGION)
    for contact in [SOURCE_CONTACT, DRAIN_CONTACT, BODY_CONTACT]:
        CreateSiliconDriftDiffusionAtContact(device, SILICON_REGION, contact)
    solve(
        type="dc",
        absolute_error=params.solver_absolute_error,
        relative_error=params.solver_relative_error,
        maximum_iterations=params.solver_max_iterations,
    )


def solve_bias(
    gate_voltage: float,
    drain_voltage: float,
    params: MOSFET2DParameters | None = None,
    device: str = DEVICE,
) -> MOSFET2DPoint:
    effective_gate_voltage = gate_voltage + (charge_coupled_gate_shift_v(params) if params is not None else 0.0)
    set_parameter(device=device, name=GetContactBiasName(GATE_CONTACT), value=effective_gate_voltage)
    set_parameter(device=device, name=GetContactBiasName(DRAIN_CONTACT), value=drain_voltage)
    set_parameter(device=device, name=GetContactBiasName(SOURCE_CONTACT), value=0.0)
    set_parameter(device=device, name=GetContactBiasName(BODY_CONTACT), value=0.0)
    solve(
        type="dc",
        absolute_error=params.solver_absolute_error if params is not None else 1e10,
        relative_error=params.solver_relative_error if params is not None else 1e-10,
        maximum_iterations=params.solver_max_iterations if params is not None else 80,
    )
    electron_current = float(
        get_contact_current(device=device, contact=DRAIN_CONTACT, equation="ElectronContinuityEquation")
    )
    hole_current = float(get_contact_current(device=device, contact=DRAIN_CONTACT, equation="HoleContinuityEquation"))
    multiplier = impact_ionization_multiplier(params, drain_voltage) if params is not None else 1.0
    electron_current *= multiplier
    hole_current *= multiplier
    total = electron_current + hole_current
    return MOSFET2DPoint(
        sweep_type="",
        gate_voltage_v=gate_voltage,
        drain_voltage_v=drain_voltage,
        drain_electron_current_a=electron_current,
        drain_hole_current_a=hole_current,
        drain_total_current_a=total,
        abs_drain_current_a=abs(total),
    )


def run_idvg_sweep(
    start: float,
    stop: float,
    step: float,
    drain_voltage: float,
    params: MOSFET2DParameters | None = None,
) -> list[MOSFET2DPoint]:
    points = []
    for gate_voltage in voltage_targets(start, stop, step):
        point = solve_bias(gate_voltage, drain_voltage, params)
        point.sweep_type = "idvg"
        points.append(point)
    return points


def run_idvd_sweep(
    start: float,
    stop: float,
    step: float,
    gate_voltage: float,
    params: MOSFET2DParameters | None = None,
) -> list[MOSFET2DPoint]:
    points = []
    for drain_voltage in voltage_targets(start, stop, step):
        point = solve_bias(gate_voltage, drain_voltage, params)
        point.sweep_type = "idvd"
        points.append(point)
    return points


def write_csv(path: Path, points: list[MOSFET2DPoint]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(points[0]).keys()))
        writer.writeheader()
        for point in points:
            writer.writerow(asdict(point))


def write_plot(path: Path, points: list[MOSFET2DPoint]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    idvg = [point for point in points if point.sweep_type == "idvg"]
    idvd = [point for point in points if point.sweep_type == "idvd"]
    if idvg:
        axes[0].semilogy(
            [point.gate_voltage_v for point in idvg],
            [max(point.abs_drain_current_a, 1e-30) for point in idvg],
            marker="o",
            linewidth=1.6,
        )
    axes[0].set_xlabel("Gate voltage (V)")
    axes[0].set_ylabel("|Idrain| (A)")
    axes[0].set_title("2D MOSFET Id-Vg")
    axes[0].grid(True, which="both", alpha=0.35)
    if idvd:
        axes[1].plot(
            [point.drain_voltage_v for point in idvd],
            [point.abs_drain_current_a for point in idvd],
            marker="o",
            linewidth=1.6,
        )
    axes[1].set_xlabel("Drain voltage (V)")
    axes[1].set_ylabel("|Idrain| (A)")
    axes[1].set_title("2D MOSFET Id-Vd")
    axes[1].grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def metric_points(points: list[MOSFET2DPoint]) -> list[MOSFETPoint]:
    return [
        MOSFETPoint(
            sweep_type=point.sweep_type,
            gate_voltage_v=point.gate_voltage_v,
            drain_voltage_v=point.drain_voltage_v,
            drain_electron_current_a=point.drain_electron_current_a,
            drain_hole_current_a=point.drain_hole_current_a,
            drain_total_current_a=point.drain_total_current_a,
        )
        for point in points
    ]


def write_summary(
    path: Path,
    points: list[MOSFET2DPoint],
    run_dir: Path,
    params: MOSFET2DParameters,
    threshold_current_a: float,
) -> None:
    metrics = extract_mosfet_metrics(metric_points(points), threshold_current_a=threshold_current_a)
    physics_models = physics_model_summary(params)
    summary = {
        "task": "mosfet_2d_id_sweep",
        "status": "completed",
        "device": DEVICE,
        "oxide_region": OXIDE_REGION,
        "silicon_region": SILICON_REGION,
        "parameters": asdict(params),
        "physics_models": physics_models,
        "metrics": metrics,
        "artifacts": {
            "csv": str(run_dir / "mosfet_id_sweep.csv"),
            "plot": str(run_dir / "mosfet_id_curves.png"),
            "tecplot": str(run_dir / "device_tecplot.dat"),
            "log": str(run_dir / "devsim.log"),
        },
    }
    summary.update(metrics)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def create_run_dir(root: Path, run_id: str | None) -> Path:
    actual_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "mosfet_2d" / actual_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simplified DEVSIM 2D MOSFET Id-Vg / Id-Vd sweep.")
    parser.add_argument("--sweep-type", choices=["idvg", "idvd", "both"], default="both")
    parser.add_argument("--gate-start", type=float, default=0.0)
    parser.add_argument("--gate-stop", type=float, default=1.0)
    parser.add_argument("--gate-step", type=float, default=0.5)
    parser.add_argument("--drain-voltage", type=float, default=0.05, help="Fixed Vds for Id-Vg.")
    parser.add_argument("--drain-start", type=float, default=0.0)
    parser.add_argument("--drain-stop", type=float, default=0.1)
    parser.add_argument("--drain-step", type=float, default=0.05)
    parser.add_argument("--idvd-gate-voltage", type=float, default=1.0, help="Fixed Vgs for Id-Vd.")
    parser.add_argument("--threshold-current-a", type=float, default=1e-6)
    parser.add_argument("--length-um", type=float, default=0.2)
    parser.add_argument("--oxide-thickness-nm", type=float, default=5.0)
    parser.add_argument("--silicon-thickness-um", type=float, default=0.05)
    parser.add_argument("--source-drain-length-um", type=float, default=0.04)
    parser.add_argument("--source-drain-depth-um", type=float, default=0.015)
    parser.add_argument("--substrate-doping-cm3", type=float, default=1.0e17)
    parser.add_argument("--source-drain-doping-cm3", type=float, default=1.0e20)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--x-divisions", type=int, default=12)
    parser.add_argument("--silicon-y-divisions", type=int, default=4)
    parser.add_argument("--mobility-model", choices=["constant", "doping_dependent"], default="constant")
    parser.add_argument("--electron-mobility-cm2-v-s", type=float, default=None)
    parser.add_argument("--hole-mobility-cm2-v-s", type=float, default=None)
    parser.add_argument("--recombination-model", choices=["none", "srh"], default="srh")
    parser.add_argument("--electron-lifetime-s", type=float, default=1.0e-5)
    parser.add_argument("--hole-lifetime-s", type=float, default=1.0e-5)
    parser.add_argument("--interface-trap-density-cm2", type=float, default=0.0)
    parser.add_argument("--fixed-oxide-charge-cm2", type=float, default=0.0)
    parser.add_argument("--impact-ionization-model", choices=["none", "selberherr"], default="none")
    parser.add_argument("--model-strategy", choices=["poisson_then_dd", "dd_direct"], default="poisson_then_dd")
    parser.add_argument("--solver-initial-absolute-error", type=float, default=1.0)
    parser.add_argument("--solver-absolute-error", type=float, default=1.0e10)
    parser.add_argument("--solver-relative-error", type=float, default=1.0e-10)
    parser.add_argument("--solver-max-iterations", type=int, default=80)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root if args.run_root.is_absolute() else PROJECT_ROOT / args.run_root
    run_dir = create_run_dir(run_root, args.run_id)
    params = MOSFET2DParameters(
        length_um=args.length_um,
        oxide_thickness_nm=args.oxide_thickness_nm,
        silicon_thickness_um=args.silicon_thickness_um,
        source_drain_length_um=args.source_drain_length_um,
        source_drain_depth_um=args.source_drain_depth_um,
        substrate_doping_cm3=args.substrate_doping_cm3,
        source_drain_doping_cm3=args.source_drain_doping_cm3,
        temperature_k=args.temperature_k,
        x_divisions=args.x_divisions,
        silicon_y_divisions=args.silicon_y_divisions,
        mobility_model=args.mobility_model,
        electron_mobility_cm2_v_s=args.electron_mobility_cm2_v_s,
        hole_mobility_cm2_v_s=args.hole_mobility_cm2_v_s,
        recombination_model=args.recombination_model,
        electron_lifetime_s=args.electron_lifetime_s,
        hole_lifetime_s=args.hole_lifetime_s,
        interface_trap_density_cm2=args.interface_trap_density_cm2,
        fixed_oxide_charge_cm2=args.fixed_oxide_charge_cm2,
        impact_ionization_model=args.impact_ionization_model,
        model_strategy=args.model_strategy,
        solver_initial_absolute_error=args.solver_initial_absolute_error,
        solver_absolute_error=args.solver_absolute_error,
        solver_relative_error=args.solver_relative_error,
        solver_max_iterations=args.solver_max_iterations,
    )
    points: list[MOSFET2DPoint] = []
    with redirect_stdout(run_dir / "devsim.log"):
        build_mosfet(params)
        if args.sweep_type in {"idvg", "both"}:
            points.extend(run_idvg_sweep(args.gate_start, args.gate_stop, args.gate_step, args.drain_voltage, params))
        if args.sweep_type in {"idvd", "both"}:
            points.extend(run_idvd_sweep(args.drain_start, args.drain_stop, args.drain_step, args.idvd_gate_voltage, params))
        write_devices(file=str(run_dir / "device_tecplot.dat"), type="tecplot")

    write_csv(run_dir / "mosfet_id_sweep.csv", points)
    write_plot(run_dir / "mosfet_id_curves.png", points)
    write_summary(run_dir / "summary.json", points, run_dir, params, args.threshold_current_a)
    print(json.dumps({"status": "completed", "run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
