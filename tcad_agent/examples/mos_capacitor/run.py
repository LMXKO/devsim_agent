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
    add_1d_contact,
    add_1d_interface,
    add_1d_mesh_line,
    add_1d_region,
    create_1d_mesh,
    create_device,
    finalize_mesh,
    get_contact_charge,
    set_parameter,
    solve,
    write_devices,
)
from devsim.python_packages.model_create import CreateNodeModel
from devsim.python_packages.simple_physics import (
    CreateOxideContact,
    CreateOxidePotentialOnly,
    CreateSiliconOxideInterface,
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact,
    GetContactBiasName,
    SetOxideParameters,
    SetSiliconParameters,
)


DEVICE = "MOSCapacitorDevice"
OXIDE_REGION = "OxideRegion"
SILICON_REGION = "SiliconRegion"
INTERFACE = "oxide_silicon"
GATE_CONTACT = "gate"
SUBSTRATE_CONTACT = "substrate"
ELEMENTARY_CHARGE_C = 1.602176634e-19
EPS0_F_PER_CM = 8.8541878128e-14
SIO2_RELATIVE_PERMITTIVITY = 3.9


@dataclass
class MOSCapacitorParameters:
    oxide_thickness_nm: float = 5.0
    silicon_thickness_um: float = 0.2
    substrate_doping_cm3: float = 1.0e17
    temperature_k: float = 300.0
    oxide_spacing_nm: float = 0.25
    silicon_spacing_um: float = 0.002
    fixed_oxide_charge_cm2: float = 0.0

    def validate(self) -> None:
        if self.oxide_thickness_nm <= 0:
            raise ValueError("oxide_thickness_nm must be positive")
        if self.silicon_thickness_um <= 0:
            raise ValueError("silicon_thickness_um must be positive")
        if self.substrate_doping_cm3 <= 0:
            raise ValueError("substrate_doping_cm3 must be positive")
        if self.temperature_k <= 0:
            raise ValueError("temperature_k must be positive")
        if self.oxide_spacing_nm <= 0 or self.silicon_spacing_um <= 0:
            raise ValueError("mesh spacings must be positive")
        if self.fixed_oxide_charge_cm2 < 0:
            raise ValueError("fixed_oxide_charge_cm2 must be non-negative")


@dataclass
class CVPoint:
    gate_voltage_v: float
    gate_charge_c_per_cm2: float
    capacitance_f_per_cm2: float | None


def nm_to_cm(value_nm: float) -> float:
    return value_nm * 1.0e-7


def um_to_cm(value_um: float) -> float:
    return value_um * 1.0e-4


def oxide_capacitance_f_per_cm2(oxide_thickness_nm: float) -> float:
    return EPS0_F_PER_CM * SIO2_RELATIVE_PERMITTIVITY / nm_to_cm(oxide_thickness_nm)


def fixed_charge_voltage_shift_v(params: MOSCapacitorParameters) -> float:
    if params.fixed_oxide_charge_cm2 <= 0:
        return 0.0
    cox = oxide_capacitance_f_per_cm2(params.oxide_thickness_nm)
    return ELEMENTARY_CHARGE_C * params.fixed_oxide_charge_cm2 / cox


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
    targets: list[float] = []
    direction = 1.0 if stop >= start else -1.0
    signed_step = abs(step) * direction
    value = start
    while (value <= stop + abs(step) * 1e-9) if direction > 0 else (value >= stop - abs(step) * 1e-9):
        targets.append(round(value, 12))
        value += signed_step
    return targets


def create_mesh(params: MOSCapacitorParameters, device: str = DEVICE) -> None:
    oxide_cm = nm_to_cm(params.oxide_thickness_nm)
    total_cm = oxide_cm + um_to_cm(params.silicon_thickness_um)
    create_1d_mesh(mesh="moscap")
    add_1d_mesh_line(mesh="moscap", pos=0.0, ps=nm_to_cm(params.oxide_spacing_nm), tag=GATE_CONTACT)
    add_1d_mesh_line(mesh="moscap", pos=oxide_cm, ps=nm_to_cm(params.oxide_spacing_nm), tag=INTERFACE)
    add_1d_mesh_line(mesh="moscap", pos=total_cm, ps=um_to_cm(params.silicon_spacing_um), tag=SUBSTRATE_CONTACT)
    add_1d_contact(mesh="moscap", name=GATE_CONTACT, tag=GATE_CONTACT, material="metal")
    add_1d_contact(mesh="moscap", name=SUBSTRATE_CONTACT, tag=SUBSTRATE_CONTACT, material="metal")
    add_1d_interface(mesh="moscap", tag=INTERFACE, name=INTERFACE)
    add_1d_region(mesh="moscap", material="Ox", region=OXIDE_REGION, tag1=GATE_CONTACT, tag2=INTERFACE)
    add_1d_region(mesh="moscap", material="Si", region=SILICON_REGION, tag1=INTERFACE, tag2=SUBSTRATE_CONTACT)
    finalize_mesh(mesh="moscap")
    create_device(mesh="moscap", device=device)


def set_substrate_doping(params: MOSCapacitorParameters, device: str = DEVICE) -> None:
    CreateNodeModel(device, SILICON_REGION, "Acceptors", f"{params.substrate_doping_cm3:.12e}")
    CreateNodeModel(device, SILICON_REGION, "Donors", "0")
    CreateNodeModel(device, SILICON_REGION, "NetDoping", "Donors-Acceptors")


def build_mos_capacitor(params: MOSCapacitorParameters, device: str = DEVICE) -> None:
    params.validate()
    create_mesh(params, device)

    SetOxideParameters(device, OXIDE_REGION, params.temperature_k)
    SetSiliconParameters(device, SILICON_REGION, params.temperature_k)
    set_substrate_doping(params, device)

    set_parameter(device=device, name=GetContactBiasName(GATE_CONTACT), value=0.0)
    set_parameter(device=device, name=GetContactBiasName(SUBSTRATE_CONTACT), value=0.0)

    CreateOxidePotentialOnly(device, OXIDE_REGION)
    CreateSiliconPotentialOnly(device, SILICON_REGION)
    CreateOxideContact(device, OXIDE_REGION, GATE_CONTACT)
    CreateSiliconPotentialOnlyContact(device, SILICON_REGION, SUBSTRATE_CONTACT)
    CreateSiliconOxideInterface(device, INTERFACE)
    solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=60)


def solve_gate_bias(voltage: float, device: str = DEVICE) -> float:
    set_parameter(device=device, name=GetContactBiasName(GATE_CONTACT), value=voltage)
    solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=60)
    return float(get_contact_charge(device=device, contact=GATE_CONTACT, equation="PotentialEquation"))


def run_cv_sweep(start: float, stop: float, step: float, *, voltage_shift_v: float = 0.0) -> list[CVPoint]:
    raw: list[tuple[float, float]] = []
    for voltage in voltage_targets(start, stop, step):
        raw.append((voltage, solve_gate_bias(voltage + voltage_shift_v)))
    points: list[CVPoint] = []
    for index, (voltage, charge) in enumerate(raw):
        capacitance = None
        if index > 0:
            previous_voltage, previous_charge = raw[index - 1]
            delta_v = voltage - previous_voltage
            if delta_v != 0:
                capacitance = (charge - previous_charge) / delta_v
        points.append(CVPoint(gate_voltage_v=voltage, gate_charge_c_per_cm2=charge, capacitance_f_per_cm2=capacitance))
    return points


def write_csv(path: Path, points: list[CVPoint]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(points[0]).keys()))
        writer.writeheader()
        for point in points:
            writer.writerow(asdict(point))


def write_plot(path: Path, points: list[CVPoint]) -> None:
    voltages = [point.gate_voltage_v for point in points if point.capacitance_f_per_cm2 is not None]
    capacitances = [abs(point.capacitance_f_per_cm2 or 0.0) for point in points if point.capacitance_f_per_cm2 is not None]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(voltages, capacitances, marker="o", linewidth=1.6)
    ax.set_xlabel("Gate bias (V)")
    ax.set_ylabel("|Cgate| (F/cm^2)")
    ax.set_title("1D MOS Capacitor C-V Sweep")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize_cv(points: list[CVPoint]) -> dict[str, float | list[float] | int | None]:
    caps = [abs(point.capacitance_f_per_cm2) for point in points if point.capacitance_f_per_cm2 is not None]
    charges = [point.gate_charge_c_per_cm2 for point in points]
    return {
        "points": len(points),
        "voltage_range_v": [points[0].gate_voltage_v, points[-1].gate_voltage_v],
        "min_gate_charge_c_per_cm2": min(charges),
        "max_gate_charge_c_per_cm2": max(charges),
        "min_capacitance_f_per_cm2": min(caps) if caps else None,
        "max_capacitance_f_per_cm2": max(caps) if caps else None,
        "final_capacitance_f_per_cm2": caps[-1] if caps else None,
    }


def write_summary(path: Path, points: list[CVPoint], run_dir: Path, params: MOSCapacitorParameters) -> None:
    summary = {
        "task": "mos_capacitor_cv_sweep",
        "status": "completed",
        "device": DEVICE,
        "oxide_region": OXIDE_REGION,
        "silicon_region": SILICON_REGION,
        "parameters": asdict(params),
        **summarize_cv(points),
        "fixed_charge_voltage_shift_v": fixed_charge_voltage_shift_v(params),
        "artifacts": {
            "csv": str(run_dir / "cv_sweep.csv"),
            "plot": str(run_dir / "cv_curve.png"),
            "tecplot": str(run_dir / "device_tecplot.dat"),
            "log": str(run_dir / "devsim.log"),
        },
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def create_run_dir(root: Path, run_id: str | None) -> Path:
    actual_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "mos_capacitor" / actual_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DEVSIM 1D MOS capacitor C-V sweep.")
    parser.add_argument("--start", type=float, default=-1.0, help="Start gate bias in volts.")
    parser.add_argument("--stop", type=float, default=1.0, help="Stop gate bias in volts.")
    parser.add_argument("--step", type=float, default=0.25, help="Gate bias step in volts.")
    parser.add_argument("--oxide-thickness-nm", type=float, default=5.0)
    parser.add_argument("--silicon-thickness-um", type=float, default=0.2)
    parser.add_argument("--substrate-doping-cm3", type=float, default=1.0e17)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--oxide-spacing-nm", type=float, default=0.25)
    parser.add_argument("--silicon-spacing-um", type=float, default=0.002)
    parser.add_argument("--fixed-oxide-charge-cm2", type=float, default=0.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root if args.run_root.is_absolute() else PROJECT_ROOT / args.run_root
    run_dir = create_run_dir(run_root, args.run_id)
    params = MOSCapacitorParameters(
        oxide_thickness_nm=args.oxide_thickness_nm,
        silicon_thickness_um=args.silicon_thickness_um,
        substrate_doping_cm3=args.substrate_doping_cm3,
        temperature_k=args.temperature_k,
        oxide_spacing_nm=args.oxide_spacing_nm,
        silicon_spacing_um=args.silicon_spacing_um,
        fixed_oxide_charge_cm2=args.fixed_oxide_charge_cm2,
    )
    with redirect_stdout(run_dir / "devsim.log"):
        build_mos_capacitor(params)
        points = run_cv_sweep(args.start, args.stop, args.step, voltage_shift_v=fixed_charge_voltage_shift_v(params))
        write_devices(file=str(run_dir / "device_tecplot.dat"), type="tecplot")

    write_csv(run_dir / "cv_sweep.csv", points)
    write_plot(run_dir / "cv_curve.png", points)
    write_summary(run_dir / "summary.json", points, run_dir, params)
    print(json.dumps({"status": "completed", "run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
