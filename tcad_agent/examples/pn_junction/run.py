from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from contextlib import contextmanager
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
    add_1d_mesh_line,
    add_1d_region,
    create_1d_mesh,
    create_device,
    finalize_mesh,
    get_contact_current,
    get_contact_list,
    get_parameter,
    set_node_values,
    set_parameter,
    solve,
    write_devices,
)
from devsim.python_packages.model_create import CreateNodeModel, CreateSolution
from devsim.python_packages.simple_physics import (
    CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact,
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact,
    GetContactBiasName,
    SetSiliconParameters,
)

from tcad_agent.metrics import IVPoint, extract_pn_iv_metrics


DEVICE = "PNJunctionDevice"
REGION = "SiliconRegion"
TOP_CONTACT = "top"
BOT_CONTACT = "bot"


@dataclass
class SweepPoint:
    voltage_v: float
    electron_current_a: float
    hole_current_a: float
    total_current_a: float


@dataclass
class PNJunctionParameters:
    length_um: float = 0.1
    junction_um: float = 0.05
    p_doping_cm3: float = 1.0e18
    n_doping_cm3: float = 1.0e18
    temperature_k: float = 300.0
    electron_lifetime_s: float = 1.0e-8
    hole_lifetime_s: float = 1.0e-8
    contact_spacing_um: float = 0.001
    junction_spacing_um: float = 1.0e-5

    def validate(self) -> None:
        if self.length_um <= 0:
            raise ValueError("length_um must be positive")
        if not 0 < self.junction_um < self.length_um:
            raise ValueError("junction_um must be between 0 and length_um")
        if self.p_doping_cm3 <= 0 or self.n_doping_cm3 <= 0:
            raise ValueError("doping concentrations must be positive")
        if self.temperature_k <= 0:
            raise ValueError("temperature_k must be positive")
        if self.electron_lifetime_s <= 0 or self.hole_lifetime_s <= 0:
            raise ValueError("carrier lifetimes must be positive")
        if self.contact_spacing_um <= 0 or self.junction_spacing_um <= 0:
            raise ValueError("mesh spacings must be positive")


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


def create_mesh(
    params: PNJunctionParameters,
    device: str = DEVICE,
    region: str = REGION,
) -> None:
    create_1d_mesh(mesh="pn_diode")
    add_1d_mesh_line(
        mesh="pn_diode",
        pos=0,
        ps=um_to_cm(params.contact_spacing_um),
        tag=TOP_CONTACT,
    )
    add_1d_mesh_line(
        mesh="pn_diode",
        pos=um_to_cm(params.junction_um),
        ps=um_to_cm(params.junction_spacing_um),
        tag="junction",
    )
    add_1d_mesh_line(
        mesh="pn_diode",
        pos=um_to_cm(params.length_um),
        ps=um_to_cm(params.contact_spacing_um),
        tag=BOT_CONTACT,
    )
    add_1d_contact(mesh="pn_diode", name=TOP_CONTACT, tag=TOP_CONTACT, material="metal")
    add_1d_contact(mesh="pn_diode", name=BOT_CONTACT, tag=BOT_CONTACT, material="metal")
    add_1d_region(
        mesh="pn_diode",
        material="Si",
        region=region,
        tag1=TOP_CONTACT,
        tag2=BOT_CONTACT,
    )
    finalize_mesh(mesh="pn_diode")
    create_device(mesh="pn_diode", device=device)


def set_doping(
    params: PNJunctionParameters,
    device: str = DEVICE,
    region: str = REGION,
) -> None:
    junction_cm = um_to_cm(params.junction_um)
    CreateNodeModel(device, region, "Acceptors", f"{params.p_doping_cm3:.12e}*step({junction_cm:.12e}-x)")
    CreateNodeModel(device, region, "Donors", f"{params.n_doping_cm3:.12e}*step(x-{junction_cm:.12e})")
    CreateNodeModel(device, region, "NetDoping", "Donors-Acceptors")


def initialize_solution(device: str = DEVICE, region: str = REGION) -> None:
    CreateSolution(device, region, "Potential")
    CreateSiliconPotentialOnly(device, region)
    for contact in get_contact_list(device=device):
        set_parameter(device=device, name=GetContactBiasName(contact), value=0.0)
        CreateSiliconPotentialOnlyContact(device, region, contact)


def enable_drift_diffusion(device: str = DEVICE, region: str = REGION) -> None:
    CreateSolution(device, region, "Electrons")
    CreateSolution(device, region, "Holes")
    set_node_values(
        device=device,
        region=region,
        name="Electrons",
        init_from="IntrinsicElectrons",
    )
    set_node_values(
        device=device,
        region=region,
        name="Holes",
        init_from="IntrinsicHoles",
    )
    CreateSiliconDriftDiffusion(device, region)
    for contact in get_contact_list(device=device):
        CreateSiliconDriftDiffusionAtContact(device, region, contact)


def build_pn_junction(
    params: PNJunctionParameters,
    device: str = DEVICE,
    region: str = REGION,
) -> None:
    params.validate()
    create_mesh(params, device, region)
    SetSiliconParameters(device, region, params.temperature_k)
    set_parameter(device=device, region=region, name="taun", value=params.electron_lifetime_s)
    set_parameter(device=device, region=region, name="taup", value=params.hole_lifetime_s)
    set_doping(params, device, region)
    initialize_solution(device, region)
    solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=30)
    enable_drift_diffusion(device, region)
    solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)


def voltage_targets(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("--step must be positive")
    targets: list[float] = []
    value = start
    if stop >= start:
        while value <= stop + step * 1e-9:
            targets.append(round(value, 12))
            value += step
    else:
        while value >= stop - step * 1e-9:
            targets.append(round(value, 12))
            value -= step
    return targets


def solve_at_bias(voltage: float, device: str = DEVICE, contact: str = TOP_CONTACT) -> SweepPoint:
    set_parameter(device=device, name=GetContactBiasName(contact), value=voltage)
    solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)
    electron_current = get_contact_current(
        device=device,
        contact=contact,
        equation="ElectronContinuityEquation",
    )
    hole_current = get_contact_current(
        device=device,
        contact=contact,
        equation="HoleContinuityEquation",
    )
    actual_voltage = get_parameter(device=device, name=GetContactBiasName(contact))
    return SweepPoint(
        voltage_v=float(actual_voltage),
        electron_current_a=float(electron_current),
        hole_current_a=float(hole_current),
        total_current_a=float(electron_current + hole_current),
    )


def run_sweep(start: float, stop: float, step: float) -> list[SweepPoint]:
    points = []
    for voltage in voltage_targets(start, stop, step):
        points.append(solve_at_bias(voltage))
    return points


def write_csv(path: Path, points: list[SweepPoint]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(points[0]).keys()))
        writer.writeheader()
        for point in points:
            writer.writerow(asdict(point))


def write_plot(path: Path, points: list[SweepPoint]) -> None:
    voltages = [point.voltage_v for point in points]
    currents = [abs(point.total_current_a) for point in points]
    floor = max(min((current for current in currents if current > 0), default=1e-30), 1e-30)
    plot_currents = [max(current, floor) for current in currents]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(voltages, plot_currents, marker="o", linewidth=1.6)
    ax.set_xlabel("Top contact bias (V)")
    ax.set_ylabel("|Total current| (A)")
    ax.set_title("1D PN Junction IV Sweep")
    ax.grid(True, which="both", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_summary(
    path: Path,
    points: list[SweepPoint],
    run_dir: Path,
    params: PNJunctionParameters,
) -> None:
    total_currents = [point.total_current_a for point in points]
    extracted_metrics = extract_pn_iv_metrics(
        [
            IVPoint(
                voltage_v=point.voltage_v,
                electron_current_a=point.electron_current_a,
                hole_current_a=point.hole_current_a,
                total_current_a=point.total_current_a,
            )
            for point in points
        ],
        temperature_k=params.temperature_k,
    )
    summary = {
        "task": "pn_junction_iv_sweep",
        "status": "completed",
        "device": DEVICE,
        "region": REGION,
        "parameters": asdict(params),
        "points": len(points),
        "voltage_range_v": [points[0].voltage_v, points[-1].voltage_v],
        "min_total_current_a": min(total_currents),
        "max_total_current_a": max(total_currents),
        "final_total_current_a": total_currents[-1],
        "extracted_metrics": extracted_metrics,
        "artifacts": {
            "csv": str(run_dir / "iv_sweep.csv"),
            "plot": str(run_dir / "iv_curve.png"),
            "tecplot": str(run_dir / "device_tecplot.dat"),
            "log": str(run_dir / "devsim.log"),
        },
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def create_run_dir(root: Path, run_id: str | None) -> Path:
    actual_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "pn_junction" / actual_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DEVSIM 1D PN junction IV sweep.")
    parser.add_argument("--start", type=float, default=0.0, help="Start bias in volts.")
    parser.add_argument("--stop", type=float, default=0.5, help="Stop bias in volts.")
    parser.add_argument("--step", type=float, default=0.1, help="Bias step in volts.")
    parser.add_argument("--length-um", type=float, default=0.1)
    parser.add_argument("--junction-um", type=float, default=0.05)
    parser.add_argument("--p-doping-cm3", type=float, default=1.0e18)
    parser.add_argument("--n-doping-cm3", type=float, default=1.0e18)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--electron-lifetime-s", type=float, default=1.0e-8)
    parser.add_argument("--hole-lifetime-s", type=float, default=1.0e-8)
    parser.add_argument("--contact-spacing-um", type=float, default=0.001)
    parser.add_argument("--junction-spacing-um", type=float, default=1.0e-5)
    parser.add_argument("--run-id", default=None, help="Optional run directory id.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=PROJECT_ROOT / "runs",
        help="Root directory for generated run artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root if args.run_root.is_absolute() else PROJECT_ROOT / args.run_root
    run_dir = create_run_dir(run_root, args.run_id)
    params = PNJunctionParameters(
        length_um=args.length_um,
        junction_um=args.junction_um,
        p_doping_cm3=args.p_doping_cm3,
        n_doping_cm3=args.n_doping_cm3,
        temperature_k=args.temperature_k,
        electron_lifetime_s=args.electron_lifetime_s,
        hole_lifetime_s=args.hole_lifetime_s,
        contact_spacing_um=args.contact_spacing_um,
        junction_spacing_um=args.junction_spacing_um,
    )

    with redirect_stdout(run_dir / "devsim.log"):
        build_pn_junction(params)
        points = run_sweep(args.start, args.stop, args.step)
        write_devices(file=str(run_dir / "device_tecplot.dat"), type="tecplot")

    write_csv(run_dir / "iv_sweep.csv", points)
    write_plot(run_dir / "iv_curve.png", points)
    write_summary(run_dir / "summary.json", points, run_dir, params)

    print(json.dumps({"status": "completed", "run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
