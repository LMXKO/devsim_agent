from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("ACTSOFT_PROJECT_ROOT", Path(__file__).resolve().parents[3])).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tcad_agent.examples.pn_junction.run import (  # noqa: E402
    PNJunctionParameters,
    build_pn_junction,
    redirect_stdout,
    run_sweep,
    write_csv,
    write_devices,
    write_plot,
    write_summary,
)


geometry = {
    "length_um": 0.1,
    "junction_um": 0.05,
}

doping = {
    "p_doping_cm3": 1.0e18,
    "n_doping_cm3": 1.0e18,
}

physics_models = {
    "electron_lifetime_s": 1.0e-8,
    "hole_lifetime_s": 1.0e-8,
    "temperature_k": 300.0,
}

mesh = {
    "contact_spacing_um": 0.001,
    "junction_spacing_um": 1.0e-5,
}

bias = {
    "start": 0.0,
    "stop": 0.2,
    "step": 0.1,
}


def build_parameters() -> PNJunctionParameters:
    return PNJunctionParameters(
        length_um=float(geometry["length_um"]),
        junction_um=float(geometry["junction_um"]),
        p_doping_cm3=float(doping["p_doping_cm3"]),
        n_doping_cm3=float(doping["n_doping_cm3"]),
        temperature_k=float(physics_models["temperature_k"]),
        electron_lifetime_s=float(physics_models["electron_lifetime_s"]),
        hole_lifetime_s=float(physics_models["hole_lifetime_s"]),
        contact_spacing_um=float(mesh["contact_spacing_um"]),
        junction_spacing_um=float(mesh["junction_spacing_um"]),
    )


def output_root() -> Path:
    raw_root = os.environ.get("ACTSOFT_USER_DECK_ACCEPTANCE_ROOT")
    return Path(raw_root).expanduser().resolve() if raw_root else PROJECT_ROOT / "runs" / "user_deck_acceptance"


def run_acceptance_deck() -> dict[str, object]:
    params = build_parameters()
    run_id = f"public_pn_diode_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir = output_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    with redirect_stdout(run_dir / "devsim.log"):
        build_pn_junction(params)
        points = run_sweep(float(bias["start"]), float(bias["stop"]), float(bias["step"]))
        write_devices(file=str(run_dir / "device_tecplot.dat"), type="tecplot")

    csv_path = run_dir / "iv_sweep.csv"
    plot_path = run_dir / "iv_curve.png"
    summary_path = run_dir / "summary.json"
    write_csv(csv_path, points)
    write_plot(plot_path, points)
    write_summary(summary_path, points, run_dir, params)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    extracted = summary.get("extracted_metrics") if isinstance(summary.get("extracted_metrics"), dict) else {}
    metrics = {
        "curve_points": len(points),
        "final_total_current_a": summary.get("final_total_current_a"),
        "min_total_current_a": summary.get("min_total_current_a"),
        "max_total_current_a": summary.get("max_total_current_a"),
        "p_doping_cm3": params.p_doping_cm3,
        "n_doping_cm3": params.n_doping_cm3,
        "electron_lifetime_s": params.electron_lifetime_s,
        "hole_lifetime_s": params.hole_lifetime_s,
        **{f"pn_{key}": value for key, value in extracted.items()},
    }
    artifacts = {
        "csv": str(csv_path.resolve()),
        "plot": str(plot_path.resolve()),
        "tecplot": str((run_dir / "device_tecplot.dat").resolve()),
        "log": str((run_dir / "devsim.log").resolve()),
        "summary": str(summary_path.resolve()),
    }
    quality_report = {
        "status": "passed" if len(points) >= 3 else "suspicious",
        "issues": [] if len(points) >= 3 else [{"code": "too_few_points", "severity": "warning"}],
        "metrics": metrics,
    }
    return {
        "status": "completed",
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "summary_path": str(summary_path.resolve()),
        "metrics": metrics,
        "artifacts": artifacts,
        "quality_report": quality_report,
    }


if __name__ == "__main__":
    print(json.dumps(run_acceptance_deck(), sort_keys=True))
