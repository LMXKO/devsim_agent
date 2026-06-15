from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from tcad_agent.examples.pn_junction.run import (
    PNJunctionParameters,
    build_pn_junction,
    redirect_stdout,
    run_sweep,
    write_devices,
    write_plot,
    write_summary,
)


def output_root() -> Path:
    raw_root = os.environ.get("ACTSOFT_USER_DECK_CORPUS_ROOT")
    if raw_root:
        return Path(raw_root).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "runs" / "user_deck_corpus"


def as_float(config: dict[str, Any], section: str, key: str) -> float:
    return float((config.get(section) or {})[key])


def build_parameters(config: dict[str, Any]) -> PNJunctionParameters:
    return PNJunctionParameters(
        length_um=as_float(config, "geometry", "length_um"),
        junction_um=as_float(config, "geometry", "junction_um"),
        p_doping_cm3=as_float(config, "doping", "p_doping_cm3"),
        n_doping_cm3=as_float(config, "doping", "n_doping_cm3"),
        temperature_k=as_float(config, "physics_models", "temperature_k"),
        electron_lifetime_s=as_float(config, "physics_models", "electron_lifetime_s"),
        hole_lifetime_s=as_float(config, "physics_models", "hole_lifetime_s"),
        contact_spacing_um=as_float(config, "mesh", "contact_spacing_um"),
        junction_spacing_um=as_float(config, "mesh", "junction_spacing_um"),
    )


def configured_sweeps(config: dict[str, Any]) -> list[dict[str, Any]]:
    bias = config.get("bias") or {}
    sweeps = bias.get("sweeps")
    if isinstance(sweeps, list) and sweeps:
        return [dict(item) for item in sweeps if isinstance(item, dict)]
    return [
        {
            "name": str(bias.get("name") or "iv"),
            "start": bias.get("start", 0.0),
            "stop": bias.get("stop", 0.2),
            "step": bias.get("step", 0.1),
        }
    ]


def write_multi_sweep_csv(path: Path, sweep_rows: list[tuple[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sweep", "voltage_v", "electron_current_a", "hole_current_a", "total_current_a"])
        for sweep_name, point in sweep_rows:
            writer.writerow(
                [
                    sweep_name,
                    point.voltage_v,
                    point.electron_current_a,
                    point.hole_current_a,
                    point.total_current_a,
                ]
            )


def run_public_pn_deck(deck_id: str, config: dict[str, Any]) -> dict[str, Any]:
    params = build_parameters(config)
    run_id = f"{deck_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir = output_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    sweep_rows: list[tuple[str, Any]] = []
    points = []
    with redirect_stdout(run_dir / "devsim.log"):
        build_pn_junction(params)
        for sweep in configured_sweeps(config):
            sweep_name = str(sweep.get("name") or "iv")
            sweep_points = run_sweep(float(sweep["start"]), float(sweep["stop"]), float(sweep["step"]))
            points.extend(sweep_points)
            sweep_rows.extend((sweep_name, point) for point in sweep_points)
        write_devices(file=str(run_dir / "device_tecplot.dat"), type="tecplot")

    csv_path = run_dir / "iv_sweep.csv"
    plot_path = run_dir / "iv_curve.png"
    summary_path = run_dir / "summary.json"
    write_multi_sweep_csv(csv_path, sweep_rows)
    write_plot(plot_path, points)
    write_summary(summary_path, points, run_dir, params)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    extracted = summary.get("extracted_metrics") if isinstance(summary.get("extracted_metrics"), dict) else {}
    metrics = {
        "deck_id": deck_id,
        "curve_points": len(points),
        "sweep_count": len(configured_sweeps(config)),
        "final_total_current_a": summary.get("final_total_current_a"),
        "min_total_current_a": summary.get("min_total_current_a"),
        "max_total_current_a": summary.get("max_total_current_a"),
        "length_um": params.length_um,
        "junction_um": params.junction_um,
        "p_doping_cm3": params.p_doping_cm3,
        "n_doping_cm3": params.n_doping_cm3,
        "electron_lifetime_s": params.electron_lifetime_s,
        "hole_lifetime_s": params.hole_lifetime_s,
        **{f"pn_{key}": value for key, value in extracted.items()},
    }
    quality_report = {
        "status": "passed" if len(points) >= 3 else "suspicious",
        "issues": [] if len(points) >= 3 else [{"code": "too_few_points", "severity": "warning"}],
        "metrics": metrics,
    }
    artifacts = {
        "csv": str(csv_path.resolve()),
        "plot": str(plot_path.resolve()),
        "tecplot": str((run_dir / "device_tecplot.dat").resolve()),
        "log": str((run_dir / "devsim.log").resolve()),
        "summary": str(summary_path.resolve()),
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

