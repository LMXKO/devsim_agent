from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.schottky_calibration import SchottkyCalibrationRequest, run_schottky_calibration
from tcad_agent.task_spec import PROJECT_ROOT


def parse_float_list(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate Schottky IV parameters against a trusted curve.")
    parser.add_argument("--calibration-id", default=None)
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs" / "agent_tools")
    parser.add_argument("--target-curve", type=Path, default=None)
    parser.add_argument("--voltage-column", default="voltage_v")
    parser.add_argument("--current-column", default="current_a")
    parser.add_argument("--start", type=float, default=-0.2)
    parser.add_argument("--stop", type=float, default=0.4)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--area-cm2", type=float, default=1.0e-8)
    parser.add_argument("--richardson-a-per-cm2-k2", type=float, default=112.0)
    parser.add_argument("--barrier-values", default=None)
    parser.add_argument("--ideality-values", default=None)
    parser.add_argument("--series-resistance-values", default=None)
    parser.add_argument("--image-force-lowering-values", default=None)
    parser.add_argument("--verify-with-devsim", action="store_true")
    parser.add_argument("--devsim-timeout-seconds", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = {
        "calibration_id": args.calibration_id,
        "run_root": args.run_root,
        "target_curve_path": args.target_curve,
        "voltage_column": args.voltage_column,
        "current_column": args.current_column,
        "start": args.start,
        "stop": args.stop,
        "step": args.step,
        "temperature_k": args.temperature_k,
        "area_cm2": args.area_cm2,
        "richardson_a_per_cm2_k2": args.richardson_a_per_cm2_k2,
        "verify_with_devsim": args.verify_with_devsim,
        "devsim_timeout_seconds": args.devsim_timeout_seconds,
    }
    overrides = {
        "barrier_values_ev": parse_float_list(args.barrier_values),
        "ideality_values": parse_float_list(args.ideality_values),
        "series_resistance_values_ohm": parse_float_list(args.series_resistance_values),
        "image_force_lowering_values_ev": parse_float_list(args.image_force_lowering_values),
    }
    data.update({key: value for key, value in overrides.items() if value is not None})
    request = SchottkyCalibrationRequest.model_validate(data)
    state = run_schottky_calibration(request)
    print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if state.status == "completed" else 2)


if __name__ == "__main__":
    main()
