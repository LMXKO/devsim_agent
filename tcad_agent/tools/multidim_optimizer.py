from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from tcad_agent.multidim_optimizer import (
    AxisScale,
    MultiDimAxis,
    MultiDimOptimizationRequest,
    MultiDimOptimizationStatus,
    run_multidim_optimization,
)
from tcad_agent.parameter_sweep import SweepDirection, SweepObjective
from tcad_agent.task_spec import PROJECT_ROOT
from tcad_agent.tools.task_runner import resolve_task_spec


def parse_axis_spec(value: str) -> MultiDimAxis:
    parts = [part.strip() for part in value.split(":")]
    if len(parts) not in {5, 6}:
        raise ValueError(
            "axis must use path:scale:min:max:initial_points[:max_new_points_per_round] syntax"
        )
    path, scale, min_value, max_value, initial_points, *rest = parts
    return MultiDimAxis(
        path=path,
        scale=AxisScale(scale),
        min_value=float(min_value),
        max_value=float(max_value),
        initial_points=int(initial_points),
        max_new_points_per_round=int(rest[0]) if rest else 2,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a checkpointed multi-dimensional TCAD optimizer.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Base natural-language TCAD task.")
    source.add_argument("--task", type=Path, help="Base task.json path.")
    parser.add_argument("--optimize-id", default=None)
    parser.add_argument("--task-id", default=None, help="Task id for the base task when using --text.")
    parser.add_argument(
        "--axis",
        action="append",
        required=True,
        help=(
            "Axis as path:scale:min:max:initial_points[:max_new_points_per_round]. "
            "Example: parameters.p_doping_cm3:log:1e16:1e18:2"
        ),
    )
    parser.add_argument(
        "--objective-metric",
        default="final_quality_report.metrics.final_total_current_a",
    )
    parser.add_argument("--direction", choices=["minimize", "maximize"], default="minimize")
    parser.add_argument("--raw-objective", action="store_true", help="Do not take absolute value of objective.")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=80)
    parser.add_argument("--max-cases-per-round", type=int, default=12)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--optimize-root", type=Path, default=PROJECT_ROOT / "runs" / "optimizations")
    parser.add_argument("--planner", choices=["deterministic", "llm"], default="deterministic")
    parser.add_argument("--no-planner-fallback", action="store_true")
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument("--use-llm", dest="use_llm", action="store_true")
    llm_group.add_argument("--no-llm", dest="use_llm", action="store_false")
    parser.set_defaults(use_llm=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        base_spec, planning_result = resolve_task_spec(
            text=args.text,
            task_path=args.task,
            task_id=args.task_id,
            use_llm=args.use_llm,
            planner=args.planner,
            allow_planner_fallback=not args.no_planner_fallback,
        )
        request = MultiDimOptimizationRequest(
            optimize_id=args.optimize_id,
            axes=[parse_axis_spec(axis) for axis in args.axis],
            objective=SweepObjective(
                metric_path=args.objective_metric,
                direction=SweepDirection(args.direction),
                absolute=not args.raw_objective,
            ),
            execute=args.execute,
            overwrite=args.overwrite,
            use_llm=args.use_llm,
            optimize_root=args.optimize_root,
            max_rounds=args.max_rounds,
            max_cases=args.max_cases,
            max_cases_per_round=args.max_cases_per_round,
        )
        state = run_multidim_optimization(base_spec, request)
        if planning_result:
            plan_path = Path(state.optimize_dir) / "base_task_plan_result.json"
            plan_path.write_text(
                json.dumps(planning_result.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if state.status != MultiDimOptimizationStatus.FAILED else 1)
    except (ValidationError, ValueError, FileExistsError, FileNotFoundError) as exc:
        print(
            json.dumps(
                {
                    "tool_name": "multidim_optimizer",
                    "status": MultiDimOptimizationStatus.FAILED,
                    "failure_reason": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
