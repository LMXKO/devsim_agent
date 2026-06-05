from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from tcad_agent.parameter_sweep import (
    ParameterSweepRequest,
    SweepDirection,
    SweepObjective,
    parse_axis_spec,
    run_parameter_sweep,
)
from tcad_agent.sweep_planner import (
    SweepPlannerStatus,
    deterministic_sweep_plan,
    plan_sweep_text_with_llm,
    sweep_plan_from_result,
    write_sweep_planning_result,
)
from tcad_agent.task_spec import PROJECT_ROOT
from tcad_agent.tools.task_runner import resolve_task_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a parameter sweep over TaskSpec fields.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Base natural-language TCAD task.")
    source.add_argument("--task", type=Path, help="Base task.json path.")
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--task-id", default=None, help="Task id for the base task when using --text.")
    parser.add_argument(
        "--axis",
        action="append",
        help="Sweep axis as path=value1,value2. Example: parameters.p_doping_cm3=1e16,1e17",
    )
    parser.add_argument(
        "--objective-metric",
        default="final_quality_report.metrics.final_total_current_a",
    )
    parser.add_argument("--direction", choices=["minimize", "maximize"], default="minimize")
    parser.add_argument("--raw-objective", action="store_true", help="Do not take absolute value of objective.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-cases", type=int, default=100)
    parser.add_argument("--sweep-root", type=Path, default=PROJECT_ROOT / "runs" / "sweeps")
    parser.add_argument("--sweep-planner", choices=["manual", "deterministic", "llm"], default="manual")
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
        if args.sweep_planner == "manual":
            if not args.axis:
                raise ValueError("--axis is required when --sweep-planner manual is used")
            base_spec, _ = resolve_task_spec(
                text=args.text,
                task_path=args.task,
                task_id=args.task_id,
                use_llm=args.use_llm,
                planner=args.planner,
                allow_planner_fallback=not args.no_planner_fallback,
            )
            request = ParameterSweepRequest(
                sweep_id=args.sweep_id,
                axes=[parse_axis_spec(axis) for axis in args.axis],
                objective=SweepObjective(
                    metric_path=args.objective_metric,
                    direction=SweepDirection(args.direction),
                    absolute=not args.raw_objective,
                ),
                execute=args.execute,
                overwrite=args.overwrite,
                use_llm=args.use_llm,
                sweep_root=args.sweep_root,
                max_cases=args.max_cases,
            )
        elif args.sweep_planner == "deterministic":
            if not args.text:
                raise ValueError("--text is required for deterministic sweep planning")
            base_spec, request, _ = deterministic_sweep_plan(
                args.text,
                sweep_id=args.sweep_id,
                task_id=args.task_id,
                execution_use_llm=args.use_llm,
                execute=args.execute,
                max_cases=args.max_cases,
            )
            request = request.model_copy(
                update={
                    "overwrite": args.overwrite,
                    "sweep_root": args.sweep_root,
                }
            )
        else:
            if not args.text:
                raise ValueError("--text is required for LLM sweep planning")
            planning_result = plan_sweep_text_with_llm(
                args.text,
                sweep_id=args.sweep_id,
                task_id=args.task_id,
                execution_use_llm=args.use_llm,
                execute=args.execute,
                max_cases=args.max_cases,
                allow_fallback=not args.no_planner_fallback,
            )
            plan_id = planning_result.sweep_id or args.sweep_id or "planned_sweep"
            write_sweep_planning_result(
                planning_result,
                PROJECT_ROOT / "runs" / "sweep_plans" / plan_id / "sweep_plan_result.json",
            )
            if planning_result.status == SweepPlannerStatus.FAILED:
                raise ValueError("; ".join(planning_result.validation_errors) or "LLM sweep planner failed")
            base_spec, request = sweep_plan_from_result(planning_result)
            request = request.model_copy(
                update={
                    "execute": args.execute,
                    "overwrite": args.overwrite,
                    "use_llm": args.use_llm,
                    "sweep_root": args.sweep_root,
                    "max_cases": args.max_cases,
                }
            )
        state = run_parameter_sweep(base_spec, request)
        print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if state.status != "failed" else 1)
    except (ValidationError, ValueError, FileExistsError, FileNotFoundError) as exc:
        print(
            json.dumps(
                {
                    "tool_name": "parameter_sweep",
                    "status": "failed",
                    "failure_reason": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
