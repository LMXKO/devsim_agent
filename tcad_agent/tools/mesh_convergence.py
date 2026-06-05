from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from tcad_agent.mesh_convergence import (
    MeshConvergenceRequest,
    MeshConvergenceStatus,
    run_mesh_convergence,
)
from tcad_agent.parameter_sweep import SweepDirection, SweepObjective
from tcad_agent.task_spec import PROJECT_ROOT
from tcad_agent.tools.task_runner import resolve_task_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a checkpointed TCAD mesh convergence check.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Base natural-language TCAD task.")
    source.add_argument("--task", type=Path, help="Base task.json path.")
    parser.add_argument("--convergence-id", default=None)
    parser.add_argument("--task-id", default=None, help="Task id for the base task when using --text.")
    parser.add_argument("--axis-path", default="mesh.junction_spacing_um")
    parser.add_argument(
        "--value",
        action="append",
        type=float,
        help="Mesh value in um. Repeat for coarse-to-fine values.",
    )
    parser.add_argument(
        "--objective-metric",
        default="final_quality_report.metrics.final_total_current_a",
    )
    parser.add_argument("--direction", choices=["minimize", "maximize"], default="minimize")
    parser.add_argument("--raw-objective", action="store_true", help="Do not take absolute value of objective.")
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--convergence-root", type=Path, default=PROJECT_ROOT / "runs" / "mesh_convergence")
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
        request = MeshConvergenceRequest(
            convergence_id=args.convergence_id,
            axis_path=args.axis_path,
            values=args.value or [2e-5, 1e-5, 5e-6],
            objective=SweepObjective(
                metric_path=args.objective_metric,
                direction=SweepDirection(args.direction),
                absolute=not args.raw_objective,
            ),
            relative_tolerance=args.relative_tolerance,
            execute=args.execute,
            overwrite=args.overwrite,
            use_llm=args.use_llm,
            convergence_root=args.convergence_root,
            max_cases=args.max_cases,
        )
        state = run_mesh_convergence(base_spec, request)
        if planning_result:
            plan_path = Path(state.convergence_dir) / "base_task_plan_result.json"
            plan_path.write_text(
                json.dumps(planning_result.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if state.status != MeshConvergenceStatus.FAILED else 1)
    except (ValidationError, ValueError, FileExistsError, FileNotFoundError) as exc:
        print(
            json.dumps(
                {
                    "tool_name": "mesh_convergence",
                    "status": MeshConvergenceStatus.FAILED,
                    "failure_reason": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
