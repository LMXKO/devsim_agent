from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.task_planner import (
    PlannerStatus,
    plan_task_text_with_llm,
    task_spec_from_planning_result,
    write_planning_result,
)
from tcad_agent.task_spec import PROJECT_ROOT, write_task_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a TCAD TaskSpec from natural-language text.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--task-output", type=Path, default=None)
    parser.add_argument("--no-fallback", action="store_true")
    loop_llm = parser.add_mutually_exclusive_group()
    loop_llm.add_argument("--loop-use-llm", dest="execution_use_llm", action="store_true")
    loop_llm.add_argument("--loop-no-llm", dest="execution_use_llm", action="store_false")
    parser.set_defaults(execution_use_llm=None)
    return parser.parse_args()


def default_output_path(task_id: str | None) -> Path:
    stem = task_id or "planned_task"
    return PROJECT_ROOT / "runs" / "task_plans" / stem / "task_plan_result.json"


def default_task_output_path(task_id: str | None) -> Path:
    stem = task_id or "planned_task"
    return PROJECT_ROOT / "runs" / "task_plans" / stem / "task.json"


def main() -> None:
    args = parse_args()
    result = plan_task_text_with_llm(
        args.text,
        task_id=args.task_id,
        execution_use_llm=args.execution_use_llm,
        allow_fallback=not args.no_fallback,
    )
    output = args.output or default_output_path(result.task_id or args.task_id)
    write_planning_result(result, output)

    if result.task_spec:
        task_output = args.task_output or default_task_output_path(result.task_id or args.task_id)
        write_task_spec(task_spec_from_planning_result(result), task_output)

    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.status != PlannerStatus.FAILED else 1)


if __name__ == "__main__":
    main()
