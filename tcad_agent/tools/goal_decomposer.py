from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.goal_decomposer import (
    DecompositionStatus,
    decompose_goal_with_llm,
    deterministic_decompose_goal,
    write_decomposition_result,
)
from tcad_agent.task_spec import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decompose a long-horizon TCAD goal into durable agent steps.")
    parser.add_argument("--goal", required=True)
    parser.add_argument("--plan-id", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    return parser.parse_args()


def default_output(plan_id: str | None) -> Path:
    actual = plan_id or "goal_plan"
    return PROJECT_ROOT / "runs" / "goal_plans" / actual / "goal_decomposition.json"


def main() -> None:
    args = parse_args()
    if args.use_llm:
        result = decompose_goal_with_llm(
            args.goal,
            plan_id=args.plan_id,
            allow_fallback=not args.no_fallback,
        )
    else:
        result = deterministic_decompose_goal(args.goal, plan_id=args.plan_id)
    output = args.output or default_output(result.plan_id or args.plan_id)
    write_decomposition_result(result, output)
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.status != DecompositionStatus.FAILED else 1)


if __name__ == "__main__":
    main()
