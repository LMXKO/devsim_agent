from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.industrial_runner_promotion import build_industrial_runner_promotion_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a runner-promotion work package for industrial TCAD device routes.")
    parser.add_argument("--goal", "--goal-text", dest="goal_text", required=True)
    parser.add_argument("--template-id", default=None)
    parser.add_argument("--simulator", default=None)
    parser.add_argument("--output", "--output-path", dest="output_path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        plan = build_industrial_runner_promotion_plan(
            args.goal_text,
            template_id=args.template_id,
            simulator=args.simulator,
            output_path=args.output_path,
        )
        print(json.dumps(plan.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if plan.status == "completed" else 1)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "tool_name": "industrial_runner_promotion",
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
