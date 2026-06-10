from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.sentaurus_patch_refiner import SentaurusPatchRefinerRequest, build_sentaurus_patch_refinement_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine the next Sentaurus semantic patch from baseline-vs-mutation effect evidence.")
    parser.add_argument("--state", "--source-state-path", dest="source_state_path", type=Path, required=True)
    parser.add_argument("--goal", "--goal-text", dest="goal_text", default="")
    parser.add_argument("--output", "--output-path", dest="output_path", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--allow-high-risk", action="store_true")
    return parser.parse_args()


def request_from_args(args: argparse.Namespace) -> SentaurusPatchRefinerRequest:
    return SentaurusPatchRefinerRequest(
        source_state_path=args.source_state_path,
        goal_text=args.goal_text,
        output_path=args.output_path,
        max_candidates=args.max_candidates,
        allow_high_risk=args.allow_high_risk,
    )


def main() -> None:
    try:
        plan = build_sentaurus_patch_refinement_plan(request_from_args(parse_args()))
        print(json.dumps(plan.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if plan.status in {"completed", "blocked_for_user_confirmation", "blocked_for_pareto_review", "no_actionable_candidates"} else 1)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "tool_name": "sentaurus_patch_refiner",
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
