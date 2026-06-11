from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.sentaurus_patch_planner import SentaurusPatchPlannerRequest, plan_sentaurus_patches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan verified Sentaurus semantic patch candidates from a natural-language goal.")
    parser.add_argument("--goal", "--goal-text", dest="goal_text", required=True)
    parser.add_argument("--state", "--source-state-path", dest="source_state_path", type=Path, default=None)
    parser.add_argument("--project", "--project-path", dest="project_path", type=Path, default=None)
    parser.add_argument("--deck-file", action="append", default=[])
    parser.add_argument("--output", "--output-path", dest="output_path", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--allow-high-risk", action="store_true")
    parser.add_argument("--enable-live-lookup", action="store_true", help="Fetch registry public sources and include findings in the evidence dossier.")
    parser.add_argument("--live-lookup-max-sources", type=int, default=6)
    return parser.parse_args()


def request_from_args(args: argparse.Namespace) -> SentaurusPatchPlannerRequest:
    return SentaurusPatchPlannerRequest(
        goal_text=args.goal_text,
        source_state_path=args.source_state_path,
        project_path=args.project_path,
        deck_files=args.deck_file,
        output_path=args.output_path,
        max_candidates=args.max_candidates,
        allow_high_risk=args.allow_high_risk,
        enable_live_lookup=args.enable_live_lookup,
        live_lookup_max_sources=args.live_lookup_max_sources,
    )


def main() -> None:
    try:
        plan = plan_sentaurus_patches(request_from_args(parse_args()))
        print(json.dumps(plan.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if plan.status in {"completed", "blocked_for_user_confirmation", "no_actionable_candidates"} else 1)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "tool_name": "sentaurus_patch_planner",
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
