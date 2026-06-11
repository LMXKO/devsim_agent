from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.agent_goal_router import AgentGoalRouteRequest, route_agent_goal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route a top-level natural-language TCAD agent goal into an executable mission plan.")
    parser.add_argument("--goal", "--goal-text", dest="goal_text", required=True)
    parser.add_argument("--simulator", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--source-deck-path", default=None)
    parser.add_argument("--sentaurus-project-path", type=Path, default=None)
    parser.add_argument("--sentaurus-profile-path", type=Path, default=None)
    parser.add_argument("--reference-curve-path", type=Path, default=None)
    parser.add_argument("--output", "--output-path", dest="output_path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        result = route_agent_goal(
            AgentGoalRouteRequest(
                goal_text=args.goal_text,
                simulator=args.simulator,
                execute=args.execute,
                max_steps=args.max_steps,
                source_deck_path=args.source_deck_path,
                sentaurus_project_path=args.sentaurus_project_path,
                sentaurus_profile_path=args.sentaurus_profile_path,
                reference_curve_path=str(args.reference_curve_path) if args.reference_curve_path else None,
            ),
            output_path=args.output_path,
        )
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if result.status in {"matched", "needs_input"} else 1)
    except Exception as exc:
        print(json.dumps({"tool_name": "agent_goal_router", "status": "failed", "failure_reason": str(exc)}, indent=2, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()

