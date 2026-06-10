from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.sentaurus_mutation_effect import SentaurusMutationEffectRequest, analyze_sentaurus_mutation_effect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline vs patched Sentaurus states and decide whether the patch direction is worth continuing.")
    parser.add_argument("--baseline", "--baseline-state-path", dest="baseline_state_path", type=Path, required=True)
    parser.add_argument("--mutation", "--mutation-state-path", dest="mutation_state_path", type=Path, required=True)
    parser.add_argument("--candidate-json", default=None, help="JSON object for the Sentaurus patch candidate that produced the mutation run.")
    parser.add_argument("--goal", "--goal-text", dest="goal_text", default="")
    parser.add_argument("--output", "--output-path", dest="output_path", type=Path, default=None)
    parser.add_argument("--overlay-output-path", type=Path, default=None)
    return parser.parse_args()


def request_from_args(args: argparse.Namespace) -> SentaurusMutationEffectRequest:
    candidate = json.loads(args.candidate_json) if args.candidate_json else {}
    if not isinstance(candidate, dict):
        raise ValueError("--candidate-json must decode to a JSON object")
    return SentaurusMutationEffectRequest(
        baseline_state_path=args.baseline_state_path,
        mutation_state_path=args.mutation_state_path,
        candidate=candidate,
        goal_text=args.goal_text,
        output_path=args.output_path,
        overlay_output_path=args.overlay_output_path,
    )


def main() -> None:
    try:
        result = analyze_sentaurus_mutation_effect(request_from_args(parse_args()))
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if result.status == "completed" else 1)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "tool_name": "sentaurus_mutation_effect_analyzer",
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
