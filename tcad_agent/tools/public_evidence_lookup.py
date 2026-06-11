from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.evidence_lookup import PublicEvidenceLookupRequest, run_public_evidence_lookup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch or summarize public TCAD evidence sources for an agent task.")
    parser.add_argument("--goal", "--goal-text", dest="goal_text", required=True)
    parser.add_argument("--simulator", default=None)
    parser.add_argument("--template-id", action="append", default=[])
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--max-sources", type=int, default=6)
    parser.add_argument("--live", action="store_true", help="Fetch public source URLs instead of registry-only summaries.")
    parser.add_argument("--timeout-seconds", type=float, default=8.0)
    parser.add_argument("--output", "--output-path", dest="output_path", type=Path, default=None)
    return parser.parse_args()


def request_from_args(args: argparse.Namespace) -> PublicEvidenceLookupRequest:
    return PublicEvidenceLookupRequest(
        goal_text=args.goal_text,
        simulator=args.simulator,
        template_ids=args.template_id,
        source_ids=args.source_id,
        max_sources=args.max_sources,
        live=args.live,
        timeout_seconds=args.timeout_seconds,
        output_path=args.output_path,
    )


def main() -> None:
    try:
        result = run_public_evidence_lookup(request_from_args(parse_args()))
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if result.status != "failed" else 1)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "tool_name": "public_evidence_lookup",
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
