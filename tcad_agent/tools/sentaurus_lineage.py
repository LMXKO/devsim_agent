from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.sentaurus_lineage import SentaurusLineageArchiveRequest, build_sentaurus_lineage_archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact multi-run Sentaurus patch lineage and Pareto archive.")
    parser.add_argument("--state", "--source-state-path", dest="source_state_path", type=Path, required=True)
    parser.add_argument("--output", "--output-path", dest="output_path", type=Path, default=None)
    parser.add_argument("--max-depth", type=int, default=24)
    return parser.parse_args()


def request_from_args(args: argparse.Namespace) -> SentaurusLineageArchiveRequest:
    return SentaurusLineageArchiveRequest(
        source_state_path=args.source_state_path,
        output_path=args.output_path,
        max_depth=args.max_depth,
    )


def main() -> None:
    try:
        archive = build_sentaurus_lineage_archive(request_from_args(parse_args()))
        print(json.dumps(archive.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if archive.status == "completed" else 1)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "tool_name": "sentaurus_lineage_archive",
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
