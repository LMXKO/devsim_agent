from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.agent_cockpit import generate_agent_cockpit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a minimal TCAD agent cockpit HTML page.")
    parser.add_argument("--source", "--state", dest="source", type=Path, required=True)
    parser.add_argument("--output", "--output-path", dest="output_path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        result = generate_agent_cockpit(args.source, args.output_path)
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"tool_name": "agent_cockpit", "status": "failed", "failure_reason": str(exc)}, indent=2, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()

