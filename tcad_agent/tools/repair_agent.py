from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.repair_agent import RepairAgentStatus, decide_repair_action_with_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the LLM repair agent to choose the next TCAD repair action.")
    parser.add_argument("--state", type=Path, required=True, help="Source run state.json.")
    parser.add_argument("--no-fallback", action="store_true", help="Fail instead of returning the deterministic repair fallback.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = decide_repair_action_with_agent(args.state, allow_fallback=not args.no_fallback)
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.status != RepairAgentStatus.FAILED else 1)


if __name__ == "__main__":
    main()
