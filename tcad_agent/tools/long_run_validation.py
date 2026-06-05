from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.long_run_validation import (
    LongRunValidationRequest,
    LongRunValidationStatus,
    run_long_run_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an unattended long-run TCAD agent validation.")
    parser.add_argument("--validation-id", default=None)
    parser.add_argument("--validation-root", type=Path, default=None)
    parser.add_argument("--queue-goals-json", default=None)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.0)
    parser.add_argument("--max-idle-loops", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = {
        "validation_id": args.validation_id,
        "poll_interval_seconds": args.poll_interval_seconds,
        "max_idle_loops": args.max_idle_loops,
    }
    if args.validation_root is not None:
        data["validation_root"] = args.validation_root
    if args.queue_goals_json:
        goals = json.loads(args.queue_goals_json)
        if not isinstance(goals, list):
            raise ValueError("--queue-goals-json must decode to a list")
        data["queue_goals"] = goals
    state = run_long_run_validation(LongRunValidationRequest.model_validate(data))
    print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if state.status == LongRunValidationStatus.COMPLETED else 2)


if __name__ == "__main__":
    main()
