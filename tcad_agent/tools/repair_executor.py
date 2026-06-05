from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.repair_executor import RepairExecutionStatus, run_repair_executor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute TCAD repair plans in a closed loop.")
    parser.add_argument("--state", type=Path, required=True, help="Source run state.json.")
    parser.add_argument("--execution-id", default=None)
    parser.add_argument("--execution-root", type=Path, default=None)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-user-confirmation-actions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_repair_executor(
        args.state,
        execution_id=args.execution_id,
        execution_root=args.execution_root,
        execute=args.execute,
        resume=args.resume,
        max_rounds=args.max_rounds,
        allow_user_confirmation_actions=args.allow_user_confirmation_actions,
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.status != RepairExecutionStatus.FAILED else 1)


if __name__ == "__main__":
    main()
