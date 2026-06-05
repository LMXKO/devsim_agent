from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.repair_strategy import RepairPlanStatus, build_repair_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TCAD-specific repair strategy from a run state.")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_repair_plan(args.state, args.output)
    print(json.dumps(plan.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if plan.status != RepairPlanStatus.FAILED else 1)


if __name__ == "__main__":
    main()
