from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.dashboard import DashboardStatus, generate_experiment_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a static HTML dashboard for a TCAD sweep or optimization.")
    parser.add_argument(
        "--state",
        type=Path,
        required=True,
        help="Path to sweep_state.json, optimization_state.json, or the containing run directory.",
    )
    parser.add_argument("--output", type=Path, default=None, help="HTML dashboard path. Defaults to dashboard.html.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_experiment_dashboard(args.state, args.output)
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.status == DashboardStatus.COMPLETED else 2)


if __name__ == "__main__":
    main()
