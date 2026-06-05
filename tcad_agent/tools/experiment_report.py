from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.reporting import ReportStatus, generate_experiment_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Markdown report for a TCAD sweep or optimization.")
    parser.add_argument(
        "--state",
        type=Path,
        required=True,
        help="Path to sweep_state.json, optimization_state.json, or the containing run directory.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Markdown report path. Defaults to report.md.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_experiment_report(args.state, args.output)
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.status == ReportStatus.COMPLETED else 2)


if __name__ == "__main__":
    main()
