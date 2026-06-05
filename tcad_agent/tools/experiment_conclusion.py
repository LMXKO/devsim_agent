from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.conclusion import ConclusionStatus, generate_experiment_conclusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a conclusion-oriented TCAD experiment report.")
    parser.add_argument(
        "--state",
        type=Path,
        required=True,
        help="Path to a state file or containing run directory.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Markdown conclusion path. Defaults to conclusion.md.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_experiment_conclusion(args.state, args.output)
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.status == ConclusionStatus.COMPLETED else 2)


if __name__ == "__main__":
    main()
