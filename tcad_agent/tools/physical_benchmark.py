from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.physical_benchmark import BenchmarkStatus, run_physical_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run physical benchmark/golden checks for a TCAD state.")
    parser.add_argument("--state", type=Path, required=True, help="Path to a state file or containing run directory.")
    parser.add_argument("--output", type=Path, default=None, help="Benchmark JSON path. Defaults to benchmark.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_physical_benchmark(args.state, args.output)
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.status in {BenchmarkStatus.PASSED, BenchmarkStatus.SUSPICIOUS} else 2)


if __name__ == "__main__":
    main()
