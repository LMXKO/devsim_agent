from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.golden_curve import GoldenCurveComparisonRequest, run_golden_curve_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a TCAD curve against a golden/measured reference curve.")
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--reference-curve", required=True, type=Path)
    parser.add_argument("--comparison-id", default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--source-x-column", default=None)
    parser.add_argument("--source-y-column", default=None)
    parser.add_argument("--reference-x-column", default=None)
    parser.add_argument("--reference-y-column", default=None)
    parser.add_argument("--max-pass-rmse-log-dec", type=float, default=0.2)
    parser.add_argument("--max-warn-rmse-log-dec", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = {
        "comparison_id": args.comparison_id,
        "source_state_path": args.source_state,
        "reference_curve_path": args.reference_curve,
        "source_x_column": args.source_x_column,
        "source_y_column": args.source_y_column,
        "reference_x_column": args.reference_x_column,
        "reference_y_column": args.reference_y_column,
        "max_pass_rmse_log_dec": args.max_pass_rmse_log_dec,
        "max_warn_rmse_log_dec": args.max_warn_rmse_log_dec,
    }
    if args.run_root is not None:
        data["run_root"] = args.run_root
    state = run_golden_curve_comparison(GoldenCurveComparisonRequest.model_validate(data))
    print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if state.status == "completed" else 2)


if __name__ == "__main__":
    main()
