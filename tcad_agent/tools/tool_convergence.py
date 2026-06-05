from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from tcad_agent.task_spec import PROJECT_ROOT
from tcad_agent.tool_convergence import ToolConvergenceRequest, ToolConvergenceStatus, run_tool_convergence


def parse_scalar(raw: str) -> object:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def parse_json_object(raw: str) -> dict[str, object]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("--base-request-json must be a JSON object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run convergence checks over agent-tool request fields.")
    parser.add_argument("--convergence-id", default=None)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--base-request-json", required=True)
    parser.add_argument("--axis-path", required=True)
    parser.add_argument("--value", action="append", required=True)
    parser.add_argument("--metric-path", default="quality_report.metrics.max_abs_current_a")
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--convergence-root", type=Path, default=PROJECT_ROOT / "runs" / "tool_convergence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        request = ToolConvergenceRequest(
            convergence_id=args.convergence_id,
            tool_name=args.tool,
            base_request=parse_json_object(args.base_request_json),
            axis_path=args.axis_path,
            values=[parse_scalar(value) for value in args.value],
            metric_path=args.metric_path,
            relative_tolerance=args.relative_tolerance,
            execute=args.execute,
            overwrite=args.overwrite,
            max_cases=args.max_cases,
            convergence_root=args.convergence_root,
        )
        state = run_tool_convergence(request)
        print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if state.status != ToolConvergenceStatus.FAILED else 1)
    except (ValidationError, ValueError, FileExistsError, FileNotFoundError) as exc:
        print(
            json.dumps(
                {
                    "tool_name": "tool_convergence",
                    "status": ToolConvergenceStatus.FAILED,
                    "failure_reason": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
