from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.engineering_objectives import (
    ConstraintOperator,
    EngineeringConstraint,
    EngineeringObjective,
    ObjectiveDirection,
    evaluate_engineering_objectives,
)


def parse_objective(raw: str) -> EngineeringObjective:
    parts = raw.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError("objective must be metric_path:direction[:weight]")
    return EngineeringObjective(
        metric_path=parts[0],
        direction=ObjectiveDirection(parts[1]),
        weight=float(parts[2]) if len(parts) == 3 else 1.0,
    )


def parse_constraint(raw: str) -> EngineeringConstraint:
    for operator in ["<=", ">=", "==", "<", ">"]:
        if operator in raw:
            left, right = raw.split(operator, 1)
            return EngineeringConstraint(
                metric_path=left.strip(),
                operator=ConstraintOperator(operator),
                value=float(right.strip()),
            )
    raise ValueError("constraint must be like metric<=value")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate engineering constraints and Pareto objectives for TCAD results.")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--objective", action="append", default=[])
    parser.add_argument("--constraint", action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_engineering_objectives(
        args.state,
        objectives=[parse_objective(item) for item in args.objective] or None,
        constraints=[parse_constraint(item) for item in args.constraint],
        output_path=args.output,
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.status == "completed" else 2)


if __name__ == "__main__":
    main()
