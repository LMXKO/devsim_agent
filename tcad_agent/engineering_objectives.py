from __future__ import annotations

import json
import math
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.reporting import final_metrics, load_final_state, resolve_state_path


class ObjectiveDirection(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    MINIMIZE_ABS = "minimize_abs"
    MAXIMIZE_ABS = "maximize_abs"


class ConstraintOperator(str, Enum):
    LE = "<="
    LT = "<"
    GE = ">="
    GT = ">"
    EQ = "=="


class EngineeringObjective(BaseModel):
    metric_path: str
    direction: ObjectiveDirection
    weight: float = 1.0


class EngineeringConstraint(BaseModel):
    metric_path: str
    operator: ConstraintOperator
    value: float
    required: bool = True


class ObjectiveCandidate(BaseModel):
    candidate_id: str
    source_state_path: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    feasible: bool = True
    violations: list[dict[str, Any]] = Field(default_factory=list)
    objective_values: dict[str, float | None] = Field(default_factory=dict)
    score: float | None = None
    pareto_front: bool = False


class EngineeringObjectiveResult(BaseModel):
    tool_name: str = "engineering_objective_evaluation"
    status: str
    source_state_path: str
    output_path: str | None = None
    objectives: list[EngineeringObjective]
    constraints: list[EngineeringConstraint] = Field(default_factory=list)
    candidates: list[ObjectiveCandidate] = Field(default_factory=list)
    best_candidate: ObjectiveCandidate | None = None
    pareto_front: list[ObjectiveCandidate] = Field(default_factory=list)
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def nested_get(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def item_metrics(item: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    metrics = dict(item.get("metrics") or {})
    final_path = item.get("final_state_path")
    if final_path:
        path = Path(final_path)
        if not path.is_absolute():
            path = base_dir / path
        metrics.update(final_metrics(load_final_state(str(path))))
    return metrics


def extract_candidates(state: dict[str, Any], state_path: Path) -> list[ObjectiveCandidate]:
    items = state.get("observations") or state.get("cases") or []
    if items:
        candidates = []
        for index, item in enumerate(items, start=1):
            candidate_id = str(item.get("task_id") or item.get("run_id") or f"candidate_{index:03d}")
            candidates.append(
                ObjectiveCandidate(
                    candidate_id=candidate_id,
                    source_state_path=item.get("final_state_path"),
                    metrics=item_metrics(item, state_path.parent),
                )
            )
        return candidates
    metrics = (state.get("quality_report") or {}).get("metrics") or {}
    if metrics:
        return [
            ObjectiveCandidate(
                candidate_id=str(state.get("run_id") or state.get("task_id") or state_path.parent.name),
                source_state_path=str(state_path),
                metrics=metrics,
            )
        ]
    return []


def default_objectives_for_candidates(candidates: list[ObjectiveCandidate]) -> list[EngineeringObjective]:
    keys = {key for candidate in candidates for key in candidate.metrics}
    objectives: list[EngineeringObjective] = []
    if "ion_ioff_ratio" in keys:
        objectives.append(EngineeringObjective(metric_path="ion_ioff_ratio", direction=ObjectiveDirection.MAXIMIZE))
    if "leakage_abs_current_at_target_a" in keys:
        objectives.append(EngineeringObjective(metric_path="leakage_abs_current_at_target_a", direction=ObjectiveDirection.MINIMIZE))
    if "breakdown_voltage_v" in keys:
        objectives.append(EngineeringObjective(metric_path="breakdown_voltage_v", direction=ObjectiveDirection.MAXIMIZE_ABS))
    if "specific_on_resistance_ohm_cm2" in keys:
        objectives.append(EngineeringObjective(metric_path="specific_on_resistance_ohm_cm2", direction=ObjectiveDirection.MINIMIZE))
    if "responsivity_a_per_w" in keys:
        objectives.append(EngineeringObjective(metric_path="responsivity_a_per_w", direction=ObjectiveDirection.MAXIMIZE))
    return objectives or [EngineeringObjective(metric_path="objective_value", direction=ObjectiveDirection.MINIMIZE)]


def compare_constraint(value: float, constraint: EngineeringConstraint) -> bool:
    if constraint.operator == ConstraintOperator.LE:
        return value <= constraint.value
    if constraint.operator == ConstraintOperator.LT:
        return value < constraint.value
    if constraint.operator == ConstraintOperator.GE:
        return value >= constraint.value
    if constraint.operator == ConstraintOperator.GT:
        return value > constraint.value
    if constraint.operator == ConstraintOperator.EQ:
        return math.isclose(value, constraint.value, rel_tol=1e-9, abs_tol=1e-300)
    return False


def transformed_objective_value(value: float, direction: ObjectiveDirection) -> float:
    if direction == ObjectiveDirection.MINIMIZE:
        return value
    if direction == ObjectiveDirection.MAXIMIZE:
        return -value
    if direction == ObjectiveDirection.MINIMIZE_ABS:
        return abs(value)
    if direction == ObjectiveDirection.MAXIMIZE_ABS:
        return -abs(value)
    return value


def evaluate_candidate(
    candidate: ObjectiveCandidate,
    objectives: list[EngineeringObjective],
    constraints: list[EngineeringConstraint],
) -> ObjectiveCandidate:
    updated = candidate.model_copy(deep=True)
    for constraint in constraints:
        value = float_or_none(nested_get(updated.metrics, constraint.metric_path))
        if value is None or not compare_constraint(value, constraint):
            updated.violations.append(
                {
                    "metric_path": constraint.metric_path,
                    "operator": constraint.operator.value,
                    "expected": constraint.value,
                    "observed": value,
                    "required": constraint.required,
                }
            )
    updated.feasible = not any(item.get("required") for item in updated.violations)

    score = 0.0
    missing_objective = False
    for objective in objectives:
        raw = float_or_none(nested_get(updated.metrics, objective.metric_path))
        updated.objective_values[objective.metric_path] = raw
        if raw is None:
            missing_objective = True
            continue
        score += objective.weight * transformed_objective_value(raw, objective.direction)
    updated.score = None if missing_objective else score
    return updated


def dominates(left: ObjectiveCandidate, right: ObjectiveCandidate, objectives: list[EngineeringObjective]) -> bool:
    if not left.feasible or not right.feasible:
        return False
    left_values = []
    right_values = []
    for objective in objectives:
        left_raw = left.objective_values.get(objective.metric_path)
        right_raw = right.objective_values.get(objective.metric_path)
        if left_raw is None or right_raw is None:
            return False
        left_values.append(transformed_objective_value(left_raw, objective.direction))
        right_values.append(transformed_objective_value(right_raw, objective.direction))
    return all(l <= r for l, r in zip(left_values, right_values)) and any(l < r for l, r in zip(left_values, right_values))


def assign_pareto_front(candidates: list[ObjectiveCandidate], objectives: list[EngineeringObjective]) -> list[ObjectiveCandidate]:
    updated = [candidate.model_copy(deep=True) for candidate in candidates]
    for candidate in updated:
        candidate.pareto_front = candidate.feasible and not any(
            dominates(other, candidate, objectives) for other in updated if other.candidate_id != candidate.candidate_id
        )
    return updated


def evaluate_engineering_objectives(
    source: Path,
    *,
    objectives: list[EngineeringObjective] | None = None,
    constraints: list[EngineeringConstraint] | None = None,
    output_path: Path | None = None,
) -> EngineeringObjectiveResult:
    try:
        state_path = resolve_state_path(source).resolve()
        state = read_json(state_path)
        raw_candidates = extract_candidates(state, state_path)
        actual_objectives = objectives or default_objectives_for_candidates(raw_candidates)
        actual_constraints = constraints or []
        evaluated = [
            evaluate_candidate(candidate, actual_objectives, actual_constraints)
            for candidate in raw_candidates
        ]
        evaluated = assign_pareto_front(evaluated, actual_objectives)
        feasible_scored = [candidate for candidate in evaluated if candidate.feasible and candidate.score is not None]
        best = sorted(feasible_scored, key=lambda item: float(item.score))[0] if feasible_scored else None
        result = EngineeringObjectiveResult(
            status="completed",
            source_state_path=str(state_path),
            objectives=actual_objectives,
            constraints=actual_constraints,
            candidates=evaluated,
            best_candidate=best,
            pareto_front=[candidate for candidate in evaluated if candidate.pareto_front],
        )
        target = (output_path or state_path.with_name("engineering_objectives.json")).resolve()
        result.output_path = str(target)
        write_json(target, result.model_dump(mode="json"))
        return result
    except Exception as exc:
        return EngineeringObjectiveResult(
            status="failed",
            source_state_path=str(source),
            objectives=objectives or [],
            constraints=constraints or [],
            failure_reason=str(exc),
        )
