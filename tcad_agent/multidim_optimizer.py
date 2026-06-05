from __future__ import annotations

import csv
import itertools
import json
import math
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from tcad_agent.parameter_sweep import (
    ParameterSweepRequest,
    ParameterSweepState,
    SweepAxis,
    SweepDirection,
    SweepObjective,
    run_parameter_sweep,
)
from tcad_agent.task_spec import PROJECT_ROOT, TaskSpec, load_task_spec, write_task_spec
from tcad_agent.tools.task_runner import run_task


class MultiDimOptimizationStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AxisScale(str, Enum):
    LINEAR = "linear"
    LOG = "log"


class MultiDimAxis(BaseModel):
    path: str
    min_value: float
    max_value: float
    scale: AxisScale = AxisScale.LOG
    initial_points: int = Field(default=2, ge=2)
    max_new_points_per_round: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def validate_axis(self) -> "MultiDimAxis":
        if not self.path:
            raise ValueError("axis path is required")
        if self.path.split(".")[0] not in {"sweep", "parameters", "mesh", "quality", "execution"}:
            raise ValueError(f"unsupported optimization path: {self.path}")
        if self.max_value <= self.min_value:
            raise ValueError("max_value must be greater than min_value")
        if self.scale == AxisScale.LOG and self.min_value <= 0:
            raise ValueError("log-scale optimization requires min_value > 0")
        return self


class MultiDimOptimizationRequest(BaseModel):
    optimize_id: str | None = None
    axes: list[MultiDimAxis]
    objective: SweepObjective = Field(default_factory=SweepObjective)
    execute: bool = False
    overwrite: bool = False
    use_llm: bool | None = None
    optimize_root: Path = PROJECT_ROOT / "runs" / "optimizations"
    max_rounds: int = Field(default=3, ge=1)
    max_cases: int = Field(default=80, ge=1)
    max_cases_per_round: int = Field(default=12, ge=1)

    @model_validator(mode="after")
    def validate_request(self) -> "MultiDimOptimizationRequest":
        if len(self.axes) < 2:
            raise ValueError("multi-dimensional optimization requires at least two axes")
        paths = [axis.path for axis in self.axes]
        if len(paths) != len(set(paths)):
            raise ValueError("optimization axis paths must be unique")
        initial_cases = 1
        for axis in self.axes:
            initial_cases *= axis.initial_points
        if initial_cases > self.max_cases:
            raise ValueError(
                f"initial grid would create {initial_cases} cases, exceeding max_cases={self.max_cases}"
            )
        if initial_cases > self.max_cases_per_round:
            raise ValueError(
                "initial grid would create "
                f"{initial_cases} cases, exceeding max_cases_per_round={self.max_cases_per_round}"
            )
        return self


class MultiDimObservation(BaseModel):
    round_index: int
    point_index: int
    sweep_id: str
    case_index: int
    task_id: str
    values: dict[str, float]
    status: str | None = None
    quality_status: str | None = None
    objective_value: float | None = None
    task_run_state_path: str | None = None
    final_state_path: str | None = None
    error: str | None = None


class MultiDimOptimizationRound(BaseModel):
    index: int
    round_id: str
    candidate_values: list[dict[str, float]]
    status: str
    sweep_state_paths: list[str] = Field(default_factory=list)
    summary_csv_path: str | None = None
    started_at: str
    finished_at: str | None = None


class MultiDimOptimizationState(BaseModel):
    tool_name: str = "multidim_optimizer"
    status: MultiDimOptimizationStatus
    optimize_id: str
    optimize_dir: str
    base_task_path: str
    created_at: str
    updated_at: str
    execute: bool
    axes: list[dict[str, Any]]
    objective: dict[str, Any]
    max_rounds: int
    max_cases: int
    max_cases_per_round: int
    rounds: list[MultiDimOptimizationRound] = Field(default_factory=list)
    observations: list[MultiDimObservation] = Field(default_factory=list)
    best_observation: dict[str, Any] | None = None
    next_action: str | None = None
    failure_reason: str | None = None


TaskRunner = Any


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_optimize_id() -> str:
    return f"multiopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_optimization_state(state: MultiDimOptimizationState, path: Path) -> None:
    state.updated_at = utc_timestamp()
    write_json(path, state.model_dump(mode="json"))


def load_optimization_state(path: Path) -> MultiDimOptimizationState:
    return MultiDimOptimizationState.model_validate_json(path.read_text(encoding="utf-8"))


def create_initial_state(
    request: MultiDimOptimizationRequest,
    base_task_path: Path,
    optimize_id: str,
    optimize_dir: Path,
) -> MultiDimOptimizationState:
    now = utc_timestamp()
    return MultiDimOptimizationState(
        status=MultiDimOptimizationStatus.RUNNING if request.execute else MultiDimOptimizationStatus.PLANNED,
        optimize_id=optimize_id,
        optimize_dir=str(optimize_dir),
        base_task_path=str(base_task_path),
        created_at=now,
        updated_at=now,
        execute=request.execute,
        axes=[axis.model_dump(mode="json") for axis in request.axes],
        objective=request.objective.model_dump(mode="json"),
        max_rounds=request.max_rounds,
        max_cases=request.max_cases,
        max_cases_per_round=request.max_cases_per_round,
    )


def almost_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-300)


def dedupe_values(values: list[float]) -> list[float]:
    unique: list[float] = []
    for value in values:
        if not any(almost_equal(value, existing) for existing in unique):
            unique.append(float(value))
    return unique


def value_seen(value: float, seen: list[float]) -> bool:
    return any(almost_equal(value, item) for item in seen)


def initial_axis_values(axis: MultiDimAxis) -> list[float]:
    if axis.scale == AxisScale.LOG:
        ratio = (axis.max_value / axis.min_value) ** (1.0 / (axis.initial_points - 1))
        return [float(axis.min_value * (ratio**index)) for index in range(axis.initial_points)]

    step = (axis.max_value - axis.min_value) / (axis.initial_points - 1)
    return [float(axis.min_value + index * step) for index in range(axis.initial_points)]


def midpoint(left: float, right: float, scale: AxisScale) -> float:
    if scale == AxisScale.LOG:
        return float(math.sqrt(left * right))
    return float((left + right) / 2.0)


def gap_size(left: float, right: float, scale: AxisScale) -> float:
    if scale == AxisScale.LOG:
        return math.log(right / left)
    return right - left


def completed_observations(state: MultiDimOptimizationState) -> list[MultiDimObservation]:
    return [
        observation
        for observation in state.observations
        if observation.status == "completed" and observation.objective_value is not None
    ]


def choose_best_observation(
    observations: list[MultiDimObservation],
    objective: SweepObjective,
) -> MultiDimObservation | None:
    eligible = [
        observation
        for observation in observations
        if observation.status == "completed" and observation.objective_value is not None
    ]
    if not eligible:
        return None
    reverse = objective.direction == SweepDirection.MAXIMIZE
    return sorted(eligible, key=lambda item: item.objective_value or 0.0, reverse=reverse)[0]


def candidate_seen(
    candidate: dict[str, float],
    seen: list[dict[str, float]],
    axes: list[MultiDimAxis],
) -> bool:
    for item in seen:
        if all(almost_equal(candidate[axis.path], float(item[axis.path])) for axis in axes if axis.path in item):
            return True
    return False


def dedupe_candidates(
    candidates: list[dict[str, float]],
    axes: list[MultiDimAxis],
) -> list[dict[str, float]]:
    unique: list[dict[str, float]] = []
    for candidate in candidates:
        if not candidate_seen(candidate, unique, axes):
            unique.append({axis.path: float(candidate[axis.path]) for axis in axes})
    return unique


def seen_candidates(state: MultiDimOptimizationState, axes: list[MultiDimAxis]) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    for observation in state.observations:
        if observation.status == "planned":
            continue
        values = observation.values
        if all(axis.path in values for axis in axes):
            candidates.append({axis.path: float(values[axis.path]) for axis in axes})
    return dedupe_candidates(candidates, axes)


def seen_axis_values(
    state: MultiDimOptimizationState,
    axis: MultiDimAxis,
) -> list[float]:
    return sorted(
        dedupe_values(
            [
                float(observation.values[axis.path])
                for observation in state.observations
                if observation.status != "planned" and axis.path in observation.values
            ]
        )
    )


def initial_candidates(axes: list[MultiDimAxis]) -> list[dict[str, float]]:
    products = itertools.product(*(initial_axis_values(axis) for axis in axes))
    return [{axis.path: float(value) for axis, value in zip(axes, values)} for values in products]


def propose_axis_refinement_values(
    axis: MultiDimAxis,
    state: MultiDimOptimizationState,
    best: MultiDimObservation,
) -> list[float]:
    seen = seen_axis_values(state, axis)
    if len(seen) < 2 or axis.path not in best.values:
        return []

    best_value = float(best.values[axis.path])
    proposals: list[float] = []
    best_index = min(range(len(seen)), key=lambda index: abs(seen[index] - best_value))
    for neighbor_index in [best_index - 1, best_index + 1]:
        if 0 <= neighbor_index < len(seen):
            candidate = midpoint(best_value, seen[neighbor_index], axis.scale)
            if axis.min_value <= candidate <= axis.max_value and not value_seen(candidate, seen):
                proposals.append(candidate)

    if len(proposals) < axis.max_new_points_per_round:
        gaps = [
            (gap_size(left, right, axis.scale), left, right)
            for left, right in zip(seen[:-1], seen[1:])
            if right > left
        ]
        for _, left, right in sorted(gaps, reverse=True):
            candidate = midpoint(left, right, axis.scale)
            if not value_seen(candidate, seen) and not value_seen(candidate, proposals):
                proposals.append(candidate)
            if len(proposals) >= axis.max_new_points_per_round:
                break

    return dedupe_values(proposals)[: axis.max_new_points_per_round]


def candidate_distance_score(
    candidate: dict[str, float],
    best_values: dict[str, float],
    axes: list[MultiDimAxis],
) -> float:
    score = 0.0
    for axis in axes:
        left = float(candidate[axis.path])
        right = float(best_values[axis.path])
        if axis.scale == AxisScale.LOG and left > 0 and right > 0:
            span = math.log(axis.max_value / axis.min_value)
            delta = abs(math.log(left / right))
        else:
            span = axis.max_value - axis.min_value
            delta = abs(left - right)
        score += delta / (span or 1.0)
    return score


def propose_refinement_candidates(
    axes: list[MultiDimAxis],
    state: MultiDimOptimizationState,
    objective: SweepObjective,
    max_candidates: int,
) -> list[dict[str, float]]:
    seen = seen_candidates(state, axes)
    if not state.observations:
        return initial_candidates(axes)[:max_candidates]

    best = choose_best_observation(completed_observations(state), objective)
    if best is None:
        return []

    best_values = {axis.path: float(best.values[axis.path]) for axis in axes}
    per_axis_proposals = {
        axis.path: propose_axis_refinement_values(axis, state, best)
        for axis in axes
    }
    if not any(per_axis_proposals.values()):
        return []

    candidates: list[dict[str, float]] = []

    # Local response-surface refinement: evaluate all-new neighborhood points.
    product_sources = [
        per_axis_proposals[axis.path] or [best_values[axis.path]]
        for axis in axes
    ]
    for values in itertools.product(*product_sources):
        candidates.append({axis.path: float(value) for axis, value in zip(axes, values)})

    # Coordinate refinements: hold other axes at the best point and move one axis.
    for axis in axes:
        for value in per_axis_proposals[axis.path]:
            candidate = dict(best_values)
            candidate[axis.path] = float(value)
            candidates.append(candidate)

    candidates = [
        candidate
        for candidate in dedupe_candidates(candidates, axes)
        if not candidate_seen(candidate, seen, axes)
    ]
    candidates.sort(key=lambda item: candidate_distance_score(item, best_values, axes))
    return candidates[:max_candidates]


def observations_from_sweep(
    sweep_state: ParameterSweepState,
    axes: list[MultiDimAxis],
    round_index: int,
    point_index: int,
) -> list[MultiDimObservation]:
    observations: list[MultiDimObservation] = []
    for case in sweep_state.cases:
        if not all(axis.path in case.values for axis in axes):
            continue
        observations.append(
            MultiDimObservation(
                round_index=round_index,
                point_index=point_index,
                sweep_id=sweep_state.sweep_id,
                case_index=case.index,
                task_id=case.task_id,
                values={axis.path: float(case.values[axis.path]) for axis in axes},
                status=case.status,
                quality_status=case.quality_status,
                objective_value=case.objective_value,
                task_run_state_path=case.task_run_state_path,
                final_state_path=case.final_state_path,
                error=case.error,
            )
        )
    return observations


def write_round_summary_csv(
    path: Path,
    observations: list[MultiDimObservation],
    axes: list[MultiDimAxis],
) -> None:
    fieldnames = [
        "round_index",
        "point_index",
        "sweep_id",
        "case_index",
        "task_id",
        *[axis.path for axis in axes],
        "status",
        "quality_status",
        "objective_value",
        "final_state_path",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for observation in observations:
            row = {
                "round_index": observation.round_index,
                "point_index": observation.point_index,
                "sweep_id": observation.sweep_id,
                "case_index": observation.case_index,
                "task_id": observation.task_id,
                "status": observation.status,
                "quality_status": observation.quality_status,
                "objective_value": observation.objective_value,
                "final_state_path": observation.final_state_path,
                "error": observation.error,
            }
            row.update(observation.values)
            writer.writerow(row)


def run_optimization_round(
    base_spec: TaskSpec,
    request: MultiDimOptimizationRequest,
    optimize_id: str,
    optimize_dir: Path,
    round_index: int,
    candidates: list[dict[str, float]],
    task_runner: TaskRunner,
) -> tuple[MultiDimOptimizationRound, list[MultiDimObservation]]:
    round_id = f"{optimize_id}_round_{round_index:03d}"
    started_at = utc_timestamp()
    round_observations: list[MultiDimObservation] = []
    sweep_paths: list[str] = []
    statuses: list[str] = []
    for point_index, candidate in enumerate(candidates, start=1):
        sweep_id = f"{round_id}_point_{point_index:03d}"
        sweep_state = run_parameter_sweep(
            base_spec,
            ParameterSweepRequest(
                sweep_id=sweep_id,
                axes=[SweepAxis(path=axis.path, values=[candidate[axis.path]]) for axis in request.axes],
                objective=request.objective,
                execute=request.execute,
                overwrite=True,
                use_llm=request.use_llm,
                sweep_root=optimize_dir / "sweeps",
                max_cases=1,
            ),
            task_runner=task_runner,
        )
        sweep_paths.append(str(Path(sweep_state.sweep_dir) / "sweep_state.json"))
        statuses.append(str(getattr(sweep_state.status, "value", sweep_state.status)))
        round_observations.extend(
            observations_from_sweep(sweep_state, request.axes, round_index, point_index)
        )

    summary_csv = optimize_dir / "rounds" / round_id / "summary.csv"
    write_round_summary_csv(summary_csv, round_observations, request.axes)
    if not request.execute:
        status = "planned"
    elif any(status == "failed" for status in statuses):
        status = "failed"
    else:
        status = "completed"
    round_state = MultiDimOptimizationRound(
        index=round_index,
        round_id=round_id,
        candidate_values=candidates,
        status=status,
        sweep_state_paths=sweep_paths,
        summary_csv_path=str(summary_csv),
        started_at=started_at,
        finished_at=utc_timestamp(),
    )
    return round_state, round_observations


def run_multidim_optimization(
    base_spec: TaskSpec,
    request: MultiDimOptimizationRequest,
    task_runner: TaskRunner = run_task,
) -> MultiDimOptimizationState:
    optimize_id = request.optimize_id or default_optimize_id()
    optimize_dir = request.optimize_root / optimize_id
    state_path = optimize_dir / "optimization_state.json"
    base_task_path = optimize_dir / "base_task.json"

    if state_path.exists() and not request.overwrite:
        state = load_optimization_state(state_path)
        if base_task_path.exists():
            base_spec = load_task_spec(base_task_path)
        if request.execute:
            state.rounds = [round_state for round_state in state.rounds if round_state.status != "planned"]
            state.observations = [
                observation for observation in state.observations if observation.status != "planned"
            ]
        state.execute = request.execute
        state.axes = [axis.model_dump(mode="json") for axis in request.axes]
        state.objective = request.objective.model_dump(mode="json")
        state.max_rounds = request.max_rounds
        state.max_cases = request.max_cases
        state.max_cases_per_round = request.max_cases_per_round
        state.status = MultiDimOptimizationStatus.RUNNING if request.execute else MultiDimOptimizationStatus.PLANNED
        write_optimization_state(state, state_path)
    else:
        optimize_dir.mkdir(parents=True, exist_ok=True)
        write_task_spec(base_spec, base_task_path)
        state = create_initial_state(request, base_task_path, optimize_id, optimize_dir)
        write_optimization_state(state, state_path)

    while len(state.rounds) < request.max_rounds and len(state.observations) < request.max_cases:
        remaining_budget = request.max_cases - len(state.observations)
        candidates = propose_refinement_candidates(
            request.axes,
            state,
            request.objective,
            min(request.max_cases_per_round, remaining_budget),
        )
        if not candidates:
            state.status = MultiDimOptimizationStatus.COMPLETED
            state.next_action = "no new candidate points inside optimization bounds"
            write_optimization_state(state, state_path)
            return state

        round_index = len(state.rounds) + 1
        round_state, observations = run_optimization_round(
            base_spec,
            request,
            optimize_id,
            optimize_dir,
            round_index,
            candidates,
            task_runner,
        )
        state.rounds.append(round_state)
        state.observations.extend(observations)
        best = choose_best_observation(state.observations, request.objective)
        state.best_observation = best.model_dump(mode="json") if best else None
        write_optimization_state(state, state_path)

        if not request.execute:
            state.status = MultiDimOptimizationStatus.PLANNED
            state.next_action = "execute planned multi-dimensional optimization round"
            write_optimization_state(state, state_path)
            return state

        if not completed_observations(state):
            state.status = MultiDimOptimizationStatus.FAILED
            state.failure_reason = "no completed observations were produced"
            state.next_action = "inspect latest sweep and task checkpoints"
            write_optimization_state(state, state_path)
            return state

    if len(state.rounds) >= request.max_rounds:
        state.status = MultiDimOptimizationStatus.COMPLETED
        state.next_action = "maximum optimization rounds reached"
    elif len(state.observations) >= request.max_cases:
        state.status = MultiDimOptimizationStatus.COMPLETED
        state.next_action = "maximum optimization case budget reached"
    else:
        state.status = MultiDimOptimizationStatus.COMPLETED
        state.next_action = "optimization stopped"
    write_optimization_state(state, state_path)
    return state
