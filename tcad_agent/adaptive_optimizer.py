from __future__ import annotations

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
from tcad_agent.tools.task_runner import TaskRunState, run_task


class OptimizationStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AxisScale(str, Enum):
    LINEAR = "linear"
    LOG = "log"


class AdaptiveAxis(BaseModel):
    path: str
    min_value: float
    max_value: float
    scale: AxisScale = AxisScale.LOG
    initial_points: int = Field(default=3, ge=2)
    max_new_points_per_round: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def validate_axis(self) -> "AdaptiveAxis":
        if not self.path:
            raise ValueError("axis path is required")
        if self.path.split(".")[0] not in {"sweep", "parameters", "mesh", "quality", "execution"}:
            raise ValueError(f"unsupported optimization path: {self.path}")
        if self.max_value <= self.min_value:
            raise ValueError("max_value must be greater than min_value")
        if self.scale == AxisScale.LOG and self.min_value <= 0:
            raise ValueError("log-scale optimization requires min_value > 0")
        return self


class AdaptiveOptimizationRequest(BaseModel):
    optimize_id: str | None = None
    axis: AdaptiveAxis
    objective: SweepObjective = Field(default_factory=SweepObjective)
    execute: bool = False
    overwrite: bool = False
    use_llm: bool | None = None
    optimize_root: Path = PROJECT_ROOT / "runs" / "optimizations"
    max_rounds: int = Field(default=3, ge=1)
    max_cases: int = Field(default=50, ge=1)


class OptimizationObservation(BaseModel):
    round_index: int
    sweep_id: str
    case_index: int
    task_id: str
    value: float
    status: str | None = None
    quality_status: str | None = None
    objective_value: float | None = None
    task_run_state_path: str | None = None
    final_state_path: str | None = None
    error: str | None = None


class OptimizationRound(BaseModel):
    index: int
    sweep_id: str
    values: list[float]
    status: str
    sweep_state_path: str
    summary_csv_path: str | None = None
    started_at: str
    finished_at: str | None = None


class AdaptiveOptimizationState(BaseModel):
    tool_name: str = "adaptive_optimizer"
    status: OptimizationStatus
    optimize_id: str
    optimize_dir: str
    base_task_path: str
    created_at: str
    updated_at: str
    execute: bool
    axis: dict[str, Any]
    objective: dict[str, Any]
    max_rounds: int
    max_cases: int
    rounds: list[OptimizationRound] = Field(default_factory=list)
    observations: list[OptimizationObservation] = Field(default_factory=list)
    best_observation: dict[str, Any] | None = None
    next_action: str | None = None
    failure_reason: str | None = None


TaskRunner = Any


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_optimize_id() -> str:
    return f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_optimization_state(state: AdaptiveOptimizationState, path: Path) -> None:
    state.updated_at = utc_timestamp()
    write_json(path, state.model_dump(mode="json"))


def load_optimization_state(path: Path) -> AdaptiveOptimizationState:
    return AdaptiveOptimizationState.model_validate_json(path.read_text(encoding="utf-8"))


def create_initial_state(
    request: AdaptiveOptimizationRequest,
    base_task_path: Path,
    optimize_id: str,
    optimize_dir: Path,
) -> AdaptiveOptimizationState:
    now = utc_timestamp()
    return AdaptiveOptimizationState(
        status=OptimizationStatus.RUNNING if request.execute else OptimizationStatus.PLANNED,
        optimize_id=optimize_id,
        optimize_dir=str(optimize_dir),
        base_task_path=str(base_task_path),
        created_at=now,
        updated_at=now,
        execute=request.execute,
        axis=request.axis.model_dump(mode="json"),
        objective=request.objective.model_dump(mode="json"),
        max_rounds=request.max_rounds,
        max_cases=request.max_cases,
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


def initial_axis_values(axis: AdaptiveAxis) -> list[float]:
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


def completed_observations(state: AdaptiveOptimizationState) -> list[OptimizationObservation]:
    return [
        observation
        for observation in state.observations
        if observation.status == "completed" and observation.objective_value is not None
    ]


def choose_best_observation(
    observations: list[OptimizationObservation],
    objective: SweepObjective,
) -> OptimizationObservation | None:
    eligible = [
        observation
        for observation in observations
        if observation.status == "completed" and observation.objective_value is not None
    ]
    if not eligible:
        return None
    reverse = objective.direction == SweepDirection.MAXIMIZE
    return sorted(eligible, key=lambda item: item.objective_value or 0.0, reverse=reverse)[0]


def sorted_seen_values(state: AdaptiveOptimizationState) -> list[float]:
    return sorted(
        dedupe_values(
            [observation.value for observation in state.observations if observation.status != "planned"]
        )
    )


def propose_refinement_values(
    axis: AdaptiveAxis,
    state: AdaptiveOptimizationState,
    objective: SweepObjective,
) -> list[float]:
    if not state.observations:
        return dedupe_values(initial_axis_values(axis))[: axis.initial_points]

    seen = sorted_seen_values(state)
    best = choose_best_observation(completed_observations(state), objective)
    proposals: list[float] = []
    if best is not None and seen:
        best_index = min(range(len(seen)), key=lambda index: abs(seen[index] - best.value))
        neighbor_indexes = [best_index - 1, best_index + 1]
        for neighbor_index in neighbor_indexes:
            if 0 <= neighbor_index < len(seen):
                candidate = midpoint(best.value, seen[neighbor_index], axis.scale)
                if axis.min_value <= candidate <= axis.max_value and not value_seen(candidate, seen):
                    proposals.append(candidate)

    if len(proposals) < axis.max_new_points_per_round and len(seen) >= 2:
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


def observations_from_sweep(
    sweep_state: ParameterSweepState,
    axis: AdaptiveAxis,
    round_index: int,
) -> list[OptimizationObservation]:
    observations: list[OptimizationObservation] = []
    for case in sweep_state.cases:
        raw_value = case.values.get(axis.path)
        if raw_value is None:
            continue
        observations.append(
            OptimizationObservation(
                round_index=round_index,
                sweep_id=sweep_state.sweep_id,
                case_index=case.index,
                task_id=case.task_id,
                value=float(raw_value),
                status=case.status,
                quality_status=case.quality_status,
                objective_value=case.objective_value,
                task_run_state_path=case.task_run_state_path,
                final_state_path=case.final_state_path,
                error=case.error,
            )
        )
    return observations


def run_optimization_round(
    base_spec: TaskSpec,
    request: AdaptiveOptimizationRequest,
    optimize_id: str,
    optimize_dir: Path,
    round_index: int,
    values: list[float],
    task_runner: TaskRunner,
) -> tuple[OptimizationRound, ParameterSweepState]:
    sweep_id = f"{optimize_id}_round_{round_index:03d}"
    started_at = utc_timestamp()
    sweep_state = run_parameter_sweep(
        base_spec,
        ParameterSweepRequest(
            sweep_id=sweep_id,
            axes=[SweepAxis(path=request.axis.path, values=values)],
            objective=request.objective,
            execute=request.execute,
            overwrite=True,
            use_llm=request.use_llm,
            sweep_root=optimize_dir / "sweeps",
            max_cases=max(len(values), 1),
        ),
        task_runner=task_runner,
    )
    round_state = OptimizationRound(
        index=round_index,
        sweep_id=sweep_id,
        values=values,
        status=sweep_state.status,
        sweep_state_path=str(Path(sweep_state.sweep_dir) / "sweep_state.json"),
        summary_csv_path=sweep_state.summary_csv_path,
        started_at=started_at,
        finished_at=utc_timestamp(),
    )
    return round_state, sweep_state


def run_adaptive_optimization(
    base_spec: TaskSpec,
    request: AdaptiveOptimizationRequest,
    task_runner: TaskRunner = run_task,
) -> AdaptiveOptimizationState:
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
        state.max_rounds = request.max_rounds
        state.max_cases = request.max_cases
        state.axis = request.axis.model_dump(mode="json")
        state.objective = request.objective.model_dump(mode="json")
        state.status = OptimizationStatus.RUNNING if request.execute else OptimizationStatus.PLANNED
        write_optimization_state(state, state_path)
    else:
        optimize_dir.mkdir(parents=True, exist_ok=True)
        write_task_spec(base_spec, base_task_path)
        state = create_initial_state(request, base_task_path, optimize_id, optimize_dir)
        write_optimization_state(state, state_path)

    while len(state.rounds) < request.max_rounds and len(state.observations) < request.max_cases:
        values = propose_refinement_values(request.axis, state, request.objective)
        remaining_budget = request.max_cases - len(state.observations)
        values = values[:remaining_budget]
        if not values:
            state.status = OptimizationStatus.COMPLETED
            state.next_action = "no new candidate values inside optimization bounds"
            write_optimization_state(state, state_path)
            return state

        round_index = len(state.rounds) + 1
        round_state, sweep_state = run_optimization_round(
            base_spec,
            request,
            optimize_id,
            optimize_dir,
            round_index,
            values,
            task_runner,
        )
        state.rounds.append(round_state)
        state.observations.extend(observations_from_sweep(sweep_state, request.axis, round_index))
        best = choose_best_observation(state.observations, request.objective)
        state.best_observation = best.model_dump(mode="json") if best else None
        write_optimization_state(state, state_path)

        if not request.execute:
            state.status = OptimizationStatus.PLANNED
            state.next_action = "execute planned optimization round"
            write_optimization_state(state, state_path)
            return state

        if not completed_observations(state):
            state.status = OptimizationStatus.FAILED
            state.failure_reason = "no completed observations were produced"
            state.next_action = "inspect latest sweep and task checkpoints"
            write_optimization_state(state, state_path)
            return state

    if len(state.rounds) >= request.max_rounds:
        state.status = OptimizationStatus.COMPLETED
        state.next_action = "maximum optimization rounds reached"
    elif len(state.observations) >= request.max_cases:
        state.status = OptimizationStatus.COMPLETED
        state.next_action = "maximum optimization case budget reached"
    else:
        state.status = OptimizationStatus.COMPLETED
        state.next_action = "optimization stopped"
    write_optimization_state(state, state_path)
    return state
