from __future__ import annotations

import csv
import itertools
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, model_validator

from tcad_agent.task_spec import PROJECT_ROOT, TaskSpec, write_task_spec
from tcad_agent.tools.task_runner import TaskRunState, run_task


class SweepStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SweepDirection(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class SweepAxis(BaseModel):
    path: str
    values: list[Any]

    @model_validator(mode="after")
    def validate_axis(self) -> "SweepAxis":
        if not self.path:
            raise ValueError("axis path is required")
        if not self.values:
            raise ValueError("axis values must not be empty")
        if self.path.split(".")[0] not in {"sweep", "parameters", "mesh", "quality", "execution"}:
            raise ValueError(f"unsupported sweep path: {self.path}")
        return self


class SweepObjective(BaseModel):
    metric_path: str = "final_quality_report.metrics.final_total_current_a"
    direction: SweepDirection = SweepDirection.MINIMIZE
    absolute: bool = True


class ParameterSweepRequest(BaseModel):
    sweep_id: str | None = None
    axes: list[SweepAxis]
    objective: SweepObjective = Field(default_factory=SweepObjective)
    execute: bool = False
    overwrite: bool = False
    use_llm: bool | None = None
    sweep_root: Path = PROJECT_ROOT / "runs" / "sweeps"
    max_cases: int = Field(default=100, ge=1)

    @model_validator(mode="after")
    def validate_case_count(self) -> "ParameterSweepRequest":
        total = 1
        for axis in self.axes:
            total *= len(axis.values)
        if total > self.max_cases:
            raise ValueError(f"sweep would create {total} cases, exceeding max_cases={self.max_cases}")
        return self


class SweepCase(BaseModel):
    index: int
    task_id: str
    values: dict[str, Any]
    status: str | None = None
    task_path: str | None = None
    task_run_state_path: str | None = None
    final_state_path: str | None = None
    quality_status: str | None = None
    objective_value: float | None = None
    error: str | None = None


class ParameterSweepState(BaseModel):
    tool_name: str = "parameter_sweep"
    status: SweepStatus
    sweep_id: str
    sweep_dir: str
    base_task_path: str
    created_at: str
    updated_at: str
    execute: bool
    axes: list[dict[str, Any]]
    objective: dict[str, Any]
    cases: list[SweepCase] = Field(default_factory=list)
    best_case_index: int | None = None
    best_case: dict[str, Any] | None = None
    summary_csv_path: str | None = None
    failure_reason: str | None = None


TaskRunner = Callable[..., TaskRunState]


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_sweep_id() -> str:
    return f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_sweep_state(state: ParameterSweepState, path: Path) -> None:
    state.updated_at = utc_timestamp()
    write_json(path, state.model_dump(mode="json"))


def parse_scalar(value: str) -> Any:
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(marker in lowered for marker in [".", "e"]):
            return float(stripped)
        return int(stripped)
    except ValueError:
        return stripped


def parse_axis_spec(value: str) -> SweepAxis:
    if "=" not in value:
        raise ValueError("axis must use path=value1,value2 syntax")
    path, raw_values = value.split("=", 1)
    values = [parse_scalar(item) for item in raw_values.split(",") if item.strip()]
    return SweepAxis(path=path.strip(), values=values)


def set_nested_value(data: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cursor: dict[str, Any] = data
    for key in keys[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set nested path through non-object: {path}")
        cursor = next_value
    cursor[keys[-1]] = value


def get_nested_value(data: dict[str, Any], path: str) -> Any:
    cursor: Any = data
    for key in path.split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def generate_case_specs(
    base_spec: TaskSpec,
    sweep_id: str,
    axes: list[SweepAxis],
) -> list[tuple[TaskSpec, dict[str, Any]]]:
    cases: list[tuple[TaskSpec, dict[str, Any]]] = []
    value_products = itertools.product(*(axis.values for axis in axes))
    for index, values in enumerate(value_products, start=1):
        data = base_spec.model_dump(mode="json")
        case_values: dict[str, Any] = {}
        for axis, value in zip(axes, values):
            set_nested_value(data, axis.path, value)
            case_values[axis.path] = value
        data["task_id"] = f"{sweep_id}_case_{index:03d}"
        data["title"] = f"{base_spec.title} case {index:03d}"
        cases.append((TaskSpec.model_validate(data), case_values))
    return cases


def metric_value(data: dict[str, Any], objective: SweepObjective) -> float | None:
    raw_value = get_nested_value(data, objective.metric_path)
    if raw_value is None:
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    return abs(value) if objective.absolute else value


def choose_best_case(
    cases: list[SweepCase],
    objective: SweepObjective,
) -> SweepCase | None:
    eligible = [case for case in cases if case.objective_value is not None and case.status == "completed"]
    if not eligible:
        return None
    reverse = objective.direction == SweepDirection.MAXIMIZE
    return sorted(eligible, key=lambda case: case.objective_value or 0.0, reverse=reverse)[0]


def write_summary_csv(path: Path, cases: list[SweepCase], axes: list[SweepAxis]) -> None:
    fieldnames = [
        "index",
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
        for case in cases:
            row = {
                "index": case.index,
                "task_id": case.task_id,
                "status": case.status,
                "quality_status": case.quality_status,
                "objective_value": case.objective_value,
                "final_state_path": case.final_state_path,
                "error": case.error,
            }
            row.update(case.values)
            writer.writerow(row)


def create_initial_state(
    request: ParameterSweepRequest,
    base_task_path: Path,
    sweep_id: str,
    sweep_dir: Path,
) -> ParameterSweepState:
    now = utc_timestamp()
    return ParameterSweepState(
        status=SweepStatus.RUNNING if request.execute else SweepStatus.PLANNED,
        sweep_id=sweep_id,
        sweep_dir=str(sweep_dir),
        base_task_path=str(base_task_path),
        created_at=now,
        updated_at=now,
        execute=request.execute,
        axes=[axis.model_dump(mode="json") for axis in request.axes],
        objective=request.objective.model_dump(mode="json"),
    )


def run_parameter_sweep(
    base_spec: TaskSpec,
    request: ParameterSweepRequest,
    task_runner: TaskRunner = run_task,
) -> ParameterSweepState:
    sweep_id = request.sweep_id or default_sweep_id()
    sweep_dir = request.sweep_root / sweep_id
    if sweep_dir.exists() and not request.overwrite:
        raise FileExistsError(f"Sweep already exists; use --overwrite: {sweep_dir}")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    base_task_path = sweep_dir / "base_task.json"
    write_task_spec(base_spec, base_task_path)
    state_path = sweep_dir / "sweep_state.json"
    state = create_initial_state(request, base_task_path, sweep_id, sweep_dir)
    write_sweep_state(state, state_path)

    case_specs = generate_case_specs(base_spec, sweep_id, request.axes)
    for index, (case_spec, values) in enumerate(case_specs, start=1):
        case = SweepCase(index=index, task_id=case_spec.task_id, values=values)
        state.cases.append(case)
        case_task_path = sweep_dir / "tasks" / case_spec.task_id / "task.json"
        write_sweep_state(state, state_path)

        try:
            task_state = task_runner(
                case_spec,
                task_root=sweep_dir / "tasks",
                run_root=sweep_dir / "agent_tools",
                execute=request.execute,
                overwrite=True,
            )
        except Exception as exc:
            case.status = "failed"
            case.error = str(exc)
            write_sweep_state(state, state_path)
            continue

        task_dump = task_state.model_dump(mode="json")
        case.status = task_dump.get("status")
        case.task_path = str(case_task_path)
        case.task_run_state_path = str(sweep_dir / "tasks" / case_spec.task_id / "task_run_state.json")
        case.final_state_path = task_dump.get("final_state_path")
        case.quality_status = (task_dump.get("final_quality_report") or {}).get("status")
        case.objective_value = metric_value(task_dump, request.objective)
        if case.status == "failed":
            case.error = task_dump.get("failure_reason")
        write_sweep_state(state, state_path)

    best_case = choose_best_case(state.cases, request.objective)
    if best_case:
        state.best_case_index = best_case.index
        state.best_case = best_case.model_dump(mode="json")

    summary_csv = sweep_dir / "summary.csv"
    write_summary_csv(summary_csv, state.cases, request.axes)
    state.summary_csv_path = str(summary_csv)
    if request.execute and any(case.status == "failed" for case in state.cases):
        state.status = SweepStatus.FAILED
        state.failure_reason = "one or more sweep cases failed"
    else:
        state.status = SweepStatus.COMPLETED if request.execute else SweepStatus.PLANNED
    write_sweep_state(state, state_path)
    return state
