from __future__ import annotations

import argparse
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from tcad_agent.task_planner import (
    PlannerStatus,
    TaskPlanningResult,
    plan_task_text_with_llm,
    task_spec_from_planning_result,
    write_planning_result,
)
from tcad_agent.task_spec import (
    PROJECT_ROOT,
    TaskSpec,
    load_task_spec,
    parse_task_text,
    task_spec_to_pn_request,
    write_task_spec,
)
from tcad_agent.tools.pn_junction_iv import PNJunctionIVRequest, ToolStatus, run_pn_junction_iv_sweep


class TaskRunStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskRunState(BaseModel):
    tool_name: str = "tcad_task_runner"
    status: TaskRunStatus
    task_id: str
    task_path: str
    task_run_dir: str
    created_at: str
    updated_at: str
    execute: bool = False
    execution_request: dict[str, Any] | None = None
    execution_state_path: str | None = None
    execution_result: dict[str, Any] | None = None
    final_state_path: str | None = None
    final_quality_report: dict[str, Any] | None = None
    failure_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    planner: str = "deterministic"
    planner_status: str | None = None
    planning_result_path: str | None = None


ToolRunner = Callable[[PNJunctionIVRequest], dict[str, Any]]


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def default_task_root() -> Path:
    return PROJECT_ROOT / "runs" / "tasks"


def task_run_dir(task_root: Path, task_id: str) -> Path:
    return task_root / task_id


def task_state_path(run_dir: Path) -> Path:
    return run_dir / "task_run_state.json"


def planning_result_path(run_dir: Path) -> Path:
    return run_dir / "task_plan_result.json"


def execution_state_path_from_request(request: PNJunctionIVRequest) -> Path:
    return request.run_root / "pn_junction_iv" / (request.run_id or "") / "state.json"


def create_initial_state(
    spec: TaskSpec,
    run_dir: Path,
    task_path: Path,
    execute: bool,
    execution_request: PNJunctionIVRequest,
    planner: str,
    planning_result: TaskPlanningResult | None,
) -> TaskRunState:
    now = utc_timestamp()
    return TaskRunState(
        status=TaskRunStatus.RUNNING if execute else TaskRunStatus.PLANNED,
        task_id=spec.task_id,
        task_path=str(task_path),
        task_run_dir=str(run_dir),
        created_at=now,
        updated_at=now,
        execute=execute,
        execution_request=execution_request.model_dump(mode="json"),
        execution_state_path=str(execution_state_path_from_request(execution_request)),
        warnings=[*spec.assumptions, *spec.warnings],
        planner=planner,
        planner_status=planning_result.status if planning_result else None,
        planning_result_path=str(planning_result_path(run_dir)) if planning_result else None,
    )


def write_task_run_state(state: TaskRunState, path: Path) -> None:
    state.updated_at = utc_timestamp()
    write_json(path, state.model_dump(mode="json"))


def prepare_task_files(
    spec: TaskSpec,
    task_root: Path,
    overwrite: bool,
    resume: bool,
) -> tuple[Path, Path]:
    run_dir = task_run_dir(task_root, spec.task_id)
    task_path = run_dir / "task.json"
    if task_path.exists() and not (overwrite or resume):
        raise FileExistsError(f"Task already exists; use --resume or --overwrite: {task_path}")
    if not resume:
        write_task_spec(spec, task_path)
    return run_dir, task_path


def resolve_task_spec(
    *,
    text: str | None,
    task_path: Path | None,
    task_id: str | None,
    use_llm: bool | None,
    planner: str,
    allow_planner_fallback: bool,
) -> tuple[TaskSpec, TaskPlanningResult | None]:
    if task_path:
        return load_task_spec(task_path), None
    if text is None:
        raise ValueError("Either --text or --task must be provided.")
    if planner == "llm":
        result = plan_task_text_with_llm(
            text,
            task_id=task_id,
            execution_use_llm=use_llm,
            allow_fallback=allow_planner_fallback,
        )
        if result.status == PlannerStatus.FAILED:
            raise ValueError("; ".join(result.validation_errors) or "LLM planner failed")
        return task_spec_from_planning_result(result), result
    return parse_task_text(text=text, task_id=task_id, use_llm=use_llm), None


def run_task(
    spec: TaskSpec,
    *,
    task_root: Path | None = None,
    run_root: Path | None = None,
    execute: bool = False,
    overwrite: bool = False,
    resume: bool = False,
    planner: str = "deterministic",
    planning_result: TaskPlanningResult | None = None,
    tool_runner: ToolRunner = run_pn_junction_iv_sweep,
) -> TaskRunState:
    actual_task_root = task_root or default_task_root()
    run_dir, task_path = prepare_task_files(spec, actual_task_root, overwrite=overwrite, resume=resume)
    execution_request = task_spec_to_pn_request(
        spec,
        run_id=spec.task_id,
        run_root=run_root,
        resume=resume,
    )
    state = create_initial_state(
        spec,
        run_dir,
        task_path,
        execute,
        execution_request,
        planner,
        planning_result,
    )
    state_path = task_state_path(run_dir)
    if planning_result:
        write_planning_result(planning_result, planning_result_path(run_dir))
    write_task_run_state(state, state_path)

    if not execute:
        return state

    try:
        execution_result = tool_runner(execution_request)
    except Exception as exc:
        state.status = TaskRunStatus.FAILED
        state.failure_reason = str(exc)
        write_task_run_state(state, state_path)
        return state

    state.execution_result = execution_result
    state.final_state_path = state.execution_state_path
    state.final_quality_report = execution_result.get("quality_report")
    raw_status = execution_result.get("status")
    status = raw_status.value if isinstance(raw_status, ToolStatus) else str(raw_status)
    if status == ToolStatus.COMPLETED.value:
        state.status = TaskRunStatus.COMPLETED
    else:
        state.status = TaskRunStatus.FAILED
        state.failure_reason = (
            execution_result.get("failure_reason")
            or execution_result.get("next_action")
            or "task execution did not complete"
        )
    write_task_run_state(state, state_path)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or execute a standardized TCAD task spec.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Natural-language TCAD task.")
    source.add_argument("--task", type=Path, help="Existing task.json path.")
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--task-root", type=Path, default=default_task_root())
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs" / "agent_tools")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--planner", choices=["deterministic", "llm"], default="deterministic")
    parser.add_argument("--no-planner-fallback", action="store_true")
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument("--use-llm", dest="use_llm", action="store_true")
    llm_group.add_argument("--no-llm", dest="use_llm", action="store_false")
    parser.set_defaults(use_llm=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        spec, planning_result = resolve_task_spec(
            text=args.text,
            task_path=args.task,
            task_id=args.task_id,
            use_llm=args.use_llm,
            planner=args.planner,
            allow_planner_fallback=not args.no_planner_fallback,
        )
        state = run_task(
            spec,
            task_root=args.task_root,
            run_root=args.run_root,
            execute=args.execute,
            overwrite=args.overwrite,
            resume=args.resume,
            planner=args.planner,
            planning_result=planning_result,
        )
        print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if state.status != TaskRunStatus.FAILED else 1)
    except (ValidationError, ValueError, FileExistsError, FileNotFoundError) as exc:
        print(
            json.dumps(
                {
                    "tool_name": "tcad_task_runner",
                    "status": TaskRunStatus.FAILED,
                    "failure_reason": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
