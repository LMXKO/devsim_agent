from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from tcad_agent.parameter_sweep import (
    ParameterSweepRequest,
    ParameterSweepState,
    SweepAxis,
    SweepDirection,
    SweepObjective,
    run_parameter_sweep,
)
from tcad_agent.task_spec import PROJECT_ROOT, TaskSpec, load_task_spec, write_task_spec
from tcad_agent.tools.task_runner import resolve_task_spec, run_task


class MeshConvergenceStatus(str, Enum):
    PLANNED = "planned"
    COMPLETED = "completed"
    FAILED = "failed"


class MeshConvergenceQuality(str, Enum):
    PASSED = "passed"
    SUSPICIOUS = "suspicious"
    FAILED = "failed"
    PLANNED = "planned"


class MeshConvergenceRequest(BaseModel):
    convergence_id: str | None = None
    axis_path: str = "mesh.junction_spacing_um"
    values: list[float] = Field(default_factory=lambda: [2e-5, 1e-5, 5e-6])
    objective: SweepObjective = Field(default_factory=SweepObjective)
    relative_tolerance: float = Field(default=0.05, ge=0.0)
    execute: bool = False
    overwrite: bool = False
    use_llm: bool | None = None
    convergence_root: Path = PROJECT_ROOT / "runs" / "mesh_convergence"
    max_cases: int = Field(default=10, ge=2)

    @model_validator(mode="after")
    def validate_request(self) -> "MeshConvergenceRequest":
        if not self.axis_path.startswith("mesh."):
            raise ValueError("mesh convergence axis_path must start with mesh.")
        if len(self.values) < 2:
            raise ValueError("at least two mesh values are required")
        if any(value <= 0 for value in self.values):
            raise ValueError("mesh values must be positive")
        if len(self.values) > self.max_cases:
            raise ValueError(f"mesh convergence would create {len(self.values)} cases, exceeding max_cases={self.max_cases}")
        return self


class MeshConvergenceState(BaseModel):
    tool_name: str = "mesh_convergence"
    status: MeshConvergenceStatus
    convergence_id: str
    convergence_dir: str
    base_task_path: str
    created_at: str
    updated_at: str
    execute: bool
    axis_path: str
    values: list[float]
    objective: dict[str, Any]
    relative_tolerance: float
    sweep_state_path: str | None = None
    summary_csv_path: str | None = None
    cases: list[dict[str, Any]] = Field(default_factory=list)
    quality_report: dict[str, Any] | None = None
    next_action: str | None = None
    failure_reason: str | None = None


TaskRunner = Any


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_convergence_id() -> str:
    return f"meshconv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_state(state: MeshConvergenceState, path: Path) -> None:
    state.updated_at = utc_timestamp()
    write_json(path, state.model_dump(mode="json"))


def load_state(path: Path) -> MeshConvergenceState:
    return MeshConvergenceState.model_validate_json(path.read_text(encoding="utf-8"))


def create_initial_state(
    request: MeshConvergenceRequest,
    base_task_path: Path,
    convergence_id: str,
    convergence_dir: Path,
) -> MeshConvergenceState:
    now = utc_timestamp()
    return MeshConvergenceState(
        status=MeshConvergenceStatus.PLANNED,
        convergence_id=convergence_id,
        convergence_dir=str(convergence_dir),
        base_task_path=str(base_task_path),
        created_at=now,
        updated_at=now,
        execute=request.execute,
        axis_path=request.axis_path,
        values=[float(value) for value in request.values],
        objective=request.objective.model_dump(mode="json"),
        relative_tolerance=request.relative_tolerance,
        next_action="execute mesh convergence sweep" if not request.execute else "run mesh convergence sweep",
    )


def relative_delta(left: float, right: float) -> float:
    denominator = max(abs(left), abs(right), 1e-300)
    return abs(left - right) / denominator


def completed_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        case
        for case in cases
        if case.get("status") == "completed" and case.get("objective_value") is not None
    ]


def mesh_value(case: dict[str, Any], axis_path: str) -> float | None:
    values = case.get("values") or {}
    try:
        return float(values.get(axis_path))
    except (TypeError, ValueError):
        return None


def build_quality_report(
    sweep_state: ParameterSweepState,
    request: MeshConvergenceRequest,
) -> dict[str, Any]:
    cases = [case.model_dump(mode="json") for case in sweep_state.cases]
    if not request.execute:
        return {
            "status": MeshConvergenceQuality.PLANNED,
            "issues": [],
            "metrics": {"cases": len(cases), "completed_cases": 0},
            "recommended_next_action": "execute planned mesh convergence sweep",
        }

    completed = completed_cases(cases)
    issues: list[dict[str, Any]] = []
    if len(completed) < 2:
        issues.append(
            {
                "code": "too_few_completed_mesh_cases",
                "severity": "error",
                "message": "At least two completed mesh cases are required for convergence comparison.",
                "evidence": {"completed_cases": len(completed)},
            }
        )
        return {
            "status": MeshConvergenceQuality.FAILED,
            "issues": issues,
            "metrics": {"cases": len(cases), "completed_cases": len(completed)},
            "recommended_next_action": "rerun failed mesh cases before trusting convergence",
        }

    ordered = sorted(
        completed,
        key=lambda case: mesh_value(case, request.axis_path) or math.inf,
    )
    finest = ordered[0]
    previous = ordered[1]
    finest_metric = float(finest["objective_value"])
    previous_metric = float(previous["objective_value"])
    delta = relative_delta(finest_metric, previous_metric)
    if delta > request.relative_tolerance:
        issues.append(
            {
                "code": "mesh_not_converged",
                "severity": "warning",
                "message": "Objective changed more than the configured tolerance between the two finest meshes.",
                "evidence": {
                    "relative_delta": delta,
                    "relative_tolerance": request.relative_tolerance,
                    "finest_mesh_value": mesh_value(finest, request.axis_path),
                    "previous_mesh_value": mesh_value(previous, request.axis_path),
                    "finest_objective": finest_metric,
                    "previous_objective": previous_metric,
                },
            }
        )
    failed_cases = [case for case in cases if case.get("status") == "failed"]
    if failed_cases:
        issues.append(
            {
                "code": "mesh_case_failures",
                "severity": "warning",
                "message": "One or more mesh cases failed while enough cases completed for comparison.",
                "evidence": {"failed_cases": len(failed_cases)},
            }
        )
    status = MeshConvergenceQuality.SUSPICIOUS if issues else MeshConvergenceQuality.PASSED
    return {
        "status": status,
        "issues": issues,
        "metrics": {
            "cases": len(cases),
            "completed_cases": len(completed),
            "axis_path": request.axis_path,
            "finest_mesh_value": mesh_value(finest, request.axis_path),
            "previous_mesh_value": mesh_value(previous, request.axis_path),
            "finest_objective": finest_metric,
            "previous_objective": previous_metric,
            "relative_delta": delta,
            "relative_tolerance": request.relative_tolerance,
        },
        "recommended_next_action": (
            "accept mesh convergence for this objective"
            if status == MeshConvergenceQuality.PASSED
            else "refine the mesh further or inspect failed/suspicious mesh cases"
        ),
    }


def run_mesh_convergence(
    base_spec: TaskSpec,
    request: MeshConvergenceRequest,
    task_runner: TaskRunner = run_task,
) -> MeshConvergenceState:
    convergence_id = request.convergence_id or default_convergence_id()
    convergence_dir = request.convergence_root / convergence_id
    state_path = convergence_dir / "state.json"
    base_task_path = convergence_dir / "base_task.json"

    if state_path.exists() and not request.overwrite:
        state = load_state(state_path)
        if base_task_path.exists():
            base_spec = load_task_spec(base_task_path)
    else:
        convergence_dir.mkdir(parents=True, exist_ok=True)
        write_task_spec(base_spec, base_task_path)
        state = create_initial_state(request, base_task_path, convergence_id, convergence_dir)
        write_state(state, state_path)

    sweep_state = run_parameter_sweep(
        base_spec,
        ParameterSweepRequest(
            sweep_id=f"{convergence_id}_mesh_sweep",
            axes=[SweepAxis(path=request.axis_path, values=request.values)],
            objective=request.objective,
            execute=request.execute,
            overwrite=True,
            use_llm=request.use_llm,
            sweep_root=convergence_dir / "sweeps",
            max_cases=len(request.values),
        ),
        task_runner=task_runner,
    )
    quality_report = build_quality_report(sweep_state, request)
    state.execute = request.execute
    state.values = [float(value) for value in request.values]
    state.objective = request.objective.model_dump(mode="json")
    state.sweep_state_path = str(Path(sweep_state.sweep_dir) / "sweep_state.json")
    state.summary_csv_path = sweep_state.summary_csv_path
    state.cases = [case.model_dump(mode="json") for case in sweep_state.cases]
    state.quality_report = quality_report
    state.status = (
        MeshConvergenceStatus.FAILED
        if quality_report["status"] == MeshConvergenceQuality.FAILED
        else MeshConvergenceStatus.COMPLETED if request.execute else MeshConvergenceStatus.PLANNED
    )
    state.next_action = quality_report["recommended_next_action"]
    state.failure_reason = "mesh convergence failed" if state.status == MeshConvergenceStatus.FAILED else None
    write_state(state, state_path)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a checkpointed TCAD mesh convergence check.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Base natural-language TCAD task.")
    source.add_argument("--task", type=Path, help="Base task.json path.")
    parser.add_argument("--convergence-id", default=None)
    parser.add_argument("--task-id", default=None, help="Task id for the base task when using --text.")
    parser.add_argument("--axis-path", default="mesh.junction_spacing_um")
    parser.add_argument(
        "--value",
        action="append",
        type=float,
        help="Mesh value in um. Repeat for coarse-to-fine values.",
    )
    parser.add_argument(
        "--objective-metric",
        default="final_quality_report.metrics.final_total_current_a",
    )
    parser.add_argument("--direction", choices=["minimize", "maximize"], default="minimize")
    parser.add_argument("--raw-objective", action="store_true", help="Do not take absolute value of objective.")
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--convergence-root", type=Path, default=PROJECT_ROOT / "runs" / "mesh_convergence")
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
        base_spec, planning_result = resolve_task_spec(
            text=args.text,
            task_path=args.task,
            task_id=args.task_id,
            use_llm=args.use_llm,
            planner=args.planner,
            allow_planner_fallback=not args.no_planner_fallback,
        )
        request = MeshConvergenceRequest(
            convergence_id=args.convergence_id,
            axis_path=args.axis_path,
            values=args.value or [2e-5, 1e-5, 5e-6],
            objective=SweepObjective(
                metric_path=args.objective_metric,
                direction=SweepDirection(args.direction),
                absolute=not args.raw_objective,
            ),
            relative_tolerance=args.relative_tolerance,
            execute=args.execute,
            overwrite=args.overwrite,
            use_llm=args.use_llm,
            convergence_root=args.convergence_root,
            max_cases=args.max_cases,
        )
        state = run_mesh_convergence(base_spec, request)
        if planning_result:
            plan_path = Path(state.convergence_dir) / "base_task_plan_result.json"
            plan_path.write_text(
                json.dumps(planning_result.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if state.status != MeshConvergenceStatus.FAILED else 1)
    except (ValidationError, ValueError, FileExistsError, FileNotFoundError) as exc:
        print(
            json.dumps(
                {
                    "tool_name": "mesh_convergence",
                    "status": MeshConvergenceStatus.FAILED,
                    "failure_reason": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
