from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.experiment_index import list_records, rebuild_index
from tcad_agent.physical_benchmark import run_physical_benchmark
from tcad_agent.run_queue import enqueue_run, get_item, run_queue_daemon
from tcad_agent.task_spec import PROJECT_ROOT


class LongRunValidationStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class LongRunValidationRequest(BaseModel):
    validation_id: str | None = None
    validation_root: Path = PROJECT_ROOT / "runs" / "long_run_validation"
    queue_goals: list[dict[str, Any]] = Field(default_factory=list)
    poll_interval_seconds: float = 0.0
    max_idle_loops: int = 1


class LongRunValidationState(BaseModel):
    tool_name: str = "long_run_validation"
    status: LongRunValidationStatus
    validation_id: str
    validation_dir: str
    created_at: str
    updated_at: str
    queue_db_path: str
    queued_items: list[dict[str, Any]] = Field(default_factory=list)
    daemon_result: dict[str, Any] | None = None
    benchmark_results: list[dict[str, Any]] = Field(default_factory=list)
    index_summary: dict[str, Any] | None = None
    indexed_records: list[dict[str, Any]] = Field(default_factory=list)
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_validation_id() -> str:
    return f"longrun_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def default_queue_goals(validation_dir: Path) -> list[dict[str, Any]]:
    run_root = validation_dir / "agent_tools"
    return [
        {
            "queue_id": "longrun_schottky",
            "tool_name": "extended_device_sweep",
            "request": {
                "device_type": "schottky_diode",
                "fidelity": "devsim_1d",
                "evidence_level": "tcad_executable",
                "start": -0.1,
                "stop": 0.1,
                "step": 0.1,
                "run_id": "longrun_schottky",
                "run_root": str(run_root),
            },
            "priority": 10,
        },
        {
            "queue_id": "longrun_power_mosfet",
            "tool_name": "extended_device_sweep",
            "request": {
                "device_type": "power_mosfet_bv_ron",
                "fidelity": "physics_1d",
                "evidence_level": "tcad_executable",
                "run_id": "longrun_power_mosfet",
                "run_root": str(run_root),
            },
            "priority": 5,
        },
        {
            "queue_id": "longrun_bjt",
            "tool_name": "extended_device_sweep",
            "request": {
                "device_type": "bjt_gummel_output",
                "fidelity": "physics_1d",
                "evidence_level": "tcad_executable",
                "run_id": "longrun_bjt",
                "run_root": str(run_root),
            },
            "priority": 4,
        },
        {
            "queue_id": "longrun_power_mosfet_convergence",
            "tool_name": "tool_convergence",
            "request": {
                "convergence_id": "longrun_power_mosfet_convergence",
                "tool_name": "extended_device_sweep",
                "base_request": {
                    "device_type": "power_mosfet_bv_ron",
                    "fidelity": "physics_1d",
                    "evidence_level": "tcad_executable",
                    "start": 0.0,
                    "stop": -90.0,
                    "step": 5.0,
                },
                "axis_path": "power_mos_drift_region_doping_cm3",
                "values": [5.0e15, 1.0e16, 2.0e16],
                "metric_path": "quality_report.metrics.specific_on_resistance_ohm_cm2",
                "relative_tolerance": 0.25,
                "execute": True,
                "convergence_root": str(validation_dir / "tool_convergence"),
                "overwrite": True,
            },
            "priority": 3,
        },
        {
            "queue_id": "longrun_bjt_convergence",
            "tool_name": "tool_convergence",
            "request": {
                "convergence_id": "longrun_bjt_convergence",
                "tool_name": "extended_device_sweep",
                "base_request": {
                    "device_type": "bjt_gummel_output",
                    "fidelity": "physics_1d",
                    "evidence_level": "tcad_executable",
                    "start": 0.55,
                    "stop": 0.8,
                    "step": 0.025,
                },
                "axis_path": "bjt_base_width_um",
                "values": [0.15, 0.2, 0.3],
                "metric_path": "quality_report.metrics.current_gain_beta",
                "relative_tolerance": 0.3,
                "execute": True,
                "convergence_root": str(validation_dir / "tool_convergence"),
                "overwrite": True,
            },
            "priority": 2,
        },
    ]


def write_state(state: LongRunValidationState, path: Path) -> None:
    state.updated_at = utc_timestamp()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")


def run_long_run_validation(request: LongRunValidationRequest) -> LongRunValidationState:
    validation_id = request.validation_id or default_validation_id()
    validation_dir = request.validation_root / validation_id
    validation_dir.mkdir(parents=True, exist_ok=True)
    state_path = validation_dir / "validation_state.json"
    queue_db = validation_dir / "run_queue.sqlite"
    now = utc_timestamp()
    state = LongRunValidationState(
        status=LongRunValidationStatus.COMPLETED,
        validation_id=validation_id,
        validation_dir=str(validation_dir),
        created_at=now,
        updated_at=now,
        queue_db_path=str(queue_db),
    )
    write_state(state, state_path)
    try:
        queue_goals = request.queue_goals or default_queue_goals(validation_dir)
        for item in queue_goals:
            queued = enqueue_run(
                queue_db,
                queue_id=item.get("queue_id"),
                tool_name=item["tool_name"],
                request=item.get("request") or {},
                priority=int(item.get("priority") or 0),
                max_attempts=int(item.get("max_attempts") or 1),
                tags=item.get("tags") or ["long_run_validation"],
            )
            state.queued_items.append(queued.model_dump(mode="json"))
        write_state(state, state_path)

        daemon = run_queue_daemon(
            queue_db,
            owner=f"{validation_id}_daemon",
            concurrency=1,
            poll_interval_seconds=request.poll_interval_seconds,
            max_idle_loops=request.max_idle_loops,
        )
        state.daemon_result = daemon.model_dump(mode="json")

        completed_items = []
        for queued in state.queued_items:
            item = get_item(queue_db, queued["queue_id"])
            if item is None:
                raise RuntimeError(f"queue item disappeared: {queued['queue_id']}")
            completed_items.append(item.model_dump(mode="json"))
            if item.status != "completed":
                raise RuntimeError(f"queue item did not complete: {queued['queue_id']} status={item.status}")
            if item.result_state_path:
                benchmark = run_physical_benchmark(Path(item.result_state_path))
                state.benchmark_results.append(benchmark.model_dump(mode="json"))
        state.queued_items = completed_items

        failed_benchmarks = [item for item in state.benchmark_results if item.get("status") == "failed"]
        if failed_benchmarks:
            raise RuntimeError(f"{len(failed_benchmarks)} benchmark(s) failed")

        index_db = validation_dir / "experiment_index.sqlite"
        state.index_summary = rebuild_index(validation_dir, index_db)
        state.indexed_records = list_records(index_db, limit=20)
        state.status = LongRunValidationStatus.COMPLETED
    except Exception as exc:
        state.status = LongRunValidationStatus.FAILED
        state.failure_reason = str(exc)
    write_state(state, state_path)
    return state
