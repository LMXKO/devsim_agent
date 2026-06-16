from __future__ import annotations

import argparse

import json
import sqlite3
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from tcad_agent.task_spec import PROJECT_ROOT


SCHEMA_VERSION = 1
Runner = Callable[[dict[str, Any]], Any]


class QueueStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QueueItem(BaseModel):
    queue_id: str
    status: QueueStatus
    priority: int = 0
    tool_name: str
    request: dict[str, Any]
    tags: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None
    attempts: int = 0
    max_attempts: int = 1
    lease_owner: str | None = None
    lease_expires_at: str | None = None
    budget_seconds: float | None = None
    budget_cases: int | None = None
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    result_state_path: str | None = None
    failure_reason: str | None = None


class QueueWorkerResult(BaseModel):
    db_path: str
    owner: str
    claimed: int
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    items: list[dict[str, Any]] = Field(default_factory=list)


class QueueDaemonResult(BaseModel):
    db_path: str
    owner: str
    loops: int = 0
    idle_loops: int = 0
    stopped_by: str
    claimed: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    worker_results: list[dict[str, Any]] = Field(default_factory=list)


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_queue_db_path() -> Path:
    return PROJECT_ROOT / "runs" / "run_queue.sqlite"


def default_queue_id(prefix: str = "queue") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def default_worker_owner() -> str:
    return f"worker_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_db(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS run_queue (
          queue_id TEXT PRIMARY KEY,
          status TEXT NOT NULL,
          priority INTEGER NOT NULL,
          tool_name TEXT NOT NULL,
          request_json TEXT NOT NULL,
          tags_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          started_at TEXT,
          completed_at TEXT,
          attempts INTEGER NOT NULL,
          max_attempts INTEGER NOT NULL,
          lease_owner TEXT,
          lease_expires_at TEXT,
          budget_seconds REAL,
          budget_cases INTEGER,
          checkpoint_json TEXT NOT NULL,
          result_json TEXT,
          result_state_path TEXT,
          failure_reason TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_run_queue_status ON run_queue(status);
        CREATE INDEX IF NOT EXISTS idx_run_queue_priority ON run_queue(priority);
        CREATE INDEX IF NOT EXISTS idx_run_queue_tool ON run_queue(tool_name);
        """
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )
    connection.commit()


def json_loads_or_default(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def row_to_item(row: sqlite3.Row) -> QueueItem:
    return QueueItem(
        queue_id=row["queue_id"],
        status=QueueStatus(row["status"]),
        priority=int(row["priority"]),
        tool_name=row["tool_name"],
        request=json_loads_or_default(row["request_json"], {}),
        tags=json_loads_or_default(row["tags_json"], []),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        attempts=int(row["attempts"]),
        max_attempts=int(row["max_attempts"]),
        lease_owner=row["lease_owner"],
        lease_expires_at=row["lease_expires_at"],
        budget_seconds=row["budget_seconds"],
        budget_cases=row["budget_cases"],
        checkpoint=json_loads_or_default(row["checkpoint_json"], {}),
        result=json_loads_or_default(row["result_json"], None),
        result_state_path=row["result_state_path"],
        failure_reason=row["failure_reason"],
    )


def fetch_item(connection: sqlite3.Connection, queue_id: str) -> QueueItem | None:
    row = connection.execute("SELECT * FROM run_queue WHERE queue_id = ?", (queue_id,)).fetchone()
    return row_to_item(row) if row else None


def get_item(db_path: Path, queue_id: str) -> QueueItem | None:
    connection = connect(db_path)
    initialize_db(connection)
    try:
        return fetch_item(connection, queue_id)
    finally:
        connection.close()


def enqueue_run(
    db_path: Path,
    *,
    tool_name: str,
    request: dict[str, Any],
    queue_id: str | None = None,
    priority: int = 0,
    tags: list[str] | None = None,
    max_attempts: int = 1,
    budget_seconds: float | None = None,
    budget_cases: int | None = None,
) -> QueueItem:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if budget_seconds is not None and budget_seconds < 0:
        raise ValueError("budget_seconds must be non-negative")
    if budget_cases is not None and budget_cases < 0:
        raise ValueError("budget_cases must be non-negative")

    actual_id = queue_id or default_queue_id("run")
    now = utc_timestamp()
    connection = connect(db_path)
    initialize_db(connection)
    try:
        connection.execute(
            """
            INSERT INTO run_queue (
              queue_id, status, priority, tool_name, request_json, tags_json,
              created_at, updated_at, attempts, max_attempts, budget_seconds,
              budget_cases, checkpoint_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                actual_id,
                QueueStatus.QUEUED.value,
                priority,
                tool_name,
                json.dumps(request, ensure_ascii=False),
                json.dumps(tags or [], ensure_ascii=False),
                now,
                now,
                0,
                max_attempts,
                budget_seconds,
                budget_cases,
                json.dumps({"queued_at": now}, ensure_ascii=False),
            ),
        )
        connection.commit()
        item = fetch_item(connection, actual_id)
        if item is None:
            raise RuntimeError(f"failed to enqueue run {actual_id}")
        return item
    finally:
        connection.close()


def list_items(
    db_path: Path,
    *,
    status: QueueStatus | str | None = None,
    tool_name: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    connection = connect(db_path)
    initialize_db(connection)
    try:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(QueueStatus(status).value)
        if tool_name:
            clauses.append("tool_name = ?")
            params.append(tool_name)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = connection.execute(
            f"""
            SELECT * FROM run_queue
            {where}
            ORDER BY
              CASE status
                WHEN 'running' THEN 0
                WHEN 'queued' THEN 1
                WHEN 'paused' THEN 2
                WHEN 'failed' THEN 3
                WHEN 'completed' THEN 4
                WHEN 'cancelled' THEN 5
                ELSE 6
              END,
              priority DESC,
              created_at ASC
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()
        return [row_to_item(row).model_dump(mode="json") for row in rows]
    finally:
        connection.close()


def set_item_status(
    db_path: Path,
    queue_id: str,
    status: QueueStatus,
    *,
    failure_reason: str | None = None,
) -> QueueItem:
    now = utc_timestamp()
    connection = connect(db_path)
    initialize_db(connection)
    try:
        item = fetch_item(connection, queue_id)
        if item is None:
            raise FileNotFoundError(f"queue item does not exist: {queue_id}")
        if item.status in {QueueStatus.COMPLETED, QueueStatus.CANCELLED}:
            raise ValueError(f"cannot change terminal queue item {queue_id} from {item.status}")
        updates: dict[str, Any] = {
            "status": status.value,
            "updated_at": now,
            "failure_reason": failure_reason,
        }
        if status in {QueueStatus.QUEUED, QueueStatus.PAUSED, QueueStatus.CANCELLED}:
            updates["lease_owner"] = None
            updates["lease_expires_at"] = None
        if status in {QueueStatus.CANCELLED, QueueStatus.FAILED}:
            updates["completed_at"] = now
        assignments = ", ".join(f"{key} = ?" for key in updates)
        connection.execute(
            f"UPDATE run_queue SET {assignments} WHERE queue_id = ?",
            [*updates.values(), queue_id],
        )
        connection.commit()
        updated = fetch_item(connection, queue_id)
        if updated is None:
            raise RuntimeError(f"failed to update queue item {queue_id}")
        return updated
    finally:
        connection.close()


def pause_item(db_path: Path, queue_id: str) -> QueueItem:
    return set_item_status(db_path, queue_id, QueueStatus.PAUSED)


def resume_item(db_path: Path, queue_id: str) -> QueueItem:
    return set_item_status(db_path, queue_id, QueueStatus.QUEUED)


def autonomous_agent_control_dir(item: QueueItem) -> Path | None:
    if item.tool_name == "autonomous_devsim_agent":
        raw_root = item.request.get("agent_root")
        agent_root = Path(str(raw_root)) if raw_root else PROJECT_ROOT / "runs" / "autonomous_devsim_agent"
        agent_id = str(item.request.get("agent_id") or item.queue_id)
        return agent_root / agent_id
    if item.tool_name == "agent_soak":
        raw_root = item.request.get("soak_root")
        soak_root = Path(str(raw_root)) if raw_root else PROJECT_ROOT / "runs" / "agent_soak"
        soak_id = str(item.request.get("soak_id") or item.queue_id)
        return soak_root / soak_id
    return None


def request_autonomous_agent_cancel(item: QueueItem) -> str | None:
    control_dir = autonomous_agent_control_dir(item)
    if control_dir is None:
        return None
    control_dir.mkdir(parents=True, exist_ok=True)
    cancel_file = Path(str(item.request.get("cancel_file") or control_dir / "cancel.requested"))
    cancel_file.parent.mkdir(parents=True, exist_ok=True)
    cancel_file.write_text(json.dumps({"queue_id": item.queue_id, "cancelled_at": utc_timestamp()}, ensure_ascii=False), encoding="utf-8")
    return str(cancel_file)


def cancel_item(db_path: Path, queue_id: str) -> QueueItem:
    item = get_item(db_path, queue_id)
    if item:
        request_autonomous_agent_cancel(item)
    return set_item_status(db_path, queue_id, QueueStatus.CANCELLED)


def update_item_request(
    db_path: Path,
    queue_id: str,
    patch: dict[str, Any],
    *,
    checkpoint_patch: dict[str, Any] | None = None,
) -> QueueItem:
    now = utc_timestamp()
    connection = connect(db_path)
    initialize_db(connection)
    try:
        item = fetch_item(connection, queue_id)
        if item is None:
            raise FileNotFoundError(f"queue item does not exist: {queue_id}")
        request = {**item.request, **patch}
        checkpoint = {**item.checkpoint, **(checkpoint_patch or {}), "request_updated_at": now}
        connection.execute(
            """
            UPDATE run_queue
            SET request_json = ?, checkpoint_json = ?, updated_at = ?
            WHERE queue_id = ?
            """,
            (
                json.dumps(request, ensure_ascii=False),
                json.dumps(checkpoint, ensure_ascii=False),
                now,
                queue_id,
            ),
        )
        connection.commit()
        updated = fetch_item(connection, queue_id)
        if updated is None:
            raise RuntimeError(f"failed to update queue item {queue_id}")
        return updated
    finally:
        connection.close()


def recover_stale_items(db_path: Path, *, now: str | None = None) -> dict[str, Any]:
    actual_now = now or utc_timestamp()
    recovered = 0
    failed = 0
    connection = connect(db_path)
    initialize_db(connection)
    try:
        rows = connection.execute(
            """
            SELECT * FROM run_queue
            WHERE status = ? AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?
            """,
            (QueueStatus.RUNNING.value, actual_now),
        ).fetchall()
        for row in rows:
            item = row_to_item(row)
            if item.attempts >= item.max_attempts:
                connection.execute(
                    """
                    UPDATE run_queue
                    SET status = ?, updated_at = ?, completed_at = ?, lease_owner = NULL,
                        lease_expires_at = NULL, failure_reason = ?
                    WHERE queue_id = ?
                    """,
                    (
                        QueueStatus.FAILED.value,
                        actual_now,
                        actual_now,
                        "lease expired and max_attempts was reached",
                        item.queue_id,
                    ),
                )
                failed += 1
            else:
                checkpoint = {
                    **item.checkpoint,
                    "recovered_at": actual_now,
                    "last_lease_owner": item.lease_owner,
                    "last_lease_expires_at": item.lease_expires_at,
                }
                connection.execute(
                    """
                    UPDATE run_queue
                    SET status = ?, updated_at = ?, lease_owner = NULL,
                        lease_expires_at = NULL, checkpoint_json = ?
                    WHERE queue_id = ?
                    """,
                    (
                        QueueStatus.QUEUED.value,
                        actual_now,
                        json.dumps(checkpoint, ensure_ascii=False),
                        item.queue_id,
                    ),
                )
                recovered += 1
        connection.commit()
        return {"recovered": recovered, "failed": failed}
    finally:
        connection.close()


def recover_owner_running_items(db_path: Path, *, owner: str, now: str | None = None) -> dict[str, Any]:
    """Recover running items owned by an inactive web worker process.

    The web UI uses a stable owner name so users can see one queue namespace
    across server restarts. If the server is restarted while a worker lease is
    active, the row can remain RUNNING until the long lease expires even though
    no worker thread exists anymore. This helper is intentionally owner-scoped
    so it does not disturb external workers.
    """
    actual_now = now or utc_timestamp()
    recovered = 0
    failed = 0
    connection = connect(db_path)
    initialize_db(connection)
    try:
        rows = connection.execute(
            """
            SELECT * FROM run_queue
            WHERE status = ? AND lease_owner = ?
            """,
            (QueueStatus.RUNNING.value, owner),
        ).fetchall()
        for row in rows:
            item = row_to_item(row)
            checkpoint = {
                **item.checkpoint,
                "owner_recovered_at": actual_now,
                "owner_recovery_count": int(item.checkpoint.get("owner_recovery_count") or 0) + 1,
                "last_lease_owner": item.lease_owner,
                "last_lease_expires_at": item.lease_expires_at,
            }
            connection.execute(
                """
                UPDATE run_queue
                SET status = ?, updated_at = ?, attempts = MAX(attempts - 1, 0),
                    lease_owner = NULL, lease_expires_at = NULL, checkpoint_json = ?
                WHERE queue_id = ?
                """,
                (
                    QueueStatus.QUEUED.value,
                    actual_now,
                    json.dumps(checkpoint, ensure_ascii=False),
                    item.queue_id,
                ),
            )
            recovered += 1
        connection.commit()
        return {"recovered": recovered, "failed": failed}
    finally:
        connection.close()


def claim_next_items(
    db_path: Path,
    *,
    owner: str,
    limit: int = 1,
    lease_seconds: float = 3600.0,
) -> list[QueueItem]:
    if limit < 1:
        return []
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    recover_stale_items(db_path)
    now = utc_timestamp()
    lease_expires_at = datetime.utcfromtimestamp(time.time() + lease_seconds).replace(microsecond=0).isoformat() + "Z"
    connection = connect(db_path)
    initialize_db(connection)
    try:
        rows = connection.execute(
            """
            SELECT * FROM run_queue
            WHERE status = ? AND attempts < max_attempts
            ORDER BY priority DESC, created_at ASC
            LIMIT ?
            """,
            (QueueStatus.QUEUED.value, limit),
        ).fetchall()
        claimed: list[QueueItem] = []
        for row in rows:
            item = row_to_item(row)
            checkpoint = {
                **item.checkpoint,
                "claimed_at": now,
                "lease_owner": owner,
                "lease_expires_at": lease_expires_at,
            }
            connection.execute(
                """
                UPDATE run_queue
                SET status = ?, updated_at = ?, started_at = COALESCE(started_at, ?),
                    attempts = attempts + 1, lease_owner = ?, lease_expires_at = ?,
                    checkpoint_json = ?
                WHERE queue_id = ? AND status = ?
                """,
                (
                    QueueStatus.RUNNING.value,
                    now,
                    now,
                    owner,
                    lease_expires_at,
                    json.dumps(checkpoint, ensure_ascii=False),
                    item.queue_id,
                    QueueStatus.QUEUED.value,
                ),
            )
            updated = fetch_item(connection, item.queue_id)
            if updated and updated.status == QueueStatus.RUNNING and updated.lease_owner == owner:
                claimed.append(updated)
        connection.commit()
        return claimed
    finally:
        connection.close()


def heartbeat_item(db_path: Path, queue_id: str, *, owner: str, lease_seconds: float = 3600.0) -> QueueItem:
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    now = utc_timestamp()
    lease_expires_at = datetime.utcfromtimestamp(time.time() + lease_seconds).replace(microsecond=0).isoformat() + "Z"
    connection = connect(db_path)
    initialize_db(connection)
    try:
        item = fetch_item(connection, queue_id)
        if item is None:
            raise FileNotFoundError(f"queue item does not exist: {queue_id}")
        if item.status != QueueStatus.RUNNING or item.lease_owner != owner:
            raise ValueError(f"queue item {queue_id} is not leased by {owner}")
        checkpoint = {**item.checkpoint, "heartbeat_at": now, "lease_expires_at": lease_expires_at}
        connection.execute(
            """
            UPDATE run_queue
            SET updated_at = ?, lease_expires_at = ?, checkpoint_json = ?
            WHERE queue_id = ?
            """,
            (now, lease_expires_at, json.dumps(checkpoint, ensure_ascii=False), queue_id),
        )
        connection.commit()
        updated = fetch_item(connection, queue_id)
        if updated is None:
            raise RuntimeError(f"failed to heartbeat queue item {queue_id}")
        return updated
    finally:
        connection.close()


def infer_result_state_path(result: dict[str, Any]) -> str | None:
    for key in ["state_path", "source_state_path", "sweep_state_path", "optimization_state_path", "mission_state_path"]:
        value = result.get(key)
        if value:
            return str(value)
    run_dir = (
        result.get("run_dir")
        or result.get("supervisor_dir")
        or result.get("convergence_dir")
        or result.get("mission_dir")
        or result.get("agent_dir")
    )
    if run_dir:
        for name in [
            "state.json",
            "supervisor_state.json",
            "mission_state.json",
            "autonomous_devsim_agent_state.json",
            "sentaurus_state.json",
            "sweep_state.json",
            "optimization_state.json",
        ]:
            candidate = Path(run_dir) / name
            if candidate.exists():
                return str(candidate.resolve())
    return None


def default_runner_registry() -> dict[str, Runner]:
    from tcad_agent.adaptive_optimizer import AdaptiveOptimizationRequest, run_adaptive_optimization
    from tcad_agent.agent_cockpit import generate_agent_cockpit
    from tcad_agent.agent_soak import AgentSoakRequest, run_agent_soak
    from tcad_agent.autonomous_devsim_agent import AutonomousDevsimRequest, run_autonomous_devsim_agent
    from tcad_agent.conclusion import generate_experiment_conclusion
    from tcad_agent.engineering_objectives import (
        EngineeringConstraint,
        EngineeringObjective,
        evaluate_engineering_objectives,
    )
    from tcad_agent.industrial_external_runner import IndustrialExternalRunnerRequest, run_industrial_external_runner
    from tcad_agent.mesh_convergence import MeshConvergenceRequest, run_mesh_convergence
    from tcad_agent.mission_agent import run_mission_agent
    from tcad_agent.multidim_optimizer import MultiDimOptimizationRequest, run_multidim_optimization
    from tcad_agent.mutation_schema_agent import MutationSchemaExtensionRequest, run_mutation_schema_extension
    from tcad_agent.parameter_sweep import ParameterSweepRequest, run_parameter_sweep
    from tcad_agent.physical_benchmark import run_physical_benchmark
    from tcad_agent.power_mosfet_signoff import PowerMOSFETSignoffRequest, run_power_mosfet_signoff
    from tcad_agent.dashboard import generate_experiment_dashboard
    from tcad_agent.reporting import generate_experiment_report
    from tcad_agent.schottky_calibration import SchottkyCalibrationRequest, run_schottky_calibration
    from tcad_agent.sentaurus import SentaurusRunRequest, run_sentaurus
    from tcad_agent.sentaurus_lineage import SentaurusLineageArchiveRequest, build_sentaurus_lineage_archive
    from tcad_agent.sentaurus_mutation_effect import SentaurusMutationEffectRequest, analyze_sentaurus_mutation_effect
    from tcad_agent.sentaurus_patch_planner import SentaurusPatchPlannerRequest, plan_sentaurus_patches
    from tcad_agent.sentaurus_patch_refiner import SentaurusPatchRefinerRequest, build_sentaurus_patch_refinement_plan
    from tcad_agent.sentaurus_preflight import SentaurusPreflightRequest, run_sentaurus_preflight
    from tcad_agent.sentaurus_replay import SentaurusReplayRequest, run_sentaurus_replay
    from tcad_agent.golden_curve import GoldenCurveComparisonRequest, run_golden_curve_comparison
    from tcad_agent.supervisor import run_supervisor
    from tcad_agent.task_spec import TaskSpec, load_task_spec
    from tcad_agent.tool_convergence import ToolConvergenceRequest, run_tool_convergence
    from tcad_agent.user_deck_runner import UserDeckRunRequest, run_user_deck
    from tcad_agent.tools.diode_breakdown import DiodeBreakdownRequest, run_diode_breakdown_sweep
    from tcad_agent.tools.extended_device_sweep import ExtendedDeviceRequest, run_extended_device_sweep
    from tcad_agent.tools.mos_capacitor_cv import MOSCapacitorCVRequest, run_mos_capacitor_cv_sweep
    from tcad_agent.tools.mosfet_2d_id import MOSFET2DIDRequest, run_mosfet_2d_id_sweep
    from tcad_agent.tools.pn_junction_iv import PNJunctionIVRequest, run_pn_junction_iv_sweep

    def request_bool(request: dict[str, Any], key: str, default: bool) -> bool:
        value = request.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def mission_use_llm_decomposer(request: dict[str, Any]) -> bool:
        if "use_llm_decomposer" in request:
            return request_bool(request, "use_llm_decomposer", True)
        if "use_llm" in request:
            return request_bool(request, "use_llm", True)
        return True

    def mission_allow_llm_fallback(request: dict[str, Any]) -> bool:
        if "allow_llm_fallback" in request:
            return request_bool(request, "allow_llm_fallback", True)
        if "no_llm_fallback" in request:
            return not request_bool(request, "no_llm_fallback", False)
        return True

    def task_spec_from_request(request: dict[str, Any]) -> TaskSpec:
        if request.get("base_task_path"):
            return load_task_spec(Path(str(request["base_task_path"])))
        if request.get("base_spec"):
            return TaskSpec.model_validate(request["base_spec"])
        raise ValueError("queue item requires base_task_path or base_spec")

    def supervisor_runner(request: dict[str, Any]) -> dict[str, Any]:
        goal_text = str(request.get("goal_text") or request.get("goal") or "")
        if not goal_text:
            raise ValueError("supervisor queue item requires goal_text")
        supervisor_root = request.get("supervisor_root")
        state = run_supervisor(
            goal_text,
            supervisor_id=request.get("supervisor_id"),
            supervisor_root=Path(supervisor_root) if supervisor_root else None,
            execute=bool(request.get("execute", True)),
            resume=bool(request.get("resume", False)),
            max_cycles=int(request.get("max_cycles", 3)),
            use_agent_policy=request_bool(request, "use_agent_policy", True),
        )
        return state.model_dump(mode="json")

    def mission_runner(request: dict[str, Any]) -> dict[str, Any]:
        goal_text = str(request.get("goal_text") or request.get("goal") or "")
        if not goal_text:
            raise ValueError("mission_agent queue item requires goal_text")
        mission_root = request.get("mission_root")
        state = run_mission_agent(
            goal_text,
            mission_id=request.get("mission_id"),
            mission_root=Path(mission_root) if mission_root else None,
            execute=bool(request.get("execute", True)),
            resume=bool(request.get("resume", False)),
            max_cycles=int(request.get("max_cycles", 8)),
            supervisor_max_cycles=int(request.get("supervisor_max_cycles", 3)),
            use_llm_decomposer=mission_use_llm_decomposer(request),
            allow_llm_fallback=mission_allow_llm_fallback(request),
        )
        return state.model_dump(mode="json")

    def extended_device_runner(request: dict[str, Any]) -> dict[str, Any]:
        return run_extended_device_sweep(ExtendedDeviceRequest.model_validate(request)).model_dump(mode="json")

    def extended_device_alias(device_type: str, fidelity: str) -> Runner:
        def runner(request: dict[str, Any]) -> dict[str, Any]:
            payload = {**request, "device_type": device_type, "fidelity": fidelity}
            return extended_device_runner(payload)

        return runner

    def schottky_calibration_runner(request: dict[str, Any]) -> dict[str, Any]:
        return run_schottky_calibration(SchottkyCalibrationRequest.model_validate(request)).model_dump(mode="json")

    def golden_curve_runner(request: dict[str, Any]) -> dict[str, Any]:
        return run_golden_curve_comparison(GoldenCurveComparisonRequest.model_validate(request)).model_dump(mode="json")

    def parameter_sweep_runner(request: dict[str, Any]) -> dict[str, Any]:
        return run_parameter_sweep(
            task_spec_from_request(request),
            ParameterSweepRequest.model_validate(request),
        ).model_dump(mode="json")

    def adaptive_optimizer_runner(request: dict[str, Any]) -> dict[str, Any]:
        return run_adaptive_optimization(
            task_spec_from_request(request),
            AdaptiveOptimizationRequest.model_validate(request),
        ).model_dump(mode="json")

    def multidim_optimizer_runner(request: dict[str, Any]) -> dict[str, Any]:
        return run_multidim_optimization(
            task_spec_from_request(request),
            MultiDimOptimizationRequest.model_validate(request),
        ).model_dump(mode="json")

    def mesh_convergence_runner(request: dict[str, Any]) -> dict[str, Any]:
        return run_mesh_convergence(
            task_spec_from_request(request),
            MeshConvergenceRequest.model_validate(request),
        ).model_dump(mode="json")

    def engineering_objectives_runner(request: dict[str, Any]) -> dict[str, Any]:
        source = request.get("source") or request.get("state") or request.get("source_state_path")
        if not source:
            raise ValueError("engineering objective queue item requires source/state")
        objectives = [
            EngineeringObjective.model_validate(item)
            for item in request.get("objectives", [])
        ] or None
        constraints = [
            EngineeringConstraint.model_validate(item)
            for item in request.get("constraints", [])
        ] or None
        output_path = Path(str(request["output_path"])) if request.get("output_path") else None
        return evaluate_engineering_objectives(
            Path(str(source)),
            objectives=objectives,
            constraints=constraints,
            output_path=output_path,
        ).model_dump(mode="json")

    def physical_benchmark_runner(request: dict[str, Any]) -> dict[str, Any]:
        source = request.get("source") or request.get("state") or request.get("source_state_path")
        if not source:
            raise ValueError("physical benchmark queue item requires source/state")
        output_path = Path(str(request["output_path"])) if request.get("output_path") else None
        return run_physical_benchmark(Path(str(source)), output_path).model_dump(mode="json")

    def experiment_report_runner(request: dict[str, Any]) -> dict[str, Any]:
        source = request.get("source") or request.get("state") or request.get("source_state_path")
        if not source:
            raise ValueError("experiment report queue item requires source/state")
        output_path = Path(str(request["output_path"])) if request.get("output_path") else None
        return generate_experiment_report(Path(str(source)), output_path).model_dump(mode="json")

    def experiment_dashboard_runner(request: dict[str, Any]) -> dict[str, Any]:
        source = request.get("source") or request.get("state") or request.get("source_state_path")
        if not source:
            raise ValueError("experiment dashboard queue item requires source/state")
        output_path = Path(str(request["output_path"])) if request.get("output_path") else None
        return generate_experiment_dashboard(Path(str(source)), output_path).model_dump(mode="json")

    def experiment_conclusion_runner(request: dict[str, Any]) -> dict[str, Any]:
        source = request.get("source") or request.get("state") or request.get("source_state_path")
        if not source:
            raise ValueError("experiment conclusion queue item requires source/state")
        output_path = Path(str(request["output_path"])) if request.get("output_path") else None
        return generate_experiment_conclusion(Path(str(source)), output_path).model_dump(mode="json")

    def agent_cockpit_runner(request: dict[str, Any]) -> dict[str, Any]:
        source = request.get("source") or request.get("state") or request.get("source_state_path")
        if not source:
            raise ValueError("agent cockpit queue item requires source/state")
        output_path = Path(str(request["output_path"])) if request.get("output_path") else None
        return generate_agent_cockpit(Path(str(source)), output_path).model_dump(mode="json")

    registry: dict[str, Runner] = {
        "supervisor": supervisor_runner,
        "mission_agent": mission_runner,
        "pn_junction_iv_sweep": lambda request: result_to_dict(
            run_pn_junction_iv_sweep(PNJunctionIVRequest.model_validate(request))
        ),
        "mos_capacitor_cv_sweep": lambda request: result_to_dict(
            run_mos_capacitor_cv_sweep(MOSCapacitorCVRequest.model_validate(request))
        ),
        "diode_breakdown_leakage_sweep": lambda request: result_to_dict(
            run_diode_breakdown_sweep(DiodeBreakdownRequest.model_validate(request))
        ),
        "mosfet_2d_id_sweep": lambda request: result_to_dict(
            run_mosfet_2d_id_sweep(MOSFET2DIDRequest.model_validate(request))
        ),
        "extended_device_sweep": extended_device_runner,
        "power_mosfet_bv_ron_1d_runner": extended_device_alias("power_mosfet_bv_ron", "physics_1d"),
        "power_mosfet_bv_ron_2d_runner": extended_device_alias("power_mosfet_bv_ron", "devsim_2d_field_plate"),
        "gan_hemt_id_bv_runner": extended_device_alias("gan_hemt_id_bv", "physics_1d"),
        "sic_power_diode_bv_leakage_runner": extended_device_alias("sic_power_diode_bv_leakage", "physics_1d"),
        "igbt_output_turnoff_runner": extended_device_alias("igbt_output_turnoff", "physics_1d"),
        "industrial_external_tcad_runner": lambda request: result_to_dict(
            run_industrial_external_runner(IndustrialExternalRunnerRequest.model_validate(request))
        ),
        "schottky_iv_calibration": schottky_calibration_runner,
        "golden_curve_comparison": golden_curve_runner,
        "tool_convergence": lambda request: result_to_dict(
            run_tool_convergence(ToolConvergenceRequest.model_validate(request))
        ),
        "mesh_convergence": mesh_convergence_runner,
        "parameter_sweep": parameter_sweep_runner,
        "adaptive_optimizer": adaptive_optimizer_runner,
        "multidim_optimizer": multidim_optimizer_runner,
        "engineering_objectives": engineering_objectives_runner,
        "physical_benchmark": physical_benchmark_runner,
        "power_mosfet_signoff": lambda request: result_to_dict(
            run_power_mosfet_signoff(PowerMOSFETSignoffRequest.model_validate(request))
        ),
        "experiment_report": experiment_report_runner,
        "experiment_dashboard": experiment_dashboard_runner,
        "agent_cockpit": agent_cockpit_runner,
        "experiment_conclusion": experiment_conclusion_runner,
        "sentaurus_run": lambda request: result_to_dict(
            run_sentaurus(SentaurusRunRequest.model_validate(request))
        ),
        "sentaurus_preflight": lambda request: result_to_dict(
            run_sentaurus_preflight(SentaurusPreflightRequest.model_validate(request))
        ),
        "sentaurus_replay": lambda request: result_to_dict(
            run_sentaurus_replay(SentaurusReplayRequest.model_validate(request))
        ),
        "sentaurus_mutation_effect_analyzer": lambda request: result_to_dict(
            analyze_sentaurus_mutation_effect(SentaurusMutationEffectRequest.model_validate(request))
        ),
        "sentaurus_patch_planner": lambda request: result_to_dict(
            plan_sentaurus_patches(SentaurusPatchPlannerRequest.model_validate(request))
        ),
        "sentaurus_patch_refiner": lambda request: result_to_dict(
            build_sentaurus_patch_refinement_plan(SentaurusPatchRefinerRequest.model_validate(request))
        ),
        "sentaurus_lineage_archive": lambda request: result_to_dict(
            build_sentaurus_lineage_archive(SentaurusLineageArchiveRequest.model_validate(request))
        ),
        "mutation_schema_agent": lambda request: result_to_dict(
            run_mutation_schema_extension(MutationSchemaExtensionRequest.model_validate(request))
        ),
        "user_deck_execution": lambda request: run_user_deck(UserDeckRunRequest.model_validate(request)),
    }
    registry["autonomous_devsim_agent"] = lambda request: run_autonomous_devsim_agent(
        AutonomousDevsimRequest.model_validate(request),
        runner_registry=registry,
    ).model_dump(mode="json")
    registry["agent_soak"] = lambda request: run_agent_soak(
        AgentSoakRequest.model_validate(request),
        runner_registry=registry,
    ).model_dump(mode="json")
    return registry


def result_to_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return {"value": result}


def complete_item(
    db_path: Path,
    item: QueueItem,
    *,
    result: dict[str, Any],
    result_state_path: str | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> QueueItem:
    now = utc_timestamp()
    merged_checkpoint = {**item.checkpoint, **(checkpoint or {}), "completed_at": now}
    connection = connect(db_path)
    initialize_db(connection)
    try:
        current = fetch_item(connection, item.queue_id)
        if current and current.status == QueueStatus.CANCELLED:
            return current
        connection.execute(
            """
            UPDATE run_queue
            SET status = ?, updated_at = ?, completed_at = ?, lease_owner = NULL,
                lease_expires_at = NULL, checkpoint_json = ?, result_json = ?,
                result_state_path = ?, failure_reason = NULL
            WHERE queue_id = ?
            """,
            (
                QueueStatus.COMPLETED.value,
                now,
                now,
                json.dumps(merged_checkpoint, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
                result_state_path,
                item.queue_id,
            ),
        )
        connection.commit()
        updated = fetch_item(connection, item.queue_id)
        if updated is None:
            raise RuntimeError(f"failed to complete queue item {item.queue_id}")
        return updated
    finally:
        connection.close()


def pause_item_with_result(
    db_path: Path,
    item: QueueItem,
    *,
    result: dict[str, Any],
    result_state_path: str | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> QueueItem:
    now = utc_timestamp()
    merged_checkpoint = {**item.checkpoint, **(checkpoint or {}), "paused_at": now}
    connection = connect(db_path)
    initialize_db(connection)
    try:
        current = fetch_item(connection, item.queue_id)
        if current and current.status == QueueStatus.CANCELLED:
            return current
        connection.execute(
            """
            UPDATE run_queue
            SET status = ?, updated_at = ?, lease_owner = NULL, lease_expires_at = NULL,
                checkpoint_json = ?, result_json = ?, result_state_path = ?, failure_reason = NULL
            WHERE queue_id = ?
            """,
            (
                QueueStatus.PAUSED.value,
                now,
                json.dumps(merged_checkpoint, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
                result_state_path,
                item.queue_id,
            ),
        )
        connection.commit()
        updated = fetch_item(connection, item.queue_id)
        if updated is None:
            raise RuntimeError(f"failed to pause queue item {item.queue_id}")
        return updated
    finally:
        connection.close()


def fail_item(
    db_path: Path,
    item: QueueItem,
    *,
    failure_reason: str,
    result: dict[str, Any] | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> QueueItem:
    now = utc_timestamp()
    merged_checkpoint = {**item.checkpoint, **(checkpoint or {}), "failed_at": now}
    connection = connect(db_path)
    initialize_db(connection)
    try:
        current = fetch_item(connection, item.queue_id)
        if current and current.status == QueueStatus.CANCELLED:
            return current
        connection.execute(
            """
            UPDATE run_queue
            SET status = ?, updated_at = ?, completed_at = ?, lease_owner = NULL,
                lease_expires_at = NULL, checkpoint_json = ?, result_json = ?,
                failure_reason = ?
            WHERE queue_id = ?
            """,
            (
                QueueStatus.FAILED.value,
                now,
                now,
                json.dumps(merged_checkpoint, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False) if result is not None else None,
                failure_reason,
                item.queue_id,
            ),
        )
        connection.commit()
        updated = fetch_item(connection, item.queue_id)
        if updated is None:
            raise RuntimeError(f"failed to mark queue item {item.queue_id} failed")
        return updated
    finally:
        connection.close()


def execute_item(
    db_path: Path,
    item: QueueItem,
    *,
    registry: dict[str, Runner] | None = None,
) -> QueueItem:
    runners = registry or default_runner_registry()
    runner = runners.get(item.tool_name)
    if runner is None:
        return fail_item(db_path, item, failure_reason=f"unknown queued tool: {item.tool_name}")
    if item.budget_seconds is not None and item.budget_seconds <= 0:
        return fail_item(db_path, item, failure_reason="budget_seconds exhausted before execution")

    tool_request = dict(item.request)
    tool_request.setdefault("queue_id", item.queue_id)
    if item.tool_name == "autonomous_devsim_agent":
        tool_request.setdefault("agent_id", item.queue_id)
        control_dir = autonomous_agent_control_dir(item)
        if control_dir:
            tool_request.setdefault("agent_root", str(control_dir.parent))
            tool_request.setdefault("cancel_file", str(control_dir / "cancel.requested"))
            tool_request.setdefault("heartbeat_path", str(control_dir / "heartbeat.json"))
    if item.tool_name == "agent_soak":
        tool_request.setdefault("soak_id", item.queue_id)
        control_dir = autonomous_agent_control_dir(item)
        if control_dir:
            tool_request.setdefault("soak_root", str(control_dir.parent))
            tool_request.setdefault("cancel_file", str(control_dir / "cancel.requested"))
            tool_request.setdefault("heartbeat_path", str(control_dir / "agent_soak_heartbeat.json"))

    started = time.monotonic()
    try:
        raw_result = runner(tool_request)
        result = result_to_dict(raw_result)
    except Exception as exc:
        elapsed = time.monotonic() - started
        return fail_item(
            db_path,
            item,
            failure_reason=str(exc),
            checkpoint={"elapsed_seconds": elapsed},
        )

    elapsed = time.monotonic() - started
    checkpoint = {"elapsed_seconds": elapsed}
    result_state_path = infer_result_state_path(result)
    if item.budget_seconds is not None and elapsed > item.budget_seconds:
        return fail_item(
            db_path,
            item,
            failure_reason=f"budget_seconds exceeded: elapsed={elapsed:.3f}, budget={item.budget_seconds:.3f}",
            result=result,
            checkpoint=checkpoint,
        )
    if str(result.get("status") or "") == "waiting_for_user":
        return pause_item_with_result(
            db_path,
            item,
            result=result,
            result_state_path=result_state_path,
            checkpoint={**checkpoint, "paused_reason": "waiting_for_user"},
        )
    if str(result.get("status") or "") == "cancelled":
        return fail_item(
            db_path,
            item,
            failure_reason="cancelled",
            result=result,
            checkpoint=checkpoint,
        )
    return complete_item(db_path, item, result=result, result_state_path=result_state_path, checkpoint=checkpoint)


def run_queue_worker(
    db_path: Path,
    *,
    owner: str | None = None,
    concurrency: int = 1,
    lease_seconds: float = 3600.0,
    max_items: int | None = None,
    registry: dict[str, Runner] | None = None,
) -> QueueWorkerResult:
    actual_owner = owner or default_worker_owner()
    limit = max(1, concurrency)
    if max_items is not None:
        limit = min(limit, max_items)
    claimed = claim_next_items(db_path, owner=actual_owner, limit=limit, lease_seconds=lease_seconds)
    result = QueueWorkerResult(db_path=str(db_path), owner=actual_owner, claimed=len(claimed))
    for item in claimed:
        latest = get_item(db_path, item.queue_id)
        if latest is None or latest.status != QueueStatus.RUNNING:
            result.skipped += 1
            continue
        finished = execute_item(db_path, latest, registry=registry)
        if finished.status == QueueStatus.COMPLETED:
            result.completed += 1
        elif finished.status == QueueStatus.FAILED:
            result.failed += 1
        else:
            result.skipped += 1
        result.items.append(finished.model_dump(mode="json"))
    return result


def run_queue_daemon(
    db_path: Path,
    *,
    owner: str | None = None,
    concurrency: int = 1,
    lease_seconds: float = 3600.0,
    poll_interval_seconds: float = 5.0,
    max_loops: int | None = None,
    max_idle_loops: int | None = None,
    stop_file: Path | None = None,
    registry: dict[str, Runner] | None = None,
) -> QueueDaemonResult:
    if poll_interval_seconds < 0:
        raise ValueError("poll_interval_seconds must be non-negative")
    actual_owner = owner or default_worker_owner()
    result = QueueDaemonResult(db_path=str(db_path), owner=actual_owner, stopped_by="max_loops")
    while True:
        if stop_file is not None and stop_file.exists():
            result.stopped_by = "stop_file"
            break
        if max_loops is not None and result.loops >= max_loops:
            result.stopped_by = "max_loops"
            break
        if max_idle_loops is not None and result.idle_loops >= max_idle_loops:
            result.stopped_by = "idle"
            break

        worker = run_queue_worker(
            db_path,
            owner=actual_owner,
            concurrency=concurrency,
            lease_seconds=lease_seconds,
            registry=registry,
        )
        result.loops += 1
        result.claimed += worker.claimed
        result.completed += worker.completed
        result.failed += worker.failed
        result.skipped += worker.skipped
        result.worker_results.append(worker.model_dump(mode="json"))
        if worker.claimed == 0:
            result.idle_loops += 1
            if poll_interval_seconds:
                time.sleep(poll_interval_seconds)
        else:
            result.idle_loops = 0
    return result


def parse_json_object(raw: str | None) -> dict[str, object]:
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("JSON value must be an object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Durable TCAD run queue.")
    parser.add_argument("--db", type=Path, default=default_queue_db_path())
    subparsers = parser.add_subparsers(dest="command", required=True)

    enqueue = subparsers.add_parser("enqueue", help="Enqueue a TCAD run.")
    enqueue.add_argument("--queue-id", default=None)
    enqueue.add_argument("--tool", default="supervisor")
    enqueue.add_argument("--request-json", default=None)
    enqueue.add_argument("--goal", default=None, help="Convenience goal_text for supervisor items.")
    enqueue.add_argument("--priority", type=int, default=0)
    enqueue.add_argument("--tag", action="append", default=[])
    enqueue.add_argument("--max-attempts", type=int, default=1)
    enqueue.add_argument("--budget-seconds", type=float, default=None)
    enqueue.add_argument("--budget-cases", type=int, default=None)

    list_parser = subparsers.add_parser("list", help="List queued/running/history items.")
    list_parser.add_argument("--status", choices=[status.value for status in QueueStatus], default=None)
    list_parser.add_argument("--tool", default=None)
    list_parser.add_argument("--limit", type=int, default=50)

    show = subparsers.add_parser("show", help="Show one queue item.")
    show.add_argument("queue_id")

    worker = subparsers.add_parser("worker", help="Claim and execute queued runs.")
    worker.add_argument("--owner", default=None)
    worker.add_argument("--concurrency", type=int, default=1)
    worker.add_argument("--lease-seconds", type=float, default=3600.0)
    worker.add_argument("--max-items", type=int, default=None)

    daemon = subparsers.add_parser("daemon", help="Run a polling queue worker loop.")
    daemon.add_argument("--owner", default=None)
    daemon.add_argument("--concurrency", type=int, default=1)
    daemon.add_argument("--lease-seconds", type=float, default=3600.0)
    daemon.add_argument("--poll-interval-seconds", type=float, default=5.0)
    daemon.add_argument("--max-loops", type=int, default=None)
    daemon.add_argument("--max-idle-loops", type=int, default=None)
    daemon.add_argument("--stop-file", type=Path, default=None)

    claim = subparsers.add_parser("claim", help="Claim queued runs without executing them.")
    claim.add_argument("--owner", default=None)
    claim.add_argument("--limit", type=int, default=1)
    claim.add_argument("--lease-seconds", type=float, default=3600.0)

    heartbeat = subparsers.add_parser("heartbeat", help="Extend a running item's lease.")
    heartbeat.add_argument("queue_id")
    heartbeat.add_argument("--owner", required=True)
    heartbeat.add_argument("--lease-seconds", type=float, default=3600.0)

    pause = subparsers.add_parser("pause", help="Pause a queued or running item.")
    pause.add_argument("queue_id")

    resume = subparsers.add_parser("resume", help="Resume a paused item.")
    resume.add_argument("queue_id")

    cancel = subparsers.add_parser("cancel", help="Cancel a non-terminal item.")
    cancel.add_argument("queue_id")

    subparsers.add_parser("recover", help="Recover expired running leases.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "enqueue":
            request = parse_json_object(args.request_json)
            if args.goal:
                request = {**request, "goal_text": args.goal}
            item = enqueue_run(
                args.db,
                tool_name=args.tool,
                request=request,
                queue_id=args.queue_id,
                priority=args.priority,
                tags=args.tag,
                max_attempts=args.max_attempts,
                budget_seconds=args.budget_seconds,
                budget_cases=args.budget_cases,
            )
            output = item.model_dump(mode="json")
        elif args.command == "list":
            output = {
                "items": list_items(
                    args.db,
                    status=args.status,
                    tool_name=args.tool,
                    limit=args.limit,
                )
            }
        elif args.command == "show":
            item = get_item(args.db, args.queue_id)
            output = item.model_dump(mode="json") if item else {"error": f"queue item not found: {args.queue_id}"}
        elif args.command == "worker":
            output = run_queue_worker(
                args.db,
                owner=args.owner,
                concurrency=args.concurrency,
                lease_seconds=args.lease_seconds,
                max_items=args.max_items,
            ).model_dump(mode="json")
        elif args.command == "daemon":
            output = run_queue_daemon(
                args.db,
                owner=args.owner,
                concurrency=args.concurrency,
                lease_seconds=args.lease_seconds,
                poll_interval_seconds=args.poll_interval_seconds,
                max_loops=args.max_loops,
                max_idle_loops=args.max_idle_loops,
                stop_file=args.stop_file,
            ).model_dump(mode="json")
        elif args.command == "claim":
            owner = args.owner or default_worker_owner()
            output = {
                "owner": owner,
                "items": [
                    item.model_dump(mode="json")
                    for item in claim_next_items(
                        args.db,
                        owner=owner,
                        limit=args.limit,
                        lease_seconds=args.lease_seconds,
                    )
                ],
            }
        elif args.command == "heartbeat":
            output = heartbeat_item(
                args.db,
                args.queue_id,
                owner=args.owner,
                lease_seconds=args.lease_seconds,
            ).model_dump(mode="json")
        elif args.command == "pause":
            output = pause_item(args.db, args.queue_id).model_dump(mode="json")
        elif args.command == "resume":
            output = resume_item(args.db, args.queue_id).model_dump(mode="json")
        elif args.command == "cancel":
            output = cancel_item(args.db, args.queue_id).model_dump(mode="json")
        elif args.command == "recover":
            output = recover_stale_items(args.db)
        else:
            raise ValueError(f"unknown command: {args.command}")
        print(json.dumps(output, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"status": "failed", "failure_reason": str(exc)}, indent=2, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
