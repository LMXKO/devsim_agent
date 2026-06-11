from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tcad_agent.task_spec import PROJECT_ROOT


SCHEMA_VERSION = 1


@dataclass
class ExperimentRecord:
    experiment_id: str
    kind: str
    status: str | None
    state_path: str
    created_at: str | None = None
    updated_at: str | None = None
    objective_value: float | None = None
    best_axis_path: str | None = None
    best_axis_value: float | None = None
    quality_status: str | None = None
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_index_db_path() -> Path:
    return PROJECT_ROOT / "runs" / "experiment_index.sqlite"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        CREATE TABLE IF NOT EXISTS experiments (
          experiment_id TEXT NOT NULL,
          kind TEXT NOT NULL,
          status TEXT,
          state_path TEXT NOT NULL UNIQUE,
          created_at TEXT,
          updated_at TEXT,
          indexed_at TEXT NOT NULL,
          objective_value REAL,
          best_axis_path TEXT,
          best_axis_value REAL,
          quality_status TEXT,
          failure_reason TEXT,
          PRIMARY KEY (experiment_id, kind)
        );
        CREATE INDEX IF NOT EXISTS idx_experiments_kind ON experiments(kind);
        CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
        CREATE INDEX IF NOT EXISTS idx_experiments_objective ON experiments(objective_value);
        """
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )
    connection.commit()


def float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def axis_from_best(best: dict[str, Any] | None) -> tuple[str | None, float | None]:
    if not best:
        return None, None
    if "value" in best:
        return None, float_or_none(best.get("value"))
    values = best.get("values") or {}
    if not values:
        return None, None
    key = next(iter(values))
    return key, float_or_none(values.get(key))


def record_from_sweep(path: Path, state: dict[str, Any]) -> ExperimentRecord:
    best = state.get("best_case") or {}
    axis_path, axis_value = axis_from_best(best)
    if not axis_path:
        axes = state.get("axes") or []
        axis_path = axes[0].get("path") if axes else None
    return ExperimentRecord(
        experiment_id=str(state.get("sweep_id")),
        kind="parameter_sweep",
        status=state.get("status"),
        state_path=str(path.resolve()),
        created_at=state.get("created_at"),
        updated_at=state.get("updated_at"),
        objective_value=float_or_none(best.get("objective_value")),
        best_axis_path=axis_path,
        best_axis_value=axis_value,
        quality_status=best.get("quality_status"),
        failure_reason=state.get("failure_reason"),
    )


def record_from_optimization(path: Path, state: dict[str, Any]) -> ExperimentRecord:
    best = state.get("best_observation") or {}
    axis = state.get("axis") or {}
    _, axis_value = axis_from_best(best)
    return ExperimentRecord(
        experiment_id=str(state.get("optimize_id")),
        kind="adaptive_optimization",
        status=state.get("status"),
        state_path=str(path.resolve()),
        created_at=state.get("created_at"),
        updated_at=state.get("updated_at"),
        objective_value=float_or_none(best.get("objective_value")),
        best_axis_path=axis.get("path"),
        best_axis_value=axis_value,
        quality_status=best.get("quality_status"),
        failure_reason=state.get("failure_reason"),
    )


def record_from_multidim_optimization(path: Path, state: dict[str, Any]) -> ExperimentRecord:
    best = state.get("best_observation") or {}
    values = best.get("values") or {}
    axes = state.get("axes") or []
    axis_paths = [axis.get("path") for axis in axes if axis.get("path")]
    first_axis_path = axis_paths[0] if axis_paths else None
    first_axis_value = float_or_none(values.get(first_axis_path)) if first_axis_path else None
    return ExperimentRecord(
        experiment_id=str(state.get("optimize_id")),
        kind="multidim_optimization",
        status=state.get("status"),
        state_path=str(path.resolve()),
        created_at=state.get("created_at"),
        updated_at=state.get("updated_at"),
        objective_value=float_or_none(best.get("objective_value")),
        best_axis_path=", ".join(axis_paths) if axis_paths else None,
        best_axis_value=first_axis_value,
        quality_status=best.get("quality_status"),
        failure_reason=state.get("failure_reason"),
    )


def record_from_task(path: Path, state: dict[str, Any]) -> ExperimentRecord:
    metrics = state.get("final_quality_report", {}).get("metrics", {}) if state.get("final_quality_report") else {}
    return ExperimentRecord(
        experiment_id=str(state.get("task_id")),
        kind="task_run",
        status=state.get("status"),
        state_path=str(path.resolve()),
        created_at=state.get("created_at"),
        updated_at=state.get("updated_at"),
        objective_value=float_or_none(metrics.get("final_total_current_a")),
        quality_status=(state.get("final_quality_report") or {}).get("status"),
        failure_reason=state.get("failure_reason"),
    )


def tool_objective_value(state: dict[str, Any]) -> float | None:
    metrics = (state.get("quality_report") or {}).get("metrics") or {}
    for key in [
        "ion_ioff_ratio",
        "vth_at_threshold_current_v",
        "max_abs_drain_current_a",
        "final_abs_drain_current_a",
        "barrier_height_ev",
        "best_rmse_log_current_dec",
        "golden_curve_rmse_log_dec",
        "current_gain_beta",
        "breakdown_voltage_v",
        "relative_delta",
        "leakage_abs_current_at_target_a",
        "breakdown_voltage_at_threshold_v",
        "final_total_current_a",
        "final_capacitance_f_per_cm2",
        "max_abs_current_a",
    ]:
        value = float_or_none(metrics.get(key))
        if value is not None:
            return value
    return None


def record_from_tool_state(path: Path, state: dict[str, Any]) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=str(
            state.get("run_id")
            or state.get("convergence_id")
            or state.get("comparison_id")
            or state.get("optimize_id")
            or state.get("sweep_id")
            or path.parent.name
        ),
        kind=str(state.get("tool_name")),
        status=state.get("status"),
        state_path=str(path.resolve()),
        created_at=state.get("created_at"),
        updated_at=state.get("updated_at"),
        objective_value=tool_objective_value(state),
        quality_status=(state.get("quality_report") or {}).get("status"),
        failure_reason=(state.get("checkpoint") or {}).get("last_failure_reason"),
    )


def record_from_benchmark(path: Path, state: dict[str, Any]) -> ExperimentRecord:
    counts = (state.get("summary") or {}).get("counts") or {}
    source = Path(str(state.get("source_state_path") or path))
    return ExperimentRecord(
        experiment_id=f"{source.parent.name}_{source.stem}_benchmark",
        kind="physical_benchmark",
        status=state.get("status"),
        state_path=str(path.resolve()),
        updated_at=(state.get("summary") or {}).get("generated_at"),
        objective_value=float_or_none(counts.get("error") or 0),
        quality_status=state.get("status"),
        failure_reason=state.get("failure_reason"),
    )


def record_from_engineering_objectives(path: Path, state: dict[str, Any]) -> ExperimentRecord:
    best = state.get("best_candidate") or {}
    return ExperimentRecord(
        experiment_id=f"{path.parent.name}_engineering_objectives",
        kind="engineering_objective_evaluation",
        status=state.get("status"),
        state_path=str(path.resolve()),
        objective_value=float_or_none(best.get("score")),
        quality_status="feasible" if best else "no_feasible_candidate",
        failure_reason=state.get("failure_reason"),
    )


def record_from_state(path: Path) -> ExperimentRecord | None:
    try:
        state = read_json(path)
    except (json.JSONDecodeError, OSError):
        return None
    tool_name = state.get("tool_name")
    if tool_name == "parameter_sweep":
        return record_from_sweep(path, state)
    if tool_name == "adaptive_optimizer":
        return record_from_optimization(path, state)
    if tool_name == "multidim_optimizer":
        return record_from_multidim_optimization(path, state)
    if tool_name == "tcad_task_runner":
        return record_from_task(path, state)
    if tool_name == "physical_benchmark":
        return record_from_benchmark(path, state)
    if tool_name == "engineering_objective_evaluation":
        return record_from_engineering_objectives(path, state)
    if tool_name in {
        "pn_junction_iv_sweep",
        "mos_capacitor_cv_sweep",
        "diode_breakdown_leakage_sweep",
        "mesh_convergence",
        "tool_convergence",
        "mosfet_2d_id_sweep",
        "extended_device_sweep",
        "schottky_iv_calibration",
        "golden_curve_comparison",
    }:
        return record_from_tool_state(path, state)
    return None


def discover_state_paths(root: Path) -> list[Path]:
    names = {
        "task_run_state.json",
        "sweep_state.json",
        "optimization_state.json",
        "state.json",
        "benchmark.json",
        "engineering_objectives.json",
    }
    return sorted(path for path in root.rglob("*.json") if path.name in names)


def upsert_record(connection: sqlite3.Connection, record: ExperimentRecord) -> None:
    connection.execute(
        """
        INSERT INTO experiments (
          experiment_id, kind, status, state_path, created_at, updated_at, indexed_at,
          objective_value, best_axis_path, best_axis_value, quality_status, failure_reason
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(state_path) DO UPDATE SET
          experiment_id=excluded.experiment_id,
          kind=excluded.kind,
          status=excluded.status,
          created_at=excluded.created_at,
          updated_at=excluded.updated_at,
          indexed_at=excluded.indexed_at,
          objective_value=excluded.objective_value,
          best_axis_path=excluded.best_axis_path,
          best_axis_value=excluded.best_axis_value,
          quality_status=excluded.quality_status,
          failure_reason=excluded.failure_reason
        """,
        (
            record.experiment_id,
            record.kind,
            record.status,
            record.state_path,
            record.created_at,
            record.updated_at,
            utc_timestamp(),
            record.objective_value,
            record.best_axis_path,
            record.best_axis_value,
            record.quality_status,
            record.failure_reason,
        ),
    )


def rebuild_index(root: Path, db_path: Path | None = None) -> dict[str, Any]:
    actual_db_path = db_path or default_index_db_path()
    connection = connect(actual_db_path)
    initialize_db(connection)
    connection.execute("DELETE FROM experiments")
    records: list[ExperimentRecord] = []
    for state_path in discover_state_paths(root):
        record = record_from_state(state_path)
        if record is None:
            continue
        records.append(record)
        upsert_record(connection, record)
    connection.commit()
    connection.close()
    return {
        "tool_name": "experiment_index",
        "status": "completed",
        "db_path": str(actual_db_path.resolve()),
        "root": str(root.resolve()),
        "records_indexed": len(records),
    }


def list_records(
    db_path: Path | None = None,
    *,
    kind: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    actual_db_path = db_path or default_index_db_path()
    connection = connect(actual_db_path)
    initialize_db(connection)
    clauses: list[str] = []
    values: list[Any] = []
    if kind:
        clauses.append("kind = ?")
        values.append(kind)
    if status:
        clauses.append("status = ?")
        values.append(status)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    rows = connection.execute(
        f"""
        SELECT experiment_id, kind, status, state_path, created_at, updated_at,
               objective_value, best_axis_path, best_axis_value, quality_status, failure_reason
        FROM experiments
        {where}
        ORDER BY COALESCE(updated_at, created_at, indexed_at) DESC
        LIMIT ?
        """,
        (*values, limit),
    ).fetchall()
    connection.close()
    return [dict(row) for row in rows]
