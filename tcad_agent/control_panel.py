from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tcad_agent.experiment_index import default_index_db_path, list_records, rebuild_index
from tcad_agent.llm_health import configured_llm_status
from tcad_agent.run_queue import default_queue_db_path, list_items
from tcad_agent.task_spec import PROJECT_ROOT


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def safe_list_queue(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    try:
        return list_items(db_path, limit=100)
    except Exception:
        return []


def safe_list_records(root: Path, db_path: Path, rebuild: bool) -> list[dict[str, Any]]:
    try:
        if rebuild or not db_path.exists():
            rebuild_index(root, db_path)
        return list_records(db_path, limit=100)
    except Exception:
        return []


def collect_benchmarks(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("benchmark.json")):
        data = read_json(path)
        if not data:
            continue
        rows.append(
            {
                "status": data.get("status"),
                "source_tool_name": data.get("source_tool_name"),
                "source_state_path": data.get("source_state_path"),
                "benchmark_path": str(path),
                "counts": (data.get("summary") or {}).get("counts") or {},
            }
        )
    return rows


def collect_validations(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("validation_state.json")):
        data = read_json(path)
        if not data:
            continue
        rows.append(
            {
                "validation_id": data.get("validation_id"),
                "status": data.get("status"),
                "queued_items": len(data.get("queued_items") or []),
                "benchmarks": len(data.get("benchmark_results") or []),
                "path": str(path),
                "failure_reason": data.get("failure_reason"),
            }
        )
    return rows


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("status") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def collect_control_panel_data(
    root: Path | None = None,
    *,
    queue_db_path: Path | None = None,
    index_db_path: Path | None = None,
    rebuild: bool = True,
) -> dict[str, Any]:
    actual_root = (root or PROJECT_ROOT / "runs").resolve()
    queue_db = queue_db_path or (actual_root / "run_queue.sqlite" if root else default_queue_db_path())
    index_db = index_db_path or (actual_root / "experiment_index.sqlite" if root else default_index_db_path())
    queue_rows = safe_list_queue(queue_db)
    records = safe_list_records(actual_root, index_db, rebuild)
    benchmarks = collect_benchmarks(actual_root)
    validations = collect_validations(actual_root)
    llm_status = configured_llm_status().model_dump(mode="json")
    return {
        "generated_at": utc_timestamp(),
        "root": str(actual_root),
        "queue_db_path": str(queue_db),
        "index_db_path": str(index_db),
        "llm_status": llm_status,
        "queue_status_counts": status_counts(queue_rows),
        "experiment_status_counts": status_counts(records),
        "benchmark_status_counts": status_counts(benchmarks),
        "validation_status_counts": status_counts(validations),
        "counts": {
            "queue_items": len(queue_rows),
            "experiment_records": len(records),
            "benchmarks": len(benchmarks),
            "validations": len(validations),
        },
        "queue_items": queue_rows,
        "experiment_records": records,
        "benchmarks": benchmarks,
        "validations": validations,
    }
