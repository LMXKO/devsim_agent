from __future__ import annotations

import html
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.experiment_index import default_index_db_path, list_records, rebuild_index
from tcad_agent.llm_health import configured_llm_status
from tcad_agent.run_queue import default_queue_db_path, list_items
from tcad_agent.task_spec import PROJECT_ROOT


class ControlPanelStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class ControlPanelResult(BaseModel):
    tool_name: str = "tcad_control_panel"
    status: ControlPanelStatus
    root: str
    output_dir: str
    html_path: str | None = None
    data_path: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    failure_reason: str | None = None


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


def esc(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return html.escape(str(value))


def table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="empty">No records.</p>'
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body_lines = []
    for row in rows:
        body_lines.append("<tr>" + "".join(f"<td>{esc(row.get(header))}</td>" for header in headers) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_lines)}</tbody></table>"


def render_html(data: dict[str, Any]) -> str:
    generated_at = esc(data.get("generated_at"))
    counts = data.get("counts") or {}
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TCAD Agent Control Panel</title>
  <style>
    :root {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; background: #f8fafc; }}
    body {{ margin: 0; }}
    header {{ background: #ffffff; border-bottom: 1px solid #d1d5db; padding: 16px 24px; }}
    main {{ padding: 20px 24px 40px; display: grid; gap: 20px; }}
    h1 {{ font-size: 20px; margin: 0 0 4px; letter-spacing: 0; }}
    h2 {{ font-size: 15px; margin: 0 0 10px; letter-spacing: 0; }}
    .meta {{ color: #4b5563; font-size: 13px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }}
    .metric {{ background: #ffffff; border: 1px solid #d1d5db; border-radius: 6px; padding: 10px 12px; }}
    .metric strong {{ display: block; font-size: 18px; }}
    section {{ background: #ffffff; border: 1px solid #d1d5db; border-radius: 6px; padding: 14px; overflow: auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ text-align: left; border-bottom: 1px solid #e5e7eb; padding: 7px 8px; vertical-align: top; }}
    th {{ color: #374151; background: #f3f4f6; position: sticky; top: 0; }}
    .empty {{ color: #6b7280; font-size: 13px; margin: 0; }}
  </style>
</head>
<body>
  <header>
    <h1>TCAD Agent Control Panel</h1>
    <div class="meta">Generated {generated_at} from {esc(data.get("root"))}</div>
  </header>
  <main>
    <div class="summary">
      <div class="metric"><span>Queue</span><strong>{esc(counts.get("queue_items", 0))}</strong></div>
      <div class="metric"><span>Experiments</span><strong>{esc(counts.get("experiment_records", 0))}</strong></div>
      <div class="metric"><span>Benchmarks</span><strong>{esc(counts.get("benchmarks", 0))}</strong></div>
      <div class="metric"><span>Validations</span><strong>{esc(counts.get("validations", 0))}</strong></div>
    </div>
    <section>
      <h2>Run Queue</h2>
      {table(["queue_id", "status", "tool_name", "priority", "attempts", "failure_reason", "result_state_path"], data.get("queue_items") or [])}
    </section>
    <section>
      <h2>Experiment Index</h2>
      {table(["experiment_id", "kind", "status", "quality_status", "objective_value", "state_path"], data.get("experiment_records") or [])}
    </section>
    <section>
      <h2>Physical Benchmarks</h2>
      {table(["status", "source_tool_name", "counts", "source_state_path", "benchmark_path"], data.get("benchmarks") or [])}
    </section>
    <section>
      <h2>Long-Run Validations</h2>
      {table(["validation_id", "status", "queued_items", "benchmarks", "failure_reason", "path"], data.get("validations") or [])}
    </section>
  </main>
</body>
</html>
"""


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


def generate_control_panel(
    root: Path | None = None,
    *,
    output_dir: Path | None = None,
    queue_db_path: Path | None = None,
    index_db_path: Path | None = None,
    rebuild: bool = True,
) -> ControlPanelResult:
    actual_root = (root or PROJECT_ROOT / "runs").resolve()
    actual_output = (output_dir or actual_root / "control_panel").resolve()
    html_path = actual_output / "index.html"
    data_path = actual_output / "control_panel.json"
    try:
        data = collect_control_panel_data(
            actual_root,
            queue_db_path=queue_db_path,
            index_db_path=index_db_path,
            rebuild=rebuild,
        )
        actual_output.mkdir(parents=True, exist_ok=True)
        data_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        html_path.write_text(render_html(data), encoding="utf-8")
        return ControlPanelResult(
            status=ControlPanelStatus.COMPLETED,
            root=str(actual_root),
            output_dir=str(actual_output),
            html_path=str(html_path),
            data_path=str(data_path),
            counts=data["counts"],
        )
    except Exception as exc:
        return ControlPanelResult(
            status=ControlPanelStatus.FAILED,
            root=str(actual_root),
            output_dir=str(actual_output),
            failure_reason=str(exc),
        )
