from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.experiment_index import default_index_db_path, list_records, rebuild_index
from tcad_agent.task_spec import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index and query TCAD experiment states.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "runs", help="Run root to scan.")
    parser.add_argument("--db", type=Path, default=default_index_db_path(), help="SQLite index path.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the experiment index from state files.")
    parser.add_argument("--list", action="store_true", help="List indexed experiments.")
    parser.add_argument("--kind", default=None, help="Filter by kind: task_run, parameter_sweep, adaptive_optimization.")
    parser.add_argument("--status", default=None, help="Filter by status.")
    parser.add_argument("--limit", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload: dict[str, object] = {
        "tool_name": "experiment_index",
        "status": "completed",
    }
    if args.rebuild:
        payload["rebuild"] = rebuild_index(args.root, args.db)
    if args.list or not args.rebuild:
        payload["records"] = list_records(args.db, kind=args.kind, status=args.status, limit=args.limit)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
