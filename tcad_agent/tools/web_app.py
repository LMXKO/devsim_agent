from __future__ import annotations

import argparse
from pathlib import Path

from tcad_agent.run_queue import default_queue_db_path
from tcad_agent.task_spec import PROJECT_ROOT
from tcad_agent.web_app import WebAppConfig, serve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TCAD Mission Workbench web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--queue-db", type=Path, default=default_queue_db_path())
    parser.add_argument("--index-db", type=Path, default=None)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--worker-owner", default="tcad_web_worker")
    parser.add_argument("--worker-stop-file", type=Path, default=PROJECT_ROOT / "runs" / "tcad_web_worker.stop")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = WebAppConfig(
        root=args.root,
        queue_db_path=args.queue_db,
        index_db_path=args.index_db,
        host=args.host,
        port=args.port,
        rebuild_index=args.rebuild_index,
        worker_owner=args.worker_owner,
        worker_stop_file=args.worker_stop_file,
    )
    print(f"TCAD Mission Workbench listening on http://{args.host}:{args.port}")
    serve(config)


if __name__ == "__main__":
    main()
