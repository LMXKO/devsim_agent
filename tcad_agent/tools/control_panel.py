from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.control_panel import ControlPanelStatus, generate_control_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a static TCAD agent control panel.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--queue-db", type=Path, default=None)
    parser.add_argument("--index-db", type=Path, default=None)
    parser.add_argument("--no-rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_control_panel(
        args.root,
        output_dir=args.output_dir,
        queue_db_path=args.queue_db,
        index_db_path=args.index_db,
        rebuild=not args.no_rebuild,
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.status == ControlPanelStatus.COMPLETED else 2)


if __name__ == "__main__":
    main()
