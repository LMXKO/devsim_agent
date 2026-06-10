from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.sentaurus_contract import (
    default_fixture_root,
    validate_fixture_corpus,
    validate_sentaurus_contract,
)
from tcad_agent.task_spec import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate offline Sentaurus agent fixture and runner contracts.")
    parser.add_argument("--project", "--project-path", dest="project_path", type=Path, default=None)
    parser.add_argument("--fixtures-root", type=Path, default=default_fixture_root())
    parser.add_argument("--all-fixtures", action="store_true", help="Validate every fixture under --fixtures-root.")
    parser.add_argument("--run-fake-e2e", action="store_true", help="Run the interface-only fake backend through sentaurus_run.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "sentaurus_contract")
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.all_fixtures:
            results = validate_fixture_corpus(
                args.fixtures_root,
                run_fake_e2e=args.run_fake_e2e,
                output_root=args.output_root,
            )
            payload = {
                "tool_name": "sentaurus_contract",
                "status": "failed" if any(result.status == "failed" for result in results) else "passed",
                "results": [result.model_dump(mode="json") for result in results],
            }
            if args.report_path:
                args.report_path.parent.mkdir(parents=True, exist_ok=True)
                args.report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            raise SystemExit(0 if payload["status"] == "passed" else 1)
        if not args.project_path:
            raise ValueError("--project is required unless --all-fixtures is set")
        result = validate_sentaurus_contract(
            args.project_path,
            run_fake_e2e=args.run_fake_e2e,
            output_root=args.output_root,
            report_path=args.report_path,
        )
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if result.status == "passed" else 1)
    except Exception as exc:
        print(
            json.dumps(
                {"tool_name": "sentaurus_contract", "status": "failed", "failure_reason": str(exc)},
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()

