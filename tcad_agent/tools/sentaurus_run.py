from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.sentaurus import SentaurusRunRequest, run_sentaurus


def parse_keyed_args(raw_items: list[str]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for raw in raw_items:
        if "=" not in raw:
            raise ValueError("--command-arg must use STEP=ARG or STEP=JSON_LIST")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("--command-arg step must not be empty")
        value = value.strip()
        if value.startswith("["):
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise ValueError("--command-arg JSON value must be a list")
            output.setdefault(key, []).extend(str(item) for item in parsed)
        else:
            output.setdefault(key, []).append(value)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local Sentaurus project through the ActSoft agent runner.")
    parser.add_argument("--goal", "--goal-text", dest="goal_text", required=True)
    parser.add_argument("--project", "--project-path", dest="project_path", type=Path, required=True)
    parser.add_argument("--profile", "--profile-path", dest="profile_path", type=Path, default=None)
    parser.add_argument("--profile-json", default=None, help="Inline SentaurusRuntimeProfile JSON object.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--flow", action="append", default=[], help="Flow step, e.g. sdevice or svisual. Repeatable.")
    parser.add_argument("--command-arg", action="append", default=[], help="STEP=ARG or STEP=[JSON,args]. Repeatable.")
    parser.add_argument("--deck-file", action="append", default=[])
    parser.add_argument("--patch-json", default=None, help="JSON list of SentaurusPatch objects.")
    parser.add_argument("--reference-curve-path", type=Path, default=None)
    parser.add_argument("--breakdown-current-threshold-a", type=float, default=1.0e-6)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--cancel-file", default=None)
    parser.add_argument("--no-execute", action="store_true")
    return parser.parse_args()


def request_from_args(args: argparse.Namespace) -> SentaurusRunRequest:
    profile = json.loads(args.profile_json) if args.profile_json else None
    patches = json.loads(args.patch_json) if args.patch_json else []
    if patches and not isinstance(patches, list):
        raise ValueError("--patch-json must decode to a JSON list")
    return SentaurusRunRequest(
        goal_text=args.goal_text,
        project_path=args.project_path,
        profile_path=args.profile_path,
        profile=profile,
        run_id=args.run_id,
        run_root=args.run_root,
        flow=args.flow,
        command_args=parse_keyed_args(args.command_arg),
        deck_files=args.deck_file,
        patches=patches,
        reference_curve_path=args.reference_curve_path,
        breakdown_current_threshold_a=args.breakdown_current_threshold_a,
        timeout_seconds=args.timeout_seconds,
        cancel_file=args.cancel_file,
        execute=not args.no_execute,
    )


def main() -> None:
    try:
        state = run_sentaurus(request_from_args(parse_args()))
        print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if state.status in {"completed", "planned"} else 1)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "tool_name": "sentaurus_run",
                    "status": "failed",
                    "failure_reason": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()

