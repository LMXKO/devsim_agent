from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.industrial_external_runner import IndustrialExternalRunnerRequest, run_industrial_external_runner


def parse_keyed_args(raw_items: list[str]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for raw in raw_items:
        if "=" not in raw:
            raise ValueError("--command-arg must use STEP=ARG")
        key, value = raw.split("=", 1)
        output.setdefault(key.strip(), []).append(value.strip())
    return {key: values for key, values in output.items() if key}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or gate a user-owned external industrial TCAD workspace.")
    parser.add_argument("--goal", "--goal-text", dest="goal_text", required=True)
    parser.add_argument("--template-id", default=None)
    parser.add_argument("--simulator", default="sentaurus")
    parser.add_argument("--project", "--project-path", dest="project_path", type=Path, default=None)
    parser.add_argument("--profile", "--profile-path", dest="profile_path", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--flow", action="append", default=[])
    parser.add_argument("--command-arg", action="append", default=[])
    parser.add_argument("--deck-file", action="append", default=[])
    parser.add_argument("--patch-json", default=None)
    parser.add_argument("--reference-curve-path", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--cancel-file", default=None)
    parser.add_argument("--no-execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        patches = json.loads(args.patch_json) if args.patch_json else []
        request = IndustrialExternalRunnerRequest(
            goal_text=args.goal_text,
            template_id=args.template_id,
            simulator=args.simulator,
            project_path=args.project_path,
            profile_path=args.profile_path,
            run_id=args.run_id,
            run_root=args.run_root or IndustrialExternalRunnerRequest.model_fields["run_root"].default,
            flow=args.flow,
            command_args=parse_keyed_args(args.command_arg),
            deck_files=args.deck_file,
            patches=patches,
            reference_curve_path=args.reference_curve_path,
            timeout_seconds=args.timeout_seconds,
            cancel_file=args.cancel_file,
            execute=not args.no_execute,
        )
        state = run_industrial_external_runner(request)
        print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if state.status in {"completed", "planned", "waiting_for_external_workspace"} else 1)
    except Exception as exc:
        print(json.dumps({"tool_name": "industrial_external_tcad_runner", "status": "failed", "failure_reason": str(exc)}, indent=2, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()

