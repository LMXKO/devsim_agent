from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.long_run_validation import (
    LongRunValidationRequest,
    LongRunValidationMode,
    LongRunValidationStatus,
    LongRunValidationSuite,
    run_long_run_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an unattended long-run TCAD agent validation.")
    parser.add_argument("--validation-id", default=None)
    parser.add_argument("--validation-root", type=Path, default=None)
    parser.add_argument(
        "--suite",
        choices=[item.value for item in LongRunValidationSuite],
        default=LongRunValidationSuite.QUEUE_SMOKE.value,
        help="Validation suite to run. queue_smoke preserves the original fast queue regression.",
    )
    parser.add_argument(
        "--mode",
        choices=[item.value for item in LongRunValidationMode],
        default=LongRunValidationMode.SIMULATED.value,
        help="simulated uses deterministic local runners; real runs the provided/default autonomous agent request with real tools.",
    )
    parser.add_argument("--scenario-id", action="append", default=None, help="Run one scenario. Repeat to run multiple scenarios.")
    parser.add_argument("--agent-max-steps", type=int, default=12)
    parser.add_argument("--use-llm", action="store_true", help="Allow the real autonomous scenario to call the configured LLM.")
    parser.add_argument("--no-llm-fallback", action="store_true", help="Fail the real autonomous scenario if LLM output is invalid/unavailable.")
    parser.add_argument("--real-agent-request-json", default=None, help="JSON object merged into the real autonomous agent request.")
    parser.add_argument("--queue-goals-json", default=None)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.0)
    parser.add_argument("--max-idle-loops", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = {
        "validation_id": args.validation_id,
        "suite": args.suite,
        "mode": args.mode,
        "scenario_ids": args.scenario_id or [],
        "agent_max_steps": args.agent_max_steps,
        "use_llm": args.use_llm,
        "allow_llm_fallback": not args.no_llm_fallback,
        "poll_interval_seconds": args.poll_interval_seconds,
        "max_idle_loops": args.max_idle_loops,
    }
    if args.validation_root is not None:
        data["validation_root"] = args.validation_root
    if args.queue_goals_json:
        goals = json.loads(args.queue_goals_json)
        if not isinstance(goals, list):
            raise ValueError("--queue-goals-json must decode to a list")
        data["queue_goals"] = goals
    if args.real_agent_request_json:
        real_request = json.loads(args.real_agent_request_json)
        if not isinstance(real_request, dict):
            raise ValueError("--real-agent-request-json must decode to an object")
        data["real_agent_request"] = real_request
    state = run_long_run_validation(LongRunValidationRequest.model_validate(data))
    print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if state.status == LongRunValidationStatus.COMPLETED else 2)


if __name__ == "__main__":
    main()
