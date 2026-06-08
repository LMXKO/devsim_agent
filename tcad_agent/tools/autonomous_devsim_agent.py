from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.autonomous_devsim_agent import (
    AutonomousDevsimRequest,
    DevsimAgentStatus,
    run_autonomous_devsim_agent,
)
from tcad_agent.task_spec import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the long-horizon autonomous DEVSIM agent runtime.")
    parser.add_argument("--goal", "--goal-text", dest="goal_text", required=True)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--agent-root", type=Path, default=PROJECT_ROOT / "runs" / "autonomous_devsim_agent")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--initial-tool-name", default=None)
    parser.add_argument("--initial-request-json", default=None, help="JSON object request for --initial-tool-name.")
    parser.add_argument("--source-state-path", default=None)
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic policy only.")
    parser.add_argument("--no-llm-fallback", action="store_true", help="Fail if the LLM action is invalid/unavailable.")
    parser.add_argument("--no-agent-repair-policy", action="store_true", help="Disable LLM repair policy inside repair_executor.")
    parser.add_argument("--allow-user-confirmation-actions", action="store_true")
    parser.add_argument("--supervisor-max-cycles", type=int, default=3)
    parser.add_argument("--repair-max-rounds", type=int, default=3)
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--no-dashboard", action="store_true")
    return parser.parse_args()


def request_from_args(args: argparse.Namespace) -> AutonomousDevsimRequest:
    initial_request = {}
    if args.initial_request_json:
        parsed = json.loads(args.initial_request_json)
        if not isinstance(parsed, dict):
            raise ValueError("--initial-request-json must decode to a JSON object")
        initial_request = parsed
    return AutonomousDevsimRequest(
        goal_text=args.goal_text,
        agent_id=args.agent_id,
        agent_root=args.agent_root,
        execute=args.execute,
        resume=args.resume,
        max_steps=args.max_steps,
        initial_tool_name=args.initial_tool_name,
        initial_request=initial_request,
        source_state_path=args.source_state_path,
        use_llm=not args.no_llm,
        allow_llm_fallback=not args.no_llm_fallback,
        use_agent_policy=not args.no_agent_repair_policy,
        allow_user_confirmation_actions=args.allow_user_confirmation_actions,
        supervisor_max_cycles=args.supervisor_max_cycles,
        repair_max_rounds=args.repair_max_rounds,
        generate_report=not args.no_report,
        generate_dashboard=not args.no_dashboard,
    )


def main() -> None:
    try:
        state = run_autonomous_devsim_agent(request_from_args(parse_args()))
        print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if state.status != DevsimAgentStatus.FAILED else 1)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "tool_name": "autonomous_devsim_agent",
                    "status": DevsimAgentStatus.FAILED,
                    "failure_reason": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
