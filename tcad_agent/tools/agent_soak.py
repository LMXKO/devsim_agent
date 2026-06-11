from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.agent_soak import AgentSoakRequest, AgentSoakStatus, run_agent_soak
from tcad_agent.task_spec import PROJECT_ROOT


def parse_json_object(raw: str | None, flag: str) -> dict[str, object]:
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{flag} must decode to a JSON object")
    return parsed


def parse_json_list(raw: str | None, flag: str) -> list[object]:
    if not raw:
        return []
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError(f"{flag} must decode to a JSON list")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a long-duration autonomous DEVSIM agent soak.")
    parser.add_argument("--goal", "--goal-text", dest="goal_text", required=True)
    parser.add_argument("--soak-id", default=None)
    parser.add_argument("--soak-root", type=Path, default=PROJECT_ROOT / "runs" / "agent_soak")
    parser.add_argument("--duration-hours", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--step-slice", type=int, default=4)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--agent-root", type=Path, default=None)
    parser.add_argument("--cancel-file", type=Path, default=None)
    parser.add_argument("--heartbeat-path", type=Path, default=None)
    parser.add_argument("--no-cockpit", action="store_true")
    parser.add_argument("--cockpit-interval-steps", type=int, default=1)
    parser.add_argument("--no-mission-spec", action="store_true")
    parser.add_argument("--no-recovery", action="store_true")
    parser.add_argument("--max-recovery-attempts", type=int, default=2)
    parser.add_argument("--no-agent-memory", action="store_true")
    parser.add_argument("--memory-path", type=Path, default=None)
    parser.add_argument("--no-curve-guidance", action="store_true")
    parser.add_argument("--no-auto-curve-guidance", action="store_true")
    parser.add_argument("--max-curve-guided-patches", type=int, default=1)
    parser.add_argument("--autonomous-request-json", default=None, help="JSON object merged into the autonomous agent request.")
    parser.add_argument("--initial-tool-name", default=None)
    parser.add_argument("--initial-request-json", default=None)
    parser.add_argument("--source-state-path", default=None)
    parser.add_argument("--source-deck-path", default=None)
    parser.add_argument("--deck-patches-json", default=None)
    parser.add_argument("--sentaurus-project-path", type=Path, default=None)
    parser.add_argument("--sentaurus-profile-path", type=Path, default=None)
    parser.add_argument("--sentaurus-request-json", default=None)
    parser.add_argument("--objectives-json", default=None)
    parser.add_argument("--constraints-json", default=None)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-llm-fallback", action="store_true")
    parser.add_argument("--no-agent-repair-policy", action="store_true")
    parser.add_argument("--enable-live-evidence-lookup", action="store_true")
    parser.add_argument("--live-evidence-max-sources", type=int, default=6)
    parser.add_argument("--allow-live-evidence-gaps", action="store_true")
    parser.add_argument("--allow-user-confirmation-actions", action="store_true")
    parser.add_argument("--supervisor-max-cycles", type=int, default=3)
    parser.add_argument("--repair-max-rounds", type=int, default=3)
    parser.add_argument("--max-mutation-refinements", type=int, default=1)
    parser.add_argument("--no-auto-mutation-refinement", action="store_true")
    parser.add_argument("--enable-experiment-design", action="store_true")
    parser.add_argument("--max-experiment-design-rounds", type=int, default=1)
    parser.add_argument("--no-auto-experiment-design", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--require-capability-audit", action="store_true")
    return parser.parse_args()


def request_from_args(args: argparse.Namespace) -> AgentSoakRequest:
    autonomous_request = parse_json_object(args.autonomous_request_json, "--autonomous-request-json")
    initial_request = parse_json_object(args.initial_request_json, "--initial-request-json")
    deck_patches = parse_json_list(args.deck_patches_json, "--deck-patches-json")
    sentaurus_request = parse_json_object(args.sentaurus_request_json, "--sentaurus-request-json")
    objectives = parse_json_list(args.objectives_json, "--objectives-json")
    constraints = parse_json_list(args.constraints_json, "--constraints-json")
    explicit = {
        "initial_tool_name": args.initial_tool_name,
        "initial_request": initial_request,
        "source_state_path": args.source_state_path,
        "source_deck_path": args.source_deck_path,
        "deck_patches": deck_patches,
        "sentaurus_project_path": args.sentaurus_project_path,
        "sentaurus_profile_path": args.sentaurus_profile_path,
        "sentaurus_request": sentaurus_request,
        "objectives": objectives,
        "constraints": constraints,
        "use_llm": not args.no_llm,
        "allow_llm_fallback": not args.no_llm_fallback,
        "use_agent_policy": not args.no_agent_repair_policy,
        "enable_live_evidence_lookup": args.enable_live_evidence_lookup,
        "live_evidence_max_sources": args.live_evidence_max_sources,
        "allow_live_evidence_gaps": args.allow_live_evidence_gaps,
        "allow_user_confirmation_actions": args.allow_user_confirmation_actions,
        "supervisor_max_cycles": args.supervisor_max_cycles,
        "repair_max_rounds": args.repair_max_rounds,
        "max_mutation_refinements": args.max_mutation_refinements,
        "auto_execute_mutation_refinements": not args.no_auto_mutation_refinement,
        "enable_experiment_design": args.enable_experiment_design,
        "max_experiment_design_rounds": args.max_experiment_design_rounds,
        "auto_execute_experiment_design": not args.no_auto_experiment_design,
        "generate_report": not args.no_report,
        "generate_dashboard": not args.no_dashboard,
        "require_capability_audit": args.require_capability_audit,
    }
    for key, value in explicit.items():
        if value is not None and value != [] and value != {}:
            autonomous_request[key] = value
    return AgentSoakRequest(
        goal_text=args.goal_text,
        soak_id=args.soak_id,
        soak_root=args.soak_root,
        execute=args.execute,
        resume=args.resume,
        duration_hours=args.duration_hours,
        max_steps=args.max_steps,
        step_slice=args.step_slice,
        poll_interval_seconds=args.poll_interval_seconds,
        agent_id=args.agent_id,
        agent_root=args.agent_root,
        autonomous_request=autonomous_request,
        cancel_file=args.cancel_file,
        heartbeat_path=args.heartbeat_path,
        generate_cockpit=not args.no_cockpit,
        cockpit_interval_steps=args.cockpit_interval_steps,
        compile_mission_spec=not args.no_mission_spec,
        enable_recovery=not args.no_recovery,
        max_recovery_attempts=args.max_recovery_attempts,
        enable_agent_memory=not args.no_agent_memory,
        memory_path=args.memory_path,
        enable_curve_guidance=not args.no_curve_guidance,
        auto_execute_curve_guidance=not args.no_auto_curve_guidance,
        max_curve_guided_patches=args.max_curve_guided_patches,
    )


def main() -> None:
    try:
        state = run_agent_soak(request_from_args(parse_args()))
        print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if state.status in {AgentSoakStatus.COMPLETED, AgentSoakStatus.WAITING_FOR_USER, AgentSoakStatus.CANCELLED} else 1)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "tool_name": "agent_soak",
                    "status": AgentSoakStatus.FAILED,
                    "failure_reason": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
