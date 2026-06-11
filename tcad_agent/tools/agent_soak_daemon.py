from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.agent_soak_daemon import AgentSoakDaemonRequest, run_agent_soak_daemon
from tcad_agent.run_queue import default_queue_db_path
from tcad_agent.task_spec import PROJECT_ROOT


def parse_json_object(raw: str | None, flag: str) -> dict[str, object]:
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{flag} must decode to a JSON object")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an agent_soak queue daemon for a natural-language TCAD goal.")
    parser.add_argument("--goal", "--goal-text", dest="goal_text", required=True)
    parser.add_argument("--daemon-id", default=None)
    parser.add_argument("--queue-id", default=None)
    parser.add_argument("--db", type=Path, default=default_queue_db_path())
    parser.add_argument("--daemon-root", type=Path, default=PROJECT_ROOT / "runs" / "agent_soak_daemon")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--duration-hours", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--step-slice", type=int, default=4)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-llm-fallback", action="store_true")
    parser.add_argument("--priority", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--owner", default="agent_soak_daemon")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--lease-seconds", type=float, default=7200.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--max-loops", type=int, default=None)
    parser.add_argument("--max-idle-loops", type=int, default=1)
    parser.add_argument("--stop-file", type=Path, default=None)
    parser.add_argument("--autonomous-request-json", default=None)
    return parser.parse_args()


def request_from_args(args: argparse.Namespace) -> AgentSoakDaemonRequest:
    return AgentSoakDaemonRequest(
        daemon_id=args.daemon_id,
        goal_text=args.goal_text,
        queue_id=args.queue_id,
        queue_db_path=args.db,
        daemon_root=args.daemon_root,
        execute=args.execute,
        duration_hours=args.duration_hours,
        max_steps=args.max_steps,
        step_slice=args.step_slice,
        use_llm=not args.no_llm,
        allow_llm_fallback=not args.no_llm_fallback,
        priority=args.priority,
        max_attempts=args.max_attempts,
        owner=args.owner,
        concurrency=args.concurrency,
        lease_seconds=args.lease_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        max_loops=args.max_loops,
        max_idle_loops=args.max_idle_loops,
        stop_file=args.stop_file,
        autonomous_request=parse_json_object(args.autonomous_request_json, "--autonomous-request-json"),
    )


def main() -> None:
    state = run_agent_soak_daemon(request_from_args(parse_args()))
    print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if state.status in {"completed", "waiting_for_user", "idle", "stopped"} else 1)


if __name__ == "__main__":
    main()
