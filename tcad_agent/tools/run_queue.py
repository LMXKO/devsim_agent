from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcad_agent.run_queue import (
    QueueStatus,
    cancel_item,
    claim_next_items,
    default_queue_db_path,
    default_worker_owner,
    enqueue_run,
    get_item,
    heartbeat_item,
    list_items,
    pause_item,
    recover_stale_items,
    run_queue_daemon,
    resume_item,
    run_queue_worker,
)


def parse_json_object(raw: str | None) -> dict[str, object]:
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("JSON value must be an object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Durable TCAD run queue.")
    parser.add_argument("--db", type=Path, default=default_queue_db_path())
    subparsers = parser.add_subparsers(dest="command", required=True)

    enqueue = subparsers.add_parser("enqueue", help="Enqueue a TCAD run.")
    enqueue.add_argument("--queue-id", default=None)
    enqueue.add_argument("--tool", default="supervisor")
    enqueue.add_argument("--request-json", default=None)
    enqueue.add_argument("--goal", default=None, help="Convenience goal_text for supervisor items.")
    enqueue.add_argument("--priority", type=int, default=0)
    enqueue.add_argument("--tag", action="append", default=[])
    enqueue.add_argument("--max-attempts", type=int, default=1)
    enqueue.add_argument("--budget-seconds", type=float, default=None)
    enqueue.add_argument("--budget-cases", type=int, default=None)

    list_parser = subparsers.add_parser("list", help="List queued/running/history items.")
    list_parser.add_argument("--status", choices=[status.value for status in QueueStatus], default=None)
    list_parser.add_argument("--tool", default=None)
    list_parser.add_argument("--limit", type=int, default=50)

    show = subparsers.add_parser("show", help="Show one queue item.")
    show.add_argument("queue_id")

    worker = subparsers.add_parser("worker", help="Claim and execute queued runs.")
    worker.add_argument("--owner", default=None)
    worker.add_argument("--concurrency", type=int, default=1)
    worker.add_argument("--lease-seconds", type=float, default=3600.0)
    worker.add_argument("--max-items", type=int, default=None)

    daemon = subparsers.add_parser("daemon", help="Run a polling queue worker loop.")
    daemon.add_argument("--owner", default=None)
    daemon.add_argument("--concurrency", type=int, default=1)
    daemon.add_argument("--lease-seconds", type=float, default=3600.0)
    daemon.add_argument("--poll-interval-seconds", type=float, default=5.0)
    daemon.add_argument("--max-loops", type=int, default=None)
    daemon.add_argument("--max-idle-loops", type=int, default=None)
    daemon.add_argument("--stop-file", type=Path, default=None)

    claim = subparsers.add_parser("claim", help="Claim queued runs without executing them.")
    claim.add_argument("--owner", default=None)
    claim.add_argument("--limit", type=int, default=1)
    claim.add_argument("--lease-seconds", type=float, default=3600.0)

    heartbeat = subparsers.add_parser("heartbeat", help="Extend a running item's lease.")
    heartbeat.add_argument("queue_id")
    heartbeat.add_argument("--owner", required=True)
    heartbeat.add_argument("--lease-seconds", type=float, default=3600.0)

    pause = subparsers.add_parser("pause", help="Pause a queued or running item.")
    pause.add_argument("queue_id")

    resume = subparsers.add_parser("resume", help="Resume a paused item.")
    resume.add_argument("queue_id")

    cancel = subparsers.add_parser("cancel", help="Cancel a non-terminal item.")
    cancel.add_argument("queue_id")

    subparsers.add_parser("recover", help="Recover expired running leases.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "enqueue":
            request = parse_json_object(args.request_json)
            if args.goal:
                request = {**request, "goal_text": args.goal}
            item = enqueue_run(
                args.db,
                tool_name=args.tool,
                request=request,
                queue_id=args.queue_id,
                priority=args.priority,
                tags=args.tag,
                max_attempts=args.max_attempts,
                budget_seconds=args.budget_seconds,
                budget_cases=args.budget_cases,
            )
            output = item.model_dump(mode="json")
        elif args.command == "list":
            output = {
                "items": list_items(
                    args.db,
                    status=args.status,
                    tool_name=args.tool,
                    limit=args.limit,
                )
            }
        elif args.command == "show":
            item = get_item(args.db, args.queue_id)
            output = item.model_dump(mode="json") if item else {"error": f"queue item not found: {args.queue_id}"}
        elif args.command == "worker":
            output = run_queue_worker(
                args.db,
                owner=args.owner,
                concurrency=args.concurrency,
                lease_seconds=args.lease_seconds,
                max_items=args.max_items,
            ).model_dump(mode="json")
        elif args.command == "daemon":
            output = run_queue_daemon(
                args.db,
                owner=args.owner,
                concurrency=args.concurrency,
                lease_seconds=args.lease_seconds,
                poll_interval_seconds=args.poll_interval_seconds,
                max_loops=args.max_loops,
                max_idle_loops=args.max_idle_loops,
                stop_file=args.stop_file,
            ).model_dump(mode="json")
        elif args.command == "claim":
            owner = args.owner or default_worker_owner()
            output = {
                "owner": owner,
                "items": [
                    item.model_dump(mode="json")
                    for item in claim_next_items(
                        args.db,
                        owner=owner,
                        limit=args.limit,
                        lease_seconds=args.lease_seconds,
                    )
                ],
            }
        elif args.command == "heartbeat":
            output = heartbeat_item(
                args.db,
                args.queue_id,
                owner=args.owner,
                lease_seconds=args.lease_seconds,
            ).model_dump(mode="json")
        elif args.command == "pause":
            output = pause_item(args.db, args.queue_id).model_dump(mode="json")
        elif args.command == "resume":
            output = resume_item(args.db, args.queue_id).model_dump(mode="json")
        elif args.command == "cancel":
            output = cancel_item(args.db, args.queue_id).model_dump(mode="json")
        elif args.command == "recover":
            output = recover_stale_items(args.db)
        else:
            raise ValueError(f"unknown command: {args.command}")
        print(json.dumps(output, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"status": "failed", "failure_reason": str(exc)}, indent=2, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
