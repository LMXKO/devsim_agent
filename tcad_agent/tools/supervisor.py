from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from tcad_agent.supervisor import SupervisorStatus, run_supervisor
from tcad_agent.task_spec import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpointed long-running TCAD supervisor.")
    parser.add_argument("--goal", required=True, help="Natural-language long-running TCAD goal.")
    parser.add_argument("--supervisor-id", default=None)
    parser.add_argument("--supervisor-root", type=Path, default=PROJECT_ROOT / "runs" / "supervisor")
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-agent-policy", action="store_true", help="Disable agent-first supervisor routing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        state = run_supervisor(
            args.goal,
            supervisor_id=args.supervisor_id,
            supervisor_root=args.supervisor_root,
            execute=args.execute,
            resume=args.resume,
            max_cycles=args.max_cycles,
            use_agent_policy=not args.no_agent_policy,
        )
        print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if state.status != SupervisorStatus.FAILED else 1)
    except (ValidationError, ValueError, FileExistsError, FileNotFoundError) as exc:
        print(
            json.dumps(
                {
                    "tool_name": "tcad_supervisor",
                    "status": SupervisorStatus.FAILED,
                    "failure_reason": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
