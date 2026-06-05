from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from tcad_agent.mission_agent import MissionStatus, run_mission_agent
from tcad_agent.task_spec import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-horizon TCAD mission agent.")
    parser.add_argument("--goal", required=True)
    parser.add_argument("--mission-id", default=None)
    parser.add_argument("--mission-root", type=Path, default=PROJECT_ROOT / "runs" / "missions")
    parser.add_argument("--max-cycles", type=int, default=8)
    parser.add_argument("--supervisor-max-cycles", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-llm", action="store_true", help="Use the configured OpenAI-compatible model for goal decomposition.")
    parser.add_argument("--no-llm-fallback", action="store_true", help="Fail instead of falling back to deterministic decomposition.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        state = run_mission_agent(
            args.goal,
            mission_id=args.mission_id,
            mission_root=args.mission_root,
            execute=args.execute,
            resume=args.resume,
            max_cycles=args.max_cycles,
            supervisor_max_cycles=args.supervisor_max_cycles,
            use_llm_decomposer=args.use_llm,
            allow_llm_fallback=not args.no_llm_fallback,
        )
        print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if state.status != MissionStatus.FAILED else 1)
    except (ValidationError, ValueError, FileExistsError, FileNotFoundError) as exc:
        print(
            json.dumps(
                {
                    "tool_name": "tcad_mission_agent",
                    "status": MissionStatus.FAILED,
                    "failure_reason": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
