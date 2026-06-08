from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.process_control import run_cancellable
from tcad_agent.task_spec import PROJECT_ROOT


class UserDeckRunRequest(BaseModel):
    deck_path: str
    run_id: str | None = None
    run_root: Path = PROJECT_ROOT / "runs" / "user_decks"
    timeout_seconds: float = Field(default=600.0, gt=0)
    cancel_file: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_run_id() -> str:
    return f"user_deck_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def tail(text: str | None, limit: int = 4000) -> str:
    return (text or "")[-limit:]


def parse_stdout_json(stdout: str | None) -> dict[str, Any]:
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def run_user_deck(request: UserDeckRunRequest) -> dict[str, Any]:
    deck_path = Path(request.deck_path).expanduser().resolve()
    if not deck_path.exists():
        raise FileNotFoundError(f"user deck does not exist: {deck_path}")
    run_id = request.run_id or safe_run_id()
    run_dir = (request.run_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    state_path = run_dir / "user_deck_state.json"
    command = [sys.executable, str(deck_path)]
    started_at = utc_timestamp()
    completed = run_cancellable(
        command,
        cwd=deck_path.parent,
        capture_output=True,
        text=True,
        timeout=request.timeout_seconds,
        check=False,
        cancel_file=request.cancel_file,
    )
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    parsed = parse_stdout_json(completed.stdout)
    status = "completed" if completed.returncode == 0 else "failed"
    quality_status = "passed" if completed.returncode == 0 else "failed"
    issues = [] if completed.returncode == 0 else [{"code": "user_deck_execution_failed", "severity": "error"}]
    if "ACTSOFT_CANCELLED" in (completed.stderr or ""):
        status = "cancelled"
        quality_status = "failed"
        issues = [{"code": "user_deck_execution_cancelled", "severity": "error"}]
    artifacts = {
        "source_deck": str(deck_path),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    if parsed.get("run_dir"):
        artifacts["reported_run_dir"] = str(parsed["run_dir"])
    if parsed.get("summary_path"):
        artifacts["reported_summary"] = str(parsed["summary_path"])
    state = {
        "tool_name": "user_deck_execution",
        "status": status,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "source_deck_path": str(deck_path),
        "started_at": started_at,
        "completed_at": utc_timestamp(),
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": tail(completed.stdout),
        "stderr_tail": tail(completed.stderr),
        "reported_stdout_json": parsed,
        "final_summary": {
            "artifacts": artifacts,
            "metrics": {
                "returncode": completed.returncode,
            },
        },
        "quality_report": {
            "status": quality_status,
            "issues": issues,
            "metrics": {
                "returncode": completed.returncode,
            },
        },
        "next_action": "inspect user deck artifacts" if status != "completed" else "benchmark or inspect user deck artifacts",
    }
    write_json(state_path, state)
    return {**state, "state_path": str(state_path)}
