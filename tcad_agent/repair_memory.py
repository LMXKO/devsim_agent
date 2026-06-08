from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from tcad_agent.task_spec import PROJECT_ROOT


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_repair_memory_path() -> Path:
    override = os.environ.get("ACTSOFT_REPAIR_MEMORY_PATH")
    if override:
        return Path(override)
    return PROJECT_ROOT / "runs" / "repair_case_memory.jsonl"


def append_repair_case_memory(
    *,
    baseline_state_path: str,
    mutation_state_path: str,
    action_name: str,
    issue_codes: list[str],
    mutation_effect_analysis: dict[str, Any],
    memory_path: Path | None = None,
) -> str:
    path = memory_path or default_repair_memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "actsoft.tcad.repair_case_memory.v1",
        "created_at": utc_timestamp(),
        "baseline_state_path": baseline_state_path,
        "mutation_state_path": mutation_state_path,
        "action_name": action_name,
        "issue_codes": issue_codes,
        "mutation_target": mutation_effect_analysis.get("mutation_target"),
        "primary_metric": mutation_effect_analysis.get("primary_metric"),
        "decision": mutation_effect_analysis.get("decision"),
        "worth_continuing": mutation_effect_analysis.get("worth_continuing"),
        "rationale": mutation_effect_analysis.get("rationale"),
        "improved_metrics": mutation_effect_analysis.get("improved_metrics") or [],
        "regressed_metrics": mutation_effect_analysis.get("regressed_metrics") or [],
        "tradeoff_violations": mutation_effect_analysis.get("tradeoff_violations") or [],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return str(path.resolve())


def recent_repair_case_memory(limit: int = 20, *, memory_path: Path | None = None) -> list[dict[str, Any]]:
    path = memory_path or default_repair_memory_path()
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows[-limit:]
