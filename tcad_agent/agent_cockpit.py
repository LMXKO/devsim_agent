from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AgentCockpitResult(BaseModel):
    tool_name: str = "agent_cockpit"
    status: str
    source_path: str
    output_path: str
    sections: list[str] = Field(default_factory=list)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value))


def artifact_link(path_value: Any) -> str:
    if not path_value:
        return ""
    value = str(path_value)
    return f'<a href="{esc(Path(value).resolve().as_uri() if Path(value).exists() else value)}">{esc(Path(value).name or value)}</a>'


def table(headers: list[str], rows: list[list[Any]], *, raw_columns: set[int] | None = None) -> str:
    if not rows:
        return '<p class="muted">None</p>'
    raw_columns = raw_columns or set()
    head = "".join(f"<th>{esc(item)}</th>" for item in headers)
    body = "\n".join(
        "<tr>"
        + "".join(f"<td>{cell if index in raw_columns else esc(cell)}</td>" for index, cell in enumerate(row))
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def status_summary(state: dict[str, Any]) -> str:
    metrics = ((state.get("final_summary") or {}).get("metrics") or {}) if isinstance(state.get("final_summary"), dict) else {}
    rows = [
        ["Status", state.get("status")],
        ["Tool", state.get("tool_name")],
        ["Run/Mission", state.get("run_id") or state.get("agent_id") or state.get("mission_id")],
        ["Next", state.get("next_action")],
        ["Signoff", metrics.get("signoff_verdict") or (state.get("signoff_gate") or {}).get("verdict")],
    ]
    return table(["Field", "Value"], rows)


def decision_rows(state: dict[str, Any]) -> list[list[Any]]:
    checkpoint = state.get("checkpoint") if isinstance(state.get("checkpoint"), dict) else {}
    ledger = checkpoint.get("agent_decision_ledger") or checkpoint.get("controller_cycles") or []
    rows: list[list[Any]] = []
    for item in ledger[-12:]:
        if not isinstance(item, dict):
            continue
        action = item.get("action") or item.get("decision") or {}
        rows.append(
            [
                item.get("step_index") or item.get("cycle"),
                action.get("kind") or action.get("action"),
                action.get("tool_name") or action.get("tool"),
                item.get("hypothesis_zh") or item.get("observation_summary") or action.get("reason"),
                item.get("fallback_used"),
            ]
        )
    return rows


def lineage_rows(state: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for key in ["mutation_effect_analysis", "sentaurus_mutation_effect_analysis"]:
        effect = state.get(key)
        if isinstance(effect, dict) and effect:
            rows.extend(
                [
                    [key, "decision", effect.get("decision")],
                    [key, "primary_metric", effect.get("primary_metric")],
                    [key, "improved", ", ".join(effect.get("improved_metrics") or [])],
                    [key, "regressed", ", ".join(effect.get("regressed_metrics") or [])],
                    [key, "overlay", effect.get("overlay_svg_path") or (effect.get("curve_overlay") or {}).get("overlay_svg")],
                ]
            )
    for patch in state.get("patches") or state.get("tcad_deck_mutations") or []:
        if isinstance(patch, dict):
            rows.append(["patch", patch.get("operation") or patch.get("target"), patch.get("reason") or patch.get("request_path")])
    for patch in state.get("applied_patches") or []:
        if isinstance(patch, dict):
            rows.append(["applied_patch", patch.get("section"), patch.get("reason")])
    return rows


def artifact_rows(state: dict[str, Any]) -> list[list[Any]]:
    artifacts: dict[str, Any] = {}
    raw = state.get("artifacts")
    if isinstance(raw, dict):
        artifacts.update(raw)
    summary = state.get("final_summary") if isinstance(state.get("final_summary"), dict) else {}
    if isinstance(summary.get("artifacts"), dict):
        artifacts.update(summary["artifacts"])
    rows = []
    for key, value in sorted(artifacts.items()):
        if any(token in key.lower() for token in ["csv", "plot", "overlay", "diff", "report", "dashboard", "state", "benchmark", "contract", "lineage", "gate"]):
            rows.append([key, artifact_link(value)])
    return rows[:24]


def signoff_rows(state: dict[str, Any]) -> list[list[Any]]:
    gate = state.get("signoff_gate") if isinstance(state.get("signoff_gate"), dict) else {}
    pack = gate.get("benchmark_signoff_pack") or ((state.get("summary") or {}).get("signoff_evidence_pack") if isinstance(state.get("summary"), dict) else {})
    rows = [
        ["verdict", gate.get("verdict") or pack.get("verdict")],
        ["missing", ", ".join(gate.get("missing_evidence") or pack.get("missing_evidence") or [])],
        ["blocking", ", ".join(gate.get("blocking_reasons") or pack.get("blocking_reasons") or [])],
    ]
    next_actions = gate.get("next_actions") or pack.get("next_actions") or []
    for action in next_actions[:6]:
        if isinstance(action, dict):
            rows.append(["next_action", action.get("action"), action.get("reason")])
    return rows


def render_agent_cockpit(state: dict[str, Any], *, source_path: Path) -> str:
    title = state.get("goal_text") or state.get("run_id") or state.get("agent_id") or "TCAD Agent"
    generated = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>
body {{ margin: 0; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #15171a; background: #fff; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size: 22px; margin: 0 0 6px; font-weight: 650; }}
h2 {{ font-size: 15px; margin: 28px 0 8px; font-weight: 650; }}
.meta {{ color: #5b6472; margin-bottom: 18px; }}
table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
th, td {{ border-bottom: 1px solid #e6e8eb; padding: 7px 8px; text-align: left; vertical-align: top; word-break: break-word; }}
th {{ color: #4b5563; font-weight: 600; background: #fafafa; }}
a {{ color: #0f5cc0; text-decoration: none; }}
.muted {{ color: #6b7280; }}
</style>
</head>
<body>
<main>
<h1>{esc(title)}</h1>
<div class="meta">Generated {esc(generated)} from {esc(source_path)}</div>
<h2>Summary</h2>
{status_summary(state)}
<h2>Decisions</h2>
{table(["Step", "Action", "Tool", "Rationale", "Fallback"], decision_rows(state))}
<h2>Lineage</h2>
{table(["Kind", "Field", "Value"], lineage_rows(state))}
<h2>Signoff Gate</h2>
{table(["Field", "Value", "Reason"], signoff_rows(state))}
<h2>Artifacts</h2>
{table(["Name", "Path"], artifact_rows(state), raw_columns={1})}
</main>
</body>
</html>
"""


def generate_agent_cockpit(source: Path, output_path: Path | None = None) -> AgentCockpitResult:
    state = read_json(source)
    output = output_path or source.with_name("agent_cockpit.html")
    html_text = render_agent_cockpit(state, source_path=source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")
    return AgentCockpitResult(
        status="completed",
        source_path=str(source.resolve()),
        output_path=str(output.resolve()),
        sections=["summary", "decisions", "lineage", "signoff_gate", "artifacts"],
    )
