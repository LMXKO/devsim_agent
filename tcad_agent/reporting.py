from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ReportKind(str, Enum):
    PARAMETER_SWEEP = "parameter_sweep"
    ADAPTIVE_OPTIMIZATION = "adaptive_optimization"
    MULTIDIM_OPTIMIZATION = "multidim_optimization"


class ReportStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class ReportResult(BaseModel):
    tool_name: str = "experiment_report"
    status: ReportStatus
    kind: ReportKind | None = None
    source_state_path: str
    report_path: str | None = None
    best_artifact_plot_path: str | None = None
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve_state_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"State path does not exist: {path}")
    for candidate in ["optimization_state.json", "sweep_state.json"]:
        state_path = path / candidate
        if state_path.exists():
            return state_path
    raise FileNotFoundError(f"No optimization_state.json or sweep_state.json found under: {path}")


def detect_report_kind(state: dict[str, Any]) -> ReportKind:
    tool_name = state.get("tool_name")
    if tool_name == "adaptive_optimizer":
        return ReportKind.ADAPTIVE_OPTIMIZATION
    if tool_name == "multidim_optimizer":
        return ReportKind.MULTIDIM_OPTIMIZATION
    if tool_name == "parameter_sweep":
        return ReportKind.PARAMETER_SWEEP
    raise ValueError(f"Unsupported state tool_name: {tool_name}")


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return ", ".join(format_value(item) for item in value)
    if isinstance(value, dict):
        return "`" + json.dumps(value, ensure_ascii=False, sort_keys=True) + "`"
    return str(value)


def escape_cell(value: Any) -> str:
    return format_value(value).replace("|", "\\|").replace("\n", "<br>")


def markdown_target(path: str | Path, base_dir: Path) -> str:
    target = Path(path)
    if target.is_absolute():
        rendered = os.path.relpath(target, base_dir)
    else:
        rendered = str(target)
    return rendered.replace(os.sep, "/")


def markdown_link(label: str, path: str | Path | None, base_dir: Path) -> str:
    if not path:
        return ""
    return f"[{label}](<{markdown_target(path, base_dir)}>)"


def markdown_image(label: str, path: str | Path | None, base_dir: Path) -> str:
    if not path:
        return ""
    return f"![{label}](<{markdown_target(path, base_dir)}>)"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(escape_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def load_final_state(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    state_path = Path(path)
    if not state_path.exists():
        return None
    try:
        return read_json(state_path)
    except json.JSONDecodeError:
        return None


def final_artifacts(final_state: dict[str, Any] | None) -> dict[str, str]:
    if not final_state:
        return {}
    summary = final_state.get("final_summary") or {}
    artifacts = summary.get("artifacts") or {}
    return {key: str(value) for key, value in artifacts.items() if value}


def final_metrics(final_state: dict[str, Any] | None) -> dict[str, Any]:
    if not final_state:
        return {}
    quality = final_state.get("quality_report") or {}
    metrics = quality.get("metrics") or {}
    summary = final_state.get("final_summary") or {}
    merged = dict(summary)
    merged.update(metrics)
    return merged


def objective_direction(state: dict[str, Any]) -> str:
    objective = state.get("objective") or {}
    return str(objective.get("direction") or "minimize")


def sorted_by_objective(items: list[dict[str, Any]], direction: str) -> list[dict[str, Any]]:
    eligible = [item for item in items if item.get("objective_value") is not None]
    reverse = direction == "maximize"
    return sorted(eligible, key=lambda item: float(item.get("objective_value") or 0.0), reverse=reverse)


def best_sweep_case(state: dict[str, Any]) -> dict[str, Any] | None:
    if state.get("best_case"):
        return state["best_case"]
    ranked = sorted_by_objective(state.get("cases") or [], objective_direction(state))
    return ranked[0] if ranked else None


def best_optimization_observation(state: dict[str, Any]) -> dict[str, Any] | None:
    if state.get("best_observation"):
        return state["best_observation"]
    ranked = sorted_by_objective(state.get("observations") or [], objective_direction(state))
    return ranked[0] if ranked else None


def axis_names_from_sweep(state: dict[str, Any]) -> list[str]:
    return [axis.get("path", "") for axis in state.get("axes") or [] if axis.get("path")]


def axis_names_from_multidim_optimization(state: dict[str, Any]) -> list[str]:
    return [axis.get("path", "") for axis in state.get("axes") or [] if axis.get("path")]


def best_value_for_axis(best: dict[str, Any] | None, axis_name: str | None) -> Any:
    if not best or not axis_name:
        return None
    if "value" in best:
        return best.get("value")
    values = best.get("values") or {}
    if axis_name not in values and values:
        return values
    return values.get(axis_name)


def artifact_rows(artifacts: dict[str, str], base_dir: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for key in ["csv", "plot", "tecplot", "log"]:
        if key in artifacts:
            rows.append([key, markdown_link(Path(artifacts[key]).name, artifacts[key], base_dir)])
    for key, value in artifacts.items():
        if key not in {"csv", "plot", "tecplot", "log"}:
            rows.append([key, markdown_link(Path(value).name, value, base_dir)])
    return rows


def deck_lineage_rows(final_state: dict[str, Any] | None, base_dir: Path) -> list[list[Any]]:
    if not final_state:
        return []
    request = final_state.get("request") or {}
    active_mutation = request.get("active_deck_mutation") or {}
    repair_context = final_state.get("repair_context") or {}
    mutation_effect = final_state.get("mutation_effect_analysis") or {}
    sentaurus_effect = final_state.get("sentaurus_mutation_effect_analysis") or {}
    artifacts = final_artifacts(final_state)
    agent_policy = repair_context.get("agent_policy") if isinstance(repair_context, dict) else {}
    if not isinstance(agent_policy, dict):
        agent_policy = {}
    rows: list[list[Any]] = []
    if active_mutation:
        rows.append(["Active mutation", active_mutation.get("target") or active_mutation.get("name")])
        rows.append(["Mutation reason", active_mutation.get("reason")])
    if repair_context:
        rows.append(["Repair action", repair_context.get("action_name")])
        rows.append(["Parent state", markdown_link("parent", repair_context.get("parent_state_path"), base_dir)])
        rows.append(["Baseline state", markdown_link("baseline", repair_context.get("baseline_state_path"), base_dir)])
        rows.append(["Worth continuing", repair_context.get("worth_continuing_mutation")])
        rows.append(["Recommended next target", repair_context.get("recommended_next_target")])
    if mutation_effect:
        rows.append(["Mutation decision", mutation_effect.get("decision")])
        rows.append(["Mutation rationale", mutation_effect.get("rationale")])
        rows.append(["Primary metric", mutation_effect.get("primary_metric")])
        rows.append(["Improved metrics", mutation_effect.get("improved_metrics")])
        rows.append(["Regressed metrics", mutation_effect.get("regressed_metrics")])
    if sentaurus_effect:
        rows.append(["Sentaurus patch decision", sentaurus_effect.get("decision")])
        rows.append(["Sentaurus patch rationale", sentaurus_effect.get("rationale")])
        rows.append(["Sentaurus primary metric", sentaurus_effect.get("primary_metric")])
        rows.append(["Sentaurus improved metrics", sentaurus_effect.get("improved_metrics")])
        rows.append(["Sentaurus regressed metrics", sentaurus_effect.get("regressed_metrics")])
        if sentaurus_effect.get("tradeoff_violations"):
            rows.append(["Sentaurus tradeoffs", sentaurus_effect.get("tradeoff_violations")])
    observation = repair_context.get("agent_observation_summary") or agent_policy.get("observation_summary")
    hypothesis = repair_context.get("agent_hypothesis_zh") or agent_policy.get("hypothesis_zh")
    tool_plan = repair_context.get("agent_tool_plan") if isinstance(repair_context.get("agent_tool_plan"), list) else agent_policy.get("tool_plan")
    safety_review = (
        repair_context.get("agent_safety_review")
        if isinstance(repair_context.get("agent_safety_review"), dict)
        else agent_policy.get("safety_review")
    )
    if observation or hypothesis or tool_plan or safety_review:
        rows.append(["Agent observation", observation])
        rows.append(["Agent hypothesis", hypothesis])
        rows.append(["Agent tool plan", tool_plan])
        rows.append(["Agent safety review", safety_review])
    for key in [
        "deck_patch_history",
        "tcad_deck_ir",
        "semantic_deck_diff",
        "patched_source_deck",
        "tcad_deck_artifact",
        "baseline_mutation_overlay",
        "sentaurus_mutation_effect",
        "sentaurus_baseline_mutation_overlay",
        "sentaurus_lineage_archive",
    ]:
        if artifacts.get(key):
            rows.append([key, markdown_link(Path(artifacts[key]).name, artifacts[key], base_dir)])
    return rows


def render_best_section(
    *,
    best: dict[str, Any] | None,
    axis_name: str | None,
    base_dir: Path,
) -> tuple[str, str | None]:
    if not best:
        return "## Best Result\n\nNo completed result with an objective value was found.\n", None

    final_state = load_final_state(best.get("final_state_path"))
    artifacts = final_artifacts(final_state)
    metrics = final_metrics(final_state)
    plot_path = artifacts.get("plot")
    rows = [
        ["Task", best.get("task_id")],
        ["Axis", axis_name],
        ["Axis value", best_value_for_axis(best, axis_name)],
        ["Objective value", best.get("objective_value")],
        ["Status", best.get("status")],
        ["Quality", best.get("quality_status")],
        ["Final current A", metrics.get("final_total_current_a")],
        ["Max abs current A", metrics.get("max_abs_current_a")],
        ["Leakage current A", metrics.get("leakage_current_a")],
        ["Turn-on @ 1 uA V", metrics.get("turn_on_voltage_at_1ua_v")],
        ["Ideality factor", metrics.get("ideality_factor_estimate")],
        ["Last dV/dI ohm", metrics.get("differential_resistance_last_ohm")],
        ["Rectification ratio", metrics.get("rectification_ratio_final_to_leakage")],
        ["Breakdown @ 1 uA V", metrics.get("breakdown_voltage_at_1ua_v")],
        ["Points", metrics.get("points")],
    ]
    lines = ["## Best Result", "", table(["Field", "Value"], rows)]
    if plot_path:
        lines.extend(["", markdown_image("Best IV curve", plot_path, base_dir)])
    if artifacts:
        lines.extend(["", "### Best Artifacts", "", table(["Artifact", "Path"], artifact_rows(artifacts, base_dir))])
    lineage_rows = deck_lineage_rows(final_state, base_dir)
    if lineage_rows:
        lines.extend(["", "### Deck Patch Lineage", "", table(["Field", "Value"], lineage_rows)])
    return "\n".join(lines) + "\n", plot_path


def render_sweep_report(state: dict[str, Any], source_path: Path, output_path: Path) -> tuple[str, str | None]:
    base_dir = output_path.parent
    axis_names = axis_names_from_sweep(state)
    axis_name = axis_names[0] if axis_names else None
    best = best_sweep_case(state)
    best_section, plot_path = render_best_section(best=best, axis_name=axis_name, base_dir=base_dir)
    case_rows = []
    for case in sorted_by_objective(state.get("cases") or [], objective_direction(state)):
        values = case.get("values") or {}
        case_rows.append(
            [
                case.get("index"),
                case.get("task_id"),
                values.get(axis_name) if axis_name else format_value(values),
                case.get("objective_value"),
                case.get("quality_status"),
                case.get("status"),
                markdown_link("state", case.get("final_state_path"), base_dir),
            ]
        )

    overview_rows = [
        ["Sweep id", state.get("sweep_id")],
        ["Status", state.get("status")],
        ["Execute", state.get("execute")],
        ["Cases", len(state.get("cases") or [])],
        ["Objective", state.get("objective")],
        ["Source state", markdown_link(source_path.name, source_path, base_dir)],
        ["Summary CSV", markdown_link("summary.csv", state.get("summary_csv_path"), base_dir)],
    ]
    lines = [
        f"# TCAD Sweep Report: {state.get('sweep_id')}",
        "",
        f"Generated: {utc_timestamp()}",
        "",
        "## Overview",
        "",
        table(["Field", "Value"], overview_rows),
        "",
        best_section.rstrip(),
        "",
        "## Ranked Cases",
        "",
        table(["Index", "Task", axis_name or "Values", "Objective", "Quality", "Status", "State"], case_rows)
        if case_rows
        else "No completed cases with objective values were found.",
        "",
    ]
    return "\n".join(lines), plot_path


def render_optimization_report(state: dict[str, Any], source_path: Path, output_path: Path) -> tuple[str, str | None]:
    base_dir = output_path.parent
    axis = state.get("axis") or {}
    axis_name = axis.get("path")
    best = best_optimization_observation(state)
    best_section, plot_path = render_best_section(best=best, axis_name=axis_name, base_dir=base_dir)

    round_rows = [
        [
            round_state.get("index"),
            round_state.get("sweep_id"),
            round_state.get("status"),
            round_state.get("values"),
            markdown_link("summary.csv", round_state.get("summary_csv_path"), base_dir),
            markdown_link("sweep_state", round_state.get("sweep_state_path"), base_dir),
        ]
        for round_state in state.get("rounds") or []
    ]
    observation_rows = []
    for rank, observation in enumerate(
        sorted_by_objective(state.get("observations") or [], objective_direction(state)),
        start=1,
    ):
        observation_rows.append(
            [
                rank,
                observation.get("round_index"),
                observation.get("case_index"),
                observation.get("task_id"),
                observation.get("value"),
                observation.get("objective_value"),
                observation.get("quality_status"),
                observation.get("status"),
                markdown_link("state", observation.get("final_state_path"), base_dir),
            ]
        )

    overview_rows = [
        ["Optimize id", state.get("optimize_id")],
        ["Status", state.get("status")],
        ["Execute", state.get("execute")],
        ["Rounds", len(state.get("rounds") or [])],
        ["Observations", len(state.get("observations") or [])],
        ["Axis", axis],
        ["Objective", state.get("objective")],
        ["Next action", state.get("next_action")],
        ["Source state", markdown_link(source_path.name, source_path, base_dir)],
    ]
    lines = [
        f"# TCAD Optimization Report: {state.get('optimize_id')}",
        "",
        f"Generated: {utc_timestamp()}",
        "",
        "## Overview",
        "",
        table(["Field", "Value"], overview_rows),
        "",
        best_section.rstrip(),
        "",
        "## Rounds",
        "",
        table(["Round", "Sweep", "Status", "Values", "Summary", "State"], round_rows)
        if round_rows
        else "No optimization rounds were recorded.",
        "",
        "## Ranked Observations",
        "",
        table(
            ["Rank", "Round", "Case", "Task", axis_name or "Value", "Objective", "Quality", "Status", "State"],
            observation_rows,
        )
        if observation_rows
        else "No completed observations with objective values were found.",
        "",
    ]
    return "\n".join(lines), plot_path


def render_multidim_optimization_report(
    state: dict[str, Any],
    source_path: Path,
    output_path: Path,
) -> tuple[str, str | None]:
    base_dir = output_path.parent
    axis_names = axis_names_from_multidim_optimization(state)
    axis_label = ", ".join(axis_names) if axis_names else None
    best = best_optimization_observation(state)
    best_section, plot_path = render_best_section(best=best, axis_name=axis_label, base_dir=base_dir)

    round_rows = []
    for round_state in state.get("rounds") or []:
        sweep_links = []
        for index, sweep_path in enumerate(round_state.get("sweep_state_paths") or [], start=1):
            sweep_links.append(markdown_link(f"state {index}", sweep_path, base_dir))
        round_rows.append(
            [
                round_state.get("index"),
                round_state.get("round_id"),
                round_state.get("status"),
                len(round_state.get("candidate_values") or []),
                markdown_link("summary.csv", round_state.get("summary_csv_path"), base_dir),
                ", ".join(sweep_links),
            ]
        )

    observation_rows = []
    for rank, observation in enumerate(
        sorted_by_objective(state.get("observations") or [], objective_direction(state)),
        start=1,
    ):
        observation_rows.append(
            [
                rank,
                observation.get("round_index"),
                observation.get("point_index"),
                observation.get("task_id"),
                observation.get("values"),
                observation.get("objective_value"),
                observation.get("quality_status"),
                observation.get("status"),
                markdown_link("state", observation.get("final_state_path"), base_dir),
            ]
        )

    overview_rows = [
        ["Optimize id", state.get("optimize_id")],
        ["Status", state.get("status")],
        ["Execute", state.get("execute")],
        ["Rounds", len(state.get("rounds") or [])],
        ["Observations", len(state.get("observations") or [])],
        ["Axes", state.get("axes")],
        ["Objective", state.get("objective")],
        ["Max cases per round", state.get("max_cases_per_round")],
        ["Next action", state.get("next_action")],
        ["Source state", markdown_link(source_path.name, source_path, base_dir)],
    ]
    lines = [
        f"# TCAD Multi-Dimensional Optimization Report: {state.get('optimize_id')}",
        "",
        f"Generated: {utc_timestamp()}",
        "",
        "## Overview",
        "",
        table(["Field", "Value"], overview_rows),
        "",
        best_section.rstrip(),
        "",
        "## Rounds",
        "",
        table(["Round", "Round id", "Status", "Candidate points", "Summary", "Sweep states"], round_rows)
        if round_rows
        else "No optimization rounds were recorded.",
        "",
        "## Ranked Observations",
        "",
        table(
            ["Rank", "Round", "Point", "Task", "Values", "Objective", "Quality", "Status", "State"],
            observation_rows,
        )
        if observation_rows
        else "No completed observations with objective values were found.",
        "",
    ]
    return "\n".join(lines), plot_path


def generate_experiment_report(source: Path, output_path: Path | None = None) -> ReportResult:
    try:
        state_path = resolve_state_path(source).resolve()
        state = read_json(state_path)
        kind = detect_report_kind(state)
        report_path = (output_path or state_path.with_name("report.md")).resolve()
        if kind == ReportKind.ADAPTIVE_OPTIMIZATION:
            content, plot_path = render_optimization_report(state, state_path, report_path)
        elif kind == ReportKind.MULTIDIM_OPTIMIZATION:
            content, plot_path = render_multidim_optimization_report(state, state_path, report_path)
        else:
            content, plot_path = render_sweep_report(state, state_path, report_path)
        write_text(report_path, content)
        return ReportResult(
            status=ReportStatus.COMPLETED,
            kind=kind,
            source_state_path=str(state_path),
            report_path=str(report_path),
            best_artifact_plot_path=plot_path,
        )
    except Exception as exc:
        return ReportResult(
            status=ReportStatus.FAILED,
            source_state_path=str(source),
            failure_reason=str(exc),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Markdown report for a TCAD run state.")
    parser.add_argument("--state", type=Path, required=True, help="State file or containing run directory.")
    parser.add_argument("--output", type=Path, default=None, help="Markdown report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_experiment_report(args.state, args.output)
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.status == ReportStatus.COMPLETED else 2)


if __name__ == "__main__":
    main()
