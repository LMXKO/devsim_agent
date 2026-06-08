from __future__ import annotations

import html
import json
import math
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from tcad_agent.reporting import (
    ReportKind,
    best_optimization_observation,
    best_sweep_case,
    best_value_for_axis,
    detect_report_kind,
    final_artifacts,
    final_metrics,
    load_final_state,
    objective_direction,
    read_json,
    resolve_state_path,
    sorted_by_objective,
    write_text,
)


class DashboardStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class DashboardResult(BaseModel):
    tool_name: str = "experiment_dashboard"
    status: DashboardStatus
    kind: ReportKind | None = None
    source_state_path: str
    dashboard_path: str | None = None
    best_artifact_plot_path: str | None = None
    failure_reason: str | None = None


class DashboardPoint(BaseModel):
    rank: int | None = None
    task_id: str | None = None
    round_index: int | None = None
    case_index: int | None = None
    value: float | None = None
    values: dict[str, Any] | None = None
    objective_value: float | None = None
    status: str | None = None
    quality_status: str | None = None
    final_state_path: str | None = None
    plot_path: str | None = None
    csv_path: str | None = None
    log_path: str | None = None
    final_current_a: float | None = None
    max_abs_current_a: float | None = None
    points: int | None = None
    leakage_current_a: float | None = None
    turn_on_voltage_at_1ua_v: float | None = None
    ideality_factor_estimate: float | None = None
    differential_resistance_last_ohm: float | None = None
    active_deck_mutation: str | None = None
    deck_patch_decision: str | None = None
    deck_patch_rationale: str | None = None
    recommended_next_target: str | None = None
    deck_patch_history_path: str | None = None
    semantic_deck_diff_path: str | None = None
    baseline_mutation_overlay_path: str | None = None
    agent_observation_summary: str | None = None
    agent_hypothesis_zh: str | None = None
    agent_tool_plan: list[dict[str, Any]] | None = None
    agent_safety_review: dict[str, Any] | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def h(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


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
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def dashboard_target(path: str | Path | None, base_dir: Path) -> str:
    if not path:
        return ""
    target = Path(path)
    if target.is_absolute():
        rendered = os.path.relpath(target, base_dir)
    else:
        rendered = str(target)
    return rendered.replace(os.sep, "/")


def link(label: str, path: str | Path | None, base_dir: Path, class_name: str = "") -> str:
    if not path:
        return ""
    class_attr = f' class="{h(class_name)}"' if class_name else ""
    return f'<a{class_attr} href="{h(dashboard_target(path, base_dir))}">{h(label)}</a>'


def value_from_case(case: dict[str, Any], axis_name: str | None) -> float | None:
    if "value" in case:
        try:
            return float(case["value"])
        except (TypeError, ValueError):
            return None
    if not axis_name:
        return None
    values = case.get("values") or {}
    try:
        return float(values.get(axis_name))
    except (TypeError, ValueError):
        return None


def point_from_item(item: dict[str, Any], axis_name: str | None, rank: int | None = None) -> DashboardPoint:
    final_state = load_final_state(item.get("final_state_path"))
    artifacts = final_artifacts(final_state)
    metrics = final_metrics(final_state)
    request = (final_state or {}).get("request") or {}
    active_mutation = request.get("active_deck_mutation") if isinstance(request.get("active_deck_mutation"), dict) else {}
    mutation_effect = (final_state or {}).get("mutation_effect_analysis") or {}
    repair_context = (final_state or {}).get("repair_context") or {}
    agent_policy = repair_context.get("agent_policy") if isinstance(repair_context, dict) else {}
    if not isinstance(agent_policy, dict):
        agent_policy = {}
    try:
        objective = float(item.get("objective_value")) if item.get("objective_value") is not None else None
    except (TypeError, ValueError):
        objective = None
    return DashboardPoint(
        rank=rank,
        task_id=item.get("task_id"),
        round_index=item.get("round_index"),
        case_index=item.get("case_index") or item.get("index"),
        value=value_from_case(item, axis_name),
        values=item.get("values") if isinstance(item.get("values"), dict) else None,
        objective_value=objective,
        status=item.get("status"),
        quality_status=item.get("quality_status"),
        final_state_path=item.get("final_state_path"),
        plot_path=artifacts.get("plot"),
        csv_path=artifacts.get("csv"),
        log_path=artifacts.get("log"),
        final_current_a=float(metrics["final_total_current_a"])
        if metrics.get("final_total_current_a") is not None
        else None,
        max_abs_current_a=float(metrics["max_abs_current_a"]) if metrics.get("max_abs_current_a") is not None else None,
        points=int(metrics["points"]) if metrics.get("points") is not None else None,
        leakage_current_a=float(metrics["leakage_current_a"]) if metrics.get("leakage_current_a") is not None else None,
        turn_on_voltage_at_1ua_v=float(metrics["turn_on_voltage_at_1ua_v"])
        if metrics.get("turn_on_voltage_at_1ua_v") is not None
        else None,
        ideality_factor_estimate=float(metrics["ideality_factor_estimate"])
        if metrics.get("ideality_factor_estimate") is not None
        else None,
        differential_resistance_last_ohm=float(metrics["differential_resistance_last_ohm"])
        if metrics.get("differential_resistance_last_ohm") is not None
        else None,
        active_deck_mutation=(active_mutation.get("target") or active_mutation.get("name")) if active_mutation else None,
        deck_patch_decision=mutation_effect.get("decision") if isinstance(mutation_effect, dict) else None,
        deck_patch_rationale=mutation_effect.get("rationale") if isinstance(mutation_effect, dict) else None,
        recommended_next_target=(
            repair_context.get("recommended_next_target")
            if isinstance(repair_context, dict)
            else None
        ),
        deck_patch_history_path=artifacts.get("deck_patch_history"),
        semantic_deck_diff_path=artifacts.get("semantic_deck_diff"),
        baseline_mutation_overlay_path=artifacts.get("baseline_mutation_overlay"),
        agent_observation_summary=(
            repair_context.get("agent_observation_summary")
            or agent_policy.get("observation_summary")
            if isinstance(repair_context, dict)
            else agent_policy.get("observation_summary")
        ),
        agent_hypothesis_zh=(
            repair_context.get("agent_hypothesis_zh")
            or agent_policy.get("hypothesis_zh")
            if isinstance(repair_context, dict)
            else agent_policy.get("hypothesis_zh")
        ),
        agent_tool_plan=(
            repair_context.get("agent_tool_plan")
            if isinstance(repair_context.get("agent_tool_plan"), list)
            else agent_policy.get("tool_plan")
            if isinstance(agent_policy.get("tool_plan"), list)
            else []
        ),
        agent_safety_review=(
            repair_context.get("agent_safety_review")
            if isinstance(repair_context.get("agent_safety_review"), dict)
            else agent_policy.get("safety_review")
            if isinstance(agent_policy.get("safety_review"), dict)
            else {}
        ),
    )


def ranked_points(items: list[dict[str, Any]], axis_name: str | None, direction: str) -> list[DashboardPoint]:
    ranked = sorted_by_objective(items, direction)
    return [point_from_item(item, axis_name, rank=index) for index, item in enumerate(ranked, start=1)]


def optimization_axis_name(state: dict[str, Any]) -> str | None:
    axis = state.get("axis") or {}
    return axis.get("path")


def sweep_axis_name(state: dict[str, Any]) -> str | None:
    axes = state.get("axes") or []
    if not axes:
        return None
    return axes[0].get("path")


def axis_names_from_state(state: dict[str, Any], kind: ReportKind) -> list[str]:
    if kind == ReportKind.ADAPTIVE_OPTIMIZATION:
        axis_name = optimization_axis_name(state)
        return [axis_name] if axis_name else []
    axes = state.get("axes") or []
    return [axis.get("path", "") for axis in axes if axis.get("path")]


def objective_label(state: dict[str, Any]) -> str:
    objective = state.get("objective") or {}
    metric = objective.get("metric_path") or "objective"
    direction = objective.get("direction") or "minimize"
    absolute = " abs" if objective.get("absolute") else ""
    return f"{direction}{absolute} {metric}"


def should_use_log_x(state: dict[str, Any], points: list[DashboardPoint]) -> bool:
    if (state.get("axis") or {}).get("scale") == "log":
        return True
    values = [point.value for point in points if point.value and point.value > 0]
    if len(values) < 2:
        return False
    return max(values) / min(values) >= 100.0


def chart_svg(points: list[DashboardPoint], *, log_x: bool) -> str:
    chart_points = [
        point
        for point in points
        if point.value is not None and point.objective_value is not None and point.status == "completed"
    ]
    if len(chart_points) < 2:
        return '<div class="empty">Not enough completed points for a trend chart.</div>'

    def x_transform(value: float) -> float:
        return math.log10(value) if log_x and value > 0 else value

    x_values = [x_transform(point.value or 0.0) for point in chart_points]
    y_values = [point.objective_value or 0.0 for point in chart_points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if math.isclose(x_min, x_max):
        x_min -= 1.0
        x_max += 1.0
    if math.isclose(y_min, y_max):
        y_min -= abs(y_min) * 0.1 or 1.0
        y_max += abs(y_max) * 0.1 or 1.0

    width = 760
    height = 280
    left = 62
    right = 24
    top = 20
    bottom = 48
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(value: float) -> float:
        return left + (x_transform(value) - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    ordered = sorted(chart_points, key=lambda point: x_transform(point.value or 0.0))
    polyline = " ".join(f"{sx(point.value or 0.0):.2f},{sy(point.objective_value or 0.0):.2f}" for point in ordered)
    circles = []
    for point in ordered:
        x = sx(point.value or 0.0)
        y = sy(point.objective_value or 0.0)
        label = f"{format_value(point.value)} -> {format_value(point.objective_value)}"
        circles.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5"><title>{h(label)}</title></circle>'
        )

    x_ticks = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    x_labels = []
    for point in x_ticks:
        x = sx(point.value or 0.0)
        x_labels.append(
            f'<text x="{x:.2f}" y="{height - 16}" text-anchor="middle">{h(format_value(point.value))}</text>'
        )
    y_labels = []
    for value in [y_min, (y_min + y_max) / 2.0, y_max]:
        y = sy(value)
        y_labels.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{h(format_value(value))}</text>')

    return f"""
<svg class="trend-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Objective trend chart">
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" class="axis-line" />
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" class="axis-line" />
  <line x1="{left}" y1="{top}" x2="{width - right}" y2="{top}" class="grid-line" />
  <line x1="{left}" y1="{top + plot_h / 2:.2f}" x2="{width - right}" y2="{top + plot_h / 2:.2f}" class="grid-line" />
  <polyline points="{polyline}" class="trend-line" />
  {''.join(circles)}
  {''.join(x_labels)}
  {''.join(y_labels)}
  <text x="{left + plot_w / 2:.2f}" y="{height - 2}" text-anchor="middle" class="axis-title">axis value</text>
  <text x="14" y="{top + plot_h / 2:.2f}" text-anchor="middle" transform="rotate(-90 14 {top + plot_h / 2:.2f})" class="axis-title">objective</text>
</svg>
""".strip()


def heatmap_items(
    items: list[dict[str, Any]],
    axis_names: list[str],
    direction: str,
) -> list[dict[str, Any]]:
    if len(axis_names) != 2:
        return []
    best_by_coord: dict[tuple[float, float], dict[str, Any]] = {}
    reverse = direction == "maximize"
    for item in items:
        values = item.get("values") or {}
        if item.get("objective_value") is None:
            continue
        try:
            x = float(values[axis_names[0]])
            y = float(values[axis_names[1]])
            objective = float(item["objective_value"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (x, y)
        current = best_by_coord.get(key)
        if current is None:
            best_by_coord[key] = {"x": x, "y": y, "objective": objective, "item": item}
            continue
        current_objective = float(current["objective"])
        if (reverse and objective > current_objective) or (not reverse and objective < current_objective):
            best_by_coord[key] = {"x": x, "y": y, "objective": objective, "item": item}
    return list(best_by_coord.values())


def heat_color(fraction: float) -> str:
    clamped = max(0.0, min(1.0, fraction))
    anchors = [
        (0.0, (21, 128, 61)),
        (0.5, (234, 179, 8)),
        (1.0, (194, 65, 12)),
    ]
    for (left_stop, left_rgb), (right_stop, right_rgb) in zip(anchors[:-1], anchors[1:]):
        if clamped <= right_stop:
            local = (clamped - left_stop) / (right_stop - left_stop)
            rgb = [
                round(left_rgb[index] + (right_rgb[index] - left_rgb[index]) * local)
                for index in range(3)
            ]
            return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
    return "rgb(194, 65, 12)"


def heatmap_svg(items: list[dict[str, Any]], axis_names: list[str], direction: str) -> str:
    points = heatmap_items(items, axis_names, direction)
    if len(points) < 2:
        return '<div class="empty">Not enough completed two-axis points for a heatmap.</div>'

    x_values = sorted({point["x"] for point in points})
    y_values = sorted({point["y"] for point in points}, reverse=True)
    objectives = [float(point["objective"]) for point in points]
    best = max(objectives) if direction == "maximize" else min(objectives)
    worst = min(objectives) if direction == "maximize" else max(objectives)
    span = abs(worst - best)
    by_coord = {(point["x"], point["y"]): point for point in points}

    cell = 58
    left = 150
    top = 26
    right = 22
    bottom = 78
    width = left + len(x_values) * cell + right
    height = top + len(y_values) * cell + bottom
    rects = []
    for row, y_value in enumerate(y_values):
        for col, x_value in enumerate(x_values):
            x = left + col * cell
            y = top + row * cell
            point = by_coord.get((x_value, y_value))
            if point:
                objective = float(point["objective"])
                if span == 0:
                    fraction = 0.0
                elif direction == "maximize":
                    fraction = (best - objective) / span
                else:
                    fraction = (objective - best) / span
                fill = heat_color(fraction)
                item = point["item"]
                label = (
                    f"{axis_names[0]}={format_value(x_value)}, "
                    f"{axis_names[1]}={format_value(y_value)}, "
                    f"objective={format_value(objective)}, task={item.get('task_id')}"
                )
                rects.append(
                    f'<rect class="heat-cell" x="{x}" y="{y}" width="{cell - 4}" height="{cell - 4}" '
                    f'fill="{h(fill)}"><title>{h(label)}</title></rect>'
                )
                rects.append(
                    f'<text x="{x + (cell - 4) / 2:.2f}" y="{y + cell / 2 + 4:.2f}" '
                    f'text-anchor="middle" class="heat-value">{h(format_value(objective))}</text>'
                )
            else:
                rects.append(
                    f'<rect class="heat-missing" x="{x}" y="{y}" width="{cell - 4}" height="{cell - 4}">'
                    f"<title>{h(axis_names[0])}={h(format_value(x_value))}, "
                    f"{h(axis_names[1])}={h(format_value(y_value))}: not sampled</title></rect>"
                )

    x_labels = []
    for col, x_value in enumerate(x_values):
        x = left + col * cell + (cell - 4) / 2
        x_labels.append(
            f'<text x="{x:.2f}" y="{height - 36}" text-anchor="middle">{h(format_value(x_value))}</text>'
        )
    y_labels = []
    for row, y_value in enumerate(y_values):
        y = top + row * cell + cell / 2 + 4
        y_labels.append(f'<text x="{left - 12}" y="{y:.2f}" text-anchor="end">{h(format_value(y_value))}</text>')

    return f"""
<svg class="heatmap-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Objective heatmap">
  {''.join(rects)}
  {''.join(x_labels)}
  {''.join(y_labels)}
  <text x="{left + len(x_values) * cell / 2:.2f}" y="{height - 8}" text-anchor="middle" class="axis-title">{h(axis_names[0])}</text>
  <text x="18" y="{top + len(y_values) * cell / 2:.2f}" text-anchor="middle" transform="rotate(-90 18 {top + len(y_values) * cell / 2:.2f})" class="axis-title">{h(axis_names[1])}</text>
  <text x="{width - right}" y="16" text-anchor="end" class="axis-title">best {h(format_value(best))}</text>
</svg>
""".strip()


def status_counts(points: list[DashboardPoint]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for point in points:
        key = point.status or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def failure_counts(points: list[DashboardPoint]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for point in points:
        key = point.quality_status or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def stat_block(label: str, value: Any, hint: str = "") -> str:
    return f"""
<div class="stat">
  <div class="stat-label">{h(label)}</div>
  <div class="stat-value">{h(format_value(value))}</div>
  <div class="stat-hint">{h(hint)}</div>
</div>
""".strip()


def data_table(points: list[DashboardPoint], axis_name: str | None, base_dir: Path) -> str:
    if not points:
        return '<div class="empty">No ranked results were found.</div>'
    rows = []
    for point in points:
        rendered_value = point.value if point.value is not None else point.values
        rows.append(
            "<tr>"
            f"<td>{h(point.rank)}</td>"
            f"<td>{h(point.round_index)}</td>"
            f"<td>{h(point.case_index)}</td>"
            f"<td class=\"task-id\">{h(point.task_id)}</td>"
            f"<td>{h(format_value(rendered_value))}</td>"
            f"<td>{h(format_value(point.objective_value))}</td>"
            f"<td>{h(point.quality_status)}</td>"
            f"<td>{h(point.status)}</td>"
            f"<td>{link('state', point.final_state_path, base_dir)} {link('csv', point.csv_path, base_dir)} {link('log', point.log_path, base_dir)}</td>"
            "</tr>"
        )
    return f"""
<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th>Rank</th>
        <th>Round</th>
        <th>Case</th>
        <th>Task</th>
        <th>{h(axis_name or "Value")}</th>
        <th>Objective</th>
        <th>Quality</th>
        <th>Status</th>
        <th>Artifacts</th>
      </tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</div>
""".strip()


def rounds_table(state: dict[str, Any], base_dir: Path) -> str:
    rounds = state.get("rounds") or []
    if not rounds:
        return ""
    rows = []
    for round_state in rounds:
        sweep_links = []
        if round_state.get("sweep_state_path"):
            sweep_links.append(link("state", round_state.get("sweep_state_path"), base_dir))
        for index, sweep_path in enumerate(round_state.get("sweep_state_paths") or [], start=1):
            sweep_links.append(link(f"state {index}", sweep_path, base_dir))
        summary_link = link("summary", round_state.get("summary_csv_path"), base_dir)
        values = round_state.get("values")
        if values is None:
            values = round_state.get("candidate_values")
        rows.append(
            "<tr>"
            f"<td>{h(round_state.get('index'))}</td>"
            f"<td class=\"task-id\">{h(round_state.get('sweep_id') or round_state.get('round_id'))}</td>"
            f"<td>{h(round_state.get('status'))}</td>"
            f"<td>{h(format_value(values))}</td>"
            f"<td>{summary_link} {' '.join(sweep_links)}</td>"
            "</tr>"
        )
    return f"""
<section class="band">
  <div class="section-head">
    <h2>Rounds</h2>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Round</th><th>Sweep</th><th>Status</th><th>Values</th><th>Files</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</section>
""".strip()


def best_plot(point: DashboardPoint | None, base_dir: Path) -> str:
    if not point or not point.plot_path:
        return '<div class="plot-placeholder">No IV plot artifact was found for the best result.</div>'
    return f'<img class="iv-plot" src="{h(dashboard_target(point.plot_path, base_dir))}" alt="Best IV curve">'


def deck_lineage_panel(point: DashboardPoint | None, base_dir: Path) -> str:
    if not point or not any([point.active_deck_mutation, point.deck_patch_decision, point.deck_patch_history_path]):
        return ""
    tool_plan_items = []
    for item in (point.agent_tool_plan or [])[:4]:
        if isinstance(item, dict):
            label = item.get("tool") or "agent tool"
            detail = item.get("expected_evidence") or item.get("why") or ""
            tool_plan_items.append(f"<li><strong>{h(label)}</strong><span>{h(detail)}</span></li>")
    safety_review = point.agent_safety_review or {}
    safety_text = ""
    if safety_review:
        safety_text = ", ".join(
            f"{key}: {format_value(value)}"
            for key, value in safety_review.items()
            if value not in (None, "", [])
        )
    links = " ".join(
        item
        for item in [
            link("patch history", point.deck_patch_history_path, base_dir, "pill-link"),
            link("semantic diff", point.semantic_deck_diff_path, base_dir, "pill-link"),
            link("curve overlay", point.baseline_mutation_overlay_path, base_dir, "pill-link"),
        ]
        if item
    )
    agent_rows = "".join(
        row
        for row in [
            f'<div class="lineage-row"><span>Observation</span><strong>{h(point.agent_observation_summary)}</strong></div>'
            if point.agent_observation_summary
            else "",
            f'<div class="lineage-row"><span>Hypothesis</span><strong>{h(point.agent_hypothesis_zh)}</strong></div>'
            if point.agent_hypothesis_zh
            else "",
            f'<div class="lineage-row"><span>Safety</span><strong>{h(safety_text)}</strong></div>' if safety_text else "",
        ]
    )
    tool_plan_html = f'<ul class="lineage-tool-plan">{"".join(tool_plan_items)}</ul>' if tool_plan_items else ""
    return f"""
          <div class="lineage">
            <div class="lineage-title">Deck Patch Lineage</div>
            <div class="lineage-row"><span>Mutation</span><strong>{h(point.active_deck_mutation)}</strong></div>
            <div class="lineage-row"><span>Decision</span><strong>{h(point.deck_patch_decision)}</strong></div>
            <div class="lineage-row"><span>Next target</span><strong>{h(point.recommended_next_target)}</strong></div>
            {agent_rows}
            {tool_plan_html}
            <div class="lineage-note">{h(point.deck_patch_rationale)}</div>
            <div class="lineage-links">{links}</div>
          </div>
""".strip()


def count_summary_text(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}: {value}" for key, value in sorted(counts.items())) or "none"


def render_dashboard(
    *,
    state: dict[str, Any],
    kind: ReportKind,
    source_path: Path,
    output_path: Path,
) -> tuple[str, str | None]:
    base_dir = output_path.parent
    direction = objective_direction(state)
    if kind == ReportKind.ADAPTIVE_OPTIMIZATION:
        axis_name = optimization_axis_name(state)
        axis_names = axis_names_from_state(state, kind)
        items = state.get("observations") or []
        best_item = best_optimization_observation(state)
        title = f"TCAD Optimization Dashboard: {state.get('optimize_id')}"
        rounds = len(state.get("rounds") or [])
        total_label = "Observations"
        total_count = len(items)
    elif kind == ReportKind.MULTIDIM_OPTIMIZATION:
        axis_names = axis_names_from_state(state, kind)
        axis_name = ", ".join(axis_names) if axis_names else None
        items = state.get("observations") or []
        best_item = best_optimization_observation(state)
        title = f"TCAD Multi-Dimensional Optimization Dashboard: {state.get('optimize_id')}"
        rounds = len(state.get("rounds") or [])
        total_label = "Observations"
        total_count = len(items)
    else:
        axis_name = sweep_axis_name(state)
        axis_names = axis_names_from_state(state, kind)
        items = state.get("cases") or []
        best_item = best_sweep_case(state)
        title = f"TCAD Sweep Dashboard: {state.get('sweep_id')}"
        rounds = 1
        total_label = "Cases"
        total_count = len(items)

    points = ranked_points(items, axis_name, direction)
    best_point = point_from_item(best_item, axis_name, rank=1) if best_item else None
    log_x = should_use_log_x(state, points)
    use_heatmap = len(axis_names) == 2
    chart_html = heatmap_svg(items, axis_names, direction) if use_heatmap else chart_svg(points, log_x=log_x)
    chart_title = "Objective Heatmap" if use_heatmap else "Objective Trend"
    chart_meta = ", ".join(axis_names) if use_heatmap else ("log x-axis" if log_x else "linear x-axis")
    plot_path = best_point.plot_path if best_point else None
    source_link = link(Path(source_path).name, source_path, base_dir)
    best_axis_value = None
    if best_point:
        best_axis_value = best_point.value if best_point.value is not None else best_point.values

    stats = [
        stat_block("Status", state.get("status"), "execution state"),
        stat_block("Rounds", rounds, "completed or planned"),
        stat_block(total_label, total_count, count_summary_text(status_counts(points))),
        stat_block("Best objective", best_point.objective_value if best_point else None, objective_label(state)),
        stat_block("Best axis value", best_axis_value, axis_name or ""),
        stat_block("Quality", best_point.quality_status if best_point else None, count_summary_text(failure_counts(points))),
    ]

    top_artifact_links = ""
    if best_point:
        top_artifact_links = " ".join(
            item
            for item in [
                link("state", best_point.final_state_path, base_dir, "pill-link"),
                link("csv", best_point.csv_path, base_dir, "pill-link"),
                link("plot", best_point.plot_path, base_dir, "pill-link"),
                link("overlay", best_point.baseline_mutation_overlay_path, base_dir, "pill-link"),
                link("log", best_point.log_path, base_dir, "pill-link"),
            ]
            if item
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{h(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8faf9;
      --ink: #17201b;
      --muted: #65716b;
      --line: #d8e0dc;
      --panel: #ffffff;
      --accent: #0f766e;
      --accent-ink: #0b4f49;
      --warn: #9a5b00;
      --good-bg: #e8f4ef;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }}
    a {{ color: var(--accent-ink); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .shell {{ max-width: 1180px; margin: 0 auto; padding: 28px 24px 44px; }}
    header {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 16px;
      align-items: end;
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
    }}
    h1 {{ margin: 0; font-size: 26px; font-weight: 720; letter-spacing: 0; }}
    h2 {{ margin: 0; font-size: 17px; letter-spacing: 0; }}
    .meta {{ color: var(--muted); margin-top: 7px; }}
    .source {{ text-align: right; color: var(--muted); }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin: 18px 0;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 96px;
    }}
    .stat-label {{ color: var(--muted); font-size: 12px; }}
    .stat-value {{ margin-top: 7px; font-size: 20px; font-weight: 720; word-break: break-word; }}
    .stat-hint {{ margin-top: 6px; color: var(--muted); font-size: 12px; word-break: break-word; }}
    .band {{
      border-top: 1px solid var(--line);
      padding: 22px 0;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(280px, 0.85fr);
      gap: 18px;
      align-items: start;
    }}
    .chart-panel, .plot-panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 314px;
    }}
    .trend-chart {{ width: 100%; height: 280px; display: block; }}
    .heatmap-chart {{ width: 100%; min-height: 280px; display: block; }}
    .axis-line {{ stroke: #93a09a; stroke-width: 1; }}
    .grid-line {{ stroke: #e6ece9; stroke-width: 1; }}
    .trend-line {{ fill: none; stroke: var(--accent); stroke-width: 2.2; }}
    circle {{ fill: #ffffff; stroke: var(--accent); stroke-width: 2; }}
    .heat-cell {{ stroke: #ffffff; stroke-width: 2; rx: 6; }}
    .heat-missing {{ fill: #eef3f0; stroke: #ffffff; stroke-width: 2; rx: 6; }}
    .heat-value {{ fill: #111827; font-size: 10px; font-weight: 700; }}
    text {{ fill: var(--muted); font-size: 11px; }}
    .axis-title {{ font-size: 12px; }}
    .iv-plot {{
      width: 100%;
      max-height: 280px;
      object-fit: contain;
      display: block;
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
    .plot-placeholder, .empty {{
      min-height: 220px;
      display: grid;
      place-items: center;
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: #fbfdfc;
      text-align: center;
      padding: 20px;
    }}
    .best-meta {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }}
    .kv {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 10px;
    }}
    .kv span {{ display: block; color: var(--muted); font-size: 12px; }}
    .kv strong {{ display: block; margin-top: 4px; font-size: 15px; word-break: break-word; }}
    .pill-link {{
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 10px;
      margin: 0 4px 4px 0;
      background: var(--good-bg);
    }}
    .lineage {{
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdfc;
      padding: 12px;
    }}
    .lineage-title {{ font-weight: 720; margin-bottom: 8px; }}
    .lineage-row {{ display: grid; grid-template-columns: 112px minmax(0, 1fr); gap: 8px; margin-top: 6px; }}
    .lineage-row span {{ color: var(--muted); }}
    .lineage-row strong {{ word-break: break-word; }}
    .lineage-note {{ color: var(--muted); margin-top: 8px; }}
    .lineage-links {{ margin-top: 8px; }}
    .lineage-tool-plan {{
      margin: 10px 0 0;
      padding-left: 18px;
      color: var(--muted);
    }}
    .lineage-tool-plan li {{ margin: 5px 0; }}
    .lineage-tool-plan strong {{ color: var(--ink); margin-right: 6px; }}
    .lineage-tool-plan span {{ word-break: break-word; }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    table {{ width: 100%; border-collapse: collapse; min-width: 860px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 650; background: #f2f6f4; }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    .task-id {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    footer {{ color: var(--muted); border-top: 1px solid var(--line); padding-top: 14px; margin-top: 8px; }}
    @media (max-width: 960px) {{
      header {{ grid-template-columns: 1fr; }}
      .source {{ text-align: left; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .grid-2 {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 560px) {{
      .shell {{ padding: 20px 14px 32px; }}
      .stats {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 22px; }}
      .best-meta {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>{h(title)}</h1>
        <div class="meta">Generated {h(utc_timestamp())}</div>
      </div>
      <div class="source">Source {source_link}</div>
    </header>

    <div class="stats">{''.join(stats)}</div>

    <section class="band">
      <div class="section-head">
        <h2>{h(chart_title)}</h2>
        <div class="meta">{h(chart_meta)}</div>
      </div>
      <div class="grid-2">
        <div class="chart-panel">{chart_html}</div>
        <div class="plot-panel">
          {best_plot(best_point, base_dir)}
          <div class="best-meta">
            <div class="kv"><span>Best task</span><strong>{h(best_point.task_id if best_point else None)}</strong></div>
            <div class="kv"><span>Artifacts</span><strong>{top_artifact_links}</strong></div>
            <div class="kv"><span>Ideality factor</span><strong>{h(format_value(best_point.ideality_factor_estimate if best_point else None))}</strong></div>
            <div class="kv"><span>Turn-on @ 1 uA</span><strong>{h(format_value(best_point.turn_on_voltage_at_1ua_v if best_point else None))} V</strong></div>
            <div class="kv"><span>Leakage current</span><strong>{h(format_value(best_point.leakage_current_a if best_point else None))} A</strong></div>
            <div class="kv"><span>Last dV/dI</span><strong>{h(format_value(best_point.differential_resistance_last_ohm if best_point else None))} ohm</strong></div>
          </div>
          {deck_lineage_panel(best_point, base_dir)}
        </div>
      </div>
    </section>

    {rounds_table(state, base_dir)}

    <section class="band">
      <div class="section-head">
        <h2>Ranked Results</h2>
        <div class="meta">{h(objective_label(state))}</div>
      </div>
      {data_table(points, axis_name, base_dir)}
    </section>

    <footer>Static dashboard generated from checkpointed TCAD state.</footer>
  </div>
</body>
</html>
""", plot_path


def generate_experiment_dashboard(source: Path, output_path: Path | None = None) -> DashboardResult:
    try:
        state_path = resolve_state_path(source).resolve()
        state = read_json(state_path)
        kind = detect_report_kind(state)
        dashboard_path = (output_path or state_path.with_name("dashboard.html")).resolve()
        content, plot_path = render_dashboard(
            state=state,
            kind=kind,
            source_path=state_path,
            output_path=dashboard_path,
        )
        write_text(dashboard_path, content)
        return DashboardResult(
            status=DashboardStatus.COMPLETED,
            kind=kind,
            source_state_path=str(state_path),
            dashboard_path=str(dashboard_path),
            best_artifact_plot_path=plot_path,
        )
    except Exception as exc:
        return DashboardResult(
            status=DashboardStatus.FAILED,
            source_state_path=str(source),
            failure_reason=str(exc),
        )
