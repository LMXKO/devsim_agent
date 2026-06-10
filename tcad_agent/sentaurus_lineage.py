from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.curve_diagnostics import finite_float
from tcad_agent.engineering_objectives import ObjectiveCandidate, assign_pareto_front, evaluate_candidate
from tcad_agent.reporting import final_artifacts, final_metrics
from tcad_agent.sentaurus_mutation_effect import (
    BV_ALIASES,
    FIELD_ALIASES,
    LOWER_BETTER_ALIASES,
    RON_ALIASES,
    objective_for_metric,
)


WATCHED_METRICS = LOWER_BETTER_ALIASES + BV_ALIASES + FIELD_ALIASES + RON_ALIASES + ["curve_points"]


class SentaurusLineageEntry(BaseModel):
    lineage_id: str
    state_path: str
    run_id: str | None = None
    status: str | None = None
    quality_status: str | None = None
    baseline_state_path: str | None = None
    candidate_id: str | None = None
    candidate_title: str | None = None
    decision: str | None = None
    worth_continuing: bool | None = None
    rationale: str | None = None
    primary_metric: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    improved_metrics: list[str] = Field(default_factory=list)
    regressed_metrics: list[str] = Field(default_factory=list)
    tradeoff_violations: list[dict[str, Any]] = Field(default_factory=list)
    patches: list[dict[str, Any]] = Field(default_factory=list)
    overlay_svg_path: str | None = None
    pareto_front: bool = False
    score: float | None = None


class SentaurusLineageArchive(BaseModel):
    tool_name: str = "sentaurus_lineage_archive"
    schema_version: str = "actsoft.tcad.sentaurus_lineage.v1"
    status: str
    source_state_path: str
    entries: list[SentaurusLineageEntry] = Field(default_factory=list)
    objectives: list[dict[str, Any]] = Field(default_factory=list)
    pareto_front: list[str] = Field(default_factory=list)
    best_entry: SentaurusLineageEntry | None = None
    output_path: str | None = None
    failure_reason: str | None = None


class SentaurusLineageArchiveRequest(BaseModel):
    source_state_path: Path
    output_path: Path | None = None
    max_depth: int = Field(default=24, ge=1, le=200)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def chain_state_paths(source: Path, *, max_depth: int) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    current = source.expanduser().resolve()
    for _ in range(max_depth):
        if current in seen or not current.exists():
            break
        seen.add(current)
        paths.append(current)
        try:
            state = read_json(current)
        except Exception:
            break
        repair_context = state.get("repair_context") if isinstance(state.get("repair_context"), dict) else {}
        baseline = repair_context.get("baseline_state_path")
        if not baseline:
            break
        next_path = Path(str(baseline)).expanduser()
        if not next_path.is_absolute():
            next_path = current.parent / next_path
        current = next_path.resolve()
    return list(reversed(paths))


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in WATCHED_METRICS:
        if key in metrics:
            output[key] = metrics[key]
    for key in ["solver_backend", "tcad_solver_invoked", "curve_x_key", "curve_y_key", "curve_field_key"]:
        if key in metrics:
            output[key] = metrics[key]
    return output


def compact_patch(patch: dict[str, Any]) -> dict[str, Any]:
    keys = ["file", "operation", "variable", "section_path", "selector", "parameter", "model", "value", "reason"]
    return {key: patch[key] for key in keys if key in patch}


def entry_from_state(path: Path, index: int) -> SentaurusLineageEntry:
    state = read_json(path)
    effect = state.get("sentaurus_mutation_effect_analysis") if isinstance(state.get("sentaurus_mutation_effect_analysis"), dict) else {}
    candidate = effect.get("candidate") if isinstance(effect.get("candidate"), dict) else {}
    repair_context = state.get("repair_context") if isinstance(state.get("repair_context"), dict) else {}
    quality = state.get("quality_report") if isinstance(state.get("quality_report"), dict) else {}
    artifacts = final_artifacts(state)
    metrics = compact_metrics(final_metrics(state))
    return SentaurusLineageEntry(
        lineage_id=f"sentaurus_{index:03d}",
        state_path=str(path),
        run_id=str(state.get("run_id") or path.parent.name),
        status=str(state.get("status")) if state.get("status") else None,
        quality_status=str(quality.get("status")) if quality.get("status") else None,
        baseline_state_path=repair_context.get("baseline_state_path"),
        candidate_id=str(effect.get("candidate_id") or candidate.get("candidate_id") or "") or None,
        candidate_title=str(candidate.get("title") or "") or None,
        decision=str(effect.get("decision") or "") or None,
        worth_continuing=effect.get("worth_continuing") if isinstance(effect.get("worth_continuing"), bool) else None,
        rationale=str(effect.get("rationale") or "") or None,
        primary_metric=str(effect.get("primary_metric") or "") or None,
        metrics=metrics,
        improved_metrics=[str(item) for item in effect.get("improved_metrics") or []],
        regressed_metrics=[str(item) for item in effect.get("regressed_metrics") or []],
        tradeoff_violations=[item for item in effect.get("tradeoff_violations") or [] if isinstance(item, dict)],
        patches=[compact_patch(patch) for patch in candidate.get("patches") or [] if isinstance(patch, dict)],
        overlay_svg_path=effect.get("overlay_svg_path") or artifacts.get("sentaurus_baseline_mutation_overlay"),
    )


def objective_metrics(entries: list[SentaurusLineageEntry]) -> list[str]:
    metrics: list[str] = []
    for metric in WATCHED_METRICS:
        comparable = sum(1 for entry in entries if finite_float(entry.metrics.get(metric)) is not None)
        if comparable >= 2:
            metrics.append(metric)
    return metrics


def assign_archive_pareto(entries: list[SentaurusLineageEntry]) -> tuple[list[dict[str, Any]], list[str], SentaurusLineageEntry | None]:
    metrics = objective_metrics(entries)
    if not metrics:
        if entries:
            entries[-1].pareto_front = True
            return [], [entries[-1].lineage_id], entries[-1]
        return [], [], None
    objectives = [objective_for_metric(metric) for metric in metrics]
    candidates = [
        evaluate_candidate(
            ObjectiveCandidate(candidate_id=entry.lineage_id, source_state_path=entry.state_path, metrics=entry.metrics),
            objectives,
            [],
        )
        for entry in entries
    ]
    candidates = assign_pareto_front(candidates, objectives)
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    for entry in entries:
        candidate = by_id.get(entry.lineage_id)
        if not candidate:
            continue
        entry.pareto_front = candidate.pareto_front
        entry.score = candidate.score
    scored = [entry for entry in entries if entry.score is not None]
    best = sorted(scored, key=lambda item: float(item.score))[0] if scored else entries[-1] if entries else None
    return [objective.model_dump(mode="json") for objective in objectives], [entry.lineage_id for entry in entries if entry.pareto_front], best


def build_sentaurus_lineage_archive(request: SentaurusLineageArchiveRequest) -> SentaurusLineageArchive:
    source = request.source_state_path.expanduser().resolve()
    try:
        paths = chain_state_paths(source, max_depth=request.max_depth)
        entries = [entry_from_state(path, index) for index, path in enumerate(paths, start=1)]
        objectives, pareto_front, best = assign_archive_pareto(entries)
        archive = SentaurusLineageArchive(
            status="completed",
            source_state_path=str(source),
            entries=entries,
            objectives=objectives,
            pareto_front=pareto_front,
            best_entry=best,
        )
    except Exception as exc:
        archive = SentaurusLineageArchive(status="failed", source_state_path=str(source), failure_reason=str(exc))
    if request.output_path is not None:
        output_path = request.output_path.expanduser().resolve()
        archive.output_path = str(output_path)
        write_json(output_path, archive.model_dump(mode="json"))
    return archive
