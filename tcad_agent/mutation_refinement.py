from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.repair_strategy import curve_guided_mutation_value, mutation_target, ordered_mutations_for_analysis


class MutationRefinementPlan(BaseModel):
    schema_version: str = "actsoft.tcad.mutation_refinement.v1"
    status: str
    source_state_path: str
    output_path: str | None = None
    target_tool: str | None = None
    action: str = "none"
    mutation_target: str | None = None
    decision: str | None = None
    worth_continuing: bool = False
    requires_user_confirmation: bool = False
    reason: str = ""
    mutation_effect_analysis: dict[str, Any] = Field(default_factory=dict)
    selected_mutation: dict[str, Any] = Field(default_factory=dict)
    request_patch: dict[str, Any] = Field(default_factory=dict)
    deck_patch: dict[str, Any] = Field(default_factory=dict)
    next_request: dict[str, Any] | None = None
    tradeoff_violations: list[dict[str, Any]] = Field(default_factory=list)
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def state_request(state: dict[str, Any]) -> dict[str, Any]:
    request = state.get("request")
    return dict(request) if isinstance(request, dict) else {}


def dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def state_mutations(state: dict[str, Any], request: dict[str, Any]) -> list[dict[str, Any]]:
    summary = dict_or_empty(state.get("final_summary"))
    repair_context = dict_or_empty(state.get("repair_context"))
    candidates: list[Any] = [
        request.get("tcad_deck_mutations"),
        state.get("tcad_deck_mutations"),
        summary.get("tcad_deck_mutations"),
        repair_context.get("deck_mutations"),
    ]
    for candidate in candidates:
        if isinstance(candidate, list):
            mutations = [item for item in candidate if isinstance(item, dict)]
            if mutations:
                return mutations
    return []


def append_deck_patch_history(request: dict[str, Any], deck_patch: dict[str, Any], source_state_path: Path) -> list[dict[str, Any]]:
    history = request.get("deck_patch_history")
    if not isinstance(history, list):
        history = []
    return [
        *history,
        {
            **deck_patch,
            "action_name": "agent_mutation_refinement",
            "source_state_path": str(source_state_path),
            "created_at": utc_timestamp(),
        },
    ]


def select_mutation(mutations: list[dict[str, Any]], analysis: dict[str, Any]) -> dict[str, Any] | None:
    ordered = ordered_mutations_for_analysis(mutations, analysis)
    recommended = str(analysis.get("recommended_next_target") or "")
    if recommended:
        for mutation in ordered:
            if mutation_target(mutation) == recommended:
                return mutation
    return ordered[0] if ordered else None


def sensitive_mutation(mutation: dict[str, Any], target: str | None) -> bool:
    if mutation.get("requires_user_confirmation"):
        return True
    return str(target or "") in {
        "field_plate",
        "guard_ring",
        "junction_depth",
        "implant_dose",
        "trench_corner_radius",
        "trap_density",
        "region_lifetime",
    }


def build_mutation_refinement_plan(source_state_path: Path, output_path: Path | None = None) -> MutationRefinementPlan:
    actual_source = source_state_path.resolve()
    try:
        state = read_json(actual_source)
    except Exception as exc:
        return MutationRefinementPlan(status="failed", source_state_path=str(actual_source), failure_reason=str(exc))

    analysis = state.get("mutation_effect_analysis")
    if not isinstance(analysis, dict) or not analysis:
        return MutationRefinementPlan(
            status="failed",
            source_state_path=str(actual_source),
            failure_reason="source state does not contain mutation_effect_analysis",
        )

    request = state_request(state)
    mutations = state_mutations(state, request)
    selected = select_mutation(mutations, analysis)
    tradeoffs = [item for item in analysis.get("tradeoff_violations") or [] if isinstance(item, dict)]
    if tradeoffs:
        plan = MutationRefinementPlan(
            status="blocked_for_pareto_review",
            source_state_path=str(actual_source),
            target_tool=str(state.get("tool_name") or "") or None,
            action="pareto_review",
            mutation_target=analysis.get("recommended_next_target") or analysis.get("mutation_target"),
            decision=str(analysis.get("decision") or ""),
            worth_continuing=bool(analysis.get("worth_continuing")),
            requires_user_confirmation=True,
            reason="Mutation changed the curve but introduced tradeoff violations; require Pareto/constraint review before continuing.",
            mutation_effect_analysis=analysis,
            tradeoff_violations=tradeoffs,
        )
    elif not selected:
        plan = MutationRefinementPlan(
            status="failed",
            source_state_path=str(actual_source),
            decision=str(analysis.get("decision") or ""),
            worth_continuing=bool(analysis.get("worth_continuing")),
            mutation_effect_analysis=analysis,
            failure_reason="no executable tcad_deck_mutations are available for refinement",
        )
    else:
        target = mutation_target(selected)
        path = str(selected.get("request_path") or "")
        value = curve_guided_mutation_value(request, selected, analysis)
        if not path or value is None:
            plan = MutationRefinementPlan(
                status="failed",
                source_state_path=str(actual_source),
                target_tool=str(state.get("tool_name") or "") or None,
                mutation_target=target,
                decision=str(analysis.get("decision") or ""),
                worth_continuing=bool(analysis.get("worth_continuing")),
                mutation_effect_analysis=analysis,
                selected_mutation=selected,
                failure_reason="selected mutation did not produce a runnable next value",
            )
        else:
            request_patch: dict[str, Any] = {
                path: value,
                "active_deck_mutation": selected,
                "deck_repair_hint": f"agent refinement for {target or selected.get('name') or path}",
            }
            if path == "electron_lifetime_s" and "hole_lifetime_s" in request:
                request_patch["hole_lifetime_s"] = value
            deck_patch = {
                "operation": selected.get("operation") or "set",
                "request_path": path,
                "deck_path": selected.get("deck_path"),
                "value": value,
                "baseline_value": request.get(path),
                "target": target,
                "source_mutation": selected.get("name"),
                "curve_guided_decision": analysis.get("decision"),
                "curve_guided_rationale": analysis.get("rationale"),
            }
            next_request = {**request, **request_patch}
            next_request["deck_patch_history"] = append_deck_patch_history(request, deck_patch, actual_source)
            repair_context = dict_or_empty(state.get("repair_context"))
            next_request.setdefault("repair_baseline_state_path", repair_context.get("baseline_state_path") or str(actual_source))
            next_request["repair_source_state_path"] = str(actual_source)
            next_request["mutation_refinement_id"] = f"mutation_refinement_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            next_request["run_id"] = f"{str(state.get('run_id') or actual_source.parent.name)}_agent_refine"
            plan = MutationRefinementPlan(
                status="completed",
                source_state_path=str(actual_source),
                target_tool=str(state.get("tool_name") or "") or None,
                action="continue_same_target" if analysis.get("worth_continuing") else "switch_or_probe_target",
                mutation_target=target,
                decision=str(analysis.get("decision") or ""),
                worth_continuing=bool(analysis.get("worth_continuing")),
                requires_user_confirmation=sensitive_mutation(selected, target),
                reason=(
                    "Primary mutation improved without blocking tradeoffs; continue with a half-step refinement."
                    if analysis.get("worth_continuing")
                    else "Previous mutation was not convincing; probe the recommended alternate target."
                ),
                mutation_effect_analysis=analysis,
                selected_mutation=selected,
                request_patch=request_patch,
                deck_patch=deck_patch,
                next_request=next_request,
            )

    if output_path is not None:
        plan.output_path = str(output_path.resolve())
        write_json(output_path, plan.model_dump(mode="json"))
    return plan
