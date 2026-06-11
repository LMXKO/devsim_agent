from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.deck_writer import plan_deck_mutations
from tcad_agent.mutation_refinement import append_deck_patch_history
from tcad_agent.repair_strategy import (
    curve_guided_mutation_value,
    float_or_none,
    mutation_target,
    next_mutation_value,
    repair_request,
    repair_target_tool,
)


ACTIONABLE_GUIDANCE_ACTIONS = {
    "reduce_field_peak",
    "improve_tradeoff",
    "bracket_breakdown",
    "reduce_leakage",
    "refine_effective_mutation",
    "switch_mutation_target",
}

TARGET_ALIASES = {
    "region_specific_lifetime": "region_lifetime",
}

TARGET_KEYWORDS = {
    "field_plate": "field plate field peak",
    "guard_ring": "guard ring breakdown field peak",
    "drift_doping": "drift doping BV Ron leakage",
    "junction_depth": "junction depth field crowding",
    "implant_dose": "implant dose drift doping",
    "trench_corner_radius": "trench corner radius field peak",
    "oxide_thickness": "oxide thickness CV field coupling",
    "trap_density": "trap density leakage current collapse",
    "region_lifetime": "region specific lifetime leakage",
    "lifetime": "lifetime leakage",
}


class GuidancePatchPlan(BaseModel):
    schema_version: str = "actsoft.tcad.guidance_patch_plan.v1"
    status: str
    source_state_path: str
    output_path: str | None = None
    target_tool: str | None = None
    action: str = "none"
    mutation_target: str | None = None
    recommended_direction: str | None = None
    requires_user_confirmation: bool = False
    reason: str = ""
    curve_guidance: dict[str, Any] = Field(default_factory=dict)
    selected_mutation: dict[str, Any] = Field(default_factory=dict)
    request_patch: dict[str, Any] = Field(default_factory=dict)
    deck_patch: dict[str, Any] = Field(default_factory=dict)
    next_request: dict[str, Any] | None = None
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_target(value: Any) -> str | None:
    target = str(value or "").strip().replace(" ", "_")
    target = TARGET_ALIASES.get(target, target)
    return target or None


def guidance_action(guidance: dict[str, Any]) -> str:
    return str(guidance.get("recommended_action") or "")


def guidance_target(guidance: dict[str, Any]) -> str | None:
    hint = guidance.get("next_patch_hint") if isinstance(guidance.get("next_patch_hint"), dict) else {}
    return normalize_target(hint.get("target") or guidance.get("recommended_target"))


def guidance_is_actionable_patch(guidance: dict[str, Any] | None) -> bool:
    if not isinstance(guidance, dict) or not guidance:
        return False
    action = guidance_action(guidance)
    target = guidance_target(guidance)
    return action in ACTIONABLE_GUIDANCE_ACTIONS and bool(target)


def guidance_requires_confirmation(guidance: dict[str, Any], selected_mutation: dict[str, Any]) -> bool:
    hint = guidance.get("next_patch_hint") if isinstance(guidance.get("next_patch_hint"), dict) else {}
    return bool(hint.get("requires_user_confirmation") or selected_mutation.get("requires_user_confirmation"))


def direction_value(request: dict[str, Any], mutation: dict[str, Any], direction: str | None) -> Any:
    path = str(mutation.get("request_path") or "")
    current = float_or_none(request.get(path))
    values = mutation.get("values") if isinstance(mutation.get("values"), list) else []
    numeric_values = [(value, float_or_none(value)) for value in values]
    numeric_values = [(raw, numeric) for raw, numeric in numeric_values if numeric is not None]
    if current is None or not numeric_values:
        return next_mutation_value(request, mutation)
    if direction in {"increase", "adjust", "smaller_step_same_direction"}:
        higher = [item for item in numeric_values if item[1] > current]
        if higher:
            return sorted(higher, key=lambda item: item[1])[0][0]
    if direction in {"decrease", "decrease_leakage"}:
        lower = [item for item in numeric_values if item[1] < current]
        if lower:
            return sorted(lower, key=lambda item: item[1], reverse=True)[0][0]
    return next_mutation_value(request, mutation)


def available_mutations(goal_text: str, tool_name: str | None, request: dict[str, Any], target: str | None) -> list[dict[str, Any]]:
    existing = request.get("tcad_deck_mutations")
    mutations = [item for item in existing if isinstance(item, dict)] if isinstance(existing, list) else []
    if mutations:
        return mutations
    keywords = TARGET_KEYWORDS.get(str(target or ""), str(target or ""))
    planned = plan_deck_mutations(f"{goal_text} {keywords}", tool_name, request)
    return [item.model_dump(mode="json") for item in planned]


def select_mutation(mutations: list[dict[str, Any]], target: str | None) -> dict[str, Any] | None:
    if not mutations:
        return None
    if target:
        for mutation in mutations:
            if normalize_target(mutation_target(mutation)) == target:
                return mutation
    return mutations[0]


def mutation_value_from_guidance(request: dict[str, Any], mutation: dict[str, Any], guidance: dict[str, Any]) -> Any:
    effect = guidance.get("mutation_effect") if isinstance(guidance.get("mutation_effect"), dict) else None
    if effect:
        value = curve_guided_mutation_value(request, mutation, effect)
        if value is not None:
            return value
    direction = str(guidance.get("recommended_direction") or "")
    hint = guidance.get("next_patch_hint") if isinstance(guidance.get("next_patch_hint"), dict) else {}
    direction = str(hint.get("direction") or direction or "")
    return direction_value(request, mutation, direction)


def build_guidance_patch_plan(
    source_state_path: Path,
    *,
    curve_guidance: dict[str, Any],
    goal_text: str,
    output_path: Path | None = None,
) -> GuidancePatchPlan:
    actual_source = source_state_path.resolve()
    try:
        state = read_json(actual_source)
    except Exception as exc:
        return GuidancePatchPlan(status="failed", source_state_path=str(actual_source), failure_reason=str(exc))
    if not guidance_is_actionable_patch(curve_guidance):
        return GuidancePatchPlan(
            status="no_action",
            source_state_path=str(actual_source),
            curve_guidance=curve_guidance,
            reason="Curve guidance did not recommend an executable physical/model patch.",
        )
    tool_name = repair_target_tool(state)
    request = repair_request(state)
    target = guidance_target(curve_guidance)
    mutations = available_mutations(goal_text, tool_name, request, target)
    selected = select_mutation(mutations, target)
    if not selected:
        return GuidancePatchPlan(
            status="failed",
            source_state_path=str(actual_source),
            target_tool=tool_name,
            curve_guidance=curve_guidance,
            mutation_target=target,
            failure_reason="no executable deck mutation matched the curve guidance target",
        )
    path = str(selected.get("request_path") or "")
    value = mutation_value_from_guidance(request, selected, curve_guidance)
    if not path or value is None:
        return GuidancePatchPlan(
            status="failed",
            source_state_path=str(actual_source),
            target_tool=tool_name,
            curve_guidance=curve_guidance,
            selected_mutation=selected,
            mutation_target=target,
            failure_reason="selected guidance mutation did not produce a runnable next value",
        )
    request_patch: dict[str, Any] = {
        path: value,
        "active_deck_mutation": selected,
        "deck_repair_hint": f"curve guidance patch for {target or selected.get('name') or path}",
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
        "curve_guidance_action": curve_guidance.get("recommended_action"),
        "curve_guidance_reason": curve_guidance.get("reason"),
    }
    next_request = {**request, **request_patch}
    next_request["deck_patch_history"] = append_deck_patch_history(request, deck_patch, actual_source)
    next_request.setdefault("repair_baseline_state_path", str(actual_source))
    next_request["guidance_source_state_path"] = str(actual_source)
    next_request["guidance_patch_id"] = f"guidance_patch_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    next_request["run_id"] = f"{str(state.get('run_id') or actual_source.parent.name)}_guidance_patch"
    plan = GuidancePatchPlan(
        status="completed",
        source_state_path=str(actual_source),
        target_tool=tool_name,
        action=str(curve_guidance.get("recommended_action") or ""),
        mutation_target=target,
        recommended_direction=str(curve_guidance.get("recommended_direction") or ""),
        requires_user_confirmation=guidance_requires_confirmation(curve_guidance, selected),
        reason=str(curve_guidance.get("reason") or "Curve guidance selected the next deck/request patch."),
        curve_guidance=curve_guidance,
        selected_mutation=selected,
        request_patch=request_patch,
        deck_patch=deck_patch,
        next_request=next_request,
    )
    if output_path is not None:
        plan.output_path = str(output_path.resolve())
        write_json(output_path, plan.model_dump(mode="json"))
    return plan
