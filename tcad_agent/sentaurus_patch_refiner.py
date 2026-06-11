from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from tcad_agent.llm import LLMClient, LLMConfig
from tcad_agent.sentaurus_deck import apply_sentaurus_semantic_patch_text
from tcad_agent.sentaurus_patch_planner import (
    DeckContext,
    SentaurusPatchPlannerRequest,
    goal_tags,
    numeric_value,
    plan_sentaurus_patches,
    project_path_from_state,
    resolve_decks,
    variable_classes,
)
from tcad_agent.task_planner import parse_json_object


class ChatClient(Protocol):
    config: LLMConfig

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        ...


class SentaurusPatchRefinementCandidate(BaseModel):
    candidate_id: str
    title: str
    action: str
    source_candidate_id: str | None = None
    score: float
    hypothesis: str
    risk_level: str = "medium"
    requires_user_confirmation: bool = False
    target_file: str | None = None
    patches: list[dict[str, Any]] = Field(default_factory=list)
    validation_records: list[dict[str, Any]] = Field(default_factory=list)
    verified_patch_count: int = 0
    expected_observation: str
    stop_condition: str
    fallback_alternatives: list[str] = Field(default_factory=list)
    rationale: str
    evidence_used: list[dict[str, Any]] = Field(default_factory=list)
    next_request_patch: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.5


class SentaurusPatchRefinementPlan(BaseModel):
    tool_name: str = "sentaurus_patch_refiner"
    schema_version: str = "actsoft.tcad.sentaurus_patch_refiner.v1"
    status: str
    source_state_path: str
    goal_text: str = ""
    analysis: dict[str, Any] = Field(default_factory=dict)
    project_path: str | None = None
    deck_files_inspected: list[str] = Field(default_factory=list)
    candidates: list[SentaurusPatchRefinementCandidate] = Field(default_factory=list)
    selected_candidate: SentaurusPatchRefinementCandidate | None = None
    agent_policy: dict[str, Any] = Field(default_factory=dict)
    final_summary: dict[str, Any] = Field(default_factory=dict)
    output_path: str | None = None
    failure_reason: str | None = None


class SentaurusPatchRefinerRequest(BaseModel):
    source_state_path: Path
    goal_text: str = ""
    output_path: Path | None = None
    max_candidates: int = Field(default=4, ge=1, le=16)
    allow_high_risk: bool = False
    use_llm: bool = False
    allow_llm_fallback: bool = True


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def state_goal_text(state: dict[str, Any]) -> str:
    request = state.get("request") if isinstance(state.get("request"), dict) else {}
    for key in ["goal_text", "goal", "task"]:
        value = request.get(key)
        if value:
            return str(value)
    return ""


def patch_signature(patch: dict[str, Any]) -> tuple[Any, ...]:
    operation = str(patch.get("operation") or "")
    if operation == "sentaurus_set_variable":
        return (operation, patch.get("file"), patch.get("variable"))
    if operation in {"sentaurus_update_assignment", "sentaurus_upsert_assignment"}:
        return (
            operation,
            patch.get("file"),
            tuple(patch.get("section_path") or []),
            json.dumps(patch.get("selector") or {}, sort_keys=True, ensure_ascii=True),
            patch.get("parameter"),
        )
    if operation == "sentaurus_add_model":
        return (operation, patch.get("file"), tuple(patch.get("section_path") or []), patch.get("model"))
    return tuple(sorted(patch.items()))


def patch_target_token(patch: dict[str, Any]) -> str:
    if patch.get("variable"):
        return str(patch["variable"])
    if patch.get("parameter"):
        return str(patch["parameter"])
    if patch.get("model"):
        return str(patch["model"])
    return str(patch.get("operation") or "patch")


def candidate_target_classes(candidate: dict[str, Any]) -> set[str]:
    classes: set[str] = set()
    for patch in candidate.get("patches") or []:
        if not isinstance(patch, dict):
            continue
        variable = patch.get("variable")
        if variable:
            classes.update(variable_classes(str(variable)))
    text = json.dumps(candidate, ensure_ascii=False, sort_keys=True).lower()
    for token in [
        "lifetime",
        "region_specific_lifetime",
        "trap_density",
        "drift_doping",
        "field_plate",
        "guard_ring",
        "oxide_thickness",
        "implant_dose",
        "junction_depth",
        "trench_corner_radius",
    ]:
        if token in text:
            classes.add(token)
    return classes


def matching_validation_record(patch: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any] | None:
    signature = patch_signature(patch)
    for record in records:
        if not isinstance(record, dict) or not record.get("verified"):
            continue
        pseudo_patch = {
            "operation": record.get("operation"),
            "file": record.get("file") or patch.get("file"),
            "variable": record.get("variable"),
            "section_path": record.get("section_path") or record.get("block_path"),
            "selector": record.get("selector"),
            "parameter": record.get("parameter"),
            "model": record.get("model"),
        }
        if patch_signature({**patch, **{key: value for key, value in pseudo_patch.items() if value is not None}}) == signature:
            return record
    if len(records) == 1 and isinstance(records[0], dict) and records[0].get("verified"):
        return records[0]
    return None


def refined_numeric_text(old_value: Any, new_value: Any) -> str | None:
    old = numeric_value(old_value)
    new = numeric_value(new_value)
    if old is None or new is None:
        return None
    step = new - old
    if step == 0:
        return None
    refined = new + 0.5 * step
    if old > 0 and new > 0 and refined <= 0:
        refined = new * 0.5
    return f"{refined:.6g}"


def validate_refinement_candidates(candidates: list[SentaurusPatchRefinementCandidate], contexts: list[DeckContext]) -> None:
    text_by_file = {ctx.rel_file: ctx.text for ctx in contexts}
    for candidate in candidates:
        working_text = dict(text_by_file)
        records: list[dict[str, Any]] = []
        for patch in candidate.patches:
            file_name = str(patch.get("file") or "")
            text = working_text.get(file_name)
            if text is None:
                records.append(
                    {
                        "file": file_name,
                        "operation": patch.get("operation"),
                        "applied": False,
                        "verified": False,
                        "error": "deck text not available",
                    }
                )
                continue
            try:
                after, record, _ = apply_sentaurus_semantic_patch_text(text, patch, source_path=file_name)
            except Exception as exc:
                records.append(
                    {
                        "file": file_name,
                        "operation": patch.get("operation"),
                        "applied": False,
                        "verified": False,
                        "error": str(exc),
                    }
                )
                continue
            record["file"] = file_name
            records.append(record)
            if record.get("applied"):
                working_text[file_name] = after
        candidate.validation_records = records
        candidate.verified_patch_count = sum(1 for record in records if record.get("verified"))
        if candidate.verified_patch_count < len(candidate.patches):
            candidate.requires_user_confirmation = True
            candidate.risk_level = "high"
            candidate.confidence = min(candidate.confidence, 0.35)


def continue_same_direction_candidates(
    *,
    analysis: dict[str, Any],
    max_candidates: int,
) -> list[SentaurusPatchRefinementCandidate]:
    source_candidate = analysis.get("candidate") if isinstance(analysis.get("candidate"), dict) else {}
    source_candidate_id = str(analysis.get("candidate_id") or source_candidate.get("candidate_id") or "") or None
    source_records = source_candidate.get("validation_records") if isinstance(source_candidate.get("validation_records"), list) else []
    candidates: list[SentaurusPatchRefinementCandidate] = []
    seen: set[tuple[Any, ...]] = set()
    for patch in source_candidate.get("patches") or []:
        if not isinstance(patch, dict) or patch.get("operation") not in {"sentaurus_set_variable", "sentaurus_update_assignment", "sentaurus_upsert_assignment"}:
            continue
        record = matching_validation_record(patch, source_records)
        old_value = (record or {}).get("old_value")
        previous_new_value = (record or {}).get("value", patch.get("value"))
        next_value = refined_numeric_text(old_value, previous_new_value)
        if next_value is None:
            continue
        next_patch = dict(patch)
        next_patch["value"] = next_value
        next_patch["reason"] = (
            f"Curve comparison marked the previous Sentaurus patch as worth continuing; "
            f"take a half step beyond {previous_new_value} from baseline {old_value}."
        )
        signature = patch_signature(next_patch)
        if signature in seen:
            continue
        seen.add(signature)
        target = patch_target_token(patch)
        risk = str(source_candidate.get("risk_level") or "medium")
        candidates.append(
            SentaurusPatchRefinementCandidate(
                candidate_id=f"{source_candidate_id or 'sentaurus_patch'}:refine:{target}",
                title=f"Refine {target} in the same Sentaurus direction",
                action="continue_same_direction",
                source_candidate_id=source_candidate_id,
                score=0.82,
                hypothesis="The last Sentaurus mutation improved the primary curve metric without blocking tradeoffs, so a smaller continuation step should test monotonicity.",
                risk_level=risk,
                requires_user_confirmation=bool(source_candidate.get("requires_user_confirmation") or risk == "high"),
                target_file=str(next_patch.get("file") or source_candidate.get("target_file") or ""),
                patches=[next_patch],
                expected_observation=str(
                    source_candidate.get("expected_observation")
                    or "The primary metric should keep moving in the favorable direction with no new BV/Ron/field/leakage tradeoff."
                ),
                stop_condition=str(
                    source_candidate.get("stop_condition")
                    or "Stop if the next curve flattens, reverses, or introduces a Pareto/constraint regression."
                ),
                fallback_alternatives=[
                    "switch target if the primary metric stops improving",
                    "pause for Pareto review if BV/Ron/field/leakage tradeoffs appear",
                    "collect denser CSV extraction around the knee or breakdown bracket if evidence is sparse",
                ],
                rationale=str(analysis.get("rationale") or "Previous patch direction was useful; refine with a smaller step."),
                evidence_used=[
                    {"kind": "sentaurus_mutation_effect_analysis", "decision": analysis.get("decision"), "primary_metric": analysis.get("primary_metric")},
                    {"kind": "source_patch_validation", "record": record or {}},
                ],
                next_request_patch={
                    "sentaurus_patch_candidate_id": f"{source_candidate_id or 'sentaurus_patch'}:refine:{target}",
                    "patches": [next_patch],
                },
                confidence=0.74,
            )
        )
        if len(candidates) >= max_candidates:
            break
    return candidates


def convert_planner_candidate(
    candidate: Any,
    *,
    action: str,
    source_candidate_id: str | None,
    analysis: dict[str, Any],
) -> SentaurusPatchRefinementCandidate:
    data = candidate.model_dump(mode="json") if hasattr(candidate, "model_dump") else dict(candidate)
    return SentaurusPatchRefinementCandidate(
        candidate_id=f"{data.get('candidate_id')}:switch_from:{source_candidate_id or 'unknown'}",
        title=str(data.get("title") or "Switch Sentaurus patch target"),
        action=action,
        source_candidate_id=source_candidate_id,
        score=float(data.get("score") or 0.5) * 0.95,
        hypothesis=str(data.get("hypothesis") or "The prior Sentaurus patch did not improve the target curve metric; test a different verified deck target."),
        risk_level=str(data.get("risk_level") or "medium"),
        requires_user_confirmation=bool(data.get("requires_user_confirmation") or data.get("risk_level") == "high"),
        target_file=data.get("target_file"),
        patches=[patch for patch in data.get("patches") or [] if isinstance(patch, dict)],
        validation_records=[record for record in data.get("validation_records") or [] if isinstance(record, dict)],
        verified_patch_count=int(data.get("verified_patch_count") or 0),
        expected_observation=str(data.get("expected_observation") or "The next curve should move the primary metric or expose a clearer tradeoff boundary."),
        stop_condition=str(data.get("stop_condition") or "Stop if the switched target also fails to improve the primary metric."),
        fallback_alternatives=[str(item) for item in data.get("fallback_alternatives") or []],
        rationale=f"Switching target because effect analyzer decided `{analysis.get('decision')}`: {analysis.get('rationale') or ''}".strip(),
        evidence_used=[
            {"kind": "sentaurus_mutation_effect_analysis", "decision": analysis.get("decision"), "recommended_next_target": analysis.get("recommended_next_target")},
            *[item for item in data.get("evidence_used") or [] if isinstance(item, dict)],
        ],
        next_request_patch={
            "sentaurus_patch_candidate_id": f"{data.get('candidate_id')}:switch_from:{source_candidate_id or 'unknown'}",
            "patches": [patch for patch in data.get("patches") or [] if isinstance(patch, dict)],
        },
        confidence=float(data.get("confidence") or data.get("score") or 0.5) * 0.9,
    )


def switch_target_candidates(
    *,
    source_state_path: Path,
    state: dict[str, Any],
    analysis: dict[str, Any],
    goal_text_value: str,
    max_candidates: int,
    allow_high_risk: bool,
) -> list[SentaurusPatchRefinementCandidate]:
    source_candidate = analysis.get("candidate") if isinstance(analysis.get("candidate"), dict) else {}
    source_candidate_id = str(analysis.get("candidate_id") or source_candidate.get("candidate_id") or "") or None
    source_signatures = {
        patch_signature(patch)
        for patch in source_candidate.get("patches") or []
        if isinstance(patch, dict)
    }
    source_classes = candidate_target_classes(source_candidate)
    recommended = str(analysis.get("recommended_next_target") or "").strip() or None
    if recommended in source_classes:
        recommended = None
    target_hint = f" Prefer target class {recommended}." if recommended else " Prefer a different verified variable class than the previous patch."
    project = project_path_from_state(state)
    planner = plan_sentaurus_patches(
        SentaurusPatchPlannerRequest(
            goal_text=f"{goal_text_value}{target_hint}",
            source_state_path=source_state_path,
            project_path=project,
            output_path=None,
            max_candidates=max(max_candidates * 3, 8),
            allow_high_risk=allow_high_risk,
        )
    )
    converted: list[SentaurusPatchRefinementCandidate] = []
    for candidate in planner.candidates:
        data = candidate.model_dump(mode="json")
        patches = [patch for patch in data.get("patches") or [] if isinstance(patch, dict)]
        if not patches or any(patch_signature(patch) in source_signatures for patch in patches):
            continue
        if recommended and recommended not in candidate_target_classes(data):
            continue
        converted.append(
            convert_planner_candidate(
                data,
                action="switch_target" if analysis.get("decision") == "switch_target" else "replace_rejected_direction",
                source_candidate_id=source_candidate_id,
                analysis=analysis,
            )
        )
    if not converted and recommended:
        return switch_target_candidates(
            source_state_path=source_state_path,
            state=state,
            analysis={**analysis, "recommended_next_target": None},
            goal_text_value=goal_text_value,
            max_candidates=max_candidates,
            allow_high_risk=allow_high_risk,
        )
    return converted[:max_candidates]


def select_candidate(
    candidates: list[SentaurusPatchRefinementCandidate],
    *,
    allow_high_risk: bool,
) -> SentaurusPatchRefinementCandidate | None:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.patches
        and candidate.verified_patch_count == len(candidate.patches)
        and (allow_high_risk or (candidate.risk_level != "high" and not candidate.requires_user_confirmation))
    ]
    if not eligible:
        return None
    return sorted(eligible, key=lambda item: item.score, reverse=True)[0]


def eligible_candidates(
    candidates: list[SentaurusPatchRefinementCandidate],
    *,
    allow_high_risk: bool,
) -> list[SentaurusPatchRefinementCandidate]:
    return [
        candidate
        for candidate in candidates
        if candidate.patches
        and candidate.verified_patch_count == len(candidate.patches)
        and (allow_high_risk or (candidate.risk_level != "high" and not candidate.requires_user_confirmation))
    ]


def choose_candidate_with_agent(
    *,
    candidates: list[SentaurusPatchRefinementCandidate],
    analysis: dict[str, Any],
    goal_text_value: str,
    allow_high_risk: bool,
    llm_client: ChatClient | None,
) -> tuple[SentaurusPatchRefinementCandidate | None, dict[str, Any]]:
    eligible = eligible_candidates(candidates, allow_high_risk=allow_high_risk)
    fallback = sorted(eligible, key=lambda item: item.score, reverse=True)[0] if eligible else None
    policy: dict[str, Any] = {
        "policy": "sentaurus_refinement_candidate_selection",
        "status": "not_used",
        "fallback_candidate_id": fallback.candidate_id if fallback else None,
        "selected_candidate_id": fallback.candidate_id if fallback else None,
        "reason": "deterministic_score_order",
    }
    if not eligible:
        policy["status"] = "no_eligible_candidates"
        return None, policy
    if llm_client is None:
        return fallback, policy

    payload = {
        "goal_text": goal_text_value,
        "effect_analysis": {
            "decision": analysis.get("decision"),
            "primary_metric": analysis.get("primary_metric"),
            "improved_metrics": analysis.get("improved_metrics"),
            "regressed_metrics": analysis.get("regressed_metrics"),
            "tradeoff_violations": analysis.get("tradeoff_violations"),
            "recommended_next_target": analysis.get("recommended_next_target"),
            "rationale": analysis.get("rationale"),
        },
        "eligible_candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "action": candidate.action,
                "score": candidate.score,
                "risk_level": candidate.risk_level,
                "hypothesis": candidate.hypothesis,
                "expected_observation": candidate.expected_observation,
                "stop_condition": candidate.stop_condition,
                "patches": candidate.patches,
                "validation_records": candidate.validation_records,
            }
            for candidate in eligible
        ],
    }
    system = (
        "You are a TCAD engineering agent selecting the next Sentaurus deck patch from verified candidates only. "
        "Do not invent patches. Choose exactly one candidate_id from eligible_candidates. "
        "Prefer the candidate that best tests the user's goal while respecting BV/Ron/field/leakage tradeoffs."
    )
    user = (
        "Return JSON only with keys: selected_candidate_id, rationale, expected_observation, stop_condition, rejected_candidate_ids.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    raw = llm_client.chat(system, user, temperature=0.1)
    parsed = parse_json_object(raw) or {}
    selected_id = str(parsed.get("selected_candidate_id") or "")
    by_id = {candidate.candidate_id: candidate for candidate in eligible}
    selected = by_id.get(selected_id)
    policy.update(
        {
            "status": "completed" if selected else "invalid_selection",
            "model": getattr(llm_client.config, "model", None),
            "raw_response": raw,
            "parsed_response": parsed,
            "selected_candidate_id": selected.candidate_id if selected else fallback.candidate_id if fallback else None,
            "reason": parsed.get("rationale") or "model selection" if selected else "model selected an ineligible or unknown candidate; deterministic fallback used",
            "rejected_candidate_ids": parsed.get("rejected_candidate_ids") if isinstance(parsed.get("rejected_candidate_ids"), list) else [],
        }
    )
    return selected or fallback, policy


def llm_client_for_request(request: SentaurusPatchRefinerRequest, llm_client: ChatClient | None) -> tuple[ChatClient | None, dict[str, Any]]:
    if not request.use_llm:
        return None, {"llm_enabled": False}
    if llm_client is not None:
        return llm_client, {"llm_enabled": True, "source": "injected"}
    try:
        actual = LLMClient()
        if not actual.config.base_url or not actual.config.model:
            return None, {"llm_enabled": True, "source": "env", "status": "unconfigured"}
        return actual, {"llm_enabled": True, "source": "env", "model": actual.config.model}
    except Exception as exc:
        return None, {"llm_enabled": True, "source": "env", "status": "failed", "failure_reason": str(exc)}


def build_sentaurus_patch_refinement_plan(
    request: SentaurusPatchRefinerRequest,
    *,
    llm_client: ChatClient | None = None,
) -> SentaurusPatchRefinementPlan:
    source = request.source_state_path.expanduser().resolve()
    try:
        state = read_json(source)
        analysis = state.get("sentaurus_mutation_effect_analysis") if isinstance(state.get("sentaurus_mutation_effect_analysis"), dict) else {}
        goal_text_value = request.goal_text or state_goal_text(state)
        actual_llm_client, llm_policy = llm_client_for_request(request, llm_client)
        if not analysis:
            plan = SentaurusPatchRefinementPlan(
                status="insufficient_evidence",
                source_state_path=str(source),
                goal_text=goal_text_value,
                agent_policy=llm_policy,
                failure_reason="Source state has no sentaurus_mutation_effect_analysis.",
            )
        else:
            planner_request = SentaurusPatchPlannerRequest(goal_text=goal_text_value, source_state_path=source)
            project_root, contexts, warnings = resolve_decks(planner_request, state)
            decision = str(analysis.get("decision") or "")
            failure_reason = None
            if decision == "blocked_for_pareto_review":
                candidates: list[SentaurusPatchRefinementCandidate] = []
                selected = None
                status = "blocked_for_pareto_review"
            elif decision == "continue_refine":
                candidates = continue_same_direction_candidates(analysis=analysis, max_candidates=request.max_candidates)
                validate_refinement_candidates(candidates, contexts)
                if request.use_llm:
                    selected, selection_policy = choose_candidate_with_agent(
                        candidates=candidates,
                        analysis=analysis,
                        goal_text_value=goal_text_value,
                        allow_high_risk=request.allow_high_risk,
                        llm_client=actual_llm_client,
                    )
                    llm_policy = {**llm_policy, **selection_policy}
                    if selection_policy.get("status") == "invalid_selection" and not request.allow_llm_fallback:
                        selected = None
                        status = "failed"
                        failure_reason = "LLM selected an unknown or ineligible Sentaurus refinement candidate."
                    else:
                        status = "completed" if selected else "blocked_for_user_confirmation" if candidates else "no_actionable_candidates"
                else:
                    selected = select_candidate(candidates, allow_high_risk=request.allow_high_risk)
                    status = "completed" if selected else "blocked_for_user_confirmation" if candidates else "no_actionable_candidates"
            elif decision in {"switch_target", "reject_candidate"}:
                candidates = switch_target_candidates(
                    source_state_path=source,
                    state=state,
                    analysis=analysis,
                    goal_text_value=goal_text_value,
                    max_candidates=request.max_candidates,
                    allow_high_risk=request.allow_high_risk,
                )
                if request.use_llm:
                    selected, selection_policy = choose_candidate_with_agent(
                        candidates=candidates,
                        analysis=analysis,
                        goal_text_value=goal_text_value,
                        allow_high_risk=request.allow_high_risk,
                        llm_client=actual_llm_client,
                    )
                    llm_policy = {**llm_policy, **selection_policy}
                    if selection_policy.get("status") == "invalid_selection" and not request.allow_llm_fallback:
                        selected = None
                        status = "failed"
                        failure_reason = "LLM selected an unknown or ineligible Sentaurus refinement candidate."
                    else:
                        status = "completed" if selected else "blocked_for_user_confirmation" if candidates else "no_actionable_candidates"
                else:
                    selected = select_candidate(candidates, allow_high_risk=request.allow_high_risk)
                    status = "completed" if selected else "blocked_for_user_confirmation" if candidates else "no_actionable_candidates"
            else:
                candidates = []
                selected = None
                status = "insufficient_evidence"
            plan = SentaurusPatchRefinementPlan(
                status=status,
                source_state_path=str(source),
                goal_text=goal_text_value,
                analysis=analysis,
                project_path=str(project_root) if project_root else None,
                deck_files_inspected=[ctx.rel_file for ctx in contexts],
                candidates=candidates,
                selected_candidate=selected,
                agent_policy=llm_policy,
                failure_reason=failure_reason,
                final_summary={
                    "decision": decision,
                    "candidate_count": len(candidates),
                    "selected_candidate_id": selected.candidate_id if selected else None,
                    "requires_user_confirmation_count": sum(1 for candidate in candidates if candidate.requires_user_confirmation),
                    "verified_candidate_count": sum(1 for candidate in candidates if candidate.verified_patch_count == len(candidate.patches)),
                    "warnings": warnings,
                    "goal_tags": goal_tags(goal_text_value, state),
                },
            )
    except Exception as exc:
        plan = SentaurusPatchRefinementPlan(status="failed", source_state_path=str(source), goal_text=request.goal_text, failure_reason=str(exc))
    if request.output_path is not None:
        output_path = request.output_path.expanduser().resolve()
        plan.output_path = str(output_path)
        write_json(output_path, plan.model_dump(mode="json"))
    return plan
