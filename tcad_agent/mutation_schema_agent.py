from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from tcad_agent.evidence_lookup import PublicEvidenceLookupRequest, run_public_evidence_lookup
from tcad_agent.llm import LLMClient, LLMConfig
from tcad_agent.mutation_vocabulary import classify_mutation_variable, mutation_class_ids
from tcad_agent.public_sources import build_public_evidence_dossier
from tcad_agent.reporting import load_final_state
from tcad_agent.sentaurus_deck import apply_sentaurus_semantic_patch_text
from tcad_agent.sentaurus_patch_planner import (
    DeckContext,
    SentaurusPatchPlannerRequest,
    deck_evidence,
    goal_tags,
    resolve_decks,
)
from tcad_agent.task_planner import parse_json_object
from tcad_agent.task_spec import PROJECT_ROOT


class ChatClient(Protocol):
    config: LLMConfig

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        ...


ALLOWED_SEMANTIC_OPERATIONS = {
    "sentaurus_set_variable",
    "sentaurus_update_assignment",
    "sentaurus_upsert_assignment",
    "sentaurus_add_model",
}


class MutationSchemaDeckBinding(BaseModel):
    file: str
    variable: str
    value: str
    numeric_value: float | None = None
    token_overlap: list[str] = Field(default_factory=list)
    existing_vocabulary_classes: list[str] = Field(default_factory=list)


class MutationSchemaCandidate(BaseModel):
    class_id: str
    display_name: str
    target_kind: str
    default_risk_level: str
    requires_user_confirmation: bool = True
    variable_name_tokens: list[list[str]] = Field(default_factory=list)
    goal_tags: list[str] = Field(default_factory=list)
    primary_metrics: list[str] = Field(default_factory=list)
    tradeoff_metrics: list[str] = Field(default_factory=list)
    semantic_patch_operations: list[str] = Field(default_factory=list)
    expected_curve_evidence: list[str] = Field(default_factory=list)
    stop_conditions: list[str] = Field(default_factory=list)
    public_source_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    bindings: list[MutationSchemaDeckBinding] = Field(default_factory=list)
    schema_patch: dict[str, Any] = Field(default_factory=dict)
    validation_patch: dict[str, Any] = Field(default_factory=dict)
    validation_records: list[dict[str, Any]] = Field(default_factory=list)
    fixture_deck_path: str | None = None
    fixture_validation_records: list[dict[str, Any]] = Field(default_factory=list)
    verified_patch_count: int = 0
    ready_for_review: bool = False
    evidence_used: list[dict[str, Any]] = Field(default_factory=list)
    llm_rationale: str | None = None
    confidence: float = 0.4


class MutationSchemaExtensionRequest(BaseModel):
    goal_text: str
    source_state_path: Path | None = None
    project_path: Path | None = None
    deck_files: list[str] = Field(default_factory=list)
    proposed_target: str | None = None
    simulator: str = "sentaurus"
    output_dir: Path = PROJECT_ROOT / "runs" / "mutation_schema_extensions"
    output_path: Path | None = None
    max_candidates: int = Field(default=4, ge=1, le=12)
    enable_live_lookup: bool = False
    live_lookup_max_sources: int = Field(default=6, ge=1, le=24)
    use_llm: bool = False
    allow_llm_fallback: bool = True


class MutationSchemaExtensionResult(BaseModel):
    tool_name: str = "mutation_schema_agent"
    schema_version: str = "actsoft.tcad.mutation_schema_agent.v1"
    status: str
    goal_text: str
    proposed_target: str | None = None
    source_state_path: str | None = None
    project_path: str | None = None
    deck_files_inspected: list[str] = Field(default_factory=list)
    candidates: list[MutationSchemaCandidate] = Field(default_factory=list)
    selected_candidate: MutationSchemaCandidate | None = None
    public_evidence_dossier: dict[str, Any] = Field(default_factory=dict)
    live_lookup_result: dict[str, Any] | None = None
    model_decision: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    final_summary: dict[str, Any] = Field(default_factory=dict)
    output_path: str | None = None
    failure_reason: str | None = None


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def token_words(value: str) -> list[str]:
    raw = re.findall(r"[A-Za-z][A-Za-z0-9]*|\d+", value.replace("-", "_"))
    parts: list[str] = []
    for item in raw:
        parts.extend(piece for piece in item.split("_") if piece)
    return [part.upper() for part in parts if len(part) > 1]


def snake_id(value: str) -> str:
    words = [word.lower() for word in token_words(value)]
    if not words:
        return "proposed_mutation"
    return "_".join(words[:8])


def numeric_value(value: Any) -> float | None:
    text = str(value).strip()
    if text.startswith("@") and text.endswith("@"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def scaled_value_text(value: Any, goal_text: str) -> str | None:
    current = numeric_value(value)
    if current is None:
        return None
    text = goal_text.lower()
    factor = 0.9 if any(token in text for token in ["reduce", "lower", "decrease", "降低", "减少", "下降"]) else 1.1
    return f"{current * factor:.6g}"


def state_from_request(request: MutationSchemaExtensionRequest) -> dict[str, Any] | None:
    if not request.source_state_path:
        return None
    loaded = load_final_state(str(request.source_state_path.expanduser()))
    if isinstance(loaded, dict):
        return loaded
    try:
        return json.loads(request.source_state_path.expanduser().read_text(encoding="utf-8"))
    except Exception:
        return None


def contexts_from_request(
    request: MutationSchemaExtensionRequest,
    state: dict[str, Any] | None,
) -> tuple[Path | None, list[DeckContext], list[str]]:
    planner_request = SentaurusPatchPlannerRequest(
        goal_text=request.goal_text,
        source_state_path=request.source_state_path,
        project_path=request.project_path,
        deck_files=request.deck_files,
    )
    return resolve_decks(planner_request, state)


def relevant_target_tokens(request: MutationSchemaExtensionRequest) -> set[str]:
    source = request.proposed_target or request.goal_text
    stop = {
        "THE",
        "AND",
        "WITH",
        "WITHOUT",
        "NOT",
        "WORSE",
        "REDUCE",
        "LOWER",
        "IMPROVE",
        "TUNE",
        "PATCH",
        "SENTAURUS",
        "DEVSIM",
    }
    return {token for token in token_words(source) if token not in stop and not token.isdigit()}


def find_deck_bindings(
    request: MutationSchemaExtensionRequest,
    contexts: list[DeckContext],
) -> list[MutationSchemaDeckBinding]:
    target_tokens = relevant_target_tokens(request)
    bindings: list[MutationSchemaDeckBinding] = []
    for ctx in contexts:
        for assignment in ctx.ir.set_variables:
            variable = assignment.key
            existing = sorted(classify_mutation_variable(variable))
            if existing:
                continue
            variable_tokens = set(token_words(variable))
            overlap = sorted(variable_tokens & target_tokens)
            if not overlap and request.proposed_target:
                continue
            if not overlap and not any(token in variable_tokens for token in {"SURFACE", "RECOMB", "POLARIZATION", "WORKFUNCTION", "INTERFACE", "CHARGE"}):
                continue
            bindings.append(
                MutationSchemaDeckBinding(
                    file=ctx.rel_file,
                    variable=variable,
                    value=str(assignment.value),
                    numeric_value=numeric_value(assignment.value),
                    token_overlap=overlap,
                    existing_vocabulary_classes=existing,
                )
            )
    return bindings


def risk_from_target(class_id: str, target_kind: str, tokens: set[str]) -> tuple[str, bool]:
    text = " ".join([class_id, target_kind, *tokens]).lower()
    if any(token in text for token in ["geometry", "process", "dose", "implant", "junction", "trench", "oxide", "doping", "polarization", "workfunction"]):
        return "high", True
    if any(token in text for token in ["trap", "interface", "surface", "recomb", "model", "lifetime"]):
        return "medium", True
    return "medium", True


def metrics_from_goal(goal_text: str) -> tuple[list[str], list[str], list[str], list[str]]:
    tags = goal_tags(goal_text)
    goal_tag_list = [key for key, active in tags.items() if active]
    primary: list[str] = []
    tradeoff: list[str] = []
    expected: list[str] = []
    stops: list[str] = []
    if tags.get("leakage"):
        primary.extend(["leakage_abs_current_at_target_a", "reverse_leakage_current_a", "ioff_current_a"])
        expected.append("leakage window moves in the intended direction against baseline")
        stops.append("leakage slope or low-current floor does not move")
    if tags.get("bv"):
        primary.append("breakdown_voltage_at_threshold_v")
        tradeoff.append("breakdown_voltage_at_threshold_v")
        expected.append("BV bracket moves consistently with the mutation hypothesis")
        stops.append("BV bracket regresses beyond the configured constraint")
    if tags.get("field"):
        primary.append("max_electric_field_v_per_cm")
        tradeoff.append("max_electric_field_v_per_cm")
        expected.append("field peak value or location changes in overlay comparison")
        stops.append("field peak shifts into a worse corner or oxide/interface hotspot")
    if tags.get("ron"):
        primary.append("specific_on_resistance_ohm_cm2")
        tradeoff.append("specific_on_resistance_ohm_cm2")
        expected.append("Ron movement is compared against BV/leakage/field constraints")
        stops.append("Ron tradeoff dominates the primary improvement")
    if not primary:
        primary = ["leakage_abs_current_at_target_a"]
        expected.append("primary curve or extracted metric changes measurably versus baseline")
        stops.append("no comparable metric or curve-shape movement is observed")
    tradeoff = list(dict.fromkeys([*tradeoff, "breakdown_voltage_at_threshold_v", "specific_on_resistance_ohm_cm2", "max_electric_field_v_per_cm"]))
    return goal_tag_list, list(dict.fromkeys(primary)), tradeoff, expected, stops


def sanitize_llm_schema(raw: dict[str, Any], fallback_class_id: str, target_tokens: set[str]) -> dict[str, Any]:
    candidate = raw.get("schema") if isinstance(raw.get("schema"), dict) else raw
    class_id = snake_id(str(candidate.get("class_id") or fallback_class_id))
    if class_id in set(mutation_class_ids()):
        class_id = f"{class_id}_extension"
    target_kind = str(candidate.get("target_kind") or "model_or_process_parameter")
    risk, confirmation = risk_from_target(class_id, target_kind, target_tokens)
    raw_risk = str(candidate.get("default_risk_level") or risk).lower()
    if raw_risk in {"low", "medium", "high"}:
        risk = raw_risk
    operations = [
        str(item)
        for item in candidate.get("semantic_patch_operations") or ["sentaurus_set_variable"]
        if str(item) in ALLOWED_SEMANTIC_OPERATIONS
    ] or ["sentaurus_set_variable"]
    tokens = candidate.get("variable_name_tokens")
    if not isinstance(tokens, list) or not tokens:
        tokens = [[token for token in sorted(target_tokens)[:4]]]
    normalized_tokens = []
    for group in tokens:
        if isinstance(group, list):
            normalized_tokens.append([str(item).upper() for item in group if str(item).strip()])
    return {
        "class_id": class_id,
        "display_name": str(candidate.get("display_name") or class_id.replace("_", " ").title()),
        "target_kind": target_kind,
        "default_risk_level": risk,
        "requires_user_confirmation": bool(candidate.get("requires_user_confirmation", confirmation or risk != "low")),
        "variable_name_tokens": normalized_tokens or [[token for token in sorted(target_tokens)[:4]]],
        "semantic_patch_operations": operations,
        "llm_rationale": str(candidate.get("rationale") or raw.get("rationale") or "") or None,
        "confidence": float(candidate.get("confidence") or raw.get("confidence") or 0.55),
    }


def deterministic_schema_seed(request: MutationSchemaExtensionRequest, bindings: list[MutationSchemaDeckBinding]) -> dict[str, Any]:
    raw_target = request.proposed_target or (bindings[0].variable if bindings else request.goal_text)
    target_tokens = relevant_target_tokens(request) or set(token_words(raw_target))
    class_id = snake_id(request.proposed_target or raw_target)
    if class_id in set(mutation_class_ids()):
        class_id = f"{class_id}_extension"
    target_kind = "model_or_process_parameter"
    if any(token in target_tokens for token in {"SURFACE", "RECOMB", "INTERFACE", "CHARGE"}):
        target_kind = "interface_or_surface_model_parameter"
    if any(token in target_tokens for token in {"GEOMETRY", "OXIDE", "JUNCTION", "TRENCH", "IMPLANT", "DOSE"}):
        target_kind = "geometry_or_process_parameter"
    risk, confirmation = risk_from_target(class_id, target_kind, target_tokens)
    return {
        "class_id": class_id,
        "display_name": class_id.replace("_", " ").title(),
        "target_kind": target_kind,
        "default_risk_level": risk,
        "requires_user_confirmation": confirmation,
        "variable_name_tokens": [[token for token in sorted(target_tokens)[:4]]] if target_tokens else [],
        "semantic_patch_operations": ["sentaurus_set_variable"],
        "llm_rationale": None,
        "confidence": 0.45,
    }


def ask_llm_for_schema(
    request: MutationSchemaExtensionRequest,
    bindings: list[MutationSchemaDeckBinding],
    public_evidence: dict[str, Any],
    *,
    llm_client: ChatClient | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not request.use_llm:
        return None, {"status": "skipped", "fallback_used": True, "reason": "use_llm is false"}
    client = llm_client or LLMClient()
    binding_payload = [binding.model_dump(mode="json") for binding in bindings[:12]]
    system = (
        "You propose safe, review-only TCAD mutation vocabulary schemas. "
        "Return JSON only. Do not invent proprietary syntax. "
        "Use only the deck bindings and public evidence provided."
    )
    user = {
        "task": "propose one mutation vocabulary schema extension",
        "goal_text": request.goal_text,
        "proposed_target": request.proposed_target,
        "deck_bindings": binding_payload,
        "public_evidence_gate": public_evidence.get("evidence_gate"),
        "public_sources": [
            {"source_id": card.get("source_id"), "name": card.get("name"), "useful_for": card.get("useful_for")}
            for card in public_evidence.get("source_cards", [])[:8]
            if isinstance(card, dict)
        ],
        "response_schema": {
            "schema": {
                "class_id": "snake_case_new_class_id",
                "display_name": "human name",
                "target_kind": "model_parameter|process_parameter|geometry_parameter|interface_or_surface_model_parameter",
                "default_risk_level": "low|medium|high",
                "requires_user_confirmation": True,
                "variable_name_tokens": [["TOKENS", "MATCHING", "LOCAL", "DECK"]],
                "semantic_patch_operations": ["sentaurus_set_variable"],
                "rationale": "why this schema is justified by deck evidence and public evidence",
                "confidence": 0.0,
            }
        },
    }
    try:
        raw = client.chat(system=system, user=json.dumps(user, ensure_ascii=False, indent=2), temperature=0.1)
        parsed = parse_json_object(raw)
        if not isinstance(parsed, dict):
            raise ValueError("LLM did not return a JSON object")
        seed = sanitize_llm_schema(parsed, snake_id(request.proposed_target or request.goal_text), relevant_target_tokens(request))
        return seed, {
            "status": "completed",
            "fallback_used": False,
            "model": getattr(getattr(client, "config", None), "model", None),
            "raw_response": raw,
            "parsed_response": parsed,
        }
    except Exception as exc:
        if not request.allow_llm_fallback:
            raise
        return None, {"status": "fallback", "fallback_used": True, "failure_reason": str(exc)}


def source_ids_from_public_evidence(public_evidence: dict[str, Any]) -> list[str]:
    output = []
    for card in public_evidence.get("source_cards") or []:
        if isinstance(card, dict) and card.get("source_id"):
            output.append(str(card["source_id"]))
    return list(dict.fromkeys(output))


def fixture_text_for_binding(binding: MutationSchemaDeckBinding) -> str:
    return (
        f"set {binding.variable} {binding.value}\n\n"
        "Physics {\n"
        "  Mobility( DopingDep )\n"
        "}\n\n"
        "Math {\n"
        "  Iterations=20\n"
        "}\n"
    )


def build_schema_candidate(
    request: MutationSchemaExtensionRequest,
    seed: dict[str, Any],
    binding: MutationSchemaDeckBinding,
    ctx_by_file: dict[str, DeckContext],
    public_evidence: dict[str, Any],
    output_dir: Path,
) -> MutationSchemaCandidate:
    tags, primary, tradeoff, expected, stops = metrics_from_goal(request.goal_text)
    next_value = scaled_value_text(binding.value, request.goal_text) or binding.value
    binding_token_group = [token for token in token_words(binding.variable) if not token.isdigit()]
    seed_token_groups = seed["variable_name_tokens"]
    if binding_token_group and not any(set(group).issubset(set(binding_token_group)) for group in seed_token_groups):
        seed_token_groups = [binding_token_group]
    patch = {
        "file": binding.file,
        "operation": "sentaurus_set_variable",
        "variable": binding.variable,
        "value": next_value,
        "reason": f"Schema-extension validation probe for `{seed['class_id']}`; review before execution.",
        "required": True,
    }
    validation_records: list[dict[str, Any]] = []
    ctx = ctx_by_file.get(binding.file)
    if ctx:
        try:
            _, record, _ = apply_sentaurus_semantic_patch_text(ctx.text, patch, source_path=binding.file)
            record["file"] = binding.file
            validation_records.append(record)
        except Exception as exc:
            validation_records.append({"file": binding.file, "operation": patch["operation"], "applied": False, "verified": False, "error": str(exc)})
    fixture_dir = output_dir / "fixtures"
    fixture_path = fixture_dir / f"{seed['class_id']}_{binding.variable}.cmd"
    fixture_text = fixture_text_for_binding(binding)
    write_text(fixture_path, fixture_text)
    fixture_records: list[dict[str, Any]] = []
    try:
        _, fixture_record, _ = apply_sentaurus_semantic_patch_text(fixture_text, {**patch, "file": fixture_path.name}, source_path=fixture_path.name)
        fixture_record["file"] = fixture_path.name
        fixture_records.append(fixture_record)
    except Exception as exc:
        fixture_records.append({"file": fixture_path.name, "operation": patch["operation"], "applied": False, "verified": False, "error": str(exc)})
    verified = sum(1 for record in validation_records if record.get("verified"))
    fixture_verified = any(record.get("verified") for record in fixture_records)
    gate = public_evidence.get("evidence_gate") if isinstance(public_evidence.get("evidence_gate"), dict) else {}
    schema_patch = {
        "class_id": seed["class_id"],
        "display_name": seed["display_name"],
        "target_kind": seed["target_kind"],
        "default_risk_level": seed["default_risk_level"],
        "requires_user_confirmation": seed["requires_user_confirmation"],
        "variable_name_tokens": seed_token_groups,
        "goal_tags": tags,
        "primary_metrics": primary,
        "tradeoff_metrics": tradeoff,
        "semantic_patch_operations": seed["semantic_patch_operations"],
        "expected_curve_evidence": expected,
        "stop_conditions": stops,
        "public_source_ids": source_ids_from_public_evidence(public_evidence),
        "notes": [
            "Generated by mutation_schema_agent as a review-only promotion package.",
            "Do not add to static vocabulary until fixture, public evidence, and real project validation pass.",
        ],
    }
    ready = bool(gate.get("passed")) and verified > 0 and fixture_verified
    return MutationSchemaCandidate(
        class_id=seed["class_id"],
        display_name=seed["display_name"],
        target_kind=seed["target_kind"],
        default_risk_level=seed["default_risk_level"],
        requires_user_confirmation=bool(seed["requires_user_confirmation"]),
        variable_name_tokens=seed_token_groups,
        goal_tags=tags,
        primary_metrics=primary,
        tradeoff_metrics=tradeoff,
        semantic_patch_operations=seed["semantic_patch_operations"],
        expected_curve_evidence=expected,
        stop_conditions=stops,
        public_source_ids=schema_patch["public_source_ids"],
        notes=schema_patch["notes"],
        bindings=[binding],
        schema_patch=schema_patch,
        validation_patch=patch,
        validation_records=validation_records,
        fixture_deck_path=str(fixture_path.resolve()),
        fixture_validation_records=fixture_records,
        verified_patch_count=verified,
        ready_for_review=ready,
        evidence_used=[
            {"kind": "public_evidence_gate", "evidence_gate": gate},
            deck_evidence(ctx) if ctx else {"kind": "sentaurus_deck_ir", "file": binding.file, "status": "missing_context"},
            {"kind": "deck_binding", **binding.model_dump(mode="json")},
        ],
        llm_rationale=seed.get("llm_rationale"),
        confidence=float(seed.get("confidence") or 0.45),
    )


def run_mutation_schema_extension(
    request: MutationSchemaExtensionRequest,
    *,
    llm_client: ChatClient | None = None,
) -> MutationSchemaExtensionResult:
    output_dir = request.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_state_path = str(request.source_state_path.expanduser().resolve()) if request.source_state_path else None
    live_lookup_result = None
    if request.enable_live_lookup:
        lookup = run_public_evidence_lookup(
            PublicEvidenceLookupRequest(
                goal_text=request.goal_text,
                simulator=request.simulator,
                live=True,
                max_sources=request.live_lookup_max_sources,
                output_path=output_dir / "public_evidence_lookup.json",
            )
        )
        live_lookup_result = lookup.model_dump(mode="json")
    public_evidence = build_public_evidence_dossier(
        request.goal_text,
        simulator=request.simulator,
        live_lookup_result=live_lookup_result,
    ).model_dump(mode="json")
    state = state_from_request(request)
    project_root, contexts, warnings = contexts_from_request(request, state)
    artifacts: dict[str, str] = {}
    if live_lookup_result and live_lookup_result.get("output_path"):
        artifacts["public_evidence_lookup"] = str(live_lookup_result["output_path"])
    model_decision: dict[str, Any] = {}
    candidates: list[MutationSchemaCandidate] = []
    failure_reason = None
    try:
        if not contexts:
            status = "no_deck_context"
            failure_reason = "No parseable Sentaurus deck files were available for schema validation."
        elif not bool((public_evidence.get("evidence_gate") or {}).get("passed")):
            status = "blocked_no_public_evidence"
            failure_reason = "No public evidence category matched this mutation schema request."
        else:
            bindings = find_deck_bindings(request, contexts)
            if not bindings:
                status = "blocked_no_deck_binding"
                failure_reason = "No unknown deck variable matched the proposed target or goal text."
            else:
                llm_seed, model_decision = ask_llm_for_schema(
                    request,
                    bindings,
                    public_evidence,
                    llm_client=llm_client,
                )
                seed = llm_seed or deterministic_schema_seed(request, bindings)
                ctx_by_file = {ctx.rel_file: ctx for ctx in contexts}
                for binding in bindings[: request.max_candidates]:
                    candidates.append(build_schema_candidate(request, seed, binding, ctx_by_file, public_evidence, output_dir))
                status = "completed" if any(candidate.ready_for_review for candidate in candidates) else "candidate_only"
    except Exception as exc:
        status = "failed"
        failure_reason = str(exc)
    selected = sorted(
        [candidate for candidate in candidates if candidate.ready_for_review],
        key=lambda item: item.confidence,
        reverse=True,
    )
    result = MutationSchemaExtensionResult(
        status=status,
        goal_text=request.goal_text,
        proposed_target=request.proposed_target,
        source_state_path=source_state_path,
        project_path=str(project_root) if project_root else str(request.project_path.expanduser()) if request.project_path else None,
        deck_files_inspected=[ctx.rel_file for ctx in contexts],
        candidates=candidates,
        selected_candidate=selected[0] if selected else None,
        public_evidence_dossier=public_evidence,
        live_lookup_result=live_lookup_result,
        model_decision=model_decision,
        artifacts=artifacts,
        final_summary={
            "candidate_count": len(candidates),
            "ready_for_review_count": sum(1 for candidate in candidates if candidate.ready_for_review),
            "deck_binding_count": sum(len(candidate.bindings) for candidate in candidates),
            "warnings": warnings,
            "does_not_modify_static_vocabulary": True,
            "does_not_execute_solver": True,
        },
        failure_reason=failure_reason,
    )
    output_path = request.output_path.expanduser().resolve() if request.output_path else output_dir / "mutation_schema_extension.json"
    result.output_path = str(output_path)
    artifacts["mutation_schema_extension"] = str(output_path)
    result.artifacts = artifacts
    write_json(output_path, result.model_dump(mode="json"))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a review-only mutation vocabulary schema extension package.")
    parser.add_argument("--goal", "--goal-text", dest="goal_text", required=True)
    parser.add_argument("--state", "--source-state-path", dest="source_state_path", type=Path, default=None)
    parser.add_argument("--project", "--project-path", dest="project_path", type=Path, default=None)
    parser.add_argument("--deck-file", action="append", default=[])
    parser.add_argument("--target", "--proposed-target", dest="proposed_target", default=None)
    parser.add_argument("--simulator", default="sentaurus")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "mutation_schema_extensions")
    parser.add_argument("--output", "--output-path", dest="output_path", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--enable-live-lookup", action="store_true")
    parser.add_argument("--live-lookup-max-sources", type=int, default=6)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--no-llm-fallback", action="store_true")
    return parser.parse_args()


def request_from_args(args: argparse.Namespace) -> MutationSchemaExtensionRequest:
    return MutationSchemaExtensionRequest(
        goal_text=args.goal_text,
        source_state_path=args.source_state_path,
        project_path=args.project_path,
        deck_files=args.deck_file,
        proposed_target=args.proposed_target,
        simulator=args.simulator,
        output_dir=args.output_dir,
        output_path=args.output_path,
        max_candidates=args.max_candidates,
        enable_live_lookup=args.enable_live_lookup,
        live_lookup_max_sources=args.live_lookup_max_sources,
        use_llm=args.use_llm,
        allow_llm_fallback=not args.no_llm_fallback,
    )


def main() -> None:
    result = run_mutation_schema_extension(request_from_args(parse_args()))
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.status in {"completed", "candidate_only", "blocked_no_deck_binding"} else 1)


if __name__ == "__main__":
    main()
