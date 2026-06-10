from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.reporting import final_artifacts, final_metrics, load_final_state
from tcad_agent.sentaurus_deck import (
    SentaurusDeckBlock,
    SentaurusDeckIR,
    apply_sentaurus_semantic_patch_text,
    parse_sentaurus_deck_file,
    unquote,
)


class SentaurusPatchCandidate(BaseModel):
    candidate_id: str
    title: str
    score: float
    hypothesis: str
    risk_level: str = "medium"
    requires_user_confirmation: bool = False
    target_file: str
    patches: list[dict[str, Any]] = Field(default_factory=list)
    validation_records: list[dict[str, Any]] = Field(default_factory=list)
    verified_patch_count: int = 0
    expected_observation: str
    stop_condition: str
    fallback_alternatives: list[str] = Field(default_factory=list)
    rationale: str
    evidence_used: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.5


class SentaurusPatchPlan(BaseModel):
    tool_name: str = "sentaurus_patch_planner"
    schema_version: str = "actsoft.tcad.sentaurus_patch_planner.v1"
    status: str
    goal_text: str
    source_state_path: str | None = None
    project_path: str | None = None
    deck_files_inspected: list[str] = Field(default_factory=list)
    candidates: list[SentaurusPatchCandidate] = Field(default_factory=list)
    selected_candidate: SentaurusPatchCandidate | None = None
    evidence_summary: dict[str, Any] = Field(default_factory=dict)
    final_summary: dict[str, Any] = Field(default_factory=dict)
    output_path: str | None = None
    failure_reason: str | None = None


class SentaurusPatchPlannerRequest(BaseModel):
    goal_text: str
    source_state_path: Path | None = None
    project_path: Path | None = None
    deck_files: list[str] = Field(default_factory=list)
    output_path: Path | None = None
    max_candidates: int = Field(default=8, ge=1, le=24)
    allow_high_risk: bool = False


@dataclass
class DeckContext:
    path: Path
    rel_file: str
    ir: SentaurusDeckIR
    text: str


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def normalized_text(value: str) -> str:
    return value.lower().replace("_", " ").replace("-", " ")


def goal_tags(goal_text: str, state: dict[str, Any] | None = None) -> dict[str, bool]:
    text = normalized_text(goal_text)
    issues = []
    if isinstance(state, dict):
        quality = state.get("quality_report") or {}
        for issue in quality.get("issues") or []:
            if isinstance(issue, dict) and issue.get("code"):
                issues.append(str(issue["code"]).lower())
        for item in state.get("log_diagnostics") or []:
            if isinstance(item, dict) and item.get("code"):
                issues.append(str(item["code"]).lower())
    issue_text = " ".join(issues)
    return {
        "convergence": any(token in text for token in ["converge", "convergence", "newton", "step too small", "收敛", "步长", "牛顿"])
        or "convergence" in issue_text,
        "bv": any(token in text for token in ["breakdown", "bv", "击穿", "耐压", "反偏", "reverse bias"]),
        "reverse": any(token in text for token in ["reverse", "反偏", "负偏"]),
        "leakage": any(token in text for token in ["leakage", "off current", "ioff", "漏电", "关态"]),
        "field": any(token in text for token in ["electric field", "field peak", "peak field", "电场", "场峰", "field"]),
        "ron": any(token in text for token in ["ron", "on resistance", "导通电阻", "比导通", "导通"]),
        "transient": any(token in text for token in ["transient", "瞬态", "time step"]),
        "tradeoff_guard": any(token in text for token in ["不能变差", "不能恶化", "not worse", "without hurting", "不牺牲", "保持"]),
    }


def extract_voltage_target(goal_text: str) -> float | None:
    matches: list[float] = []
    for match in re.finditer(r"(?<![\w.])([+-]?\d+(?:\.\d+)?)(?:\s*)(kv|v|伏)", goal_text, flags=re.IGNORECASE):
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit == "kv":
            value *= 1000.0
        matches.append(value)
    if not matches:
        return None
    return max(matches, key=lambda item: abs(item))


def numeric_value(value: Any) -> float | None:
    text = str(value).strip()
    if text.startswith("@") and text.endswith("@"):
        return None
    match = re.match(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$", text)
    if not match:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def scaled_numeric_text(value: Any, factor: float) -> str | None:
    numeric = numeric_value(value)
    if numeric is None:
        return None
    scaled = numeric * factor
    return f"{scaled:.6g}"


def same_token(left: str, right: str) -> bool:
    return left.strip().lower() == right.strip().lower()


def path_endswith(path: list[str], suffix: list[str]) -> bool:
    if len(path) < len(suffix):
        return False
    return [part.lower() for part in path[-len(suffix) :]] == [part.lower() for part in suffix]


def first_block(ctx: DeckContext, suffix: list[str]) -> SentaurusDeckBlock | None:
    for block in ctx.ir.sections:
        if path_endswith(block.path, suffix):
            return block
    return None


def block_assignment(block: SentaurusDeckBlock, key: str) -> str | None:
    for assignment in block.assignments:
        if same_token(assignment.key, key):
            return unquote(assignment.value)
    return None


def section_assignment_patch(
    ctx: DeckContext,
    block: SentaurusDeckBlock,
    *,
    parameter: str,
    value: Any,
    selector: dict[str, Any] | None = None,
    upsert: bool = False,
    reason: str,
) -> dict[str, Any]:
    return {
        "file": ctx.rel_file,
        "operation": "sentaurus_upsert_assignment" if upsert else "sentaurus_update_assignment",
        "section_path": block.path,
        "selector": selector or {},
        "parameter": parameter,
        "value": value,
        "reason": reason,
        "required": True,
    }


def set_variable_patch(ctx: DeckContext, variable: str, value: Any, *, reason: str) -> dict[str, Any]:
    return {
        "file": ctx.rel_file,
        "operation": "sentaurus_set_variable",
        "variable": variable,
        "value": value,
        "reason": reason,
        "required": True,
    }


def deck_files_from_state(state: dict[str, Any] | None) -> list[str]:
    if not isinstance(state, dict):
        return []
    request = state.get("request") if isinstance(state.get("request"), dict) else {}
    summary = state.get("final_summary") if isinstance(state.get("final_summary"), dict) else {}
    parameters = summary.get("parameters") if isinstance(summary.get("parameters"), dict) else {}
    for value in [request.get("deck_files"), parameters.get("deck_files")]:
        if isinstance(value, list) and value:
            return [str(item) for item in value if item]
    return []


def project_path_from_state(state: dict[str, Any] | None) -> Path | None:
    if not isinstance(state, dict):
        return None
    for key in ["project_copy_path", "project_path"]:
        value = state.get(key)
        if value and Path(str(value)).exists():
            return Path(str(value)).expanduser().resolve()
    artifacts = final_artifacts(state)
    for key in ["project_copy", "sentaurus_project_copy"]:
        value = artifacts.get(key)
        if value and Path(value).exists():
            return Path(value).expanduser().resolve()
    return None


def default_deck_files(project_root: Path) -> list[str]:
    files = sorted(project_root.rglob("*.cmd"))
    return [str(path.relative_to(project_root)) for path in files[:12]]


def resolve_decks(request: SentaurusPatchPlannerRequest, state: dict[str, Any] | None) -> tuple[Path | None, list[DeckContext], list[str]]:
    project_root = request.project_path.expanduser().resolve() if request.project_path else project_path_from_state(state)
    warnings: list[str] = []
    if project_root is None:
        return None, [], ["No project_path or Sentaurus project_copy_path was available."]
    if not project_root.exists():
        return project_root, [], [f"Project path does not exist: {project_root}"]
    deck_files = request.deck_files or deck_files_from_state(state) or default_deck_files(project_root)
    contexts: list[DeckContext] = []
    for raw in deck_files:
        raw_path = Path(raw)
        path = raw_path.expanduser().resolve() if raw_path.is_absolute() else (project_root / raw_path).resolve()
        if not path.exists() or not path.is_file():
            warnings.append(f"Deck file not found: {raw}")
            continue
        try:
            rel_file = str(path.relative_to(project_root))
        except ValueError:
            rel_file = path.name
        try:
            contexts.append(
                DeckContext(
                    path=path,
                    rel_file=rel_file,
                    ir=parse_sentaurus_deck_file(path),
                    text=path.read_text(encoding="utf-8", errors="replace"),
                )
            )
        except Exception as exc:
            warnings.append(f"Failed to parse deck {raw}: {exc}")
    return project_root, contexts, warnings


def deck_evidence(ctx: DeckContext) -> dict[str, Any]:
    variables = [assignment.key for assignment in ctx.ir.set_variables]
    section_paths = [".".join(block.path) for block in ctx.ir.sections[:24]]
    return {
        "kind": "sentaurus_deck_ir",
        "file": ctx.rel_file,
        "variables": variables[:24],
        "section_paths": section_paths,
        "warnings": ctx.ir.warnings,
    }


def state_evidence(state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    quality = state.get("quality_report") if isinstance(state.get("quality_report"), dict) else {}
    issues = [
        issue.get("code")
        for issue in quality.get("issues") or []
        if isinstance(issue, dict) and issue.get("code")
    ]
    diagnostics = [
        item.get("code")
        for item in state.get("log_diagnostics") or []
        if isinstance(item, dict) and item.get("code")
    ]
    metrics = final_metrics(state)
    return {
        "tool_name": state.get("tool_name"),
        "status": state.get("status"),
        "quality_status": quality.get("status"),
        "issue_codes": issues,
        "log_diagnostic_codes": diagnostics,
        "metrics": {key: metrics[key] for key in sorted(metrics) if key in {"breakdown_voltage_v", "leakage_current_a", "max_electric_field_v_per_cm", "specific_on_resistance_ohm_cm2", "curve_points"}},
    }


def convergence_candidate(ctx: DeckContext, tags: dict[str, bool], state: dict[str, Any] | None) -> SentaurusPatchCandidate | None:
    state_info = state_evidence(state)
    has_state_issue = any("convergence" in str(code).lower() for code in state_info.get("issue_codes", []) + state_info.get("log_diagnostic_codes", []))
    if not (tags["convergence"] or has_state_issue or tags["bv"] or tags["transient"]):
        return None

    patches: list[dict[str, Any]] = []
    math = first_block(ctx, ["Math"])
    if math:
        current_iterations = block_assignment(math, "Iterations")
        numeric_iterations = numeric_value(current_iterations)
        next_iterations = int(max((numeric_iterations or 20) * 1.5, (numeric_iterations or 20) + 10))
        patches.append(
            section_assignment_patch(
                ctx,
                math,
                parameter="Iterations",
                value=next_iterations,
                reason="Give Newton iterations more room before declaring non-convergence.",
            )
        )
        current_not_damped = block_assignment(math, "NotDamped")
        if current_not_damped is None:
            patches.append(
                section_assignment_patch(
                    ctx,
                    math,
                    parameter="NotDamped",
                    value=100,
                    upsert=True,
                    reason="Add conservative damping control for difficult high-bias continuation.",
                )
            )
        else:
            numeric_not_damped = numeric_value(current_not_damped) or 50
            patches.append(
                section_assignment_patch(
                    ctx,
                    math,
                    parameter="NotDamped",
                    value=int(max(numeric_not_damped * 1.5, numeric_not_damped + 25)),
                    reason="Increase the non-damped iteration window for difficult bias points.",
                )
            )

    for suffix in (["Solve", "Quasistationary"], ["Solve", "Transient"]):
        block = first_block(ctx, suffix)
        if not block:
            continue
        for parameter, factor in [("InitialStep", 0.1), ("MaxStep", 0.5)]:
            current = block_assignment(block, parameter)
            next_value = scaled_numeric_text(current, factor)
            if next_value is None:
                continue
            patches.append(
                section_assignment_patch(
                    ctx,
                    block,
                    parameter=parameter,
                    value=next_value,
                    reason=f"Reduce {'.'.join(block.path)} {parameter} to make continuation less aggressive.",
                )
            )
        if len(patches) >= 4:
            break
    if not patches:
        return None
    score = 0.88 if tags["convergence"] or has_state_issue else 0.66
    return SentaurusPatchCandidate(
        candidate_id=f"{ctx.rel_file}:convergence_step_control",
        title="Conservative continuation and Newton controls",
        score=score,
        hypothesis="The next Sentaurus run is limited by nonlinear continuation aggressiveness rather than device physics.",
        risk_level="low",
        target_file=ctx.rel_file,
        patches=patches[:4],
        expected_observation="Log should advance to later bias/time points with fewer Newton or step-size failures; extracted curve should keep the same polarity and trend.",
        stop_condition="Stop using this direction once the baseline bias range completes and remaining gaps are physical metrics instead of solver failure.",
        fallback_alternatives=["tighten mesh near peak field", "split the bias ramp into staged goals", "inspect contact and model setup if the first bias point fails"],
        rationale="Continuation and Math controls are reversible numeric edits, so they are the safest first Sentaurus patch when convergence risk exists.",
        evidence_used=[
            deck_evidence(ctx),
            {"kind": "public_methodology", "source_id": "sentaurus_quasistationary_training", "note": "InitialStep/MaxStep/MinStep continuation control pattern"},
            state_info,
        ],
        confidence=0.78,
    )


def goal_blocks(ctx: DeckContext) -> list[SentaurusDeckBlock]:
    return [block for block in ctx.ir.sections if same_token(block.name, "Goal")]


def bias_candidate(ctx: DeckContext, tags: dict[str, bool], goal_text: str) -> SentaurusPatchCandidate | None:
    voltage = extract_voltage_target(goal_text)
    if voltage is None or not tags["bv"]:
        return None
    preferred = {"cathode", "drain", "collector"}
    for block in goal_blocks(ctx):
        name = block_assignment(block, "Name")
        if not name or name.lower() not in preferred:
            continue
        current_voltage = block_assignment(block, "Voltage")
        current_numeric = numeric_value(current_voltage)
        sign = -1.0 if tags["reverse"] or name.lower() == "cathode" or (current_numeric is not None and current_numeric < 0) else 1.0
        target = sign * abs(voltage)
        patch = section_assignment_patch(
            ctx,
            block,
            selector={"Name": name},
            parameter="Voltage",
            value=f"{target:.6g}",
            reason=f"Set the Sentaurus Goal voltage to the requested BV target of {abs(voltage):.6g} V.",
        )
        return SentaurusPatchCandidate(
            candidate_id=f"{ctx.rel_file}:bv_goal_{name}_{abs(voltage):.6g}v",
            title=f"Move {name} Goal voltage toward requested BV target",
            score=0.86,
            hypothesis="The deck already has a Quasistationary Goal block, so the natural-language BV target can be expressed as a verified Goal Voltage patch.",
            risk_level="medium",
            target_file=ctx.rel_file,
            patches=[patch],
            expected_observation="The next extracted IV curve should bracket the requested breakdown voltage or fail near a narrower high-field region.",
            stop_condition="Stop once breakdown is bracketed around the requested voltage or leakage/field constraints clearly fail before the target.",
            fallback_alternatives=["reduce continuation step size first", "add a measured/reference extraction contract", "inspect field peak location before geometry edits"],
            rationale="Changing the sweep endpoint is a semantic bias patch, not a geometry or model change, and it is directly tied to the requested BV target.",
            evidence_used=[deck_evidence(ctx), {"kind": "natural_language_target", "target_voltage_v": voltage, "goal_tags": tags}],
            confidence=0.75,
        )
    return None


def variable_classes(name: str) -> set[str]:
    upper = name.upper()
    classes: set[str] = set()
    if "DOP" in upper and "DRIFT" in upper:
        classes.add("drift_doping")
    if "LIFETIME" in upper or upper.startswith("TAU") or "_TAU" in upper:
        classes.add("lifetime")
        if any(token in upper for token in ["P_BODY", "N_DRIFT", "REGION", "ANODE", "CATHODE", "BASE", "EMITTER", "COLLECTOR"]):
            classes.add("region_specific_lifetime")
    if "TRAP" in upper and any(token in upper for token in ["DENS", "CONC", "N"]):
        classes.add("trap_density")
    if "FIELD" in upper and "PLATE" in upper:
        classes.add("field_plate")
    if "GUARD" in upper and "RING" in upper:
        classes.add("guard_ring")
    if "OXIDE" in upper or upper.startswith("TOX") or "_TOX" in upper:
        classes.add("oxide_thickness")
    if "IMPLANT" in upper and "DOSE" in upper:
        classes.add("implant_dose")
    if "JUNCTION" in upper and ("DEPTH" in upper or upper.endswith("_XJ")):
        classes.add("junction_depth")
    if "TRENCH" in upper and ("RADIUS" in upper or "CORNER" in upper):
        classes.add("trench_corner_radius")
    return classes


def variable_mutation_direction(var_class: str, tags: dict[str, bool]) -> tuple[float, str, str, str, str, float] | None:
    if var_class == "lifetime":
        if tags["leakage"]:
            return (2.0, "low", "Increase lifetime to reduce SRH generation-driven reverse leakage.", "Leakage should fall with limited BV/Ron movement.", "Stop if leakage does not move or transient/storage metrics become the active constraint.", 0.8)
        return None
    if var_class == "region_specific_lifetime":
        if tags["leakage"]:
            return (1.5, "medium", "Adjust region-specific lifetime before global process geometry changes.", "Leakage should improve in the targeted region with less global side effect.", "Stop if the curve change is not localized or stored charge becomes worse.", 0.72)
        return None
    if var_class == "trap_density":
        if tags["leakage"]:
            return (0.5, "medium", "Reduce trap density to test whether trap-assisted leakage dominates.", "Leakage should drop while BV/Ron remain inside constraints.", "Stop if leakage slope or subthreshold shape is unchanged.", 0.72)
        return None
    if var_class == "drift_doping":
        if tags["ron"]:
            risk = "high" if tags["tradeoff_guard"] or tags["bv"] or tags["field"] else "medium"
            return (1.15, risk, "Raise drift doping as a Ron-improvement probe with explicit BV/field tradeoff review.", "Ron should improve; BV and peak field must be checked before continuing.", "Stop if BV drops, field peak rises beyond constraint, or leakage worsens.", 0.65)
        if tags["bv"] or tags["field"] or tags["leakage"]:
            return (0.8, "medium", "Lower drift doping to reduce peak field and improve reverse-bias margin.", "BV/field should improve but Ron may worsen, so compare Pareto movement.", "Stop if Ron penalty violates constraints or BV/field does not improve.", 0.7)
        return None
    if var_class == "field_plate":
        if tags["field"] or tags["bv"]:
            return (1.15, "high", "Extend field plate length as a geometry-level field redistribution probe.", "Peak field should shift and reduce near the critical junction edge.", "Stop if field peak moves into oxide/corner or Ron/capacitance tradeoff becomes unacceptable.", 0.62)
        return None
    if var_class == "guard_ring":
        if tags["bv"] or tags["field"]:
            return (1.1, "high", "Adjust guard ring geometry as a termination/BV field-shaping probe.", "Breakdown bracket should move upward and field peak should spread across termination.", "Stop if termination peak worsens or active area penalty is too large.", 0.58)
        return None
    if var_class == "oxide_thickness":
        if tags["field"] or tags["leakage"] or tags["bv"]:
            return (1.1, "high", "Increase oxide thickness as a high-risk field/leakage sensitivity probe.", "Oxide/interface field should reduce if oxide coupling is the driver.", "Stop if threshold/capacitance or process constraints are violated.", 0.55)
        return None
    if var_class == "implant_dose":
        if tags["bv"] or tags["leakage"] or tags["ron"]:
            factor = 0.9 if tags["bv"] or tags["leakage"] else 1.1
            return (factor, "high", "Perturb implant dose as a process-level tradeoff probe.", "Curve movement should identify whether dose controls the active failure mode.", "Stop unless the next curve gives a Pareto improvement under constraints.", 0.52)
        return None
    if var_class == "junction_depth":
        if tags["bv"] or tags["field"]:
            return (1.05, "high", "Increase junction depth as a process geometry field-relief probe.", "Peak field should reduce or move away from the shallow junction edge.", "Stop if leakage/Ron or process limits degrade.", 0.5)
        return None
    if var_class == "trench_corner_radius":
        if tags["field"] or tags["bv"]:
            return (1.2, "high", "Increase trench corner radius to test whether corner crowding drives peak field.", "Field peak at the trench corner should reduce in overlay comparison.", "Stop if the peak is elsewhere or geometry constraints dominate.", 0.5)
        return None
    return None


def variable_candidates(ctx: DeckContext, tags: dict[str, bool]) -> list[SentaurusPatchCandidate]:
    candidates: list[SentaurusPatchCandidate] = []
    for assignment in ctx.ir.set_variables:
        classes = variable_classes(assignment.key)
        if not classes:
            continue
        for var_class in sorted(classes):
            direction = variable_mutation_direction(var_class, tags)
            if direction is None:
                continue
            factor, risk, hypothesis, expected, stop, score = direction
            next_value = scaled_numeric_text(assignment.value, factor)
            if next_value is None:
                continue
            requires_confirmation = risk == "high"
            candidate = SentaurusPatchCandidate(
                candidate_id=f"{ctx.rel_file}:{var_class}:{assignment.key}",
                title=f"{assignment.key} {factor:.3g}x {var_class} probe",
                score=score,
                hypothesis=hypothesis,
                risk_level=risk,
                requires_user_confirmation=requires_confirmation,
                target_file=ctx.rel_file,
                patches=[
                    set_variable_patch(
                        ctx,
                        assignment.key,
                        next_value,
                        reason=f"Natural-language goal matched {var_class}; scale {assignment.key} by {factor:.3g}.",
                    )
                ],
                expected_observation=expected,
                stop_condition=stop,
                fallback_alternatives=["compare baseline/mutation overlay", "run a smaller factor if direction helps", "switch to model or extraction debugging if curve shape is unchanged"],
                rationale=f"Variable {assignment.key} is present in the user deck and matches the extensible Sentaurus mutation vocabulary class `{var_class}`.",
                evidence_used=[deck_evidence(ctx), {"kind": "mutation_vocabulary", "class": var_class, "variable": assignment.key, "factor": factor, "goal_tags": tags}],
                confidence=score,
            )
            candidates.append(candidate)
    return candidates


def validate_candidates(candidates: list[SentaurusPatchCandidate], contexts: list[DeckContext]) -> None:
    text_by_file = {ctx.rel_file: ctx.text for ctx in contexts}
    for candidate in candidates:
        working_text = dict(text_by_file)
        records: list[dict[str, Any]] = []
        for patch in candidate.patches:
            file_name = str(patch.get("file") or "")
            text = working_text.get(file_name)
            if text is None:
                records.append({"file": file_name, "operation": patch.get("operation"), "applied": False, "verified": False, "error": "deck text not available"})
                continue
            try:
                after, record, _ = apply_sentaurus_semantic_patch_text(text, patch, source_path=file_name)
            except Exception as exc:
                records.append({"file": file_name, "operation": patch.get("operation"), "applied": False, "verified": False, "error": str(exc)})
                continue
            record["file"] = file_name
            records.append(record)
            if record.get("applied"):
                working_text[file_name] = after
        candidate.validation_records = records
        candidate.verified_patch_count = sum(1 for record in records if record.get("verified"))
        if candidate.verified_patch_count < len(candidate.patches):
            candidate.requires_user_confirmation = True
            candidate.risk_level = "high" if candidate.risk_level != "high" else candidate.risk_level
            candidate.confidence = min(candidate.confidence, 0.35)


def dedupe_candidates(candidates: list[SentaurusPatchCandidate]) -> list[SentaurusPatchCandidate]:
    seen: set[str] = set()
    output: list[SentaurusPatchCandidate] = []
    for candidate in candidates:
        signature = json.dumps(candidate.patches, sort_keys=True, ensure_ascii=True)
        if signature in seen:
            continue
        seen.add(signature)
        output.append(candidate)
    return output


def select_candidate(candidates: list[SentaurusPatchCandidate], *, allow_high_risk: bool) -> SentaurusPatchCandidate | None:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.patches
        and candidate.verified_patch_count == len(candidate.patches)
        and (allow_high_risk or (candidate.risk_level != "high" and not candidate.requires_user_confirmation))
    ]
    if not eligible:
        return None
    return sorted(eligible, key=lambda candidate: candidate.score, reverse=True)[0]


def build_candidates(goal_text: str, state: dict[str, Any] | None, contexts: list[DeckContext]) -> list[SentaurusPatchCandidate]:
    tags = goal_tags(goal_text, state)
    candidates: list[SentaurusPatchCandidate] = []
    for ctx in contexts:
        candidate = convergence_candidate(ctx, tags, state)
        if candidate:
            candidates.append(candidate)
        candidate = bias_candidate(ctx, tags, goal_text)
        if candidate:
            candidates.append(candidate)
        candidates.extend(variable_candidates(ctx, tags))
    candidates = dedupe_candidates(candidates)
    validate_candidates(candidates, contexts)
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def plan_sentaurus_patches(request: SentaurusPatchPlannerRequest) -> SentaurusPatchPlan:
    source_state_path = str(request.source_state_path.expanduser().resolve()) if request.source_state_path else None
    state = load_final_state(source_state_path) if source_state_path else None
    try:
        project_root, contexts, warnings = resolve_decks(request, state)
        if not contexts:
            plan = SentaurusPatchPlan(
                status="no_deck_context",
                goal_text=request.goal_text,
                source_state_path=source_state_path,
                project_path=str(project_root) if project_root else None,
                evidence_summary={"warnings": warnings, "state": state_evidence(state)},
                failure_reason="No parseable Sentaurus deck files were available.",
            )
        else:
            candidates = build_candidates(request.goal_text, state, contexts)[: request.max_candidates]
            selected = select_candidate(candidates, allow_high_risk=request.allow_high_risk)
            if selected:
                status = "completed"
            elif candidates:
                status = "blocked_for_user_confirmation"
            else:
                status = "no_actionable_candidates"
            plan = SentaurusPatchPlan(
                status=status,
                goal_text=request.goal_text,
                source_state_path=source_state_path,
                project_path=str(project_root) if project_root else None,
                deck_files_inspected=[ctx.rel_file for ctx in contexts],
                candidates=candidates,
                selected_candidate=selected,
                evidence_summary={
                    "goal_tags": goal_tags(request.goal_text, state),
                    "state": state_evidence(state),
                    "deck_count": len(contexts),
                    "warnings": warnings,
                },
                final_summary={
                    "candidate_count": len(candidates),
                    "selected_candidate_id": selected.candidate_id if selected else None,
                    "requires_user_confirmation_count": sum(1 for candidate in candidates if candidate.requires_user_confirmation),
                    "verified_candidate_count": sum(1 for candidate in candidates if candidate.verified_patch_count == len(candidate.patches)),
                },
            )
    except Exception as exc:
        plan = SentaurusPatchPlan(
            status="failed",
            goal_text=request.goal_text,
            source_state_path=source_state_path,
            failure_reason=str(exc),
        )
    if request.output_path is not None:
        output_path = request.output_path.expanduser().resolve()
        plan.output_path = str(output_path)
        write_json(output_path, plan.model_dump(mode="json"))
    return plan

