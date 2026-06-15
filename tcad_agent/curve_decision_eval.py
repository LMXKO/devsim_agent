from __future__ import annotations

import argparse
import csv
import json
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from tcad_agent.agent_curve_guidance import AgentCurveGuidance, build_agent_curve_guidance
from tcad_agent.curve_diagnostics import compare_state_mutation_effect
from tcad_agent.llm import LLMClient, LLMConfig
from tcad_agent.task_planner import parse_json_object
from tcad_agent.task_spec import PROJECT_ROOT


ALLOWED_ACTIONS = [
    "refine_effective_mutation",
    "switch_mutation_target",
    "pareto_review_before_next_patch",
    "repair_curve_shape",
    "collect_more_curve_evidence",
]

ALLOWED_TARGETS = [
    "field_plate",
    "guard_ring",
    "drift_doping",
    "implant_dose",
    "junction_depth",
    "region_specific_lifetime",
    "trap_density",
    "oxide_thickness",
    "trench_corner_radius",
    "bias_or_mesh_refinement",
]

ACTION_ALIASES = {
    "continue_same_target": "refine_effective_mutation",
    "continue_refine": "refine_effective_mutation",
    "refine": "refine_effective_mutation",
    "switch_target": "switch_mutation_target",
    "pareto_review": "pareto_review_before_next_patch",
    "review_constraints": "pareto_review_before_next_patch",
    "blocked_for_pareto_review": "pareto_review_before_next_patch",
    "repair_curve": "repair_curve_shape",
    "collect_more_evidence": "collect_more_curve_evidence",
}

TARGET_ALIASES = {
    "lifetime": "region_specific_lifetime",
    "local_lifetime": "region_specific_lifetime",
    "region_lifetime": "region_specific_lifetime",
    "mesh": "bias_or_mesh_refinement",
    "bias": "bias_or_mesh_refinement",
    "bias_refinement": "bias_or_mesh_refinement",
    "mesh_refinement": "bias_or_mesh_refinement",
    "fieldplate": "field_plate",
    "guardring": "guard_ring",
    "drift": "drift_doping",
    "doping": "drift_doping",
}


class ChatClient(Protocol):
    config: LLMConfig

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        ...


class CurveDecisionEvalStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class CurveDecisionEvalCase(BaseModel):
    case_id: str
    title: str
    goal_text: str
    baseline_metrics: dict[str, Any]
    mutation_metrics: dict[str, Any]
    baseline_rows: list[dict[str, Any]]
    mutation_rows: list[dict[str, Any]]
    deck_patch: dict[str, Any]
    issue_codes: list[str] = Field(default_factory=list)
    expected_action: str
    expected_targets: list[str] = Field(default_factory=list)


class CurveDecisionEvalRequest(BaseModel):
    eval_id: str = "curve_decision_eval"
    eval_root: Path = PROJECT_ROOT / "runs" / "curve_decision_eval"
    use_llm: bool = False
    allow_llm_fallback: bool = True
    cases: list[CurveDecisionEvalCase] = Field(default_factory=list)


class CurveDecisionCaseResult(BaseModel):
    case_id: str
    title: str
    status: CurveDecisionEvalStatus
    expected_action: str
    expected_targets: list[str] = Field(default_factory=list)
    recommended_action: str | None = None
    recommended_target: str | None = None
    decision_source: str
    fallback_used: bool = False
    model: str | None = None
    raw_response: str | None = None
    parsed_response: dict[str, Any] | None = None
    rationale: str | None = None
    evidence_used: list[str] = Field(default_factory=list)
    failure_reason: str | None = None
    baseline_state_path: str
    mutation_state_path: str
    mutation_effect_path: str
    guidance_path: str
    overlay_svg_path: str | None = None
    mutation_effect_analysis: dict[str, Any] = Field(default_factory=dict)
    oracle_guidance: dict[str, Any] = Field(default_factory=dict)


class CurveDecisionEvalResult(BaseModel):
    schema_version: str = "actsoft.tcad.curve_decision_eval.v1"
    status: CurveDecisionEvalStatus
    eval_id: str
    eval_dir: str
    use_llm: bool
    allow_llm_fallback: bool
    case_count: int
    passed_count: int
    failed_count: int
    llm_decision_count: int = 0
    fallback_count: int = 0
    raw_response_count: int = 0
    models: list[str] = Field(default_factory=list)
    cases: list[CurveDecisionCaseResult] = Field(default_factory=list)
    result_path: str | None = None
    failure_reason: str | None = None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def normalize_action(value: Any) -> str | None:
    label = normalized_label(value)
    label = ACTION_ALIASES.get(label, label)
    return label if label in ALLOWED_ACTIONS else None


def normalize_target(value: Any) -> str | None:
    label = normalized_label(value)
    label = TARGET_ALIASES.get(label, label)
    return label if label in ALLOWED_TARGETS else None


def csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(key)
    return ordered


def write_curve_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = csv_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_case_state(
    state_path: Path,
    *,
    case: CurveDecisionEvalCase,
    variant: str,
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
) -> Path:
    curve_path = state_path.parent / "curve.csv"
    write_curve_csv(curve_path, rows)
    payload = {
        "tool_name": "curve_decision_eval_fixture",
        "status": "completed",
        "run_id": f"{case.case_id}_{variant}",
        "request": {
            "goal_text": case.goal_text,
            "curve_decision_case_id": case.case_id,
            "variant": variant,
        },
        "run_dir": str(state_path.parent),
        "final_summary": {"artifacts": {"csv": str(curve_path)}, "metrics": metrics},
        "quality_report": {"status": "passed", "issues": [], "metrics": metrics},
    }
    write_json(state_path, payload)
    return state_path


def default_curve_decision_cases() -> list[CurveDecisionEvalCase]:
    common_threshold = {"breakdown_current_threshold_a": 1.0e-6}
    return [
        CurveDecisionEvalCase(
            case_id="lifetime_leakage_improved",
            title="Localized lifetime cut reduces leakage without blocking tradeoffs",
            goal_text="降低 power diode 反偏漏电，同时保持 BV/Ron/field peak 不变坏。",
            baseline_metrics={
                **common_threshold,
                "leakage_current_a": 2.0e-8,
                "breakdown_voltage_v": 78.0,
                "specific_on_resistance_ohm_cm2": 6.0e-3,
                "max_electric_field_v_per_cm": 2.80e5,
            },
            mutation_metrics={
                **common_threshold,
                "leakage_current_a": 4.0e-9,
                "breakdown_voltage_v": 80.0,
                "specific_on_resistance_ohm_cm2": 6.1e-3,
                "max_electric_field_v_per_cm": 2.65e5,
            },
            baseline_rows=[
                {"reverse_voltage_v": 0, "current_a": -2.0e-12, "electric_field_v_per_cm": 1.0e4},
                {"reverse_voltage_v": -20, "current_a": -4.0e-10, "electric_field_v_per_cm": 9.0e4},
                {"reverse_voltage_v": -40, "current_a": -2.0e-8, "electric_field_v_per_cm": 1.7e5},
                {"reverse_voltage_v": -60, "current_a": -2.0e-7, "electric_field_v_per_cm": 2.5e5},
                {"reverse_voltage_v": -80, "current_a": -1.5e-5, "electric_field_v_per_cm": 2.8e5},
            ],
            mutation_rows=[
                {"reverse_voltage_v": 0, "current_a": -4.0e-13, "electric_field_v_per_cm": 0.9e4},
                {"reverse_voltage_v": -20, "current_a": -8.0e-11, "electric_field_v_per_cm": 8.0e4},
                {"reverse_voltage_v": -40, "current_a": -4.0e-9, "electric_field_v_per_cm": 1.5e5},
                {"reverse_voltage_v": -60, "current_a": -6.0e-8, "electric_field_v_per_cm": 2.3e5},
                {"reverse_voltage_v": -80, "current_a": -8.0e-6, "electric_field_v_per_cm": 2.65e5},
            ],
            deck_patch={
                "target": "region_specific_lifetime",
                "request_path": "lifetime.drift_region_ns",
                "baseline_value": 50.0,
                "value": 30.0,
            },
            issue_codes=["leakage_high"],
            expected_action="refine_effective_mutation",
            expected_targets=["region_specific_lifetime"],
        ),
        CurveDecisionEvalCase(
            case_id="field_plate_ron_tradeoff",
            title="Field plate lowers field but creates a Ron tradeoff",
            goal_text="降低 power MOSFET field peak，但不能让 Ron 明显变坏。",
            baseline_metrics={
                **common_threshold,
                "leakage_current_a": 1.0e-8,
                "breakdown_voltage_v": 82.0,
                "specific_on_resistance_ohm_cm2": 4.5e-3,
                "max_electric_field_v_per_cm": 3.10e5,
            },
            mutation_metrics={
                **common_threshold,
                "leakage_current_a": 7.0e-9,
                "breakdown_voltage_v": 80.0,
                "specific_on_resistance_ohm_cm2": 6.6e-3,
                "max_electric_field_v_per_cm": 2.50e5,
            },
            baseline_rows=[
                {"reverse_voltage_v": 0, "current_a": -1.0e-12, "electric_field_v_per_cm": 1.1e4},
                {"reverse_voltage_v": -25, "current_a": -2.0e-10, "electric_field_v_per_cm": 1.2e5},
                {"reverse_voltage_v": -50, "current_a": -1.0e-8, "electric_field_v_per_cm": 2.2e5},
                {"reverse_voltage_v": -75, "current_a": -4.0e-7, "electric_field_v_per_cm": 3.1e5},
                {"reverse_voltage_v": -85, "current_a": -1.2e-5, "electric_field_v_per_cm": 3.0e5},
            ],
            mutation_rows=[
                {"reverse_voltage_v": 0, "current_a": -8.0e-13, "electric_field_v_per_cm": 1.0e4},
                {"reverse_voltage_v": -25, "current_a": -1.5e-10, "electric_field_v_per_cm": 9.0e4},
                {"reverse_voltage_v": -50, "current_a": -7.0e-9, "electric_field_v_per_cm": 1.7e5},
                {"reverse_voltage_v": -75, "current_a": -3.0e-7, "electric_field_v_per_cm": 2.5e5},
                {"reverse_voltage_v": -85, "current_a": -1.0e-5, "electric_field_v_per_cm": 2.45e5},
            ],
            deck_patch={
                "target": "field_plate",
                "request_path": "geometry.field_plate_length_um",
                "baseline_value": 1.5,
                "value": 2.8,
            },
            issue_codes=["field_peak_high"],
            expected_action="pareto_review_before_next_patch",
            expected_targets=["guard_ring", "field_plate", "drift_doping"],
        ),
        CurveDecisionEvalCase(
            case_id="drift_doping_ron_not_improved",
            title="Drift doping probe fails to improve Ron",
            goal_text="优化 Ron/BV 折中，先试 drift doping，如果曲线无效就换方向。",
            baseline_metrics={
                **common_threshold,
                "leakage_current_a": 1.0e-8,
                "breakdown_voltage_v": 80.0,
                "specific_on_resistance_ohm_cm2": 5.0e-3,
                "max_electric_field_v_per_cm": 2.80e5,
            },
            mutation_metrics={
                **common_threshold,
                "leakage_current_a": 1.05e-8,
                "breakdown_voltage_v": 79.0,
                "specific_on_resistance_ohm_cm2": 5.8e-3,
                "max_electric_field_v_per_cm": 2.90e5,
            },
            baseline_rows=[
                {"reverse_voltage_v": 0, "current_a": -1.0e-12, "electric_field_v_per_cm": 1.0e4},
                {"reverse_voltage_v": -20, "current_a": -2.0e-10, "electric_field_v_per_cm": 8.0e4},
                {"reverse_voltage_v": -40, "current_a": -1.0e-8, "electric_field_v_per_cm": 1.8e5},
                {"reverse_voltage_v": -60, "current_a": -2.0e-7, "electric_field_v_per_cm": 2.5e5},
                {"reverse_voltage_v": -80, "current_a": -1.0e-5, "electric_field_v_per_cm": 2.8e5},
            ],
            mutation_rows=[
                {"reverse_voltage_v": 0, "current_a": -1.1e-12, "electric_field_v_per_cm": 1.0e4},
                {"reverse_voltage_v": -20, "current_a": -2.2e-10, "electric_field_v_per_cm": 8.3e4},
                {"reverse_voltage_v": -40, "current_a": -1.05e-8, "electric_field_v_per_cm": 1.85e5},
                {"reverse_voltage_v": -60, "current_a": -2.1e-7, "electric_field_v_per_cm": 2.55e5},
                {"reverse_voltage_v": -80, "current_a": -1.05e-5, "electric_field_v_per_cm": 2.9e5},
            ],
            deck_patch={
                "target": "drift_doping",
                "request_path": "power_mos_drift_region_doping_cm3",
                "baseline_value": 1.0e16,
                "value": 1.4e16,
            },
            issue_codes=["ron_high"],
            expected_action="switch_mutation_target",
            expected_targets=["implant_dose", "junction_depth", "drift_doping"],
        ),
        CurveDecisionEvalCase(
            case_id="nonmonotonic_curve_requires_repair",
            title="Curve shape breaks before physical interpretation",
            goal_text="曲线有反常拐点时，先修 bias/mesh，再继续 lifetime 或 field plate patch。",
            baseline_metrics={
                **common_threshold,
                "leakage_current_a": 2.0e-8,
                "breakdown_voltage_v": 78.0,
                "specific_on_resistance_ohm_cm2": 6.0e-3,
                "max_electric_field_v_per_cm": 2.80e5,
            },
            mutation_metrics={
                **common_threshold,
                "leakage_current_a": 5.0e-9,
                "breakdown_voltage_v": 79.0,
                "specific_on_resistance_ohm_cm2": 6.05e-3,
                "max_electric_field_v_per_cm": 2.70e5,
            },
            baseline_rows=[
                {"reverse_voltage_v": 0, "current_a": -2.0e-12, "electric_field_v_per_cm": 1.0e4},
                {"reverse_voltage_v": -20, "current_a": -3.0e-10, "electric_field_v_per_cm": 8.5e4},
                {"reverse_voltage_v": -40, "current_a": -2.0e-8, "electric_field_v_per_cm": 1.7e5},
                {"reverse_voltage_v": -60, "current_a": -2.0e-7, "electric_field_v_per_cm": 2.5e5},
                {"reverse_voltage_v": -80, "current_a": -1.0e-5, "electric_field_v_per_cm": 2.8e5},
            ],
            mutation_rows=[
                {"reverse_voltage_v": 0, "current_a": -4.0e-13, "electric_field_v_per_cm": 9.0e3},
                {"reverse_voltage_v": -20, "current_a": -9.0e-10, "electric_field_v_per_cm": 8.0e4},
                {"reverse_voltage_v": -40, "current_a": -2.0e-10, "electric_field_v_per_cm": 1.6e5},
                {"reverse_voltage_v": -60, "current_a": -2.0e-8, "electric_field_v_per_cm": 2.3e5},
                {"reverse_voltage_v": -80, "current_a": -1.0e-6, "electric_field_v_per_cm": 2.7e5},
            ],
            deck_patch={
                "target": "region_specific_lifetime",
                "request_path": "lifetime.drift_region_ns",
                "baseline_value": 50.0,
                "value": 35.0,
            },
            issue_codes=["leakage_high"],
            expected_action="repair_curve_shape",
            expected_targets=["bias_or_mesh_refinement"],
        ),
    ]


def oracle_decision(guidance: AgentCurveGuidance) -> dict[str, Any]:
    return {
        "recommended_action": normalize_action(guidance.recommended_action) or guidance.recommended_action,
        "recommended_target": normalize_target(guidance.recommended_target) or guidance.recommended_target,
        "rationale": guidance.reason,
        "evidence_used": guidance.decision_basis,
        "next_patch_hint": guidance.next_patch_hint,
    }


def build_llm_messages(
    *,
    case: CurveDecisionEvalCase,
    mutation_effect: dict[str, Any],
) -> tuple[str, str]:
    system = (
        "You are a TCAD curve-review agent. Compare baseline vs mutation evidence and choose the next "
        "deck-patch direction from the allowed labels only. Do not invent labels or patches. Return JSON only."
    )
    payload = {
        "task": "choose next mutation direction after seeing baseline-vs-mutation curves",
        "response_schema": {
            "recommended_action": ALLOWED_ACTIONS,
            "recommended_target": ALLOWED_TARGETS,
            "rationale": "short explanation tied to curve shape, deltas, tradeoffs, and overlay",
            "evidence_used": ["metric_deltas", "curve_shape", "overlay", "tradeoff_violations"],
            "next_patch_hint": {"direction": "smaller_step_same_direction | probe_alternate | add_points | pareto_review"},
        },
        "action_meanings": {
            "refine_effective_mutation": "primary metric improved and tradeoffs are acceptable; continue with a smaller patch on the same useful direction",
            "switch_mutation_target": "the tested mutation did not improve its primary metric enough; probe another physical target",
            "pareto_review_before_next_patch": "primary movement exists but BV/Ron/field/leakage tradeoffs require constraint/Pareto review first",
            "repair_curve_shape": "curve is too sparse or nonmonotonic; fix bias stepping, mesh, or numerical setup before physical edits",
            "collect_more_curve_evidence": "there is not enough comparable curve evidence to choose a physical patch",
        },
        "case": {
            "case_id": case.case_id,
            "title": case.title,
            "goal_text": case.goal_text,
            "tested_deck_patch": case.deck_patch,
            "issue_codes": case.issue_codes,
        },
        "curve_evidence": {
            "primary_metric": mutation_effect.get("primary_metric"),
            "primary_improved": mutation_effect.get("primary_improved"),
            "worth_continuing": mutation_effect.get("worth_continuing"),
            "metric_deltas": mutation_effect.get("metric_deltas"),
            "improved_metrics": mutation_effect.get("improved_metrics"),
            "regressed_metrics": mutation_effect.get("regressed_metrics"),
            "tradeoff_violations": mutation_effect.get("tradeoff_violations"),
            "baseline_shape": mutation_effect.get("baseline_shape"),
            "mutation_shape": mutation_effect.get("mutation_shape"),
            "curve_overlay": mutation_effect.get("curve_overlay"),
        },
    }
    return system, json.dumps(payload, indent=2, ensure_ascii=False)


def validate_decision(
    *,
    decision: dict[str, Any],
    case: CurveDecisionEvalCase,
) -> tuple[bool, str | None, str | None, str | None]:
    action = normalize_action(decision.get("recommended_action") or decision.get("action"))
    target = normalize_target(decision.get("recommended_target") or decision.get("target"))
    if action is None:
        return False, None, target, "decision did not choose an allowed action"
    if target is None:
        return False, action, None, "decision did not choose an allowed target"
    expected_action = normalize_action(case.expected_action)
    if action != expected_action:
        return False, action, target, f"expected action {case.expected_action}, got {action}"
    expected_targets = [normalize_target(item) for item in case.expected_targets]
    expected_targets = [item for item in expected_targets if item]
    if expected_targets and target not in expected_targets:
        return False, action, target, f"expected target in {expected_targets}, got {target}"
    evidence = decision.get("evidence_used")
    if not isinstance(evidence, list) or not evidence:
        return False, action, target, "decision did not cite evidence_used"
    rationale = str(decision.get("rationale") or decision.get("reason") or "").strip()
    if not rationale:
        return False, action, target, "decision did not include a rationale"
    return True, action, target, None


def model_name(llm_client: ChatClient | None) -> str | None:
    config = getattr(llm_client, "config", None)
    return getattr(config, "model", None)


def choose_case_decision(
    *,
    case: CurveDecisionEvalCase,
    mutation_effect: dict[str, Any],
    guidance: AgentCurveGuidance,
    request: CurveDecisionEvalRequest,
    llm_client: ChatClient | None,
) -> tuple[dict[str, Any], str, bool, str | None, dict[str, Any] | None, str | None, str | None]:
    deterministic = oracle_decision(guidance)
    if not request.use_llm:
        return deterministic, "deterministic_guidance", False, None, None, None, None

    chat_client = llm_client or LLMClient()
    system, user = build_llm_messages(case=case, mutation_effect=mutation_effect)
    try:
        raw = chat_client.chat(system=system, user=user, temperature=0.1)
    except Exception as exc:
        if request.allow_llm_fallback:
            return deterministic, "deterministic_guidance", True, None, None, None, str(exc)
        return {}, "llm", False, None, None, None, str(exc)

    parsed = parse_json_object(raw)
    if not parsed:
        if request.allow_llm_fallback:
            return deterministic, "deterministic_guidance", True, raw, None, model_name(chat_client), "LLM response did not contain a JSON object"
        return {}, "llm", False, raw, None, model_name(chat_client), "LLM response did not contain a JSON object"
    return parsed, "llm", False, raw, parsed, model_name(chat_client), None


def run_curve_decision_eval(
    request: CurveDecisionEvalRequest | None = None,
    *,
    llm_client: ChatClient | None = None,
) -> CurveDecisionEvalResult:
    actual = request or CurveDecisionEvalRequest()
    cases = actual.cases or default_curve_decision_cases()
    eval_dir = actual.eval_root.expanduser().resolve() / actual.eval_id
    eval_dir.mkdir(parents=True, exist_ok=True)
    results: list[CurveDecisionCaseResult] = []
    for case in cases:
        case_dir = eval_dir / case.case_id
        baseline_state_path = write_case_state(
            case_dir / "baseline" / "state.json",
            case=case,
            variant="baseline",
            metrics=case.baseline_metrics,
            rows=case.baseline_rows,
        )
        mutation_state_path = write_case_state(
            case_dir / "mutation" / "state.json",
            case=case,
            variant="mutation",
            metrics=case.mutation_metrics,
            rows=case.mutation_rows,
        )
        effect = compare_state_mutation_effect(
            baseline_state_path,
            mutation_state_path,
            deck_patch=case.deck_patch,
            issue_codes=case.issue_codes,
            overlay_output_path=case_dir / "baseline_mutation_overlay.svg",
        ).model_dump(mode="json")
        mutation_state = read_json(mutation_state_path)
        mutation_state["mutation_effect_analysis"] = effect
        write_json(mutation_state_path, mutation_state)
        effect_path = case_dir / "mutation_effect_analysis.json"
        write_json(effect_path, effect)
        guidance = build_agent_curve_guidance(goal_text=case.goal_text, source_state_path=str(mutation_state_path))
        guidance_path = case_dir / "agent_curve_guidance.json"
        guidance_data = guidance.model_dump(mode="json")
        write_json(guidance_path, guidance_data)

        decision, source, fallback_used, raw, parsed, model, call_failure = choose_case_decision(
            case=case,
            mutation_effect=effect,
            guidance=guidance,
            request=actual,
            llm_client=llm_client,
        )
        valid, action, target, validation_failure = validate_decision(decision=decision, case=case) if decision else (
            False,
            None,
            None,
            "no decision was produced",
        )
        failure_reason = validation_failure or (call_failure if not fallback_used else None)
        status = CurveDecisionEvalStatus.COMPLETED if valid and not failure_reason else CurveDecisionEvalStatus.FAILED
        evidence = decision.get("evidence_used") if isinstance(decision.get("evidence_used"), list) else []
        result = CurveDecisionCaseResult(
            case_id=case.case_id,
            title=case.title,
            status=status,
            expected_action=case.expected_action,
            expected_targets=case.expected_targets,
            recommended_action=action,
            recommended_target=target,
            decision_source=source,
            fallback_used=fallback_used,
            model=model,
            raw_response=raw,
            parsed_response=parsed,
            rationale=str(decision.get("rationale") or decision.get("reason") or "") if decision else None,
            evidence_used=[str(item) for item in evidence],
            failure_reason=failure_reason,
            baseline_state_path=str(baseline_state_path),
            mutation_state_path=str(mutation_state_path),
            mutation_effect_path=str(effect_path),
            guidance_path=str(guidance_path),
            overlay_svg_path=effect.get("overlay_svg_path"),
            mutation_effect_analysis=effect,
            oracle_guidance=guidance_data,
        )
        write_json(case_dir / "curve_decision_case_result.json", result.model_dump(mode="json"))
        results.append(result)

    passed = sum(1 for item in results if item.status == CurveDecisionEvalStatus.COMPLETED)
    failed = len(results) - passed
    models = sorted({item.model for item in results if item.model})
    result = CurveDecisionEvalResult(
        status=CurveDecisionEvalStatus.COMPLETED if failed == 0 else CurveDecisionEvalStatus.FAILED,
        eval_id=actual.eval_id,
        eval_dir=str(eval_dir),
        use_llm=actual.use_llm,
        allow_llm_fallback=actual.allow_llm_fallback,
        case_count=len(results),
        passed_count=passed,
        failed_count=failed,
        llm_decision_count=sum(1 for item in results if item.decision_source == "llm" and item.status == CurveDecisionEvalStatus.COMPLETED),
        fallback_count=sum(1 for item in results if item.fallback_used),
        raw_response_count=sum(1 for item in results if item.raw_response),
        models=models,
        cases=results,
        failure_reason=None if failed == 0 else f"{failed} curve decision eval case(s) failed",
    )
    result_path = eval_dir / "curve_decision_eval_result.json"
    result.result_path = str(result_path)
    write_json(result_path, result.model_dump(mode="json"))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate agent/LLM curve-driven next mutation decisions.")
    parser.add_argument("--eval-id", default="curve_decision_eval")
    parser.add_argument("--eval-root", type=Path, default=PROJECT_ROOT / "runs" / "curve_decision_eval")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--no-llm-fallback", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_curve_decision_eval(
        CurveDecisionEvalRequest(
            eval_id=args.eval_id,
            eval_root=args.eval_root,
            use_llm=bool(args.use_llm),
            allow_llm_fallback=not bool(args.no_llm_fallback),
        )
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.status == CurveDecisionEvalStatus.COMPLETED else 1)


if __name__ == "__main__":
    main()
