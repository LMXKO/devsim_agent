from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.curve_diagnostics import compare_state_mutation_effect
from tcad_agent.mutation_refinement import state_mutations, state_request
from tcad_agent.physical_benchmark import PhysicalBenchmarkResult, run_physical_benchmark
from tcad_agent.repair_strategy import curve_guided_mutation_value, mutation_target, next_mutation_value
from tcad_agent.reporting import final_metrics, load_final_state


class AgentExperimentCandidate(BaseModel):
    candidate_id: str
    action_kind: str
    score: float
    reason: str
    tool_name: str | None = None
    request: dict[str, Any] = Field(default_factory=dict)
    source_state_path: str | None = None
    evidence_gap: str | None = None
    expected_effect: str | None = None
    requires_user_confirmation: bool = False
    risk_notes: list[str] = Field(default_factory=list)


class AgentExperimentDesignPlan(BaseModel):
    schema_version: str = "actsoft.tcad.agent_experiment_design.v1"
    status: str
    source_state_path: str
    output_path: str | None = None
    benchmark_path: str | None = None
    evidence_gaps: list[str] = Field(default_factory=list)
    signoff_verdict: str | None = None
    curve_engineering_review: dict[str, Any] = Field(default_factory=dict)
    mutation_effect_analysis: dict[str, Any] = Field(default_factory=dict)
    candidates: list[AgentExperimentCandidate] = Field(default_factory=list)
    selected_candidate: AgentExperimentCandidate | None = None
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def benchmark_for_state(source_state_path: Path, benchmark_path: Path | None = None) -> PhysicalBenchmarkResult:
    if benchmark_path and benchmark_path.exists():
        try:
            return PhysicalBenchmarkResult.model_validate_json(benchmark_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return run_physical_benchmark(source_state_path)


def request_reference_curve_path(state: dict[str, Any], request: dict[str, Any]) -> str | None:
    deck = state.get("tcad_deck_spec") or request.get("tcad_deck_spec") or {}
    signoff = deck.get("signoff_requirements") if isinstance(deck, dict) else {}
    candidates = [
        request.get("reference_curve_path"),
        request.get("golden_curve_path"),
        request.get("measured_curve_path"),
        request.get("measured_curve_comparison"),
        signoff.get("measured_curve_path") if isinstance(signoff, dict) else None,
        signoff.get("reference_curve_path") if isinstance(signoff, dict) else None,
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return None


def infer_metric_path(tool_name: str, state: dict[str, Any]) -> str:
    metrics = final_metrics(state)
    if tool_name == "mosfet_2d_id_sweep":
        if "vth_at_threshold_current_v" in metrics:
            return "quality_report.metrics.vth_at_threshold_current_v"
        if "ion_ioff_ratio" in metrics:
            return "quality_report.metrics.ion_ioff_ratio"
    if tool_name == "extended_device_sweep":
        if "max_electric_field_v_per_cm" in metrics:
            return "quality_report.metrics.max_electric_field_v_per_cm"
        if "specific_on_resistance_ohm_cm2" in metrics:
            return "quality_report.metrics.specific_on_resistance_ohm_cm2"
        if "current_gain_beta" in metrics:
            return "quality_report.metrics.current_gain_beta"
    if "leakage_current_a" in metrics:
        return "quality_report.metrics.leakage_current_a"
    return "quality_report.metrics.objective_value"


def infer_convergence_axis_and_values(state: dict[str, Any], request: dict[str, Any]) -> tuple[str, list[Any]]:
    tool_name = str(state.get("tool_name") or "")
    metrics = final_metrics(state)
    if tool_name == "mosfet_2d_id_sweep":
        current = request.get("x_divisions") or metrics.get("x_divisions") or 12
        base = max(int(float(current)), 4)
        return "x_divisions", sorted({max(4, base - 4), base, base + 4})
    if tool_name == "extended_device_sweep":
        device_type = str(metrics.get("device_type") or request.get("device_type") or "")
        if device_type == "power_mosfet_bv_ron":
            spacing = float(request.get("power_mos_junction_mesh_spacing_um") or metrics.get("junction_mesh_spacing_um") or 0.01)
            return "power_mos_junction_mesh_spacing_um", [spacing * 2.0, spacing, max(spacing / 2.0, 1.0e-5)]
        if device_type == "bjt_gummel_output":
            return "fidelity", [request.get("fidelity") or "physics_1d", "physics_1d"]
    if "step" in request:
        step = float(request.get("step") or 0.1)
        return "step", [step, max(step / 2.0, 1.0e-6)]
    return "mesh_refinement_level", [1, 2, 3]


def convergence_candidate(source_state_path: Path, state: dict[str, Any], request: dict[str, Any], score: float) -> AgentExperimentCandidate | None:
    tool_name = str(state.get("tool_name") or "")
    if tool_name in {"tool_convergence", "mesh_convergence", "golden_curve_comparison", ""}:
        return None
    axis_path, values = infer_convergence_axis_and_values(state, request)
    if len(values) < 2:
        return None
    return AgentExperimentCandidate(
        candidate_id="collect_convergence_evidence",
        action_kind="run_tool",
        tool_name="tool_convergence",
        score=score,
        source_state_path=str(source_state_path),
        evidence_gap="convergence_evidence",
        reason="Benchmark/signoff evidence says convergence is missing; run a tool-level mesh/model/bias convergence check.",
        expected_effect="Turns a single accepted curve into explicit convergence evidence before signoff.",
        request={
            "convergence_id": f"{source_state_path.parent.name}_agent_convergence",
            "tool_name": tool_name,
            "base_request": request,
            "axis_path": axis_path,
            "values": values,
            "metric_path": infer_metric_path(tool_name, state),
            "relative_tolerance": 0.1,
            "execute": True,
            "overwrite": True,
            "convergence_root": str(source_state_path.parent / "agent_tool_convergence"),
        },
    )


def golden_candidate(source_state_path: Path, state: dict[str, Any], request: dict[str, Any], score: float) -> AgentExperimentCandidate | None:
    reference = request_reference_curve_path(state, request)
    if not reference:
        return None
    return AgentExperimentCandidate(
        candidate_id="collect_golden_measured_correlation",
        action_kind="run_tool",
        tool_name="golden_curve_comparison",
        score=score,
        source_state_path=str(source_state_path),
        evidence_gap="golden_or_measured_comparison",
        reason="A reference/measured curve is available; correlate the current TCAD curve before stronger engineering signoff.",
        expected_effect="Quantifies log-domain curve error and blocks/accepts calibration evidence.",
        request={
            "comparison_id": f"{source_state_path.parent.name}_agent_reference",
            "source_state_path": str(source_state_path),
            "reference_curve_path": reference,
            "run_root": str(source_state_path.parent / "agent_golden_curve_comparison"),
        },
    )


def mutation_candidates(source_state_path: Path, state: dict[str, Any], request: dict[str, Any]) -> list[AgentExperimentCandidate]:
    mutations = state_mutations(state, request)
    if not mutations:
        return []
    analysis = state.get("mutation_effect_analysis") if isinstance(state.get("mutation_effect_analysis"), dict) else {}
    output: list[AgentExperimentCandidate] = []
    for index, mutation in enumerate(mutations[:4], start=1):
        path = str(mutation.get("request_path") or "")
        if not path:
            continue
        value = curve_guided_mutation_value(request, mutation, analysis) if analysis else next_mutation_value(request, mutation)
        if value is None:
            continue
        target = mutation_target(mutation)
        next_request = {
            **request,
            path: value,
            "active_deck_mutation": mutation,
            "deck_repair_hint": f"agent experiment design candidate for {target or path}",
            "repair_source_state_path": str(source_state_path),
            "run_id": f"{str(state.get('run_id') or source_state_path.parent.name)}_agent_candidate_{index:02d}",
        }
        output.append(
            AgentExperimentCandidate(
                candidate_id=f"mutation_{target or path}_{index}",
                action_kind="run_tool",
                tool_name=str(state.get("tool_name") or ""),
                score=0.58 - index * 0.02,
                source_state_path=str(source_state_path),
                evidence_gap="curve_guided_mutation_probe",
                reason=f"Probe {target or path} as an explicit candidate instead of relying on a single repair rule.",
                expected_effect=str(mutation.get("expected_effect") or "Compare curve/metric movement against baseline."),
                request=next_request,
                requires_user_confirmation=bool(mutation.get("requires_user_confirmation")),
                risk_notes=["geometry/process/model mutation" if mutation.get("requires_user_confirmation") else "low-risk request mutation"],
            )
        )
    return output


def repair_candidate(source_state_path: Path, state: dict[str, Any], benchmark: PhysicalBenchmarkResult) -> AgentExperimentCandidate | None:
    quality = state.get("quality_report") or {}
    warning_codes = (benchmark.summary or {}).get("warning_codes") or []
    blocking_codes = (benchmark.summary or {}).get("blocking_codes") or []
    if quality.get("status") not in {"failed", "suspicious"} and not warning_codes and not blocking_codes:
        return None
    return AgentExperimentCandidate(
        candidate_id="repair_from_benchmark_and_curve",
        action_kind="run_repair_executor",
        score=0.7 if blocking_codes else 0.62,
        source_state_path=str(source_state_path),
        evidence_gap="quality_or_benchmark_issue",
        reason="Quality/benchmark evidence is not clean; run the repair executor with curve and benchmark context.",
        expected_effect="Produces a safer patched request/deck before further signoff work.",
        request={},
        requires_user_confirmation=False,
        risk_notes=[*(blocking_codes[:4]), *(warning_codes[:4])],
    )


def sentaurus_schema_extension_candidate(source_state_path: Path, state: dict[str, Any], request: dict[str, Any]) -> AgentExperimentCandidate | None:
    if state.get("tool_name") != "sentaurus_run":
        return None
    project = state.get("project_copy_path") or state.get("project_path")
    deck_files = request.get("deck_files") or ((state.get("final_summary") or {}).get("parameters") or {}).get("deck_files")
    if not project and not deck_files:
        return None
    return AgentExperimentCandidate(
        candidate_id="extend_mutation_schema_from_sentaurus_deck",
        action_kind="plan_mutation_schema_extension",
        score=0.56,
        source_state_path=str(source_state_path),
        evidence_gap="mutation_vocabulary_gap",
        reason="Sentaurus state is available; if the existing mutation vocabulary has no verified target, generate a review-only schema extension from public evidence and local deck bindings.",
        expected_effect="Produces a schema promotion package with fixture deck and semantic patch validation, without executing a new untrusted mutation.",
        request={"project_path": project, "deck_files": deck_files if isinstance(deck_files, list) else []},
        requires_user_confirmation=False,
        risk_notes=["schema proposal only", "does not modify static vocabulary", "does not run solver"],
    )


def mutation_effect_from_state(source_state_path: Path, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    existing = state.get("mutation_effect_analysis") or state.get("sentaurus_mutation_effect_analysis")
    if isinstance(existing, dict) and existing:
        return existing
    baseline = (
        request.get("repair_baseline_state_path")
        or request.get("repair_source_state_path")
        or state.get("repair_baseline_state_path")
        or state.get("baseline_state_path")
    )
    if not baseline:
        return {}
    try:
        overlay = source_state_path.parent / "agent_experiment_baseline_mutation_overlay.svg"
        return compare_state_mutation_effect(Path(str(baseline)), source_state_path, overlay_output_path=overlay).model_dump(mode="json")
    except Exception:
        return {}


def curve_review_from_effect(effect: dict[str, Any]) -> dict[str, Any]:
    if not effect:
        return {}
    baseline_shape = effect.get("baseline_shape") or {}
    mutation_shape = effect.get("mutation_shape") or {}
    overlay = effect.get("curve_overlay") or {}
    pareto = effect.get("pareto_decision") or effect.get("pareto_summary") or {}
    return {
        "decision": effect.get("decision"),
        "primary_metric": effect.get("primary_metric"),
        "primary_improved": effect.get("primary_improved"),
        "worth_continuing": effect.get("worth_continuing"),
        "improved_metrics": effect.get("improved_metrics") or [],
        "regressed_metrics": effect.get("regressed_metrics") or [],
        "tradeoff_violations": effect.get("tradeoff_violations") or [],
        "baseline_knee_x": baseline_shape.get("knee_x"),
        "mutation_knee_x": mutation_shape.get("knee_x"),
        "baseline_field_peak_x": baseline_shape.get("field_peak_x"),
        "mutation_field_peak_x": mutation_shape.get("field_peak_x"),
        "overlay_svg": effect.get("overlay_svg_path") or overlay.get("overlay_svg"),
        "pareto": pareto,
    }


def refinement_candidate_from_effect(source_state_path: Path, effect: dict[str, Any]) -> AgentExperimentCandidate | None:
    if not effect:
        return None
    decision = str(effect.get("decision") or "")
    target = effect.get("recommended_next_target") or effect.get("mutation_target")
    if decision in {"continue_refine", "continue", "accept_and_refine"} or effect.get("worth_continuing"):
        return AgentExperimentCandidate(
            candidate_id="refine_effective_mutation_direction",
            action_kind="plan_mutation_refinement",
            score=0.94,
            source_state_path=str(source_state_path),
            evidence_gap="curve_effect_refinement",
            reason="Baseline/mutation curve comparison says this direction helped; generate a smaller follow-up patch instead of restarting from generic rules.",
            expected_effect=f"Refine {target or 'the effective mutation target'} while keeping Pareto regressions visible.",
            request={},
            risk_notes=[f"decision={decision}", f"target={target}"],
        )
    if decision in {"blocked_for_pareto_review", "reject_candidate"} or effect.get("tradeoff_violations"):
        return AgentExperimentCandidate(
            candidate_id="pareto_review_before_more_patches",
            action_kind="ask_user",
            score=0.91,
            source_state_path=str(source_state_path),
            evidence_gap="pareto_or_constraint_review",
            reason="Curve comparison found tradeoffs; pause for Pareto/constraint review before applying another geometry/process patch.",
            expected_effect="Prevents optimizing leakage/BV/Ron by silently damaging another engineering objective.",
            request={"mutation_effect_analysis": effect},
            requires_user_confirmation=True,
            risk_notes=[str(item) for item in (effect.get("regressed_metrics") or [])[:4]],
        )
    return None


def power_mosfet_signoff_candidate(source_state_path: Path, state: dict[str, Any], request: dict[str, Any]) -> AgentExperimentCandidate | None:
    metrics = final_metrics(state)
    summary = state.get("final_summary") if isinstance(state.get("final_summary"), dict) else {}
    if isinstance(summary.get("metrics"), dict):
        metrics.update(summary["metrics"])
    quality = state.get("quality_report") if isinstance(state.get("quality_report"), dict) else {}
    if isinstance(quality.get("metrics"), dict):
        metrics.update(quality["metrics"])
    device_type = metrics.get("device_type") or request.get("device_type")
    fidelity = metrics.get("fidelity") or request.get("fidelity")
    signoff_gaps = metrics.get("signoff_gaps") or []
    if device_type != "power_mosfet_bv_ron" or fidelity != "devsim_2d_field_plate":
        return None
    if not signoff_gaps:
        return None
    baseline_request = dict(request)
    baseline_request.setdefault("run_id", f"{source_state_path.parent.name}_signoff_baseline")
    return AgentExperimentCandidate(
        candidate_id="power_mosfet_2d_signoff_evidence_pack",
        action_kind="run_tool",
        tool_name="power_mosfet_signoff",
        score=0.92,
        source_state_path=str(source_state_path),
        evidence_gap="power_mosfet_2d_signoff_gaps",
        reason="Power MOSFET 2D runner completed but still has mesh/golden/process signoff gaps; run the bundled evidence workflow.",
        expected_effect="Collects 2D baseline, physical benchmark, mesh/model convergence, and optional golden correlation into one gate.",
        request={
            "run_id": f"{source_state_path.parent.name}_signoff",
            "baseline_request": baseline_request,
            "run_convergence": True,
            "execute": True,
            "run_root": str(source_state_path.parent / "power_mosfet_signoff"),
        },
        risk_notes=[str(item) for item in signoff_gaps],
    )


def select_candidate(candidates: list[AgentExperimentCandidate]) -> AgentExperimentCandidate | None:
    executable = [candidate for candidate in candidates if candidate.tool_name or candidate.action_kind != "run_tool"]
    if not executable:
        return None
    return sorted(executable, key=lambda candidate: candidate.score, reverse=True)[0]


def build_agent_experiment_design_plan(
    source_state_path: Path,
    *,
    benchmark_path: Path | None = None,
    output_path: Path | None = None,
) -> AgentExperimentDesignPlan:
    actual_source = source_state_path.resolve()
    try:
        state = load_final_state(str(actual_source)) or read_json(actual_source)
        request = state_request(state)
        benchmark = benchmark_for_state(actual_source, benchmark_path)
        summary = benchmark.summary or {}
        pack = summary.get("signoff_evidence_pack") or {}
        missing = [str(item) for item in pack.get("missing_evidence") or []]
        warning_codes = [str(item) for item in summary.get("warning_codes") or []]
        blocking_codes = [str(item) for item in summary.get("blocking_codes") or []]
        effect = mutation_effect_from_state(actual_source, state, request)
        curve_review = curve_review_from_effect(effect)
        candidates: list[AgentExperimentCandidate] = []
        refinement = refinement_candidate_from_effect(actual_source, effect)
        if refinement:
            candidates.append(refinement)
        power_signoff = power_mosfet_signoff_candidate(actual_source, state, request)
        if power_signoff:
            candidates.append(power_signoff)
        if "convergence_evidence" in missing or "physics_1d_mesh_convergence_missing" in warning_codes:
            candidate = convergence_candidate(actual_source, state, request, score=0.9)
            if candidate:
                candidates.append(candidate)
        if "golden_or_measured_comparison" in missing or "physics_1d_reference_correlation_missing" in warning_codes:
            candidate = golden_candidate(actual_source, state, request, score=0.86)
            if candidate:
                candidates.append(candidate)
        repair = repair_candidate(actual_source, state, benchmark)
        if repair:
            candidates.append(repair)
        schema_extension = sentaurus_schema_extension_candidate(actual_source, state, request)
        if schema_extension:
            candidates.append(schema_extension)
        candidates.extend(mutation_candidates(actual_source, state, request))
        selected = select_candidate(candidates)
        status = "completed" if candidates else "no_actionable_candidates"
        plan = AgentExperimentDesignPlan(
            status=status,
            source_state_path=str(actual_source),
            benchmark_path=benchmark.benchmark_path,
            evidence_gaps=missing,
            signoff_verdict=pack.get("verdict"),
            curve_engineering_review=curve_review,
            mutation_effect_analysis=effect,
            candidates=sorted(candidates, key=lambda candidate: candidate.score, reverse=True),
            selected_candidate=selected,
        )
    except Exception as exc:
        plan = AgentExperimentDesignPlan(status="failed", source_state_path=str(actual_source), failure_reason=str(exc))
    if output_path is not None:
        plan.output_path = str(output_path.resolve())
        write_json(output_path, plan.model_dump(mode="json"))
    return plan
