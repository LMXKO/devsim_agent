from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.device_templates import DeviceTaskTemplate, route_device_goal, device_templates
from tcad_agent.public_sources import build_public_evidence_dossier, public_categories_for_template


class RunnerPromotionStage(BaseModel):
    stage_id: str
    title: str
    status: str = "planned"
    rationale: str
    actions: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class IndustrialRunnerPromotionPlan(BaseModel):
    tool_name: str = "industrial_runner_promotion"
    schema_version: str = "actsoft.tcad.industrial_runner_promotion.v1"
    status: str
    goal_text: str
    template_id: str | None = None
    display_name: str | None = None
    support: str | None = None
    current_tool: str | None = None
    current_fidelity: str | None = None
    promotion_required: bool = False
    evidence_dossier: dict[str, Any] = Field(default_factory=dict)
    stages: list[RunnerPromotionStage] = Field(default_factory=list)
    acceptance_tests: list[str] = Field(default_factory=list)
    next_action: str | None = None
    output_path: str | None = None
    failure_reason: str | None = None


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def template_by_id(template_id: str) -> DeviceTaskTemplate | None:
    for template in device_templates():
        if template.template_id == template_id:
            return template
    return None


def promotion_required(template: DeviceTaskTemplate) -> bool:
    if template.support.value != "executable":
        return True
    if template.tcad_fidelity.startswith("physics_1d"):
        return True
    if any(token in " ".join(template.evidence_requirements).lower() for token in ["3d", "surrogate", "wide_bandgap", "transient"]):
        return True
    return False


def source_ids_for_template(template: DeviceTaskTemplate) -> list[str]:
    ids: list[str] = []
    for category in public_categories_for_template(template.template_id):
        ids.extend(str(item) for item in category.get("source_ids", []) if item)
    return list(dict.fromkeys(ids))


def promotion_stages(template: DeviceTaskTemplate, goal_text: str, evidence_dossier: dict[str, Any]) -> list[RunnerPromotionStage]:
    source_ids = source_ids_for_template(template)
    category_steps: list[str] = []
    for category in public_categories_for_template(template.template_id):
        category_steps.extend(str(item) for item in category.get("promotion_steps", []) if item)
    return [
        RunnerPromotionStage(
            stage_id="public_evidence_and_license_gate",
            title="Public Evidence And License Gate",
            rationale="Ground runner design in public or user-owned evidence before implementation.",
            actions=[
                "Run public_evidence_lookup with live mode for the matched sources.",
                "Record source URLs, license/access notes, and non-vendoring boundaries.",
                "Reject any plan that requires committing proprietary decks, PDKs, model files, binaries, or license strings.",
            ],
            required_artifacts=["public_evidence_lookup.json", "public_evidence_dossier.json"],
            acceptance_criteria=[
                "At least one public or user-provided source is verified for the simulator operation.",
                "Every source is labeled as runnable seed, methodology reference, or external-only vendor material.",
            ],
            source_ids=source_ids,
        ),
        RunnerPromotionStage(
            stage_id="runner_contract",
            title="Runner Contract",
            rationale="Create a stable agent-callable interface before physics implementation expands.",
            actions=[
                f"Define request/state schema for `{template.template_id}` using current tool `{template.executable_tool or 'new_runner'}` as the seed.",
                "Declare required inputs, sweep axes, output CSV columns, log/artifact globs, and cancellation behavior.",
                "Add fixture manifests for fake/interface-only execution when proprietary tools are unavailable.",
            ],
            required_artifacts=["runner_contract.json", "fixture_manifest.json", "state_schema.json"],
            acceptance_criteria=[
                "The runner can produce a deterministic state.json with metrics, artifacts, quality_report, and final_summary.",
                "The contract can run without private software when using fake/interface-only fixtures.",
            ],
            source_ids=source_ids,
        ),
        RunnerPromotionStage(
            stage_id="geometry_mesh_model_implementation",
            title="Geometry, Mesh, And Model Implementation",
            rationale="Promote the existing first-pass route into device-specific TCAD physics.",
            actions=[
                *template.next_implementation_steps,
                *category_steps[:3],
                "Add mesh or bias-convergence knobs that the repair loop can change safely.",
            ],
            required_artifacts=["generated_or_user_deck", "mesh_summary", "model_coupling_summary"],
            acceptance_criteria=[
                "The state records geometry/model parameters and capability boundary explicitly.",
                "Physical benchmark can distinguish executable evidence from compact/surrogate evidence.",
            ],
            source_ids=source_ids,
        ),
        RunnerPromotionStage(
            stage_id="metric_extraction",
            title="Metric Extraction",
            rationale="Industrial runners need stable metrics before the agent can optimize them.",
            actions=[
                f"Extract benchmark metrics: {', '.join(template.benchmark_metrics)}.",
                f"Extract industrial metrics: {', '.join(template.industrial_metrics)}.",
                "Export numeric CSV curves with unit-bearing columns for curve diagnostics.",
            ],
            required_artifacts=["curve.csv", "metrics.json", "extraction_log.txt"],
            acceptance_criteria=[
                "Curve diagnostics can identify leakage/BV/Ron/field or device-specific brackets.",
                "Missing or ambiguous units fail loudly instead of being treated as signoff evidence.",
            ],
            source_ids=source_ids,
        ),
        RunnerPromotionStage(
            stage_id="convergence_and_quality",
            title="Convergence And Quality Gates",
            rationale="The agent must know whether a better metric is physically credible.",
            actions=[
                *template.recommended_convergence,
                "Add mesh/bias/model convergence checks and repair actions for known failure modes.",
            ],
            required_artifacts=["convergence_state.json", "quality_report", "repair_strategy_cases"],
            acceptance_criteria=[
                "At least one convergence smoke test passes for the nominal case.",
                "Known bad units, sparse curves, and nonphysical shapes produce failed/suspicious quality status.",
            ],
            source_ids=source_ids,
        ),
        RunnerPromotionStage(
            stage_id="golden_correlation_and_signoff",
            title="Golden Correlation And Signoff",
            rationale="Strong claims need measured/golden correlation, not just a completed simulation.",
            actions=[
                "Add optional golden/measured curve comparison for the primary metric.",
                "Define signoff evidence pack requirements and blocking reasons.",
                "Ensure compact/surrogate/physics_1d evidence remains conditional until correlation gates pass.",
            ],
            required_artifacts=["golden_curve_comparison.json", "physical_benchmark.json", "signoff_evidence_pack"],
            acceptance_criteria=[
                "Physical benchmark reports ready/conditional/blocked with explicit missing evidence.",
                "Conclusion names the evidence level and does not overclaim signoff.",
            ],
            source_ids=source_ids,
        ),
        RunnerPromotionStage(
            stage_id="autonomous_e2e_validation",
            title="Autonomous E2E Validation",
            rationale="A runner is not agent-ready until the loop can improve or stop from evidence.",
            actions=[
                "Add long_run_validation scenario: baseline -> patch/mutation -> curve effect -> refinement -> benchmark -> report.",
                "Add queue resume/cancel coverage for long-running cases.",
                "Add sensitive patch approval tests for geometry/process/model edits.",
            ],
            required_artifacts=["long_run_validation_result.json", "dashboard.html", "lineage_or_mutation_effect.json"],
            acceptance_criteria=[
                "The autonomous agent completes or pauses at an explicit confirmation gate.",
                "Lineage records what changed, why, curve movement, Pareto decision, and next action.",
            ],
            source_ids=source_ids,
        ),
    ]


def build_industrial_runner_promotion_plan(
    goal_text: str,
    *,
    template_id: str | None = None,
    simulator: str | None = None,
    live_lookup_result: dict[str, Any] | None = None,
    output_path: Path | None = None,
) -> IndustrialRunnerPromotionPlan:
    try:
        template = template_by_id(template_id) if template_id else None
        route = route_device_goal(goal_text) if template is None else None
        template = template or (route.template if route else None)
        if template is None:
            return IndustrialRunnerPromotionPlan(
                status="unmatched",
                goal_text=goal_text,
                failure_reason="No device template matched the goal or template_id.",
            )
        dossier = build_public_evidence_dossier(
            goal_text,
            simulator=simulator,
            template_ids=[template.template_id],
            live_lookup_result=live_lookup_result,
        ).model_dump(mode="json")
        required = promotion_required(template)
        stages = promotion_stages(template, goal_text, dossier)
        tests = [
            f"python3.11 -m tcad_agent.tools.device_templates route --goal {json.dumps(goal_text, ensure_ascii=False)}",
            "python3.11 -m tcad_agent.tools.public_evidence_lookup --live --goal <goal> --template-id <template>",
            "python3.11 -m tcad_agent.tools.long_run_validation --suite autonomous_e2e --validation-id <runner>_promotion",
            "python3.11 -m unittest tests.test_physical_benchmark tests.test_autonomous_devsim_agent",
        ]
        plan = IndustrialRunnerPromotionPlan(
            status="completed",
            goal_text=goal_text,
            template_id=template.template_id,
            display_name=template.display_name,
            support=template.support.value,
            current_tool=template.executable_tool,
            current_fidelity=template.tcad_fidelity,
            promotion_required=required,
            evidence_dossier=dossier,
            stages=stages,
            acceptance_tests=tests,
            next_action="implement_runner_contract" if required else "run_nominal_convergence_and_golden_correlation",
        )
    except Exception as exc:
        plan = IndustrialRunnerPromotionPlan(status="failed", goal_text=goal_text, template_id=template_id, failure_reason=str(exc))
    if output_path is not None:
        target = output_path.expanduser().resolve()
        plan.output_path = str(target)
        write_json(target, plan.model_dump(mode="json"))
    return plan
