from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.engineering_intent import DeviceSupport, EngineeringIntent, parse_engineering_intent


class CompiledMissionSpec(BaseModel):
    schema_version: str = "actsoft.tcad.compiled_mission_spec.v1"
    created_at: str
    goal_text: str
    intent: dict[str, Any]
    selected_tool: str | None = None
    initial_request: dict[str, Any] = Field(default_factory=dict)
    objectives: list[dict[str, Any]] = Field(default_factory=list)
    constraints: list[dict[str, Any]] = Field(default_factory=list)
    allowed_mutations: list[dict[str, Any]] = Field(default_factory=list)
    stop_conditions: list[str] = Field(default_factory=list)
    validation_plan: list[str] = Field(default_factory=list)
    risk_gates: list[dict[str, Any]] = Field(default_factory=list)
    memory_context: list[dict[str, Any]] = Field(default_factory=list)
    status: str = "compiled"
    summary: str = ""


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def mutation_entry(name: str, *, risk: str, reason: str, evidence: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "risk": risk,
        "reason": reason,
        "required_evidence": evidence,
    }


def allowed_mutations_for_intent(intent: EngineeringIntent) -> list[dict[str, Any]]:
    mutations: list[dict[str, Any]] = []
    device = intent.device_family
    hints = set(intent.model_hints)
    metrics = set(intent.metrics)
    analyses = set(intent.analyses)
    if device in {"power_mosfet", "sic_power_diode", "gan_hemt"} or {"field_plate", "drift_doping"} & hints:
        mutations.extend(
            [
                mutation_entry("field_plate", risk="high", reason="shape electric-field peak and BV tradeoff", evidence=["field_peak", "bv_bracket", "overlay"]),
                mutation_entry("guard_ring", risk="high", reason="spread edge field for high-voltage junctions", evidence=["field_peak_position", "bv_bracket", "layout/deck evidence"]),
                mutation_entry("drift_doping", risk="medium", reason="trade BV against Ron/leakage", evidence=["bv", "ron", "leakage_window"]),
                mutation_entry("junction_depth", risk="medium", reason="move junction curvature and depletion profile", evidence=["field_peak_position", "bv", "process/deck binding"]),
                mutation_entry("implant_dose", risk="medium", reason="process-level proxy for active doping and Ron/BV movement", evidence=["dose binding", "bv", "ron"]),
                mutation_entry("trench_corner_radius", risk="high", reason="reduce high-field crowding around corners", evidence=["field_peak_position", "mesh", "geometry binding"]),
            ]
        )
    if "cv" in analyses or device == "mos_capacitor" or "fixed_oxide_charge" in hints:
        mutations.extend(
            [
                mutation_entry("oxide_thickness", risk="medium", reason="change Cox and flatband/CV shape", evidence=["cox", "cv_overlay", "oxide region binding"]),
                mutation_entry("fixed_oxide_charge", risk="medium", reason="explain flatband shifts", evidence=["flatband_shift", "cv_overlay"]),
                mutation_entry("interface_trap_density", risk="medium", reason="shape stretch-out and C-V transition", evidence=["cv_shape", "trap model binding"]),
            ]
        )
    if "leakage" in metrics or "leakage" in analyses or "srh_lifetime" in hints:
        mutations.extend(
            [
                mutation_entry("lifetime", risk="medium", reason="change recombination/leakage and stored charge", evidence=["leakage_window", "carrier_lifetime binding"]),
                mutation_entry("region_specific_lifetime", risk="medium", reason="localize leakage repair with less global tradeoff", evidence=["region binding", "leakage_window", "overlay"]),
                mutation_entry("trap_density", risk="medium", reason="capture leakage/current-collapse style effects", evidence=["trap model binding", "hysteresis/stress evidence"]),
            ]
        )
    if not mutations:
        mutations.append(
            mutation_entry(
                "bias_or_mesh_refinement",
                risk="low",
                reason="first repair numerical coverage before physical/process edits",
                evidence=["convergence_log", "curve_shape", "unit_check"],
            )
        )
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for item in mutations:
        if item["name"] in seen:
            continue
        output.append(item)
        seen.add(item["name"])
    return output


def objectives_for_intent(intent: EngineeringIntent) -> list[dict[str, Any]]:
    objectives: list[dict[str, Any]] = []
    for name in intent.objectives:
        if name == "minimize_leakage":
            objectives.append({"metric": "leakage_current_a", "direction": "minimize", "weight": 1.0})
        elif name == "meet_bv":
            objectives.append({"metric": "breakdown_voltage_v", "direction": "maximize", "weight": 1.0})
        elif name == "maximize_ion_ioff":
            objectives.append({"metric": "ion_ioff_ratio", "direction": "maximize", "weight": 1.0})
        elif name == "fit_measured_curve":
            objectives.append({"metric": "golden_curve_rmse_log_dec", "direction": "minimize", "weight": 1.0})
        else:
            objectives.append({"name": name, "direction": "extract", "weight": 0.5})
    if not objectives:
        for metric in intent.metrics[:4]:
            objectives.append({"metric": metric, "direction": "extract", "weight": 0.5})
    return objectives


def constraints_for_intent(intent: EngineeringIntent) -> list[dict[str, Any]]:
    constraints = [{"expression": item, "source": "natural_language"} for item in intent.constraints]
    if "ron" in intent.metrics and "bv" in intent.metrics:
        constraints.append({"expression": "do_not_improve_bv_by_destroying_ron", "source": "tradeoff_guardrail"})
    if "leakage" in intent.metrics and "bv" in intent.metrics:
        constraints.append({"expression": "leakage_bv_tradeoff_requires_pareto_review", "source": "tradeoff_guardrail"})
    return constraints


def validation_plan_for_intent(intent: EngineeringIntent) -> list[str]:
    plan = ["run_initial_tool_or_supervisor", "curve_shape_diagnostic", "physical_benchmark"]
    if "mesh_convergence" in intent.evidence_requirements or "convergence" in intent.analyses:
        plan.append("mesh_or_bias_convergence_check")
    if "golden_or_measured" in intent.evidence_requirements or "calibration" in intent.analyses:
        plan.append("golden_or_measured_curve_compare")
    if intent.objectives or intent.constraints:
        plan.append("pareto_or_constraint_review")
    plan.append("engineer_readable_report")
    return plan


def stop_conditions_for_intent(intent: EngineeringIntent) -> list[str]:
    stops = [
        "state_status_completed_or_waiting_for_user",
        "physical_benchmark_done",
        "curve_shape_review_recorded",
    ]
    if intent.objectives or intent.constraints:
        stops.append("objective_or_constraint_decision_recorded")
    if intent.evidence_policy in {"needs_clarification", "blocked_until_runner_implemented"}:
        stops.append("ask_user_before_execution")
    else:
        stops.append("report_or_cockpit_artifact_written")
    return stops


def risk_gates_for_intent(intent: EngineeringIntent) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    if intent.support in {DeviceSupport.UNKNOWN, DeviceSupport.PLANNED}:
        gates.append({"gate": "capability", "action": "ask_user", "reason": intent.evidence_policy})
    if any(item["risk"] == "high" for item in allowed_mutations_for_intent(intent)):
        gates.append({"gate": "high_risk_mutation", "action": "require_confirmation", "reason": "geometry/process/model edit"})
    if "golden_or_measured" in intent.evidence_requirements:
        gates.append({"gate": "reference_curve", "action": "require_artifact_or_pause", "reason": "measured/golden comparison requested"})
    return gates


def compact_memory_context(records: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records[:limit]:
        output.append(
            {
                key: record.get(key)
                for key in [
                    "record_id",
                    "goal_text",
                    "device_family",
                    "template_id",
                    "status",
                    "outcome",
                    "final_state_path",
                    "curve_guidance_summary",
                    "recovery_summary",
                    "next_action",
                ]
                if record.get(key) is not None
            }
        )
    return output


def compile_mission_spec(goal_text: str, *, memory_records: list[dict[str, Any]] | None = None) -> CompiledMissionSpec:
    intent = parse_engineering_intent(goal_text)
    memory_context = compact_memory_context(memory_records or [])
    status = "needs_clarification" if intent.evidence_policy in {"needs_clarification", "blocked_until_runner_implemented"} else "compiled"
    return CompiledMissionSpec(
        created_at=utc_timestamp(),
        goal_text=goal_text,
        intent=intent.model_dump(mode="json"),
        selected_tool=intent.suggested_tool,
        initial_request=intent.request_hint,
        objectives=objectives_for_intent(intent),
        constraints=constraints_for_intent(intent),
        allowed_mutations=allowed_mutations_for_intent(intent),
        stop_conditions=stop_conditions_for_intent(intent),
        validation_plan=validation_plan_for_intent(intent),
        risk_gates=risk_gates_for_intent(intent),
        memory_context=memory_context,
        status=status,
        summary=intent.summary_zh,
    )


def apply_mission_spec_to_autonomous_request(
    autonomous_request: dict[str, Any],
    spec: CompiledMissionSpec,
) -> dict[str, Any]:
    payload = dict(autonomous_request)
    payload.setdefault("mission_spec", spec.model_dump(mode="json"))
    payload.setdefault("agent_memory_context", spec.memory_context)
    if spec.selected_tool and not payload.get("initial_tool_name") and spec.status == "compiled":
        payload["initial_tool_name"] = spec.selected_tool
        payload.setdefault("initial_request", spec.initial_request)
    elif spec.initial_request and not payload.get("initial_request"):
        payload["initial_request"] = spec.initial_request
    if spec.constraints and "constraints" not in payload:
        payload["natural_language_constraints"] = spec.constraints
    if spec.objectives and "objectives" not in payload:
        payload["natural_language_objectives"] = spec.objectives
    if any(step in spec.validation_plan for step in ["pareto_or_constraint_review", "golden_or_measured_curve_compare"]):
        payload.setdefault("enable_experiment_design", True)
    if spec.status == "needs_clarification" and not payload.get("initial_tool_name"):
        payload.setdefault("require_capability_audit", True)
    return payload
