from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RepairPlanStatus(str, Enum):
    PLANNED = "planned"
    NO_ACTION = "no_action"
    FAILED = "failed"


class RepairAction(BaseModel):
    name: str
    priority: int
    reason: str
    target_tool: str | None = None
    request_patch: dict[str, Any] = Field(default_factory=dict)
    checklist: list[str] = Field(default_factory=list)
    expected_effect: str
    user_confirmation_required: bool = False


class RepairPlan(BaseModel):
    status: RepairPlanStatus
    state_path: str
    output_path: str | None = None
    tool_name: str | None = None
    run_id: str | None = None
    quality_status: str | None = None
    failure_classes: list[str] = Field(default_factory=list)
    issue_codes: list[str] = Field(default_factory=list)
    actions: list[RepairAction] = Field(default_factory=list)
    next_action: str | None = None
    created_at: str
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def default_output_path(state_path: Path) -> Path:
    return state_path.parent / "repair_plan.json"


def float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def issue_codes(state: dict[str, Any]) -> set[str]:
    quality_report = state.get("quality_report") or state.get("final_quality_report") or {}
    codes = {str(issue.get("code")) for issue in quality_report.get("issues", []) if issue.get("code")}
    if state.get("tool_name") == "tool_convergence":
        failed_cases = [case for case in state.get("cases") or [] if isinstance(case, dict) and case.get("status") == "failed"]
        if failed_cases:
            codes.add("tool_convergence_case_failures")
            if any(case.get("failure_reason") for case in failed_cases):
                codes.add("runner_exception")
    return codes


def classify_failure_text(text: str) -> str | None:
    lowered = text.lower()
    if "string should match pattern" in lowered or "validation" in lowered or "pydantic" in lowered:
        return "validation"
    if "converge" in lowered or "convergence" in lowered or "did not converge" in lowered:
        return "convergence"
    if "traceback" in lowered or "exception" in lowered or "runner" in lowered:
        return "runtime_exception"
    return None


def failure_classes(state: dict[str, Any]) -> list[str]:
    classes = []
    for attempt in state.get("attempts") or []:
        value = attempt.get("failure_class")
        if value and value != "none":
            classes.append(str(value))
    checkpoint = state.get("checkpoint") or {}
    value = checkpoint.get("last_failure_class")
    if value and value != "none":
        classes.append(str(value))
    for case in state.get("cases") or []:
        if not isinstance(case, dict) or case.get("status") != "failed":
            continue
        failure_class = classify_failure_text(str(case.get("failure_reason") or ""))
        if failure_class:
            classes.append(failure_class)
    if str(state.get("failure_reason") or ""):
        failure_class = classify_failure_text(str(state.get("failure_reason") or ""))
        if failure_class:
            classes.append(failure_class)
    return sorted(set(classes))


def quality_status(state: dict[str, Any]) -> str | None:
    quality_report = state.get("quality_report") or state.get("final_quality_report") or {}
    return quality_report.get("status")


def halve_step_patch(request: dict[str, Any]) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    for field, min_field in [("step", "min_step"), ("gate_step", "min_gate_step"), ("drain_step", "min_drain_step")]:
        value = float_or_none(request.get(field))
        if value is None:
            continue
        min_value = float_or_none(request.get(min_field)) or max(value / 8.0, 1e-6)
        patch[field] = max(value / 2.0, min_value)
        patch[min_field] = min(min_value, max(patch[field] / 4.0, 1e-6))
    if "max_attempts" in request:
        patch["max_attempts"] = max(int(request.get("max_attempts") or 1), 3)
    return patch


def continuation_patch(request: dict[str, Any]) -> dict[str, Any]:
    patch = halve_step_patch(request)
    if "max_attempts" in request:
        patch["max_attempts"] = max(int(request.get("max_attempts") or 1) + 2, 5)
    patch["resume"] = False
    return patch


def mesh_relax_patch(request: dict[str, Any]) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    for key in ["contact_spacing_um", "junction_spacing_um", "oxide_spacing_nm", "silicon_spacing_um"]:
        value = float_or_none(request.get(key))
        if value is not None:
            patch[key] = value * 2.0
    for key in ["x_divisions", "silicon_y_divisions"]:
        value = float_or_none(request.get(key))
        if value is not None:
            patch[key] = max(int(value / 2.0), 4 if key == "x_divisions" else 3)
    return patch


def mesh_refine_patch(request: dict[str, Any]) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    for key in ["contact_spacing_um", "junction_spacing_um", "oxide_spacing_nm", "silicon_spacing_um"]:
        value = float_or_none(request.get(key))
        if value is not None:
            patch[key] = max(value / 2.0, 1e-9)
    for key in ["x_divisions", "silicon_y_divisions"]:
        value = float_or_none(request.get(key))
        if value is not None:
            patch[key] = int(value * 1.5) + 1
    return patch


def solver_adjustment_patch(request: dict[str, Any]) -> dict[str, Any]:
    max_iterations = int(float_or_none(request.get("solver_max_iterations")) or 80)
    relative_error = float_or_none(request.get("solver_relative_error")) or 1e-10
    absolute_error = float_or_none(request.get("solver_absolute_error")) or 1e10
    return {
        "solver_strategy": "increase_iterations_and_relax_error",
        "solver_max_iterations": max(max_iterations * 2, 120),
        "solver_relative_error": max(relative_error, 1e-9),
        "solver_absolute_error": max(absolute_error, 1e10),
        "solver_initial_absolute_error": max(float_or_none(request.get("solver_initial_absolute_error")) or 1.0, 1.0),
    }


def model_staging_patch(request: dict[str, Any]) -> dict[str, Any]:
    patch: dict[str, Any] = {
        "model_strategy": "poisson_then_dd",
        "advanced_model_staging": ["poisson", "drift_diffusion", "advanced_models"],
    }
    if request.get("impact_ionization_model") not in {None, "none"}:
        patch["impact_ionization_model"] = "none"
        patch["deferred_impact_ionization_model"] = request.get("impact_ionization_model")
    return patch


def initial_solution_reuse_patch(request: dict[str, Any]) -> dict[str, Any]:
    patch = halve_step_patch(request)
    patch.update(
        {
            "resume": True,
            "initial_condition_strategy": "reuse_last_successful_bias",
            "continuation_from_checkpoint": True,
        }
    )
    return patch


def convergence_case_safe_retry_patch(request: dict[str, Any]) -> dict[str, Any]:
    patch = continuation_patch(request)
    patch.update(
        {
            "resume": True,
            "initial_condition_strategy": "reuse_last_successful_bias",
            "continuation_from_checkpoint": True,
            "repair_scope": "failed_tool_convergence_case",
            "tool_convergence_repair_hint": "rerun failed convergence case with smaller bias step before accepting aggregate evidence",
        }
    )
    return patch


def reverse_range_extension_patch(request: dict[str, Any]) -> dict[str, Any]:
    start = float_or_none(request.get("start"))
    stop = float_or_none(request.get("stop"))
    step = abs(float_or_none(request.get("step")) or 0.5)
    if start is None:
        start = 0.0
    if stop is None or stop >= -1.0:
        stop = -10.0
    else:
        stop = min(stop * 1.5, stop - 5.0)
    return {"start": start, "stop": stop, "step": min(step, max(abs(stop - start) / 40.0, 0.05)), "min_step": min(step / 4.0, 0.05)}


def lifetime_sweep_seed_patch(request: dict[str, Any]) -> dict[str, Any]:
    electron = float_or_none(request.get("electron_lifetime_s")) or 1.0e-8
    hole = float_or_none(request.get("hole_lifetime_s")) or electron
    return {
        "electron_lifetime_s": max(electron * 10.0, 1.0e-9),
        "hole_lifetime_s": max(hole * 10.0, 1.0e-9),
        "lifetime_repair_hint": "compare leakage against one-decade SRH lifetime perturbation",
    }


def geometry_sanity_patch(request: dict[str, Any]) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    length = float_or_none(request.get("length_um"))
    junction = float_or_none(request.get("junction_um"))
    if length is not None and junction is not None and junction >= length:
        patch["junction_um"] = length / 2.0
    silicon = float_or_none(request.get("silicon_thickness_um"))
    depth = float_or_none(request.get("source_drain_depth_um"))
    if silicon is not None and depth is not None and depth >= silicon:
        patch["source_drain_depth_um"] = silicon / 2.0
    source_drain_length = float_or_none(request.get("source_drain_length_um"))
    if length is not None and source_drain_length is not None and source_drain_length * 2.0 >= length:
        patch["source_drain_length_um"] = length / 4.0
    return patch


def unit_bias_patch(request: dict[str, Any]) -> dict[str, Any]:
    start = float_or_none(request.get("start"))
    stop = float_or_none(request.get("stop"))
    if start is None or stop is None:
        return {}
    span = stop - start
    if abs(span) <= 50.0:
        return {}
    limited_span = 50.0 if span > 0 else -50.0
    new_stop = start + limited_span
    step = abs(limited_span) / 10.0
    return {"stop": new_stop, "step": min(float_or_none(request.get("step")) or step, step), "min_step": step / 4.0}


def add_action(actions: list[RepairAction], action: RepairAction) -> None:
    if any(existing.name == action.name for existing in actions):
        return
    actions.append(action)


def repair_target_tool(state: dict[str, Any]) -> str | None:
    if state.get("tool_name") == "tool_convergence":
        target = state.get("target_tool")
        return str(target) if target else None
    tool_name = state.get("tool_name")
    return str(tool_name) if tool_name else None


def repair_request(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("tool_name") != "tool_convergence":
        return dict(state.get("request") or {})
    cases = [case for case in state.get("cases") or [] if isinstance(case, dict)]
    failed = [case for case in cases if case.get("status") == "failed" and isinstance(case.get("request"), dict)]
    if failed:
        return dict(failed[0]["request"])
    with_requests = [case for case in cases if isinstance(case.get("request"), dict)]
    if with_requests:
        return dict(with_requests[-1]["request"])
    return dict(state.get("base_request") or {})


def schema_normalization_actions(tool_name: str | None, request: dict[str, Any], actions: list[RepairAction]) -> None:
    add_action(
        actions,
        RepairAction(
            name="schema_field_alias_normalization",
            priority=98,
            reason="Tool validation failed or the plan used human/LLM field aliases instead of executable request fields.",
            target_tool=tool_name,
            request_patch={},
            checklist=[
                "Normalize sweep aliases such as output_characteristic -> idvd and transfer_characteristic -> idvg.",
                "Map informal mesh_refinement_level to executable mesh fields such as x_divisions.",
                "Validate the patched request with the Pydantic request model before running the simulator.",
            ],
            expected_effect="Repairs field-name/schema mismatches without changing the intended TCAD physics.",
        ),
    )


def convergence_actions(tool_name: str | None, request: dict[str, Any], actions: list[RepairAction]) -> None:
    add_action(
        actions,
        RepairAction(
            name="continuation_bias_ramp",
            priority=100,
            reason="Convergence failure usually needs a smoother bias continuation path before changing physics.",
            target_tool=tool_name,
            request_patch=continuation_patch(request),
            checklist=[
                "Start from the last stable low-bias point.",
                "Ramp the failing terminal in smaller increments.",
                "Keep the same geometry and models for the first retry.",
            ],
            expected_effect="Improves nonlinear solver convergence by reducing the state jump between bias points.",
        ),
    )
    add_action(
        actions,
        RepairAction(
            name="reuse_last_successful_initial_solution",
            priority=92,
            reason="A previous low-bias solution can often seed the failing continuation point more reliably than a cold start.",
            target_tool=tool_name,
            request_patch=initial_solution_reuse_patch(request),
            checklist=[
                "Reuse the last successful bias point as the initial condition.",
                "Keep the same geometry and physical models for this retry.",
                "If the retry passes, run a benchmark or convergence check before signoff.",
            ],
            expected_effect="Improves convergence without changing the intended device or model setup.",
        ),
    )
    add_action(
        actions,
        RepairAction(
            name="mesh_relax_for_initial_solution",
            priority=80,
            reason="A slightly coarser mesh can produce an initial solution when the fine mesh fails immediately.",
            target_tool=tool_name,
            request_patch=mesh_relax_patch(request),
            checklist=[
                "Use relaxed mesh only to obtain an initial solution.",
                "Follow with a mesh convergence check before trusting extracted metrics.",
            ],
            expected_effect="Reduces nonlinear stiffness and makes the initial solve easier.",
        ),
    )
    add_action(
        actions,
        RepairAction(
            name="solver_parameter_adjustment",
            priority=70,
            reason="Repeated convergence failures may need solver damping or iteration-limit changes.",
            target_tool=tool_name,
            request_patch=solver_adjustment_patch(request),
            checklist=[
                "Increase maximum nonlinear iterations.",
                "Enable damping or line-search if exposed by the runner.",
                "Keep logs for comparison with the original failure.",
            ],
            expected_effect="Gives the Newton solve more room to converge without changing device intent.",
        ),
    )
    model_staging = model_staging_patch(request)
    add_action(
        actions,
        RepairAction(
            name="model_switch_staging",
            priority=90 if model_staging.get("deferred_impact_ionization_model") else 60,
            reason="Hard physics models should be staged after a simpler Poisson or drift-diffusion initialization.",
            target_tool=tool_name,
            request_patch=model_staging,
            checklist=[
                "Solve Poisson-only first.",
                "Enable drift-diffusion after electrostatic convergence.",
                "Add mobility/recombination/interface models one at a time.",
            ],
            expected_effect="Separates physical model difficulty from geometry or bias-ramp problems.",
        ),
    )


def quality_issue_actions(tool_name: str | None, request: dict[str, Any], codes: set[str], actions: list[RepairAction]) -> None:
    if codes & {"string_pattern_mismatch", "invalid_tool_schema", "field_alias_mismatch", "schema_validation_failed"}:
        schema_normalization_actions(tool_name, request, actions)

    if codes & {"too_many_convergence_failures"}:
        convergence_actions(tool_name, request, actions)

    if codes & {"too_few_completed_convergence_cases", "tool_convergence_case_failures", "runner_exception"}:
        add_action(
            actions,
            RepairAction(
                name="rerun_failed_convergence_cases_with_safe_bias",
                priority=105,
                reason="工具收敛验证中有 case 失败或完成点不足，需要先用更保守 bias continuation 重跑失败 case。",
                target_tool=tool_name,
                request_patch=convergence_case_safe_retry_patch(request),
                checklist=[
                    "从失败 convergence case 的真实 TCAD 请求重跑，而不是重跑整个聚合状态。",
                    "缩小 gate/drain/reverse bias step，并复用最近成功初值。",
                    "重跑后再回到 tool_convergence 或 physical_benchmark 验证证据密度。",
                ],
                expected_effect="把聚合收敛失败转化为目标 TCAD 工具上的可执行安全重试。",
            ),
        )

    if codes & {"mesh_not_converged", "mesh_spacing_too_coarse_for_device"}:
        add_action(
            actions,
            RepairAction(
                name="mesh_refinement_and_convergence_check",
                priority=95,
                reason="Metrics changed with mesh or mesh spacing is too coarse.",
                target_tool=tool_name,
                request_patch=mesh_refine_patch(request),
                checklist=[
                    "Halve the relevant mesh spacing.",
                    "Run mesh_convergence across at least three mesh values.",
                    "Accept only if the objective changes below tolerance on the two finest meshes.",
                ],
                expected_effect="Separates physical trends from discretization artifacts.",
            ),
        )

    if codes & {"voltage_span_unusually_large", "mos_cv_voltage_span_unusually_large", "invalid_voltage_range"}:
        add_action(
            actions,
            RepairAction(
                name="unit_and_bias_range_repair",
                priority=90,
                reason="Bias range looks like a unit mistake or an overly aggressive first sweep.",
                target_tool=tool_name,
                request_patch=unit_bias_patch(request),
                checklist=[
                    "Confirm all voltages are in volts.",
                    "Run a narrow smoke sweep before restoring the requested span.",
                ],
                expected_effect="Prevents solver and quality failures caused by accidental mV/V or extreme-bias confusion.",
                user_confirmation_required=True,
            ),
        )

    if codes & {
        "junction_not_inside_device",
        "invalid_geometry_value",
        "source_drain_depth_exceeds_silicon",
        "source_drain_regions_leave_no_channel",
    }:
        add_action(
            actions,
            RepairAction(
                name="geometry_sanity_repair",
                priority=90,
                reason="Geometry is invalid or leaves no physically meaningful active region.",
                target_tool=tool_name,
                request_patch=geometry_sanity_patch(request),
                checklist=[
                    "Check geometry units, especially nm vs um.",
                    "Ensure junctions and contacts lie inside the simulated domain.",
                    "Review boundary condition names after geometry changes.",
                ],
                expected_effect="Restores a physically valid TCAD domain before retrying the solver.",
                user_confirmation_required=True,
            ),
        )

    if codes & {
        "diode_reverse_sweep_missing_negative_bias",
        "breakdown_not_reached",
        "diode_breakdown_not_reached",
    }:
        add_action(
            actions,
            RepairAction(
                name="extend_reverse_bias_window",
                priority=88,
                reason="Reverse leakage or breakdown extraction did not cover the necessary reverse-bias region.",
                target_tool=tool_name,
                request_patch=reverse_range_extension_patch(request),
                checklist=[
                    "Extend the reverse-bias stop voltage while keeping a conservative step.",
                    "Do not claim BV pass/fail unless the threshold is bracketed or the requested range is documented.",
                    "Watch reverse-current monotonicity after extending the sweep.",
                ],
                expected_effect="Covers the leakage/BV region before drawing an engineering conclusion.",
            ),
        )

    if codes & {
        "diode_leakage_above_policy",
        "reverse_current_gain_below_one",
        "leakage_exceeds_max_reverse_current",
    }:
        add_action(
            actions,
            RepairAction(
                name="srh_lifetime_and_boundary_sanity",
                priority=76,
                reason="Leakage magnitude or shape suggests SRH lifetime, contact, or boundary-condition sensitivity.",
                target_tool=tool_name,
                request_patch=lifetime_sweep_seed_patch(request),
                checklist=[
                    "Perturb SRH lifetime by one decade to check leakage sensitivity.",
                    "Confirm reverse-bias contact polarity and current sign convention.",
                    "Compare leakage at a fixed target voltage before interpreting BV.",
                ],
                expected_effect="Separates physical leakage sensitivity from boundary/sign-convention artifacts.",
            ),
        )

    if codes & {"doping_out_of_expected_range", "invalid_doping_value", "capacitance_exceeds_oxide_capacitance"}:
        add_action(
            actions,
            RepairAction(
                name="doping_and_unit_sanity_review",
                priority=75,
                reason="Material parameters or extracted capacitance suggest a units or deck-parameter problem.",
                target_tool=tool_name,
                request_patch={},
                checklist=[
                    "Confirm doping is in cm^-3.",
                    "Confirm oxide thickness is in nm and lengths are in um.",
                    "Compare MOS C-V capacitance against Cox before accepting trends.",
                ],
                expected_effect="Avoids optimizing against physically impossible parameterization.",
                user_confirmation_required=True,
            ),
        )

    if codes & {
        "idvg_not_monotonic",
        "reverse_current_not_monotonic",
        "current_not_monotonic",
        "idvd_negative_differential_conductance",
        "idvd_kink_suspected",
    }:
        add_action(
            actions,
            RepairAction(
                name="local_bias_step_refinement",
                priority=85,
                reason="Curve shape is non-monotonic where a smooth TCAD response is expected.",
                target_tool=tool_name,
                request_patch=halve_step_patch(request),
                checklist=[
                    "Rerun with smaller local bias step around the suspicious segment.",
                    "Inspect whether the sign convention or contact current changed.",
                    "If the artifact persists, run mesh convergence at the same bias range.",
                ],
                expected_effect="Distinguishes real device behavior from continuation or discretization artifacts.",
            ),
        )

    if codes & {"idvd_kink_suspected", "idvd_saturation_not_observed"}:
        add_action(
            actions,
            RepairAction(
                name="mosfet_output_physics_triage",
                priority=82,
                reason="Id-Vd output curve suggests kink behavior, missing saturation, or high-drain numerical artifacts.",
                target_tool=tool_name,
                request_patch={
                    **halve_step_patch(request),
                    "impact_ionization_model": request.get("impact_ionization_model") or "none",
                    "model_strategy": "poisson_then_dd",
                },
                checklist=[
                    "Rerun Id-Vd with smaller drain step near the suspicious high-Vd segment.",
                    "Compare constant vs doping-dependent mobility before blaming impact ionization.",
                    "Run x_divisions convergence on idvd_final_current_a before accepting kink as physical.",
                ],
                expected_effect="Separates real high-field behavior from continuation, mobility-model, and mesh artifacts.",
            ),
        )

    if codes & {"threshold_not_crossed", "low_ion_ioff_ratio", "mosfet_ion_ioff_ratio_low", "mosfet_vth_outside_gate_sweep"}:
        add_action(
            actions,
            RepairAction(
                name="mosfet_sweep_range_extension",
                priority=65,
                reason="The Id-Vg sweep does not expose a reliable on/off transition.",
                target_tool=tool_name,
                request_patch={"gate_stop": max(float_or_none(request.get("gate_stop")) or 1.0, 1.0)},
                checklist=[
                    "Extend gate sweep only after confirming oxide thickness and doping are reasonable.",
                    "Keep drain voltage low for Vth/SS extraction.",
                ],
                expected_effect="Gives Vth, SS, Ion, and Ioff extraction enough dynamic range.",
            ),
        )

    if codes & {
        "moscap_cv_dynamic_range_too_low",
        "capacitance_far_below_oxide_capacitance",
        "fixed_charge_shift_exceeds_sweep_window",
        "fixed_oxide_charge_not_accounted_in_metrics",
    }:
        add_action(
            actions,
            RepairAction(
                name="moscap_bias_and_charge_window_review",
                priority=78,
                reason="MOS C-V shape or fixed-charge shift suggests the bias window may not expose accumulation/depletion reliably.",
                target_tool=tool_name,
                request_patch={
                    "start": min(float_or_none(request.get("start")) or -1.0, -2.0),
                    "stop": max(float_or_none(request.get("stop")) or 1.0, 2.0),
                    "step": min(float_or_none(request.get("step")) or 0.25, 0.25),
                },
                checklist=[
                    "Check Cox from oxide thickness before interpreting flat-band shift.",
                    "Run fixed_oxide_charge=0 as a baseline when debugging Qf.",
                    "Use a wider gate sweep if the equivalent fixed-charge shift approaches the sweep span.",
                ],
                expected_effect="Makes MOS C-V trend and flat-band/fixed-charge interpretation more robust.",
            ),
        )

    if codes & {
        "interface_trap_model_metadata_only",
        "fixed_oxide_charge_metadata_only",
        "impact_ionization_model_metadata_only",
        "deck_physics_model_coupling_needs_confirmation",
        "deck_spec_contains_model_warnings",
        "subthreshold_swing_below_thermal_limit",
    }:
        add_action(
            actions,
            RepairAction(
                name="model_coupling_and_extraction_review",
                priority=72,
                reason="Requested physical model or extracted metric may not be consistently coupled into the equations.",
                target_tool=tool_name,
                request_patch={"model_strategy": "poisson_then_dd"},
                checklist=[
                    "Confirm whether requested traps, fixed charge, or avalanche terms are equation-coupled or metadata-only.",
                    "Repeat extraction with a conservative current floor and bias window.",
                    "Do not use the metric for signoff until the coupling state is explicit.",
                ],
                expected_effect="Prevents accepting plausible-looking curves whose physical model is not actually active.",
            ),
        )

    if codes & {"deck_signoff_convergence_evidence_missing"}:
        add_action(
            actions,
            RepairAction(
                name="signoff_evidence_density_retry",
                priority=84,
                reason="工程签核任务缺少收敛证据，先用更保守 bias step 和更细网格补一条可复核结果。",
                target_tool=tool_name,
                request_patch={**halve_step_patch(request), **mesh_refine_patch(request), "signoff_repair_hint": "add convergence evidence before signoff"},
                checklist=[
                    "保留原始物理意图，只增加证据密度。",
                    "优先在影响结论的 bias 区间做局部细化。",
                    "重跑后必须再次执行 physical_benchmark。",
                ],
                expected_effect="把签核证据缺口转化成可执行的局部细化/收敛补证。",
            ),
        )

    if codes & {"deck_measured_curve_comparison_missing"}:
        add_action(
            actions,
            RepairAction(
                name="measured_curve_comparison_required",
                priority=40,
                reason="任务要求与实测/可信曲线对比，但当前结果缺少曲线对齐和误差评估。",
                target_tool=tool_name,
                request_patch={},
                checklist=[
                    "加载实测/可信曲线并确认单位、电流符号和面积归一化。",
                    "计算 log-current RMSE 或关键 bias 点误差。",
                    "完成对比前不要给出强签核结论。",
                ],
                expected_effect="避免在缺少实测/golden 对比时误判模型可信度。",
                user_confirmation_required=True,
            ),
        )


def build_repair_actions(state: dict[str, Any]) -> list[RepairAction]:
    tool_name = repair_target_tool(state)
    request = repair_request(state)
    codes = issue_codes(state)
    classes = set(failure_classes(state))
    actions: list[RepairAction] = []

    if "validation" in classes:
        schema_normalization_actions(tool_name, request, actions)
    if "convergence" in classes:
        convergence_actions(tool_name, request, actions)
    if "runtime_exception" in classes and not (codes & {"runner_exception", "tool_convergence_case_failures"}):
        add_action(
            actions,
            RepairAction(
                name="runner_exception_safe_retry",
                priority=87,
                reason="Runner raised an exception; retry once with conservative continuation before escalating to manual log inspection.",
                target_tool=tool_name,
                request_patch=convergence_case_safe_retry_patch(request),
                checklist=[
                    "Preserve the original device geometry and physical models.",
                    "Shrink only bias steps and reuse a nearby stable solution if available.",
                    "If the exception repeats, classify stderr/log tail before further retries.",
                ],
                expected_effect="Separates transient runner/continuation failures from persistent deck or model defects.",
            ),
        )
    quality_issue_actions(tool_name, request, codes, actions)

    if state.get("status") == "failed" and not actions:
        add_action(
            actions,
            RepairAction(
                name="inspect_failed_artifacts",
                priority=50,
                reason="Run failed without a recognized repair signature.",
                target_tool=tool_name,
                request_patch={},
                checklist=[
                    "Open simulator log tail.",
                    "Check missing artifacts and stderr.",
                    "Classify the failure before retrying.",
                ],
                expected_effect="Turns an unknown failure into a classified repair path.",
            ),
        )
    return sorted(actions, key=lambda action: action.priority, reverse=True)


def build_repair_plan(state_path: Path, output_path: Path | None = None) -> RepairPlan:
    actual_output = output_path or default_output_path(state_path)
    try:
        state = read_json(state_path)
        actions = build_repair_actions(state)
        status = RepairPlanStatus.PLANNED if actions else RepairPlanStatus.NO_ACTION
        plan = RepairPlan(
            status=status,
            state_path=str(state_path),
            output_path=str(actual_output),
            tool_name=state.get("tool_name"),
            run_id=state.get("run_id") or state.get("task_id") or state.get("convergence_id"),
            quality_status=quality_status(state),
            failure_classes=failure_classes(state),
            issue_codes=sorted(issue_codes(state)),
            actions=actions,
            next_action=actions[0].name if actions else "no repair action needed",
            created_at=utc_timestamp(),
        )
    except Exception as exc:
        plan = RepairPlan(
            status=RepairPlanStatus.FAILED,
            state_path=str(state_path),
            output_path=str(actual_output),
            created_at=utc_timestamp(),
            failure_reason=str(exc),
        )
    write_json(actual_output, plan.model_dump(mode="json"))
    return plan
