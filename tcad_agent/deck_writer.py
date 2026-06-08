from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.deck_ir import parse_devsim_deck_source, write_semantic_deck_patch_artifacts


class DeckMutation(BaseModel):
    schema_version: str = "actsoft.tcad.deck_mutation.v1"
    name: str
    target: str
    operation: str = "sweep"
    request_path: str
    deck_path: str
    values: list[Any] = Field(default_factory=list)
    units: str | None = None
    reason: str
    source_text: str
    executable: bool = True
    requires_user_confirmation: bool = False
    section: str | None = None
    parameter_kind: str | None = None
    validation_metric_paths: list[str] = Field(default_factory=list)
    expected_tradeoffs: list[str] = Field(default_factory=list)
    safety_constraints: list[str] = Field(default_factory=list)
    next_value_strategy: str = "centered_sweep_then_curve_guided_refine"


class DeckArtifact(BaseModel):
    schema_version: str = "actsoft.tcad.deck_artifact.v1"
    tool_name: str
    deck_kind: str = "python_replay"
    simulator: str = "devsim_or_physics_runner"
    source_goal_text: str | None = None
    generated_at: str
    deck_path: str
    request_path: str
    spec_path: str
    mutations_path: str | None = None
    patch_history_path: str | None = None
    source_ir_path: str | None = None
    semantic_diff_path: str | None = None
    patched_source_path: str | None = None


FIELD_PLATE_KEYWORDS = ["field plate", "field-plate", "field_plate", "场板", "场板结构"]
GUARD_RING_KEYWORDS = ["guard ring", "guard-ring", "guard_ring", "field ring", "termination ring", "保护环", "场限环", "终端环"]
JUNCTION_DEPTH_KEYWORDS = ["junction depth", "junction_depth", "结深", "source drain depth", "source/drain depth", "扩散深度"]
OXIDE_THICKNESS_KEYWORDS = [
    "oxide thickness",
    "gate oxide thickness",
    "oxide_thickness",
    "tox",
    "氧化层厚度",
    "栅氧厚度",
]
IMPLANT_DOSE_KEYWORDS = ["implant dose", "implant_dose", "dose", "注入剂量", "implant dosage", "离子注入剂量"]
TRENCH_RADIUS_KEYWORDS = [
    "trench corner radius",
    "trench radius",
    "corner radius",
    "trench_corner_radius",
    "沟槽圆角",
    "沟槽拐角半径",
]
TRAP_DENSITY_KEYWORDS = ["trap density", "trap_density", "interface trap", "interface traps", "陷阱密度", "界面态密度", "陷阱"]
DRIFT_DOPING_KEYWORDS = [
    "drift doping",
    "drift-region doping",
    "drift region doping",
    "drift_region_doping",
    "drift doping concentration",
    "drift concentration",
    "drift dose",
    "漂移区掺杂",
    "漂移区浓度",
    "漂移区剂量",
]
LIFETIME_KEYWORDS = ["lifetime", "srh", "carrier lifetime", "minority lifetime", "寿命", "复合寿命", "载流子寿命"]
REGION_LIFETIME_KEYWORDS = [
    "region-specific lifetime",
    "region specific lifetime",
    "drift lifetime",
    "drift-region lifetime",
    "body lifetime",
    "局部寿命",
    "区域寿命",
    "漂移区寿命",
]
POWER_DEVICE_CONTEXT = [
    "power mos",
    "power mosfet",
    "ldmos",
    "vdmos",
    "功率mos",
    "功率 mos",
    "功率mosfet",
    "高压",
    "耐压",
    "blocking",
    "ron",
    "r_on",
    "场峰值",
]


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def normalized_text(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ")


def text_has_any(text: str, keywords: list[str]) -> bool:
    lowered = normalized_text(text)
    return any(normalized_text(keyword) in lowered for keyword in keywords)


def goal_prefers_power_device(text: str) -> bool:
    structure_keywords = (
        FIELD_PLATE_KEYWORDS
        + GUARD_RING_KEYWORDS
        + DRIFT_DOPING_KEYWORDS
        + IMPLANT_DOSE_KEYWORDS
        + TRENCH_RADIUS_KEYWORDS
        + REGION_LIFETIME_KEYWORDS
    )
    return text_has_any(text, structure_keywords) or (
        text_has_any(text, POWER_DEVICE_CONTEXT) and text_has_any(text, ["leakage", "漏电", "bv", "breakdown", "击穿"])
    )


def float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def compact_float(value: float) -> float:
    return float(f"{value:.6g}")


def centered_linear_values(current: float, *, low_factor: float, high_factor: float, minimum: float = 0.0) -> list[float]:
    values = [max(current * low_factor, minimum), current, max(current * high_factor, minimum)]
    output: list[float] = []
    for value in values:
        compact = compact_float(value)
        if compact not in output:
            output.append(compact)
    return output if len(output) >= 2 else [compact_float(current), compact_float(current * high_factor)]


def centered_log_values(current: float, *, minimum: float = 1.0e-30) -> list[float]:
    safe = max(current, minimum)
    values = [safe / 10.0, safe, safe * 10.0]
    return [compact_float(value) for value in values]


def mutation_metrics(target: str) -> list[str]:
    if target in {"field_plate", "guard_ring", "trench_corner_radius"}:
        return [
            "quality_report.metrics.max_electric_field_v_per_cm",
            "quality_report.metrics.leakage_current_a",
            "quality_report.metrics.breakdown_voltage_v",
        ]
    if target in {"drift_doping", "implant_dose", "junction_depth"}:
        return [
            "quality_report.metrics.breakdown_voltage_v",
            "quality_report.metrics.specific_on_resistance_ohm_cm2",
            "quality_report.metrics.max_electric_field_v_per_cm",
        ]
    if target in {"lifetime", "region_lifetime", "trap_density"}:
        return [
            "quality_report.metrics.leakage_current_a",
            "quality_report.metrics.leakage_abs_current_at_target_a",
            "quality_report.metrics.breakdown_voltage_v",
        ]
    if target == "oxide_thickness":
        return [
            "quality_report.metrics.max_electric_field_v_per_cm",
            "quality_report.metrics.gate_capacitance_f",
            "quality_report.metrics.vth_at_threshold_current_v",
        ]
    return ["quality_report.metrics.leakage_current_a"]


def mutation_tradeoffs(target: str) -> list[str]:
    mapping = {
        "field_plate": ["May reduce field peak but increase capacitance or shift termination field crowding."],
        "guard_ring": ["May improve termination BV but consume layout area and add termination leakage sensitivity."],
        "trench_corner_radius": ["May reduce corner field but changes cell pitch and etch/process assumptions."],
        "drift_doping": ["Lower doping can improve BV but usually worsens Ron; higher doping does the inverse."],
        "implant_dose": ["Dose changes can alter junction field and leakage while shifting threshold/body effects."],
        "junction_depth": ["Deeper junctions may reduce surface field but increase capacitance or punch-through risk."],
        "lifetime": ["Longer lifetime can reduce SRH leakage but may hurt switching or stored charge assumptions."],
        "region_lifetime": ["Local lifetime changes can improve leakage in one region while moving recombination elsewhere."],
        "trap_density": ["Trap density can explain leakage/current collapse but must not be tuned without measured evidence."],
        "oxide_thickness": ["Oxide changes affect field coupling, capacitance, threshold, and reliability margins."],
    }
    return mapping.get(target, [])


def mutation_constraints(target: str) -> list[str]:
    common = [
        "Do not accept leakage improvement if BV magnitude regresses by more than 10%.",
        "Do not accept field-peak improvement if Ron regresses by more than 20% without an explicit tradeoff goal.",
    ]
    if target in {"trap_density", "region_lifetime"}:
        return [*common, "Require measured/golden correlation before treating model-only leakage tuning as signoff evidence."]
    if target in {"oxide_thickness", "trench_corner_radius", "guard_ring"}:
        return [*common, "Require geometry/process confirmation before applying this patch to a proprietary deck."]
    return common


def deck_mutation(
    *,
    name: str,
    target: str,
    request_path: str,
    deck_path: str,
    values: list[Any],
    units: str | None,
    reason: str,
    source_text: str,
    section: str,
    parameter_kind: str,
    requires_user_confirmation: bool = False,
) -> DeckMutation:
    return DeckMutation(
        name=name,
        target=target,
        request_path=request_path,
        deck_path=deck_path,
        values=values,
        units=units,
        reason=reason,
        source_text=source_text,
        requires_user_confirmation=requires_user_confirmation,
        section=section,
        parameter_kind=parameter_kind,
        validation_metric_paths=mutation_metrics(target),
        expected_tradeoffs=mutation_tradeoffs(target),
        safety_constraints=mutation_constraints(target),
    )


def planned_power_device_request(goal_text: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
    base = dict(request or {})
    base.setdefault("device_type", "power_mosfet_bv_ron")
    base.setdefault("fidelity", "physics_1d")
    base.setdefault("evidence_level", "tcad_executable")
    base.setdefault("start", 0.0)
    base.setdefault("stop", -90.0)
    base.setdefault("step", 5.0)
    return base


def metric_path_for_goal(goal_text: str) -> str:
    if text_has_any(goal_text, ["leakage", "漏电", "暗电流"]):
        return "quality_report.metrics.leakage_current_a"
    if text_has_any(goal_text, ["ron", "r_on", "导通电阻"]):
        return "quality_report.metrics.specific_on_resistance_ohm_cm2"
    if text_has_any(goal_text, ["field", "场峰值", "电场"]):
        return "quality_report.metrics.max_electric_field_v_per_cm"
    if text_has_any(goal_text, ["bv", "breakdown", "击穿", "耐压"]):
        return "quality_report.metrics.breakdown_voltage_v"
    return "quality_report.metrics.leakage_current_a"


def plan_deck_mutations(goal_text: str, tool_name: str | None, request: dict[str, Any] | None = None) -> list[DeckMutation]:
    request_data = dict(request or {})
    device_type = str(request_data.get("device_type") or "")
    power_context = (
        tool_name == "extended_device_sweep"
        and device_type in {"", "power_mosfet_bv_ron"}
    ) or goal_prefers_power_device(goal_text)
    mutations: list[DeckMutation] = []
    if power_context and text_has_any(goal_text, FIELD_PLATE_KEYWORDS):
        current = float_or_none(request_data.get("power_mos_field_plate_length_um")) or 1.5
        mutations.append(
            deck_mutation(
                name="sweep_field_plate_length",
                target="field_plate",
                request_path="power_mos_field_plate_length_um",
                deck_path="geometry.field_plate_length_um",
                values=centered_linear_values(current, low_factor=0.6, high_factor=1.4, minimum=0.0),
                units="um",
                reason="User asked to vary field plate geometry to inspect leakage, field peak, or BV sensitivity.",
                source_text=goal_text,
                section="geometry",
                parameter_kind="termination_geometry",
            )
        )
    if power_context and text_has_any(goal_text, GUARD_RING_KEYWORDS):
        current = float_or_none(request_data.get("power_mos_guard_ring_spacing_um")) or 1.0
        mutations.append(
            deck_mutation(
                name="sweep_guard_ring_spacing",
                target="guard_ring",
                request_path="power_mos_guard_ring_spacing_um",
                deck_path="geometry.guard_ring_spacing_um",
                values=centered_linear_values(current, low_factor=0.7, high_factor=1.4, minimum=0.05),
                units="um",
                reason="User asked to vary guard-ring termination geometry to inspect BV, leakage, and field crowding.",
                source_text=goal_text,
                section="geometry",
                parameter_kind="termination_geometry",
                requires_user_confirmation=True,
            )
        )
    if text_has_any(goal_text, JUNCTION_DEPTH_KEYWORDS):
        if power_context:
            current = float_or_none(request_data.get("power_mos_junction_depth_um")) or 0.35
            request_path = "power_mos_junction_depth_um"
            deck_path = "geometry.junction_depth_um"
        elif tool_name == "mosfet_2d_id_sweep":
            current = float_or_none(request_data.get("source_drain_depth_um")) or 0.08
            request_path = "source_drain_depth_um"
            deck_path = "geometry.source_drain_depth_um"
        else:
            current = float_or_none(request_data.get("junction_um")) or 0.05
            request_path = "junction_um"
            deck_path = "geometry.junction_um"
        mutations.append(
            deck_mutation(
                name="sweep_junction_depth",
                target="junction_depth",
                request_path=request_path,
                deck_path=deck_path,
                values=centered_linear_values(current, low_factor=0.75, high_factor=1.35, minimum=0.001),
                units="um",
                reason="User asked to vary junction depth to inspect field crowding, leakage, or threshold sensitivity.",
                source_text=goal_text,
                section="geometry",
                parameter_kind="junction_geometry",
                requires_user_confirmation=True,
            )
        )
    if text_has_any(goal_text, OXIDE_THICKNESS_KEYWORDS):
        if power_context:
            current = float_or_none(request_data.get("power_mos_gate_oxide_thickness_nm")) or 50.0
            request_path = "power_mos_gate_oxide_thickness_nm"
            deck_path = "geometry.gate_oxide_thickness_nm"
        else:
            current = float_or_none(request_data.get("oxide_thickness_nm")) or 10.0
            request_path = "oxide_thickness_nm"
            deck_path = "geometry.oxide_thickness_nm"
        mutations.append(
            deck_mutation(
                name="sweep_oxide_thickness",
                target="oxide_thickness",
                request_path=request_path,
                deck_path=deck_path,
                values=centered_linear_values(current, low_factor=0.8, high_factor=1.25, minimum=0.1),
                units="nm",
                reason="User asked to vary oxide thickness to inspect field coupling, capacitance, or threshold tradeoffs.",
                source_text=goal_text,
                section="geometry",
                parameter_kind="dielectric_geometry",
                requires_user_confirmation=True,
            )
        )
    if power_context and text_has_any(goal_text, IMPLANT_DOSE_KEYWORDS):
        current = float_or_none(request_data.get("power_mos_implant_dose_cm2")) or 1.0e13
        mutations.append(
            deck_mutation(
                name="sweep_implant_dose",
                target="implant_dose",
                request_path="power_mos_implant_dose_cm2",
                deck_path="doping.implant_dose_cm2",
                values=centered_log_values(current, minimum=1.0e10),
                units="cm^-2",
                reason="User asked to vary implant dose to inspect junction field, leakage, and Ron/BV sensitivity.",
                source_text=goal_text,
                section="doping",
                parameter_kind="process_dose",
                requires_user_confirmation=True,
            )
        )
    if power_context and text_has_any(goal_text, TRENCH_RADIUS_KEYWORDS):
        current = float_or_none(request_data.get("power_mos_trench_corner_radius_um")) or 0.08
        mutations.append(
            deck_mutation(
                name="sweep_trench_corner_radius",
                target="trench_corner_radius",
                request_path="power_mos_trench_corner_radius_um",
                deck_path="geometry.trench_corner_radius_um",
                values=centered_linear_values(current, low_factor=0.5, high_factor=1.75, minimum=0.005),
                units="um",
                reason="User asked to vary trench corner radius to inspect local electric-field crowding.",
                source_text=goal_text,
                section="geometry",
                parameter_kind="corner_geometry",
                requires_user_confirmation=True,
            )
        )
    if power_context and text_has_any(goal_text, DRIFT_DOPING_KEYWORDS):
        current = float_or_none(request_data.get("power_mos_drift_region_doping_cm3")) or 1.0e16
        mutations.append(
            deck_mutation(
                name="sweep_drift_region_doping",
                target="drift_doping",
                request_path="power_mos_drift_region_doping_cm3",
                deck_path="doping.drift_region_doping_cm3",
                values=centered_log_values(current, minimum=1.0e12),
                units="cm^-3",
                reason="User asked to vary drift-region doping to inspect leakage, BV, field, or Ron tradeoffs.",
                source_text=goal_text,
                section="doping",
                parameter_kind="region_doping",
            )
        )
    if text_has_any(goal_text, TRAP_DENSITY_KEYWORDS):
        if power_context:
            current = float_or_none(request_data.get("power_mos_trap_density_cm2")) or 1.0e11
            request_path = "power_mos_trap_density_cm2"
            deck_path = "physics_models.trap_density_cm2"
        elif device_type == "gan_hemt_id_bv":
            current = float_or_none(request_data.get("gan_trap_density_cm2")) or 5.0e12
            request_path = "gan_trap_density_cm2"
            deck_path = "physics_models.trap_density_cm2"
        else:
            current = float_or_none(request_data.get("interface_trap_density_cm2")) or 1.0e11
            request_path = "interface_trap_density_cm2"
            deck_path = "physics_models.interface_trap_density_cm2"
        mutations.append(
            deck_mutation(
                name="sweep_trap_density",
                target="trap_density",
                request_path=request_path,
                deck_path=deck_path,
                values=centered_log_values(current, minimum=1.0e6),
                units="cm^-2",
                reason="User asked to vary trap density to inspect leakage, current collapse, or model-coupling sensitivity.",
                source_text=goal_text,
                section="model",
                parameter_kind="trap_model",
                requires_user_confirmation=True,
            )
        )
    if power_context and text_has_any(goal_text, REGION_LIFETIME_KEYWORDS):
        current = float_or_none(request_data.get("power_mos_drift_region_lifetime_s")) or float_or_none(
            request_data.get("power_mos_carrier_lifetime_s")
        ) or 1.0e-6
        mutations.append(
            deck_mutation(
                name="sweep_drift_region_lifetime",
                target="region_lifetime",
                request_path="power_mos_drift_region_lifetime_s",
                deck_path="physics_models.regions.drift.carrier_lifetime_s",
                values=centered_log_values(current, minimum=1.0e-12),
                units="s",
                reason="User asked to vary a region-specific carrier lifetime instead of a global lifetime.",
                source_text=goal_text,
                section="model",
                parameter_kind="region_lifetime",
            )
        )
    if text_has_any(goal_text, LIFETIME_KEYWORDS):
        if power_context:
            current = float_or_none(request_data.get("power_mos_carrier_lifetime_s")) or 1.0e-6
            mutations.append(
                deck_mutation(
                    name="sweep_power_carrier_lifetime",
                    target="lifetime",
                    request_path="power_mos_carrier_lifetime_s",
                    deck_path="physics_models.carrier_lifetime_s",
                    values=centered_log_values(current, minimum=1.0e-12),
                    units="s",
                    reason="User asked to vary carrier lifetime to inspect leakage sensitivity.",
                    source_text=goal_text,
                    section="model",
                    parameter_kind="global_lifetime",
                )
            )
        else:
            current = float_or_none(request_data.get("electron_lifetime_s")) or 1.0e-8
            mutations.append(
                deck_mutation(
                    name="sweep_srh_lifetime",
                    target="lifetime",
                    request_path="electron_lifetime_s",
                    deck_path="physics_models.electron_lifetime_s",
                    values=centered_log_values(current, minimum=1.0e-12),
                    units="s",
                    reason="User asked to vary SRH lifetime to inspect leakage sensitivity.",
                    source_text=goal_text,
                    section="model",
                    parameter_kind="global_lifetime",
                )
            )
    return mutations


def deck_mutation_convergence_requests(
    goal_text: str,
    tool_name: str | None,
    request: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    base_request = planned_power_device_request(goal_text, request) if goal_prefers_power_device(goal_text) else dict(request or {})
    actual_tool = tool_name or ("extended_device_sweep" if goal_prefers_power_device(goal_text) else None)
    if not actual_tool:
        return []
    requests: list[dict[str, Any]] = []
    mutations = plan_deck_mutations(goal_text, actual_tool, base_request)
    mutation_dicts = [mutation.model_dump(mode="json") for mutation in mutations]
    if mutation_dicts:
        base_request.setdefault("tcad_deck_mutations", mutation_dicts)
    for mutation in mutations:
        requests.append(
            {
                "tool_name": actual_tool,
                "base_request": dict(base_request),
                "axis_path": mutation.request_path,
                "values": mutation.values,
                "metric_path": metric_path_for_goal(goal_text),
                "relative_tolerance": 0.25 if mutation.target != "lifetime" else 1.0,
                "deck_mutation": mutation.model_dump(mode="json"),
            }
        )
    return requests


def generated_deck_source(tool_name: str) -> str:
    imports = {
        "pn_junction_iv_sweep": ("tcad_agent.tools.pn_junction_iv", "PNJunctionIVRequest", "run_pn_junction_iv_sweep"),
        "mos_capacitor_cv_sweep": ("tcad_agent.tools.mos_capacitor_cv", "MOSCapacitorCVRequest", "run_mos_capacitor_cv_sweep"),
        "diode_breakdown_leakage_sweep": ("tcad_agent.tools.diode_breakdown", "DiodeBreakdownRequest", "run_diode_breakdown_sweep"),
        "mosfet_2d_id_sweep": ("tcad_agent.tools.mosfet_2d_id", "MOSFET2DIDRequest", "run_mosfet_2d_id_sweep"),
        "extended_device_sweep": ("tcad_agent.tools.extended_device_sweep", "ExtendedDeviceRequest", "run_extended_device_sweep"),
    }
    module, request_model, runner = imports.get(tool_name, imports["extended_device_sweep"])
    return f'''from __future__ import annotations

import json
from pathlib import Path

from {module} import {request_model}, {runner}


def main() -> None:
    request_path = Path(__file__).with_name("deck_request.json")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    result = {runner}({request_model}.model_validate(request))
    if hasattr(result, "model_dump"):
        result = result.model_dump(mode="json")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''


def write_deck_artifacts(
    run_dir: Path,
    *,
    tool_name: str,
    request: dict[str, Any],
    deck_spec: dict[str, Any] | None,
    mutations: list[dict[str, Any]] | None = None,
    source_goal_text: str | None = None,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    request_path = run_dir / "deck_request.json"
    spec_path = run_dir / "tcad_deck_spec.json"
    deck_path = run_dir / "generated_deck.py"
    mutations_path = run_dir / "tcad_deck_mutations.json"
    patch_history_path = run_dir / "deck_patch_history.json"

    request_path.write_text(json.dumps(request, indent=2, ensure_ascii=False), encoding="utf-8")
    spec_path.write_text(json.dumps(deck_spec or {}, indent=2, ensure_ascii=False), encoding="utf-8")
    deck_path.write_text(generated_deck_source(tool_name), encoding="utf-8")
    actual_mutations = list(mutations or [])
    if actual_mutations:
        mutations_path.write_text(json.dumps(actual_mutations, indent=2, ensure_ascii=False), encoding="utf-8")
    actual_deck_patch_history = request.get("deck_patch_history") if isinstance(request.get("deck_patch_history"), list) else []
    patch_history = {
        "schema_version": "actsoft.tcad.deck_patch_history.v1",
        "tool_name": tool_name,
        "generated_at": utc_timestamp(),
        "source_goal_text": source_goal_text,
        "mutations": actual_mutations,
        "applied_patches": actual_deck_patch_history,
    }
    patch_history_path.write_text(json.dumps(patch_history, indent=2, ensure_ascii=False), encoding="utf-8")
    source_ir_path = run_dir / "tcad_deck_ir.json"
    generated_ir = parse_devsim_deck_source(deck_path.read_text(encoding="utf-8"), source_path=str(deck_path.resolve()))
    source_ir_path.write_text(json.dumps(generated_ir.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    semantic_diff_path: Path | None = None
    patched_source_path: Path | None = None
    source_deck_path_value = request.get("source_deck_path") or request.get("deck_source_path")
    if source_deck_path_value and actual_deck_patch_history:
        source_deck_path = Path(str(source_deck_path_value))
        if source_deck_path.exists():
            semantic_result = write_semantic_deck_patch_artifacts(source_deck_path, actual_deck_patch_history, run_dir / "semantic_deck_patch")
            semantic_diff_path = Path(str(semantic_result.diff_path)) if semantic_result.diff_path else None
            patched_source_path = Path(str(semantic_result.patched_source_path)) if semantic_result.patched_source_path else None
    artifact = DeckArtifact(
        tool_name=tool_name,
        source_goal_text=source_goal_text,
        generated_at=utc_timestamp(),
        deck_path=str(deck_path.resolve()),
        request_path=str(request_path.resolve()),
        spec_path=str(spec_path.resolve()),
        mutations_path=str(mutations_path.resolve()) if actual_mutations else None,
        patch_history_path=str(patch_history_path.resolve()),
        source_ir_path=str(source_ir_path.resolve()),
        semantic_diff_path=str(semantic_diff_path.resolve()) if semantic_diff_path else None,
        patched_source_path=str(patched_source_path.resolve()) if patched_source_path else None,
    )
    artifact_path = run_dir / "tcad_deck_artifact.json"
    artifact_path.write_text(json.dumps(artifact.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    output = {
        "generated_deck": str(deck_path.resolve()),
        "deck_request": str(request_path.resolve()),
        "tcad_deck_spec": str(spec_path.resolve()),
        "deck_patch_history": str(patch_history_path.resolve()),
        "tcad_deck_artifact": str(artifact_path.resolve()),
        "tcad_deck_ir": str(source_ir_path.resolve()),
    }
    if actual_mutations:
        output["tcad_deck_mutations"] = str(mutations_path.resolve())
    if semantic_diff_path:
        output["semantic_deck_diff"] = str(semantic_diff_path.resolve())
    if patched_source_path:
        output["patched_source_deck"] = str(patched_source_path.resolve())
    return output
