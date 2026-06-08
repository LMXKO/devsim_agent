from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.deck_writer import plan_deck_mutations
from tcad_agent.device_templates import RouteStatus, route_device_goal
from tcad_agent.engineering_intent import DeviceSupport, EngineeringIntent, parse_engineering_intent


NUMBER_RE = r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"


class TCADSpec(BaseModel):
    source_text: str
    device_family: str = "unknown"
    template_id: str | None = None
    support: str = "unknown"
    execution_profile: str = "needs_clarification"
    tcad_fidelity: str | None = None
    signoff_workflow: list[str] = Field(default_factory=list)
    suggested_tool: str | None = None
    analyses: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    spec_limits: dict[str, Any] = Field(default_factory=dict)
    geometry: dict[str, Any] = Field(default_factory=dict)
    materials: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, Any] = Field(default_factory=dict)
    bias: dict[str, Any] = Field(default_factory=dict)
    corner_plan: dict[str, Any] = Field(default_factory=dict)
    calibration: dict[str, Any] = Field(default_factory=dict)
    deliverables: list[str] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)
    request_hint: dict[str, Any] = Field(default_factory=dict)
    deck_mutations: list[dict[str, Any]] = Field(default_factory=list)
    signoff_required: bool = False
    measured_or_golden_reference: str | None = None
    capability_warnings: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)


def float_value(text: str) -> float:
    return float(text)


def text_has_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def extract_first(patterns: list[str], text: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float_value(match.group(1))
    return None


def extract_geometry(text: str) -> dict[str, Any]:
    geometry: dict[str, Any] = {}
    values = {
        "oxide_thickness_nm": [
            rf"(?:tox|oxide thickness|氧化层厚度|栅氧厚度)\s*(?:=|:|为)?\s*({NUMBER_RE})\s*nm",
            rf"({NUMBER_RE})\s*nm\s*(?:tox|oxide)",
        ],
        "length_um": [
            rf"(?:gate length|channel length|length|沟道长度|栅长|器件长度|L)\s*(?:=|:|为)?\s*({NUMBER_RE})\s*um",
            rf"({NUMBER_RE})\s*um\s*(?:gate length|channel length|栅长|沟道)",
        ],
        "silicon_thickness_um": [
            rf"(?:silicon thickness|si thickness|硅厚度|硅膜厚度)\s*(?:=|:|为)?\s*({NUMBER_RE})\s*um",
        ],
        "junction_um": [
            rf"(?:junction|结位置)\s*(?:=|:|为)?\s*({NUMBER_RE})\s*um",
        ],
    }
    for key, patterns in values.items():
        value = extract_first(patterns, text)
        if value is not None:
            geometry[key] = value
    return geometry


def extract_materials(text: str) -> dict[str, Any]:
    lowered = text.lower()
    materials: dict[str, Any] = {}
    if "sic" in lowered or "碳化硅" in text:
        materials["semiconductor"] = "SiC"
    elif "gan" in lowered or "氮化镓" in text:
        materials["semiconductor"] = "GaN"
    elif "si" in lowered or "硅" in text:
        materials["semiconductor"] = "Si"
    if "sio2" in lowered or "氧化硅" in text:
        materials["oxide"] = "SiO2"
    return materials


def extract_models(text: str) -> dict[str, Any]:
    models: dict[str, Any] = {}
    lowered = text.lower()
    if "srh" in lowered or "寿命" in text:
        models["recombination_model"] = "srh"
    if "impact ionization" in lowered or "avalanche" in lowered or "碰撞电离" in text or "雪崩" in text:
        models["impact_ionization_model"] = "requested"
    if "constant mobility" in lowered:
        models["mobility_model"] = "constant"
    elif "doping-dependent" in lowered or "doping dependent" in lowered or "迁移率" in text:
        models["mobility_model"] = "doping_dependent"
    interface_trap = extract_first(
        [rf"(?:interface trap|dit|界面态|界面陷阱)\s*(?:=|:|为)?\s*({NUMBER_RE})"],
        text,
    )
    if interface_trap is not None:
        models["interface_trap_density_cm2"] = interface_trap
    fixed_charge = extract_first(
        [rf"(?:fixed charge|fixed oxide charge|qf|固定电荷)\s*(?:=|:|为)?\s*({NUMBER_RE})"],
        text,
    )
    if fixed_charge is not None:
        models["fixed_oxide_charge_cm2"] = fixed_charge
    if "field plate" in lowered or "field-plate" in lowered or "场板" in text:
        models["field_plate_edit_requested"] = True
    if "drift doping" in lowered or "drift region doping" in lowered or "漂移区掺杂" in text or "漂移区浓度" in text:
        models["drift_doping_edit_requested"] = True
    return models


def constraint_dict(intent: EngineeringIntent) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for item in intent.constraints:
        if "=" not in item:
            output.setdefault("flags", []).append(item)
            continue
        key, value = item.split("=", 1)
        output[key] = value
    return output


def scale_current_amp(value: float, unit: str | None) -> float:
    normalized = (unit or "a").lower().replace("μ", "u")
    return value * {
        "a": 1.0,
        "ma": 1e-3,
        "ua": 1e-6,
        "na": 1e-9,
        "pa": 1e-12,
    }.get(normalized, 1.0)


def extract_spec_limits(text: str, intent: EngineeringIntent) -> dict[str, Any]:
    limits: dict[str, Any] = {}
    leakage = re.search(
        rf"(?:leakage|漏电|暗电流)[^\n。；;，,]{{0,40}}?(?:<=|≤|<|不超过|低于|小于|limit|上限)?\s*({NUMBER_RE})\s*(a|ma|ua|μa|na|pa)",
        text,
        flags=re.IGNORECASE,
    )
    if leakage:
        limits["leakage_current_max_a"] = scale_current_amp(float(leakage.group(1)), leakage.group(2))
    ion = re.search(
        rf"(?:ion/ioff|开关比)[^\n。；;，,]{{0,40}}?(?:>=|≥|至少|大于|超过)?\s*({NUMBER_RE})",
        text,
        flags=re.IGNORECASE,
    )
    if ion:
        limits["ion_ioff_min"] = float(ion.group(1))
    bv = re.search(
        rf"(?:bv|breakdown|耐压|击穿)[^\n。；;，,]{{0,40}}?(?:>=|≥|至少|大于|超过)?\s*({NUMBER_RE})\s*(?:v|伏|伏特)",
        text,
        flags=re.IGNORECASE,
    )
    if bv:
        limits["breakdown_voltage_min_abs_v"] = abs(float(bv.group(1)))
    vth_window = re.search(
        rf"(?:vth|threshold|阈值)[^\n。；;，,]{{0,40}}?({NUMBER_RE})\s*(?:v|伏|伏特)?\s*(?:-|~|到|至)\s*({NUMBER_RE})\s*(?:v|伏|伏特)",
        text,
        flags=re.IGNORECASE,
    )
    if vth_window:
        limits["vth_window_v"] = [float(vth_window.group(1)), float(vth_window.group(2))]
    for raw in intent.constraints:
        if raw.startswith("leakage_current_limit=") and "leakage_current_max_a" not in limits:
            match = re.search(rf"=({NUMBER_RE})(a|ma|ua|μa|na|pa)$", raw, flags=re.IGNORECASE)
            if match:
                limits["leakage_current_max_a"] = scale_current_amp(float(match.group(1)), match.group(2))
        elif raw.startswith("ion_ioff_min=") and "ion_ioff_min" not in limits:
            limits["ion_ioff_min"] = float(raw.split("=", 1)[1])
        elif raw.startswith("breakdown_voltage_min=") and "breakdown_voltage_min_abs_v" not in limits:
            match = re.search(rf"=({NUMBER_RE})", raw)
            if match:
                limits["breakdown_voltage_min_abs_v"] = abs(float(match.group(1)))
    return limits


def numbers_after_label(text: str, labels: str, *, max_chars: int = 80) -> list[float]:
    match = re.search(rf"(?:{labels})[^\n。；;，,]{{0,{max_chars}}}", text, flags=re.IGNORECASE)
    if not match:
        return []
    return [float(value) for value in re.findall(NUMBER_RE, match.group(0), flags=re.IGNORECASE)]


def extract_bias_details(text: str, intent: EngineeringIntent) -> dict[str, Any]:
    bias = dict(intent.sweep_hints)
    drain_values = numbers_after_label(text, r"vd|drain|漏压|漏极")
    gate_values = numbers_after_label(text, r"vg|gate|栅压|栅极")
    if len(drain_values) >= 2:
        bias["drain_voltage_values_v"] = drain_values[:6]
    if len(gate_values) >= 2:
        bias.setdefault("gate_voltage_values_v", gate_values[:8])
    return bias


def extract_corner_plan(text: str, intent: EngineeringIntent) -> dict[str, Any]:
    plan: dict[str, Any] = {}
    temperatures_k = list(intent.sweep_hints.get("temperature_values_k") or [])
    for celsius in re.findall(rf"({NUMBER_RE})\s*(?:c|℃)", text, flags=re.IGNORECASE):
        temperatures_k.append(float(celsius) + 273.15)
    if temperatures_k:
        plan["temperature_values_k"] = temperatures_k[:8]
    if any(key in intent.evidence_requirements for key in ["mesh_convergence", "model_ab"]):
        plan["verification_axes"] = [
            key
            for key in ["mesh_convergence", "model_ab", "curve_shape", "unit_check", "golden_or_measured"]
            if key in intent.evidence_requirements
        ]
    if "dibl" in intent.metrics:
        plan.setdefault("bias_splits", []).append("low_high_drain_idvg")
    return plan


def extract_deliverables(text: str, intent: EngineeringIntent) -> list[str]:
    deliverables = ["state_json", "quality_report"]
    if intent.suggested_tool:
        deliverables.extend(["curve_csv", "curve_plot"])
    if intent.metrics:
        deliverables.append("extracted_metrics")
    if intent.evidence_requirements:
        deliverables.append("physical_benchmark")
    if text_has_any(text, ["conclusion", "report", "总结", "结论", "解释", "建议", "交付"]):
        deliverables.append("engineering_conclusion")
    if "golden_or_measured" in intent.evidence_requirements:
        deliverables.append("golden_or_measured_comparison")
    return list(dict.fromkeys(deliverables))


def calibration_plan(reference: str | None, intent: EngineeringIntent) -> dict[str, Any]:
    plan: dict[str, Any] = {}
    if reference:
        plan["reference_path"] = reference
    if "calibration" in intent.analyses or "golden_or_measured" in intent.evidence_requirements:
        plan["required"] = True
        plan["metrics"] = intent.metrics or ["curve_rmse"]
    return plan


def reference_path(text: str) -> str | None:
    match = re.search(
        r"(?:measured[-_ ]?curve|target[-_ ]?curve|golden[-_ ]?curve|实测曲线|目标曲线|可信曲线)\s*[:=：]?\s*([^\s,，;；]+(?:\.csv|\.json|\.txt|\.dat))",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip("\"'，,。；;") if match else None


def execution_profile(intent: EngineeringIntent) -> str:
    if intent.support == DeviceSupport.EXECUTABLE:
        return "tcad_signoff_candidate" if intent.evidence_policy == "requires_signoff_evidence" else "tcad_executable"
    if intent.support == DeviceSupport.COMPACT_BASELINE:
        return "compact_planning_baseline"
    if intent.support == DeviceSupport.PLANNED:
        return "runner_implementation_required"
    return "needs_clarification"


def missing_inputs(intent: EngineeringIntent, signoff_required: bool, reference: str | None) -> list[str]:
    missing: list[str] = []
    if intent.device_family == "unknown":
        missing.append("device_family")
    if not intent.analyses:
        missing.append("analysis_type")
    if not intent.metrics and signoff_required:
        missing.append("target_metrics")
    if signoff_required and not reference and "golden_or_measured" in intent.evidence_requirements:
        missing.append("measured_or_golden_reference")
    if intent.support == DeviceSupport.PLANNED:
        missing.append("tcad_runner")
        missing.append("quality_rules")
        missing.append("physical_benchmark_rules")
    return missing


def parse_tcad_spec(text: str) -> TCADSpec:
    intent = parse_engineering_intent(text)
    route = route_device_goal(text)
    reference = reference_path(text)
    signoff_required = intent.evidence_policy == "requires_signoff_evidence" or "engineering_signoff" in intent.evidence_requirements
    capability_warnings = list(intent.capability_warnings)
    if route.status == RouteStatus.MATCHED:
        capability_warnings.extend(route.capability_warnings)
    deck_mutations = [
        mutation.model_dump(mode="json")
        for mutation in plan_deck_mutations(text, intent.suggested_tool, intent.request_hint)
    ]
    return TCADSpec(
        source_text=text,
        device_family=intent.device_family,
        template_id=intent.template_id,
        support=intent.support.value,
        execution_profile=execution_profile(intent),
        tcad_fidelity=route.tcad_fidelity if route.status == RouteStatus.MATCHED else None,
        signoff_workflow=route.signoff_workflow if route.status == RouteStatus.MATCHED else [],
        suggested_tool=intent.suggested_tool,
        analyses=intent.analyses,
        metrics=intent.metrics,
        objectives=intent.objectives,
        constraints=constraint_dict(intent),
        spec_limits=extract_spec_limits(text, intent),
        geometry=extract_geometry(text),
        materials=extract_materials(text),
        models=extract_models(text),
        bias=extract_bias_details(text, intent),
        corner_plan=extract_corner_plan(text, intent),
        calibration=calibration_plan(reference, intent),
        deliverables=extract_deliverables(text, intent),
        evidence_requirements=intent.evidence_requirements,
        request_hint=intent.request_hint,
        deck_mutations=deck_mutations,
        signoff_required=signoff_required,
        measured_or_golden_reference=reference,
        capability_warnings=capability_warnings,
        assumptions=intent.assumptions,
        missing_inputs=missing_inputs(intent, signoff_required, reference),
        clarification_questions=intent.clarification_questions,
    )
