from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TemplateSupport(str, Enum):
    EXECUTABLE = "executable"
    PLANNED = "planned"


class RouteStatus(str, Enum):
    MATCHED = "matched"
    UNMATCHED = "unmatched"


class DeviceTaskTemplate(BaseModel):
    template_id: str
    display_name: str
    aliases: list[str]
    tasks: list[str]
    support: TemplateSupport
    executable_tool: str | None = None
    default_request: dict[str, Any] = Field(default_factory=dict)
    benchmark_metrics: list[str] = Field(default_factory=list)
    missing_capabilities: list[str] = Field(default_factory=list)
    next_implementation_steps: list[str] = Field(default_factory=list)


class DeviceRouteResult(BaseModel):
    status: RouteStatus
    goal_text: str
    template: DeviceTaskTemplate | None = None
    matched_alias: str | None = None
    executable: bool = False
    suggested_tool: str | None = None
    request_hint: dict[str, Any] = Field(default_factory=dict)
    message: str


def device_templates() -> list[DeviceTaskTemplate]:
    return [
        DeviceTaskTemplate(
            template_id="pn_junction_iv",
            display_name="PN Junction IV",
            aliases=["pn junction", "pn结", "p-n", "pn iv"],
            tasks=["forward/reverse IV", "ideality factor", "rectification ratio"],
            support=TemplateSupport.EXECUTABLE,
            executable_tool="pn_junction_iv_sweep",
            default_request={"start": 0.0, "stop": 0.5, "step": 0.1},
            benchmark_metrics=["ideality_factor_estimate", "rectification_ratio_final_to_leakage"],
        ),
        DeviceTaskTemplate(
            template_id="mos_capacitor_cv",
            display_name="MOS Capacitor C-V",
            aliases=["mos capacitor", "moscap", "mos c-v", "mos cv", "mos 电容"],
            tasks=["C-V", "oxide capacitance benchmark", "oxide/doping sweep"],
            support=TemplateSupport.EXECUTABLE,
            executable_tool="mos_capacitor_cv_sweep",
            default_request={"start": -1.0, "stop": 1.0, "step": 0.25, "oxide_thickness_nm": 5.0},
            benchmark_metrics=["max_capacitance_f_per_cm2", "oxide_capacitance_estimate_f_per_cm2"],
        ),
        DeviceTaskTemplate(
            template_id="diode_breakdown_leakage",
            display_name="Diode Breakdown / Leakage",
            aliases=["breakdown diode", "diode breakdown", "pn breakdown", "击穿", "漏电"],
            tasks=["reverse leakage", "breakdown voltage", "reverse current shape"],
            support=TemplateSupport.EXECUTABLE,
            executable_tool="diode_breakdown_leakage_sweep",
            default_request={"start": 0.0, "stop": -5.0, "step": 0.5, "breakdown_current_a": 1e-6},
            benchmark_metrics=["leakage_abs_current_at_target_a", "breakdown_voltage_at_threshold_v"],
        ),
        DeviceTaskTemplate(
            template_id="mosfet_2d_id",
            display_name="2D MOSFET Id-Vg / Id-Vd",
            aliases=["mosfet", "id-vg", "idvg", "id-vd", "idvd", "转移特性", "输出特性"],
            tasks=["Id-Vg", "Id-Vd", "Vth", "SS", "Ion/Ioff", "gm"],
            support=TemplateSupport.EXECUTABLE,
            executable_tool="mosfet_2d_id_sweep",
            default_request={"sweep_type": "both", "gate_start": 0.0, "gate_stop": 1.0, "gate_step": 0.25},
            benchmark_metrics=["vth_at_threshold_current_v", "subthreshold_swing_mv_dec", "ion_ioff_ratio"],
        ),
        DeviceTaskTemplate(
            template_id="schottky_diode",
            display_name="Schottky Diode",
            aliases=["schottky", "肖特基"],
            tasks=["forward IV", "reverse leakage", "barrier-height extraction", "C-V"],
            support=TemplateSupport.EXECUTABLE,
            executable_tool="extended_device_sweep",
            default_request={"device_type": "schottky_diode", "start": -0.5, "stop": 0.8, "step": 0.1},
            benchmark_metrics=["barrier_height_ev", "ideality_factor_estimate", "reverse_leakage_current_a"],
            next_implementation_steps=[
                "Calibrate the residual-coupled thermionic contact against trusted Schottky IV/C-V references.",
                "Add C-V extraction after the IV path is validated.",
            ],
        ),
        DeviceTaskTemplate(
            template_id="bjt_gummel_output",
            display_name="BJT Gummel / Output",
            aliases=["bjt", "bipolar", "gummel", "双极", "晶体管"],
            tasks=["Gummel plot", "Ic-Vce", "current gain beta", "Early voltage"],
            support=TemplateSupport.EXECUTABLE,
            executable_tool="extended_device_sweep",
            default_request={"device_type": "bjt_gummel_output", "start": 0.55, "stop": 0.8, "step": 0.025},
            benchmark_metrics=["current_gain_beta", "early_voltage_v", "collector_leakage_current_a"],
            next_implementation_steps=[
                "Replace compact Gummel curves with a three-terminal DEVSIM BJT geometry.",
                "Add coupled base/emitter/collector output sweeps.",
            ],
        ),
        DeviceTaskTemplate(
            template_id="jfet_transfer_output",
            display_name="JFET Transfer / Output",
            aliases=["jfet", "结型场效应", "junction fet"],
            tasks=["Id-Vg", "Id-Vd", "pinch-off voltage", "transconductance"],
            support=TemplateSupport.EXECUTABLE,
            executable_tool="extended_device_sweep",
            default_request={"device_type": "jfet_transfer_output", "start": -3.0, "stop": 0.0, "step": 0.25},
            benchmark_metrics=["pinch_off_voltage_v", "max_transconductance_s", "idss_a"],
            next_implementation_steps=[
                "Replace compact transfer curve with a 2D JFET channel/gate junction geometry.",
                "Add depletion-region mesh convergence checks.",
            ],
        ),
        DeviceTaskTemplate(
            template_id="power_mosfet_bv_ron",
            display_name="Power MOSFET BV / R_on",
            aliases=["power mos", "power mosfet", "vdmos", "ldmos", "功率mos", "功率mosfet"],
            tasks=["breakdown voltage", "specific on-resistance", "leakage", "field peak"],
            support=TemplateSupport.EXECUTABLE,
            executable_tool="extended_device_sweep",
            default_request={"device_type": "power_mosfet_bv_ron", "start": 0.0, "stop": -90.0, "step": 5.0},
            benchmark_metrics=["breakdown_voltage_v", "specific_on_resistance_ohm_cm2", "max_electric_field_v_per_cm"],
            next_implementation_steps=[
                "Replace compact BV/Ron model with drift-region geometry and high-voltage mesh strategy.",
                "Couple impact ionization into the solver before using BV as final evidence.",
            ],
        ),
        DeviceTaskTemplate(
            template_id="photodiode_iv",
            display_name="Photodiode IV",
            aliases=["photodiode", "photo diode", "光电二极管", "光电"],
            tasks=["dark IV", "illuminated IV", "responsivity", "quantum efficiency"],
            support=TemplateSupport.EXECUTABLE,
            executable_tool="extended_device_sweep",
            default_request={"device_type": "photodiode_iv", "start": -1.0, "stop": 0.8, "step": 0.1},
            benchmark_metrics=["dark_current_a", "photocurrent_a", "responsivity_a_per_w"],
            next_implementation_steps=[
                "Replace compact illuminated IV with optical generation as a spatial source term.",
                "Add quantum-efficiency extraction from wavelength-dependent photon flux.",
            ],
        ),
    ]


def normalize(text: str) -> str:
    return text.lower().replace("_", "-")


def route_device_goal(goal_text: str) -> DeviceRouteResult:
    lowered = normalize(goal_text)
    best: tuple[int, DeviceTaskTemplate, str] | None = None
    for template in device_templates():
        for alias in template.aliases:
            if normalize(alias) in lowered:
                score = len(alias)
                if best is None or score > best[0]:
                    best = (score, template, alias)
    if best is not None:
        _, template, alias = best
        return DeviceRouteResult(
            status=RouteStatus.MATCHED,
            goal_text=goal_text,
            template=template,
            matched_alias=alias,
            executable=template.support == TemplateSupport.EXECUTABLE,
            suggested_tool=template.executable_tool,
            request_hint=template.default_request,
            message=(
                f"Matched {template.display_name}; use {template.executable_tool}."
                if template.support == TemplateSupport.EXECUTABLE
                else f"Matched planned template {template.display_name}; implementation work is required before execution."
            ),
        )
    return DeviceRouteResult(
        status=RouteStatus.UNMATCHED,
        goal_text=goal_text,
        message="No device task template matched the goal text.",
    )


def list_device_templates(*, support: TemplateSupport | None = None) -> list[dict[str, Any]]:
    templates = device_templates()
    if support is not None:
        templates = [template for template in templates if template.support == support]
    return [template.model_dump(mode="json") for template in templates]
