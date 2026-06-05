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
    industrial_metrics: list[str] = Field(default_factory=list)
    natural_language_examples: list[str] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)
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
            industrial_metrics=["ideality factor", "leakage", "turn-on voltage", "series resistance"],
            natural_language_examples=["帮我看这个 PN diode 正向 IV，提取理想因子和漏电风险。"],
            evidence_requirements=["bias_step_retry", "curve_shape_check", "unit_check"],
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
            industrial_metrics=["Cox", "flat-band shift", "fixed oxide charge", "C-V dynamic range"],
            natural_language_examples=["MOSCAP 平带电压偏移，帮我判断是 tox 还是固定电荷导致。"],
            evidence_requirements=["cox_formula_check", "bias_window_check", "unit_check"],
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
            industrial_metrics=["leakage", "BV", "reverse current monotonicity", "temperature corner"],
            natural_language_examples=["这个二极管漏电偏高，帮我扫温度和 SRH 寿命，判断 BV 风险。"],
            evidence_requirements=["reverse_bias_range_check", "curve_shape_check", "temperature_corner"],
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
            industrial_metrics=["Vth", "SS", "Ion/Ioff", "DIBL", "gm", "Id-Vd saturation", "kink risk"],
            natural_language_examples=["帮我看 2D NMOS 线性区和饱和区 Id-Vg，提 Vth/SS/Ion/Ioff/DIBL。"],
            evidence_requirements=["mesh_convergence", "model_ab", "curve_shape_check", "thermal_limit_check"],
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
            industrial_metrics=["barrier height", "ideality factor", "reverse leakage", "series resistance"],
            natural_language_examples=["校准 Schottky diode 到可信 IV 曲线，给 barrier height 和风险。"],
            evidence_requirements=["golden_curve_compare", "series_resistance_check", "model_coupling_check"],
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
            industrial_metrics=["beta", "Early voltage", "Gummel slope", "collector leakage"],
            natural_language_examples=["帮我做 BJT Gummel 和输出特性，提 beta 和 Early voltage。"],
            evidence_requirements=["compact_baseline_warning", "geometry_runner_needed"],
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
            industrial_metrics=["pinch-off voltage", "gm", "Idss", "output conductance"],
            natural_language_examples=["JFET 转移曲线和输出曲线，判断 pinch-off 是否异常。"],
            evidence_requirements=["compact_baseline_warning", "depletion_mesh_check"],
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
            industrial_metrics=["BV", "specific Ron", "leakage", "peak electric field", "drift-region tradeoff"],
            natural_language_examples=["功率 MOSFET BV 和 Ron tradeoff，漏电不能太高，自动找风险点。"],
            evidence_requirements=["high_voltage_mesh", "impact_ionization_coupling", "field_peak_check"],
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
            industrial_metrics=["dark current", "photocurrent", "responsivity", "open-circuit voltage"],
            natural_language_examples=["光电二极管暗电流和光照 IV，提 responsivity 并判断异常。"],
            evidence_requirements=["optical_generation_model", "dark_light_compare"],
            next_implementation_steps=[
                "Replace compact illuminated IV with optical generation as a spatial source term.",
                "Add quantum-efficiency extraction from wavelength-dependent photon flux.",
            ],
        ),
        DeviceTaskTemplate(
            template_id="finfet_id_cv",
            display_name="FinFET / GAA Id-CV",
            aliases=["finfet", "gaa", "nanosheet", "nanowire", "纳米片", "环栅"],
            tasks=["Id-Vg", "Id-Vd", "Cgg/Cgd", "DIBL", "short-channel risk"],
            support=TemplateSupport.PLANNED,
            benchmark_metrics=["vth_at_threshold_current_v", "subthreshold_swing_mv_dec", "dibl_mv_per_v"],
            industrial_metrics=["Vth", "SS", "DIBL", "Ion/Ioff", "gate capacitance"],
            natural_language_examples=["FinFET 短沟道 DIBL 有风险，帮我跑 Id-Vg/Id-Vd 并做 mesh convergence。"],
            evidence_requirements=["3d_geometry_or_2d_surrogate", "quantum_correction", "mesh_convergence"],
            missing_capabilities=["3D/advanced geometry runner", "quantum correction model", "capacitance extraction"],
            next_implementation_steps=[
                "Add a parameterized fin/nanosheet geometry template.",
                "Add Id and capacitance extraction with mesh/quantum-correction evidence.",
            ],
        ),
        DeviceTaskTemplate(
            template_id="sic_power_diode_bv_leakage",
            display_name="SiC Power Diode BV / Leakage",
            aliases=["sic diode", "sic sbd", "sic jbs", "碳化硅", "jbs"],
            tasks=["high-voltage leakage", "breakdown", "temperature corner", "field crowding"],
            support=TemplateSupport.PLANNED,
            benchmark_metrics=["breakdown_voltage_v", "leakage_abs_current_at_target_a", "max_electric_field_v_per_cm"],
            industrial_metrics=["BV", "leakage", "field peak", "temperature sensitivity"],
            natural_language_examples=["SiC JBS 二极管高温漏电偏大，帮我判断 BV 和场峰值风险。"],
            evidence_requirements=["wide_bandgap_material_models", "impact_ionization_coupling", "thermal_corner"],
            missing_capabilities=["SiC material parameter set", "wide-bandgap impact ionization", "high-voltage mesh strategy"],
            next_implementation_steps=[
                "Add SiC material/model preset and reverse-bias high-voltage ramp.",
                "Add field-peak extraction and temperature-corner benchmark.",
            ],
        ),
        DeviceTaskTemplate(
            template_id="gan_hemt_id_bv",
            display_name="GaN HEMT Id / BV",
            aliases=["gan hemt", "hemt", "algan", "氮化镓"],
            tasks=["Id-Vg", "Id-Vd", "2DEG charge", "breakdown", "current collapse risk"],
            support=TemplateSupport.PLANNED,
            benchmark_metrics=["threshold_voltage_v", "on_current_a", "breakdown_voltage_v"],
            industrial_metrics=["2DEG density", "Vth", "Ron", "BV", "current collapse proxy"],
            natural_language_examples=["GaN HEMT 输出特性有 current collapse 风险，帮我扫栅压和漏压。"],
            evidence_requirements=["polarization_charge_model", "trap_model", "self_heating_or_warning"],
            missing_capabilities=["polarization charge model", "trap/current-collapse surrogate", "heterojunction template"],
            next_implementation_steps=[
                "Add AlGaN/GaN heterojunction template and polarization charge model.",
                "Add trap/current-collapse proxy and high-voltage output benchmark.",
            ],
        ),
        DeviceTaskTemplate(
            template_id="igbt_output_turnoff",
            display_name="IGBT Output / Turn-off",
            aliases=["igbt", "绝缘栅双极"],
            tasks=["output curve", "blocking", "turn-off tail current", "latch-up risk"],
            support=TemplateSupport.PLANNED,
            benchmark_metrics=["on_state_voltage_v", "blocking_voltage_v", "tail_current_a"],
            industrial_metrics=["Vce(sat)", "blocking voltage", "tail current", "latch-up margin"],
            natural_language_examples=["IGBT 输出特性和关断尾电流太大，帮我定位结构/寿命风险。"],
            evidence_requirements=["bipolar_transport", "transient_runner", "lifetime_sweep"],
            missing_capabilities=["transient solver wrapper", "IGBT layered geometry", "tail-current extraction"],
            next_implementation_steps=[
                "Add layered IGBT geometry and bipolar transport presets.",
                "Add transient turn-off runner and lifetime sensitivity benchmark.",
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
