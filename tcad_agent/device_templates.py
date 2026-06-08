from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.public_sources import public_sources_for_template


class TemplateSupport(str, Enum):
    EXECUTABLE = "executable"
    COMPACT_BASELINE = "compact_baseline"
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
    tcad_fidelity: str = "unknown"
    signoff_workflow: list[str] = Field(default_factory=list)
    public_source_category_ids: list[str] = Field(default_factory=list)
    recommended_convergence: list[str] = Field(default_factory=list)
    missing_capabilities: list[str] = Field(default_factory=list)
    next_implementation_steps: list[str] = Field(default_factory=list)

    @property
    def runnable(self) -> bool:
        return self.support in {TemplateSupport.EXECUTABLE, TemplateSupport.COMPACT_BASELINE}

    @property
    def signoff_ready(self) -> bool:
        return self.support == TemplateSupport.EXECUTABLE


class DeviceRouteResult(BaseModel):
    status: RouteStatus
    goal_text: str
    template: DeviceTaskTemplate | None = None
    matched_alias: str | None = None
    executable: bool = False
    runnable: bool = False
    signoff_ready: bool = False
    suggested_tool: str | None = None
    request_hint: dict[str, Any] = Field(default_factory=dict)
    capability_warnings: list[str] = Field(default_factory=list)
    tcad_fidelity: str | None = None
    signoff_workflow: list[str] = Field(default_factory=list)
    public_source_category_ids: list[str] = Field(default_factory=list)
    public_sources: list[dict[str, Any]] = Field(default_factory=list)
    recommended_convergence: list[str] = Field(default_factory=list)
    message: str


def device_templates() -> list[DeviceTaskTemplate]:
    return [
        DeviceTaskTemplate(
            template_id="pn_junction_iv",
            display_name="PN Junction IV",
            aliases=["pn junction", "pn结", "p-n", "pn iv", "pn diode", "pn 二极管", "pn二极管", "pn 任务", "pn任务"],
            tasks=["forward/reverse IV", "ideality factor", "rectification ratio"],
            support=TemplateSupport.EXECUTABLE,
            executable_tool="pn_junction_iv_sweep",
            default_request={"start": 0.0, "stop": 0.5, "step": 0.1},
            benchmark_metrics=["ideality_factor_estimate", "rectification_ratio_final_to_leakage"],
            industrial_metrics=["ideality factor", "leakage", "turn-on voltage", "series resistance"],
            natural_language_examples=["帮我看这个 diode/SBD reverse leakage 和 BV 风险，扫反偏并提取漏电阈值。"],
            evidence_requirements=["bias_step_retry", "curve_shape_check", "unit_check"],
            tcad_fidelity="devsim_1d_drift_diffusion",
            signoff_workflow=["run_primary_iv", "physical_benchmark", "bias_or_mesh_convergence", "golden_or_measured_comparison_if_requested"],
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
            tcad_fidelity="devsim_1d_quasi_static_cv",
            signoff_workflow=["run_primary_cv", "cox_analytic_benchmark", "bias_or_mesh_convergence", "golden_or_measured_comparison_if_requested"],
            public_source_category_ids=["moscap_capacitance"],
            recommended_convergence=[
                "solve_equilibrium_before_voltage_sweep",
                "sweep_accumulation_to_inversion_with_step_shrink",
                "compare_max_capacitance_to_analytic_cox",
            ],
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
            tcad_fidelity="devsim_1d_reverse_iv",
            signoff_workflow=["run_reverse_iv", "physical_benchmark", "reverse_bias_local_refinement", "leakage_or_bv_golden_comparison_if_requested"],
            public_source_category_ids=["diode_sbd_breakdown"],
            recommended_convergence=[
                "start_from_small_reverse_bias",
                "use_local_refinement_near_current_threshold",
                "switch_to_current_or_resistor_control_after_breakdown_onset",
            ],
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
            tcad_fidelity="devsim_2d_drift_diffusion",
            signoff_workflow=["run_idvg_idvd", "physical_benchmark", "mesh_model_convergence", "curve_shape_review", "golden_or_measured_comparison_if_requested"],
            public_source_category_ids=["mosfet_id_dibl"],
            recommended_convergence=[
                "solve_equilibrium_before_bias",
                "ramp_drain_before_gate_sweep",
                "split_low_high_drain_idvg_for_dibl",
                "save_load_intermediate_solution_between_bias_phases",
            ],
        ),
        DeviceTaskTemplate(
            template_id="schottky_diode",
            display_name="Schottky Diode",
            aliases=["schottky", "肖特基"],
            tasks=["forward IV", "reverse leakage", "barrier-height extraction", "C-V"],
            support=TemplateSupport.EXECUTABLE,
            executable_tool="extended_device_sweep",
            default_request={
                "device_type": "schottky_diode",
                "fidelity": "devsim_1d",
                "start": -0.5,
                "stop": 0.8,
                "step": 0.1,
            },
            benchmark_metrics=["barrier_height_ev", "ideality_factor_estimate", "reverse_leakage_current_a"],
            industrial_metrics=["barrier height", "ideality factor", "reverse leakage", "series resistance"],
            natural_language_examples=["校准 Schottky diode 到可信 IV 曲线，给 barrier height 和风险。"],
            evidence_requirements=["golden_curve_compare", "series_resistance_check", "model_coupling_check"],
            tcad_fidelity="devsim_1d_thermionic_contact",
            signoff_workflow=["run_devsim_1d_schottky_iv", "physical_benchmark", "contact_model_coupling_check", "golden_curve_calibration"],
            public_source_category_ids=["diode_sbd_breakdown"],
            recommended_convergence=[
                "start_from_small_reverse_bias",
                "use_local_refinement_near_current_threshold",
                "switch_to_current_or_resistor_control_after_breakdown_onset",
            ],
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
            default_request={
                "device_type": "bjt_gummel_output",
                "fidelity": "physics_1d",
                "evidence_level": "tcad_executable",
                "start": 0.55,
                "stop": 0.8,
                "step": 0.025,
            },
            benchmark_metrics=["current_gain_beta", "early_voltage_v", "collector_leakage_current_a"],
            industrial_metrics=["beta", "Early voltage", "Gummel slope", "collector leakage"],
            natural_language_examples=["帮我做 BJT Gummel 和输出特性，提 beta 和 Early voltage。"],
            evidence_requirements=["three_terminal_output_family", "early_effect_check", "gummel_slope_check"],
            tcad_fidelity="physics_1d_bjt_transport",
            signoff_workflow=[
                "run_gummel_and_output_family",
                "physical_benchmark",
                "base_emitter_and_collector_bias_convergence",
                "golden_or_measured_comparison_if_requested",
            ],
            public_source_category_ids=["bjt_gummel_output"],
            recommended_convergence=[
                "solve_equilibrium_then_base_emitter_ramp",
                "hold_vce_for_gummel_before_output_family",
                "sweep_collector_voltage_from_saved_base_bias_states",
            ],
            next_implementation_steps=[
                "Correlate the physics_1d BJT transport baseline against DEVSIM BJT public examples or measured Gummel curves.",
                "Add mesh-resolved three-terminal geometry when a full public runner is imported.",
            ],
        ),
        DeviceTaskTemplate(
            template_id="jfet_transfer_output",
            display_name="JFET Transfer / Output",
            aliases=["jfet", "结型场效应", "junction fet"],
            tasks=["Id-Vg", "Id-Vd", "pinch-off voltage", "transconductance"],
            support=TemplateSupport.EXECUTABLE,
            executable_tool="extended_device_sweep",
            default_request={
                "device_type": "jfet_transfer_output",
                "fidelity": "physics_1d",
                "evidence_level": "tcad_executable",
                "start": -3.0,
                "stop": 0.0,
                "step": 0.25,
            },
            benchmark_metrics=["pinch_off_voltage_v", "max_transconductance_s", "idss_a"],
            industrial_metrics=["pinch-off voltage", "gm", "Idss", "output conductance"],
            natural_language_examples=[],
            evidence_requirements=["depletion_model_coupling", "output_family", "depletion_mesh_check"],
            tcad_fidelity="physics_1d_gate_junction_depletion",
            signoff_workflow=["run_transfer_and_output_family", "physical_benchmark", "gate_junction_depletion_convergence", "golden_or_measured_comparison_if_requested"],
            recommended_convergence=["solve_gate_junction_depletion_first", "ramp_gate_reverse_bias_before_output_family"],
            next_implementation_steps=[
                "Correlate the physics_1d depletion baseline against a mesh-resolved JFET channel/gate geometry.",
                "Add local depletion-region mesh convergence checks for final layout-sensitive signoff.",
            ],
        ),
        DeviceTaskTemplate(
            template_id="power_mosfet_bv_ron",
            display_name="Power MOSFET BV / R_on",
            aliases=[
                "power mos",
                "power mosfet",
                "vdmos",
                "ldmos",
                "field plate",
                "field-plate",
                "drift doping",
                "drift region",
                "功率mos",
                "功率mosfet",
                "场板",
                "漂移区",
                "漂移区掺杂",
            ],
            tasks=["breakdown voltage", "specific on-resistance", "leakage", "field peak"],
            support=TemplateSupport.EXECUTABLE,
            executable_tool="extended_device_sweep",
            default_request={
                "device_type": "power_mosfet_bv_ron",
                "fidelity": "physics_1d",
                "evidence_level": "tcad_executable",
                "start": 0.0,
                "stop": -90.0,
                "step": 5.0,
            },
            benchmark_metrics=["breakdown_voltage_v", "specific_on_resistance_ohm_cm2", "max_electric_field_v_per_cm"],
            industrial_metrics=["BV", "specific Ron", "leakage", "peak electric field", "drift-region tradeoff"],
            natural_language_examples=["功率 MOSFET BV 和 Ron tradeoff，漏电不能太高，自动找风险点。"],
            evidence_requirements=["high_voltage_field_check", "impact_ionization_coupling", "ron_component_decomposition"],
            tcad_fidelity="physics_1d_high_voltage_drift_avalanche",
            signoff_workflow=[
                "run_off_state_bv_and_ron_tradeoff",
                "physical_benchmark",
                "high_voltage_bias_convergence",
                "golden_or_measured_comparison_if_requested",
            ],
            public_source_category_ids=["ldmos_igbt_power"],
            recommended_convergence=[
                "separate_off_state_blocking_from_on_state_output",
                "ramp_high_voltage_with_small_initial_step",
                "use_current_or_resistor_control_for_snapback_or_breakdown",
            ],
            next_implementation_steps=[
                "Correlate the physics_1d high-voltage drift/avalanche baseline against LDMOS public application examples.",
                "Add a mesh-resolved field-plate/drift-region runner for final layout-sensitive signoff.",
            ],
        ),
        DeviceTaskTemplate(
            template_id="photodiode_iv",
            display_name="Photodiode IV",
            aliases=["photodiode", "photo diode", "光电二极管", "光电"],
            tasks=["dark IV", "illuminated IV", "responsivity", "quantum efficiency"],
            support=TemplateSupport.EXECUTABLE,
            executable_tool="extended_device_sweep",
            default_request={
                "device_type": "photodiode_iv",
                "fidelity": "physics_1d",
                "evidence_level": "tcad_executable",
                "start": -1.0,
                "stop": 0.8,
                "step": 0.1,
            },
            benchmark_metrics=["dark_current_a", "photocurrent_a", "responsivity_a_per_w"],
            industrial_metrics=["dark current", "photocurrent", "responsivity", "open-circuit voltage"],
            natural_language_examples=[],
            evidence_requirements=["optical_generation_model", "dark_light_compare", "responsivity_benchmark"],
            tcad_fidelity="physics_1d_optical_generation",
            signoff_workflow=["run_dark_and_illuminated_iv", "physical_benchmark", "optical_generation_convergence", "golden_or_measured_comparison_if_requested"],
            recommended_convergence=["solve_dark_iv_before_illumination", "sweep_optical_power_after_bias_converges"],
            next_implementation_steps=[
                "Correlate optical-generation baseline against wavelength-dependent QE/reference data.",
                "Add spatial absorption profile runner for final detector signoff.",
            ],
        ),
        DeviceTaskTemplate(
            template_id="finfet_id_cv",
            display_name="FinFET / GAA Id-CV",
            aliases=["finfet", "gaa", "nanosheet", "nanowire", "纳米片", "环栅"],
            tasks=["Id-Vg", "Id-Vd", "Cgg/Cgd", "DIBL", "short-channel risk"],
            support=TemplateSupport.EXECUTABLE,
            executable_tool="extended_device_sweep",
            default_request={
                "device_type": "finfet_id_cv",
                "fidelity": "physics_1d",
                "evidence_level": "tcad_executable",
                "start": 0.0,
                "stop": 1.0,
                "step": 0.1,
            },
            benchmark_metrics=["vth_at_threshold_current_v", "subthreshold_swing_mv_dec", "dibl_mv_per_v"],
            industrial_metrics=["Vth", "SS", "DIBL", "Ion/Ioff", "gate capacitance"],
            natural_language_examples=["FinFET 短沟道 DIBL 有风险，帮我跑 Id-Vg/Id-Vd 并做 mesh convergence。"],
            evidence_requirements=["3d_geometry_or_2d_surrogate", "quantum_correction", "mesh_convergence"],
            tcad_fidelity="physics_1d_finfet_surrogate_density_gradient",
            signoff_workflow=["run_id_cv_surrogate", "physical_benchmark", "mesh_quantum_convergence", "golden_or_measured_comparison_if_requested"],
            public_source_category_ids=["finfet_soi_variability"],
            recommended_convergence=[
                "validate_planar_or_2d_surrogate_before_3d",
                "enable_density_gradient_after_drift_diffusion_converges",
                "run_nominal_geometry_before_random_trap_or_dopant_splits",
            ],
            missing_capabilities=[],
            next_implementation_steps=[
                "Correlate the fin/nanosheet surrogate against a mesh-resolved 3D runner.",
                "Add distribution-level variability campaign after nominal mesh/quantum convergence passes.",
            ],
        ),
        DeviceTaskTemplate(
            template_id="sic_power_diode_bv_leakage",
            display_name="SiC Power Diode BV / Leakage",
            aliases=["sic diode", "sic sbd", "sic jbs", "碳化硅", "jbs"],
            tasks=["high-voltage leakage", "breakdown", "temperature corner", "field crowding"],
            support=TemplateSupport.EXECUTABLE,
            executable_tool="extended_device_sweep",
            default_request={
                "device_type": "sic_power_diode_bv_leakage",
                "fidelity": "physics_1d",
                "evidence_level": "tcad_executable",
                "start": 0.0,
                "stop": -1200.0,
                "step": 50.0,
            },
            benchmark_metrics=["breakdown_voltage_v", "leakage_abs_current_at_target_a", "max_electric_field_v_per_cm"],
            industrial_metrics=["BV", "leakage", "field peak", "temperature sensitivity"],
            natural_language_examples=["SiC JBS 二极管高温漏电偏大，帮我判断 BV 和场峰值风险。"],
            evidence_requirements=["wide_bandgap_material_models", "impact_ionization_coupling", "thermal_corner"],
            tcad_fidelity="physics_1d_sic_wide_bandgap_avalanche",
            signoff_workflow=["run_high_voltage_reverse_iv", "physical_benchmark", "field_temperature_convergence", "golden_or_measured_comparison_if_requested"],
            public_source_category_ids=["diode_sbd_breakdown"],
            recommended_convergence=[
                "start_from_small_reverse_bias",
                "use_local_refinement_near_current_threshold",
                "sweep_temperature_after_room_temperature_converges",
            ],
            missing_capabilities=[],
            next_implementation_steps=[
                "Correlate SiC material/model preset against trusted SiC/JBS public examples or measured curves.",
                "Add mesh-resolved junction/field-crowding geometry for layout-sensitive signoff.",
            ],
        ),
        DeviceTaskTemplate(
            template_id="gan_hemt_id_bv",
            display_name="GaN HEMT Id / BV",
            aliases=["gan hemt", "hemt", "algan", "氮化镓"],
            tasks=["Id-Vg", "Id-Vd", "2DEG charge", "breakdown", "current collapse risk"],
            support=TemplateSupport.EXECUTABLE,
            executable_tool="extended_device_sweep",
            default_request={
                "device_type": "gan_hemt_id_bv",
                "fidelity": "physics_1d",
                "evidence_level": "tcad_executable",
                "start": -4.0,
                "stop": 2.0,
                "step": 0.25,
            },
            benchmark_metrics=["threshold_voltage_v", "on_current_a", "breakdown_voltage_v"],
            industrial_metrics=["2DEG density", "Vth", "Ron", "BV", "current collapse proxy"],
            natural_language_examples=["GaN HEMT 输出特性有 current collapse 风险，帮我扫栅压和漏压。"],
            evidence_requirements=["polarization_charge_model", "trap_model", "self_heating_or_warning"],
            tcad_fidelity="physics_1d_algan_gan_polarization_trap",
            signoff_workflow=["run_id_bv_current_collapse", "physical_benchmark", "polarization_trap_convergence", "golden_or_measured_comparison_if_requested"],
            public_source_category_ids=["gan_algan_hemt"],
            recommended_convergence=[
                "solve_heterojunction_equilibrium_with_fixed_polarization_first",
                "ramp_trap_occupancy_or_enable_traps_after_dc_converges",
                "split_output_sweeps_by_gate_bias",
            ],
            missing_capabilities=[],
            next_implementation_steps=[
                "Correlate AlGaN/GaN polarization and trap baseline against heterojunction runner or public data.",
                "Add field-plate/self-heating geometry for layout-sensitive signoff.",
            ],
        ),
        DeviceTaskTemplate(
            template_id="igbt_output_turnoff",
            display_name="IGBT Output / Turn-off",
            aliases=["igbt", "绝缘栅双极"],
            tasks=["output curve", "blocking", "turn-off tail current", "latch-up risk"],
            support=TemplateSupport.EXECUTABLE,
            executable_tool="extended_device_sweep",
            default_request={
                "device_type": "igbt_output_turnoff",
                "fidelity": "physics_1d",
                "evidence_level": "tcad_executable",
                "start": 0.0,
                "stop": 4.0,
                "step": 0.25,
            },
            benchmark_metrics=["on_state_voltage_v", "blocking_voltage_v", "tail_current_a"],
            industrial_metrics=["Vce(sat)", "blocking voltage", "tail current", "latch-up margin"],
            natural_language_examples=["IGBT 输出特性和关断尾电流太大，帮我定位结构/寿命风险。"],
            evidence_requirements=["bipolar_transport", "transient_runner", "lifetime_sweep"],
            tcad_fidelity="physics_1d_bipolar_transient_tail",
            signoff_workflow=["run_output_blocking_turnoff", "physical_benchmark", "lifetime_transient_convergence", "golden_or_measured_comparison_if_requested"],
            public_source_category_ids=["ldmos_igbt_power"],
            recommended_convergence=[
                "separate_off_state_blocking_from_on_state_output",
                "reuse_dc_solution_as_transient_initial_state",
                "tighten_mesh_at_drift_junction_and_field_plate_edges",
            ],
            missing_capabilities=[],
            next_implementation_steps=[
                "Correlate IGBT bipolar/transient baseline against a layered geometry runner.",
                "Add lifetime and drift-region mesh convergence for tail-current signoff.",
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
        capability_warnings: list[str] = []
        if template.support == TemplateSupport.COMPACT_BASELINE:
            capability_warnings.append(
                "当前只有 compact baseline 可运行；结果只能作为规划/路由证据，不能作为最终 TCAD 签核证据。"
            )
            capability_warnings.extend(template.next_implementation_steps[:3])
        elif template.support == TemplateSupport.PLANNED:
            capability_warnings.append("该工业模板尚未实现可执行 TCAD runner、质量规则和 benchmark 证据链。")
            capability_warnings.extend(template.missing_capabilities[:3])
        return DeviceRouteResult(
            status=RouteStatus.MATCHED,
            goal_text=goal_text,
            template=template,
            matched_alias=alias,
            executable=template.support == TemplateSupport.EXECUTABLE,
            runnable=template.runnable,
            signoff_ready=template.signoff_ready,
            suggested_tool=template.executable_tool if template.runnable else None,
            request_hint=template.default_request if template.runnable else {},
            capability_warnings=capability_warnings,
            tcad_fidelity=template.tcad_fidelity,
            signoff_workflow=template.signoff_workflow,
            public_source_category_ids=template.public_source_category_ids,
            public_sources=[
                {
                    "source_id": source["source_id"],
                    "name": source["name"],
                    "url": source["url"],
                    "source_type": source["source_type"],
                    "access": source["access"],
                }
                for source in public_sources_for_template(template.template_id)
            ],
            recommended_convergence=template.recommended_convergence,
            message=(
                f"Matched {template.display_name}; use {template.executable_tool} as TCAD evidence."
                if template.support == TemplateSupport.EXECUTABLE
                else f"Matched compact baseline {template.display_name}; use {template.executable_tool} only as planning evidence."
                if template.support == TemplateSupport.COMPACT_BASELINE
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
