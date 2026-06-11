from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from tcad_agent.deck_writer import plan_deck_mutations
from tcad_agent.deck_ir import (
    DeckPatchResult,
    DeckSourceIR,
    apply_semantic_deck_patch,
    parse_devsim_deck_file,
    parse_devsim_deck_source,
)


NUMBER_RE = r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"


def float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def text_has_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def first_path_near(labels: str, text: str) -> str | None:
    match = re.search(
        rf"(?:{labels})\s*[:=：]?\s*([^\s,，;；]+(?:\.csv|\.json|\.txt|\.dat))",
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip("\"'，,。；;") if match else None


def compact_number(value: Any) -> Any:
    numeric = float_or_none(value)
    if numeric is None:
        return value
    if numeric == 0:
        return 0.0
    return float(f"{numeric:.6g}")


def common_signoff_requirements(goal_text: str, request: dict[str, Any]) -> dict[str, Any]:
    strict = text_has_any(
        goal_text,
        ["signoff", "签核", "可信", "客户", "实测", "golden", "量产", "风险", "讨论", "benchmark"],
    )
    measured_curve = first_path_near(r"measured[-_ ]?curve|target[-_ ]?curve|golden[-_ ]?curve|实测曲线|目标曲线|可信曲线", goal_text)
    return {
        "required_level": "engineering_signoff" if strict else "iteration_baseline",
        "require_quality_report": True,
        "require_physical_benchmark": True,
        "require_convergence_evidence": strict,
        "require_curve_shape_check": True,
        "measured_curve_path": measured_curve,
        "golden_metrics": request.get("golden_metrics"),
    }


def physics_model_risk(goal_text: str, request: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    interface_trap = float_or_none(request.get("interface_trap_density_cm2"))
    fixed_charge = float_or_none(request.get("fixed_oxide_charge_cm2"))
    impact = request.get("impact_ionization_model")
    coupling = request.get("advanced_model_coupling") or request.get("model_coupling")
    model_coupled = coupling in {"equation_coupled", "compact_equivalent_bias_and_avalanche"}
    if interface_trap and interface_trap > 0 and not model_coupled:
        warnings.append("界面态已进入 deck/spec，但当前轻量 runner 可能只记录参数，需用 benchmark 标记耦合风险。")
    if fixed_charge and fixed_charge != 0 and not model_coupled:
        warnings.append("固定电荷已进入 deck/spec；若 runner 未耦合到 Poisson，需要把结论标为有条件可信。")
    if impact not in {None, "none"} and not model_coupled:
        warnings.append("碰撞电离/雪崩模型被请求，但当前 runner 可能不是完整方程耦合。")
    return (
        {
            "mobility_model": request.get("mobility_model"),
            "recombination_model": request.get("recombination_model"),
            "electron_lifetime_s": compact_number(request.get("electron_lifetime_s")),
            "hole_lifetime_s": compact_number(request.get("hole_lifetime_s")),
            "interface_trap_density_cm2": compact_number(interface_trap),
            "fixed_oxide_charge_cm2": compact_number(fixed_charge),
            "impact_ionization_model": impact,
            "model_strategy": request.get("model_strategy"),
            "coupling_status": "equation_coupled_or_compact_equivalent" if model_coupled else "needs_benchmark_confirmation",
        },
        warnings,
    )


def mosfet_deck(goal_text: str, request: dict[str, Any]) -> dict[str, Any]:
    physics, warnings = physics_model_risk(goal_text, request)
    sweep_type = request.get("sweep_type") or "both"
    bias_sequence: list[dict[str, Any]] = []
    if sweep_type in {"idvg", "both"}:
        bias_sequence.append(
            {
                "name": "Id-Vg",
                "gate_v": [request.get("gate_start"), request.get("gate_stop"), request.get("gate_step")],
                "drain_v": request.get("drain_voltage"),
                "continuation": "gate_ramp",
            }
        )
    if sweep_type in {"idvd", "both"}:
        bias_sequence.append(
            {
                "name": "Id-Vd",
                "gate_v": request.get("idvd_gate_voltage"),
                "drain_v": [request.get("drain_start"), request.get("drain_stop"), request.get("drain_step")],
                "continuation": "drain_ramp",
            }
        )
    return {
        "device_family": "2d_mosfet",
        "dimensionality": "2d",
        "simulator": "devsim",
        "intent_zh": "二维 MOSFET Id-Vg/Id-Vd 仿真与参数提取",
        "regions": [
            {"name": "silicon", "material": "Si", "role": "channel/source/drain/body"},
            {"name": "gate_oxide", "material": "SiO2", "role": "gate dielectric"},
        ],
        "contacts": ["gate", "source", "drain", "body"],
        "doping": {
            "substrate_doping_cm3": compact_number(request.get("substrate_doping_cm3")),
            "source_drain_doping_cm3": compact_number(request.get("source_drain_doping_cm3")),
        },
        "geometry": {
            "length_um": compact_number(request.get("length_um")),
            "oxide_thickness_nm": compact_number(request.get("oxide_thickness_nm")),
            "silicon_thickness_um": compact_number(request.get("silicon_thickness_um")),
            "source_drain_length_um": compact_number(request.get("source_drain_length_um")),
            "source_drain_depth_um": compact_number(request.get("source_drain_depth_um")),
        },
        "mesh": {
            "x_divisions": request.get("x_divisions"),
            "silicon_y_divisions": request.get("silicon_y_divisions"),
            "expected_followup": "mesh_convergence_for_vth_ss_ion_ioff_or_idvd_kink",
        },
        "physics_models": physics,
        "bias_sequence": bias_sequence,
        "extractions": [
            "vth_at_threshold_current_v",
            "subthreshold_swing_mv_dec",
            "ion_current_a",
            "ioff_current_a",
            "ion_ioff_ratio",
            "max_transconductance_s",
            "idvd_final_current_a",
            "output_conductance_last_s",
            "idvd_kink_slope_jumps",
        ],
        "signoff_requirements": common_signoff_requirements(goal_text, request),
        "assumptions": ["默认 source/body 接地，drain/gate 按 bias_sequence 扫描。"],
        "warnings": warnings,
    }


def moscap_deck(goal_text: str, request: dict[str, Any]) -> dict[str, Any]:
    return {
        "device_family": "mos_capacitor",
        "dimensionality": "1d_quasi_static",
        "simulator": "devsim",
        "intent_zh": "MOS 电容 C-V、Cox/Cmin 与固定电荷偏移判断",
        "regions": [
            {"name": "gate", "material": "metal", "role": "gate electrode"},
            {"name": "oxide", "material": "SiO2", "role": "dielectric"},
            {"name": "substrate", "material": "Si", "role": "semiconductor"},
        ],
        "contacts": ["gate", "substrate"],
        "doping": {"substrate_doping_cm3": compact_number(request.get("substrate_doping_cm3"))},
        "geometry": {
            "oxide_thickness_nm": compact_number(request.get("oxide_thickness_nm")),
            "silicon_thickness_um": compact_number(request.get("silicon_thickness_um")),
        },
        "mesh": {
            "oxide_spacing_nm": compact_number(request.get("oxide_spacing_nm")),
            "silicon_spacing_um": compact_number(request.get("silicon_spacing_um")),
        },
        "physics_models": {
            "fixed_oxide_charge_cm2": compact_number(request.get("fixed_oxide_charge_cm2")),
            "temperature_k": compact_number(request.get("temperature_k")),
            "coupling_status": "compact_equivalent_voltage_shift_or_equation_coupled_check_required",
        },
        "bias_sequence": [
            {"name": "C-V", "gate_v": [request.get("start"), request.get("stop"), request.get("step")], "continuation": "gate_ramp"}
        ],
        "extractions": [
            "min_capacitance_f_per_cm2",
            "max_capacitance_f_per_cm2",
            "oxide_capacitance_estimate_f_per_cm2",
            "capacitance_dynamic_range",
            "fixed_charge_voltage_shift_v",
        ],
        "signoff_requirements": common_signoff_requirements(goal_text, request),
        "assumptions": ["默认高频/准静态 C-V 轻量模型，Cox 用氧化层厚度解析估算复核。"],
        "warnings": [],
    }


def diode_deck(goal_text: str, request: dict[str, Any]) -> dict[str, Any]:
    return {
        "device_family": "pn_diode_breakdown_leakage",
        "dimensionality": "1d",
        "simulator": "devsim",
        "intent_zh": "PN 二极管反偏漏电、击穿与寿命/温度敏感性分析",
        "regions": [{"name": "silicon", "material": "Si", "role": "p-n junction"}],
        "contacts": ["anode", "cathode"],
        "doping": {
            "p_doping_cm3": compact_number(request.get("p_doping_cm3")),
            "n_doping_cm3": compact_number(request.get("n_doping_cm3")),
        },
        "geometry": {
            "length_um": compact_number(request.get("length_um")),
            "junction_um": compact_number(request.get("junction_um")),
        },
        "mesh": {
            "contact_spacing_um": compact_number(request.get("contact_spacing_um")),
            "junction_spacing_um": compact_number(request.get("junction_spacing_um")),
            "expected_followup": "reverse_bias_local_refinement_and_mesh_convergence_near_leakage_or_bv",
        },
        "physics_models": {
            "recombination_model": "srh",
            "electron_lifetime_s": compact_number(request.get("electron_lifetime_s")),
            "hole_lifetime_s": compact_number(request.get("hole_lifetime_s")),
            "temperature_k": compact_number(request.get("temperature_k")),
        },
        "bias_sequence": [
            {
                "name": "reverse IV",
                "terminal": "anode",
                "voltage_v": [request.get("start"), request.get("stop"), request.get("step")],
                "continuation": "reverse_bias_ramp",
            }
        ],
        "extractions": [
            "leakage_abs_current_at_target_a",
            "breakdown_voltage_at_threshold_v",
            "max_reverse_abs_current_a",
            "reverse_current_shape_violations",
            "ideality_factor_estimate",
        ],
        "signoff_requirements": {
            **common_signoff_requirements(goal_text, request),
            "leakage_voltage_v": compact_number(request.get("leakage_voltage_v")),
            "max_leakage_abs_current_a": compact_number(request.get("quality_max_leakage_abs_current_a")),
            "breakdown_current_a": compact_number(request.get("breakdown_current_a")),
            "require_breakdown": bool(request.get("require_breakdown")),
        },
        "assumptions": ["反偏约定使用负电压；漏电和击穿结论必须复核电流符号与曲线单调性。"],
        "warnings": [],
    }


def pn_iv_deck(goal_text: str, request: dict[str, Any]) -> dict[str, Any]:
    return {
        "device_family": "pn_junction_iv",
        "dimensionality": "1d",
        "simulator": "devsim",
        "intent_zh": "PN 结 I-V 仿真与理想因子/整流比提取",
        "regions": [{"name": "silicon", "material": "Si", "role": "p-n junction"}],
        "contacts": ["anode", "cathode"],
        "doping": {
            "p_doping_cm3": compact_number(request.get("p_doping_cm3")),
            "n_doping_cm3": compact_number(request.get("n_doping_cm3")),
        },
        "geometry": {
            "length_um": compact_number(request.get("length_um")),
            "junction_um": compact_number(request.get("junction_um")),
        },
        "mesh": {
            "contact_spacing_um": compact_number(request.get("contact_spacing_um")),
            "junction_spacing_um": compact_number(request.get("junction_spacing_um")),
        },
        "physics_models": {
            "recombination_model": "srh",
            "electron_lifetime_s": compact_number(request.get("electron_lifetime_s")),
            "hole_lifetime_s": compact_number(request.get("hole_lifetime_s")),
            "temperature_k": compact_number(request.get("temperature_k")),
        },
        "bias_sequence": [
            {"name": "IV", "terminal": "anode", "voltage_v": [request.get("start"), request.get("stop"), request.get("step")]}
        ],
        "extractions": ["turn_on_voltage_at_1ua_v", "ideality_factor_estimate", "rectification_ratio_final_to_leakage"],
        "signoff_requirements": common_signoff_requirements(goal_text, request),
        "assumptions": [],
        "warnings": [],
    }


def extended_device_deck(goal_text: str, request: dict[str, Any]) -> dict[str, Any]:
    device_type = str(request.get("device_type") or "extended_device")
    fidelity = str(request.get("fidelity") or "compact")
    is_executable_physics = fidelity in {"devsim_1d", "physics_1d"}
    physics_models: dict[str, Any] = {
        "fidelity": fidelity,
        "schottky_contact_model": request.get("schottky_contact_model"),
        "schottky_contact_coupling_mode": request.get("schottky_contact_coupling_mode"),
        "temperature_k": compact_number(request.get("temperature_k")),
    }
    if device_type == "bjt_gummel_output":
        physics_models.update(
            {
                "transport_model": "charge_control_transport_with_early_effect" if fidelity == "physics_1d" else "compact_gummel",
                "early_voltage_v": compact_number(request.get("bjt_early_voltage_v")),
                "collector_leakage_current_a": compact_number(request.get("bjt_collector_leakage_current_a")),
                "coupling_status": "equation_coupled" if fidelity == "physics_1d" else "compact_baseline",
            }
        )
    if device_type == "power_mosfet_bv_ron":
        physics_models.update(
            {
                "impact_ionization_model": request.get("power_mos_impact_ionization_model"),
                "critical_field_v_per_cm": compact_number(request.get("power_mos_critical_field_v_per_cm")),
                "drift_region_doping_cm3": compact_number(request.get("power_mos_drift_region_doping_cm3")),
                "carrier_lifetime_s": compact_number(request.get("power_mos_carrier_lifetime_s")),
                "drift_region_lifetime_s": compact_number(request.get("power_mos_drift_region_lifetime_s")),
                "trap_density_cm2": compact_number(request.get("power_mos_trap_density_cm2")),
                "coupling_status": "equation_coupled" if fidelity == "physics_1d" else "compact_baseline",
            }
        )
    regions = [{"name": device_type, "material": "Si", "role": "template"}]
    contacts = ["terminal_1", "terminal_2"]
    doping = {
        "schottky_n_doping_cm3": compact_number(request.get("schottky_n_doping_cm3")),
        "power_mos_drift_region_doping_cm3": compact_number(request.get("power_mos_drift_region_doping_cm3")),
    }
    geometry = {
        "area_cm2": compact_number(request.get("area_cm2")),
        "schottky_length_um": compact_number(request.get("schottky_length_um")),
        "power_mos_drift_region_length_um": compact_number(request.get("power_mos_drift_region_length_um")),
    }
    mesh = {
        "schottky_contact_spacing_um": compact_number(request.get("schottky_contact_spacing_um")),
        "schottky_bulk_spacing_um": compact_number(request.get("schottky_bulk_spacing_um")),
        "expected_followup": "bias_or_mesh_convergence_for_extended_physics_path" if is_executable_physics else "runner_promotion_required",
    }
    if device_type == "bjt_gummel_output":
        regions = [
            {"name": "emitter", "material": "Si", "role": "n+ emitter"},
            {"name": "base", "material": "Si", "role": "p base transport region"},
            {"name": "collector", "material": "Si", "role": "n collector/drift region"},
        ]
        contacts = ["emitter", "base", "collector"]
        doping = {
            "emitter_doping_cm3": compact_number(request.get("bjt_emitter_doping_cm3")),
            "base_doping_cm3": compact_number(request.get("bjt_base_doping_cm3")),
            "collector_doping_cm3": compact_number(request.get("bjt_collector_doping_cm3")),
        }
        geometry = {
            "emitter_width_um": compact_number(request.get("bjt_emitter_width_um")),
            "base_width_um": compact_number(request.get("bjt_base_width_um")),
            "collector_width_um": compact_number(request.get("bjt_collector_width_um")),
            "total_stack_width_um": compact_number(
                (float_or_none(request.get("bjt_emitter_width_um")) or 0.0)
                + (float_or_none(request.get("bjt_base_width_um")) or 0.0)
                + (float_or_none(request.get("bjt_collector_width_um")) or 0.0)
            ),
        }
        mesh = {
            "junction_spacing_um": compact_number(request.get("bjt_junction_mesh_spacing_um")),
            "refined_regions": ["emitter_base_junction", "base_collector_junction"],
            "expected_followup": "base_emitter_and_collector_bias_mesh_convergence",
        }
    elif device_type == "power_mosfet_bv_ron":
        regions = [
            {"name": "source", "material": "Si", "role": "n+ source"},
            {"name": "body", "material": "Si", "role": "p body"},
            {"name": "drift", "material": "Si", "role": "high-voltage n drift region"},
            {"name": "drain", "material": "Si", "role": "n+ drain/substrate"},
            {"name": "field_plate_oxide", "material": "SiO2", "role": "field plate dielectric"},
        ]
        contacts = ["gate", "source", "drain", "body", "field_plate"]
        doping = {
            "source_doping_cm3": compact_number(request.get("power_mos_source_doping_cm3")),
            "body_doping_cm3": compact_number(request.get("power_mos_body_doping_cm3")),
            "drift_region_doping_cm3": compact_number(request.get("power_mos_drift_region_doping_cm3")),
            "drain_doping_cm3": compact_number(request.get("power_mos_drain_doping_cm3")),
            "implant_dose_cm2": compact_number(request.get("power_mos_implant_dose_cm2")),
        }
        geometry = {
            "drift_region_length_um": compact_number(request.get("power_mos_drift_region_length_um")),
            "field_plate_length_um": compact_number(request.get("power_mos_field_plate_length_um")),
            "guard_ring_spacing_um": compact_number(request.get("power_mos_guard_ring_spacing_um")),
            "junction_depth_um": compact_number(request.get("power_mos_junction_depth_um")),
            "trench_corner_radius_um": compact_number(request.get("power_mos_trench_corner_radius_um")),
            "gate_oxide_thickness_nm": compact_number(request.get("power_mos_gate_oxide_thickness_nm")),
            "area_cm2": compact_number(request.get("area_cm2")),
        }
        mesh = {
            "junction_spacing_um": compact_number(request.get("power_mos_junction_mesh_spacing_um")),
            "refined_regions": ["body_drift_junction", "drain_drift_junction", "field_plate_edge"],
            "expected_followup": "high_voltage_field_peak_mesh_convergence",
        }
    simulator = (
        "devsim"
        if fidelity == "devsim_1d"
        else "devsim_1d_power_mos_runner"
        if device_type == "power_mosfet_bv_ron" and fidelity == "physics_1d"
        else "physics_1d_model"
        if fidelity == "physics_1d"
        else "compact_model"
    )
    return {
        "device_family": device_type,
        "dimensionality": "1d" if is_executable_physics else "compact",
        "simulator": simulator,
        "intent_zh": "扩展器件模板仿真与关键指标提取",
        "regions": regions,
        "contacts": contacts,
        "doping": doping,
        "geometry": geometry,
        "mesh": mesh,
        "physics_models": physics_models,
        "bias_sequence": [{"name": "terminal sweep", "voltage_v": [request.get("start"), request.get("stop"), request.get("step")]}],
        "extractions": [
            "barrier_height_ev",
            "ideality_factor_estimate",
            "current_gain_beta",
            "breakdown_voltage_v",
            "specific_on_resistance_ohm_cm2",
            "responsivity_a_per_w",
        ],
        "signoff_requirements": common_signoff_requirements(goal_text, request),
        "assumptions": [
            "physics_1d 路径可作为本轮可执行工程证据；最终签核仍需收敛、golden/实测相关性和版图相关几何复核。"
            if is_executable_physics
            else "紧凑模板只能作为规划基线；工程签核需要更完整的几何/模型/收敛验证。"
        ],
        "warnings": [] if is_executable_physics else ["当前为 compact fidelity，不应作为最终 TCAD 签核证据。"],
    }


def build_tcad_deck_spec(goal_text: str, tool_name: str, request: dict[str, Any]) -> dict[str, Any]:
    builders = {
        "mosfet_2d_id_sweep": mosfet_deck,
        "mos_capacitor_cv_sweep": moscap_deck,
        "diode_breakdown_leakage_sweep": diode_deck,
        "pn_junction_iv_sweep": pn_iv_deck,
        "extended_device_sweep": extended_device_deck,
        "schottky_iv_calibration": extended_device_deck,
    }
    builder = builders.get(tool_name)
    base = builder(goal_text, request) if builder else {
        "device_family": tool_name,
        "dimensionality": "unknown",
        "simulator": "unknown",
        "intent_zh": "待补充 TCAD deck/spec",
        "regions": [],
        "contacts": [],
        "doping": {},
        "geometry": {},
        "mesh": {},
        "physics_models": {},
        "bias_sequence": [],
        "extractions": [],
        "signoff_requirements": common_signoff_requirements(goal_text, request),
        "assumptions": [],
        "warnings": ["当前工具暂未建立完整 deck/spec 映射。"],
    }
    planned_mutations = [mutation.model_dump(mode="json") for mutation in plan_deck_mutations(goal_text, tool_name, request)]
    if planned_mutations:
        base["planned_mutations"] = planned_mutations
    base.update(
        {
            "schema_version": "actsoft.tcad.deck.v1",
            "tool_name": tool_name,
            "source_goal_text": goal_text,
        }
    )
    return base


def attach_tcad_deck_spec(goal_text: str, tool_name: str, request: dict[str, Any]) -> dict[str, Any]:
    updated = dict(request)
    updated["tcad_deck_spec"] = build_tcad_deck_spec(goal_text, tool_name, request)
    mutations = [mutation.model_dump(mode="json") for mutation in plan_deck_mutations(goal_text, tool_name, updated)]
    if mutations:
        updated["tcad_deck_mutations"] = mutations
    return updated


def compact_tcad_deck_spec(deck: Any) -> dict[str, Any] | None:
    if not isinstance(deck, dict):
        return None
    return {
        "device_family": deck.get("device_family"),
        "dimensionality": deck.get("dimensionality"),
        "simulator": deck.get("simulator"),
        "intent_zh": deck.get("intent_zh"),
        "physics_models": deck.get("physics_models") or {},
        "bias_sequence": (deck.get("bias_sequence") or [])[:3],
        "extractions": (deck.get("extractions") or [])[:8],
        "signoff_requirements": deck.get("signoff_requirements") or {},
        "planned_mutations": (deck.get("planned_mutations") or [])[:6],
        "warnings": (deck.get("warnings") or [])[:4],
    }


def parse_tcad_deck_source(source: str, *, source_path: str | None = None) -> DeckSourceIR:
    return parse_devsim_deck_source(source, source_path=source_path)


def parse_tcad_deck_file(path: str | Path) -> DeckSourceIR:
    return parse_devsim_deck_file(Path(path))


def semantic_patch_tcad_deck_source(
    source: str,
    patches: list[dict[str, Any]] | dict[str, Any],
    *,
    source_path: str | None = None,
) -> DeckPatchResult:
    return apply_semantic_deck_patch(source, patches, source_path=source_path)
