from __future__ import annotations

import argparse
import json
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from tcad_agent.deck_writer import deck_mutation_convergence_requests
from tcad_agent.device_templates import RouteStatus, TemplateSupport, route_device_goal
from tcad_agent.engineering_intent import DeviceSupport, parse_engineering_intent
from tcad_agent.llm import LLMClient, LLMConfig
from tcad_agent.task_planner import parse_json_object
from tcad_agent.task_spec import PROJECT_ROOT
from tcad_agent.tcad_spec import parse_tcad_spec
from tcad_agent.tool_convergence import normalize_tool_convergence_payload


class ChatClient(Protocol):
    config: LLMConfig

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        ...


class DecompositionStatus(str, Enum):
    COMPLETED = "completed"
    FALLBACK = "fallback"
    FAILED = "failed"


class GoalStepKind(str, Enum):
    RUN_SUPERVISOR = "run_supervisor"
    RUN_TOOL_CONVERGENCE = "run_tool_convergence"
    RUN_GOLDEN_COMPARISON = "run_golden_comparison"
    RUN_PHYSICAL_BENCHMARK = "run_physical_benchmark"
    RUN_REPAIR_EXECUTOR = "run_repair_executor"
    GENERATE_CONCLUSION = "generate_conclusion"
    QUERY_HISTORY = "query_history"
    ASK_USER = "ask_user"


class GoalStep(BaseModel):
    index: int
    kind: GoalStepKind
    title: str
    request: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[int] = Field(default_factory=list)
    stop_on_failure: bool = True
    requires_user_confirmation: bool = False


class GoalDecompositionResult(BaseModel):
    status: DecompositionStatus
    goal_text: str
    plan_id: str | None = None
    model: str | None = None
    raw_response: str | None = None
    parsed_response: dict[str, Any] | None = None
    steps: list[GoalStep] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    validation_errors: list[str] = Field(default_factory=list)


class ReplanDecision(BaseModel):
    status: DecompositionStatus
    goal_text: str
    plan_id: str | None = None
    model: str | None = None
    raw_response: str | None = None
    parsed_response: dict[str, Any] | None = None
    issue_family: str = "execution"
    control_action: str = "replan"
    strategy_zh: str
    recommended_actions: list[str] = Field(default_factory=list)
    append_steps: list[GoalStep] = Field(default_factory=list)
    mark_soft_failed: list[int] = Field(default_factory=list)
    skip_goal_steps: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    validation_errors: list[str] = Field(default_factory=list)


def text_has_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def primary_goal_step(text: str) -> GoalStep:
    intent = parse_engineering_intent(text)
    spec = parse_tcad_spec(text)
    return GoalStep(
        index=1,
        kind=GoalStepKind.RUN_SUPERVISOR,
        title="Run primary TCAD task through supervisor",
        request={
            "goal_text": text,
            "execute": True,
            "max_cycles": 3,
            "engineering_intent": intent.model_dump(mode="json"),
            "tcad_spec": spec.model_dump(mode="json"),
            "request_hint": intent.request_hint,
            "suggested_tool": intent.suggested_tool,
            "capability_warnings": intent.capability_warnings,
            "clarification_questions": intent.clarification_questions,
        },
        stop_on_failure=False,
    )


def needs_clarification_before_execution(intent: Any) -> bool:
    return (
        intent.support == DeviceSupport.UNKNOWN
        and not intent.suggested_tool
        and not intent.request_hint
        and len(intent.clarification_questions) >= 2
    )


def default_tool_convergence_request(text: str) -> dict[str, Any]:
    intent = parse_engineering_intent(text)
    mutation_requests = deck_mutation_convergence_requests(text, intent.suggested_tool, intent.request_hint)
    if mutation_requests:
        return mutation_requests[0]
    if intent.device_family == "power_mosfet":
        return {
            "tool_name": "extended_device_sweep",
            "base_request": {
                "device_type": "power_mosfet_bv_ron",
                "fidelity": "physics_1d",
                "evidence_level": "tcad_executable",
                "start": 0.0,
                "stop": -90.0,
                "step": 5.0,
            },
            "axis_path": "power_mos_drift_region_doping_cm3",
            "values": [5.0e15, 1.0e16, 2.0e16],
            "metric_path": "quality_report.metrics.specific_on_resistance_ohm_cm2",
            "relative_tolerance": 0.25,
        }
    if intent.device_family == "bjt":
        return {
            "tool_name": "extended_device_sweep",
            "base_request": {
                "device_type": "bjt_gummel_output",
                "fidelity": "physics_1d",
                "evidence_level": "tcad_executable",
                "start": 0.55,
                "stop": 0.8,
                "step": 0.025,
            },
            "axis_path": "bjt_base_width_um",
            "values": [0.15, 0.2, 0.3],
            "metric_path": "quality_report.metrics.current_gain_beta",
            "relative_tolerance": 0.3,
        }
    compact_device_map = {
        "finfet": {
            "device_type": "finfet_id_cv",
            "axis_path": "finfet_gate_length_nm",
            "values": [18.0, 28.0, 40.0],
            "metric_path": "quality_report.metrics.ion_ioff_ratio",
            "base_request": {"device_type": "finfet_id_cv", "fidelity": "physics_1d", "evidence_level": "tcad_executable", "start": 0.0, "stop": 1.0, "step": 0.1},
        },
        "sic_power_diode": {
            "device_type": "sic_power_diode_bv_leakage",
            "axis_path": "sic_breakdown_voltage_v",
            "values": [-800.0, -1200.0, -1700.0],
            "metric_path": "quality_report.metrics.leakage_abs_current_at_target_a",
            "base_request": {"device_type": "sic_power_diode_bv_leakage", "fidelity": "physics_1d", "evidence_level": "tcad_executable", "start": 0.0, "stop": -1200.0, "step": 50.0},
        },
        "gan_hemt": {
            "device_type": "gan_hemt_id_bv",
            "axis_path": "gan_2deg_density_cm2",
            "values": [5.0e12, 1.0e13, 1.5e13],
            "metric_path": "quality_report.metrics.on_current_a",
            "base_request": {"device_type": "gan_hemt_id_bv", "fidelity": "physics_1d", "evidence_level": "tcad_executable", "start": -4.0, "stop": 2.0, "step": 0.25},
        },
        "igbt": {
            "device_type": "igbt_output_turnoff",
            "axis_path": "igbt_tail_current_a",
            "values": [5.0e-4, 2.0e-3, 8.0e-3],
            "metric_path": "quality_report.metrics.tail_current_a",
            "base_request": {"device_type": "igbt_output_turnoff", "fidelity": "physics_1d", "evidence_level": "tcad_executable", "start": 0.0, "stop": 4.0, "step": 0.25},
        },
    }
    if intent.device_family in compact_device_map:
        config = compact_device_map[intent.device_family]
        return {
            "tool_name": "extended_device_sweep",
            "base_request": config["base_request"],
            "axis_path": config["axis_path"],
            "values": config["values"],
            "metric_path": config["metric_path"],
            "relative_tolerance": 0.25,
        }
    if text_has_any(text, ["schottky", "肖特基"]) and text_has_any(
        text,
        ["calibrate", "calibration", "fit", "trusted curve", "measured curve", "校准", "标定", "拟合", "可信曲线", "实测曲线"],
    ):
        return {
            "tool_name": "schottky_iv_calibration",
            "base_request": {
                "start": -0.2,
                "stop": 0.4,
                "step": 0.1,
                "max_pass_rmse_log_current_dec": 0.15,
            },
            "axis_path": "step",
            "values": [0.2, 0.1, 0.05],
            "metric_path": "quality_report.metrics.best_rmse_log_current_dec",
            "relative_tolerance": 0.05,
        }
    if text_has_any(text, ["mosfet", "nmos", "pmos", "id-vg", "idvg", "id-vd", "idvd", "output characteristic", "输出特性", "kink", "vth", "ion/ioff"]):
        output_like = text_has_any(text, ["id-vd", "idvd", "output characteristic", "输出特性", "输出曲线", "kink"])
        transfer_like = text_has_any(text, ["id-vg", "idvg", "transfer characteristic", "转移特性", "vth", "ss", "ion/ioff", "阈值"])
        if text_has_any(text, ["dibl", "低 vd", "高 vd", "short channel", "短沟道"]):
            return {
                "tool_name": "mosfet_2d_id_sweep",
                "base_request": {
                    "sweep_type": "idvg",
                    "gate_start": 0.0,
                    "gate_stop": 1.2,
                    "gate_step": 0.2,
                    "drain_voltage": 0.05,
                },
                "axis_path": "drain_voltage",
                "values": [0.05, 1.0],
                "metric_path": "quality_report.metrics.vth_at_threshold_current_v",
                "relative_tolerance": 1.0,
            }
        if text_has_any(text, ["mobility", "迁移率", "constant mobility", "doping-dependent", "doping dependent", "a/b", "ab"]):
            return {
                "tool_name": "mosfet_2d_id_sweep",
                "base_request": {
                    "sweep_type": "idvg",
                    "gate_start": 0.0,
                    "gate_stop": 1.2,
                    "gate_step": 0.2,
                    "drain_voltage": 0.05,
                },
                "axis_path": "mobility_model",
                "values": ["constant", "doping_dependent"],
                "metric_path": "quality_report.metrics.ion_ioff_ratio",
                "relative_tolerance": 0.5,
            }
        if text_has_any(text, ["interface trap", "dit", "界面态", "界面陷阱"]):
            return {
                "tool_name": "mosfet_2d_id_sweep",
                "base_request": {
                    "sweep_type": "idvg",
                    "gate_start": 0.0,
                    "gate_stop": 1.2,
                    "gate_step": 0.2,
                    "drain_voltage": 0.05,
                },
                "axis_path": "interface_trap_density_cm2",
                "values": [0.0, 1.0e11],
                "metric_path": "quality_report.metrics.subthreshold_swing_mv_dec",
                "relative_tolerance": 0.5,
            }
        sweep_type = "both" if output_like and transfer_like else "idvd" if output_like else "idvg"
        metric_path = "quality_report.metrics.idvd_final_current_a" if sweep_type == "idvd" else "quality_report.metrics.ion_ioff_ratio"
        return {
            "tool_name": "mosfet_2d_id_sweep",
            "base_request": {
                "sweep_type": sweep_type,
                "gate_start": 0.0,
                "gate_stop": 1.2 if transfer_like else 0.5,
                "gate_step": 0.2,
                "drain_start": 0.0,
                "drain_stop": 1.2 if output_like else 0.05,
                "drain_step": 0.2 if output_like else 0.05,
                "idvd_gate_voltage": 1.2 if output_like else 0.5,
            },
            "axis_path": "x_divisions",
            "values": [8, 12, 16],
            "metric_path": metric_path,
            "relative_tolerance": 0.1,
        }
    if text_has_any(text, ["mos c-v", "mos cv", "moscap", "capacitor", "fixed charge", "fixed oxide charge", "平带", "固定电荷", "电容"]):
        if text_has_any(text, ["tox", "oxide thickness", "氧化层厚度", "厚薄", "偏厚", "偏薄"]):
            return {
                "tool_name": "mos_capacitor_cv_sweep",
                "base_request": {"start": -2.0, "stop": 2.0, "step": 0.25, "oxide_thickness_nm": 5.0},
                "axis_path": "oxide_thickness_nm",
                "values": [4.5, 5.0, 5.5],
                "metric_path": "quality_report.metrics.max_capacitance_f_per_cm2",
                "relative_tolerance": 0.2,
            }
        if text_has_any(text, ["fixed charge", "fixed oxide charge", "qf", "固定电荷", "氧化层固定电荷", "平带"]):
            return {
                "tool_name": "mos_capacitor_cv_sweep",
                "base_request": {"start": -2.0, "stop": 2.0, "step": 0.25, "fixed_oxide_charge_cm2": 0.0},
                "axis_path": "fixed_oxide_charge_cm2",
                "values": [0.0, 5.0e11],
                "metric_path": "quality_report.metrics.fixed_charge_voltage_shift_v",
                "relative_tolerance": 1.0,
            }
        return {
            "tool_name": "mos_capacitor_cv_sweep",
            "base_request": {"start": -1.0, "stop": 1.0, "step": 0.25},
            "axis_path": "oxide_spacing_nm",
            "values": [0.5, 0.25, 0.125],
            "metric_path": "quality_report.metrics.final_capacitance_f_per_cm2",
            "relative_tolerance": 0.05,
        }
    if text_has_any(text, ["breakdown", "leakage", "击穿", "漏电"]):
        if text_has_any(text, ["temperature", "temp", "高温", "温度", "corner"]):
            return {
                "tool_name": "diode_breakdown_leakage_sweep",
                "base_request": {"start": 0.0, "stop": -10.0, "step": 0.5, "temperature_k": 300.0},
                "axis_path": "temperature_k",
                "values": [300.0, 350.0, 400.0],
                "metric_path": "quality_report.metrics.leakage_abs_current_at_target_a",
                "relative_tolerance": 1.0,
            }
        if text_has_any(text, ["lifetime", "srh", "寿命", "复合"]):
            return {
                "tool_name": "diode_breakdown_leakage_sweep",
                "base_request": {"start": 0.0, "stop": -10.0, "step": 0.5},
                "axis_path": "electron_lifetime_s",
                "values": [1.0e-9, 1.0e-8, 1.0e-7],
                "metric_path": "quality_report.metrics.leakage_abs_current_at_target_a",
                "relative_tolerance": 1.0,
            }
        return {
            "tool_name": "diode_breakdown_leakage_sweep",
            "base_request": {"start": 0.0, "stop": -5.0, "step": 0.5},
            "axis_path": "junction_spacing_um",
            "values": [2e-5, 1e-5, 5e-6],
            "metric_path": "quality_report.metrics.leakage_abs_current_at_target_a",
            "relative_tolerance": 0.1,
        }
    return {
        "tool_name": "pn_junction_iv_sweep",
        "base_request": {"start": 0.0, "stop": 0.5, "step": 0.1},
        "axis_path": "junction_spacing_um",
        "values": [2e-5, 1e-5, 5e-6],
        "metric_path": "quality_report.metrics.final_total_current_a",
        "relative_tolerance": 0.05,
    }


def deterministic_decompose_goal(goal_text: str, *, plan_id: str | None = None) -> GoalDecompositionResult:
    steps: list[GoalStep] = []
    assumptions: list[str] = []
    warnings: list[str] = []
    intent = parse_engineering_intent(goal_text)
    spec = parse_tcad_spec(goal_text)

    route = route_device_goal(goal_text)
    if needs_clarification_before_execution(intent):
        steps.append(
            GoalStep(
                index=1,
                kind=GoalStepKind.ASK_USER,
                title="Clarify TCAD mission before execution",
                request={
                    "engineering_intent": intent.model_dump(mode="json"),
                    "tcad_spec": spec.model_dump(mode="json"),
                    "questions": intent.clarification_questions,
                    "question": "当前自然语言目标还不足以安全执行 TCAD。请补充器件/结构、分析类型、关键指标和规格。",
                },
                requires_user_confirmation=True,
            )
        )
        warnings.append("目标缺少器件、分析类型或指标；已停止在澄清阶段，避免盲目运行。")
        return GoalDecompositionResult(
            status=DecompositionStatus.COMPLETED,
            goal_text=goal_text,
            plan_id=plan_id,
            steps=steps,
            assumptions=[f"工程意图：{intent.summary_zh}", *intent.assumptions],
            warnings=warnings,
        )

    if intent.support == DeviceSupport.PLANNED:
        steps.append(
            GoalStep(
                index=1,
                kind=GoalStepKind.ASK_USER,
                title=f"确认尚未实现的工业 TCAD 模板：{intent.device_family}",
                request={
                    "engineering_intent": intent.model_dump(mode="json"),
                    "tcad_spec": spec.model_dump(mode="json"),
                    "question": "该器件/结构已进入能力目录，但还没有可执行 DEVSIM runner。请确认是否先实现模板、runner、质量检查和 benchmark。",
                },
                requires_user_confirmation=True,
            )
        )
        warnings.append(f"目标匹配到尚未实现的工业器件模板：{intent.device_family}。")
        return GoalDecompositionResult(
            status=DecompositionStatus.COMPLETED,
            goal_text=goal_text,
            plan_id=plan_id,
            steps=steps,
            warnings=warnings,
        )

    if route.status == RouteStatus.MATCHED and route.template and route.template.support == TemplateSupport.PLANNED:
        steps.append(
            GoalStep(
                index=1,
                kind=GoalStepKind.ASK_USER,
                title=f"Confirm planned device template: {route.template.display_name}",
                request={
                    "template": route.template.model_dump(mode="json"),
                    "message": route.message,
                    "question": "This device template is known but not executable yet. Confirm whether to implement the runner/quality/benchmark path first.",
                },
                requires_user_confirmation=True,
            )
        )
        warnings.append(route.message)
        return GoalDecompositionResult(
            status=DecompositionStatus.COMPLETED,
            goal_text=goal_text,
            plan_id=plan_id,
            steps=steps,
            warnings=warnings,
        )

    if intent.support == DeviceSupport.COMPACT_BASELINE:
        warnings.extend(intent.capability_warnings)
        warnings.append("已允许 compact baseline 继续执行，但最终结论必须降级为规划/探索证据。")

    if text_has_any(goal_text, ["history", "历史", "检索", "已有"]):
        steps.append(
            GoalStep(
                index=len(steps) + 1,
                kind=GoalStepKind.QUERY_HISTORY,
                title="Inspect prior TCAD experiments",
                request={"limit": 20, "engineering_intent": intent.model_dump(mode="json"), "tcad_spec": spec.model_dump(mode="json")},
                stop_on_failure=False,
            )
        )

    primary_index = len(steps) + 1
    steps.append(primary_goal_step(goal_text).model_copy(update={"index": primary_index}))

    route_requires_convergence = route.status == RouteStatus.MATCHED and any(
        "convergence" in item for item in route.signoff_workflow + route.recommended_convergence
    )
    mutation_convergence_requests = deck_mutation_convergence_requests(goal_text, intent.suggested_tool, intent.request_hint)
    needs_convergence = route_requires_convergence or bool(
        {"mesh_convergence", "model_ab", "engineering_signoff", "curve_shape", "unit_check"}
        & set(intent.evidence_requirements)
    ) or bool(mutation_convergence_requests) or text_has_any(
        goal_text,
        [
            "convergence",
            "mesh",
            "model",
            "split",
            "corner",
            "a/b",
            "ab",
            "dibl",
            "mobility",
            "interface trap",
            "dit",
            "lifetime",
            "tox",
            "signoff",
            "可信",
            "签核",
            "项目会",
            "异常",
            "kink",
            "unit",
            "单位",
            "曲线形状",
            "物理可信",
            "模型收敛",
            "网格",
            "对比",
            "扫描",
            "扫参",
            "界面态",
            "迁移率",
            "寿命",
            "高温",
            "field plate",
            "field-plate",
            "drift doping",
            "drift region",
            "场板",
            "漂移区",
        ],
    )
    if needs_convergence:
        convergence_requests = mutation_convergence_requests or [default_tool_convergence_request(goal_text)]
        for convergence_request in convergence_requests:
            mutation = convergence_request.get("deck_mutation") or {}
            title = (
                f"Run deck mutation sweep: {mutation.get('target')}"
                if isinstance(mutation, dict) and mutation.get("target")
                else "Run mesh/model convergence study"
            )
            steps.append(
                GoalStep(
                    index=len(steps) + 1,
                    kind=GoalStepKind.RUN_TOOL_CONVERGENCE,
                    title=title,
                    request=convergence_request,
                    depends_on=[primary_index],
                    stop_on_failure=False,
                )
            )
    benchmark_dep = steps[-1].index
    if spec.measured_or_golden_reference:
        steps.append(
            GoalStep(
                index=len(steps) + 1,
                kind=GoalStepKind.RUN_GOLDEN_COMPARISON,
                title="Compare TCAD curve with golden/measured reference",
                request={
                    "reference_curve_path": spec.measured_or_golden_reference,
                    "tcad_spec": spec.model_dump(mode="json"),
                    "engineering_intent": intent.model_dump(mode="json"),
                },
                depends_on=[benchmark_dep],
                stop_on_failure=False,
            )
        )
        benchmark_dep = steps[-1].index
    steps.append(
        GoalStep(
            index=len(steps) + 1,
            kind=GoalStepKind.RUN_PHYSICAL_BENCHMARK,
            title="Run physical benchmark and sanity checks",
            request={"engineering_intent": intent.model_dump(mode="json"), "tcad_spec": spec.model_dump(mode="json")},
            depends_on=[benchmark_dep],
            stop_on_failure=False,
        )
    )

    steps.append(
        GoalStep(
            index=len(steps) + 1,
            kind=GoalStepKind.RUN_REPAIR_EXECUTOR,
            title="Repair suspicious or failed result if needed",
            request={
                "max_rounds": 4 if intent.risk_level == "high" else 3,
                "allow_user_confirmation_actions": False,
                "engineering_intent": intent.model_dump(mode="json"),
                "tcad_spec": spec.model_dump(mode="json"),
            },
            depends_on=[steps[-1].index],
            stop_on_failure=False,
        )
    )

    if text_has_any(goal_text, ["conclusion", "结论", "报告", "总结", "解释", "趋势", "建议"]) or True:
        steps.append(
            GoalStep(
                index=len(steps) + 1,
                kind=GoalStepKind.GENERATE_CONCLUSION,
            title="Generate engineering conclusion",
            request={"engineering_intent": intent.model_dump(mode="json"), "tcad_spec": spec.model_dump(mode="json")},
            depends_on=[steps[-1].index],
            stop_on_failure=False,
            )
        )
    if len(steps) <= 2:
        assumptions.append("Defaulted to primary supervisor execution, repair-if-needed, and final conclusion.")
    assumptions.append(f"工程意图：{intent.summary_zh}")
    assumptions.extend(intent.assumptions)
    return GoalDecompositionResult(
        status=DecompositionStatus.COMPLETED,
        goal_text=goal_text,
        plan_id=plan_id,
        steps=steps,
        assumptions=assumptions,
        warnings=warnings,
    )


def normalize_steps(raw_steps: Any, warnings: list[str]) -> list[GoalStep]:
    if not isinstance(raw_steps, list):
        warnings.append("LLM 响应中没有 steps 列表。")
        return []
    steps: list[GoalStep] = []
    for index, raw in enumerate(raw_steps, start=1):
        if not isinstance(raw, dict):
            warnings.append(f"已忽略第 {index} 个非对象步骤。")
            continue
        data = dict(raw)
        data["index"] = int(data.get("index") or index)
        normalize_goal_step_request(data, warnings)
        try:
            steps.append(GoalStep.model_validate(data))
        except ValidationError as exc:
            warnings.append(f"已忽略第 {index} 个无效步骤：{exc}")
    return steps


def normalize_goal_step_request(data: dict[str, Any], warnings: list[str]) -> None:
    if data.get("kind") != GoalStepKind.RUN_TOOL_CONVERGENCE.value:
        return
    request = data.get("request")
    if not isinstance(request, dict):
        return
    tool_name = request.get("tool_name")
    base_request = request.get("base_request")
    axis_path = request.get("axis_path")
    values = request.get("values")
    metric_path = request.get("metric_path") or "quality_report.metrics.max_abs_current_a"
    if not isinstance(tool_name, str) or not isinstance(base_request, dict) or not isinstance(axis_path, str):
        return
    if not isinstance(values, list):
        return
    normalized = normalize_tool_convergence_payload(tool_name, base_request, axis_path, values, str(metric_path))
    if (
        normalized["base_request"] != base_request
        or normalized["axis_path"] != axis_path
        or normalized["values"] != values
        or normalized["metric_path"] != metric_path
    ):
        warnings.append(f"已归一化 {tool_name} 的工具收敛 request。")
    request.update(normalized)


def build_messages(goal_text: str, baseline: GoalDecompositionResult) -> tuple[str, str]:
    system = (
        "你是谨慎的 TCAD 长程任务拆解器。"
        "只返回 JSON，生成由受支持 step kind 组成、可持久恢复的执行计划。"
        "不要包含 shell 命令；如果不确定，优先使用 run_supervisor + repair + conclusion。"
        "对于 mosfet_2d_id_sweep，sweep_type 只能是 idvg、idvd 或 both；用户说输出特性时应映射为 idvd。"
        "对于 mesh convergence，优先使用 x_divisions、silicon_y_divisions 等真实可执行字段，不要写口语字段。"
        "真实 TCAD 工程任务常会提 Vth、SS、Ion/Ioff、BV、漏电、固定氧化层电荷、kink、签核、单位检查或 golden curve；请保守映射到已支持工具。"
    )
    user = {
        "task": "拆解长程 TCAD 自主任务",
        "supported_step_kinds": [kind.value for kind in GoalStepKind],
        "step_schema": {
            "index": "integer",
            "kind": [kind.value for kind in GoalStepKind],
            "title": "中文短标题",
            "request": "object",
            "depends_on": ["依赖的 step index"],
            "stop_on_failure": "boolean",
            "requires_user_confirmation": "boolean",
        },
        "baseline_plan": baseline.model_dump(mode="json"),
        "goal_text": goal_text,
    }
    return system, json.dumps(user, ensure_ascii=False, indent=2)


def deterministic_replan_after_issue(
    goal_text: str,
    *,
    issue_context: dict[str, Any],
    plan_id: str | None = None,
) -> ReplanDecision:
    issue_text = json.dumps(issue_context, ensure_ascii=False)
    if text_has_any(issue_text, ["validation", "字段", "sweep_type", "pattern", "schema", "alias", "field"]):
        issue_family = "schema_or_field_alias"
        control_action = "replan"
        strategy = "检测到字段/别名/Schema 兼容问题：优先做字段归一化；如果失败步骤是可选检查，则标记为非阻塞并继续修复判断和工程结论。"
        recommended_actions = ["normalize_tool_fields", "rerun_optional_check_when_safe", "continue_with_primary_evidence"]
    elif text_has_any(issue_text, ["convergence", "too_few_completed_convergence_cases", "maximum_iterations", "收敛"]):
        issue_family = "solver_convergence"
        control_action = "replan"
        strategy = "检测到收敛链路问题：优先缩小 bias step 或使用 continuation ramp；可选收敛检查失败时不要阻塞主结论，但必须在结论中标注可信度风险。"
        recommended_actions = ["shrink_bias_step", "continuation_bias_ramp", "mesh_or_solver_repair", "mark_optional_convergence_soft_failed"]
    elif text_has_any(issue_text, ["benchmark", "quality", "physical", "suspicious", "物理", "曲线", "unit", "单位"]):
        issue_family = "physical_quality"
        control_action = "repair_or_verify"
        strategy = "检测到物理质量风险：保留数值结果但降低可信度，优先执行 mesh/model/单位/曲线形状检查，再生成带风险说明的工程结论。"
        recommended_actions = ["run_physical_benchmark", "run_mesh_or_model_convergence", "execute_tcad_repair", "generate_risk_aware_conclusion"]
    elif text_has_any(issue_text, ["repair", "修复"]):
        issue_family = "repair_exhausted"
        control_action = "ask_or_finish_with_risk"
        strategy = "自动修复没有产生可信新结果：保留主仿真证据，在结论中标注修复风险和下一轮建议。"
        recommended_actions = ["preserve_primary_evidence", "summarize_failed_repairs", "propose_minimal_next_experiment"]
    else:
        issue_family = "execution"
        control_action = "replan"
        strategy = "执行链路出现问题：保留已完成证据，优先继续可安全完成的后续步骤，并把异常写入结论。"
        recommended_actions = ["preserve_completed_evidence", "continue_safe_steps", "explain_failure_chain"]
    return ReplanDecision(
        status=DecompositionStatus.FALLBACK,
        goal_text=goal_text,
        plan_id=plan_id,
        issue_family=issue_family,
        control_action=control_action,
        strategy_zh=strategy,
        recommended_actions=recommended_actions,
        append_steps=[],
        warnings=["使用确定性再编排兜底。"],
        fallback_used=True,
    )


def normalize_replan_append_steps(raw_steps: Any, warnings: list[str]) -> list[GoalStep]:
    steps = normalize_steps(raw_steps, warnings)
    for step in steps:
        step.stop_on_failure = bool(step.stop_on_failure)
    return steps


def build_replan_messages(
    goal_text: str,
    *,
    current_plan: dict[str, Any],
    goal_step_statuses: dict[str, Any],
    issue_context: dict[str, Any],
    current_evidence: dict[str, Any] | None,
) -> tuple[str, str]:
    system = (
        "你是 TCAD 自主 agent 的总控，采用 observe-diagnose-plan-act-verify 循环。只返回 JSON。"
        "请诊断字段名不一致、工具调用失败、质量检查失败、DAG 步骤阻塞等执行问题。"
        "优先修正计划并安全继续，不要轻易询问用户。"
        "只有物理含义确实模糊、缺少外部实测数据、或几何/模型变更风险较高时，才向用户确认。"
        "对于 mosfet_2d_id_sweep，sweep_type 必须是 idvg、idvd 或 both；用户说输出特性时应映射为 idvd。"
        "如果是 schema/alias 失败，先归一化字段并重试，或仅把可选检查标记为非阻塞失败。"
        "如果是物理质量问题，优先选择 repair/convergence/conclusion，保留当前证据并解释风险。"
        "issue_context 里可能包含 tcad_deck_spec、quality_report、benchmark evidence_matrix 和 repair attempts；必须据此判断是模型耦合、收敛证据、曲线形状还是实测/golden 缺口。"
        "strategy_zh、title 和 warnings 尽量使用中文。"
    )
    user = {
        "task": "TCAD 执行异常后的自动再编排",
        "supported_step_kinds": [kind.value for kind in GoalStepKind],
        "response_schema": {
            "issue_family": "schema_or_field_alias、solver_convergence、physical_quality、repair_exhausted、execution 之一",
            "control_action": "continue、replan、repair_or_verify、ask_user、finish_with_risk 之一",
            "strategy_zh": "简短中文诊断和下一步策略",
            "recommended_actions": ["简短机器可读 action name"],
            "mark_soft_failed": ["应视为非阻塞失败的 goal step index"],
            "skip_goal_steps": ["可以安全跳过的 goal step index"],
            "append_steps": [
                {
                    "index": "临时整数；执行器会重新编号",
                    "kind": [kind.value for kind in GoalStepKind],
                    "title": "中文短标题",
                    "request": "object",
                    "depends_on": ["已有或临时 step index"],
                    "stop_on_failure": "boolean",
                    "requires_user_confirmation": "boolean",
                }
            ],
        },
        "rules": [
            "除非问题发生在主 TCAD 仿真本身，否则不要重复已完成的主仿真。",
            "物理 benchmark 失败只是总控证据，不等于最终真相；请选择修复/收敛验证，或生成带风险说明的结论。",
            "如果 evidence_matrix 显示 deck_spec 或 convergence/golden/measured 证据缺失，不要给强签核结论；优先追加验证或保留风险。",
            "如果 tcad_deck_spec 的 physics_models.coupling_status 需要确认，优先追加模型耦合/提取复核或修复步骤。",
            "可选收敛检查失败时，通常把该收敛步骤标记为非阻塞失败，并继续必要修复和结论生成。",
            "归一化真实可执行的工具 request 字段，不要编造工具 schema 字段。",
            "append_steps 保持最小化。",
        ],
        "goal_text": goal_text,
        "current_plan": current_plan,
        "goal_step_statuses": goal_step_statuses,
        "current_evidence": current_evidence,
        "issue_context": issue_context,
    }
    return system, json.dumps(user, ensure_ascii=False, indent=2)


def replan_goal_after_issue(
    goal_text: str,
    *,
    current_plan: dict[str, Any],
    goal_step_statuses: dict[str, Any],
    issue_context: dict[str, Any],
    current_evidence: dict[str, Any] | None = None,
    plan_id: str | None = None,
    client: ChatClient | None = None,
    allow_fallback: bool = True,
) -> ReplanDecision:
    chat_client = client or LLMClient()
    system, user = build_replan_messages(
        goal_text,
        current_plan=current_plan,
        goal_step_statuses=goal_step_statuses,
        issue_context=issue_context,
        current_evidence=current_evidence,
    )
    try:
        raw_response = chat_client.chat(system=system, user=user, temperature=0.1)
    except Exception as exc:
        if allow_fallback:
            fallback = deterministic_replan_after_issue(goal_text, issue_context=issue_context, plan_id=plan_id)
            return fallback.model_copy(
                update={
                    "model": getattr(chat_client.config, "model", None),
                    "warnings": [f"LLM 再编排失败：{exc}"] + fallback.warnings,
                }
            )
        return ReplanDecision(
            status=DecompositionStatus.FAILED,
            goal_text=goal_text,
            plan_id=plan_id,
            model=getattr(chat_client.config, "model", None),
            strategy_zh="再编排失败，需要用户确认下一步。",
            validation_errors=[str(exc)],
        )

    parsed = parse_json_object(raw_response)
    if parsed is None:
        if allow_fallback:
            fallback = deterministic_replan_after_issue(goal_text, issue_context=issue_context, plan_id=plan_id)
            return fallback.model_copy(
                update={
                    "model": getattr(chat_client.config, "model", None),
                    "raw_response": raw_response,
                    "warnings": ["LLM 再编排没有返回 JSON 对象。"] + fallback.warnings,
                }
            )
        return ReplanDecision(
            status=DecompositionStatus.FAILED,
            goal_text=goal_text,
            plan_id=plan_id,
            model=getattr(chat_client.config, "model", None),
            raw_response=raw_response,
            strategy_zh="再编排失败：模型没有返回 JSON。",
            validation_errors=["LLM 再编排没有返回 JSON 对象。"],
        )

    warnings: list[str] = []
    append_steps = normalize_replan_append_steps(parsed.get("append_steps") or [], warnings)
    return ReplanDecision(
        status=DecompositionStatus.COMPLETED,
        goal_text=goal_text,
        plan_id=plan_id,
        model=getattr(chat_client.config, "model", None),
        raw_response=raw_response,
        parsed_response=parsed,
        issue_family=str(parsed.get("issue_family") or parsed.get("failure_family") or "execution"),
        control_action=str(parsed.get("control_action") or parsed.get("action") or "replan"),
        strategy_zh=str(parsed.get("strategy_zh") or parsed.get("strategy") or "已根据执行问题调整后续计划。"),
        recommended_actions=[str(item) for item in parsed.get("recommended_actions") or []],
        append_steps=append_steps,
        mark_soft_failed=[
            int(item)
            for item in parsed.get("mark_soft_failed") or []
            if isinstance(item, int) or (isinstance(item, str) and item.isdigit())
        ],
        skip_goal_steps=[
            int(item)
            for item in parsed.get("skip_goal_steps") or []
            if isinstance(item, int) or (isinstance(item, str) and item.isdigit())
        ],
        warnings=warnings + [str(item) for item in parsed.get("warnings") or []],
    )


def decompose_goal_with_llm(
    goal_text: str,
    *,
    plan_id: str | None = None,
    client: ChatClient | None = None,
    allow_fallback: bool = True,
) -> GoalDecompositionResult:
    baseline = deterministic_decompose_goal(goal_text, plan_id=plan_id)
    chat_client = client or LLMClient()
    system, user = build_messages(goal_text, baseline)
    try:
        raw_response = chat_client.chat(system=system, user=user, temperature=0.1)
    except Exception as exc:
        if allow_fallback:
            fallback = baseline.model_copy(
                update={
                    "status": DecompositionStatus.FALLBACK,
                    "model": getattr(chat_client.config, "model", None),
                    "warnings": [f"LLM 任务拆解失败：{exc}"],
                    "fallback_used": True,
                }
            )
            return fallback
        return GoalDecompositionResult(
            status=DecompositionStatus.FAILED,
            goal_text=goal_text,
            plan_id=plan_id,
            model=getattr(chat_client.config, "model", None),
            validation_errors=[str(exc)],
        )

    parsed = parse_json_object(raw_response)
    if parsed is None:
        if allow_fallback:
            return baseline.model_copy(
                update={
                    "status": DecompositionStatus.FALLBACK,
                    "model": getattr(chat_client.config, "model", None),
                    "raw_response": raw_response,
                    "warnings": ["LLM 没有返回 JSON 对象。"],
                    "fallback_used": True,
                }
            )
        return GoalDecompositionResult(
            status=DecompositionStatus.FAILED,
            goal_text=goal_text,
            plan_id=plan_id,
            model=getattr(chat_client.config, "model", None),
            raw_response=raw_response,
            validation_errors=["LLM 没有返回 JSON 对象。"],
        )

    warnings: list[str] = []
    steps = normalize_steps(parsed.get("steps"), warnings)
    if not steps:
        if allow_fallback:
            return baseline.model_copy(
                update={
                    "status": DecompositionStatus.FALLBACK,
                    "model": getattr(chat_client.config, "model", None),
                    "raw_response": raw_response,
                    "parsed_response": parsed,
                    "warnings": warnings or ["LLM 计划中没有可用步骤。"],
                    "fallback_used": True,
                }
            )
    return GoalDecompositionResult(
        status=DecompositionStatus.COMPLETED,
        goal_text=goal_text,
        plan_id=plan_id or parsed.get("plan_id"),
        model=getattr(chat_client.config, "model", None),
        raw_response=raw_response,
        parsed_response=parsed,
        steps=steps,
        assumptions=[str(item) for item in parsed.get("assumptions") or []],
        warnings=warnings + [str(item) for item in parsed.get("warnings") or []],
    )


def write_decomposition_result(result: GoalDecompositionResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decompose a long-horizon TCAD goal into durable agent steps.")
    parser.add_argument("--goal", required=True)
    parser.add_argument("--plan-id", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    return parser.parse_args()


def default_output(plan_id: str | None) -> Path:
    actual = plan_id or "goal_plan"
    return PROJECT_ROOT / "runs" / "goal_plans" / actual / "goal_decomposition.json"


def main() -> None:
    args = parse_args()
    if args.use_llm:
        result = decompose_goal_with_llm(
            args.goal,
            plan_id=args.plan_id,
            allow_fallback=not args.no_fallback,
        )
    else:
        result = deterministic_decompose_goal(args.goal, plan_id=args.plan_id)
    output = args.output or default_output(result.plan_id or args.plan_id)
    write_decomposition_result(result, output)
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.status != DecompositionStatus.FAILED else 1)


if __name__ == "__main__":
    main()
