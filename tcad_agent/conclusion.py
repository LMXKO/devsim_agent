from __future__ import annotations

import json
import math
import statistics
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from tcad_agent.reporting import final_artifacts, final_metrics, load_final_state, read_json, resolve_state_path, write_text


class ConclusionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class ConclusionResult(BaseModel):
    tool_name: str = "experiment_conclusion"
    status: ConclusionStatus
    source_state_path: str
    conclusion_path: str | None = None
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return ", ".join(format_value(item) for item in value)
    if isinstance(value, dict):
        return "`" + json.dumps(value, ensure_ascii=False, sort_keys=True) + "`"
    return str(value)


STATUS_LABELS_ZH = {
    "completed": "已完成",
    "failed": "失败",
    "passed": "通过",
    "suspicious": "可疑",
    "unsupported": "暂不支持",
    "planned": "已计划",
    "unknown": "未知",
    "warning": "告警",
    "error": "错误",
}


def format_status(value: Any) -> str:
    text = str(value or "unknown")
    return STATUS_LABELS_ZH.get(text, text)


def objective_direction(state: dict[str, Any]) -> str:
    objective = state.get("objective") or {}
    return str(objective.get("direction") or "minimize")


def sorted_items(items: list[dict[str, Any]], direction: str) -> list[dict[str, Any]]:
    eligible = [item for item in items if item.get("objective_value") is not None]
    reverse = direction == "maximize"
    return sorted(eligible, key=lambda item: float(item.get("objective_value") or 0.0), reverse=reverse)


def item_axis_value(item: dict[str, Any], axis_name: str | None) -> float | None:
    if "value" in item:
        try:
            return float(item["value"])
        except (TypeError, ValueError):
            return None
    values = item.get("values") or {}
    if axis_name and axis_name in values:
        try:
            return float(values[axis_name])
        except (TypeError, ValueError):
            return None
    return None


def axis_names_for_state(state: dict[str, Any]) -> list[str]:
    if state.get("tool_name") == "adaptive_optimizer":
        axis_name = (state.get("axis") or {}).get("path")
        return [axis_name] if axis_name else []
    axes = state.get("axes") or []
    return [axis.get("path", "") for axis in axes if axis.get("path")]


def axis_name_for_state(state: dict[str, Any]) -> str | None:
    axis_names = axis_names_for_state(state)
    if not axis_names:
        return None
    return axis_names[0] if len(axis_names) == 1 else ", ".join(axis_names)


def item_axis_display(item: dict[str, Any], axis_names: list[str]) -> Any:
    if not axis_names:
        return None
    if len(axis_names) == 1:
        return item_axis_value(item, axis_names[0])
    values = item.get("values") or {}
    if values:
        return {axis_name: values.get(axis_name) for axis_name in axis_names}
    return None


def experiment_items(state: dict[str, Any]) -> list[dict[str, Any]]:
    if state.get("tool_name") in {"adaptive_optimizer", "multidim_optimizer"}:
        return state.get("observations") or []
    if state.get("tool_name") == "parameter_sweep":
        return state.get("cases") or []
    if state.get("quality_report"):
        metrics = (state.get("quality_report") or {}).get("metrics") or {}
        objective = metrics.get("final_total_current_a")
        if objective is None:
            objective = metrics.get("final_capacitance_f_per_cm2")
        if objective is None:
            objective = metrics.get("leakage_abs_current_at_target_a")
        if objective is None:
            objective = metrics.get("breakdown_voltage_at_threshold_v")
        if objective is None:
            objective = metrics.get("relative_delta")
        if objective is None:
            objective = metrics.get("ion_ioff_ratio")
        if objective is None:
            objective = metrics.get("max_abs_drain_current_a")
        if objective is None:
            objective = metrics.get("barrier_height_ev")
        if objective is None:
            objective = metrics.get("current_gain_beta")
        if objective is None:
            objective = metrics.get("pinch_off_voltage_v")
        if objective is None:
            objective = metrics.get("breakdown_voltage_v")
        if objective is None:
            objective = metrics.get("responsivity_a_per_w")
        return [
            {
                "task_id": state.get("run_id"),
                "status": state.get("status"),
                "quality_status": (state.get("quality_report") or {}).get("status"),
                "objective_value": objective,
                "final_state_path": None,
                "metrics": metrics,
            }
        ]
    return []


def best_item(state: dict[str, Any]) -> dict[str, Any] | None:
    if state.get("best_observation"):
        return state["best_observation"]
    if state.get("best_case"):
        return state["best_case"]
    ranked = sorted_items(experiment_items(state), objective_direction(state))
    return ranked[0] if ranked else None


def quality_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(item.get("quality_status") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def status_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(item.get("status") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def trend_sentence(
    items: list[dict[str, Any]],
    axis_names: list[str],
    direction: str,
    best: dict[str, Any] | None = None,
) -> str:
    axis_name = axis_names[0] if len(axis_names) == 1 else None
    if len(axis_names) > 1:
        if best:
            return (
                "这是多参数响应面。当前采样中最优参数组合为 "
                f"{format_value(item_axis_display(best, axis_names))}，目标值为 "
                f"{format_value(best.get('objective_value'))}；把一维单调趋势当作因果结论前，"
                "需要先查看 heatmap/dashboard。"
            )
        return "这是多参数响应面；已完成观测点不足，暂时无法判断最优区域。"
    points = [
        (item_axis_value(item, axis_name), float(item.get("objective_value")))
        for item in items
        if item.get("objective_value") is not None and item_axis_value(item, axis_name) is not None
    ]
    points = [(x, y) for x, y in points if x is not None]
    if len(points) < 2 or not axis_name:
        return "轴向观测点不足，暂时不能判断趋势。"
    ordered = sorted(points, key=lambda pair: pair[0])
    first_x, first_y = ordered[0]
    last_x, last_y = ordered[-1]
    improves = last_y < first_y if direction == "minimize" else last_y > first_y
    if improves:
        return (
            f"在当前采样范围内，{axis_name} 从 {format_value(first_x)} 变化到 {format_value(last_x)} 时，"
            f"目标值从 {format_value(first_y)} 改善到 {format_value(last_y)}。"
        )
    return (
        f"在当前采样范围内，{axis_name} 从 {format_value(first_x)} 变化到 {format_value(last_x)} 时，"
        f"目标值未改善（{format_value(first_y)} -> {format_value(last_y)}）。"
    )


def objective_points(items: list[dict[str, Any]], axis_names: list[str]) -> list[tuple[float | None, float, dict[str, Any]]]:
    axis_name = axis_names[0] if len(axis_names) == 1 else None
    points: list[tuple[float | None, float, dict[str, Any]]] = []
    for item in items:
        objective = item.get("objective_value")
        if objective is None:
            continue
        try:
            y = float(objective)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(y):
            continue
        points.append((item_axis_value(item, axis_name), y, item))
    return points


def trend_assessment(items: list[dict[str, Any]], axis_names: list[str], direction: str) -> dict[str, Any]:
    points = objective_points(items, axis_names)
    if len(axis_names) != 1:
        confidence = "medium" if len(points) >= 4 else "low"
        return {
            "kind": "multi_axis" if len(axis_names) > 1 else "not_axis_resolved",
            "confidence": confidence,
            "completed_points": len(points),
            "monotonicity": "not_applicable",
            "local_reversals": 0,
        }
    ordered = sorted([point for point in points if point[0] is not None], key=lambda point: float(point[0] or 0.0))
    if len(ordered) < 2:
        return {
            "kind": "single_axis",
            "confidence": "low",
            "completed_points": len(ordered),
            "monotonicity": "insufficient_points",
            "local_reversals": 0,
        }
    signed = [point[1] if direction == "minimize" else -point[1] for point in ordered]
    deltas = [right - left for left, right in zip(signed[:-1], signed[1:])]
    improves = sum(1 for delta in deltas if delta < 0)
    worsens = sum(1 for delta in deltas if delta > 0)
    reversals = sum(1 for left, right in zip(deltas[:-1], deltas[1:]) if left * right < 0)
    if improves and not worsens:
        monotonicity = "monotonic_improving"
    elif worsens and not improves:
        monotonicity = "monotonic_degrading"
    elif reversals:
        monotonicity = "nonmonotonic"
    else:
        monotonicity = "flat_or_mixed"
    confidence = "high" if len(ordered) >= 4 and reversals == 0 else "medium" if len(ordered) >= 3 else "low"
    return {
        "kind": "single_axis",
        "confidence": confidence,
        "completed_points": len(ordered),
        "monotonicity": monotonicity,
        "local_reversals": reversals,
        "improving_steps": improves,
        "degrading_steps": worsens,
    }


def anomaly_lines(items: list[dict[str, Any]], axis_names: list[str], direction: str) -> list[str]:
    lines: list[str] = []
    for item in items:
        label = item.get("task_id") or item.get("run_id") or "unknown"
        if item.get("status") == "failed":
            lines.append(f"`{label}` 已失败，应从设计决策证据中剔除。")
        if item.get("quality_status") == "suspicious":
            lines.append(f"`{label}` 质量可疑，作为证据前需要检查对应产物。")
        if item.get("objective_value") is None:
            lines.append(f"`{label}` 没有目标值，因此未参与排序。")

    points = objective_points(items, axis_names)
    values = [point[1] for point in points]
    if len(values) >= 4:
        median_value = statistics.median(values)
        if median_value != 0:
            for _, value, item in points:
                ratio = abs(value / median_value)
                is_outlier = ratio > 10.0 or ratio < 0.1
                if is_outlier:
                    label = item.get("task_id") or item.get("run_id") or "unknown"
                    lines.append(
                        f"`{label}` 的目标值 `{format_value(value)}` 明显偏离中位数 `{format_value(median_value)}`；需要复核收敛和单位。"
                    )

    assessment = trend_assessment(items, axis_names, direction)
    if assessment.get("monotonicity") == "nonmonotonic":
        lines.append("单轴趋势存在非单调变化；在声称因果趋势前，应先做局部加密验证。")
    return lines


def benchmark_digest(source_path: Path) -> dict[str, Any]:
    try:
        from tcad_agent.physical_benchmark import run_physical_benchmark

        result = run_physical_benchmark(source_path)
        data = result.model_dump(mode="json")
        checks = data.get("checks") or []
        return {
            "status": data.get("status"),
            "benchmark_path": data.get("benchmark_path"),
            "counts": (data.get("summary") or {}).get("counts") or {},
            "signoff_status": (data.get("summary") or {}).get("signoff_status"),
            "signoff_label_zh": (data.get("summary") or {}).get("signoff_label_zh"),
            "confidence_score": (data.get("summary") or {}).get("confidence_score"),
            "credibility": (data.get("summary") or {}).get("credibility") or {},
            "recommended_next_action_zh": (data.get("summary") or {}).get("recommended_next_action_zh"),
            "evidence_matrix": (data.get("summary") or {}).get("evidence_matrix") or {},
            "warning_codes": [check.get("code") for check in checks if check.get("severity") == "warning"][:5],
            "error_codes": [check.get("code") for check in checks if check.get("severity") == "error"][:5],
            "failure_reason": data.get("failure_reason"),
        }
    except Exception as exc:
        return {"status": "failed", "failure_reason": str(exc), "counts": {}, "warning_codes": [], "error_codes": []}


def engineering_decision(
    state: dict[str, Any],
    q_counts: dict[str, int],
    s_counts: dict[str, int],
    benchmark: dict[str, Any],
) -> str:
    benchmark_status = benchmark.get("status")
    credibility = benchmark.get("credibility") or {}
    if credibility.get("level") == "blocked":
        return "暂不信任该结果：可信度评审已阻塞，需要先修复错误级证据。"
    if state.get("status") == "failed" or q_counts.get("failed") or benchmark_status == "failed":
        return "暂不信任该结果：至少有一个质量检查或物理 benchmark 失败项需要先解决。"
    if credibility.get("level") == "limited":
        return "只能作为探索性结果：证据偏少，不能直接支撑强工程判断。"
    if s_counts.get("failed") or q_counts.get("suspicious") or benchmark_status in {"suspicious", "unsupported"}:
        return "可作为下一步规划线索，但在作为工程证据前，需要先完成建议的验证或局部细化。"
    return "可作为当前 TCAD 迭代的基线结果。"


def next_experiment_packages(
    state: dict[str, Any],
    items: list[dict[str, Any]],
    best: dict[str, Any] | None,
    axis_names: list[str],
    assessment: dict[str, Any],
) -> list[dict[str, Any]]:
    tool_name = state.get("tool_name")
    packages: list[dict[str, Any]] = []
    best_axis = item_axis_display(best, axis_names) if best else None
    if assessment.get("confidence") == "low":
        packages.append(
            {
                "title": "增加证据密度",
                "tool_name": "tool_convergence" if tool_name not in {"parameter_sweep", "adaptive_optimizer", "multidim_optimizer"} else "parameter_sweep",
                "rationale": "已完成点数太少，趋势置信度偏低。",
                "request_hint": {"seed_best_axis_value": best_axis, "add_points_near_best": True},
            }
        )
    if tool_name == "mosfet_2d_id_sweep":
        packages.extend(
            [
                {
                    "title": "MOSFET 网格/模型验证",
                    "tool_name": "tool_convergence",
                    "rationale": "验证 Id-Vg/Id-Vd 指标对网格和模型设置的敏感性。",
                    "request_hint": {
                        "tool_name": "mosfet_2d_id_sweep",
                        "axis_path": "x_divisions",
                        "values": [8, 12, 16],
                        "metric_path": "quality_report.metrics.ion_ioff_ratio",
                    },
                },
                {
                    "title": "MOSFET 设计细化",
                    "tool_name": "multidim_optimizer",
                    "rationale": "在 Vth/SS/Ion-Ioff 提取可信后，同时优化漏电和导通电流。",
                    "request_hint": {"objectives": ["maximize Ion/Ioff", "constrain leakage"], "seed_from_current_result": True},
                },
            ]
        )
    elif tool_name == "mos_capacitor_cv_sweep":
        packages.append(
            {
                "title": "MOSCAP 氧化层/掺杂扫描",
                "tool_name": "parameter_sweep",
                "rationale": "区分氧化层电容偏移与衬底掺杂效应。",
                "request_hint": {"axes": ["oxide_thickness_nm", "substrate_doping_cm3"], "seed_from_current_result": True},
            }
        )
    elif tool_name == "diode_breakdown_leakage_sweep":
        packages.append(
            {
                "title": "击穿边界细化",
                "tool_name": "tool_convergence",
                "rationale": "在漏电或击穿阈值附近细化反偏分辨率和网格。",
                "request_hint": {"axis_path": "junction_spacing_um", "metric_path": "quality_report.metrics.leakage_abs_current_at_target_a"},
            }
        )
    elif tool_name == "pn_junction_iv_sweep":
        packages.append(
            {
                "title": "PN 二极管参数扫描",
                "tool_name": "parameter_sweep",
                "rationale": "拆分掺杂、寿命和结位置对漏电/正向电流的影响。",
                "request_hint": {"axes": ["p_doping_cm3", "n_doping_cm3", "electron_lifetime_s"]},
            }
        )
    elif tool_name in {"parameter_sweep", "adaptive_optimizer", "multidim_optimizer"}:
        packages.append(
            {
                "title": "围绕最优区域细化",
                "tool_name": "adaptive_optimizer" if len(axis_names) <= 1 else "multidim_optimizer",
                "rationale": "用当前最优样本作为下一轮细化中心。",
                "request_hint": {"best_axis_value": best_axis, "reuse_best_case": True},
            }
        )
    return packages[:4]


def best_metrics(best: dict[str, Any] | None) -> dict[str, Any]:
    if not best:
        return {}
    if best.get("metrics"):
        return best["metrics"]
    final_state = load_final_state(best.get("final_state_path"))
    return final_metrics(final_state)


def artifact_summary(best: dict[str, Any] | None) -> list[str]:
    if not best:
        return []
    final_state = load_final_state(best.get("final_state_path"))
    artifacts = final_artifacts(final_state)
    return [f"- `{name}`: `{path}`" for name, path in artifacts.items()]


def recommendations(state: dict[str, Any], items: list[dict[str, Any]], best: dict[str, Any] | None) -> list[str]:
    recs: list[str] = []
    failed = [item for item in items if item.get("status") == "failed"]
    suspicious = [item for item in items if item.get("quality_status") == "suspicious"]
    if failed:
        recs.append("先检查失败 case；在信任横向对比前，用更小 bias step 或更稳健网格重跑。")
    if suspicious:
        recs.append("把可疑 case 用于优化决策前，需要先复核曲线、单位和产物。")
    if state.get("tool_name") == "adaptive_optimizer":
        recs.append("围绕当前最优值继续做若干轮自适应优化。")
    elif state.get("tool_name") == "multidim_optimizer":
        recs.append("围绕当前最优参数组合继续做多维局部细化。")
    elif state.get("tool_name") == "parameter_sweep":
        recs.append("用本轮 sweep 的最优区域作为种子启动自适应优化。")
    elif state.get("tool_name") == "mos_capacitor_cv_sweep":
        recs.append("下一步扫描氧化层厚度或衬底掺杂，解释 C-V 偏移来源。")
    elif state.get("tool_name") == "pn_junction_iv_sweep":
        recs.append("下一步扫描掺杂、结位置或温度，拆分漏电和正向电流影响。")
    elif state.get("tool_name") == "diode_breakdown_leakage_sweep":
        recs.append("扩展反偏范围，或在已提取漏电/击穿区域附近细化步长。")
    elif state.get("tool_name") == "mesh_convergence":
        recs.append("若收敛检查通过，可沿用当前网格；否则需要加密网格并重跑收敛检查。")
    elif state.get("tool_name") == "mosfet_2d_id_sweep":
        recs.append("复核 Vth、SS、Ion/Ioff 和 Id-Vd 输出电导；若有物理质量告警，需要细化几何或网格。")
    elif state.get("tool_name") == "extended_device_sweep":
        device_type = ((state.get("quality_report") or {}).get("metrics") or {}).get("device_type")
        recs.append(
            f"将 `{device_type or 'extended device'}` 的紧凑结果作为规划基线；作为最终 TCAD 证据前，需要补 golden benchmark 和收敛验证。"
        )
    if best and not recs:
        recs.append("将最优结果作为下一轮 TCAD 任务的基线。")
    return recs or ["先收集更多已完成观测点，再决定下一轮实验。"]


def render_conclusion(state: dict[str, Any], source_path: Path) -> str:
    items = experiment_items(state)
    ranked = sorted_items(items, objective_direction(state))
    best = best_item(state)
    axis_names = axis_names_for_state(state)
    axis_name = axis_name_for_state(state)
    metrics = best_metrics(best)
    q_counts = quality_counts(items)
    s_counts = status_counts(items)
    benchmark = benchmark_digest(source_path)
    assessment = trend_assessment(items, axis_names, objective_direction(state))
    anomalies = anomaly_lines(items, axis_names, objective_direction(state))
    next_packages = next_experiment_packages(state, items, best, axis_names, assessment)
    title_id = (
        state.get("optimize_id")
        or state.get("sweep_id")
        or state.get("run_id")
        or state.get("convergence_id")
        or state.get("task_id")
        or source_path.parent.name
    )

    lines = [
        f"# TCAD 工程结论：{title_id}",
        "",
        f"生成时间：{utc_timestamp()}",
        "",
        "## 摘要",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- 最优任务/结果：`{best.get('task_id') or best.get('run_id') or title_id}`。",
                f"- 目标值：`{format_value(best.get('objective_value'))}`。",
                f"- 扫描轴：`{axis_name or 'N/A'}`；最优轴取值：`{format_value(item_axis_display(best, axis_names))}`。",
                f"- 质量分布：`{format_value(q_counts)}`。",
                f"- 执行状态分布：`{format_value(s_counts)}`。",
                f"- 签核判断：`{format_value(benchmark.get('signoff_label_zh') or format_status(benchmark.get('status')))}`；置信分数 `{format_value(benchmark.get('confidence_score'))}`。",
                f"- 工程判断：{engineering_decision(state, q_counts, s_counts, benchmark)}",
            ]
        )
    else:
        lines.append("- 没有找到带目标值的已完成结果。")

    lines.extend(
        [
            "",
            "## 趋势解读",
            "",
            trend_sentence(items, axis_names, objective_direction(state), best),
            "",
            f"- 趋势置信度：`{assessment.get('confidence')}`。",
            f"- 单调性：`{assessment.get('monotonicity')}`。",
            f"- 已完成目标点数：`{assessment.get('completed_points')}`。",
            "",
            "## 关键指标",
            "",
        ]
    )
    metric_keys = [
        "final_total_current_a",
        "max_abs_current_a",
        "leakage_current_a",
        "turn_on_voltage_at_1ua_v",
        "ideality_factor_estimate",
        "rectification_ratio_final_to_leakage",
        "final_capacitance_f_per_cm2",
        "max_capacitance_f_per_cm2",
        "leakage_abs_current_at_target_a",
        "breakdown_voltage_at_threshold_v",
        "max_reverse_abs_current_a",
        "reverse_current_shape_violations",
        "relative_delta",
        "relative_tolerance",
        "finest_mesh_value",
        "previous_mesh_value",
        "finest_objective",
        "previous_objective",
        "vth_at_threshold_current_v",
        "subthreshold_swing_mv_dec",
        "ion_current_a",
        "ioff_current_a",
        "ion_ioff_ratio",
        "max_transconductance_s",
        "idvd_final_current_a",
        "output_conductance_last_s",
        "idvg_shape_violations",
        "barrier_height_ev",
        "reverse_leakage_current_a",
        "current_gain_beta",
        "early_voltage_v",
        "pinch_off_voltage_v",
        "idss_a",
        "specific_on_resistance_ohm_cm2",
        "breakdown_voltage_v",
        "max_electric_field_v_per_cm",
        "photocurrent_a",
        "responsivity_a_per_w",
        "open_circuit_voltage_v",
    ]
    metric_lines = [f"- `{key}`: `{format_value(metrics.get(key))}`" for key in metric_keys if key in metrics]
    lines.extend(metric_lines or ["- 没有可用的详细最终指标。"])

    lines.extend(["", "## 排序证据", ""])
    if ranked:
        for index, item in enumerate(ranked[:8], start=1):
            lines.append(
                f"{index}. `{item.get('task_id') or item.get('run_id')}` 目标值 `{format_value(item.get('objective_value'))}`，质量 `{format_status(item.get('quality_status'))}`。"
            )
    else:
        lines.append("没有可排序的观测点。")

    lines.extend(["", "## 物理可信度检查", ""])
    credibility = benchmark.get("credibility") or {}
    if benchmark.get("failure_reason"):
        lines.append(f"- 物理 benchmark 状态：`{format_status(benchmark.get('status'))}`；失败原因：`{benchmark.get('failure_reason')}`。")
    else:
        lines.extend(
            [
                f"- 物理 benchmark 状态：`{format_status(benchmark.get('status'))}`。",
                f"- 签核状态：`{format_value(benchmark.get('signoff_label_zh') or benchmark.get('signoff_status'))}`；置信分数：`{format_value(benchmark.get('confidence_score'))}`。",
                f"- 可信度等级：`{format_value(credibility.get('level'))}`；可信度分数：`{format_value(credibility.get('score'))}`。",
                f"- 接收建议：{format_value(credibility.get('acceptance_zh'))}。",
                f"- 证据矩阵：`{format_value(benchmark.get('evidence_matrix') or {})}`。",
                f"- 缺失证据：`{format_value(credibility.get('evidence_gaps') or [])}`。",
                f"- 签核前必须处理：`{format_value(credibility.get('must_fix_before_signoff') or [])}`。",
                f"- 检查计数：`{format_value(benchmark.get('counts'))}`。",
                f"- 告警代码：`{format_value(benchmark.get('warning_codes') or [])}`。",
                f"- 错误代码：`{format_value(benchmark.get('error_codes') or [])}`。",
                f"- benchmark 建议：{format_value(benchmark.get('recommended_next_action_zh'))}。",
                f"- benchmark 文件：`{benchmark.get('benchmark_path')}`。",
            ]
        )

    lines.extend(["", "## 异常点", ""])
    lines.extend([f"- {line}" for line in anomalies] or ["- 未发现明显的失败、可疑、缺失目标值、离群或非单调异常。"])

    artifacts = artifact_summary(best)
    if artifacts:
        lines.extend(["", "## 最优结果产物", "", *artifacts])

    lines.extend(["", "## 建议下一步", ""])
    lines.extend([f"- {item}" for item in recommendations(state, items, best)])
    if next_packages:
        lines.extend(["", "## 下一轮实验计划", ""])
        for index, package in enumerate(next_packages, start=1):
            lines.append(f"{index}. `{package['title']}` 使用 `{package['tool_name']}`：{package['rationale']}")
            lines.append(f"   请求提示：`{format_value(package.get('request_hint'))}`")
    lines.extend(["", f"来源状态：`{source_path}`", ""])
    return "\n".join(lines)


def generate_experiment_conclusion(source: Path, output_path: Path | None = None) -> ConclusionResult:
    try:
        state_path = resolve_state_path(source).resolve()
        state = read_json(state_path)
        conclusion_path = (output_path or state_path.with_name("conclusion.md")).resolve()
        write_text(conclusion_path, render_conclusion(state, state_path))
        return ConclusionResult(
            status=ConclusionStatus.COMPLETED,
            source_state_path=str(state_path),
            conclusion_path=str(conclusion_path),
        )
    except Exception as exc:
        return ConclusionResult(
            status=ConclusionStatus.FAILED,
            source_state_path=str(source),
            failure_reason=str(exc),
        )
