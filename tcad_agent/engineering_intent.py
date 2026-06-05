from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DeviceSupport(str, Enum):
    EXECUTABLE = "executable"
    COMPACT_BASELINE = "compact_baseline"
    PLANNED = "planned"
    UNKNOWN = "unknown"


class EngineeringIntent(BaseModel):
    source_text: str
    language: str = "mixed"
    device_family: str = "unknown"
    template_id: str | None = None
    support: DeviceSupport = DeviceSupport.UNKNOWN
    suggested_tool: str | None = None
    analyses: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)
    repair_preferences: list[str] = Field(default_factory=list)
    model_hints: list[str] = Field(default_factory=list)
    sweep_hints: dict[str, Any] = Field(default_factory=dict)
    request_hint: dict[str, Any] = Field(default_factory=dict)
    ambiguity: list[str] = Field(default_factory=list)
    risk_level: str = "medium"
    summary_zh: str = ""


DEVICE_CATALOG: list[dict[str, Any]] = [
    {
        "device_family": "mosfet_2d",
        "template_id": "mosfet_2d_id",
        "support": DeviceSupport.EXECUTABLE,
        "tool": "mosfet_2d_id_sweep",
        "aliases": ["2d nmos", "2d mosfet", "nmos", "pmos", "mosfet", "id-vg", "idvg", "id-vd", "idvd", "转移特性", "输出特性"],
        "request": {"sweep_type": "both", "gate_start": 0.0, "gate_stop": 1.2, "gate_step": 0.2},
    },
    {
        "device_family": "mos_capacitor",
        "template_id": "mos_capacitor_cv",
        "support": DeviceSupport.EXECUTABLE,
        "tool": "mos_capacitor_cv_sweep",
        "aliases": ["moscap", "mos capacitor", "mos c-v", "mos cv", "mos 电容", "c-v", "cv"],
        "request": {"start": -1.0, "stop": 1.0, "step": 0.25, "oxide_thickness_nm": 5.0},
    },
    {
        "device_family": "diode_breakdown",
        "template_id": "diode_breakdown_leakage",
        "support": DeviceSupport.EXECUTABLE,
        "tool": "diode_breakdown_leakage_sweep",
        "aliases": ["breakdown", "击穿", "leakage", "漏电", "reverse diode", "反偏", "pn breakdown"],
        "request": {"start": 0.0, "stop": -5.0, "step": 0.5},
    },
    {
        "device_family": "pn_junction",
        "template_id": "pn_junction_iv",
        "support": DeviceSupport.EXECUTABLE,
        "tool": "pn_junction_iv_sweep",
        "aliases": ["pn junction", "pn结", "p-n", "pn iv", "二极管 iv"],
        "request": {"start": 0.0, "stop": 0.5, "step": 0.1},
    },
    {
        "device_family": "schottky_diode",
        "template_id": "schottky_diode",
        "support": DeviceSupport.COMPACT_BASELINE,
        "tool": "extended_device_sweep",
        "aliases": ["schottky", "肖特基"],
        "request": {"device_type": "schottky_diode", "start": -0.5, "stop": 0.8, "step": 0.1},
    },
    {
        "device_family": "power_mosfet",
        "template_id": "power_mosfet_bv_ron",
        "support": DeviceSupport.COMPACT_BASELINE,
        "tool": "extended_device_sweep",
        "aliases": ["power mos", "power mosfet", "vdmos", "ldmos", "功率mos", "功率mosfet"],
        "request": {"device_type": "power_mosfet_bv_ron", "start": 0.0, "stop": -90.0, "step": 5.0},
    },
    {
        "device_family": "bjt",
        "template_id": "bjt_gummel_output",
        "support": DeviceSupport.COMPACT_BASELINE,
        "tool": "extended_device_sweep",
        "aliases": ["bjt", "gummel", "bipolar", "双极", "晶体管"],
        "request": {"device_type": "bjt_gummel_output", "start": 0.55, "stop": 0.8, "step": 0.025},
    },
    {
        "device_family": "jfet",
        "template_id": "jfet_transfer_output",
        "support": DeviceSupport.COMPACT_BASELINE,
        "tool": "extended_device_sweep",
        "aliases": ["jfet", "junction fet", "结型场效应"],
        "request": {"device_type": "jfet_transfer_output", "start": -3.0, "stop": 0.0, "step": 0.25},
    },
    {
        "device_family": "photodiode",
        "template_id": "photodiode_iv",
        "support": DeviceSupport.COMPACT_BASELINE,
        "tool": "extended_device_sweep",
        "aliases": ["photodiode", "photo diode", "光电二极管", "光电"],
        "request": {"device_type": "photodiode_iv", "start": -1.0, "stop": 0.8, "step": 0.1},
    },
    {
        "device_family": "finfet",
        "template_id": "finfet_id_cv",
        "support": DeviceSupport.PLANNED,
        "tool": None,
        "aliases": ["finfet", "gaa", "纳米片", "nanosheet", "nanowire"],
        "request": {},
    },
    {
        "device_family": "sic_power_diode",
        "template_id": "sic_power_diode_bv_leakage",
        "support": DeviceSupport.PLANNED,
        "tool": None,
        "aliases": ["sic diode", "sic jbs", "sic sbd", "碳化硅", "jbs"],
        "request": {},
    },
    {
        "device_family": "gan_hemt",
        "template_id": "gan_hemt_id_bv",
        "support": DeviceSupport.PLANNED,
        "tool": None,
        "aliases": ["gan hemt", "hemt", "algan", "氮化镓"],
        "request": {},
    },
    {
        "device_family": "igbt",
        "template_id": "igbt_output_turnoff",
        "support": DeviceSupport.PLANNED,
        "tool": None,
        "aliases": ["igbt", "绝缘栅双极"],
        "request": {},
    },
]


ANALYSIS_KEYWORDS: dict[str, list[str]] = {
    "idvg": ["id-vg", "idvg", "transfer", "转移", "vth", "ss", "ion/ioff", "阈值"],
    "idvd": ["id-vd", "idvd", "output", "输出", "饱和", "linear", "线性区", "saturation", "kink"],
    "cv": ["c-v", "cv", "capacitance", "电容", "平带"],
    "iv": ["i-v", "iv", "正向", "反向", "forward", "reverse"],
    "breakdown": ["breakdown", "击穿", "bv", "耐压"],
    "leakage": ["leakage", "漏电", "dark current", "暗电流"],
    "calibration": ["calibrate", "calibration", "fit", "拟合", "校准", "标定", "实测", "可信曲线"],
    "optimization": ["optimize", "optimization", "优化", "最优", "pareto", "约束"],
    "convergence": ["convergence", "mesh", "网格", "收敛", "model split", "模型对比", "a/b"],
    "report": ["conclusion", "report", "总结", "结论", "解释", "建议"],
}

METRIC_KEYWORDS: dict[str, list[str]] = {
    "vth": ["vth", "threshold", "阈值"],
    "ss": ["ss", "subthreshold", "亚阈", "摆幅"],
    "ion_ioff": ["ion/ioff", "ion", "ioff", "开关比"],
    "dibl": ["dibl", "短沟道"],
    "gm": ["gm", "transconductance", "跨导"],
    "bv": ["bv", "breakdown voltage", "击穿电压", "耐压"],
    "leakage": ["leakage", "漏电"],
    "ideality_factor": ["ideality", "理想因子"],
    "cox": ["cox", "oxide capacitance", "氧化层电容"],
    "flatband_shift": ["flatband", "平带", "qf", "固定电荷"],
    "ron": ["ron", "r_on", "导通电阻"],
    "responsivity": ["responsivity", "响应度"],
}

MODEL_KEYWORDS: dict[str, list[str]] = {
    "mobility_model": ["mobility", "迁移率", "constant mobility", "doping-dependent"],
    "interface_trap": ["interface trap", "dit", "界面态", "界面陷阱"],
    "fixed_oxide_charge": ["fixed charge", "fixed oxide charge", "qf", "固定电荷"],
    "impact_ionization": ["impact ionization", "avalanche", "雪崩", "碰撞电离"],
    "srh_lifetime": ["srh", "lifetime", "寿命", "复合"],
    "temperature_corner": ["temperature", "temp", "高温", "低温", "温度", "corner"],
}

EVIDENCE_KEYWORDS: dict[str, list[str]] = {
    "mesh_convergence": ["mesh", "网格", "convergence", "收敛"],
    "model_ab": ["model", "模型", "a/b", "ab", "对比"],
    "unit_check": ["unit", "单位", "量纲"],
    "curve_shape": ["curve shape", "曲线形状", "monotonic", "非单调", "kink", "异常"],
    "golden_or_measured": ["golden", "measured", "实测", "可信曲线", "标定"],
    "engineering_signoff": ["signoff", "签核", "项目会", "交付", "可信"],
}

OBJECTIVE_KEYWORDS: dict[str, list[str]] = {
    "minimize_leakage": ["漏电最小", "降低漏电", "leakage low", "min leakage"],
    "maximize_ion": ["ion 达标", "提高 ion", "drive current", "驱动电流"],
    "maximize_ion_ioff": ["ion/ioff", "开关比"],
    "meet_bv": ["bv", "耐压", "击穿达标"],
    "fit_measured_curve": ["拟合", "校准", "实测", "可信曲线"],
}


def text_has_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        output.append(item)
        seen.add(item)
    return output


def detect_language(text: str) -> str:
    has_cjk = any("\u4e00" <= char <= "\u9fff" for char in text)
    has_ascii_word = bool(re.search(r"[A-Za-z]{2,}", text))
    if has_cjk and has_ascii_word:
        return "mixed"
    if has_cjk:
        return "zh"
    return "en"


def match_device(text: str) -> dict[str, Any] | None:
    lowered = text.lower().replace("_", "-")
    best: tuple[int, dict[str, Any]] | None = None
    for entry in DEVICE_CATALOG:
        for alias in entry["aliases"]:
            normalized = alias.lower().replace("_", "-")
            if normalized in lowered:
                score = len(normalized)
                if best is None or score > best[0]:
                    best = (score, entry)
    return best[1] if best else None


def parse_voltage_point(text: str, label: str) -> float | None:
    patterns = [
        rf"{label}\s*(?:=|用|为|at)?\s*(-?\d+(?:\.\d+)?)\s*v",
        rf"{label.upper()}\s*(?:=|用|为|at)?\s*(-?\d+(?:\.\d+)?)\s*V",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def parse_range_after_label(text: str, labels: list[str]) -> dict[str, float] | None:
    label_pattern = "|".join(re.escape(label) for label in labels)
    patterns = [
        rf"(?:{label_pattern}).{{0,12}}?(-?\d+(?:\.\d+)?)\s*(?:v|V)?\s*(?:到|至|~|-|扫到|scan to)\s*(-?\d+(?:\.\d+)?)\s*(?:v|V)?",
        rf"(?:{label_pattern}).{{0,12}}?(?:from|从)\s*(-?\d+(?:\.\d+)?)\s*(?:v|V)?\s*(?:to|到)\s*(-?\d+(?:\.\d+)?)\s*(?:v|V)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return {"start": float(match.group(1)), "stop": float(match.group(2))}
    return None


def parse_sweep_hints(text: str) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    vg_range = parse_range_after_label(text, ["vg", "gate", "栅压", "栅极"])
    if vg_range:
        hints["gate_start"] = vg_range["start"]
        hints["gate_stop"] = vg_range["stop"]
    vd_range = parse_range_after_label(text, ["vd", "drain", "漏压", "漏极"])
    if vd_range:
        hints["drain_start"] = vd_range["start"]
        hints["drain_stop"] = vd_range["stop"]
    cv_range = parse_range_after_label(text, ["cv", "c-v", "gate", "栅压", "电压"])
    if cv_range and "gate_start" not in hints:
        hints["start"] = cv_range["start"]
        hints["stop"] = cv_range["stop"]
    for label, key in [("vd", "drain_voltage"), ("vg", "gate_voltage")]:
        point = parse_voltage_point(text, label)
        if point is not None:
            hints[key] = point
    all_voltages = [float(value) for value in re.findall(r"(-?\d+(?:\.\d+)?)\s*v", text, flags=re.IGNORECASE)]
    if all_voltages:
        hints["mentioned_voltages_v"] = all_voltages[:8]
    step_match = re.search(r"(?:step|步长)\s*(-?\d+(?:\.\d+)?)\s*v", text, flags=re.IGNORECASE)
    if step_match:
        hints["step"] = abs(float(step_match.group(1)))
    temp_values = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*k", text, flags=re.IGNORECASE)]
    if temp_values:
        hints["temperature_values_k"] = temp_values[:6]
    return hints


def build_request_hint(entry: dict[str, Any] | None, analyses: list[str], sweep_hints: dict[str, Any]) -> dict[str, Any]:
    request = dict((entry or {}).get("request") or {})
    if not request:
        return {}
    if "idvg" in analyses and "idvd" in analyses and request.get("sweep_type") is not None:
        request["sweep_type"] = "both"
    elif "idvd" in analyses and request.get("sweep_type") is not None:
        request["sweep_type"] = "idvd"
    elif "idvg" in analyses and request.get("sweep_type") is not None:
        request["sweep_type"] = "idvg"
    for key, value in sweep_hints.items():
        if key in {
            "gate_start",
            "gate_stop",
            "drain_start",
            "drain_stop",
            "drain_voltage",
            "start",
            "stop",
            "step",
        }:
            request[key] = value
    if "temperature_values_k" in sweep_hints:
        request["temperature_k"] = sweep_hints["temperature_values_k"][0]
    return request


def parse_engineering_intent(text: str) -> EngineeringIntent:
    entry = match_device(text)
    analyses = unique([name for name, words in ANALYSIS_KEYWORDS.items() if text_has_any(text, words)])
    metrics = unique([name for name, words in METRIC_KEYWORDS.items() if text_has_any(text, words)])
    model_hints = unique([name for name, words in MODEL_KEYWORDS.items() if text_has_any(text, words)])
    evidence = unique([name for name, words in EVIDENCE_KEYWORDS.items() if text_has_any(text, words)])
    objectives = unique([name for name, words in OBJECTIVE_KEYWORDS.items() if text_has_any(text, words)])
    repair_preferences = []
    if text_has_any(text, ["失败", "收敛失败", "自己调", "自动修复", "长时间", "不要停", "retry", "repair", "rerun"]):
        repair_preferences.append("auto_repair_without_user_until_budget_exhausted")
    if text_has_any(text, ["缩小步长", "bias step", "continuation", "ramp", "调步长"]):
        repair_preferences.append("bias_continuation_first")
    if text_has_any(text, ["网格", "mesh"]):
        repair_preferences.append("mesh_convergence_or_refinement")

    sweep_hints = parse_sweep_hints(text)
    ambiguity: list[str] = []
    if entry is None:
        ambiguity.append("没有明确匹配到已知器件模板。")
    if not analyses:
        ambiguity.append("没有明确识别仿真类型，默认先交给 supervisor 判断。")
    if "golden_or_measured" in evidence and "calibration" not in analyses:
        analyses.append("calibration")
    if "engineering_signoff" in evidence and "mesh_convergence" not in evidence:
        evidence.append("mesh_convergence")
    support = entry["support"] if entry else DeviceSupport.UNKNOWN
    risk_score = 0
    risk_score += 2 if support == DeviceSupport.PLANNED else 1 if support == DeviceSupport.COMPACT_BASELINE else 0
    risk_score += 2 if "engineering_signoff" in evidence or "golden_or_measured" in evidence else 0
    risk_score += 1 if "curve_shape" in evidence or "unit_check" in evidence or "mesh_convergence" in evidence else 0
    risk_score += 1 if ambiguity else 0
    risk_level = "high" if support == DeviceSupport.PLANNED or risk_score >= 3 else "medium" if risk_score >= 1 else "low"

    request_hint = build_request_hint(entry, analyses, sweep_hints)
    device_family = str(entry["device_family"]) if entry else "unknown"
    template_id = str(entry["template_id"]) if entry else None
    suggested_tool = entry.get("tool") if entry else None
    if not objectives and ("leakage" in metrics or "leakage" in analyses):
        objectives.append("minimize_leakage")
    if not objectives and ("ion_ioff" in metrics or "idvg" in analyses):
        objectives.append("extract_and_improve_key_metrics")
    constraints = []
    if text_has_any(text, ["达标", "约束", "不能超过", "至少", "小于", "大于", "limit", "spec"]):
        constraints.append("natural_language_spec_constraint_present")

    summary_parts = [
        f"器件={device_family}",
        f"分析={','.join(analyses) or '待定'}",
        f"指标={','.join(metrics) or '待定'}",
        f"证据={','.join(evidence) or '基础质量检查'}",
        f"风险={risk_level}",
    ]
    return EngineeringIntent(
        source_text=text,
        language=detect_language(text),
        device_family=device_family,
        template_id=template_id,
        support=support,
        suggested_tool=suggested_tool,
        analyses=analyses,
        metrics=metrics,
        objectives=objectives,
        constraints=constraints,
        evidence_requirements=evidence,
        repair_preferences=repair_preferences,
        model_hints=model_hints,
        sweep_hints=sweep_hints,
        request_hint=request_hint,
        ambiguity=ambiguity,
        risk_level=risk_level,
        summary_zh="；".join(summary_parts),
    )


def list_industrial_capabilities() -> list[dict[str, Any]]:
    return [
        {
            "device_family": entry["device_family"],
            "template_id": entry["template_id"],
            "support": entry["support"].value if isinstance(entry["support"], DeviceSupport) else str(entry["support"]),
            "suggested_tool": entry.get("tool"),
            "aliases": entry["aliases"],
            "default_request": entry["request"],
        }
        for entry in DEVICE_CATALOG
    ]
