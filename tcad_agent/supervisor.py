from __future__ import annotations

import json
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from tcad_agent.conclusion import generate_experiment_conclusion
from tcad_agent.dashboard import generate_experiment_dashboard
from tcad_agent.device_templates import RouteStatus, TemplateSupport, route_device_goal
from tcad_agent.experiment_index import default_index_db_path, list_records, rebuild_index
from tcad_agent.llm import LLMClient, LLMConfig
from tcad_agent.repair_strategy import build_repair_plan
from tcad_agent.reporting import generate_experiment_report
from tcad_agent.schottky_calibration import SchottkyCalibrationRequest, run_schottky_calibration
from tcad_agent.task_spec import PROJECT_ROOT, parse_task_text
from tcad_agent.mesh_convergence import MeshConvergenceRequest, run_mesh_convergence
from tcad_agent.tcad_deck import attach_tcad_deck_spec
from tcad_agent.tools.diode_breakdown import DiodeBreakdownRequest, run_diode_breakdown_sweep
from tcad_agent.tools.extended_device_sweep import ExtendedDeviceRequest, run_extended_device_sweep
from tcad_agent.tools.mos_capacitor_cv import MOSCapacitorCVRequest, run_mos_capacitor_cv_sweep
from tcad_agent.tools.mosfet_2d_id import MOSFET2DIDRequest, run_mosfet_2d_id_sweep
from tcad_agent.tools.task_runner import run_task
from tcad_agent.task_planner import parse_json_object


NUMBER_RE = r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"


class SupervisorStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    WAITING_FOR_USER = "waiting_for_user"
    COMPLETED = "completed"
    FAILED = "failed"


class SupervisorActionKind(str, Enum):
    REBUILD_INDEX = "rebuild_index"
    QUERY_INDEX = "query_index"
    RUN_PN_IV = "run_pn_iv"
    RUN_MOS_CV = "run_mos_cv"
    RUN_DIODE_BREAKDOWN = "run_diode_breakdown"
    RUN_MOSFET_2D = "run_mosfet_2d"
    RUN_EXTENDED_DEVICE = "run_extended_device"
    RUN_SCHOTTKY_CALIBRATION = "run_schottky_calibration"
    RUN_MESH_CONVERGENCE = "run_mesh_convergence"
    GENERATE_REPORT = "generate_report"
    GENERATE_DASHBOARD = "generate_dashboard"
    GENERATE_CONCLUSION = "generate_conclusion"
    GENERATE_REPAIR_PLAN = "generate_repair_plan"
    PLAN_DEVICE_TEMPLATE = "plan_device_template"
    ASK_USER = "ask_user"
    NOOP = "noop"


class SupervisorActionStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SupervisorAction(BaseModel):
    index: int
    kind: SupervisorActionKind
    status: SupervisorActionStatus
    reason: str
    request: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class SupervisorState(BaseModel):
    tool_name: str = "tcad_supervisor"
    status: SupervisorStatus
    supervisor_id: str
    goal_text: str
    supervisor_dir: str
    created_at: str
    updated_at: str
    execute: bool
    max_cycles: int
    completed_cycles: int = 0
    last_index_summary: dict[str, Any] | None = None
    recent_records: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[SupervisorAction] = Field(default_factory=list)
    next_action: str | None = None
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    failure_reason: str | None = None


class ChatClient(Protocol):
    config: LLMConfig

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        ...


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_supervisor_id() -> str:
    return f"supervisor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def result_to_json_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):
        dumped = result.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    return {"value": result}


def with_action_deck_result(result: Any, action: SupervisorAction) -> dict[str, Any]:
    data = result_to_json_dict(result)
    deck = action.request.get("tcad_deck_spec")
    if isinstance(deck, dict):
        data.setdefault("tcad_deck_spec", deck)
    return data


def write_supervisor_state(state: SupervisorState, path: Path) -> None:
    state.updated_at = utc_timestamp()
    write_json(path, state.model_dump(mode="json"))


def load_supervisor_state(path: Path) -> SupervisorState:
    return SupervisorState.model_validate_json(path.read_text(encoding="utf-8"))


def state_path(supervisor_root: Path, supervisor_id: str) -> Path:
    return supervisor_root / supervisor_id / "supervisor_state.json"


def first_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def first_number_after(labels: str, text: str) -> float | None:
    match = re.search(rf"(?:{labels})\s*[:=：]?\s*({NUMBER_RE})", text, re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def scale_current_amp(value: float, unit: str | None) -> float:
    normalized = (unit or "a").lower().replace("μ", "u")
    scale = {
        "a": 1.0,
        "ma": 1e-3,
        "ua": 1e-6,
        "na": 1e-9,
        "pa": 1e-12,
    }.get(normalized, 1.0)
    return value * scale


def first_current_after(labels: str, text: str) -> float | None:
    match = re.search(
        rf"(?:{labels})[^\n。；;，,<>≤≥]{{0,40}}?(?:<=|≤|<|不超过|低于|小于|limit|上限|目标)?\s*({NUMBER_RE})\s*(a|ma|ua|μa|na|pa)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    return scale_current_amp(float(match.group(1)), match.group(2))


def reverse_voltage_near(labels: str, text: str) -> float | None:
    patterns = [
        rf"({NUMBER_RE})\s*(?:v|伏|伏特)?[^\n。；;，,]{{0,40}}(?:{labels})",
        rf"(?:{labels})[^\n。；;，,]{{0,40}}({NUMBER_RE})\s*(?:v|伏|伏特)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        value = float(match.group(1))
        if value == 0:
            return 0.0
        return -abs(value) if text_has_any(text, ["reverse", "反偏", "漏电", "击穿", "bv"]) else value
    return None


def numbers_near_label(labels: str, text: str, *, max_chars: int = 120) -> list[float]:
    match = re.search(rf"(?:{labels})\s*[:=：]?\s*([^\n。；;，,]{{0,{max_chars}}})", text, re.IGNORECASE)
    if not match:
        return []
    segment = match.group(1)
    values = []
    for number in re.findall(NUMBER_RE, segment, re.IGNORECASE):
        try:
            values.append(float(number))
        except ValueError:
            pass
    return values


def range_near_label(labels: str, text: str) -> tuple[float | None, float | None]:
    values = numbers_near_label(labels, text)
    if len(values) >= 2:
        return values[0], values[1]
    if len(values) == 1:
        return None, values[0]
    return None, None


def text_has_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def first_path(pattern: str, text: str) -> Path | None:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1).strip().strip("\"'，,。；;")
    return Path(raw) if raw else None


def infer_schottky_calibration_request(
    goal_text: str,
    calibration_id: str,
    run_root: Path,
) -> SchottkyCalibrationRequest:
    start = first_float(r"(?:start|from|起始|从)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:v|伏|伏特)?", goal_text)
    stop = first_float(r"(?:stop|to|end|终止|结束|到|扫到)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:v|伏|伏特)?", goal_text)
    step = first_float(r"(?:step|步长|间隔)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:v|伏|伏特)?", goal_text)
    temperature = first_float(r"(?:temperature|温度)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:k|开尔文)?", goal_text)
    area = first_float(r"(?:area|面积)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*(?:cm2|cm\^2)?", goal_text)
    rmse_threshold = first_float(
        r"(?:rmse[-_ ]?threshold|max[-_ ]?rmse|误差阈值|rmse阈值)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)",
        goal_text,
    )
    target_curve = first_path(
        r"(?:target[-_ ]?curve|trusted[-_ ]?curve|measured[-_ ]?curve|目标曲线|可信曲线|实测曲线)\s*[:=：]?\s*([^\s,，;；]+(?:\.csv|\.json))",
        goal_text,
    )
    return SchottkyCalibrationRequest(
        calibration_id=calibration_id,
        run_root=run_root,
        target_curve_path=target_curve,
        start=start if start is not None else -0.2,
        stop=stop if stop is not None else 0.4,
        step=step if step is not None else 0.1,
        temperature_k=temperature if temperature is not None else 300.0,
        area_cm2=area if area is not None else 1.0e-8,
        max_pass_rmse_log_current_dec=rmse_threshold if rmse_threshold is not None else 0.15,
        verify_with_devsim=text_has_any(goal_text, ["devsim", "verify", "验证", "复核", "可信"]),
    )


def infer_mos_request(goal_text: str, run_id: str, run_root: Path) -> MOSCapacitorCVRequest:
    start = first_float(r"(?:start|from|起始|从)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:v|伏|伏特)?", goal_text)
    stop = first_float(r"(?:stop|to|end|终止|结束|到|扫到)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:v|伏|伏特)?", goal_text)
    step = first_float(r"(?:step|步长|间隔)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:v|伏|伏特)?", goal_text)
    oxide = first_float(r"(?:oxide[-_ ]?thickness|tox|氧化层厚度|氧化层)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:nm|纳米)?", goal_text)
    doping = first_float(r"(?:substrate[-_ ]?doping|substrate|衬底掺杂|衬底)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", goal_text)
    temperature = first_number_after(r"temperature|temp|温度", goal_text)
    fixed_charge = first_number_after(
        r"fixed[-_ ]?oxide[-_ ]?charge|fixed[-_ ]?charge|qf|oxide[-_ ]?charge|固定电荷|氧化层固定电荷",
        goal_text,
    )
    max_attempts = first_float(r"(?:max[-_ ]?attempts|attempts|最大尝试次数|重试次数)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)", goal_text)
    if start is None or stop is None:
        range_start, range_stop = range_near_label(r"gate|vg|栅压|栅极|c-v|cv|扫描|sweep", goal_text)
        start = start if start is not None else range_start
        stop = stop if stop is not None else range_stop
    return MOSCapacitorCVRequest(
        start=start if start is not None else -1.0,
        stop=stop if stop is not None else 1.0,
        step=step if step is not None else 0.25,
        min_step=(step / 4.0) if step is not None else 0.0625,
        max_attempts=int(max_attempts) if max_attempts is not None else 3,
        oxide_thickness_nm=oxide if oxide is not None else 5.0,
        substrate_doping_cm3=doping if doping is not None else 1.0e17,
        temperature_k=temperature if temperature is not None else 300.0,
        fixed_oxide_charge_cm2=fixed_charge if fixed_charge is not None else 0.0,
        run_id=run_id,
        run_root=run_root,
    )


def infer_diode_breakdown_request(goal_text: str, run_id: str, run_root: Path) -> DiodeBreakdownRequest:
    start = first_float(r"(?:start|from|起始|从)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:v|伏|伏特)?", goal_text)
    stop = first_float(r"(?:stop|to|end|终止|结束|到|扫到)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:v|伏|伏特)?", goal_text)
    step = first_float(r"(?:step|步长|间隔)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:v|伏|伏特)?", goal_text)
    breakdown_current = first_float(r"(?:breakdown[-_ ]?current|击穿电流|bv[-_ ]?current)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", goal_text)
    leakage_voltage = first_float(r"(?:leakage[-_ ]?voltage|漏电电压)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:v|伏|伏特)?", goal_text)
    if leakage_voltage is None:
        leakage_voltage = reverse_voltage_near(r"leakage|漏电|暗电流|反偏电流", goal_text)
    leakage_limit = first_current_after(
        r"quality[-_ ]?max[-_ ]?leakage|leakage[-_ ]?limit|漏电(?:上限|目标|规格|spec)?|暗电流",
        goal_text,
    )
    temperature = first_number_after(r"temperature|temp|温度|高温", goal_text)
    lifetime = first_number_after(r"electron[-_ ]?lifetime|hole[-_ ]?lifetime|lifetime|srh|寿命|复合寿命", goal_text)
    p_doping = first_number_after(r"p[-_ ]?doping|p区掺杂|p掺杂|acceptor(?:s)?", goal_text)
    n_doping = first_number_after(r"n[-_ ]?doping|n区掺杂|n掺杂|donor(?:s)?", goal_text)
    max_attempts = first_float(r"(?:max[-_ ]?attempts|attempts|最大尝试次数|重试次数)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)", goal_text)
    if start is None or stop is None:
        range_start, range_stop = range_near_label(r"reverse|反偏|bias|电压|sweep|扫描", goal_text)
        start = start if start is not None else range_start
        stop = stop if stop is not None else range_stop
    if stop is None and leakage_voltage is not None:
        stop = min(leakage_voltage, -5.0)
    strict_leakage_investigation = text_has_any(goal_text, ["数量级", "order of magnitude", "偏高", "高很多", "客户数据", "实测"])
    return DiodeBreakdownRequest(
        start=start if start is not None else 0.0,
        stop=stop if stop is not None else -5.0,
        step=step if step is not None else 0.5,
        min_step=(step / 4.0) if step is not None else 0.125,
        max_attempts=int(max_attempts) if max_attempts is not None else 5,
        breakdown_current_a=breakdown_current if breakdown_current is not None else 1e-6,
        leakage_voltage_v=leakage_voltage if leakage_voltage is not None else -1.0,
        require_breakdown=text_has_any(goal_text, ["breakdown", "击穿", "bv", "耐压"]),
        quality_max_leakage_abs_current_a=(
            leakage_limit if leakage_limit is not None else 1.0e-6 if strict_leakage_investigation else 1.0e-3
        ),
        p_doping_cm3=p_doping if p_doping is not None else 1.0e18,
        n_doping_cm3=n_doping if n_doping is not None else 1.0e18,
        temperature_k=temperature if temperature is not None and temperature > 100 else 350.0 if text_has_any(goal_text, ["高温", "temperature corner", "温度 corner"]) else 300.0,
        electron_lifetime_s=lifetime if lifetime is not None else 1.0e-8,
        hole_lifetime_s=lifetime if lifetime is not None else 1.0e-8,
        run_id=run_id,
        run_root=run_root,
    )


def infer_mosfet_2d_request(goal_text: str, run_id: str, run_root: Path) -> MOSFET2DIDRequest:
    gate_start = first_float(r"(?:gate[-_ ]?start|vg[-_ ]?start|栅压起始|栅极起始)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:v|伏|伏特)?", goal_text)
    gate_stop = first_float(r"(?:gate[-_ ]?stop|vg[-_ ]?stop|栅压结束|栅极结束|扫到)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:v|伏|伏特)?", goal_text)
    gate_step = first_float(r"(?:gate[-_ ]?step|vg[-_ ]?step|栅压步长|栅极步长|步长)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:v|伏|伏特)?", goal_text)
    drain_voltage = first_float(r"(?:drain[-_ ]?voltage|vds|漏压|漏极电压)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:v|伏|伏特)?", goal_text)
    drain_start = first_float(r"(?:drain[-_ ]?start|vd[-_ ]?start|漏压起始|漏极起始)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:v|伏|伏特)?", goal_text)
    drain_stop = first_float(r"(?:drain[-_ ]?stop|vd[-_ ]?stop|漏压结束|漏极结束)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:v|伏|伏特)?", goal_text)
    drain_step = first_float(r"(?:drain[-_ ]?step|vd[-_ ]?step|漏压步长|漏极步长)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:v|伏|伏特)?", goal_text)
    idvd_gate_voltage = first_float(r"(?:idvd[-_ ]?gate[-_ ]?voltage|output[-_ ]?gate[-_ ]?voltage|输出栅压)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:v|伏|伏特)?", goal_text)
    gate_values = numbers_near_label(r"vg|gate|栅压|栅极|固定\s*vg", goal_text)
    drain_values = numbers_near_label(r"vd|vds|drain|漏压|漏极", goal_text)
    if gate_start is None or gate_stop is None:
        range_start, range_stop = range_near_label(r"vg|gate|栅压|栅极", goal_text)
        gate_start = gate_start if gate_start is not None else range_start
        gate_stop = gate_stop if gate_stop is not None else range_stop
    if drain_start is None or drain_stop is None:
        range_start, range_stop = range_near_label(r"vd|vds|drain|漏压|漏极", goal_text)
        drain_start = drain_start if drain_start is not None else range_start
        drain_stop = drain_stop if drain_stop is not None else range_stop
    if idvd_gate_voltage is None and gate_values:
        idvd_gate_voltage = max(gate_values)
    oxide = first_number_after(r"oxide[-_ ]?thickness|tox|氧化层厚度|氧化层", goal_text)
    substrate_doping = first_number_after(r"substrate[-_ ]?doping|channel[-_ ]?doping|沟道掺杂|衬底掺杂|衬底", goal_text)
    sd_doping = first_number_after(r"source[-_ ]?drain[-_ ]?doping|sd[-_ ]?doping|源漏掺杂", goal_text)
    temperature = first_number_after(r"temperature|temp|温度", goal_text)
    electron_mobility = first_number_after(r"electron[-_ ]?mobility|electron[-_ ]?mu|电子迁移率", goal_text)
    hole_mobility = first_number_after(r"hole[-_ ]?mobility|hole[-_ ]?mu|空穴迁移率", goal_text)
    electron_lifetime = first_number_after(r"electron[-_ ]?lifetime|taun|电子寿命", goal_text)
    hole_lifetime = first_number_after(r"hole[-_ ]?lifetime|taup|空穴寿命", goal_text)
    common_lifetime = first_number_after(r"srh[-_ ]?lifetime|carrier[-_ ]?lifetime|lifetime|复合寿命|寿命", goal_text)
    fixed_charge = first_number_after(
        r"fixed[-_ ]?oxide[-_ ]?charge|fixed[-_ ]?charge|qf|oxide[-_ ]?charge|固定电荷|氧化层固定电荷",
        goal_text,
    )
    interface_trap = first_number_after(
        r"interface[-_ ]?trap(?:[-_ ]?density)?|dit|界面态|界面陷阱",
        goal_text,
    )
    asks_output = text_has_any(goal_text, ["id-vd", "idvd", "output characteristic", "输出特性", "输出曲线", "kink"])
    asks_transfer = text_has_any(goal_text, ["id-vg", "idvg", "transfer characteristic", "转移特性", "转移曲线", "vth", "ss", "ion/ioff", "阈值"])
    if asks_output and asks_transfer:
        sweep_type = "both"
    elif asks_output:
        sweep_type = "idvd"
    else:
        sweep_type = "idvg"
    default_drain_stop = 1.2 if asks_output else (drain_voltage if drain_voltage is not None else 0.05)
    default_drain_step = 0.05 if asks_output else (drain_voltage if drain_voltage is not None and drain_voltage > 0 else 0.05)
    explicit_high_field_model = text_has_any(
        goal_text,
        ["impact", "ionization", "avalanche", "雪崩", "碰撞电离"],
    )
    mobility_model = "doping_dependent" if text_has_any(
        goal_text,
        ["doping-dependent mobility", "doping dependent mobility", "掺杂相关迁移率", "迁移率模型", "mobility model"],
    ) and not text_has_any(goal_text, ["constant mobility", "常数迁移率"]) else "constant"
    recombination_model = "none" if text_has_any(goal_text, ["no srh", "关闭 srh", "不启用复合", "无复合"]) else "srh"
    model_strategy = "dd_direct" if text_has_any(goal_text, ["直接 drift diffusion", "直接 dd", "dd_direct"]) else "poisson_then_dd"
    if text_has_any(goal_text, ["dibl", "低 vd", "高 vd", "short channel", "短沟道"]):
        asks_transfer = True
        sweep_type = "idvg" if not asks_output else sweep_type
    return MOSFET2DIDRequest(
        sweep_type=sweep_type,
        gate_start=gate_start if gate_start is not None else 0.0,
        gate_stop=gate_stop if gate_stop is not None else (max(gate_values) if gate_values else 0.5),
        gate_step=gate_step if gate_step is not None else 0.5,
        min_gate_step=(gate_step / 4.0) if gate_step is not None else 0.125,
        drain_voltage=drain_voltage if drain_voltage is not None else 0.05,
        drain_start=drain_start if drain_start is not None else 0.0,
        drain_stop=drain_stop if drain_stop is not None else (max(drain_values) if len(drain_values) >= 2 else default_drain_stop),
        drain_step=drain_step if drain_step is not None else default_drain_step,
        min_drain_step=(drain_step / 4.0) if drain_step is not None else 0.0125,
        idvd_gate_voltage=idvd_gate_voltage if idvd_gate_voltage is not None else (gate_stop if gate_stop is not None else 0.5),
        oxide_thickness_nm=oxide if oxide is not None else 5.0,
        substrate_doping_cm3=substrate_doping if substrate_doping is not None else 1.0e17,
        source_drain_doping_cm3=sd_doping if sd_doping is not None else 1.0e20,
        temperature_k=temperature if temperature is not None else 350.0 if text_has_any(goal_text, ["高温", "hot corner"]) else 300.0,
        mobility_model=mobility_model,
        electron_mobility_cm2_v_s=electron_mobility,
        hole_mobility_cm2_v_s=hole_mobility,
        recombination_model=recombination_model,
        electron_lifetime_s=electron_lifetime if electron_lifetime is not None else common_lifetime if common_lifetime is not None else 1.0e-5,
        hole_lifetime_s=hole_lifetime if hole_lifetime is not None else common_lifetime if common_lifetime is not None else 1.0e-5,
        fixed_oxide_charge_cm2=fixed_charge if fixed_charge is not None else 0.0,
        interface_trap_density_cm2=interface_trap if interface_trap is not None else 0.0,
        impact_ionization_model="selberherr" if explicit_high_field_model else "none",
        model_strategy=model_strategy,
        x_divisions=12,
        silicon_y_divisions=4,
        run_id=run_id,
        run_root=run_root,
    )


def infer_mesh_convergence_request(goal_text: str, convergence_id: str, convergence_root: Path) -> MeshConvergenceRequest:
    tolerance = first_float(r"(?:relative[-_ ]?tolerance|tolerance|收敛阈值|相对误差)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", goal_text)
    return MeshConvergenceRequest(
        convergence_id=convergence_id,
        relative_tolerance=tolerance if tolerance is not None else 0.05,
        execute=True,
        overwrite=True,
        convergence_root=convergence_root,
        max_cases=3,
    )


def latest_state_path(records: list[dict[str, Any]], kinds: set[str] | None = None) -> str | None:
    for record in records:
        if kinds is None or record.get("kind") in kinds:
            path = record.get("state_path")
            if path:
                return str(path)
    return None


def asks_existing_result(text: str) -> bool:
    return text_has_any(
        text,
        [
            "recent",
            "latest",
            "previous",
            "prior",
            "existing",
            "history",
            "indexed",
            "最近",
            "最新",
            "上一次",
            "已有",
            "历史",
            "索引",
            "旧",
        ],
    )


def existing_result_action(state: SupervisorState, common: dict[str, Any]) -> SupervisorAction | None:
    text = state.goal_text
    if text_has_any(text, ["repair", "修复", "失败恢复", "恢复策略", "收敛失败", "失败原因"]):
        target = latest_state_path(
            state.recent_records,
            {
                "adaptive_optimization",
                "multidim_optimization",
                "parameter_sweep",
                "mosfet_2d_id_sweep",
                "diode_breakdown_leakage_sweep",
                "mesh_convergence",
                "mos_capacitor_cv_sweep",
                "pn_junction_iv_sweep",
                "schottky_iv_calibration",
                "task_run",
            },
        )
        if target:
            return SupervisorAction(
                **common,
                kind=SupervisorActionKind.GENERATE_REPAIR_PLAN,
                reason="user goal asks for TCAD-specific failure repair strategy over an existing result",
                request={"state": target},
            )

    if text_has_any(text, ["conclusion", "结论", "工程判断", "趋势解释", "下一轮建议", "异常点"]):
        target = latest_state_path(
            state.recent_records,
            {
                "adaptive_optimization",
                "multidim_optimization",
                "parameter_sweep",
                "mosfet_2d_id_sweep",
                "diode_breakdown_leakage_sweep",
                "mesh_convergence",
                "mos_capacitor_cv_sweep",
                "pn_junction_iv_sweep",
                "schottky_iv_calibration",
            },
        )
        if target:
            return SupervisorAction(
                **common,
                kind=SupervisorActionKind.GENERATE_CONCLUSION,
                reason="user goal asks for an engineering conclusion over an existing TCAD result",
                request={"state": target},
            )

    if text_has_any(text, ["report", "报告", "总结"]):
        target = latest_state_path(state.recent_records, {"adaptive_optimization", "multidim_optimization", "parameter_sweep"})
        if target:
            return SupervisorAction(
                **common,
                kind=SupervisorActionKind.GENERATE_REPORT,
                reason="user goal asks for a report over existing experiment results",
                request={"state": target},
            )

    if text_has_any(text, ["dashboard", "仪表盘", "可视化"]):
        target = latest_state_path(state.recent_records, {"adaptive_optimization", "multidim_optimization", "parameter_sweep"})
        if target:
            return SupervisorAction(
                **common,
                kind=SupervisorActionKind.GENERATE_DASHBOARD,
                reason="user goal asks for a visual dashboard over existing experiment results",
                request={"state": target},
            )
    return None


def choose_next_action(state: SupervisorState) -> SupervisorAction:
    text = state.goal_text
    index = len(state.actions) + 1
    now = utc_timestamp()
    common = {"index": index, "status": SupervisorActionStatus.PLANNED, "created_at": now, "updated_at": now}

    if not state.last_index_summary:
        return SupervisorAction(
            **common,
            kind=SupervisorActionKind.REBUILD_INDEX,
            reason="refresh experiment memory before deciding the next TCAD action",
            request={"root": str(PROJECT_ROOT / "runs"), "db_path": str(default_index_db_path())},
        )

    if asks_existing_result(text):
        action = existing_result_action(state, common)
        if action:
            return action

    if text_has_any(text, ["history", "历史", "检索", "index", "索引"]):
        return SupervisorAction(
            **common,
            kind=SupervisorActionKind.QUERY_INDEX,
            reason="user goal asks to inspect prior TCAD experiments",
            request={"limit": 20},
        )

    route = route_device_goal(text)
    if route.status == RouteStatus.MATCHED and route.template and route.template.support == TemplateSupport.PLANNED:
        return SupervisorAction(
            **common,
            kind=SupervisorActionKind.PLAN_DEVICE_TEMPLATE,
            reason="goal matched a known device template that is not executable yet",
            request=route.model_dump(mode="json"),
        )
    if text_has_any(text, ["schottky", "肖特基"]) and text_has_any(
        text,
        [
            "calibrate",
            "calibration",
            "fit",
            "fitting",
            "trusted curve",
            "measured curve",
            "golden curve",
            "校准",
            "标定",
            "拟合",
            "可信曲线",
            "目标曲线",
            "实测曲线",
        ],
    ):
        calibration_id = f"{state.supervisor_id}_schottky_cal_{index:03d}"
        request = infer_schottky_calibration_request(
            text,
            calibration_id,
            Path(state.supervisor_dir) / "agent_tools",
        ).model_dump(mode="json")
        request.setdefault("device_type", "schottky_diode")
        return SupervisorAction(
            **common,
            kind=SupervisorActionKind.RUN_SCHOTTKY_CALIBRATION,
            reason="goal asks to calibrate Schottky IV parameters against a trusted or measured curve",
            request=attach_tcad_deck_spec(text, "schottky_iv_calibration", request),
        )
    if route.status == RouteStatus.MATCHED and route.suggested_tool == "extended_device_sweep":
        request = dict(route.request_hint)
        run_id = f"{state.supervisor_id}_{request.get('device_type', 'extended')}_{index:03d}"
        request["run_id"] = run_id
        request["run_root"] = str(Path(state.supervisor_dir) / "agent_tools")
        if route.template and route.template.support == TemplateSupport.COMPACT_BASELINE:
            request["evidence_level"] = "compact_baseline"
            request["capability_warnings"] = route.capability_warnings
            request["requires_higher_fidelity_runner_for_signoff"] = True
        elif route.template and route.template.support == TemplateSupport.EXECUTABLE:
            request["evidence_level"] = "tcad_executable"
        return SupervisorAction(
            **common,
            kind=SupervisorActionKind.RUN_EXTENDED_DEVICE,
            reason=(
                "goal matched a compact baseline extended-device template; run only as planning evidence"
                if route.template and route.template.support == TemplateSupport.COMPACT_BASELINE
                else "goal matched an executable extended-device TCAD template"
            ),
            request=attach_tcad_deck_spec(text, "extended_device_sweep", request),
        )

    if text_has_any(
        text,
        [
            "mosfet",
            "nmos",
            "pmos",
            "id-vg",
            "idvg",
            "id-vd",
            "idvd",
            "output characteristic",
            "transfer characteristic",
            "输出特性",
            "输出曲线",
            "转移特性",
            "转移曲线",
            "vth",
            "subthreshold",
            "ion/ioff",
            "dibl",
            "kink",
            "阈值",
            "亚阈值",
            "源漏",
        ],
    ):
        run_id = f"{state.supervisor_id}_mosfet2d_{index:03d}"
        request = infer_mosfet_2d_request(text, run_id, Path(state.supervisor_dir) / "agent_tools").model_dump(mode="json")
        return SupervisorAction(
            **common,
            kind=SupervisorActionKind.RUN_MOSFET_2D,
            reason="goal appears to request a 2D MOSFET Id-Vg/Id-Vd task",
            request=attach_tcad_deck_spec(text, "mosfet_2d_id_sweep", request),
        )

    if text_has_any(text, ["mesh convergence", "网格收敛", "mesh check", "网格检查", "mesh", "网格"]):
        convergence_id = f"{state.supervisor_id}_mesh_{index:03d}"
        return SupervisorAction(
            **common,
            kind=SupervisorActionKind.RUN_MESH_CONVERGENCE,
            reason="goal appears to request a mesh convergence check",
            request={
                "base_task_text": text,
                "convergence_request": infer_mesh_convergence_request(
                    text,
                    convergence_id,
                    Path(state.supervisor_dir) / "mesh_convergence",
                ).model_dump(mode="json"),
            },
        )

    if text_has_any(text, ["breakdown", "leakage", "bv", "击穿", "漏电", "反偏", "reverse", "高温漏电"]):
        run_id = f"{state.supervisor_id}_diode_bd_{index:03d}"
        request = infer_diode_breakdown_request(text, run_id, Path(state.supervisor_dir) / "agent_tools").model_dump(mode="json")
        return SupervisorAction(
            **common,
            kind=SupervisorActionKind.RUN_DIODE_BREAKDOWN,
            reason="goal appears to request diode reverse leakage or breakdown",
            request=attach_tcad_deck_spec(text, "diode_breakdown_leakage_sweep", request),
        )

    if text_has_any(
        text,
        [
            "mos",
            "moscap",
            "capacitor",
            "c-v",
            "cv",
            "flat-band",
            "flatband",
            "fixed oxide charge",
            "fixed charge",
            "oxide charge",
            "电容",
            "平带",
            "固定电荷",
            "氧化层电荷",
        ],
    ):
        run_id = f"{state.supervisor_id}_mos_cv_{index:03d}"
        request = infer_mos_request(text, run_id, Path(state.supervisor_dir) / "agent_tools").model_dump(mode="json")
        return SupervisorAction(
            **common,
            kind=SupervisorActionKind.RUN_MOS_CV,
            reason="goal appears to request a MOS capacitor C-V task",
            request=attach_tcad_deck_spec(text, "mos_capacitor_cv_sweep", request),
        )

    if text_has_any(text, ["pn", "二极管", "junction", "iv", "i-v", "伏安", "电流"]):
        task_id = f"{state.supervisor_id}_pn_iv_{index:03d}"
        spec = parse_task_text(text, task_id=task_id, use_llm=False)
        deck_request = {
            "start": spec.sweep.start_v,
            "stop": spec.sweep.stop_v,
            "step": spec.sweep.step_v,
            "length_um": spec.parameters.length_um,
            "junction_um": spec.parameters.junction_um,
            "p_doping_cm3": spec.parameters.p_doping_cm3,
            "n_doping_cm3": spec.parameters.n_doping_cm3,
            "temperature_k": spec.parameters.temperature_k,
            "electron_lifetime_s": spec.parameters.electron_lifetime_s,
            "hole_lifetime_s": spec.parameters.hole_lifetime_s,
            "contact_spacing_um": spec.mesh.contact_spacing_um,
            "junction_spacing_um": spec.mesh.junction_spacing_um,
        }
        request = attach_tcad_deck_spec(text, "pn_junction_iv_sweep", {"task_id": task_id, "text": text, **deck_request})
        request["task_id"] = task_id
        request["text"] = text
        return SupervisorAction(
            **common,
            kind=SupervisorActionKind.RUN_PN_IV,
            reason="goal appears to request a PN junction IV task",
            request=request,
        )

    return SupervisorAction(
        **common,
        kind=SupervisorActionKind.ASK_USER,
        reason="goal is ambiguous for current deterministic routing",
        request={
            "question": "请明确要运行 PN IV、MOS C-V、2D MOSFET、Schottky、击穿/漏电、参数扫参/优化、报告还是 dashboard。",
            "questions": [
                "器件/结构是什么？",
                "要做哪类分析或扫参？",
                "关键指标和规格是什么？",
            ],
        },
    )


def build_supervisor_agent_messages(state: SupervisorState, deterministic_action: SupervisorAction) -> tuple[str, str]:
    system = (
        "你是 TCAD supervisor agent，负责在已有工具能力内选择下一步。只返回 JSON。"
        "你可以覆盖 deterministic_candidate，但只能使用 supported_action_kinds 里的 kind。"
        "不要返回 shell command。不要编造工具。"
        "优先基于目标、最近实验、已有 deck/spec、质量状态和 artifact 判断下一步。"
        "若没有足够上下文，选择最小信息增益动作，比如 rebuild/query/repair/benchmark，而不是泛泛询问用户。"
    )
    user = {
        "task": "choose next supervisor action",
        "supported_action_kinds": [kind.value for kind in SupervisorActionKind],
        "response_schema": {
            "action": {
                "kind": "one supported action kind",
                "reason": "中文，说明为什么比 deterministic candidate 更好，或为什么接受它",
                "request": "object",
            },
            "observation_summary": "中文证据摘要",
            "hypothesis_zh": "当前工程假设",
            "evidence_used": ["goal_text", "recent_records", "deterministic_candidate"],
        },
        "guardrails": [
            "如果 action.kind 是 run_*，request 必须是该工具可验证字段。",
            "不要直接给强签核结论；需要 conclusion/report/dashboard 时必须引用已有 state。",
            "对于高风险几何/工艺变更，优先走 repair/deck mutation lineage，不要直接改未知字段。",
        ],
        "context": {
            "goal_text": state.goal_text,
            "completed_cycles": state.completed_cycles,
            "recent_records": state.recent_records[:10],
            "last_index_summary": state.last_index_summary,
            "deterministic_candidate": deterministic_action.model_dump(mode="json"),
        },
    }
    return system, json.dumps(user, ensure_ascii=False, indent=2)


def supervisor_action_from_agent(
    state: SupervisorState,
    deterministic_action: SupervisorAction,
    *,
    client: ChatClient | None = None,
    allow_fallback: bool = True,
) -> tuple[SupervisorAction, dict[str, Any]]:
    chat_client = client or LLMClient()
    system, user = build_supervisor_agent_messages(state, deterministic_action)
    decision: dict[str, Any] = {
        "schema_version": "actsoft.tcad.supervisor_agent_decision.v1",
        "status": "fallback",
        "fallback_used": True,
        "deterministic_action": deterministic_action.model_dump(mode="json"),
    }
    try:
        raw = chat_client.chat(system=system, user=user, temperature=0.2)
    except Exception as exc:
        decision["failure_reason"] = str(exc)
        return deterministic_action, decision
    decision["raw_response"] = raw
    parsed = parse_json_object(raw)
    if parsed is None:
        decision["failure_reason"] = "agent did not return a JSON object"
        return deterministic_action, decision
    decision["parsed_response"] = parsed
    raw_action = parsed.get("action") if isinstance(parsed.get("action"), dict) else parsed
    kind_value = raw_action.get("kind") if isinstance(raw_action, dict) else None
    try:
        kind = SupervisorActionKind(str(kind_value))
    except Exception:
        decision["failure_reason"] = f"unsupported action kind: {kind_value}"
        return deterministic_action, decision
    request = raw_action.get("request") if isinstance(raw_action.get("request"), dict) else {}
    if raw_action.get("command") or raw_action.get("next_tool_command"):
        decision["failure_reason"] = "agent returned shell command; ignored"
        return deterministic_action, decision
    now = utc_timestamp()
    action = SupervisorAction(
        index=len(state.actions) + 1,
        kind=kind,
        status=SupervisorActionStatus.PLANNED,
        reason=str(raw_action.get("reason") or parsed.get("hypothesis_zh") or deterministic_action.reason),
        request=request,
        created_at=now,
        updated_at=now,
    )
    decision.update(
        {
            "status": "completed",
            "fallback_used": False,
            "model": getattr(chat_client.config, "model", None),
            "observation_summary": parsed.get("observation_summary"),
            "hypothesis_zh": parsed.get("hypothesis_zh"),
            "evidence_used": parsed.get("evidence_used") or [],
            "action": action.model_dump(mode="json"),
        }
    )
    return action, decision


def create_initial_state(
    supervisor_id: str,
    goal_text: str,
    supervisor_dir: Path,
    execute: bool,
    max_cycles: int,
) -> SupervisorState:
    now = utc_timestamp()
    return SupervisorState(
        status=SupervisorStatus.RUNNING if execute else SupervisorStatus.PLANNED,
        supervisor_id=supervisor_id,
        goal_text=goal_text,
        supervisor_dir=str(supervisor_dir),
        created_at=now,
        updated_at=now,
        execute=execute,
        max_cycles=max_cycles,
        next_action="rebuild experiment index",
        checkpoint={"completed_cycles": 0},
    )


def execute_action(action: SupervisorAction, state: SupervisorState) -> SupervisorAction:
    action.status = SupervisorActionStatus.RUNNING
    action.updated_at = utc_timestamp()
    try:
        if action.kind == SupervisorActionKind.REBUILD_INDEX:
            root = Path(action.request.get("root") or PROJECT_ROOT / "runs")
            db_path = Path(action.request.get("db_path") or default_index_db_path())
            result = rebuild_index(root, db_path)
            state.last_index_summary = result
            state.recent_records = list_records(db_path, limit=20)
            action.result = {"index": result, "recent_records": state.recent_records[:5]}
        elif action.kind == SupervisorActionKind.QUERY_INDEX:
            action.result = {
                "records": list_records(
                    default_index_db_path(),
                    kind=action.request.get("kind"),
                    status=action.request.get("status"),
                    limit=int(action.request.get("limit") or 20),
                )
            }
        elif action.kind == SupervisorActionKind.RUN_PN_IV:
            spec = parse_task_text(action.request["text"], task_id=action.request["task_id"], use_llm=False)
            task_state = run_task(
                spec,
                task_root=Path(state.supervisor_dir) / "tasks",
                run_root=Path(state.supervisor_dir) / "agent_tools",
                execute=state.execute,
                overwrite=True,
            )
            action.result = with_action_deck_result(task_state, action)
        elif action.kind == SupervisorActionKind.RUN_MOS_CV:
            request = MOSCapacitorCVRequest.model_validate(action.request)
            action.result = with_action_deck_result(run_mos_capacitor_cv_sweep(request), action)
        elif action.kind == SupervisorActionKind.RUN_DIODE_BREAKDOWN:
            request = DiodeBreakdownRequest.model_validate(action.request)
            result = run_diode_breakdown_sweep(request)
            action.result = with_action_deck_result(result, action)
        elif action.kind == SupervisorActionKind.RUN_MOSFET_2D:
            request = MOSFET2DIDRequest.model_validate(action.request)
            result = run_mosfet_2d_id_sweep(request)
            action.result = with_action_deck_result(result, action)
        elif action.kind == SupervisorActionKind.RUN_EXTENDED_DEVICE:
            request = ExtendedDeviceRequest.model_validate(action.request)
            result = run_extended_device_sweep(request)
            action.result = with_action_deck_result(result, action)
        elif action.kind == SupervisorActionKind.RUN_SCHOTTKY_CALIBRATION:
            request = SchottkyCalibrationRequest.model_validate(action.request)
            result = run_schottky_calibration(request)
            action.result = with_action_deck_result(result, action)
        elif action.kind == SupervisorActionKind.RUN_MESH_CONVERGENCE:
            spec = parse_task_text(
                action.request["base_task_text"],
                task_id=f"{state.supervisor_id}_mesh_base_{action.index:03d}",
                use_llm=False,
            )
            request = MeshConvergenceRequest.model_validate(action.request["convergence_request"])
            result = run_mesh_convergence(spec, request)
            action.result = result_to_json_dict(result)
        elif action.kind == SupervisorActionKind.GENERATE_REPORT:
            result = generate_experiment_report(Path(action.request["state"]))
            action.result = result_to_json_dict(result)
        elif action.kind == SupervisorActionKind.GENERATE_DASHBOARD:
            result = generate_experiment_dashboard(Path(action.request["state"]))
            action.result = result_to_json_dict(result)
        elif action.kind == SupervisorActionKind.GENERATE_CONCLUSION:
            result = generate_experiment_conclusion(Path(action.request["state"]))
            action.result = result_to_json_dict(result)
        elif action.kind == SupervisorActionKind.GENERATE_REPAIR_PLAN:
            result = build_repair_plan(Path(action.request["state"]))
            action.result = result_to_json_dict(result)
        elif action.kind == SupervisorActionKind.PLAN_DEVICE_TEMPLATE:
            template = action.request.get("template") or {}
            action.result = {
                "status": "planned",
                "template_id": template.get("template_id"),
                "display_name": template.get("display_name"),
                "support": template.get("support"),
                "tasks": template.get("tasks") or [],
                "missing_capabilities": template.get("missing_capabilities") or [],
                "next_implementation_steps": template.get("next_implementation_steps") or [],
                "message": action.request.get("message"),
            }
            state.status = SupervisorStatus.WAITING_FOR_USER
        elif action.kind == SupervisorActionKind.ASK_USER:
            action.result = {"question": action.request.get("question")}
            state.status = SupervisorStatus.WAITING_FOR_USER
        else:
            action.result = {"message": "no operation"}
        action.status = SupervisorActionStatus.COMPLETED
    except Exception as exc:
        action.status = SupervisorActionStatus.FAILED
        action.error = str(exc)
        state.status = SupervisorStatus.FAILED
        state.failure_reason = str(exc)
    action.updated_at = utc_timestamp()
    return action


def refresh_experiment_memory(state: SupervisorState) -> None:
    result = rebuild_index(PROJECT_ROOT / "runs", default_index_db_path())
    state.last_index_summary = result
    state.recent_records = list_records(default_index_db_path(), limit=20)


def run_supervisor(
    goal_text: str,
    *,
    supervisor_id: str | None = None,
    supervisor_root: Path | None = None,
    execute: bool = False,
    resume: bool = False,
    max_cycles: int = 3,
    use_agent_policy: bool = True,
    llm_client: ChatClient | None = None,
) -> SupervisorState:
    actual_supervisor_id = supervisor_id or default_supervisor_id()
    actual_root = supervisor_root or PROJECT_ROOT / "runs" / "supervisor"
    actual_state_path = state_path(actual_root, actual_supervisor_id)
    supervisor_dir = actual_root / actual_supervisor_id

    if resume and actual_state_path.exists():
        state = load_supervisor_state(actual_state_path)
        state.execute = execute
        state.max_cycles = max_cycles
        state.status = SupervisorStatus.RUNNING if execute else SupervisorStatus.PLANNED
    else:
        supervisor_dir.mkdir(parents=True, exist_ok=True)
        state = create_initial_state(actual_supervisor_id, goal_text, supervisor_dir, execute, max_cycles)
        state.checkpoint["agent_first_policy"] = {
            "enabled": use_agent_policy,
            "layer": "supervisor",
            "fallback": "deterministic_route",
        }
    write_supervisor_state(state, actual_state_path)

    while state.completed_cycles < state.max_cycles and state.status in {SupervisorStatus.RUNNING, SupervisorStatus.PLANNED}:
        deterministic_action = choose_next_action(state)
        agent_decision: dict[str, Any] | None = None
        action = deterministic_action
        if use_agent_policy:
            action, agent_decision = supervisor_action_from_agent(
                state,
                deterministic_action,
                client=llm_client,
                allow_fallback=True,
            )
        state.actions.append(action)
        state.next_action = action.kind.value
        if agent_decision:
            state.checkpoint["last_supervisor_agent_decision"] = agent_decision
        write_supervisor_state(state, actual_state_path)

        if not state.execute:
            state.status = SupervisorStatus.PLANNED
            agent_first_policy = state.checkpoint.get("agent_first_policy")
            state.checkpoint = {
                "completed_cycles": state.completed_cycles,
                "planned_action": action.model_dump(mode="json"),
            }
            if agent_first_policy:
                state.checkpoint["agent_first_policy"] = agent_first_policy
            if agent_decision:
                state.checkpoint["last_supervisor_agent_decision"] = agent_decision
            write_supervisor_state(state, actual_state_path)
            return state

        action = execute_action(action, state)
        state.actions[-1] = action
        if action.status == SupervisorActionStatus.COMPLETED and action.kind != SupervisorActionKind.ASK_USER:
            state.completed_cycles += 1
        agent_first_policy = state.checkpoint.get("agent_first_policy")
        state.checkpoint = {
            "completed_cycles": state.completed_cycles,
            "last_action": action.model_dump(mode="json"),
        }
        if agent_first_policy:
            state.checkpoint["agent_first_policy"] = agent_first_policy
        if agent_decision:
            state.checkpoint["last_supervisor_agent_decision"] = agent_decision
        if (
            action.kind
            in {
                SupervisorActionKind.RUN_PN_IV,
                SupervisorActionKind.RUN_MOS_CV,
                SupervisorActionKind.RUN_DIODE_BREAKDOWN,
                SupervisorActionKind.RUN_MOSFET_2D,
                SupervisorActionKind.RUN_EXTENDED_DEVICE,
                SupervisorActionKind.RUN_SCHOTTKY_CALIBRATION,
                SupervisorActionKind.RUN_MESH_CONVERGENCE,
            }
            and action.status == SupervisorActionStatus.COMPLETED
        ):
            refresh_experiment_memory(state)
            state.checkpoint["post_action_index"] = state.last_index_summary
        write_supervisor_state(state, actual_state_path)

        if action.kind in {
            SupervisorActionKind.RUN_PN_IV,
            SupervisorActionKind.RUN_MOS_CV,
            SupervisorActionKind.RUN_DIODE_BREAKDOWN,
            SupervisorActionKind.RUN_MOSFET_2D,
            SupervisorActionKind.RUN_EXTENDED_DEVICE,
            SupervisorActionKind.RUN_SCHOTTKY_CALIBRATION,
            SupervisorActionKind.RUN_MESH_CONVERGENCE,
            SupervisorActionKind.GENERATE_REPORT,
            SupervisorActionKind.GENERATE_DASHBOARD,
            SupervisorActionKind.GENERATE_CONCLUSION,
            SupervisorActionKind.GENERATE_REPAIR_PLAN,
            SupervisorActionKind.QUERY_INDEX,
        }:
            state.status = SupervisorStatus.COMPLETED
            state.next_action = "inspect supervisor checkpoint and decide follow-up"
            write_supervisor_state(state, actual_state_path)
            return state

    if state.status not in {SupervisorStatus.FAILED, SupervisorStatus.WAITING_FOR_USER}:
        state.status = SupervisorStatus.COMPLETED
        state.next_action = "maximum supervisor cycles reached"
    write_supervisor_state(state, actual_state_path)
    return state
