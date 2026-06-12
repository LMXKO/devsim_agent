from __future__ import annotations

import json
import mimetypes
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tcad_agent.control_panel import collect_control_panel_data
from tcad_agent.llm import (
    DEFAULT_API_KEY,
    LLMClient,
    LLMConfig,
    save_persisted_llm_settings,
)
from tcad_agent.llm_health import check_llm_health
from tcad_agent.run_queue import (
    cancel_item,
    default_queue_db_path,
    enqueue_run,
    get_item,
    recover_owner_running_items,
    resume_item,
    run_queue_daemon,
    update_item_request,
)
from tcad_agent.task_spec import PROJECT_ROOT
from tcad_agent.tcad_deck import compact_tcad_deck_spec


CAPABILITIES: list[dict[str, Any]] = [
    {
        "name": "Autonomous DEVSIM Agent",
        "tool": "autonomous_devsim_agent",
        "scope": "long-horizon DEVSIM execution, observation, repair, objective checks, conclusion",
    },
    {
        "name": "Device Coverage",
        "tool": "supervisor",
        "scope": "MOSCAP, MOSFET/DIBL, diode/SBD breakdown, LDMOS/IGBT, GaN HEMT, BJT, FinFET/SOI variability",
    },
    {
        "name": "Quality",
        "tool": "physical_benchmark",
        "scope": "physical sanity checks, golden-profile comparison, convergence evidence",
    },
    {
        "name": "Optimization",
        "tool": "multidim_optimizer",
        "scope": "1D/multidimensional sweeps, Pareto objectives, engineering constraints",
    },
    {
        "name": "Reporting",
        "tool": "experiment_report",
        "scope": "ranked results, best parameter summary, artifacts, next action",
    },
]


SEMICONDUCTOR_TEST_CASES: list[dict[str, Any]] = [
    {
        "id": "moscap_cv_oxide_qc",
        "title": "MOSCAP 曲线偏移",
        "goal": (
            "业务任务：这批 MOSCAP 的 C-V 曲线整体有点往负压偏，帮我先按 P-sub 1e17、tox 5nm "
            "从 -2V 扫到 2V 看一下。重点别只给图，顺便判断 Cox、Cmin 和平带点像不像固定电荷问题。"
        ),
        "priority": 20,
        "max_cycles": 12,
        "expected_outputs": ["C-V 曲线", "Cox/Cmin", "平带偏移判断", "中文结论"],
    },
    {
        "id": "moscap_flatband_customer_curve",
        "title": "MOSCAP 客户平带点",
        "goal": (
            "业务任务：客户给的 MOSCAP C-V 平带点比我们仿真大概负偏 0.1V。"
            "帮我用 tox 5nm 和 fixed oxide charge 5e11 cm^-2 做个快速解释，看看等效平带偏移是不是一个量级，最后说值不值得继续校准。"
        ),
        "priority": 16,
        "max_cycles": 14,
        "expected_outputs": ["C-V 结果", "平带偏移估算", "Qf 合理性", "校准建议"],
    },
    {
        "id": "mosfet_idvg_split",
        "title": "MOSFET DIBL 分裂",
        "goal": (
            "业务任务：帮我看一下这个 2D NMOS 的线性区和饱和区 Id-Vg。Vd 用 0.05V 和 1.0V 两个点，"
            "Vg 大概从 0 扫到 1.2V。我要 Vth、SS、Ion/Ioff，还有 DIBL 有没有明显风险；中间收敛失败你自己调步长重跑。"
        ),
        "priority": 18,
        "max_cycles": 16,
        "expected_outputs": ["Id-Vg split", "Vth/SS/Ion-Ioff", "DIBL 风险", "重试过程"],
    },
    {
        "id": "mosfet_output_kink_debug",
        "title": "MOSFET 输出 kink",
        "goal": (
            "业务任务：客户要看 NMOS 输出特性，我想先固定几个 Vg，比如 0.8、1.0、1.2V，"
            "然后把 Vd 从 0 拉到 1.2V。帮我画 Id-Vd，看看 Ron、饱和电流和高压段有没有 kink，最后说这个结果能不能拿去讨论。"
        ),
        "priority": 16,
        "max_cycles": 14,
        "expected_outputs": ["Id-Vd 曲线", "Ron/饱和电流", "kink 检查", "可信度结论"],
    },
    {
        "id": "mesh_vs_model_signoff",
        "title": "MOSFET 签核证据",
        "goal": (
            "业务任务：这个 MOSFET 结果明天要给项目会，我需要一个能站得住的版本。"
            "请先跑主曲线，再做网格或模型可信度检查；如果中间某个 convergence case 挂了，自己重新编排，不要直接丢一个失败给我。"
        ),
        "priority": 17,
        "max_cycles": 16,
        "expected_outputs": ["主曲线", "收敛验证", "自动再编排", "签核结论"],
    },
    {
        "id": "diode_bv_leakage",
        "title": "Diode/SBD BV 漏电",
        "goal": (
            "业务任务：项目里这颗 diode/SBD 反偏漏电有点悬，帮我从 0V 往 -30V 扫一下。"
            "我关心 -5V 漏电和大概 BV，电流到 1e-6A 可以认为接近击穿；如果扫不到或者中途不收敛，你自己缩小 bias step。"
        ),
        "priority": 17,
        "max_cycles": 14,
        "expected_outputs": ["反偏 IV", "-5V 漏电", "BV 估算", "修复记录"],
    },
    {
        "id": "schottky_barrier_calibration",
        "title": "Schottky/SBD 校准",
        "goal": (
            "业务任务：Schottky/SBD 的 golden curve 最近对不上，帮我重新估一下 barrier height 和串联电阻。"
            "不用只报最小误差，把 log-current 拟合残差、ideality factor 和最可疑的偏差区间也说清楚。"
        ),
        "priority": 15,
        "max_cycles": 12,
        "expected_outputs": ["拟合曲线", "barrier/RMSE", "ideality factor", "下一轮建议"],
    },
    {
        "id": "ldmos_bv_ron_tradeoff",
        "title": "LDMOS BV/Ron 取舍",
        "goal": (
            "业务任务：我想先把 LDMOS 的 BV/Ron tradeoff 模板立起来。"
            "帮我按 power MOSFET BV 和 specific Ron 的公开来源口径做规划基线，明确哪些只是 compact baseline，哪些必须升级到高压 TCAD runner 后才能签核。"
        ),
        "priority": 13,
        "max_cycles": 14,
        "expected_outputs": ["BV/Ron 指标", "compact 边界", "高压收敛策略", "runner 晋级步骤"],
    },
    {
        "id": "igbt_turnoff_tail",
        "title": "IGBT 尾电流模板",
        "goal": (
            "业务任务：IGBT 关断尾电流这个方向后面要做长任务自动化。"
            "请按输出曲线、blocking、turn-off tail current 和 lifetime sweep 梳理一个可执行/不可执行边界，给我下一步真实 TCAD runner 的最小实现清单。"
        ),
        "priority": 13,
        "max_cycles": 14,
        "expected_outputs": ["IGBT 指标表", "瞬态缺口", "收敛 playbook", "实现清单"],
    },
    {
        "id": "gan_hemt_output_bv",
        "title": "GaN HEMT 输出/BV",
        "goal": (
            "业务任务：GaN HEMT 输出特性和 BV 风险需要进入模板库。"
            "帮我用公开来源整理 Id-Vg、Id-Vd、2DEG、BV 的指标和模型要求，尤其说明 polarization charge、trap 和 self-heating 现在缺在哪里。"
        ),
        "priority": 15,
        "max_cycles": 14,
        "expected_outputs": ["HEMT 指标", "模型缺口", "收敛策略", "签核边界"],
    },
    {
        "id": "gan_hemt_current_collapse",
        "title": "GaN current collapse",
        "goal": (
            "业务任务：客户问 GaN HEMT current collapse，我想让 agent 先给一个压力/恢复实验计划。"
            "请按 trap occupancy、off-state stress、dynamic Ron ratio 和高场 gate edge 风险组织步骤，明确当前只能规划不能签核。"
        ),
        "priority": 15,
        "max_cycles": 14,
        "expected_outputs": ["stress/recovery 计划", "dynamic Ron", "trap 缺口", "风险结论"],
    },
    {
        "id": "bjt_gummel_gain",
        "title": "BJT Gummel/beta",
        "goal": (
            "业务任务：BJT Gummel 和 beta 提取要从 compact baseline 晋级到真实 runner。"
            "帮我用公开 DEVSIM BJT 示例做路线梳理，列出 base-emitter ramp、collector bias family、beta 噪声地板和 Early voltage 的证据要求。"
        ),
        "priority": 14,
        "max_cycles": 12,
        "expected_outputs": ["Gummel 指标", "beta/Early", "公开来源", "runner 晋级步骤"],
    },
    {
        "id": "bjt_output_early",
        "title": "BJT 输出 Early",
        "goal": (
            "业务任务：BJT 输出曲线要用于 Early voltage 和 leakage review。"
            "请把 Vbe 固定族、Vce sweep、collector leakage 和保存中间解的收敛策略写成 agent 可执行任务草案，并标注当前 compact baseline 的限制。"
        ),
        "priority": 14,
        "max_cycles": 12,
        "expected_outputs": ["输出族计划", "Early 提取", "leakage 窗口", "限制说明"],
    },
    {
        "id": "finfet_dibl_cv",
        "title": "FinFET DIBL/CV",
        "goal": (
            "业务任务：FinFET/GAA 后面要看 DIBL 和 gate capacitance。"
            "帮我按 3D MOSFET、density-gradient、Cgg/Cgd 和短沟道指标整理模板，说明哪些可以用公开 DEVSIM 资料启动，哪些还缺 3D/量子修正验证。"
        ),
        "priority": 14,
        "max_cycles": 14,
        "expected_outputs": ["FinFET 指标", "3D/量子缺口", "DIBL/CV 计划", "签核边界"],
    },
    {
        "id": "soi_finfet_variability",
        "title": "SOI/FinFET 变异",
        "goal": (
            "业务任务：SOI/FinFET 这版我担心 random trap 或几何 split 导致 Vth 分布太宽。"
            "请给一个 nominal-first、mesh reuse、样本分布签核的 variability campaign 计划，别把单点结果当成最终结论。"
        ),
        "priority": 14,
        "max_cycles": 16,
        "expected_outputs": ["Vth 分布计划", "样本策略", "mesh reuse", "签核口径"],
    },
]

PRESET_GOALS: list[str] = [case["goal"] for case in SEMICONDUCTOR_TEST_CASES[:4]]


@dataclass
class WebAppConfig:
    root: Path = PROJECT_ROOT / "runs"
    queue_db_path: Path = default_queue_db_path()
    index_db_path: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8765
    rebuild_index: bool = False
    worker_owner: str = "tcad_web_worker"
    worker_stop_file: Path = PROJECT_ROOT / "runs" / "tcad_web_worker.stop"


def api_key_is_user_configured(value: str | None) -> bool:
    return bool(value) and value != DEFAULT_API_KEY


def mask_api_key(value: str | None) -> str:
    if not api_key_is_user_configured(value):
        return ""
    text = str(value)
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:3]}...{text[-4:]}"


def llm_settings_response(*, settings_path: Path | None = None) -> dict[str, Any]:
    config = LLMConfig.from_env(settings_path=settings_path)
    return {
        "status": "configured" if config.base_url and config.model else "unconfigured",
        "base_url": config.base_url,
        "model": config.model,
        "api_key_set": api_key_is_user_configured(config.api_key),
        "api_key_preview": mask_api_key(config.api_key),
    }


def save_llm_settings_from_payload(payload: dict[str, Any], *, settings_path: Path | None = None) -> dict[str, Any]:
    save_persisted_llm_settings(
        {
            "base_url": payload.get("base_url"),
            "model": payload.get("model"),
            "api_key": payload.get("api_key", ""),
        },
        settings_path=settings_path,
        allow_empty=True,
    )
    return llm_settings_response(settings_path=settings_path)


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def bool_from_payload(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def int_from_payload(payload: dict[str, Any], key: str, default: int, *, minimum: int = 0) -> int:
    value = int(payload.get(key, default))
    return max(minimum, value)


def float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def mission_request_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    goal_text = str(payload.get("goal_text") or payload.get("goal") or "").strip()
    if not goal_text:
        raise ValueError("goal_text is required")
    return {
        "goal_text": goal_text,
        "execute": bool_from_payload(payload, "execute", True),
        "max_cycles": int_from_payload(payload, "max_cycles", 12, minimum=1),
        "supervisor_max_cycles": int_from_payload(payload, "supervisor_max_cycles", 3, minimum=1),
        "use_llm_decomposer": bool_from_payload(payload, "use_llm_decomposer", True),
        "allow_llm_fallback": bool_from_payload(payload, "allow_llm_fallback", True),
    }


def autonomous_request_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    goal_text = str(payload.get("goal_text") or payload.get("goal") or "").strip()
    if not goal_text:
        raise ValueError("goal_text is required")
    request: dict[str, Any] = {
        "goal_text": goal_text,
        "execute": bool_from_payload(payload, "execute", True),
        "max_steps": int_from_payload(payload, "max_steps", int_from_payload(payload, "max_cycles", 12, minimum=1), minimum=1),
        "supervisor_max_cycles": int_from_payload(payload, "supervisor_max_cycles", 3, minimum=1),
        "use_llm": bool_from_payload(payload, "use_llm", bool_from_payload(payload, "use_llm_decomposer", True)),
        "allow_llm_fallback": bool_from_payload(payload, "allow_llm_fallback", True),
        "require_capability_audit": bool_from_payload(payload, "require_capability_audit", True),
    }
    for key in [
        "initial_tool_name",
        "source_state_path",
        "source_deck_path",
        "sentaurus_project_path",
        "sentaurus_profile_path",
    ]:
        if payload.get(key):
            request[key] = payload[key]
    if isinstance(payload.get("initial_request"), dict):
        request["initial_request"] = payload["initial_request"]
    if isinstance(payload.get("deck_patches"), list):
        request["deck_patches"] = payload["deck_patches"]
    if isinstance(payload.get("sentaurus_request"), dict):
        request["sentaurus_request"] = payload["sentaurus_request"]
    if "allow_unverified_deck_patch_execution" in payload:
        request["allow_unverified_deck_patch_execution"] = bool_from_payload(payload, "allow_unverified_deck_patch_execution", False)
    if isinstance(payload.get("objectives"), list):
        request["objectives"] = payload["objectives"]
    if isinstance(payload.get("constraints"), list):
        request["constraints"] = payload["constraints"]
    if "enable_experiment_design" in payload:
        request["enable_experiment_design"] = bool_from_payload(payload, "enable_experiment_design", False)
    if "max_experiment_design_rounds" in payload:
        request["max_experiment_design_rounds"] = int_from_payload(payload, "max_experiment_design_rounds", 1, minimum=0)
    if "auto_execute_experiment_design" in payload:
        request["auto_execute_experiment_design"] = bool_from_payload(payload, "auto_execute_experiment_design", True)
    return request


def soak_request_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    goal_text = str(payload.get("goal_text") or payload.get("goal") or "").strip()
    if not goal_text:
        raise ValueError("goal_text is required")
    max_steps = int_from_payload(
        payload,
        "max_steps",
        int_from_payload(payload, "max_cycles", 24, minimum=1),
        minimum=1,
    )
    autonomous_payload = autonomous_request_from_payload({**payload, "max_steps": max_steps})
    autonomous_payload.pop("goal_text", None)
    autonomous_payload.pop("execute", None)
    autonomous_payload.pop("max_steps", None)
    request: dict[str, Any] = {
        "goal_text": goal_text,
        "execute": bool_from_payload(payload, "execute", True),
        "resume": bool_from_payload(payload, "resume", False),
        "duration_hours": float(payload["duration_hours"]) if payload.get("duration_hours") not in {None, ""} else 1.0,
        "max_steps": max_steps,
        "step_slice": int_from_payload(payload, "step_slice", 4, minimum=1),
        "poll_interval_seconds": float(payload["poll_interval_seconds"]) if payload.get("poll_interval_seconds") not in {None, ""} else 0.0,
        "autonomous_request": autonomous_payload,
        "generate_cockpit": bool_from_payload(payload, "generate_cockpit", True),
        "cockpit_interval_steps": int_from_payload(payload, "cockpit_interval_steps", 1, minimum=1),
    }
    for key in ["soak_id", "soak_root", "agent_id", "agent_root", "cancel_file", "heartbeat_path"]:
        if payload.get(key):
            request[key] = payload[key]
    return request


def enqueue_mission_from_payload(config: WebAppConfig, payload: dict[str, Any]) -> dict[str, Any]:
    requested_tool = str(payload.get("tool_name") or payload.get("tool") or "").strip()
    if requested_tool == "mission_agent":
        request = mission_request_from_payload(payload)
        tool_name = "mission_agent"
        default_tags = ["web", "mission"]
    elif requested_tool == "autonomous_devsim_agent":
        request = autonomous_request_from_payload(payload)
        tool_name = "autonomous_devsim_agent"
        default_tags = ["web", "autonomous"]
    else:
        request = soak_request_from_payload(payload)
        tool_name = "agent_soak"
        default_tags = ["web", "agent_soak", "autonomous"]
    tags = payload.get("tags") or default_tags
    if isinstance(tags, str):
        tags = [item.strip() for item in tags.split(",") if item.strip()]
    item = enqueue_run(
        config.queue_db_path,
        tool_name=tool_name,
        request=request,
        queue_id=payload.get("queue_id") or None,
        priority=int_from_payload(payload, "priority", 10, minimum=-1000000),
        tags=tags,
        max_attempts=int_from_payload(payload, "max_attempts", 2, minimum=1),
        budget_seconds=float_or_none(payload.get("budget_seconds")),
        budget_cases=int(payload["budget_cases"]) if payload.get("budget_cases") not in {None, ""} else None,
    )
    return item.model_dump(mode="json")


def approve_item_confirmation(config: WebAppConfig, queue_id: str) -> dict[str, Any]:
    item = get_item(config.queue_db_path, queue_id)
    if item is None:
        raise FileNotFoundError(f"queue item does not exist: {queue_id}")
    patch: dict[str, Any] = {
        "resume": True,
        "execute": True,
        "allow_user_confirmation_actions": True,
        "allow_unverified_deck_patch_execution": True,
    }
    if item.tool_name == "agent_soak":
        nested = item.request.get("autonomous_request") if isinstance(item.request.get("autonomous_request"), dict) else {}
        patch["autonomous_request"] = {
            **nested,
            "allow_user_confirmation_actions": True,
            "allow_unverified_deck_patch_execution": True,
        }
    updated = update_item_request(
        config.queue_db_path,
        queue_id,
        patch,
        checkpoint_patch={"user_confirmation": "approved"},
    )
    cancel_file = updated.request.get("cancel_file") or ((updated.result or {}).get("cancel_file") if isinstance(updated.result, dict) else None)
    if cancel_file:
        path = Path(str(cancel_file))
        if path.exists():
            path.unlink()
    resumed = resume_item(config.queue_db_path, queue_id)
    return resumed.model_dump(mode="json")


def reject_item_confirmation(config: WebAppConfig, queue_id: str) -> dict[str, Any]:
    update_item_request(
        config.queue_db_path,
        queue_id,
        {"resume": False, "allow_user_confirmation_actions": False},
        checkpoint_patch={"user_confirmation": "rejected"},
    )
    return cancel_item(config.queue_db_path, queue_id).model_dump(mode="json")


class WorkerController:
    def __init__(self, config: WebAppConfig) -> None:
        self.config = config
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.started_at: str | None = None
        self.last_result: dict[str, Any] | None = None
        self.failure_reason: str | None = None
        self.last_recovery: dict[str, Any] | None = None

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def _status_unlocked(self) -> dict[str, Any]:
        return {
            "running": self.is_running(),
            "owner": self.config.worker_owner,
            "started_at": self.started_at,
            "stop_file": str(self.config.worker_stop_file),
            "last_result": self.last_result,
            "failure_reason": self.failure_reason,
            "last_recovery": self.last_recovery,
        }

    def status(self) -> dict[str, Any]:
        with self.lock:
            return self._status_unlocked()

    def start(
        self,
        *,
        concurrency: int = 1,
        lease_seconds: float = 7200.0,
        poll_interval_seconds: float = 5.0,
        max_loops: int | None = None,
        max_idle_loops: int | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            if self.is_running():
                return self._status_unlocked()
            self.config.worker_stop_file.parent.mkdir(parents=True, exist_ok=True)
            if self.config.worker_stop_file.exists():
                self.config.worker_stop_file.unlink()
            self.last_recovery = recover_owner_running_items(
                self.config.queue_db_path,
                owner=self.config.worker_owner,
            )
            self.started_at = utc_timestamp()
            self.failure_reason = None

            def run() -> None:
                try:
                    result = run_queue_daemon(
                        self.config.queue_db_path,
                        owner=self.config.worker_owner,
                        concurrency=concurrency,
                        lease_seconds=lease_seconds,
                        poll_interval_seconds=poll_interval_seconds,
                        max_loops=max_loops,
                        max_idle_loops=max_idle_loops,
                        stop_file=self.config.worker_stop_file,
                    )
                    with self.lock:
                        self.last_result = result.model_dump(mode="json")
                except Exception as exc:
                    with self.lock:
                        self.failure_reason = str(exc)

            self.thread = threading.Thread(target=run, name="tcad-web-worker", daemon=True)
            self.thread.start()
            return self._status_unlocked()

    def stop(self) -> dict[str, Any]:
        with self.lock:
            self.config.worker_stop_file.parent.mkdir(parents=True, exist_ok=True)
            self.config.worker_stop_file.write_text("stop\n", encoding="utf-8")
            return self._status_unlocked()


IMPORTANT_RESULT_KEYS = [
    "status",
    "quality_status",
    "run_id",
    "soak_id",
    "mission_id",
    "supervisor_id",
    "convergence_id",
    "optimize_id",
    "sweep_id",
    "conclusion_path",
    "report_path",
    "dashboard_path",
    "state_path",
    "result_state_path",
    "final_state_path",
    "agent_state_path",
    "latest_cockpit_path",
    "final_agent_status",
    "completed_steps",
    "model_decisions",
    "fallback_decisions",
    "failure_reason",
    "next_action",
]

IMPORTANT_METRIC_KEYS = [
    "points",
    "final_total_current_a",
    "max_abs_current_a",
    "final_capacitance_f_per_cm2",
    "min_capacitance_f_per_cm2",
    "max_capacitance_f_per_cm2",
    "vth_at_threshold_current_v",
    "subthreshold_swing_mv_dec",
    "ion_ioff_ratio",
    "leakage_abs_current_at_target_a",
    "breakdown_voltage_at_threshold_v",
    "breakdown_voltage_v",
    "ideality_factor_estimate",
    "barrier_height_ev",
    "best_rmse_log_current_dec",
    "relative_delta",
]

TEXT_ARTIFACT_EXTENSIONS = {".csv", ".json", ".log", ".md", ".txt", ".dat"}
IMAGE_ARTIFACT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".webp"}


def resolve_artifact_path(raw_path: str | Path) -> Path:
    candidate = Path(str(raw_path))
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    allowed_root = (PROJECT_ROOT / "runs").resolve()
    if not resolved.is_relative_to(allowed_root):
        raise ValueError(f"artifact path is outside runs/: {raw_path}")
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"artifact not found: {raw_path}")
    return resolved


def artifact_content_type(path: Path) -> str:
    if path.suffix.lower() == ".svg":
        return "image/svg+xml"
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def preview_artifact(path: str | Path, *, max_chars: int = 1800) -> dict[str, Any] | None:
    try:
        resolved = resolve_artifact_path(path)
    except (OSError, ValueError):
        return None
    if resolved.suffix.lower() not in TEXT_ARTIFACT_EXTENSIONS:
        return None
    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = text.splitlines()
    preview = "\n".join(lines[:28])
    if len(preview) > max_chars:
        preview = preview[: max_chars - 3] + "..."
    return {
        "path": str(resolved),
        "lines": len(lines),
        "preview": preview,
    }


def clean_markdown_summary_line(line: str, *, max_chars: int = 220) -> str:
    text = line.strip()
    text = re.sub(r"^#{1,6}\s+", "", text)
    text = re.sub(r"^[-*]\s+", "", text)
    text = re.sub(r"^\d+\.\s+", "", text)
    text = text.replace("`", "")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = translate_summary_text(text)
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


SUMMARY_TEXT_REPLACEMENTS = [
    ("TCAD Conclusion:", "TCAD 工程结论："),
    ("TCAD Conclusion：", "TCAD 工程结论："),
    ("Generated:", "生成时间："),
    ("Executive Summary", "摘要"),
    ("Engineering decision", "工程判断"),
    ("Trend Interpretation", "趋势解读"),
    ("Key Metrics", "关键指标"),
    ("Ranked Evidence", "排序证据"),
    ("Physical Benchmark", "物理可信度检查"),
    ("Recommended Next Steps", "建议下一步"),
    ("Next Experiment Plan", "下一轮实验计划"),
    ("Best task/result:", "最优任务/结果："),
    ("Objective value:", "目标值："),
    ("Axis:", "扫描轴："),
    ("; best axis value:", "；最优轴取值："),
    ("Quality distribution:", "质量分布："),
    ("Status distribution:", "执行状态分布："),
    ("Benchmark status:", "物理 benchmark 状态："),
    ("Check counts:", "检查计数："),
    ("Warning codes:", "告警代码："),
    ("Error codes:", "错误代码："),
    ("Benchmark file:", "benchmark 文件："),
    ("Request hint:", "请求提示："),
    ("Source state:", "来源状态："),
    ("Accept the result for project discussion.", "结果可以进入项目讨论。"),
    ("Accept the stable Id-Vd baseline for discussion.", "稳定的 Id-Vd 基线可以进入项目讨论。"),
    ("Do not sign off high-field kink until mesh convergence is checked.", "高场 kink 在完成网格收敛检查前不能签核。"),
    ("Run one finer mesh before signoff.", "签核前再跑一档更细网格。"),
    ("Run x_divisions convergence around the high-Vd segment.", "围绕高 Vd 段运行 x_divisions 收敛检查。"),
    ("Use the best result as the baseline for the next TCAD task.", "将最优结果作为下一轮 TCAD 任务的基线。"),
    ("Increase evidence density using tool_convergence:", "使用工具收敛验证增加证据密度："),
    ("Trend confidence is low because too few completed points were available.", "已完成点数太少，趋势置信度偏低。"),
    ("Existing quality report passed.", "已有质量报告通过。"),
    ("Metric matches the golden profile within relative tolerance.", "指标在 golden profile 相对容差内。"),
    ("At least two completed tool convergence cases are required.", "至少需要两个已完成且有指标值的工具收敛 case。"),
    ("At least two 已完成 tool convergence cases are required.", "至少需要两个已完成且有指标值的工具收敛 case。"),
    ("rerun failed convergence cases before trusting the result", "先重跑失败的收敛 case，再信任该结果"),
    ("rerun 失败 convergence cases before trusting the result", "先重跑失败的收敛 case，再信任该结果"),
    ("Error code:", "错误码："),
    ("accept convergence for this tool/metric", "接受该工具/指标的收敛结果"),
    ("use calibrated Schottky parameters in residual-coupled DEVSIM sweeps", "在残差耦合的 DEVSIM sweep 中使用已校准 Schottky 参数"),
    ("DEVSIM solver did not converge.", "DEVSIM 求解器未收敛。"),
    ("tool execution failed", "工具执行失败"),
    ("maximum repair rounds reached", "已达到最大自动修复轮数"),
    ("Runner failed for an unclassified reason.", "Runner 失败，暂未分类原因。"),
    ("accept MOS capacitor C-V artifacts", "接受 MOS 电容 C-V 产物"),
    ("inspect MOS capacitor artifacts before using the result", "使用结果前检查 MOS 电容产物"),
    ("start first DEVSIM MOS capacitor attempt", "开始第一次 DEVSIM MOS 电容尝试"),
    ("stop and report failure", "停止并报告失败"),
    ("maximum attempts exhausted", "已耗尽最大尝试次数"),
    ("launch reverse-bias PN junction sweep", "启动 PN junction 反偏扫描"),
    ("inspect underlying PN reverse sweep failure", "检查底层 PN 反偏扫描失败"),
    ("rerun the reverse sweep after fixing failed artifacts or sweep settings", "修复失败产物或扫描设置后重跑反偏扫描"),
    ("rerun with a smaller initial reverse-bias step or relaxed solver settings", "使用更小初始反偏步长或更宽松求解器设置重跑"),
    ("review the solver log and rerun with a smaller initial bias step", "检查求解器日志，并用更小初始 bias step 重跑"),
    ("accept result artifacts and proceed to the next TCAD task", "接受当前结果产物，并进入下一项 TCAD 任务"),
    ("stop and inspect failed artifacts", "停止并检查失败产物"),
    ("Doping concentration must be a positive finite value.", "掺杂浓度必须是正的有限值。"),
    ("Doping concentration is outside the configured semiconductor sanity range.", "掺杂浓度超出配置的半导体合理范围。"),
    ("Temperature must be a positive finite value in kelvin.", "温度必须是以 K 为单位的正有限值。"),
    ("Temperature is outside the expected TCAD sanity range.", "温度超出预期 TCAD 合理范围。"),
    ("Carrier lifetime must be a positive finite value in seconds.", "载流子寿命必须是以秒为单位的正有限值。"),
    ("Carrier lifetime is outside a broad semiconductor sanity range.", "载流子寿命超出宽泛半导体合理范围。"),
    ("Carrier mobility must be a positive finite value.", "载流子迁移率必须是正的有限值。"),
    ("Carrier mobility is outside a broad silicon sanity range.", "载流子迁移率超出宽泛硅材料合理范围。"),
    ("Interface trap density is extremely high; verify units and whether the model is equation-coupled.", "界面陷阱密度极高；请检查单位以及模型是否真正耦合进方程。"),
    ("Fixed oxide charge is extremely high; verify cm^-2 units and expected flat-band shift.", "固定氧化层电荷极高；请检查 cm^-2 单位和平带偏移预期。"),
    ("Geometry and mesh dimensions must be positive finite values.", "几何和网格尺寸必须是正的有限值。"),
    ("Mesh spacing is coarse compared with the simulated device length.", "网格间距相对器件长度偏粗。"),
    ("Existing quality report failed; physical benchmark cannot overrule failed numeric/artifact quality.", "已有质量报告失败；物理 benchmark 不能覆盖数值/产物质量失败。"),
    ("Existing quality report is suspicious; use benchmark results as supporting evidence only.", "已有质量报告可疑；benchmark 结果只能作为辅助证据。"),
    ("Forward/reverse current ratio is low; check leakage, contacts, doping, or bias polarity.", "正/反向电流比偏低；请检查漏电、接触、掺杂或 bias 极性。"),
    ("C-V curve is nearly flat across the requested voltage span; inspect bias window, derivative noise, or fixed charge.", "C-V 曲线在请求电压范围内几乎平坦；请检查 bias 窗口、导数噪声或固定电荷。"),
    ("Reverse leakage exceeds the benchmark; verify lifetime, doping, contacts, and reverse-bias range.", "反向漏电超过 benchmark；请检查寿命、掺杂、接触和反偏范围。"),
    ("Breakdown voltage should be negative for this reverse-bias sweep convention.", "按当前反偏扫描约定，击穿电压应为负值。"),
    ("Reverse current magnitude is not monotonic; refine bias steps around the suspicious segment.", "反向电流幅值非单调；请在可疑区间细化 bias step。"),
    ("Subthreshold swing is below the thermal limit; inspect extraction, units, and current floor.", "亚阈值摆幅低于热极限；请检查提取方式、单位和电流下限。"),
    ("Subthreshold swing is large for a usable transfer curve; inspect short-channel/mesh/model settings.", "亚阈值摆幅对可用转移曲线来说偏大；请检查短沟道、网格或模型设置。"),
    ("Id-Vd output curve has decreasing-current segments; inspect bias continuation, mesh, or sign convention.", "Id-Vd 输出曲线存在电流下降区间；请检查 bias continuation、网格或符号约定。"),
    ("Metric differs from the golden profile; inspect model, units, or extraction settings.", "指标与 golden profile 不一致；请检查模型、单位或提取设置。"),
    ("Last two convergence cases differ more than the configured tolerance.", "最后两个收敛 case 的差异超过配置容差。"),
    ("Some aggregate cases failed and should not be used as optimization evidence.", "部分聚合 case 失败，不应作为优化证据。"),
]


def translate_summary_text(text: str) -> str:
    translated = str(text or "")
    for old, new in SUMMARY_TEXT_REPLACEMENTS:
        translated = translated.replace(old, new)
    translated = re.sub(r"\bBenchmark status\b", "物理 benchmark 状态", translated)
    translated = re.sub(r"\bpassed\b", "通过", translated)
    translated = re.sub(r"\bfailed\b", "失败", translated)
    translated = re.sub(r"\bsuspicious\b", "可疑", translated)
    translated = re.sub(r"\bcompleted\b", "已完成", translated)
    translated = re.sub(r"：\s+", "：", translated)
    translated = translated.replace("通过.", "通过。").replace("失败.", "失败。").replace("可疑.", "可疑。")
    translated = translated.replace("N/A.", "N/A。")
    translated = re.sub(r"：\s*\.", "：无。", translated)
    return translated


USER_VISIBLE_TEXT_KEYS = {
    "assumptions",
    "caption",
    "checklist",
    "content",
    "detail",
    "diagnosis",
    "expected_effect",
    "failure_reason",
    "follow_up_checks",
    "llm_failure_reason",
    "message",
    "next_action",
    "rationale",
    "reason",
    "recommended_next_action",
    "strategy",
    "strategy_zh",
    "title",
    "warnings",
}


def translate_user_visible_fields(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {str(item_key): translate_user_visible_fields(item_value, str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [translate_user_visible_fields(item, key) for item in value]
    if isinstance(value, str) and (key in USER_VISIBLE_TEXT_KEYS or (key or "").endswith("_reason")):
        return translate_summary_text(value)
    return value


def translate_conclusion_summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    if normalized.get("title") is not None:
        normalized["title"] = translate_summary_text(str(normalized["title"]))
    blocks = normalized.get("blocks")
    if isinstance(blocks, list):
        translated_blocks: list[dict[str, Any]] = []
        for raw in blocks:
            if not isinstance(raw, dict):
                continue
            block = dict(raw)
            for key in ["label", "title", "content", "caption"]:
                if block.get(key) is not None:
                    block[key] = translate_summary_text(str(block[key]))
            if isinstance(block.get("items"), list):
                block["items"] = [translate_summary_text(str(item)) for item in block["items"]]
            translated_blocks.append(block)
        normalized["blocks"] = translated_blocks
    return normalized


def markdown_sections(text: str) -> tuple[str | None, dict[str, list[str]]]:
    title: str | None = None
    sections: dict[str, list[str]] = {"preamble": []}
    current = "preamble"
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("# "):
            title = clean_markdown_summary_line(line)
            continue
        if line.startswith("## "):
            current = clean_markdown_summary_line(line).lower()
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return title, sections


def section_summary_items(
    sections: dict[str, list[str]],
    section_names: list[str],
    *,
    max_items: int = 3,
) -> list[str]:
    items: list[str] = []
    wanted = {name.lower() for name in section_names}
    for name, lines in sections.items():
        if name.lower() not in wanted:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("###") or stripped.startswith("Source state:") or stripped.startswith("来源状态："):
                continue
            if stripped.lower().startswith("request hint:") or stripped.startswith("请求提示："):
                continue
            if not (stripped.startswith(("-", "*")) or re.match(r"^\d+\.\s+", stripped)):
                continue
            item = clean_markdown_summary_line(stripped)
            if item and item not in items:
                items.append(item)
            if len(items) >= max_items:
                return items
    return items


def compact_conclusion(path: str | Path, *, client: Any | None = None, allow_llm: bool = True) -> dict[str, Any] | None:
    try:
        resolved = resolve_artifact_path(path)
    except (OSError, ValueError):
        return None
    if resolved.suffix.lower() != ".md":
        return None
    cached = read_cached_conclusion_summary(resolved)
    if cached:
        return cached
    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    title, sections = markdown_sections(text[:60000])
    image_candidates = image_candidates_for_conclusion(resolved, text)
    summary: dict[str, Any] | None = None
    failure_reason: str | None = None
    if allow_llm:
        try:
            summary = llm_conclusion_summary(
                title=title,
                markdown_text=text,
                image_candidates=image_candidates,
                client=client,
            )
        except Exception as exc:
            summary = None
            failure_reason = str(exc)
    if summary is None:
        blocks = fallback_conclusion_blocks(title, sections, image_candidates)
        if not blocks:
            return None
        summary = {
            "title": title or "工程结论",
            "blocks": blocks,
            "fallback_used": True,
        }
        if failure_reason:
            summary["llm_failure_reason"] = failure_reason[:240]
    summary["path"] = str(resolved)
    summary = translate_conclusion_summary_payload(summary)
    write_cached_conclusion_summary(resolved, summary)
    return summary


def compact_artifacts(artifacts: Any) -> dict[str, str]:
    if not isinstance(artifacts, dict):
        return {}
    compact: dict[str, str] = {}
    for key, value in artifacts.items():
        if value:
            compact[str(key)] = str(value)
    return {key: compact[key] for key in list(compact)[:8]}


def artifact_previews(artifacts: dict[str, str]) -> dict[str, dict[str, Any]]:
    previews: dict[str, dict[str, Any]] = {}
    for key, path in artifacts.items():
        preview = preview_artifact(path)
        if preview:
            previews[key] = preview
    return previews


def parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def collect_image_artifacts(value: Any, found: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    items = found if found is not None else []
    if not value or len(items) >= 8:
        return items
    if isinstance(value, dict):
        artifacts = value.get("artifacts")
        if isinstance(artifacts, dict):
            for label, raw_path in artifacts.items():
                if len(items) >= 8:
                    break
                if not isinstance(raw_path, str) or not is_image_artifact_path(raw_path):
                    continue
                try:
                    resolved = resolve_artifact_path(raw_path)
                except (OSError, ValueError):
                    continue
                path = str(resolved)
                if not any(existing["path"] == path for existing in items):
                    items.append({"label": str(label), "path": path})
        for item in value.values():
            collect_image_artifacts(item, items)
    elif isinstance(value, list):
        for item in value:
            collect_image_artifacts(item, items)
    return items


def is_image_artifact_path(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_ARTIFACT_EXTENSIONS


def image_candidates_for_conclusion(conclusion_path: Path, markdown_text: str) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    state_paths = [
        conclusion_path.with_name("state.json"),
        conclusion_path.with_name("mission_state.json"),
        conclusion_path.with_name("supervisor_state.json"),
    ]
    try:
        state_paths.extend(sorted(conclusion_path.parent.glob("*.json")))
    except OSError:
        pass
    seen_state_paths: set[Path] = set()
    for state_path in state_paths:
        if state_path in seen_state_paths or state_path.name == conclusion_summary_cache_path(conclusion_path).name:
            continue
        seen_state_paths.add(state_path)
        state = read_json_if_exists(state_path)
        if state:
            collect_image_artifacts(state, candidates)
    for raw_path in re.findall(r"(/[^\\s`\\)]+\\.(?:png|jpg|jpeg|svg|webp))", markdown_text, flags=re.IGNORECASE):
        if len(candidates) >= 8:
            break
        try:
            resolved = resolve_artifact_path(raw_path)
        except (OSError, ValueError):
            continue
        path = str(resolved)
        if not any(existing["path"] == path for existing in candidates):
            candidates.append({"label": "artifact", "path": path})
    return candidates[:8]


def conclusion_summary_cache_path(conclusion_path: Path) -> Path:
    return conclusion_path.with_suffix(".web_summary.json")


def read_cached_conclusion_summary(conclusion_path: Path) -> dict[str, Any] | None:
    cache_path = conclusion_summary_cache_path(conclusion_path)
    if not cache_path.exists():
        return None
    try:
        cache_stat = cache_path.stat()
        if cache_stat.st_mtime < conclusion_path.stat().st_mtime:
            return None
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict) and payload.get("fallback_used") and time.time() - cache_stat.st_mtime > 60:
        return None
    return translate_conclusion_summary_payload(payload) if isinstance(payload, dict) else None


def write_cached_conclusion_summary(conclusion_path: Path, summary: dict[str, Any]) -> None:
    cache_path = conclusion_summary_cache_path(conclusion_path)
    try:
        cache_path.write_text(json.dumps(translate_conclusion_summary_payload(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def normalize_conclusion_blocks(parsed: dict[str, Any], image_candidates: list[dict[str, str]]) -> list[dict[str, Any]]:
    allowed_paths = {item["path"] for item in image_candidates}
    blocks: list[dict[str, Any]] = []
    raw_blocks = parsed.get("blocks")
    if not isinstance(raw_blocks, list):
        raw_blocks = parsed.get("items") if isinstance(parsed.get("items"), list) else []
    for raw in raw_blocks[:10]:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("type") or raw.get("kind") or "text").lower()
        label = clean_markdown_summary_line(str(raw.get("label") or raw.get("title") or ""), max_chars=80)
        if kind in {"image", "plot", "curve"}:
            path = str(raw.get("path") or raw.get("artifact_path") or "")
            if path not in allowed_paths:
                continue
            block = {
                "type": "image",
                "label": label or "图",
                "path": path,
            }
            caption = clean_markdown_summary_line(str(raw.get("caption") or raw.get("content") or ""), max_chars=180)
            if caption:
                block["caption"] = caption
            blocks.append(block)
            continue
        if kind in {"bullets", "list", "points"}:
            items = raw.get("items")
            if not isinstance(items, list):
                content = raw.get("content")
                items = [content] if content else []
            normalized_items = [
                clean_markdown_summary_line(str(item), max_chars=220)
                for item in items[:6]
                if clean_markdown_summary_line(str(item), max_chars=220)
            ]
            if normalized_items:
                blocks.append({"type": "bullets", "label": label or "要点", "items": normalized_items})
            continue
        content = clean_markdown_summary_line(str(raw.get("content") or raw.get("text") or ""), max_chars=420)
        if content:
            blocks.append({"type": "text", "label": label or "说明", "content": content})
    return blocks[:8]


def fallback_conclusion_blocks(title: str | None, sections: dict[str, list[str]], image_candidates: list[dict[str, str]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for label, section_names, max_items in [
        ("结论", ["摘要", "工程判断", "Executive Summary", "Engineering decision"], 3),
        ("关键指标", ["关键指标", "Key Metrics"], 5),
        ("风险", ["物理可信度检查", "异常点", "Physical Benchmark", "Anomalies"], 4),
        ("下一步", ["建议下一步", "下一轮实验计划", "Recommended Next Steps", "Next Experiment Plan"], 5),
    ]:
        items = section_summary_items(sections, section_names, max_items=max_items)
        if items:
            blocks.append({"type": "bullets", "label": label, "items": items})
    for image in image_candidates[:2]:
        blocks.append({"type": "image", "label": image.get("label") or "图", "path": image["path"]})
    if not blocks and title:
        blocks.append({"type": "text", "label": "结论", "content": title})
    return blocks[:8]


def llm_conclusion_summary(
    *,
    title: str | None,
    markdown_text: str,
    image_candidates: list[dict[str, str]],
    client: Any | None = None,
) -> dict[str, Any] | None:
    actual_client = client
    if actual_client is None:
        config = LLMConfig.from_env()
        config = LLMConfig(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
            timeout_seconds=min(config.timeout_seconds, 12.0),
        )
        actual_client = LLMClient(config)
    system = (
        "你是半导体 TCAD 仿真工程师的结果整理助手。"
        "请根据工程结论 Markdown 和可用图片，决定页面最后应该展示哪些关键内容。"
        "不要照搬固定章节，不要输出长文；可以混合文本、要点和图片。"
        "只输出 JSON。"
    )
    user = {
        "task": "提炼 TCAD 工程结论供前端展示",
        "output_schema": {
            "title": "中文短标题",
            "blocks": [
                {"type": "text", "label": "中文短标签", "content": "一段简短中文说明"},
                {"type": "bullets", "label": "中文短标签", "items": ["简短中文要点"]},
                {"type": "image", "label": "中文短标签", "path": "必须来自 image_candidates.path", "caption": "可选中文图注"},
            ],
        },
        "rules": [
            "最多 8 个 blocks。",
            "优先输出工程结论、关键指标、异常/风险、下一步建议。",
            "如果有曲线图或关键图片，选择最有解释价值的 1-3 张图片。",
            "图片 path 必须从 image_candidates 中选择，不能编造。",
            "中文输出，短句，适合前端时间线直接展示。",
        ],
        "title_hint": title,
        "image_candidates": image_candidates,
        "markdown": markdown_text[:16000],
    }
    response = actual_client.chat(system, json.dumps(user, ensure_ascii=False, indent=2), temperature=0.1)
    parsed = parse_json_object(response)
    if not parsed:
        return None
    blocks = normalize_conclusion_blocks(parsed, image_candidates)
    if not blocks:
        return None
    summary = {
        "title": clean_markdown_summary_line(str(parsed.get("title") or title or "工程结论"), max_chars=120),
        "blocks": blocks,
        "model": getattr(actual_client.config, "model", None),
        "fallback_used": False,
    }
    return translate_conclusion_summary_payload(summary)


def text_tail(value: Any, *, max_chars: int = 1800) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    return text if len(text) <= max_chars else "..." + text[-max_chars:]


def compact_command(value: Any) -> list[str] | str | None:
    if isinstance(value, list):
        command = [str(item) for item in value if item is not None]
        return command[:48] if command else None
    if isinstance(value, str) and value:
        return value
    return None


def compact_attempts(attempts: Any) -> list[dict[str, Any]]:
    if not isinstance(attempts, list):
        return []
    rows: list[dict[str, Any]] = []
    for attempt in attempts[:8]:
        if not isinstance(attempt, dict):
            continue
        row = {
            key: attempt.get(key)
            for key in [
                "index",
                "status",
                "step_v",
                "gate_step",
                "drain_step",
                "returncode",
                "failure_class",
                "failure_reason",
                "summary_path",
                "run_dir",
            ]
            if attempt.get(key) is not None
        }
        command = compact_command(attempt.get("command"))
        if command:
            row["command"] = command
        stdout_tail = text_tail(attempt.get("stdout_tail"))
        stderr_tail = text_tail(attempt.get("stderr_tail"))
        if stdout_tail:
            row["stdout_tail"] = stdout_tail
        if stderr_tail:
            row["stderr_tail"] = stderr_tail
        rows.append(row)
    return rows


def compact_cases(cases: Any) -> list[dict[str, Any]]:
    if not isinstance(cases, list):
        return []
    rows: list[dict[str, Any]] = []
    for case in cases[:8]:
        if not isinstance(case, dict):
            continue
        rows.append(
            {
                key: case.get(key)
                for key in [
                    "index",
                    "case_index",
                    "task_id",
                    "status",
                    "quality_status",
                    "value",
                    "values",
                    "metric_value",
                    "objective_value",
                    "failure_reason",
                    "final_state_path",
                    "state_path",
                ]
                if case.get(key) is not None
            }
        )
    return rows


def compact_record_summary(records: Any) -> dict[str, Any]:
    if not isinstance(records, list):
        return {}
    by_status: dict[str, int] = {}
    by_quality: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        status = record.get("status")
        quality = record.get("quality_status")
        kind = record.get("kind")
        if status:
            by_status[str(status)] = by_status.get(str(status), 0) + 1
        if quality:
            by_quality[str(quality)] = by_quality.get(str(quality), 0) + 1
        if kind:
            by_kind[str(kind)] = by_kind.get(str(kind), 0) + 1
    summary: dict[str, Any] = {"count": len([record for record in records if isinstance(record, dict)])}
    if by_status:
        summary["status_counts"] = by_status
    if by_quality:
        summary["quality_counts"] = by_quality
    if by_kind:
        summary["kind_counts"] = dict(sorted(by_kind.items())[:8])
    return summary


def read_json_if_exists(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        return None
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def short_value(value: Any, *, max_chars: int = 900) -> Any:
    if isinstance(value, dict):
        return {str(key): short_value(item, max_chars=max_chars) for key, item in value.items()}
    if isinstance(value, list):
        return [short_value(item, max_chars=max_chars) for item in value[:8]]
    if isinstance(value, str) and len(value) > max_chars:
        return value[: max_chars - 3] + "..."
    return value


def compact_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    compact = {key: metrics.get(key) for key in IMPORTANT_METRIC_KEYS if key in metrics}
    if compact:
        return compact
    return {key: metrics[key] for key in list(metrics)[:8]}


def compact_checkpoint(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        return {}
    keys = [
        "completed_attempts",
        "current_step_v",
        "gate_step_v",
        "drain_step_v",
        "last_failure_class",
        "quality_status",
        "current_round",
        "completed_rounds",
        "best_objective",
        "best_state_path",
        "goal_step_statuses",
        "planned_attempt",
        "experiment_design_plan_path",
        "mutation_refinement_plan_path",
        "semantic_deck_diff",
        "deck_patch_verified",
    ]
    compact = {key: checkpoint.get(key) for key in keys if checkpoint.get(key) is not None}
    tree = compact_agent_hypothesis_tree(checkpoint.get("agent_hypothesis_tree"))
    if tree:
        compact["agent_hypothesis_tree"] = tree
    if compact:
        return compact
    return {key: checkpoint[key] for key in list(checkpoint)[:8]}


def compact_agent_hypothesis_tree(tree: Any) -> dict[str, Any]:
    if not isinstance(tree, dict):
        return {}
    nodes = tree.get("nodes") if isinstance(tree.get("nodes"), list) else []
    last = tree.get("last_hypothesis") if isinstance(tree.get("last_hypothesis"), dict) else (nodes[-1] if nodes and isinstance(nodes[-1], dict) else {})
    output = {
        "count": len(nodes),
        "last": {
            key: last.get(key)
            for key in ["id", "step_index", "action_kind", "tool_name", "hypothesis_zh", "expected_observation", "verdict", "result_state_path"]
            if last.get(key) is not None
        },
        "open_questions": (tree.get("open_questions") or [])[:5],
    }
    return {key: value for key, value in output.items() if value}


def autonomous_cockpit_summary(state: dict[str, Any]) -> dict[str, Any]:
    checkpoint = state.get("checkpoint") if isinstance(state.get("checkpoint"), dict) else {}
    latest_state_path = state.get("latest_state_path") or checkpoint.get("latest_state_path")
    latest_state = read_json_if_exists(latest_state_path)
    final_summary = latest_state.get("final_summary") if isinstance(latest_state, dict) and isinstance(latest_state.get("final_summary"), dict) else {}
    quality = latest_state.get("quality_report") if isinstance(latest_state, dict) and isinstance(latest_state.get("quality_report"), dict) else {}
    artifacts = final_summary.get("artifacts") if isinstance(final_summary.get("artifacts"), dict) else {}
    calibration = final_summary.get("calibration") if isinstance(final_summary.get("calibration"), dict) else quality.get("calibration")
    if not isinstance(calibration, dict) and artifacts.get("calibration"):
        calibration = read_json_if_exists(artifacts.get("calibration"))
    pending_candidate = checkpoint.get("pending_agent_experiment_candidate")
    if isinstance(pending_candidate, dict):
        candidate = {
            key: pending_candidate.get(key)
            for key in ["candidate_id", "action_kind", "tool_name", "reason", "expected_effect", "executed"]
            if pending_candidate.get(key) is not None
        }
    else:
        candidate = {}
    summary = {
        "status": state.get("status"),
        "next_action": state.get("next_action"),
        "latest_state_path": latest_state_path,
        "hypothesis": compact_agent_hypothesis_tree(checkpoint.get("agent_hypothesis_tree")),
        "pending_candidate": candidate,
        "deck_patch": {
            key: checkpoint.get(key)
            for key in ["patched_source_deck", "semantic_deck_diff", "deck_patch_verified", "deck_patch_unverified"]
            if checkpoint.get(key) is not None
        },
        "calibration": {
            key: calibration.get(key)
            for key in ["source_to_reference_y_scale", "rmse_log_dec", "rmse_after_y_scale_fit_log_dec", "recommendations"]
            if isinstance(calibration, dict) and calibration.get(key) is not None
        },
    }
    return {key: value for key, value in summary.items() if value not in ({}, [], None)}


def compact_result(result: Any) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return {"value": short_value(result)} if result is not None else None
    compact = {key: result.get(key) for key in IMPORTANT_RESULT_KEYS if result.get(key) is not None}
    deck = compact_tcad_deck_spec(result.get("tcad_deck_spec"))
    if deck:
        compact["tcad_deck_spec"] = deck
    if result.get("conclusion_path"):
        conclusion = compact_conclusion(str(result["conclusion_path"]), allow_llm=False)
        if conclusion:
            compact["conclusion_summary"] = conclusion
    checkpoint = compact_checkpoint(result.get("checkpoint"))
    if checkpoint:
        compact["checkpoint"] = checkpoint
    top_artifacts = compact_artifacts(result.get("artifacts"))
    if top_artifacts:
        compact["artifacts"] = top_artifacts
        previews = artifact_previews(top_artifacts)
        if previews:
            compact["artifact_previews"] = previews
    attempts = compact_attempts(result.get("attempts"))
    if attempts:
        compact["attempts"] = attempts
    cases = compact_cases(result.get("cases"))
    if cases:
        compact["cases"] = cases
    record_summary = compact_record_summary(result.get("recent_records"))
    if record_summary:
        compact["recent_records_summary"] = record_summary
    quality_report = result.get("quality_report") or {}
    if isinstance(quality_report, dict):
        quality: dict[str, Any] = {}
        if quality_report.get("status") is not None:
            quality["status"] = quality_report.get("status")
        metrics = compact_metrics(quality_report.get("metrics") or {})
        if metrics:
            quality["metrics"] = metrics
        issues = quality_report.get("issues") or []
        if issues:
            quality["issues"] = [
                {
                    "code": issue.get("code"),
                    "severity": issue.get("severity"),
                    "message": issue.get("message"),
                }
                for issue in issues[:3]
                if isinstance(issue, dict)
            ]
        if quality_report.get("recommended_next_action"):
            quality["recommended_next_action"] = quality_report.get("recommended_next_action")
        if quality:
            compact["quality_report"] = quality
    final_summary = result.get("final_summary") or {}
    if isinstance(final_summary, dict):
        artifacts = compact_artifacts(final_summary.get("artifacts") or {})
        summary = {key: final_summary.get(key) for key in IMPORTANT_RESULT_KEYS if final_summary.get(key) is not None}
        if artifacts:
            summary["artifacts"] = artifacts
            previews = artifact_previews(artifacts)
            if previews:
                summary["artifact_previews"] = previews
        metrics = compact_metrics(final_summary)
        if metrics:
            summary["metrics"] = metrics
        if summary:
            compact["final_summary"] = summary
    if not compact:
        compact = {key: result[key] for key in list(result)[:8]}
    else:
        if isinstance(result.get("index"), dict):
            index_summary = {
                key: result["index"].get(key)
                for key in ["tool_name", "status", "records_indexed", "db_path", "root"]
                if result["index"].get(key) is not None
            }
            if index_summary:
                compact["index"] = index_summary
    return short_value(translate_user_visible_fields(compact), max_chars=1800)


STEP_KIND_LABELS_ZH = {
    "decompose_goal": "拆解任务",
    "rebuild_index": "刷新实验记忆",
    "query_history": "检索历史实验",
    "run_supervisor": "执行 TCAD 主任务",
    "run_tool_convergence": "执行收敛验证",
    "run_physical_benchmark": "执行物理可信度检查",
    "agent_replan": "Agent 自动再编排",
    "generate_repair_plan": "生成修复计划",
    "execute_repair": "执行自动修复",
    "generate_conclusion": "生成工程结论",
    "ask_user": "等待用户确认",
    "skip_goal_step": "跳过已满足步骤",
    "noop": "无后续动作",
}

ACTION_KIND_LABELS_ZH = {
    "audit_capability": "审计能力边界",
    "run_supervisor": "运行监督器",
    "run_tool": "运行工具",
    "run_repair_executor": "运行修复器",
    "run_physical_benchmark": "运行物理基准",
    "evaluate_objectives": "评估目标/约束",
    "ingest_deck": "解析 deck",
    "apply_deck_patch": "应用 deck patch",
    "run_user_deck": "运行用户 deck",
    "plan_mutation_refinement": "规划 mutation refinement",
    "plan_sentaurus_patch": "规划 Sentaurus patch",
    "plan_experiment_design": "规划下一实验",
    "generate_dashboard": "生成仪表盘",
    "stop_success": "完成",
    "rebuild_index": "刷新实验索引",
    "query_index": "查询实验索引",
    "run_pn_iv": "运行 PN IV",
    "run_mos_cv": "运行 MOS C-V",
    "run_diode_breakdown": "运行二极管 BV/漏电",
    "run_mosfet_2d": "运行 2D MOSFET",
    "run_extended_device": "运行扩展器件模板",
    "run_schottky_calibration": "运行 Schottky 校准",
    "run_mesh_convergence": "运行网格收敛",
    "generate_report": "生成报告",
    "generate_dashboard": "生成仪表盘",
    "generate_conclusion": "生成工程结论",
    "generate_repair_plan": "生成修复计划",
    "plan_device_template": "规划器件模板",
    "ask_user": "等待用户确认",
    "noop": "无后续动作",
}

REASON_LABELS_ZH = {
    "refresh global experiment memory before mission planning": "执行前刷新全局实验记忆",
    "execute goal-decomposition supervisor step": "执行目标分解中的 TCAD 主任务",
    "refresh experiment memory after the primary TCAD action": "主仿真完成后刷新实验记忆",
    "refresh experiment memory after tool convergence": "收敛验证完成后刷新实验记忆",
    "execute goal-decomposition tool convergence study before accepting TCAD evidence": "接受结果前执行工具收敛验证",
    "run physical benchmark before accepting TCAD evidence": "接受结果前执行物理可信度检查",
    "goal-decomposition physical benchmark step has no TCAD result": "物理可信度检查没有可用 TCAD 结果",
    "diagnose execution issues and adapt the mission plan": "诊断执行问题并自动调整任务编排",
    "latest TCAD result is failed or physically suspicious": "当前 TCAD 结果失败或物理可信度可疑",
    "execute the highest-priority TCAD repair action and re-evaluate quality": "执行最高优先级修复并重新评估质量",
    "skip repair step because the latest TCAD result is accepted": "当前 TCAD 结果已通过，跳过修复",
    "execute goal-decomposition engineering conclusion step": "生成目标分解要求的工程结论",
    "one or more goal-decomposition steps are blocked": "一个或多个目标步骤被阻塞",
    "automatic repair did not produce an accepted result": "自动修复没有得到可信结果",
    "all goal-decomposition steps have reached a terminal state": "所有目标步骤已结束",
    "current durable DAG progress": "当前持久化任务图进度",
    "conclusion artifact is ready": "工程结论文件已生成",
    "refresh experiment memory before deciding the next TCAD action": "决策前刷新实验记忆",
    "goal appears to request a 2D MOSFET Id-Vg/Id-Vd task": "识别为 2D MOSFET Id-Vg/Id-Vd 任务",
    "goal asks to calibrate Schottky IV parameters against a trusted or measured curve": "识别为 Schottky IV 参数校准任务",
    "goal appears to request diode reverse leakage or breakdown": "识别为二极管反偏漏电/BV 任务",
    "goal appears to request a MOS capacitor C-V task": "识别为 MOS 电容 C-V 任务",
    "goal appears to request a PN junction IV task": "识别为 PN junction IV 任务",
}


def label_step_kind(kind: Any) -> str:
    text = str(kind or "unknown")
    return STEP_KIND_LABELS_ZH.get(text, text)


def label_action_kind(kind: Any) -> str:
    text = str(kind or "unknown")
    return ACTION_KIND_LABELS_ZH.get(text, text)


def translate_detail(text: Any) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    return REASON_LABELS_ZH.get(raw, raw)


AGENT_ACTION_LABELS_ZH = {
    "continue": "继续执行",
    "replan": "重新编排",
    "continue_with_risk": "带风险继续",
    "finish": "结束任务",
    "ask_user": "等待用户确认",
}


def label_agent_action(action: Any) -> str:
    text = str(action or "unknown")
    return AGENT_ACTION_LABELS_ZH.get(text, text)


def summarize_agent_decision(decision: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    action = str(decision.get("action") or "unknown")
    primary = observation.get("primary_tcad_record") if isinstance(observation.get("primary_tcad_record"), dict) else {}
    primary_quality = str(primary.get("quality_status") or "").strip()
    observations: list[str] = []
    if primary_quality:
        observations.append(f"主仿真质量：{translate_summary_text(primary_quality)}")
    if primary.get("kind"):
        observations.append(f"主仿真类型：{primary.get('kind')}")
    if primary.get("experiment_id"):
        observations.append(f"实验 ID：{primary.get('experiment_id')}")
    soft_failure_count = observation.get("soft_failure_count")
    if soft_failure_count:
        observations.append(f"软失败次数：{soft_failure_count}")
    blocked = observation.get("blocked_goal_steps") or []
    if blocked:
        observations.append(f"阻塞步骤：{len(blocked)} 个")
    pending = observation.get("pending_goal_kinds") or []
    if pending:
        labels = [label_step_kind(item) for item in pending[:4]]
        observations.append(f"待处理步骤：{'、'.join(labels)}")
    reason = translate_detail(decision.get("reason_zh") or "")
    if action == "replan" and primary_quality in {"suspicious", "failed"}:
        reason = f"发现主仿真质量{translate_summary_text(primary_quality)}，触发重新编排。"
    elif action == "continue_with_risk":
        reason = "重规划后判断风险非阻塞，继续生成带风险说明的工程结论。"
    elif action == "finish":
        reason = reason or "工程结论已生成，本轮任务结束。"
    return {
        "action": action,
        "action_label": label_agent_action(action),
        "reason": reason or "Agent 已观察本步骤并决定下一步。",
        "next_action": decision.get("next_action"),
        "next_action_label": label_step_kind(decision.get("next_action")),
        "observations": observations,
    }


def activity_event(
    *,
    source: str,
    title: str,
    status: str | None = None,
    detail: str | None = None,
    output: dict[str, Any] | None = None,
    path: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "title": title,
        "status": status or "unknown",
        "detail": detail,
        "output": output,
        "path": path,
        "created_at": created_at,
    }


def extract_state_activity(state: dict[str, Any], *, source: str, path: str | None = None) -> list[dict[str, Any]]:
    tool_name = str(state.get("tool_name") or "tcad_state")
    if tool_name == "tcad_mission_agent":
        return extract_mission_activity(state, source=source, path=path)
    if tool_name == "tcad_supervisor":
        return extract_supervisor_activity(state, source=source, path=path)
    if tool_name == "autonomous_devsim_agent":
        return extract_autonomous_agent_activity(state, source=source, path=path)
    if tool_name == "agent_soak":
        return extract_agent_soak_activity(state, source=source, path=path)
    output = compact_result(state)
    return [
        activity_event(
            source=source,
            title=f"{tool_name} result",
            status=str(state.get("status") or "unknown"),
            detail=str(state.get("next_action") or state.get("failure_reason") or ""),
            output=output,
            path=path,
            created_at=state.get("updated_at") or state.get("created_at"),
        )
    ]


def extract_autonomous_agent_activity(state: dict[str, Any], *, source: str, path: str | None = None) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        activity_event(
            source=source,
            title="Agent cockpit",
            status=str(state.get("status") or "unknown"),
            detail=str(state.get("next_action") or state.get("failure_reason") or ""),
            output={"cockpit": autonomous_cockpit_summary(state)},
            path=path,
            created_at=state.get("updated_at") or state.get("created_at"),
        )
    ]
    for step in state.get("steps") or []:
        if not isinstance(step, dict):
            continue
        output = compact_result(step.get("result") or {})
        observation = step.get("observation") if isinstance(step.get("observation"), dict) else {}
        decision = observation.get("agent_decision") if isinstance(observation.get("agent_decision"), dict) else {}
        hypothesis = decision.get("hypothesis_tree_update") or {}
        if isinstance(hypothesis, dict) and hypothesis:
            output = {**(output or {}), "agent_hypothesis": short_value(hypothesis, max_chars=700)}
        if step.get("error"):
            output = {**(output or {}), "error": step.get("error")}
        events.append(
            activity_event(
                source=source,
                title=f"Agent 步骤 {step.get('index')}：{label_action_kind(step.get('kind'))}",
                status=str(step.get("status") or "unknown"),
                detail=translate_detail(step.get("reason")),
                output=output,
                path=step.get("result_state_path") or path,
                created_at=step.get("completed_at") or step.get("started_at"),
            )
        )
    return events


def compact_soak_cycles(cycles: Any) -> list[dict[str, Any]]:
    if not isinstance(cycles, list):
        return []
    rows: list[dict[str, Any]] = []
    for cycle in cycles[-8:]:
        if not isinstance(cycle, dict):
            continue
        rows.append(
            {
                key: cycle.get(key)
                for key in [
                    "index",
                    "status",
                    "agent_status",
                    "requested_max_steps",
                    "agent_steps",
                    "new_steps",
                    "model_decisions",
                    "fallback_decisions",
                    "agent_state_path",
                    "cockpit_path",
                    "failure_reason",
                ]
                if cycle.get(key) is not None
            }
        )
    return rows


def compact_soak_lifecycle(events: Any) -> list[dict[str, Any]]:
    if not isinstance(events, list):
        return []
    rows: list[dict[str, Any]] = []
    for event in events[-8:]:
        if not isinstance(event, dict):
            continue
        rows.append(
            {
                key: event.get(key)
                for key in ["created_at", "event", "detail", "data"]
                if event.get(key) is not None
            }
        )
    return rows


def compact_mission_spec_for_ui(spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict):
        return {}
    intent = spec.get("intent") if isinstance(spec.get("intent"), dict) else {}
    return {
        key: value
        for key, value in {
            "status": spec.get("status"),
            "summary": spec.get("summary"),
            "selected_tool": spec.get("selected_tool"),
            "device_family": intent.get("device_family"),
            "template_id": intent.get("template_id"),
            "allowed_mutations": [
                item.get("name")
                for item in (spec.get("allowed_mutations") or [])[:8]
                if isinstance(item, dict) and item.get("name")
            ],
            "validation_plan": (spec.get("validation_plan") or [])[:8],
            "risk_gates": (spec.get("risk_gates") or [])[:5],
        }.items()
        if value not in (None, [], {})
    }


def extract_agent_soak_activity(state: dict[str, Any], *, source: str, path: str | None = None) -> list[dict[str, Any]]:
    cycles = compact_soak_cycles(state.get("cycles"))
    lifecycle = compact_soak_lifecycle(state.get("lifecycle_events"))
    artifacts = {
        key: state.get(key)
        for key in ["agent_state_path", "latest_cockpit_path", "final_state_path", "heartbeat_path"]
        if state.get(key)
    }
    output = {
        key: state.get(key)
        for key in [
            "soak_id",
            "status",
            "completed_steps",
            "model_decisions",
            "fallback_decisions",
            "final_agent_status",
            "state_path",
        ]
        if state.get(key) is not None
    }
    if cycles:
        output["cycles"] = cycles
    mission_spec = compact_mission_spec_for_ui(state.get("mission_spec"))
    if mission_spec:
        output["mission_spec"] = mission_spec
    if state.get("curve_guidance"):
        output["curve_guidance"] = state.get("curve_guidance")
    if state.get("recovery_events"):
        output["recovery_events"] = (state.get("recovery_events") or [])[-5:]
    if lifecycle:
        output["lifecycle"] = lifecycle
    if state.get("memory_record_path"):
        output["memory_record_path"] = state.get("memory_record_path")
    if artifacts:
        output["artifacts"] = artifacts
    events: list[dict[str, Any]] = [
        activity_event(
            source=source,
            title="Agent Soak",
            status=str(state.get("status") or "unknown"),
            detail=str(state.get("next_action") or state.get("failure_reason") or ""),
            output=short_value(output, max_chars=1800),
            path=path,
            created_at=state.get("updated_at") or state.get("created_at"),
        )
    ]
    for cycle in cycles:
        events.append(
            activity_event(
                source=source,
                title=f"Soak 周期 {cycle.get('index')}",
                status=str(cycle.get("status") or "unknown"),
                detail=(
                    f"agent={cycle.get('agent_status')}, "
                    f"new_steps={cycle.get('new_steps')}, "
                    f"model={cycle.get('model_decisions')}, fallback={cycle.get('fallback_decisions')}"
                ),
                output=cycle,
                path=cycle.get("agent_state_path") or path,
                created_at=state.get("updated_at") or state.get("created_at"),
            )
        )
    if state.get("curve_guidance"):
        guidance = state["curve_guidance"]
        events.append(
            activity_event(
                source=source,
                title="曲线驱动建议",
                status=str(guidance.get("status") or "unknown"),
                detail=str(guidance.get("reason") or ""),
                output=short_value(guidance, max_chars=1400),
                path=guidance.get("source_state_path") or path,
                created_at=guidance.get("created_at") or state.get("updated_at"),
            )
        )
    for event in (state.get("recovery_events") or [])[-4:]:
        if not isinstance(event, dict):
            continue
        events.append(
            activity_event(
                source=source,
                title=f"恢复策略：{event.get('family') or 'unknown'}",
                status="running" if event.get("should_retry") else "planned" if event.get("should_pause_for_user") else "failed",
                detail=str(event.get("reason") or ""),
                output=short_value(event, max_chars=1200),
                path=path,
                created_at=event.get("created_at") or state.get("updated_at"),
            )
        )
    agent_state = read_json_if_exists(state.get("agent_state_path"))
    if agent_state:
        events.extend(extract_autonomous_agent_activity(agent_state, source=source, path=state.get("agent_state_path")))
    return events


def extract_mission_activity(state: dict[str, Any], *, source: str, path: str | None = None) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    checkpoint = state.get("checkpoint") or {}
    decomposition = checkpoint.get("goal_decomposition") or {}
    plan_steps = decomposition.get("steps") or []
    if not plan_steps and checkpoint.get("goal_decomposition_status") == "running":
        events.append(
            activity_event(
                source=source,
                title="任务步骤 0：拆解任务",
                status="running",
                detail="正在等待 LLM 返回目标拆解结果",
                output={
                    "agent_decision": {
                        "action": "decompose_goal",
                        "action_label": "拆解任务",
                        "reason": "Agent 正在把自然语言任务拆成可执行 TCAD 子任务。",
                        "next_action": "rebuild_index",
                        "next_action_label": "刷新实验记忆",
                        "observations": [
                            f"拆解方式：{checkpoint.get('goal_decomposer') or 'deterministic'}",
                            f"模型：{checkpoint.get('goal_decomposer_model') or '等待中'}",
                        ],
                    },
                    "goal_decomposition_status": checkpoint.get("goal_decomposition_status"),
                },
                path=path,
                created_at=state.get("updated_at") or state.get("created_at"),
            )
        )
    if plan_steps:
        events.append(
            activity_event(
                source=source,
                title="任务计划",
                status=str(decomposition.get("status") or "planned"),
                detail=f"{len(plan_steps)} 个目标步骤，由 {checkpoint.get('goal_decomposer') or 'decomposer'} 编排",
                output={
                    "model": checkpoint.get("goal_decomposer_model"),
                    "fallback_used": checkpoint.get("goal_decomposer_fallback_used"),
                    "steps": [
                        {
                            "index": step.get("index"),
                            "kind": step.get("kind"),
                            "title": step.get("title"),
                            "depends_on": step.get("depends_on"),
                        }
                        for step in plan_steps[:10]
                        if isinstance(step, dict)
                    ],
                },
                path=path,
                created_at=state.get("created_at"),
            )
        )
    goal_statuses = checkpoint.get("goal_step_statuses") or {}
    if goal_statuses:
        events.append(
            activity_event(
                source=source,
                title="目标步骤状态",
                status=str(state.get("status") or "unknown"),
                detail=translate_detail("current durable DAG progress"),
                output=goal_statuses,
                path=path,
                created_at=state.get("updated_at"),
            )
        )
    for step in state.get("steps") or []:
        if not isinstance(step, dict):
            continue
        output = compact_result(step.get("result") or {})
        if step.get("error"):
            output = {**(output or {}), "error": step.get("error")}
        display_status = mission_step_display_status(step, goal_statuses)
        events.append(
            activity_event(
                source=source,
                title=f"任务步骤 {step.get('index')}：{label_step_kind(step.get('kind'))}",
                status=display_status,
                detail=translate_detail(step.get("reason")),
                output=output,
                path=(output or {}).get("state_path") or (output or {}).get("conclusion_path") or path,
                created_at=step.get("updated_at") or step.get("created_at"),
            )
        )
        matching_cycle = next(
            (
                cycle
                for cycle in checkpoint.get("controller_cycles") or []
                if isinstance(cycle, dict)
                and ((cycle.get("observation") or {}).get("step_index") == step.get("index"))
            ),
            None,
        )
        if isinstance(matching_cycle, dict):
            decision = matching_cycle.get("decision") if isinstance(matching_cycle.get("decision"), dict) else {}
            observation = matching_cycle.get("observation") if isinstance(matching_cycle.get("observation"), dict) else {}
            agent_decision = summarize_agent_decision(decision, observation)
            events.append(
                activity_event(
                    source=source,
                    title=f"Agent 决策 {matching_cycle.get('cycle') or step.get('index')}",
                    status="completed",
                    detail=agent_decision["reason"],
                    output={
                        "agent_decision": agent_decision,
                        "action": decision.get("action"),
                        "soft_failure_count": observation.get("soft_failure_count"),
                        "goal_status_counts": observation.get("goal_status_counts") or {},
                        "primary_tcad_record": observation.get("primary_tcad_record"),
                        "blocked_goal_steps": observation.get("blocked_goal_steps") or [],
                        "pending_goal_kinds": observation.get("pending_goal_kinds") or [],
                    },
                    path=path,
                    created_at=matching_cycle.get("created_at") or step.get("updated_at") or step.get("created_at"),
                )
            )
    has_conclusion_step_event = any(
        event.get("title", "").endswith("生成工程结论") and (event.get("output") or {}).get("conclusion_summary")
        for event in events
    )
    if checkpoint.get("conclusion_path") and not has_conclusion_step_event:
        conclusion_output = compact_result({"conclusion_path": checkpoint.get("conclusion_path")})
        events.append(
            activity_event(
                source=source,
                title="工程结论",
                status="completed",
                detail=translate_detail("conclusion artifact is ready"),
                output=conclusion_output,
                path=checkpoint.get("conclusion_path"),
                created_at=state.get("updated_at"),
            )
        )
    return events


def mission_step_display_status(step: dict[str, Any], goal_statuses: dict[str, Any]) -> str:
    step_index = step.get("index")
    for raw in goal_statuses.values():
        if not isinstance(raw, dict):
            continue
        if raw.get("mission_step_index") == step_index and raw.get("status"):
            return str(raw["status"])
    return str(step.get("status") or "unknown")


def extract_supervisor_activity(state: dict[str, Any], *, source: str, path: str | None = None) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for action in state.get("actions") or []:
        if not isinstance(action, dict):
            continue
        output = compact_result(action.get("result") or {})
        if action.get("error"):
            output = {**(output or {}), "error": action.get("error")}
        events.append(
            activity_event(
                source=source,
                title=f"执行动作 {action.get('index')}：{label_action_kind(action.get('kind'))}",
                status=str(action.get("status") or "unknown"),
                detail=translate_detail(action.get("reason")),
                output=output,
                path=(output or {}).get("state_path") or path,
                created_at=action.get("updated_at") or action.get("created_at"),
            )
        )
    return events


def collect_execution_activity(queue_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for item in sorted(queue_items, key=lambda row: str(row.get("created_at") or "")):
        queue_id = str(item.get("queue_id") or "queue")
        request = item.get("request") or {}
        goal_text = request.get("goal_text") or request.get("goal") or ""
        queue_output = {
            "queue_id": queue_id,
            "tool_name": item.get("tool_name"),
            "attempts": item.get("attempts"),
            "max_attempts": item.get("max_attempts"),
            "result_state_path": item.get("result_state_path"),
            "failure_reason": item.get("failure_reason"),
        }
        events.append(
            activity_event(
                source=queue_id,
                title=f"{item.get('tool_name')} 已入队",
                status=str(item.get("status") or "unknown"),
                detail=str(goal_text),
                output={key: value for key, value in queue_output.items() if value is not None},
                path=item.get("result_state_path"),
                created_at=item.get("updated_at") or item.get("created_at"),
            )
        )

        state_path = item.get("result_state_path")
        state = read_json_if_exists(state_path) or (item.get("result") if isinstance(item.get("result"), dict) else None)
        if state:
            events.extend(extract_state_activity(state, source=queue_id, path=state_path))
    return events[-120:]


def collect_recent_experiment_activity(records: list[dict[str, Any]], *, limit: int = 6) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for record in records[:limit]:
        if not isinstance(record, dict):
            continue
        source = str(record.get("experiment_id") or record.get("kind") or "experiment")
        state_path = record.get("state_path")
        state = read_json_if_exists(state_path)
        if state:
            extracted = extract_state_activity(state, source=source, path=state_path)
            if extracted:
                events.extend(extracted)
                continue
        events.append(
            activity_event(
                source=source,
                title=f"Recent experiment: {record.get('kind')}",
                status=str(record.get("status") or "unknown"),
                detail=str(record.get("failure_reason") or ""),
                output={
                    key: record.get(key)
                    for key in ["objective_value", "best_axis_path", "best_axis_value", "quality_status", "state_path"]
                    if record.get(key) is not None
                },
                path=state_path,
                created_at=record.get("updated_at") or record.get("created_at"),
            )
        )
    return events[-60:]


def activity_has_artifacts(events: list[dict[str, Any]]) -> bool:
    for event in events:
        output = event.get("output")
        if isinstance(output, dict) and "artifacts" in json.dumps(output, ensure_ascii=False):
            return True
    return False


def activity_has_process(events: list[dict[str, Any]]) -> bool:
    for event in events:
        output = event.get("output")
        if isinstance(output, dict):
            encoded = json.dumps(output, ensure_ascii=False)
            if any(key in encoded for key in ["stdout_tail", "stderr_tail", '"command"', "failure_class"]):
                return True
    return False


def render_app_html() -> str:
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TCAD Mission</title>
  <style>
    :root {
      color: #1d1d1f;
      background: #fbfaf8;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; overflow: hidden; background: #fbfaf8; }
    button, input, textarea { font: inherit; letter-spacing: 0; }
    .app { height: 100vh; display: grid; grid-template-rows: 44px minmax(0, 1fr) auto; }
    header {
      display: flex; align-items: center; justify-content: space-between; gap: 12px;
      padding: 8px 14px; border-bottom: 1px solid #ece8df; background: rgba(251,250,248,.92);
      backdrop-filter: blur(12px);
    }
    .brand { display: flex; align-items: center; gap: 9px; min-width: 0; }
    .brand strong { font-size: 13px; font-weight: 650; white-space: nowrap; }
    .brand span {
      color: #77736c; font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
      max-width: min(54vw, 680px);
    }
    .toolbar { display: flex; gap: 5px; align-items: center; flex-wrap: wrap; justify-content: flex-end; }
    .btn {
      border: 1px solid #ded8ce; background: rgba(255,255,255,.74); color: #2b2b2d; border-radius: 7px;
      min-height: 26px; padding: 3px 8px; cursor: pointer; font-size: 12px; line-height: 18px;
    }
    .btn.primary { background: #202124; border-color: #202124; color: #fff; }
    .btn.danger { color: #9b1c1c; border-color: #ecc8c4; background: #fffaf9; }
    .btn.subtle { color: #6a6258; }
    .btn:disabled { opacity: .55; cursor: default; }
    .modal-backdrop[hidden] { display: none; }
    .modal-backdrop {
      position: fixed; inset: 0; z-index: 80; display: grid; place-items: center;
      padding: 18px; background: rgba(29,29,31,.20); backdrop-filter: blur(10px);
    }
    .settings-dialog {
      width: min(440px, calc(100vw - 28px)); border: 1px solid #e1dbd0; border-radius: 8px;
      background: rgba(255,255,255,.98); box-shadow: 0 18px 52px rgba(38,32,24,.16);
      padding: 14px; color: #252321;
    }
    .settings-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-bottom: 12px; }
    .settings-title { font-size: 14px; font-weight: 650; }
    .settings-close {
      border: 0; background: transparent; color: #777067; cursor: pointer; font-size: 18px; line-height: 1;
      width: 24px; height: 24px; border-radius: 6px;
    }
    .settings-close:hover { background: #f2eee7; color: #302d29; }
    .settings-form { display: grid; gap: 10px; }
    .settings-field { display: grid; gap: 4px; color: #6e675e; font-size: 11px; }
    .settings-field input {
      width: 100%; min-height: 32px; border: 1px solid #d8d1c4; border-radius: 7px;
      padding: 5px 8px; background: #fff; color: #1d1d1f; outline: none; font-size: 13px;
    }
    .settings-field input:focus { border-color: #9f9588; box-shadow: 0 0 0 3px rgba(78,65,48,.08); }
    .settings-note { color: #8a837a; font-size: 11px; min-height: 16px; }
    .settings-note.failed { color: #9b1c1c; }
    .settings-note.passed { color: #155034; }
    .settings-actions { display: flex; justify-content: flex-end; gap: 7px; margin-top: 2px; }
    main { min-height: 0; overflow: auto; padding: 10px 14px 18px; }
    .workspace { max-width: 940px; margin: 0 auto; }
    .meta-line {
      display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 0 0 10px;
      color: #6f6b65; font-size: 12px;
    }
    .metric {
      display: inline-flex; align-items: baseline; gap: 5px; min-height: 24px; padding: 2px 7px;
      border: 1px solid #ece8df; border-radius: 999px; background: rgba(255,255,255,.62);
    }
    .metric strong { color: #2f2f31; font-size: 12px; font-weight: 600; }
    .timeline { display: grid; gap: 0; padding-bottom: 12px; border-top: 1px solid #efebe4; }
    @keyframes entryPop {
      from { opacity: 0; transform: translateY(5px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .entry {
      display: grid; grid-template-columns: 92px minmax(0, 1fr); gap: 14px;
      border-bottom: 1px solid #efebe4; padding: 14px 0;
      animation: entryPop .18s ease-out both;
    }
    .entry.status-failed .entry-title { color: #8a1c1c; }
    .entry.status-running .entry-title { color: #174075; }
    .entry-source {
      color: #a19a90; font-size: 10px; padding-top: 4px; overflow: hidden; text-overflow: ellipsis;
      white-space: nowrap; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .entry-main {
      min-width: 0;
    }
    .entry-head { display: flex; gap: 10px; align-items: flex-start; justify-content: space-between; }
    .entry-title { font-size: 14px; font-weight: 650; line-height: 1.35; color: #202124; }
    .entry-detail { color: #6c665e; font-size: 13px; line-height: 1.5; margin-top: 4px; overflow-wrap: anywhere; }
    .pill {
      display: inline-flex; align-items: center; min-height: 22px; border-radius: 999px; padding: 2px 8px;
      color: #55514b; background: #efebe4; font-size: 11px; white-space: nowrap;
    }
    .pill.completed, .pill.passed { color: #155034; background: #e5f5eb; }
    .pill.running { color: #174075; background: #e7f0fb; }
    .pill.failed, .pill.quality-failed { color: #8a1c1c; background: #fae5e1; }
    .pill.suspicious { color: #704b00; background: #fff3d5; }
    .pill.queued, .pill.planned { color: #704b00; background: #fff3d5; }
    .metrics-grid { margin-top: 9px; display: flex; flex-wrap: wrap; gap: 6px; }
    .metric-chip {
      display: inline-flex; align-items: baseline; gap: 5px; min-height: 24px; padding: 2px 7px;
      border: 1px solid #ebe6dc; border-radius: 7px; background: #fff; font-size: 12px;
    }
    .metric-chip span { color: #817a70; }
    .metric-chip strong { color: #2e2d2b; font-weight: 600; }
    .notice {
      margin-top: 9px; display: grid; gap: 4px; padding: 8px 10px;
      border-left: 2px solid #d9d2c7; background: #fff; color: #514b44; font-size: 12px; line-height: 1.45;
    }
    .notice.warning { border-left-color: #d59b2d; background: #fff9eb; }
    .notice.failed { border-left-color: #c5534b; background: #fff1ee; }
    .notice.passed { border-left-color: #3b9665; background: #f0faf4; }
    .notice-line strong { color: #2d2b29; font-weight: 650; }
    .decision-card {
      margin-top: 9px; border: 1px solid #e8e1d7; border-left: 2px solid #9c9489;
      border-radius: 7px; background: #fff; padding: 8px 10px; color: #3d3934;
      font-size: 12px; line-height: 1.45;
    }
    .decision-card.replan, .decision-card.continue_with_risk {
      border-left-color: #d59b2d; background: #fff9eb;
    }
    .decision-card.finish, .decision-card.continue {
      border-left-color: #3b9665; background: #f5fbf7;
    }
    .decision-kicker { color: #827a70; font-size: 10px; font-weight: 700; text-transform: uppercase; }
    .decision-main { margin-top: 2px; }
    .decision-main strong { color: #24211f; font-weight: 700; }
    .decision-list { margin: 5px 0 0; padding-left: 16px; color: #605a52; }
    .decision-list li { margin: 1px 0; overflow-wrap: anywhere; }
    .cockpit-card {
      margin-top: 9px; display: grid; gap: 6px; border: 1px solid #e8e1d7;
      border-left: 2px solid #667085; border-radius: 7px; background: #fff; padding: 8px 10px;
      color: #3d3934; font-size: 12px; line-height: 1.45;
    }
    .cockpit-row { display: grid; grid-template-columns: 92px minmax(0, 1fr); gap: 8px; }
    .cockpit-row span { color: #827a70; font-size: 10px; font-weight: 700; text-transform: uppercase; }
    .cockpit-row strong { color: #24211f; font-weight: 600; overflow-wrap: anywhere; }
    .conclusion-card {
      margin-top: 10px; border: 1px solid #e5ded3; border-radius: 8px; background: #fff;
      padding: 10px 11px; color: #2e2b28; font-size: 12px; line-height: 1.5;
    }
    .conclusion-title { font-size: 13px; font-weight: 650; margin-bottom: 6px; }
    .conclusion-grid { display: grid; gap: 7px; }
    .conclusion-section strong {
      display: block; color: #6f675d; font-size: 11px; margin-bottom: 2px;
    }
    .conclusion-text { margin: 0; overflow-wrap: anywhere; }
    .conclusion-section ul { margin: 0; padding-left: 16px; }
    .conclusion-section li { margin: 1px 0; overflow-wrap: anywhere; }
    .conclusion-image {
      margin: 2px 0 0; border: 1px solid #ebe6dc; border-radius: 7px; background: #fff;
      padding: 7px;
    }
    .conclusion-image img {
      display: block; max-width: 100%; max-height: 360px; width: auto; height: auto; margin: 0 auto;
    }
    .conclusion-caption { margin-top: 5px; color: #756e64; font-size: 11px; overflow-wrap: anywhere; }
    .output {
      margin-top: 9px; border-radius: 7px; background: #fff; border: 1px solid #ebe6dc;
      padding: 9px; max-height: 260px; overflow: auto;
      font: 11px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap;
      color: #30302f;
    }
    .json-details {
      margin-top: 9px; border: 1px solid #ebe6dc; border-radius: 7px; background: rgba(255,255,255,.66);
    }
    .json-details summary {
      cursor: pointer; min-height: 26px; padding: 4px 8px; color: #756e64; font-size: 11px;
      list-style: none; user-select: none;
    }
    .json-details summary::-webkit-details-marker { display: none; }
    .json-details summary::before { content: "▸"; display: inline-block; width: 14px; color: #9a9288; }
    .json-details[open] summary::before { content: "▾"; }
    .json-details .output { margin: 0; border: 0; border-top: 1px solid #ebe6dc; border-radius: 0 0 7px 7px; }
    .terminal {
      margin-top: 8px; border-radius: 7px; background: #181817; border: 1px solid #2d2c2a;
      color: #e8e1d7; padding: 9px 10px; max-height: 320px; overflow: auto;
      font: 11px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap;
    }
    .terminal-label { color: #9fd4aa; }
    .terminal-error { color: #ffb4a8; }
    .terminal-muted { color: #b9b0a5; }
    .section-label { margin-top: 11px; color: #8a837a; font-size: 10px; font-weight: 700; text-transform: uppercase; }
    .artifact-strip { margin-top: 10px; display: grid; gap: 8px; }
    .artifact-image { border: 1px solid #ebe6dc; border-radius: 8px; background: #fff; padding: 7px; }
    .artifact-image img {
      display: block; max-width: 100%; max-height: 380px; width: auto; height: auto; margin: 0 auto;
    }
    .artifact-links { display: flex; flex-wrap: wrap; gap: 6px; }
    .artifact-link {
      border: 1px solid #ded8ce; border-radius: 7px; background: #fff; padding: 4px 7px;
      color: #315d75; text-decoration: none; font-size: 12px;
    }
    .preview { margin-top: 8px; display: grid; gap: 6px; }
    .preview-title { color: #70695f; font-size: 11px; overflow-wrap: anywhere; }
    .path { margin-top: 7px; color: #837c72; font-size: 11px; overflow-wrap: anywhere; }
    .empty { color: #77736c; text-align: center; padding: 56px 12px; }
    .composer-wrap {
      border-top: 1px solid #ece8df; background: rgba(251,250,248,.96); padding: 9px 14px 12px;
      backdrop-filter: blur(12px);
    }
    .composer { max-width: 940px; margin: 0 auto; display: grid; gap: 7px; }
    .example-menu { position: relative; align-self: flex-end; }
    .example-menu summary {
      display: inline-flex; align-items: center; min-height: 18px; padding: 0 2px;
      border: 0; background: transparent; color: #8a837a; font-size: 11px;
      cursor: pointer; list-style: none; user-select: none;
    }
    .example-menu summary::-webkit-details-marker { display: none; }
    .example-menu[open] summary { color: #3f3b36; }
    .case-rail {
      position: absolute; right: 0; bottom: calc(100% + 6px); z-index: 30;
      width: min(520px, calc(100vw - 28px)); max-height: 280px; overflow: auto;
      display: grid; gap: 6px; padding: 8px; scrollbar-width: thin;
      border: 1px solid #e1dbd0; border-radius: 8px; background: rgba(255,255,255,.98);
      box-shadow: 0 14px 34px rgba(38, 32, 24, .12);
    }
    .case-chip {
      width: 100%; text-align: left; border: 1px solid #e1dbd0; background: rgba(255,255,255,.72);
      color: #343230; border-radius: 7px; padding: 5px 8px; min-height: 28px;
      font-size: 12px; line-height: 1.35; white-space: normal; cursor: pointer;
    }
    .case-chip:hover { background: #fff; border-color: #cfc6b8; }
    .case-title { display: block; color: #24211e; font-weight: 650; }
    .case-desc { display: block; margin-top: 2px; color: #7a746c; font-size: 11px; }
    textarea {
      width: 100%; min-height: 58px; max-height: 150px; resize: vertical; border: 1px solid #d8d1c4;
      border-radius: 8px; padding: 10px 11px; background: #fff; color: #1d1d1f; line-height: 1.45;
      outline: none;
    }
    textarea:focus { border-color: #9f9588; box-shadow: 0 0 0 3px rgba(78, 65, 48, .08); }
    .composer-controls { display: flex; gap: 8px; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; }
    .left-controls, .right-controls { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
    .action-stack { display: flex; flex-direction: column; align-items: flex-end; gap: 3px; position: relative; }
    .latest-jump {
      position: fixed; right: max(16px, calc((100vw - 940px) / 2 + 8px)); bottom: 116px; z-index: 35;
      min-height: 25px; padding: 3px 9px; border-radius: 999px; border: 1px solid #d8d1c4;
      background: rgba(255,255,255,.94); color: #4c4640; font-size: 11px; box-shadow: 0 8px 24px rgba(38,32,24,.12);
      opacity: 0; pointer-events: none; transform: translateY(6px); transition: opacity .14s ease, transform .14s ease;
    }
    .latest-jump.visible { opacity: 1; pointer-events: auto; transform: translateY(0); }
    .toggle {
      display: inline-flex; gap: 5px; align-items: center; min-height: 26px; padding: 3px 7px;
      border: 1px solid #e1dbd0; border-radius: 7px; background: rgba(255,255,255,.74); color: #4b4640; font-size: 12px;
    }
    .toggle input { width: 13px; height: 13px; }
    .number {
      width: 72px; min-height: 26px; border: 1px solid #e1dbd0; border-radius: 7px; padding: 3px 6px;
      background: #fff; font-size: 12px;
    }

    /* Minimal agent UI pass */
    :root { color: #171717; background: #fff; }
    body { background: #fff; color: #171717; }
    .app { grid-template-rows: 46px minmax(0, 1fr) auto; }
    header {
      padding: 9px 18px; border-bottom: 1px solid #ededed; background: rgba(255,255,255,.86);
      backdrop-filter: blur(16px);
    }
    .brand { gap: 10px; }
    .brand strong { font-size: 12px; font-weight: 620; color: #111; }
    .brand span { color: #9a9a9a; font-size: 10px; max-width: min(48vw, 580px); }
    .toolbar { gap: 2px; }
    .btn {
      min-height: 28px; padding: 4px 8px; border: 1px solid transparent; border-radius: 8px;
      background: transparent; color: #525252; font-size: 12px;
    }
    .btn:hover { background: #f5f5f5; color: #171717; }
    .btn.primary { background: #171717; border-color: #171717; color: #fff; }
    .btn.primary:hover { background: #000; border-color: #000; color: #fff; }
    .btn.danger { color: #b42318; border-color: transparent; background: transparent; }
    .btn.danger:hover { background: #fff1f0; }
    .btn.subtle { color: #737373; }
    .settings-dialog {
      border-color: #e5e5e5; border-radius: 8px; background: rgba(255,255,255,.98);
      box-shadow: 0 18px 48px rgba(0,0,0,.12);
    }
    .settings-field { color: #737373; }
    .settings-field input {
      border-color: #e0e0e0; border-radius: 8px; background: #fff; color: #171717;
    }
    .settings-field input:focus { border-color: #171717; box-shadow: 0 0 0 3px rgba(0,0,0,.06); }
    main { padding: 12px 18px 20px; }
    .workspace, .composer { max-width: 900px; }
    .meta-line { gap: 14px; margin: 0 0 8px; color: #a3a3a3; font-size: 11px; }
    .metric { min-height: auto; padding: 0; border: 0; border-radius: 0; background: transparent; gap: 4px; }
    .metric strong { color: #525252; font-size: 11px; font-weight: 550; }
    .timeline { border-top: 0; }
    .entry {
      grid-template-columns: 108px minmax(0, 1fr); gap: 18px; padding: 17px 0;
      border-bottom: 1px solid #f0f0f0;
    }
    .entry-source { color: #adadad; font-size: 10px; padding-top: 3px; }
    .entry-title { color: #171717; font-size: 15px; font-weight: 620; }
    .entry-detail { color: #666; font-size: 13px; line-height: 1.55; }
    .entry.status-failed .entry-title { color: #b42318; }
    .entry.status-running .entry-title { color: #175cd3; }
    .pill {
      min-height: 20px; padding: 1px 7px; border: 1px solid #e5e5e5; background: #fafafa;
      color: #666; font-size: 11px;
    }
    .pill.completed, .pill.passed { color: #067647; background: #f6fef9; border-color: #dcfae6; }
    .pill.running { color: #175cd3; background: #f5f8ff; border-color: #d1e0ff; }
    .pill.failed, .pill.quality-failed { color: #b42318; background: #fffbfa; border-color: #fee4e2; }
    .pill.suspicious, .pill.queued, .pill.planned { color: #93370d; background: #fffcf5; border-color: #fedf89; }
    .metric-chip {
      min-height: 22px; padding: 1px 6px; border-color: #eeeeee; border-radius: 6px;
      background: #fafafa; font-size: 11px;
    }
    .metric-chip span { color: #737373; }
    .metric-chip strong { color: #262626; font-weight: 560; }
    .notice, .decision-card, .conclusion-card {
      border: 1px solid #eeeeee; border-left: 2px solid #d0d5dd; border-radius: 8px;
      background: #fafafa; color: #3f3f46;
    }
    .notice.warning, .decision-card.replan, .decision-card.continue_with_risk {
      border-left-color: #f79009; background: #fffdf7;
    }
    .notice.failed { border-left-color: #f04438; background: #fffafa; }
    .notice.passed, .decision-card.finish, .decision-card.continue {
      border-left-color: #12b76a; background: #fbfffd;
    }
    .decision-kicker, .section-label, .conclusion-section strong {
      color: #8a8a8a; font-size: 10px; font-weight: 650;
    }
    .decision-main strong, .notice-line strong { color: #171717; }
    .decision-list { color: #666; }
    .cockpit-card {
      border-color: #eeeeee; border-left-color: #667085; border-radius: 8px; background: #fafafa;
    }
    .cockpit-row span { color: #8a8a8a; }
    .cockpit-row strong { color: #171717; }
    .conclusion-card { padding: 10px; }
    .conclusion-title { font-size: 13px; color: #171717; }
    .conclusion-image, .artifact-image {
      border-color: #eeeeee; border-radius: 8px; background: #fff; padding: 6px;
    }
    .output {
      border-color: #eeeeee; border-radius: 8px; background: #fafafa; color: #262626;
      font-size: 10.5px;
    }
    .json-details { border-color: #eeeeee; border-radius: 8px; background: #fff; }
    .json-details summary { color: #737373; }
    .json-details .output { border-top-color: #eeeeee; }
    .terminal {
      border-color: #1f1f1f; border-radius: 8px; background: #111; color: #f4f4f5;
    }
    .artifact-link {
      border-color: transparent; background: #f6f6f6; border-radius: 7px; color: #175cd3;
    }
    .path { color: #9a9a9a; font-size: 10px; }
    .empty { color: #9a9a9a; }
    .composer-wrap {
      border-top: 1px solid #ededed; background: rgba(255,255,255,.92);
      padding: 10px 18px 14px; backdrop-filter: blur(16px);
    }
    .composer { gap: 8px; }
    textarea {
      min-height: 54px; max-height: 140px; border-color: #e0e0e0; border-radius: 8px;
      padding: 10px 12px; color: #171717; background: #fff;
    }
    textarea:focus { border-color: #171717; box-shadow: 0 0 0 3px rgba(0,0,0,.06); }
    .composer-controls { gap: 6px; align-items: center; }
    .left-controls, .right-controls { gap: 4px; }
    .advanced-menu, .example-menu { position: relative; }
    .advanced-menu summary, .example-menu summary {
      display: inline-flex; align-items: center; min-height: 26px; padding: 3px 6px;
      border-radius: 7px; color: #8a8a8a; font-size: 12px; cursor: pointer; list-style: none;
      user-select: none;
    }
    .advanced-menu summary::-webkit-details-marker, .example-menu summary::-webkit-details-marker { display: none; }
    .advanced-menu summary:hover, .example-menu summary:hover { background: #f5f5f5; color: #171717; }
    .advanced-menu[open] summary, .example-menu[open] summary { color: #171717; background: #f5f5f5; }
    .advanced-panel {
      position: absolute; left: 0; bottom: calc(100% + 6px); z-index: 34;
      width: min(360px, calc(100vw - 28px)); display: flex; flex-wrap: wrap; gap: 6px;
      padding: 8px; border: 1px solid #e5e5e5; border-radius: 8px; background: rgba(255,255,255,.98);
      box-shadow: 0 14px 36px rgba(0,0,0,.10);
    }
    .toggle {
      min-height: 26px; padding: 3px 7px; border-color: transparent; border-radius: 7px;
      background: #f6f6f6; color: #525252; font-size: 12px;
    }
    .mini-field {
      display: inline-flex; align-items: center; gap: 5px; min-height: 26px; color: #737373;
      font-size: 12px;
    }
    .number {
      width: 64px; min-height: 26px; border-color: #e0e0e0; border-radius: 7px; background: #fff;
      color: #171717;
    }
    .case-rail {
      border-color: #e5e5e5; border-radius: 8px; background: rgba(255,255,255,.98);
      box-shadow: 0 14px 36px rgba(0,0,0,.10);
    }
    .case-chip { border-color: transparent; background: #fff; border-radius: 7px; }
    .case-chip:hover { background: #f7f7f7; border-color: transparent; }
    .case-title { color: #171717; font-weight: 620; }
    .case-desc { color: #737373; }
    .latest-jump {
      border-color: #e5e5e5; background: rgba(255,255,255,.96); color: #525252;
      box-shadow: 0 10px 28px rgba(0,0,0,.10);
    }
    @media (max-width: 720px) {
      body { overflow: auto; }
      .app { min-height: 100vh; height: auto; grid-template-rows: auto minmax(360px, 1fr) auto; }
      header { align-items: stretch; flex-direction: column; }
      .toolbar { justify-content: flex-start; }
      .entry { grid-template-columns: 1fr; gap: 4px; padding: 12px 0; }
      .entry-source { padding-top: 0; }
      .composer-controls { align-items: stretch; flex-direction: column; }
      .left-controls, .right-controls { width: 100%; }
      .action-stack { width: 100%; }
      .example-menu { align-self: flex-end; }
      .right-controls .btn { flex: 1; }
      .latest-jump { right: 14px; bottom: 158px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="brand">
        <strong>TCAD Mission</strong>
        <span id="headerMeta">Loading</span>
      </div>
      <div class="toolbar">
        <button class="btn subtle" id="clearActivityBtn" type="button">清空</button>
        <button class="btn subtle" id="settingsBtn" type="button">设置</button>
      </div>
    </header>
    <main id="scrollRoot">
      <div class="workspace">
        <div class="meta-line">
          <span class="metric">Queue <strong id="metricQueue">0</strong></span>
          <span class="metric">Worker <strong id="metricWorker">off</strong></span>
          <span class="metric">LLM <strong id="metricLlm">configured</strong></span>
          <span class="metric">Experiments <strong id="metricExperiments">0</strong></span>
        </div>
        <div class="timeline" id="activity"></div>
      </div>
    </main>
    <button class="latest-jump" id="latestJumpBtn" type="button" aria-hidden="true">最新</button>
    <div class="composer-wrap">
      <form class="composer" id="missionForm">
        <textarea id="goalText" required placeholder="输入 TCAD 任务"></textarea>
        <div class="composer-controls">
          <div class="left-controls">
            <details class="advanced-menu">
              <summary>选项</summary>
              <div class="advanced-panel">
                <label class="toggle"><input id="execute" type="checkbox" checked>执行</label>
                <label class="toggle"><input id="useLlm" type="checkbox" checked>LLM</label>
                <label class="toggle"><input id="allowFallback" type="checkbox" checked>Fallback</label>
                <label class="mini-field">步数<input class="number" id="maxCycles" type="number" min="1" value="12" aria-label="Max steps"></label>
                <label class="mini-field">优先级<input class="number" id="priority" type="number" value="10" aria-label="Priority"></label>
              </div>
            </details>
          </div>
          <div class="right-controls">
            <div class="action-stack">
            <details class="example-menu">
              <summary>例子</summary>
              <div class="case-rail" id="caseRail" aria-label="Semiconductor engineering test cases"></div>
            </details>
            <button class="btn primary" id="missionActionBtn" type="button">Send</button>
            </div>
          </div>
        </div>
      </form>
    </div>
  </div>
  <div class="modal-backdrop" id="settingsModal" hidden>
    <div class="settings-dialog" role="dialog" aria-modal="true" aria-labelledby="settingsTitle">
      <div class="settings-head">
        <div class="settings-title" id="settingsTitle">大模型设置</div>
        <button class="settings-close" id="settingsCloseBtn" type="button" aria-label="关闭">×</button>
      </div>
      <form class="settings-form" id="settingsForm">
        <label class="settings-field">
          URL
          <input id="llmBaseUrl" name="base_url" type="text" autocomplete="off" required>
        </label>
        <label class="settings-field">
          模型名
          <input id="llmModelName" name="model" type="text" autocomplete="off" required>
        </label>
        <label class="settings-field">
          API Key
          <input id="llmApiKey" name="api_key" type="password" autocomplete="off" placeholder="留空保存为空">
        </label>
        <div class="settings-note" id="settingsNote"></div>
        <div class="settings-actions">
          <button class="btn subtle" id="settingsCancelBtn" type="button">取消</button>
          <button class="btn primary" id="settingsSaveBtn" type="submit">保存</button>
        </div>
      </form>
    </div>
  </div>
  <script type="application/json" id="presetData">__PRESET_JSON__</script>
  <script type="application/json" id="testCaseData">__TEST_CASE_JSON__</script>
  <script>
    const presets = JSON.parse(document.getElementById('presetData').textContent || '[]');
    const testCases = JSON.parse(document.getElementById('testCaseData').textContent || '[]');
    const activityEl = document.getElementById('activity');
    const scrollRoot = document.getElementById('scrollRoot');
    const latestJumpBtn = document.getElementById('latestJumpBtn');
    const clearKey = 'tcadMission.clearBefore';
    let clearBefore = Number(sessionStorage.getItem(clearKey) || 0);
    let workerRunning = false;
    let actionPending = false;
    let autoFollow = true;
    let latestPending = false;
    let displayedActivity = [];
    let displayedActivityKeys = new Set();
    let pendingActivity = [];
    let revealTimer = null;
    const settingsModal = document.getElementById('settingsModal');
    const settingsNote = document.getElementById('settingsNote');
    const llmBaseUrlInput = document.getElementById('llmBaseUrl');
    const llmModelInput = document.getElementById('llmModelName');
    const llmApiKeyInput = document.getElementById('llmApiKey');

    function esc(value) {
      if (value === null || value === undefined) return '';
      return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }

    const statusLabels = {
      completed: '已完成',
      running: '运行中',
      failed: '失败',
      planned: '已计划',
      queued: '排队中',
      waiting_for_user: '等待确认',
      waiting: '等待确认',
      skipped: '已跳过',
      soft_failed: '降级继续',
      suspicious: '需复核',
      unconfigured: '未配置',
      'quality-failed': '质量失败',
      passed: '通过',
      unknown: '未知',
    };

    function labelStatus(status) {
      const text = String(status || 'unknown').toLowerCase();
      return statusLabels[text] || status || '未知';
    }

    function labelNotice(label) {
      const labels = {Failure: '失败', Quality: '质量', Next: '下一步', Issue: '问题'};
      return labels[label] || label;
    }

    function translateMessage(message) {
      const text = String(message || '');
      if (text.includes('tool convergence cases are required')) {
        return '至少需要两个已完成且有指标值的工具收敛 case。';
      }
      if (text.includes('convergence cases before trusting the result')) {
        return '先重跑失败的收敛 case，再信任该结果';
      }
      const translations = [
        ['At least two completed tool convergence cases are required.', '至少需要两个完成的工具收敛 case。'],
        ['rerun failed convergence cases before trusting the result', '先重跑失败的收敛 case，再信任该结果'],
        ['String should match pattern', '字符串不符合工具字段约束'],
        ['tool convergence did not pass', '工具收敛检查未通过'],
        ['automatic repair did not produce an accepted result', '自动修复没有得到可信结果'],
        ['physical quality warning', '物理质量警告'],
        ['failed', '失败'],
        ['suspicious', '可疑'],
        ['passed', '通过'],
        ['completed', '已完成'],
      ];
      for (const [source, target] of translations) {
        if (text.includes(source)) return text.replace(source, target);
      }
      return text;
    }

    function statusPill(status) {
      const text = status || 'unknown';
      return `<span class="pill ${esc(text)}">${esc(labelStatus(text))}</span>`;
    }

    function statusClass(status) {
      const text = String(status || 'unknown').toLowerCase();
      if (text === 'soft_failed') return 'status-suspicious';
      if (text.includes('fail') || text.includes('error')) return 'status-failed';
      if (text.includes('suspicious')) return 'status-suspicious';
      if (text.includes('running')) return 'status-running';
      if (text.includes('waiting')) return 'status-waiting';
      return `status-${text.replace(/[^a-z0-9_-]+/g, '-')}`;
    }

    function outputQualityStatus(value) {
      let status = '';
      const historyKeys = new Set(['recent_records', 'records', 'index', 'recent_records_summary', 'status_counts', 'quality_counts', 'kind_counts']);
      function visit(item, key = '') {
        if (historyKeys.has(key)) return;
        if (!item || typeof item !== 'object' || status === 'failed') return;
        if (!Array.isArray(item)) {
          const report = item.quality_report || {};
          const candidate = String(report.status || item.quality_status || '').toLowerCase();
          if (candidate === 'failed') status = 'failed';
          else if (candidate === 'suspicious' && !status) status = 'suspicious';
        }
        if (Array.isArray(item)) item.forEach(child => visit(child, key));
        else Object.entries(item).forEach(([childKey, child]) => visit(child, childKey));
      }
      visit(value);
      return status;
    }

    function outputRiskStatus(value) {
      let status = '';
      const historyKeys = new Set(['recent_records', 'records', 'index', 'recent_records_summary', 'status_counts', 'quality_counts', 'kind_counts']);
      function visit(item, key = '') {
        if (historyKeys.has(key)) return;
        if (!item || typeof item !== 'object' || status === 'failed') return;
        if (!Array.isArray(item)) {
          const candidate = String(item.status || '').toLowerCase();
          if (candidate === 'failed' || item.failure_reason) status = 'failed';
        }
        if (Array.isArray(item)) item.forEach(child => visit(child, key));
        else Object.entries(item).forEach(([childKey, child]) => visit(child, childKey));
      }
      visit(value);
      return status;
    }

    function eventDisplayStatus(event) {
      const status = String((event && event.status) || 'unknown').toLowerCase();
      const risk = outputRiskStatus(event && event.output);
      if (risk === 'failed' && status !== 'failed') return ['completed', 'soft_failed'].includes(status) ? status : 'failed';
      const quality = outputQualityStatus(event && event.output);
      if (quality === 'failed' && status !== 'failed') return ['completed', 'soft_failed'].includes(status) ? status : 'quality-failed';
      if (quality === 'suspicious' && !['failed', 'quality-failed'].includes(status)) return 'suspicious';
      return status;
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: {'Content-Type': 'application/json'},
        ...options,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.failure_reason || data.error || response.statusText);
      return data;
    }

    function setSettingsNote(text, tone = '') {
      settingsNote.textContent = text || '';
      settingsNote.classList.toggle('failed', tone === 'failed');
      settingsNote.classList.toggle('passed', tone === 'passed');
    }

    function closeSettings() {
      settingsModal.hidden = true;
      setSettingsNote('');
      llmApiKeyInput.value = '';
    }

    async function openSettings() {
      settingsModal.hidden = false;
      setSettingsNote('读取中...');
      try {
        const settings = await api('/api/settings/llm');
        llmBaseUrlInput.value = settings.base_url || '';
        llmModelInput.value = settings.model || '';
        llmApiKeyInput.value = '';
        setSettingsNote(settings.api_key_set ? `API Key 已保存：${settings.api_key_preview || '******'}` : 'API Key 当前为空');
        llmBaseUrlInput.focus();
      } catch (error) {
        setSettingsNote(error.message, 'failed');
      }
    }

    async function saveSettings(event) {
      event.preventDefault();
      setSettingsNote('保存中...');
      try {
        const payload = {
          base_url: llmBaseUrlInput.value.trim(),
          model: llmModelInput.value.trim(),
          api_key: llmApiKeyInput.value.trim(),
        };
        const settings = await api('/api/settings/llm', {method: 'POST', body: JSON.stringify(payload)});
        setSettingsNote(settings.api_key_set ? '已保存' : '已保存，API Key 为空', 'passed');
        setTimeout(closeSettings, 220);
        await refresh();
      } catch (error) {
        setSettingsNote(error.message, 'failed');
      }
    }

    function outputBlock(output) {
      if (!output || Object.keys(output).length === 0) return '';
      const compact = stripVerbose(output);
      if (!compact || Object.keys(compact).length === 0) return '';
      return `<details class="json-details"><summary>JSON 明细</summary><pre class="output">${esc(JSON.stringify(compact, null, 2))}</pre></details>`;
    }

    function stripVerbose(value) {
      if (Array.isArray(value)) {
        return value.map(stripVerbose).filter(item => item !== undefined);
      }
      if (!value || typeof value !== 'object') return value;
      const removed = new Set(['stdout_tail', 'stderr_tail', 'command', 'artifact_previews']);
      const cleaned = {};
      Object.entries(value).forEach(([key, item]) => {
        if (removed.has(key)) return;
        cleaned[key] = stripVerbose(item);
      });
      return cleaned;
    }

    function collectNotices(value, found = [], key = '') {
      const historyKeys = new Set(['recent_records', 'records', 'index', 'recent_records_summary', 'status_counts', 'quality_counts', 'kind_counts']);
      if (historyKeys.has(key)) return found;
      if (!value || typeof value !== 'object' || found.length >= 8) return found;
      if (!Array.isArray(value)) {
        if (value.failure_reason) found.push({tone: 'failed', label: 'Failure', text: value.failure_reason});
        else if (String(value.status || '').toLowerCase() === 'failed') found.push({tone: 'failed', label: 'Failure', text: '工具执行失败'});
        const report = value.quality_report || {};
        if (report && typeof report === 'object') {
          const status = report.status || value.quality_status;
          if (status && status !== 'passed') found.push({tone: 'warning', label: 'Quality', text: status});
          (report.issues || []).slice(0, 4).forEach(issue => {
            if (issue && typeof issue === 'object') {
              found.push({
                tone: issue.severity === 'error' ? 'failed' : 'warning',
                label: issue.code || issue.severity || 'Issue',
                text: issue.message || issue.severity || 'physical quality warning',
              });
            }
          });
          if (status && status !== 'passed' && report.recommended_next_action) {
            found.push({tone: 'warning', label: 'Next', text: report.recommended_next_action});
          }
        }
      }
      if (Array.isArray(value)) value.forEach(item => collectNotices(item, found, key));
      else Object.entries(value).forEach(([childKey, item]) => collectNotices(item, found, childKey));
      return found;
    }

    function noticeBlock(output) {
      const seen = new Set();
      const notices = collectNotices(output).filter(item => {
        const key = `${item.label}:${item.text}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return item.text;
      }).slice(0, 5);
      if (!notices.length) return '';
      const tone = notices.some(item => item.tone === 'failed') ? 'failed' : 'warning';
      return `<div class="notice ${tone}">${notices.map(item =>
        `<div class="notice-line"><strong>${esc(labelNotice(item.label))}</strong> ${esc(translateMessage(item.text))}</div>`
      ).join('')}</div>`;
    }

    function decisionBlock(output) {
      const decision = output && output.agent_decision;
      if (!decision || typeof decision !== 'object') return '';
      const action = String(decision.action || '').replace(/[^a-z0-9_-]+/gi, '_');
      const observations = Array.isArray(decision.observations) ? decision.observations.filter(Boolean).slice(0, 5) : [];
      const next = decision.next_action_label && decision.next_action_label !== 'unknown'
        ? `<div class="decision-main">下一步：${esc(decision.next_action_label)}</div>`
        : '';
      return `<div class="decision-card ${esc(action)}">
        <div class="decision-kicker">Agent 判断</div>
        <div class="decision-main"><strong>${esc(decision.action_label || decision.action || '决策')}</strong> ${esc(decision.reason || '')}</div>
        ${next}
        ${observations.length ? `<ul class="decision-list">${observations.map(item => `<li>${esc(item)}</li>`).join('')}</ul>` : ''}
      </div>`;
    }

    function cockpitBlock(output) {
      const cockpit = output && output.cockpit;
      if (!cockpit || typeof cockpit !== 'object') return '';
      const hypothesis = cockpit.hypothesis && cockpit.hypothesis.last ? cockpit.hypothesis.last : {};
      const candidate = cockpit.pending_candidate || {};
      const deck = cockpit.deck_patch || {};
      const calibration = cockpit.calibration || {};
      const rows = [];
      if (hypothesis.hypothesis_zh) rows.push(['假设', `${hypothesis.hypothesis_zh}${hypothesis.verdict ? ` · ${labelStatus(hypothesis.verdict)}` : ''}`]);
      if (candidate.candidate_id || cockpit.next_action) rows.push(['下一步', candidate.reason || candidate.candidate_id || cockpit.next_action]);
      if (calibration.rmse_log_dec !== undefined) rows.push(['校准', `RMSE ${formatMetric(calibration.rmse_log_dec)} dec${calibration.source_to_reference_y_scale !== undefined ? ` · scale ${formatMetric(calibration.source_to_reference_y_scale)}` : ''}`]);
      if (deck.semantic_deck_diff) rows.push(['Deck', `${deck.deck_patch_verified ? 'verified' : 'review'} · ${deck.semantic_deck_diff}`]);
      if (!rows.length) return '';
      return `<div class="cockpit-card">${rows.slice(0, 4).map(([label, value]) =>
        `<div class="cockpit-row"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`
      ).join('')}</div>`;
    }

    function artifactUrl(path) {
      return `/api/artifact?path=${encodeURIComponent(path)}`;
    }

    function isImageArtifact(path) {
      return /\\.(png|jpg|jpeg|svg|webp)$/i.test(path || '');
    }

    function collectArtifacts(value, found = []) {
      if (!value || typeof value !== 'object') return found;
      if (!Array.isArray(value) && value.artifacts && typeof value.artifacts === 'object') {
        Object.entries(value.artifacts).forEach(([label, path]) => {
          if (typeof path === 'string') found.push({label, path});
        });
      }
      if (Array.isArray(value)) {
        value.forEach(item => collectArtifacts(item, found));
      } else {
        Object.values(value).forEach(item => collectArtifacts(item, found));
      }
      return found;
    }

    function artifactBlock(output) {
      const unique = [];
      const seen = new Set();
      collectArtifacts(output).forEach(item => {
        if (!seen.has(item.path)) {
          seen.add(item.path);
          unique.push(item);
        }
      });
      if (!unique.length) return '';
      const images = unique.filter(item => isImageArtifact(item.path));
      const links = unique.filter(item => !isImageArtifact(item.path));
      const imageHtml = images.map(item => `
        <div class="artifact-image">
          <img src="${artifactUrl(item.path)}" alt="${esc(item.label)}">
        </div>
      `).join('');
      const linkHtml = links.length
        ? `<div class="artifact-links">${links.map(item =>
            `<a class="artifact-link" href="${artifactUrl(item.path)}" target="_blank" rel="noreferrer">${esc(item.label)}</a>`
          ).join('')}</div>`
        : '';
      return `<div class="artifact-strip">${imageHtml}${linkHtml}</div>`;
    }

    function collectConclusions(value, found = []) {
      if (!value || typeof value !== 'object') return found;
      if (!Array.isArray(value) && value.conclusion_summary && typeof value.conclusion_summary === 'object') {
        found.push(value.conclusion_summary);
      }
      if (Array.isArray(value)) value.forEach(item => collectConclusions(item, found));
      else Object.values(value).forEach(item => collectConclusions(item, found));
      return found;
    }

    function conclusionBlock(output) {
      const conclusion = collectConclusions(output)[0];
      if (!conclusion) return '';
      const blocks = conclusionBlocks(conclusion);
      if (!blocks.length && !conclusion.title) return '';
      return `<div class="conclusion-card">
        ${conclusion.title ? `<div class="conclusion-title">${esc(conclusion.title)}</div>` : ''}
        <div class="conclusion-grid">${blocks.map(renderConclusionBlock).join('')}</div>
      </div>`;
    }

    function conclusionBlocks(conclusion) {
      if (Array.isArray(conclusion.blocks) && conclusion.blocks.length) {
        return conclusion.blocks.slice(0, 8);
      }
      return [
        {type: 'bullets', label: '结论', items: conclusion.decision},
        {type: 'bullets', label: '关键指标', items: conclusion.key_metrics},
        {type: 'bullets', label: '趋势', items: conclusion.trend},
        {type: 'bullets', label: '质量/风险', items: [...(conclusion.benchmark || []), ...(conclusion.risks || [])].slice(0, 4)},
        {type: 'bullets', label: '下一步', items: conclusion.next_steps},
      ].filter(block => Array.isArray(block.items) && block.items.length);
    }

    function renderConclusionBlock(block) {
      if (!block || typeof block !== 'object') return '';
      const kind = String(block.type || 'text').toLowerCase();
      const label = block.label ? `<strong>${esc(block.label)}</strong>` : '';
      if (kind === 'image' && block.path) {
        const caption = block.caption ? `<figcaption class="conclusion-caption">${esc(block.caption)}</figcaption>` : '';
        return `<figure class="conclusion-section conclusion-image">
          ${label}
          <img src="${esc(artifactUrl(block.path))}" alt="${esc(block.label || '结论图片')}">
          ${caption}
        </figure>`;
      }
      if (kind === 'bullets' || kind === 'list' || kind === 'points') {
        const items = Array.isArray(block.items) ? block.items.filter(Boolean).slice(0, 6) : [];
        if (!items.length) return '';
        return `<div class="conclusion-section">${label}<ul>
          ${items.map(item => `<li>${esc(item)}</li>`).join('')}
        </ul></div>`;
      }
      const content = block.content || block.text || '';
      if (!content) return '';
      return `<div class="conclusion-section">${label}<p class="conclusion-text">${esc(content)}</p></div>`;
    }

    function collectMetrics(value, found = []) {
      if (!value || typeof value !== 'object') return found;
      if (!Array.isArray(value) && value.quality_report && value.quality_report.metrics) {
        found.push(value.quality_report.metrics);
      }
      if (!Array.isArray(value) && value.final_summary && value.final_summary.metrics) {
        found.push(value.final_summary.metrics);
      }
      if (!Array.isArray(value) && value.metrics && typeof value.metrics === 'object') {
        found.push(value.metrics);
      }
      if (Array.isArray(value)) value.forEach(item => collectMetrics(item, found));
      else Object.values(value).forEach(item => collectMetrics(item, found));
      return found;
    }

    function metricsBlock(output) {
      const merged = {};
      collectMetrics(output).forEach(metrics => {
        Object.entries(metrics || {}).forEach(([key, value]) => {
          if (merged[key] === undefined && value !== null && value !== undefined) merged[key] = value;
        });
      });
      const items = Object.entries(merged).slice(0, 10);
      if (!items.length) return '';
      return `<div class="metrics-grid">${items.map(([key, value]) =>
        `<span class="metric-chip"><span>${esc(metricLabel(key))}</span><strong>${esc(formatMetric(value))}</strong></span>`
      ).join('')}</div>`;
    }

    function metricLabel(key) {
      const labels = {
        cases: 'case 数',
        completed_cases: '完成 case 数',
        relative_delta: '相对差异',
        relative_tolerance: '相对容差',
        points: '点数',
        leakage_abs_current_at_target_a: '目标电压漏电',
        breakdown_voltage_at_threshold_v: '击穿电压',
        ion_ioff_ratio: 'Ion/Ioff',
        vth_at_threshold_current_v: 'Vth',
        subthreshold_swing_mv_dec: 'SS',
      };
      return labels[key] || key;
    }

    function formatMetric(value) {
      if (typeof value === 'number') {
        if (value !== 0 && (Math.abs(value) >= 10000 || Math.abs(value) < 0.001)) return value.toExponential(3);
        return Number(value.toPrecision(5)).toString();
      }
      return value;
    }

    function collectAttempts(value, found = []) {
      if (!value || typeof value !== 'object') return found;
      if (!Array.isArray(value) && Array.isArray(value.attempts)) {
        value.attempts.forEach(attempt => {
          if (attempt && typeof attempt === 'object') found.push(attempt);
        });
      }
      if (Array.isArray(value)) value.forEach(item => collectAttempts(item, found));
      else Object.values(value).forEach(item => collectAttempts(item, found));
      return found;
    }

    function processBlock(output) {
      const attempts = collectAttempts(output).filter(attempt =>
        attempt.command || attempt.stdout_tail || attempt.stderr_tail || attempt.failure_class || attempt.returncode !== undefined
      );
      if (!attempts.length) return '';
      const body = attempts.slice(0, 6).map(attempt => {
        const title = `attempt ${attempt.index || '?'} · ${attempt.status || 'unknown'} · rc=${attempt.returncode ?? 'n/a'}`;
        const command = Array.isArray(attempt.command) ? attempt.command.join(' ') : (attempt.command || '');
        const failure = attempt.failure_class ? `failure_class=${attempt.failure_class} ${attempt.failure_reason || ''}` : '';
        return [
          `<span class="terminal-label">$ ${esc(title)}</span>`,
          command ? `<span class="terminal-muted">${esc(command)}</span>` : '',
          failure ? `<span class="terminal-error">${esc(failure)}</span>` : '',
          attempt.stdout_tail ? esc(attempt.stdout_tail) : '',
          attempt.stderr_tail ? `<span class="terminal-error">${esc(attempt.stderr_tail)}</span>` : '',
        ].filter(Boolean).join('\\n');
      }).join('\\n\\n');
      return `<div class="section-label">Process</div><pre class="terminal">${body}</pre>`;
    }

    function previewBlock(output) {
      const previews = [];
      function visit(value) {
        if (!value || typeof value !== 'object') return;
        if (!Array.isArray(value) && value.artifact_previews && typeof value.artifact_previews === 'object') {
          Object.entries(value.artifact_previews).forEach(([label, preview]) => previews.push({label, preview}));
        }
        if (Array.isArray(value)) value.forEach(visit);
        else Object.values(value).forEach(visit);
      }
      visit(output);
      if (!previews.length) return '';
      return `<div class="preview">${previews.slice(0, 4).map(item => `
        <div>
          <div class="preview-title">${esc(item.label)} · ${esc(item.preview.path || '')}</div>
          <pre class="output">${esc(item.preview.preview || '')}</pre>
        </div>
      `).join('')}</div>`;
    }

    function renderActivity(events) {
      if (!events || events.length === 0) {
        const message = clearBefore
          ? 'Transcript cleared. New TCAD activity will appear here.'
          : 'No mission activity yet.';
        activityEl.innerHTML = `<div class="empty">${esc(message)}</div>`;
        return;
      }
      activityEl.innerHTML = events.map(event => `
        <article class="entry ${statusClass(eventDisplayStatus(event))}">
          <div class="entry-source">${esc(event.source || '')}</div>
          <div class="entry-main">
            <div class="entry-head">
              <div class="entry-title">${esc(event.title || 'Step')}</div>
              ${statusPill(eventDisplayStatus(event))}
            </div>
            ${event.detail ? `<div class="entry-detail">${esc(event.detail)}</div>` : ''}
            ${cockpitBlock(event.output)}
            ${decisionBlock(event.output)}
            ${noticeBlock(event.output)}
            ${metricsBlock(event.output)}
            ${conclusionBlock(event.output)}
            ${artifactBlock(event.output)}
            ${previewBlock(event.output)}
            ${processBlock(event.output)}
            ${outputBlock(event.output)}
            ${event.path ? `<div class="path">${esc(event.path)}</div>` : ''}
          </div>
        </article>
      `).join('');
    }

    function eventKey(event, index) {
      return [
        event && event.source,
        event && event.title,
        event && event.created_at,
        event && event.path,
        index,
      ].map(value => String(value || '')).join('|');
    }

    function keyedActivity(events) {
      return (events || []).map((event, index) => ({...event, _key: eventKey(event, index)}));
    }

    function repaintActivity(events, options = {}) {
      renderActivity(events);
      window.lastActivity = events;
      if (options.follow) {
        scrollToLatest({force: true});
      } else if (options.scroller) {
        preserveScrollPosition(options.scroller, options.previousTop || 0);
      }
    }

    function scheduleActivityReveal(delay = 650) {
      if (revealTimer || !pendingActivity.length) return;
      revealTimer = setTimeout(revealNextActivityEvent, delay);
    }

    function revealNextActivityEvent() {
      revealTimer = null;
      const next = pendingActivity.shift();
      if (!next) return;
      const scroller = activeScroller();
      const previousTop = scroller.scrollTop;
      const shouldFollow = autoFollow || isNearLatest();
      displayedActivity.push(next);
      displayedActivityKeys.add(next._key);
      repaintActivity(displayedActivity, {follow: shouldFollow, scroller, previousTop});
      scheduleActivityReveal();
    }

    function resetActivityReveal() {
      displayedActivity = [];
      displayedActivityKeys = new Set();
      pendingActivity = [];
      if (revealTimer) clearTimeout(revealTimer);
      revealTimer = null;
    }

    function syncActivity(events, options = {}) {
      const keyed = keyedActivity(events);
      const latestKeys = new Set(keyed.map(event => event._key));
      if (!keyed.length) {
        resetActivityReveal();
        repaintActivity([], options);
        return;
      }
      const keyedBy = new Map(keyed.map(event => [event._key, event]));
      displayedActivity = displayedActivity
        .filter(event => latestKeys.has(event._key))
        .map(event => keyedBy.get(event._key) || event);
      pendingActivity = pendingActivity
        .filter(event => latestKeys.has(event._key))
        .map(event => keyedBy.get(event._key) || event);
      displayedActivityKeys = new Set(displayedActivity.map(event => event._key));
      const queuedKeys = new Set(pendingActivity.map(event => event._key));
      keyed.forEach(event => {
        if (!displayedActivityKeys.has(event._key) && !queuedKeys.has(event._key)) {
          pendingActivity.push(event);
          queuedKeys.add(event._key);
        }
      });
      repaintActivity(displayedActivity, options);
      scheduleActivityReveal(displayedActivity.length ? 650 : 120);
    }

    function appendLocalActivity(event) {
      const keyed = {...event, created_at: event.created_at || new Date().toISOString()};
      keyed._key = eventKey(keyed, displayedActivity.length + pendingActivity.length);
      pendingActivity.push(keyed);
      scheduleActivityReveal(50);
    }

    function eventTimeMs(event) {
      const parsed = Date.parse((event && event.created_at) || '');
      return Number.isFinite(parsed) ? parsed : 0;
    }

    function visibleEvents(events) {
      if (!clearBefore) return events || [];
      return (events || []).filter(event => eventTimeMs(event) > clearBefore);
    }

    function activeScroller() {
      const page = document.scrollingElement || document.documentElement;
      const pageScrolls = page && page.scrollHeight > page.clientHeight + 1;
      const rootScrolls = scrollRoot.scrollHeight > scrollRoot.clientHeight + 1;
      return rootScrolls || !pageScrolls ? scrollRoot : page;
    }

    function isNearLatest() {
      const scroller = activeScroller();
      return scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight <= 80;
    }

    function updateLatestJump() {
      const show = !autoFollow && (latestPending || !isNearLatest());
      latestJumpBtn.classList.toggle('visible', show);
      latestJumpBtn.setAttribute('aria-hidden', show ? 'false' : 'true');
    }

    function scrollToLatest(options = {}) {
      if (!options.force && !autoFollow) {
        latestPending = true;
        updateLatestJump();
        return;
      }
      autoFollow = true;
      latestPending = false;
      requestAnimationFrame(() => {
        const scroller = activeScroller();
        scroller.scrollTop = scroller.scrollHeight;
        updateLatestJump();
      });
    }

    function preserveScrollPosition(scroller, scrollTop) {
      requestAnimationFrame(() => {
        scroller.scrollTop = scrollTop;
        latestPending = true;
        updateLatestJump();
      });
    }

    function handleTranscriptScroll() {
      autoFollow = isNearLatest();
      if (autoFollow) latestPending = false;
      updateLatestJump();
    }

    function updateMissionActionButton() {
      const button = document.getElementById('missionActionBtn');
      if (!button) return;
      button.disabled = actionPending;
      button.classList.toggle('primary', !workerRunning);
      button.classList.toggle('danger', workerRunning);
      if (actionPending) {
        button.textContent = workerRunning ? 'Stopping...' : 'Starting...';
      } else {
        button.textContent = workerRunning ? 'Stop' : 'Send';
      }
    }

    function renderState(state) {
      const counts = state.counts || {};
      const worker = state.worker_status || {};
      const llm = state.llm_status || {};
      const filteredActivity = visibleEvents(state.activity || []);
      const scroller = activeScroller();
      const previousTop = scroller.scrollTop;
      const shouldFollow = autoFollow || isNearLatest();
      workerRunning = !!worker.running;
      document.getElementById('metricQueue').textContent = counts.queue_items || 0;
      document.getElementById('metricExperiments').textContent = counts.experiment_records || 0;
      document.getElementById('metricWorker').textContent = worker.running ? 'on' : 'off';
      document.getElementById('metricLlm').textContent = llm.checked_live ? labelStatus(llm.status) : (llm.model || labelStatus(llm.status || 'unconfigured'));
      document.getElementById('headerMeta').textContent = `${state.generated_at || ''}  ${state.root || ''}`;
      updateMissionActionButton();
      syncActivity(filteredActivity, {follow: shouldFollow, scroller, previousTop});
    }

    async function refresh() {
      renderState(await api('/api/state'));
    }

    async function submitAndStartMission() {
      const payload = {
        goal_text: document.getElementById('goalText').value,
        priority: Number(document.getElementById('priority').value || 10),
        max_cycles: Number(document.getElementById('maxCycles').value || 12),
        execute: document.getElementById('execute').checked,
        use_llm_decomposer: document.getElementById('useLlm').checked,
        allow_llm_fallback: document.getElementById('allowFallback').checked,
      };
      if (!payload.goal_text.trim()) {
        document.getElementById('goalText').focus();
        return;
      }
      actionPending = true;
      updateMissionActionButton();
      try {
        await api('/api/missions', {method: 'POST', body: JSON.stringify(payload)});
        document.getElementById('goalText').value = '';
        workerRunning = true;
        updateMissionActionButton();
        await api('/api/worker/start', {method: 'POST', body: JSON.stringify({poll_interval_seconds: 3})});
        await refresh();
        scrollToLatest({force: true});
      } catch (error) {
        appendLocalActivity({source: 'web', title: '提交失败', status: 'failed', detail: error.message});
        scrollToLatest({force: true});
      } finally {
        actionPending = false;
        updateMissionActionButton();
      }
    }

    async function stopMissionWorker() {
      actionPending = true;
      updateMissionActionButton();
      try {
        await api('/api/worker/stop', {method: 'POST', body: '{}'});
        await refresh();
      } catch (error) {
        appendLocalActivity({source: 'web', title: '停止失败', status: 'failed', detail: error.message});
        scrollToLatest({force: true});
      } finally {
        actionPending = false;
        updateMissionActionButton();
      }
    }

    async function handleMissionAction(event) {
      event.preventDefault();
      if (workerRunning) {
        await stopMissionWorker();
      } else {
        await submitAndStartMission();
      }
    }

    function renderTestCases() {
      const rail = document.getElementById('caseRail');
      if (!rail) return;
      rail.innerHTML = testCases.map(item =>
        `<button class="case-chip" type="button" data-case-id="${esc(item.id)}">
          <span class="case-title">${esc(item.title)}</span>
          <span class="case-desc">${esc(caseDescription(item))}</span>
        </button>`
      ).join('');
      rail.querySelectorAll('button[data-case-id]').forEach(button => {
        button.addEventListener('click', () => {
          const item = testCases.find(candidate => candidate.id === button.dataset.caseId);
          if (!item) return;
          document.getElementById('goalText').value = item.goal || '';
          document.getElementById('priority').value = item.priority || 10;
          document.getElementById('maxCycles').value = item.max_cycles || 12;
          const menu = button.closest('details');
          if (menu) menu.open = false;
          document.getElementById('goalText').focus();
        });
      });
    }

    function caseDescription(item) {
      const outputs = Array.isArray(item.expected_outputs) ? item.expected_outputs.slice(0, 3).join(' / ') : '';
      if (outputs) return outputs;
      return String(item.goal || '').slice(0, 72);
    }

    function clearActivity() {
      clearBefore = Date.now();
      sessionStorage.setItem(clearKey, String(clearBefore));
      resetActivityReveal();
      renderActivity([]);
      window.lastActivity = [];
      scrollRoot.scrollTop = 0;
      autoFollow = true;
      latestPending = false;
      updateLatestJump();
    }

    document.getElementById('missionForm').addEventListener('submit', handleMissionAction);
    document.getElementById('missionActionBtn').addEventListener('click', handleMissionAction);
    document.getElementById('clearActivityBtn').addEventListener('click', clearActivity);
    document.getElementById('settingsBtn').addEventListener('click', openSettings);
    document.getElementById('settingsCloseBtn').addEventListener('click', closeSettings);
    document.getElementById('settingsCancelBtn').addEventListener('click', closeSettings);
    document.getElementById('settingsForm').addEventListener('submit', saveSettings);
    settingsModal.addEventListener('click', event => {
      if (event.target === settingsModal) closeSettings();
    });
    window.addEventListener('keydown', event => {
      if (event.key === 'Escape' && !settingsModal.hidden) closeSettings();
    });
    latestJumpBtn.addEventListener('click', () => scrollToLatest({force: true}));
    scrollRoot.addEventListener('scroll', handleTranscriptScroll, {passive: true});
    window.addEventListener('scroll', handleTranscriptScroll, {passive: true});

    if (presets[0]) document.getElementById('goalText').placeholder = presets[0];
    updateMissionActionButton();
    renderTestCases();
    refresh().catch(console.error);
    setInterval(() => refresh().catch(() => {}), 3000);
  </script>
</body>
</html>
"""
    return (
        template.replace("__PRESET_JSON__", json.dumps(PRESET_GOALS, ensure_ascii=False))
        .replace("__TEST_CASE_JSON__", json.dumps(SEMICONDUCTOR_TEST_CASES, ensure_ascii=False))
    )
