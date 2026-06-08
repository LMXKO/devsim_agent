from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.conclusion import generate_experiment_conclusion
from tcad_agent.engineering_intent import parse_engineering_intent
from tcad_agent.experiment_index import default_index_db_path, list_records, rebuild_index
from tcad_agent.golden_curve import GoldenCurveComparisonRequest, run_golden_curve_comparison
from tcad_agent.goal_decomposer import (
    ChatClient,
    GoalStep,
    decompose_goal_with_llm,
    deterministic_decompose_goal,
    replan_goal_after_issue,
)
from tcad_agent.long_horizon_agent import (
    build_long_horizon_snapshot,
    decide_long_horizon_action,
    merge_risk_ledger,
)
from tcad_agent.physical_benchmark import run_physical_benchmark
from tcad_agent.repair_executor import run_repair_executor
from tcad_agent.repair_strategy import build_repair_plan
from tcad_agent.supervisor import run_supervisor
from tcad_agent.task_spec import PROJECT_ROOT
from tcad_agent.tcad_deck import compact_tcad_deck_spec
from tcad_agent.tool_convergence import ToolConvergenceRequest, run_tool_convergence


class MissionStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    WAITING_FOR_USER = "waiting_for_user"
    COMPLETED = "completed"
    FAILED = "failed"


class MissionStepKind(str, Enum):
    DECOMPOSE_GOAL = "decompose_goal"
    REBUILD_INDEX = "rebuild_index"
    QUERY_HISTORY = "query_history"
    RUN_SUPERVISOR = "run_supervisor"
    RUN_TOOL_CONVERGENCE = "run_tool_convergence"
    RUN_GOLDEN_COMPARISON = "run_golden_comparison"
    RUN_PHYSICAL_BENCHMARK = "run_physical_benchmark"
    REPLAN = "agent_replan"
    GENERATE_REPAIR_PLAN = "generate_repair_plan"
    EXECUTE_REPAIR = "execute_repair"
    GENERATE_CONCLUSION = "generate_conclusion"
    ASK_USER = "ask_user"
    SKIP_GOAL_STEP = "skip_goal_step"
    NOOP = "noop"


class MissionStepStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class MissionStep(BaseModel):
    index: int
    kind: MissionStepKind
    status: MissionStepStatus
    reason: str
    request: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class MissionState(BaseModel):
    tool_name: str = "tcad_mission_agent"
    status: MissionStatus
    mission_id: str
    goal_text: str
    mission_dir: str
    created_at: str
    updated_at: str
    execute: bool
    max_cycles: int
    supervisor_max_cycles: int = 3
    completed_cycles: int = 0
    last_index_summary: dict[str, Any] | None = None
    recent_records: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[MissionStep] = Field(default_factory=list)
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    next_action: str | None = None
    failure_reason: str | None = None


RELEVANT_KINDS = {
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
    "tool_convergence",
    "golden_curve_comparison",
}


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_mission_id() -> str:
    return f"mission_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def state_path(mission_root: Path, mission_id: str) -> Path:
    return mission_root / mission_id / "mission_state.json"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json_safe(path_value: str | Path | None) -> dict[str, Any]:
    if not path_value:
        return {}
    try:
        path = Path(path_value)
        if not path.exists() or not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_mission_state(state: MissionState, path: Path) -> None:
    state.updated_at = utc_timestamp()
    write_json(path, state.model_dump(mode="json"))


def load_mission_state(path: Path) -> MissionState:
    return MissionState.model_validate_json(path.read_text(encoding="utf-8"))


def latest_relevant_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in records:
        if record.get("kind") in RELEVANT_KINDS and record.get("state_path"):
            return record
    return None


def goal_steps(state: MissionState) -> list[dict[str, Any]]:
    decomposition = state.checkpoint.get("goal_decomposition") or {}
    steps = decomposition.get("steps") or []
    return [step for step in steps if isinstance(step, dict)]


def goal_step_statuses(state: MissionState) -> dict[str, Any]:
    statuses = state.checkpoint.setdefault("goal_step_statuses", {})
    if not isinstance(statuses, dict):
        statuses = {}
        state.checkpoint["goal_step_statuses"] = statuses
    return statuses


def goal_step_status(state: MissionState, index: int) -> str | None:
    raw = goal_step_statuses(state).get(str(index))
    if isinstance(raw, dict):
        status = raw.get("status")
        return str(status) if status is not None else None
    return str(raw) if raw is not None else None


def mark_goal_step(
    state: MissionState,
    index: int,
    status: str,
    *,
    kind: str | None = None,
    mission_step: MissionStep | None = None,
    result: dict[str, Any] | None = None,
) -> None:
    entry: dict[str, Any] = {
        "status": status,
        "updated_at": utc_timestamp(),
    }
    if kind:
        entry["kind"] = kind
    if mission_step:
        entry["mission_step_index"] = mission_step.index
        entry["mission_step_kind"] = mission_step.kind.value
    if result:
        entry["result"] = result
    goal_step_statuses(state)[str(index)] = entry


def goal_step_indexes_by_status(state: MissionState, accepted: set[str]) -> set[int]:
    indexes: set[int] = set()
    for raw_index, raw_status in goal_step_statuses(state).items():
        status = raw_status.get("status") if isinstance(raw_status, dict) else raw_status
        if status in accepted:
            try:
                indexes.add(int(raw_index))
            except (TypeError, ValueError):
                pass
    return indexes


def next_ready_goal_step(state: MissionState) -> dict[str, Any] | None:
    finished = goal_step_indexes_by_status(state, {"completed", "skipped", "soft_failed"})
    blocked = goal_step_indexes_by_status(state, {"failed", "waiting_for_user"})
    steps = goal_steps(state)
    for step in steps:
        try:
            index = int(step.get("index"))
        except (TypeError, ValueError):
            continue
        if index in finished or index in blocked:
            continue
        depends_on = step.get("depends_on") or []
        if all(int(dep) in finished for dep in depends_on):
            return step
    return None


def goal_step_request_fields(step: dict[str, Any]) -> dict[str, Any]:
    try:
        index = int(step.get("index"))
    except (TypeError, ValueError):
        index = 0
    return {
        "goal_step_index": index,
        "goal_step_kind": step.get("kind"),
        "goal_step_title": step.get("title"),
        "goal_step_stop_on_failure": bool(step.get("stop_on_failure", True)),
    }


def record_needs_repair(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    return record.get("status") == "failed" or record.get("quality_status") in {"failed", "suspicious"}


def tool_result_state_path(result: dict[str, Any]) -> str | None:
    for key in ["state_path", "result_state_path", "final_state_path", "verified_state_path"]:
        value = result.get(key)
        if value:
            return str(value)
    final_summary = result.get("final_summary") or {}
    if isinstance(final_summary, dict):
        artifacts = final_summary.get("artifacts") or {}
        if isinstance(artifacts, dict) and artifacts.get("state"):
            return str(artifacts["state"])
    run_dir = result.get("run_dir") or result.get("calibration_dir") or result.get("convergence_dir")
    if run_dir:
        candidate = Path(str(run_dir)) / "state.json"
        if candidate.exists():
            return str(candidate.resolve())
    return None


def tool_result_quality_status(result: dict[str, Any]) -> str | None:
    quality_report = result.get("quality_report") or {}
    if isinstance(quality_report, dict) and quality_report.get("status"):
        return str(quality_report["status"])
    if result.get("quality_status"):
        return str(result["quality_status"])
    final_summary = result.get("final_summary") or {}
    if isinstance(final_summary, dict) and final_summary.get("quality_status"):
        return str(final_summary["quality_status"])
    return None


def supervisor_tcad_record(supervisor_result: dict[str, Any]) -> dict[str, Any] | None:
    actions = supervisor_result.get("actions") or []
    candidates = [action for action in actions if isinstance(action, dict)]
    checkpoint_action = (supervisor_result.get("checkpoint") or {}).get("last_action")
    if isinstance(checkpoint_action, dict):
        candidates.append(checkpoint_action)
    for action in reversed(candidates):
        result = action.get("result") or {}
        if not isinstance(result, dict):
            continue
        state_path = tool_result_state_path(result)
        if not state_path:
            continue
        kind = result.get("tool_name") or action.get("kind")
        if kind not in RELEVANT_KINDS and str(kind) not in {
            "run_pn_iv",
            "run_mos_cv",
            "run_diode_breakdown",
            "run_mosfet_2d",
            "run_extended_device",
            "run_schottky_calibration",
            "run_mesh_convergence",
        }:
            continue
        return {
            "experiment_id": result.get("run_id")
            or result.get("calibration_id")
            or result.get("convergence_id")
            or f"{supervisor_result.get('supervisor_id')}_tcad",
            "kind": str(kind),
            "status": result.get("status") or action.get("status"),
            "quality_status": tool_result_quality_status(result),
            "failure_reason": result.get("failure_reason"),
            "state_path": state_path,
        }
    return None


def tool_convergence_checkpoint_record(state: MissionState) -> dict[str, Any] | None:
    state_path_value = state.checkpoint.get("tool_convergence_state_path")
    if not state_path_value:
        return None
    return {
        "experiment_id": state.checkpoint.get("tool_convergence_id") or f"{state.mission_id}_tool_convergence",
        "kind": "tool_convergence",
        "status": state.checkpoint.get("tool_convergence_status"),
        "quality_status": state.checkpoint.get("tool_convergence_quality_status"),
        "state_path": str(state_path_value),
    }


def golden_comparison_checkpoint_record(state: MissionState) -> dict[str, Any] | None:
    state_path_value = state.checkpoint.get("golden_comparison_state_path")
    if not state_path_value:
        return None
    return {
        "experiment_id": state.checkpoint.get("golden_comparison_id") or f"{state.mission_id}_golden_comparison",
        "kind": "golden_curve_comparison",
        "status": state.checkpoint.get("golden_comparison_status"),
        "quality_status": state.checkpoint.get("golden_comparison_quality_status"),
        "state_path": str(state_path_value),
    }


def current_evidence_record(state: MissionState) -> dict[str, Any] | None:
    if state.checkpoint.get("golden_comparison_completed"):
        record = golden_comparison_checkpoint_record(state)
        if record:
            return record
    if state.checkpoint.get("post_tool_convergence_index_refreshed"):
        record = tool_convergence_checkpoint_record(state)
        if record and record.get("status") != "failed" and record.get("quality_status") != "failed":
            return record
    primary_record = state.checkpoint.get("primary_tcad_record")
    if isinstance(primary_record, dict) and primary_record.get("state_path"):
        return primary_record
    if state.checkpoint.get("post_tool_convergence_index_refreshed"):
        record = tool_convergence_checkpoint_record(state)
        if record:
            return record
    if state.checkpoint.get("primary_supervisor_completed"):
        return None
    return latest_relevant_record(state.recent_records)


def repair_evidence_record(state: MissionState) -> dict[str, Any] | None:
    primary_record = state.checkpoint.get("primary_tcad_record")
    record = primary_record if isinstance(primary_record, dict) and primary_record.get("state_path") else current_evidence_record(state)
    if not isinstance(record, dict):
        return None
    benchmark_status = state.checkpoint.get("physical_benchmark_status")
    if benchmark_status in {"failed", "suspicious"}:
        patched = dict(record)
        patched["quality_status"] = benchmark_status
        patched["failure_reason"] = (
            state.checkpoint.get("physical_benchmark_failure_reason")
            or f"physical benchmark status is {benchmark_status}"
        )
        return patched
    return record


def severity_rank(value: str | None) -> int:
    return {"passed": 0, "completed": 0, "suspicious": 1, "failed": 2}.get(str(value or ""), 0)


def benchmark_augmented_repair_state_path(state: MissionState, source_state_path: str) -> str:
    benchmark_path = state.checkpoint.get("physical_benchmark_path")
    benchmark_status = state.checkpoint.get("physical_benchmark_status")
    if benchmark_status not in {"failed", "suspicious"} or not benchmark_path:
        return source_state_path
    source_data = read_json_safe(source_state_path)
    benchmark_data = read_json_safe(str(benchmark_path))
    if not source_data or not benchmark_data:
        return source_state_path
    quality = dict(source_data.get("quality_report") or {})
    existing_issues = [issue for issue in quality.get("issues") or [] if isinstance(issue, dict)]
    existing_codes = {str(issue.get("code")) for issue in existing_issues if issue.get("code")}
    for check in benchmark_data.get("checks") or []:
        if not isinstance(check, dict) or check.get("severity") not in {"warning", "error"}:
            continue
        code = str(check.get("code") or "")
        if not code or code in existing_codes:
            continue
        existing_issues.append(
            {
                "code": code,
                "severity": check.get("severity"),
                "message": check.get("message"),
                "evidence": check.get("observed") or {},
            }
        )
        existing_codes.add(code)
    current_quality_status = str(quality.get("status") or "passed")
    if severity_rank(str(benchmark_status)) > severity_rank(current_quality_status):
        quality["status"] = benchmark_status
    quality["issues"] = existing_issues
    quality["recommended_next_action"] = (
        (benchmark_data.get("summary") or {}).get("recommended_next_action_zh")
        or quality.get("recommended_next_action")
        or "先修复 benchmark 标记的物理/签核风险，再信任该结果"
    )
    source_data["quality_report"] = quality
    source_data["benchmark_context"] = {
        "status": benchmark_data.get("status"),
        "summary": benchmark_data.get("summary") or {},
        "benchmark_path": str(benchmark_path),
    }
    output = Path(state.mission_dir) / "repair_inputs" / f"{Path(source_state_path).stem}_benchmark_augmented.json"
    write_json(output, source_data)
    state.checkpoint["repair_augmented_state_path"] = str(output)
    return str(output)


def compact_quality_report(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    return {
        "status": report.get("status"),
        "issues": [
            {
                "code": issue.get("code"),
                "severity": issue.get("severity"),
                "message": issue.get("message"),
            }
            for issue in (report.get("issues") or [])[:8]
            if isinstance(issue, dict)
        ],
        "metrics": {
            key: value
            for key, value in (report.get("metrics") or {}).items()
            if key
            in {
                "relative_delta",
                "relative_tolerance",
                "leakage_abs_current_at_target_a",
                "breakdown_voltage_at_threshold_v",
                "ion_ioff_ratio",
                "subthreshold_swing_mv_dec",
                "vth_at_threshold_current_v",
                "idvd_final_current_a",
                "reverse_current_shape_violations",
            }
        },
        "recommended_next_action": report.get("recommended_next_action"),
    }


def state_digest_for_llm(path_value: str | Path | None) -> dict[str, Any]:
    data = read_json_safe(path_value)
    if not data:
        return {}
    request = data.get("request") if isinstance(data.get("request"), dict) else {}
    final_summary = data.get("final_summary") if isinstance(data.get("final_summary"), dict) else {}
    return {
        "path": str(path_value),
        "tool_name": data.get("tool_name"),
        "status": data.get("status"),
        "run_id": data.get("run_id") or data.get("convergence_id") or data.get("task_id"),
        "tcad_deck_spec": compact_tcad_deck_spec(data.get("tcad_deck_spec") or request.get("tcad_deck_spec")),
        "request_core": {
            key: request.get(key)
            for key in [
                "sweep_type",
                "start",
                "stop",
                "step",
                "gate_start",
                "gate_stop",
                "gate_step",
                "drain_start",
                "drain_stop",
                "drain_step",
                "axis_path",
                "values",
                "mobility_model",
                "recombination_model",
                "impact_ionization_model",
                "model_strategy",
            ]
            if request.get(key) is not None
        },
        "quality_report": compact_quality_report(data.get("quality_report")),
        "benchmark_context": data.get("benchmark_context"),
        "repair_context": data.get("repair_context"),
        "mutation_effect_analysis": data.get("mutation_effect_analysis"),
        "agent_reasoning": {
            "last_repair_agent_decision": (data.get("checkpoint") or {}).get("last_repair_agent_decision")
            if isinstance(data.get("checkpoint"), dict)
            else None,
            "repair_agent_policy": request.get("agent_policy"),
        },
        "final_metrics": {
            key: value
            for key, value in ((final_summary.get("metrics") or final_summary) if isinstance(final_summary, dict) else {}).items()
            if key
            in {
                "leakage_abs_current_at_target_a",
                "breakdown_voltage_at_threshold_v",
                "ion_ioff_ratio",
                "subthreshold_swing_mv_dec",
                "vth_at_threshold_current_v",
                "relative_delta",
            }
        },
    }


def append_soft_failure(state: MissionState, step: MissionStep, reason: str) -> None:
    failures = state.checkpoint.setdefault("soft_failures", [])
    if not isinstance(failures, list):
        failures = []
        state.checkpoint["soft_failures"] = failures
    failures.append(
        {
            "step_index": step.index,
            "step_kind": step.kind.value,
            "goal_step_index": step.request.get("goal_step_index"),
            "goal_step_kind": step.request.get("goal_step_kind"),
            "reason": reason,
            "created_at": utc_timestamp(),
        }
    )


def soft_failure_count(state: MissionState) -> int:
    failures = state.checkpoint.get("soft_failures") or []
    return len(failures) if isinstance(failures, list) else 0


def goal_status_counts(state: MissionState) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw_status in goal_step_statuses(state).values():
        status = raw_status.get("status") if isinstance(raw_status, dict) else raw_status
        key = str(status or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def blocked_goal_step_indexes(state: MissionState) -> list[int]:
    indexes: list[int] = []
    for raw_index, raw_status in goal_step_statuses(state).items():
        status = raw_status.get("status") if isinstance(raw_status, dict) else raw_status
        if status == "failed":
            try:
                indexes.append(int(raw_index))
            except (TypeError, ValueError):
                pass
    return sorted(indexes)


def controller_observation(state: MissionState, step: MissionStep) -> dict[str, Any]:
    """Record a compact Chinese observe/diagnose snapshot for long-running UI and resume."""
    primary = state.checkpoint.get("primary_tcad_record")
    return {
        "step_index": step.index,
        "step_kind": step.kind.value,
        "step_status": step.status.value,
        "step_error": step.error,
        "goal_step_index": step.request.get("goal_step_index"),
        "goal_step_kind": step.request.get("goal_step_kind"),
        "goal_status_counts": goal_status_counts(state),
        "soft_failure_count": soft_failure_count(state),
        "blocked_goal_steps": blocked_goal_step_indexes(state),
        "pending_goal_kinds": sorted(pending_goal_kinds(state)),
        "primary_tcad_record": primary if isinstance(primary, dict) else None,
        "tool_convergence": {
            "status": state.checkpoint.get("tool_convergence_status"),
            "quality_status": state.checkpoint.get("tool_convergence_quality_status"),
            "state_path": state.checkpoint.get("tool_convergence_state_path"),
        },
        "physical_benchmark": {
            "status": state.checkpoint.get("physical_benchmark_status"),
            "failure_reason": state.checkpoint.get("physical_benchmark_failure_reason"),
            "benchmark_path": state.checkpoint.get("physical_benchmark_path"),
        },
        "repair": {
            "status": state.checkpoint.get("repair_execution_status"),
            "quality_status": state.checkpoint.get("repaired_quality_status"),
            "failure_reason": state.checkpoint.get("repair_failure_reason"),
            "state_path": state.checkpoint.get("repaired_state_path"),
        },
    }


def controller_decision(state: MissionState, step: MissionStep, observation: dict[str, Any]) -> dict[str, Any]:
    snapshot = build_long_horizon_snapshot(state.goal_text, state.checkpoint, observation)
    policy = decide_long_horizon_action(snapshot)
    if state.status == MissionStatus.WAITING_FOR_USER:
        action = "ask_user"
        reason = "当前步骤需要用户确认或补充信息，暂停自动执行。"
    elif step.kind == MissionStepKind.GENERATE_CONCLUSION and step.status == MissionStepStatus.COMPLETED:
        action = "finish"
        reason = "工程结论已生成，本轮任务可以结束。"
    elif step.kind == MissionStepKind.NOOP:
        action = "finish"
        reason = "目标分解中的步骤都已进入终态。"
    elif step.status == MissionStepStatus.FAILED and should_agent_replan(state):
        action = "replan"
        reason = "工具链路失败，下一步调用总控重编排而不是直接终止。"
    elif should_agent_replan(state):
        action = "replan"
        reason = "存在软失败或阻塞步骤，下一步让 agent 重新诊断并调整计划。"
    elif observation.get("blocked_goal_steps"):
        action = "ask_user"
        reason = "仍有阻塞步骤，且自动重编排预算已不足，需要用户确认策略。"
    elif observation.get("soft_failure_count"):
        pending = set(observation.get("pending_goal_kinds") or [])
        if "generate_conclusion" in pending:
            action = "continue_with_risk"
            reason = "有非阻塞风险，但仍可继续生成带风险说明的工程结论。"
        else:
            action = "continue"
            reason = "软失败已记录，继续执行后续可用步骤。"
    else:
        action = "continue"
        reason = "当前步骤未发现阻塞风险，继续执行下一步。"
    if policy.action in {"replan", "repair_or_verify", "continue_with_risk", "ask_user"} and action in {"continue", "continue_with_risk"}:
        action = policy.action
        reason = policy.reason_zh
    existing_ledger = [item for item in (state.checkpoint.get("risk_ledger") or []) if isinstance(item, dict)]
    state.checkpoint["risk_ledger"] = merge_risk_ledger(existing_ledger, policy.risk_ledger_updates)
    policy_state = dict(state.checkpoint.get("long_horizon_policy") or {})
    policy_state.update(
        {
            "last_action": action,
            "last_policy_action": policy.action,
            "last_policy_reason_zh": policy.reason_zh,
            "last_risk_level": policy.risk_level,
            "missing_evidence": policy.missing_evidence,
            "required_evidence": policy.required_evidence,
            "budget": policy.budget,
            "updated_at": utc_timestamp(),
        }
    )
    state.checkpoint["long_horizon_policy"] = policy_state
    return {
        "action": action,
        "reason_zh": reason,
        "policy_action": policy.action,
        "policy_reason_zh": policy.reason_zh,
        "risk_level": policy.risk_level,
        "required_evidence": policy.required_evidence,
        "missing_evidence": policy.missing_evidence,
        "next_action": state.next_action,
        "replan_budget": {
            "attempts": int(state.checkpoint.get("agent_replan_attempts") or 0),
            "max_attempts": int(state.checkpoint.get("agent_replan_max_attempts") or 0),
        },
    }


def record_controller_cycle(state: MissionState, step: MissionStep) -> dict[str, Any]:
    observation = controller_observation(state, step)
    decision = controller_decision(state, step, observation)
    cycle = {
        "cycle": len(state.checkpoint.get("controller_cycles") or []) + 1,
        "created_at": utc_timestamp(),
        "role": "observe_diagnose_plan_act",
        "observation": observation,
        "decision": decision,
    }
    cycles = state.checkpoint.setdefault("controller_cycles", [])
    if not isinstance(cycles, list):
        cycles = []
        state.checkpoint["controller_cycles"] = cycles
    cycles.append(cycle)
    state.checkpoint["last_controller_decision"] = decision
    return cycle


def replan_issue_signature(state: MissionState) -> str | None:
    soft_failures = state.checkpoint.get("soft_failures") or []
    blocked = blocked_goal_step_indexes(state)
    if not soft_failures and not blocked:
        return None
    payload = {
        "soft_failure_count": len(soft_failures) if isinstance(soft_failures, list) else 0,
        "blocked_goal_steps": blocked,
        "last_step_error": (state.checkpoint.get("last_step") or {}).get("error"),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def should_agent_replan(state: MissionState) -> bool:
    if not state.checkpoint.get("agent_replan_enabled"):
        return False
    attempts = int(state.checkpoint.get("agent_replan_attempts") or 0)
    max_attempts = int(state.checkpoint.get("agent_replan_max_attempts") or 2)
    if attempts >= max_attempts:
        return False
    signature = replan_issue_signature(state)
    if not signature:
        return False
    return signature != state.checkpoint.get("last_replan_issue_signature")


def should_replan_before_ready_goal_step(state: MissionState) -> bool:
    if not should_agent_replan(state):
        return False
    soft_failures = state.checkpoint.get("soft_failures") or []
    if not isinstance(soft_failures, list) or not soft_failures:
        return False
    latest = soft_failures[-1]
    if not isinstance(latest, dict):
        return False
    return latest.get("goal_step_kind") == "run_tool_convergence"


def replan_issue_context(state: MissionState) -> dict[str, Any]:
    soft_failures = state.checkpoint.get("soft_failures") or []
    blocked = blocked_goal_step_indexes(state)
    primary_record = state.checkpoint.get("primary_tcad_record")
    primary_state_path = primary_record.get("state_path") if isinstance(primary_record, dict) else None
    current_record = repair_evidence_record(state)
    current_state_path = current_record.get("state_path") if isinstance(current_record, dict) else None
    benchmark_data = read_json_safe(state.checkpoint.get("physical_benchmark_path"))
    repair_data = read_json_safe(state.checkpoint.get("repair_execution_state_path"))
    return {
        "soft_failures": soft_failures[-5:] if isinstance(soft_failures, list) else [],
        "blocked_goal_steps": blocked,
        "last_step": state.checkpoint.get("last_step"),
        "primary_tcad_record": primary_record,
        "primary_evidence_digest": state_digest_for_llm(primary_state_path),
        "current_evidence_digest": state_digest_for_llm(current_state_path),
        "tool_convergence": {
            "status": state.checkpoint.get("tool_convergence_status"),
            "quality_status": state.checkpoint.get("tool_convergence_quality_status"),
            "state_path": state.checkpoint.get("tool_convergence_state_path"),
            "digest": state_digest_for_llm(state.checkpoint.get("tool_convergence_state_path")),
        },
        "repair": {
            "status": state.checkpoint.get("repair_execution_status"),
            "quality_status": state.checkpoint.get("repaired_quality_status"),
            "failure_reason": state.checkpoint.get("repair_failure_reason"),
            "attempts": [
                {
                    "index": attempt.get("index"),
                    "action_name": attempt.get("action_name"),
                    "status": attempt.get("status"),
                    "quality_status": attempt.get("quality_status"),
                    "failure_reason": attempt.get("failure_reason"),
                }
                for attempt in (repair_data.get("attempts") or [])[-5:]
                if isinstance(attempt, dict)
            ],
        },
        "physical_benchmark": {
            "status": state.checkpoint.get("physical_benchmark_status"),
            "benchmark_path": state.checkpoint.get("physical_benchmark_path"),
            "source_state_path": state.checkpoint.get("physical_benchmark_source_state_path"),
            "failure_reason": state.checkpoint.get("physical_benchmark_failure_reason"),
            "summary": benchmark_data.get("summary") if isinstance(benchmark_data, dict) else None,
            "checks": [
                {
                    "code": check.get("code"),
                    "severity": check.get("severity"),
                    "message": check.get("message"),
                }
                for check in (benchmark_data.get("checks") or [])[:8]
                if isinstance(check, dict) and check.get("severity") != "pass"
            ],
        },
    }


def pending_goal_kinds(state: MissionState) -> set[str]:
    finished = goal_step_indexes_by_status(state, {"completed", "skipped", "soft_failed", "failed", "waiting_for_user"})
    kinds: set[str] = set()
    for step in goal_steps(state):
        try:
            index = int(step.get("index"))
        except (TypeError, ValueError):
            continue
        if index not in finished and step.get("kind"):
            kinds.add(str(step["kind"]))
    return kinds


def append_replan_steps(state: MissionState, steps: list[GoalStep]) -> list[dict[str, Any]]:
    if not steps:
        return []
    decomposition = state.checkpoint.setdefault("goal_decomposition", {})
    existing = decomposition.setdefault("steps", [])
    if not isinstance(existing, list):
        existing = []
        decomposition["steps"] = existing
    existing_indexes = [int(step.get("index")) for step in existing if isinstance(step, dict) and str(step.get("index", "")).isdigit()]
    max_index = max(existing_indexes or [0])
    pending_kinds = pending_goal_kinds(state)
    appended: list[dict[str, Any]] = []
    original_to_new: dict[int, int] = {}
    candidate_steps: list[GoalStep] = []
    for step in steps:
        if step.kind.value in pending_kinds:
            continue
        candidate_steps.append(step)
        original_to_new[step.index] = max_index + len(candidate_steps)
    for step in candidate_steps:
        new_index = original_to_new[step.index]
        depends_on: list[int] = []
        for dep in step.depends_on:
            depends_on.append(original_to_new.get(dep, dep))
        if not depends_on and max_index:
            depends_on = [max_index]
        data = step.model_copy(update={"index": new_index, "depends_on": depends_on}).model_dump(mode="json")
        existing.append(data)
        appended.append(data)
    return appended


def apply_replan_decision(state: MissionState, step: MissionStep, decision: dict[str, Any]) -> dict[str, Any]:
    for index in decision.get("mark_soft_failed") or []:
        try:
            mark_goal_step(state, int(index), "soft_failed", result={"reason": "agent replanner marked this step non-blocking"})
        except (TypeError, ValueError):
            pass
    for index in decision.get("skip_goal_steps") or []:
        try:
            mark_goal_step(state, int(index), "skipped", result={"reason": "agent replanner skipped this step"})
        except (TypeError, ValueError):
            pass
    appended = append_replan_steps(
        state,
        [GoalStep.model_validate(item) for item in decision.get("append_steps") or [] if isinstance(item, dict)],
    )
    history = state.checkpoint.setdefault("agent_replans", [])
    if not isinstance(history, list):
        history = []
        state.checkpoint["agent_replans"] = history
    summary = {
        "step_index": step.index,
        "status": decision.get("status"),
        "model": decision.get("model"),
        "issue_family": decision.get("issue_family"),
        "control_action": decision.get("control_action"),
        "strategy_zh": decision.get("strategy_zh"),
        "recommended_actions": decision.get("recommended_actions") or [],
        "appended_steps": appended,
        "warnings": decision.get("warnings") or [],
        "created_at": utc_timestamp(),
    }
    history.append(summary)
    state.checkpoint["agent_replan_attempts"] = int(state.checkpoint.get("agent_replan_attempts") or 0) + 1
    signature = replan_issue_signature(state)
    if signature:
        state.checkpoint["last_replan_issue_signature"] = signature
    return summary


def create_initial_state(
    mission_id: str,
    goal_text: str,
    mission_dir: Path,
    execute: bool,
    max_cycles: int,
    supervisor_max_cycles: int,
    use_llm_decomposer: bool,
    allow_llm_fallback: bool,
    llm_client: ChatClient | None = None,
    defer_decomposition: bool = False,
) -> MissionState:
    now = utc_timestamp()
    intent = parse_engineering_intent(goal_text)
    replan_budget = 4 if intent.risk_level == "high" else 3
    checkpoint: dict[str, Any] = {
        "completed_cycles": 0,
        "goal_decomposer": "llm" if use_llm_decomposer else "deterministic",
        "goal_decomposition_status": "running" if defer_decomposition else "planned",
        "agent_replan_enabled": True,
        "agent_replan_attempts": 0,
        "agent_replan_max_attempts": replan_budget,
        "engineering_intent": intent.model_dump(mode="json"),
        "long_horizon_policy": {
            "mode": "autonomous",
            "planner": "agent_first_llm_with_deterministic_fallback" if use_llm_decomposer else "deterministic_fallback",
            "risk_level": intent.risk_level,
            "max_replan_attempts": replan_budget,
            "max_repair_rounds": 4 if intent.risk_level == "high" else 3,
            "required_evidence": intent.evidence_requirements,
            "intent_summary_zh": intent.summary_zh,
            "agent_first_policy": {
                "mission_planner": use_llm_decomposer,
                "repair_executor": True,
                "deterministic_fallback": allow_llm_fallback,
            },
        },
        "risk_ledger": [],
    }
    if defer_decomposition:
        return MissionState(
            status=MissionStatus.RUNNING if execute else MissionStatus.PLANNED,
            mission_id=mission_id,
            goal_text=goal_text,
            mission_dir=str(mission_dir),
            created_at=now,
            updated_at=now,
            execute=execute,
            max_cycles=max_cycles,
            supervisor_max_cycles=supervisor_max_cycles,
            checkpoint=checkpoint,
            next_action="decompose goal",
        )
    decomposition = (
        decompose_goal_with_llm(
            goal_text,
            plan_id=mission_id,
            client=llm_client,
            allow_fallback=allow_llm_fallback,
        )
        if use_llm_decomposer
        else deterministic_decompose_goal(goal_text, plan_id=mission_id)
    )
    failed_decomposition = decomposition.status == "failed"
    return MissionState(
        status=MissionStatus.FAILED if failed_decomposition else MissionStatus.RUNNING if execute else MissionStatus.PLANNED,
        mission_id=mission_id,
        goal_text=goal_text,
        mission_dir=str(mission_dir),
        created_at=now,
        updated_at=now,
        execute=execute,
        max_cycles=max_cycles,
        supervisor_max_cycles=supervisor_max_cycles,
        checkpoint=apply_goal_decomposition_checkpoint(checkpoint, decomposition),
        failure_reason="goal decomposition failed" if failed_decomposition else None,
        next_action="refresh experiment memory",
    )


def run_goal_decomposition(
    goal_text: str,
    *,
    mission_id: str,
    use_llm_decomposer: bool,
    allow_llm_fallback: bool,
    llm_client: ChatClient | None = None,
):
    return (
        decompose_goal_with_llm(
            goal_text,
            plan_id=mission_id,
            client=llm_client,
            allow_fallback=allow_llm_fallback,
        )
        if use_llm_decomposer
        else deterministic_decompose_goal(goal_text, plan_id=mission_id)
    )


def apply_goal_decomposition_checkpoint(checkpoint: dict[str, Any], decomposition: Any) -> dict[str, Any]:
    return {
        **checkpoint,
        "goal_decomposition": decomposition.model_dump(mode="json"),
        "goal_decomposition_status": decomposition.status,
        "goal_decomposer_fallback_used": decomposition.fallback_used,
        "goal_decomposer_model": decomposition.model,
    }


def common_step_fields(state: MissionState) -> dict[str, Any]:
    now = utc_timestamp()
    return {
        "index": len(state.steps) + 1,
        "status": MissionStepStatus.PLANNED,
        "created_at": now,
        "updated_at": now,
    }


def choose_next_step(state: MissionState) -> MissionStep:
    common = common_step_fields(state)
    if not state.last_index_summary:
        return MissionStep(
            **common,
            kind=MissionStepKind.REBUILD_INDEX,
            reason="refresh global experiment memory before mission planning",
            request={"root": str(PROJECT_ROOT / "runs"), "db_path": str(default_index_db_path())},
        )

    if state.checkpoint.get("primary_supervisor_completed") and not state.checkpoint.get("post_primary_index_refreshed"):
        return MissionStep(
            **common,
            kind=MissionStepKind.REBUILD_INDEX,
            reason="refresh experiment memory after the primary TCAD action",
            request={"root": str(PROJECT_ROOT / "runs"), "db_path": str(default_index_db_path())},
        )

    if state.checkpoint.get("tool_convergence_completed") and not state.checkpoint.get("post_tool_convergence_index_refreshed"):
        return MissionStep(
            **common,
            kind=MissionStepKind.REBUILD_INDEX,
            reason="refresh experiment memory after tool convergence",
            request={"root": str(PROJECT_ROOT / "runs"), "db_path": str(default_index_db_path())},
        )

    if state.checkpoint.get("golden_comparison_completed") and not state.checkpoint.get("post_golden_comparison_index_refreshed"):
        return MissionStep(
            **common,
            kind=MissionStepKind.REBUILD_INDEX,
            reason="refresh experiment memory after golden/measured curve comparison",
            request={"root": str(PROJECT_ROOT / "runs"), "db_path": str(default_index_db_path())},
        )

    if should_replan_before_ready_goal_step(state):
        return MissionStep(
            **common,
            kind=MissionStepKind.REPLAN,
            reason="diagnose execution issues and adapt the mission plan",
            request={"issue_context": replan_issue_context(state)},
        )

    goal_step = next_ready_goal_step(state)
    if goal_step:
        return mission_step_for_goal_step(state, goal_step, common)

    if should_agent_replan(state):
        return MissionStep(
            **common,
            kind=MissionStepKind.REPLAN,
            reason="diagnose execution issues and adapt the mission plan",
            request={"issue_context": replan_issue_context(state)},
        )

    blocking = goal_step_indexes_by_status(state, {"failed", "waiting_for_user"})
    if blocking:
        return MissionStep(
            **common,
            kind=MissionStepKind.ASK_USER,
            reason="one or more goal-decomposition steps are blocked",
            request={"question": "长期计划中有步骤失败或等待确认，请检查 mission checkpoint 后决定是否继续。"},
        )

    return MissionStep(
        **common,
        kind=MissionStepKind.NOOP,
        reason="all goal-decomposition steps have reached a terminal state",
    )


def mission_step_for_goal_step(
    state: MissionState,
    goal_step: dict[str, Any],
    common: dict[str, Any],
) -> MissionStep:
    step_kind = str(goal_step.get("kind") or "")
    request = goal_step_request_fields(goal_step)
    goal_request = goal_step.get("request") if isinstance(goal_step.get("request"), dict) else {}

    if step_kind == "query_history":
        request.update({"limit": int(goal_request.get("limit") or 20)})
        return MissionStep(
            **common,
            kind=MissionStepKind.QUERY_HISTORY,
            reason="execute goal-decomposition history query step",
            request=request,
        )

    if step_kind == "run_supervisor":
        request.update(
            {
                "goal_text": goal_request.get("goal_text") or state.goal_text,
                "supervisor_id": goal_request.get("supervisor_id") or f"{state.mission_id}_supervisor",
                "supervisor_root": goal_request.get("supervisor_root") or str(Path(state.mission_dir) / "supervisor"),
                "max_cycles": int(goal_request.get("max_cycles") or state.supervisor_max_cycles),
            }
        )
        return MissionStep(
            **common,
            kind=MissionStepKind.RUN_SUPERVISOR,
            reason="execute goal-decomposition supervisor step",
            request=request,
        )

    if step_kind == "run_tool_convergence":
        primary_record = state.checkpoint.get("primary_tcad_record")
        if state.checkpoint.get("primary_supervisor_completed") and not isinstance(primary_record, dict):
            request["question"] = "主 TCAD 步骤没有产出当前 state.json，不能用默认 convergence 替代本轮仿真结果。"
            return MissionStep(
                **common,
                kind=MissionStepKind.ASK_USER,
                reason="goal-decomposition tool convergence step has no primary TCAD result",
                request=request,
            )
        if isinstance(primary_record, dict) and record_needs_repair(primary_record):
            request["skip_reason"] = "primary TCAD result is suspicious or failed; run benchmark/repair before convergence"
            request["target_state"] = primary_record.get("state_path")
            return MissionStep(
                **common,
                kind=MissionStepKind.SKIP_GOAL_STEP,
                reason="skip automatic convergence until the primary TCAD result is repaired",
                request=request,
            )
        convergence_request = dict(goal_request)
        convergence_request.setdefault("convergence_id", f"{state.mission_id}_tool_convergence")
        convergence_request.setdefault("convergence_root", str(Path(state.mission_dir) / "tool_convergence"))
        convergence_request.setdefault("overwrite", True)
        convergence_request["execute"] = state.execute
        request["convergence_request"] = convergence_request
        return MissionStep(
            **common,
            kind=MissionStepKind.RUN_TOOL_CONVERGENCE,
            reason="execute goal-decomposition tool convergence study before accepting TCAD evidence",
            request=request,
        )

    if step_kind == "run_golden_comparison":
        primary_record = state.checkpoint.get("primary_tcad_record")
        if not isinstance(primary_record, dict) or not primary_record.get("state_path"):
            request["question"] = "没有本轮 primary TCAD state.json，不能执行 golden/measured 曲线对比。"
            return MissionStep(
                **common,
                kind=MissionStepKind.ASK_USER,
                reason="golden/measured comparison step has no primary TCAD result",
                request=request,
            )
        reference = goal_request.get("reference_curve_path")
        if not reference:
            request["question"] = "golden/measured 曲线对比缺少 reference_curve_path。"
            return MissionStep(
                **common,
                kind=MissionStepKind.ASK_USER,
                reason="golden/measured comparison step has no reference curve",
                request=request,
            )
        comparison_request = {
            "comparison_id": goal_request.get("comparison_id") or f"{state.mission_id}_golden_comparison",
            "source_state_path": primary_record["state_path"],
            "reference_curve_path": reference,
            "run_root": str(Path(state.mission_dir) / "golden_curve_comparison"),
        }
        request["comparison_request"] = comparison_request
        return MissionStep(
            **common,
            kind=MissionStepKind.RUN_GOLDEN_COMPARISON,
            reason="compare primary TCAD curve against golden/measured reference before benchmark",
            request=request,
        )

    if step_kind == "run_physical_benchmark":
        target = current_evidence_record(state)
        if not target and not state.checkpoint.get("repaired_state_path"):
            request["question"] = "没有检索到可用于物理 benchmark 的 TCAD state.json。"
            return MissionStep(
                **common,
                kind=MissionStepKind.ASK_USER,
                reason="goal-decomposition physical benchmark step has no TCAD result",
                request=request,
            )
        benchmark_state = state.checkpoint.get("repaired_state_path") or (target or {})["state_path"]
        request["state"] = goal_request.get("state") or benchmark_state
        return MissionStep(
            **common,
            kind=MissionStepKind.RUN_PHYSICAL_BENCHMARK,
            reason="run physical benchmark before accepting TCAD evidence",
            request=request,
        )

    if step_kind == "run_repair_executor":
        target = repair_evidence_record(state)
        if not target:
            request["question"] = "没有检索到可用的 TCAD 结果，请确认要分析哪个 state.json。"
            return MissionStep(
                **common,
                kind=MissionStepKind.ASK_USER,
                reason="goal-decomposition repair step has no TCAD result to inspect",
                request=request,
            )

        if not record_needs_repair(target):
            request["skip_reason"] = "latest TCAD result is already accepted by status/quality checks"
            request["target_state"] = target["state_path"]
            return MissionStep(
                **common,
                kind=MissionStepKind.SKIP_GOAL_STEP,
                reason="skip repair step because the latest TCAD result is accepted",
                request=request,
            )

        if not state.checkpoint.get("repair_plan_path"):
            request["state"] = benchmark_augmented_repair_state_path(state, target["state_path"])
            request["original_state"] = target["state_path"]
            return MissionStep(
                **common,
                kind=MissionStepKind.GENERATE_REPAIR_PLAN,
                reason="latest TCAD result is failed or physically suspicious",
                request=request,
            )

        if state.checkpoint.get("repair_requires_confirmation"):
            request["question"] = "修复计划包含需要确认的单位/几何/参数修改，请确认后再继续执行。"
            return MissionStep(
                **common,
                kind=MissionStepKind.ASK_USER,
                reason="repair plan contains actions requiring user confirmation",
                request=request,
            )

        if not state.checkpoint.get("repair_execution_state_path"):
            repair_source_state = state.checkpoint.get("repair_source_state_path") or benchmark_augmented_repair_state_path(
                state,
                target["state_path"],
            )
            request.update(
                {
                    "state": repair_source_state,
                    "original_state": target["state_path"],
                    "execution_id": goal_request.get("execution_id") or f"{state.mission_id}_repair",
                    "execution_root": goal_request.get("execution_root")
                    or str(Path(state.mission_dir) / "repair_execution"),
                    "max_rounds": int(goal_request.get("max_rounds") or 3),
                }
            )
            return MissionStep(
                **common,
                kind=MissionStepKind.EXECUTE_REPAIR,
                reason="execute the highest-priority TCAD repair action and re-evaluate quality",
                request=request,
            )

        if state.checkpoint.get("repair_execution_status") in {"failed", "waiting_for_user"}:
            request["question"] = "自动修复没有得到可信结果，请确认是否扩大修复预算、允许敏感修复，或指定新的策略。"
            return MissionStep(
                **common,
                kind=MissionStepKind.ASK_USER,
                reason="automatic repair did not produce an accepted result",
                request=request,
            )

        request["target_state"] = state.checkpoint.get("repaired_state_path") or target["state_path"]
        return MissionStep(
            **common,
            kind=MissionStepKind.SKIP_GOAL_STEP,
            reason="repair step is already satisfied by existing repair checkpoint",
            request=request,
        )

    if step_kind == "generate_conclusion":
        target = current_evidence_record(state)
        if not target and not state.checkpoint.get("repaired_state_path"):
            request["question"] = "没有检索到可用于生成结论的 TCAD state.json。"
            return MissionStep(
                **common,
                kind=MissionStepKind.ASK_USER,
                reason="goal-decomposition conclusion step has no TCAD result",
                request=request,
            )
        conclusion_state = state.checkpoint.get("repaired_state_path") or (target or {})["state_path"]
        request["state"] = goal_request.get("state") or conclusion_state
        return MissionStep(
            **common,
            kind=MissionStepKind.GENERATE_CONCLUSION,
            reason="execute goal-decomposition engineering conclusion step",
            request=request,
        )

    if step_kind == "ask_user":
        request["question"] = goal_request.get("question") or goal_request.get("message") or "请确认下一步。"
        return MissionStep(
            **common,
            kind=MissionStepKind.ASK_USER,
            reason="execute goal-decomposition user-confirmation step",
            request=request,
        )

    return MissionStep(
        **common,
        kind=MissionStepKind.SKIP_GOAL_STEP,
        reason=f"unsupported goal-decomposition step kind {step_kind} was skipped",
        request=request,
    )


def execute_step(step: MissionStep, state: MissionState, *, llm_client: ChatClient | None = None) -> MissionStep:
    step.status = MissionStepStatus.RUNNING
    step.updated_at = utc_timestamp()
    goal_step_index = int(step.request.get("goal_step_index") or 0)
    goal_step_kind = step.request.get("goal_step_kind")
    if goal_step_index:
        mark_goal_step(state, goal_step_index, "running", kind=str(goal_step_kind), mission_step=step)
    try:
        if step.kind == MissionStepKind.REBUILD_INDEX:
            root = Path(step.request.get("root") or PROJECT_ROOT / "runs")
            db_path = Path(step.request.get("db_path") or default_index_db_path())
            result = rebuild_index(root, db_path)
            state.last_index_summary = result
            state.recent_records = list_records(db_path, limit=30)
            step.result = {"index": result, "recent_records": state.recent_records[:5]}
            if state.checkpoint.get("tool_convergence_completed") and not state.checkpoint.get(
                "post_tool_convergence_index_refreshed"
            ):
                state.checkpoint["post_tool_convergence_index_refreshed"] = True
            elif state.checkpoint.get("golden_comparison_completed") and not state.checkpoint.get(
                "post_golden_comparison_index_refreshed"
            ):
                state.checkpoint["post_golden_comparison_index_refreshed"] = True
            elif state.checkpoint.get("primary_supervisor_completed"):
                state.checkpoint["post_primary_index_refreshed"] = True
        elif step.kind == MissionStepKind.QUERY_HISTORY:
            step.result = {
                "records": list_records(
                    default_index_db_path(),
                    limit=int(step.request.get("limit") or 20),
                )
            }
            if goal_step_index:
                mark_goal_step(
                    state,
                    goal_step_index,
                    "completed",
                    kind=str(goal_step_kind),
                    mission_step=step,
                    result={"records": len(step.result["records"])},
                )
        elif step.kind == MissionStepKind.RUN_SUPERVISOR:
            supervisor_state = run_supervisor(
                step.request["goal_text"],
                supervisor_id=step.request["supervisor_id"],
                supervisor_root=Path(step.request["supervisor_root"]),
                execute=state.execute,
                max_cycles=int(step.request.get("max_cycles") or state.supervisor_max_cycles),
                use_agent_policy=bool(step.request.get("use_agent_policy", True)),
                llm_client=llm_client,
            )
            result = supervisor_state.model_dump(mode="json")
            step.result = result
            state.checkpoint["primary_supervisor_completed"] = True
            state.checkpoint["primary_supervisor_state_path"] = str(
                Path(result["supervisor_dir"]) / "supervisor_state.json"
            )
            primary_record = supervisor_tcad_record(result)
            if primary_record:
                state.checkpoint["primary_tcad_record"] = primary_record
            if goal_step_index:
                if primary_record and record_needs_repair(primary_record):
                    goal_status = "failed" if step.request.get("goal_step_stop_on_failure", True) else "soft_failed"
                    if goal_status == "soft_failed":
                        append_soft_failure(
                            state,
                            step,
                            primary_record.get("failure_reason")
                            or f"primary TCAD result needs repair: status={primary_record.get('status')} quality={primary_record.get('quality_status')}",
                        )
                else:
                    goal_status = "completed"
                mark_goal_step(
                    state,
                    goal_step_index,
                    goal_status,
                    kind=str(goal_step_kind),
                    mission_step=step,
                    result={
                        "status": result.get("status"),
                        "primary_status": (primary_record or {}).get("status"),
                        "primary_quality_status": (primary_record or {}).get("quality_status"),
                        "primary_failure_reason": (primary_record or {}).get("failure_reason"),
                        "state_path": state.checkpoint["primary_supervisor_state_path"],
                    },
                )
        elif step.kind == MissionStepKind.RUN_TOOL_CONVERGENCE:
            request = ToolConvergenceRequest.model_validate(step.request["convergence_request"])
            convergence = run_tool_convergence(request)
            result = convergence.model_dump(mode="json")
            step.result = result
            state.checkpoint["tool_convergence_completed"] = True
            state.checkpoint["tool_convergence_state_path"] = str(
                Path(result["convergence_dir"]) / "state.json"
            )
            state.checkpoint["tool_convergence_id"] = result.get("convergence_id")
            state.checkpoint["tool_convergence_status"] = result.get("status")
            state.checkpoint["tool_convergence_quality_status"] = (result.get("quality_report") or {}).get("status")
            if goal_step_index:
                quality_status = state.checkpoint.get("tool_convergence_quality_status")
                if result.get("status") == "failed" or quality_status == "failed":
                    goal_status = "failed" if step.request.get("goal_step_stop_on_failure", True) else "soft_failed"
                    if goal_status == "soft_failed":
                        append_soft_failure(
                            state,
                            step,
                            (result.get("quality_report") or {}).get("recommended_next_action")
                            or result.get("failure_reason")
                            or "tool convergence did not pass",
                        )
                else:
                    goal_status = "completed"
                mark_goal_step(
                    state,
                    goal_step_index,
                    goal_status,
                    kind=str(goal_step_kind),
                    mission_step=step,
                    result={
                        "status": result.get("status"),
                        "quality_status": quality_status,
                        "state_path": state.checkpoint["tool_convergence_state_path"],
                    },
                    )
        elif step.kind == MissionStepKind.RUN_GOLDEN_COMPARISON:
            request = GoldenCurveComparisonRequest.model_validate(step.request["comparison_request"])
            comparison = run_golden_curve_comparison(request)
            result = comparison.model_dump(mode="json")
            step.result = result
            state.checkpoint["golden_comparison_completed"] = True
            state.checkpoint["golden_comparison_state_path"] = str(
                Path(result["comparison_dir"]) / "state.json"
            )
            state.checkpoint["golden_comparison_id"] = result.get("comparison_id")
            state.checkpoint["golden_comparison_status"] = result.get("status")
            state.checkpoint["golden_comparison_quality_status"] = (result.get("quality_report") or {}).get("status")
            if goal_step_index:
                quality_status = state.checkpoint.get("golden_comparison_quality_status")
                terminal_status = "completed" if quality_status == "passed" else "soft_failed"
                mark_goal_step(
                    state,
                    goal_step_index,
                    terminal_status,
                    kind=str(goal_step_kind),
                    mission_step=step,
                    result={
                        "state_path": state.checkpoint["golden_comparison_state_path"],
                        "quality_status": quality_status,
                    },
                )
                if terminal_status == "soft_failed":
                    append_soft_failure(state, step, (result.get("failure_reason") or "golden/measured comparison did not pass"))
        elif step.kind == MissionStepKind.RUN_PHYSICAL_BENCHMARK:
            benchmark = run_physical_benchmark(Path(step.request["state"]))
            result = benchmark.model_dump(mode="json")
            step.result = result
            state.checkpoint["physical_benchmark_status"] = result.get("status")
            state.checkpoint["physical_benchmark_path"] = result.get("benchmark_path")
            state.checkpoint["physical_benchmark_source_state_path"] = result.get("source_state_path")
            if result.get("failure_reason"):
                state.checkpoint["physical_benchmark_failure_reason"] = result.get("failure_reason")
            if goal_step_index:
                status = str(result.get("status") or "unsupported")
                if status in {"failed", "suspicious"}:
                    goal_status = "failed" if step.request.get("goal_step_stop_on_failure", True) else "soft_failed"
                    if goal_status == "soft_failed":
                        append_soft_failure(
                            state,
                            step,
                            result.get("failure_reason") or f"physical benchmark status is {status}",
                        )
                else:
                    goal_status = "completed"
                mark_goal_step(
                    state,
                    goal_step_index,
                    goal_status,
                    kind=str(goal_step_kind),
                    mission_step=step,
                    result={
                        "status": status,
                        "benchmark_path": result.get("benchmark_path"),
                        "source_state_path": result.get("source_state_path"),
                    },
                )
        elif step.kind == MissionStepKind.REPLAN:
            decision = replan_goal_after_issue(
                state.goal_text,
                current_plan=state.checkpoint.get("goal_decomposition") or {},
                goal_step_statuses=goal_step_statuses(state),
                issue_context=step.request.get("issue_context") or replan_issue_context(state),
                current_evidence=repair_evidence_record(state),
                plan_id=state.mission_id,
                client=llm_client,
                allow_fallback=True,
            )
            decision_payload = decision.model_dump(mode="json")
            summary = apply_replan_decision(state, step, decision_payload)
            step.result = {
                "status": decision_payload.get("status"),
                "controller_role": "llm_observe_diagnose_plan_act_verify",
                "control_action": decision_payload.get("control_action"),
                "strategy_zh": decision_payload.get("strategy_zh"),
                "model": decision_payload.get("model"),
                "fallback_used": decision_payload.get("fallback_used"),
                "append_steps": decision_payload.get("append_steps"),
                "mark_soft_failed": decision_payload.get("mark_soft_failed"),
                "skip_goal_steps": decision_payload.get("skip_goal_steps"),
                "warnings": decision_payload.get("warnings"),
                "summary": summary,
            }
        elif step.kind == MissionStepKind.GENERATE_REPAIR_PLAN:
            repair = build_repair_plan(Path(step.request["state"]))
            result = repair.model_dump(mode="json")
            step.result = result
            state.checkpoint["repair_plan_path"] = result.get("output_path")
            state.checkpoint["repair_source_state_path"] = step.request["state"]
            if step.request.get("original_state"):
                state.checkpoint["repair_original_state_path"] = step.request.get("original_state")
            state.checkpoint["repair_next_action"] = result.get("next_action")
            state.checkpoint["repair_requires_confirmation"] = any(
                action.get("user_confirmation_required") for action in result.get("actions") or []
            )
        elif step.kind == MissionStepKind.EXECUTE_REPAIR:
            execution = run_repair_executor(
                Path(step.request["state"]),
                execution_id=step.request.get("execution_id"),
                execution_root=Path(step.request["execution_root"]) if step.request.get("execution_root") else None,
                execute=state.execute,
                max_rounds=int(step.request.get("max_rounds") or 3),
                use_agent_policy=bool(step.request.get("use_agent_policy", True)),
                llm_client=llm_client,
            )
            result = execution.model_dump(mode="json")
            step.result = result
            state.checkpoint["repair_execution_state_path"] = str(
                Path(result["execution_dir"]) / "repair_execution_state.json"
            )
            state.checkpoint["repair_execution_status"] = result.get("status")
            state.checkpoint["repaired_state_path"] = result.get("final_state_path") or result.get("current_state_path")
            state.checkpoint["repaired_quality_status"] = result.get("final_quality_status")
            if result.get("status") == "waiting_for_user":
                state.checkpoint["repair_requires_confirmation"] = True
            if result.get("status") == "failed":
                state.checkpoint["repair_failure_reason"] = result.get("failure_reason")
            if goal_step_index:
                repair_status = result.get("status")
                if repair_status == "completed":
                    goal_status = "completed"
                elif repair_status == "failed":
                    goal_status = "failed" if step.request.get("goal_step_stop_on_failure", True) else "soft_failed"
                    if goal_status == "soft_failed":
                        append_soft_failure(
                            state,
                            step,
                            result.get("failure_reason") or "automatic repair did not produce an accepted result",
                        )
                elif repair_status == "waiting_for_user":
                    goal_status = "waiting_for_user"
                else:
                    goal_status = "completed"
                mark_goal_step(
                    state,
                    goal_step_index,
                    goal_status,
                    kind=str(goal_step_kind),
                    mission_step=step,
                    result={
                        "status": repair_status,
                        "state_path": state.checkpoint.get("repaired_state_path"),
                        "quality_status": state.checkpoint.get("repaired_quality_status"),
                    },
                )
        elif step.kind == MissionStepKind.GENERATE_CONCLUSION:
            conclusion = generate_experiment_conclusion(Path(step.request["state"]))
            result = conclusion.model_dump(mode="json")
            step.result = result
            state.checkpoint["conclusion_path"] = result.get("conclusion_path")
            if goal_step_index:
                mark_goal_step(
                    state,
                    goal_step_index,
                    "completed",
                    kind=str(goal_step_kind),
                    mission_step=step,
                    result={"status": result.get("status"), "conclusion_path": result.get("conclusion_path")},
                )
        elif step.kind == MissionStepKind.ASK_USER:
            step.result = {"question": step.request.get("question")}
            state.status = MissionStatus.WAITING_FOR_USER
            if goal_step_index:
                mark_goal_step(state, goal_step_index, "waiting_for_user", kind=str(goal_step_kind), mission_step=step)
        elif step.kind == MissionStepKind.SKIP_GOAL_STEP:
            step.result = {"status": "skipped", "reason": step.request.get("skip_reason") or step.reason}
            if goal_step_index:
                mark_goal_step(
                    state,
                    goal_step_index,
                    "skipped",
                    kind=str(goal_step_kind),
                    mission_step=step,
                    result=step.result,
                )
        else:
            step.result = {"message": "no operation"}
        step.status = MissionStepStatus.COMPLETED
    except Exception as exc:
        step.status = MissionStepStatus.FAILED
        step.error = str(exc)
        stop_on_failure = bool(step.request.get("goal_step_stop_on_failure", True))
        if stop_on_failure:
            if state.checkpoint.get("agent_replan_enabled") and int(state.checkpoint.get("agent_replan_attempts") or 0) < int(
                state.checkpoint.get("agent_replan_max_attempts") or 2
            ):
                state.status = MissionStatus.RUNNING
                state.failure_reason = None
            else:
                state.status = MissionStatus.FAILED
                state.failure_reason = str(exc)
            goal_status = "failed"
        else:
            goal_status = "soft_failed"
            append_soft_failure(state, step, str(exc))
        if goal_step_index:
            mark_goal_step(
                state,
                goal_step_index,
                goal_status,
                kind=str(goal_step_kind),
                mission_step=step,
                result={"error": str(exc)},
            )
    step.updated_at = utc_timestamp()
    return step


def run_mission_agent(
    goal_text: str,
    *,
    mission_id: str | None = None,
    mission_root: Path | None = None,
    execute: bool = False,
    resume: bool = False,
    max_cycles: int = 8,
    supervisor_max_cycles: int = 3,
    use_llm_decomposer: bool = True,
    allow_llm_fallback: bool = True,
    llm_client: ChatClient | None = None,
) -> MissionState:
    actual_mission_id = mission_id or default_mission_id()
    actual_root = mission_root or PROJECT_ROOT / "runs" / "missions"
    actual_state_path = state_path(actual_root, actual_mission_id)
    mission_dir = actual_root / actual_mission_id

    if resume and actual_state_path.exists():
        state = load_mission_state(actual_state_path)
        state.execute = execute
        state.max_cycles = max_cycles
        state.supervisor_max_cycles = supervisor_max_cycles
        state.status = MissionStatus.RUNNING if execute else MissionStatus.PLANNED
    else:
        mission_dir.mkdir(parents=True, exist_ok=True)
        state = create_initial_state(
            actual_mission_id,
            goal_text,
            mission_dir,
            execute,
            max_cycles,
            supervisor_max_cycles,
            use_llm_decomposer,
            allow_llm_fallback,
            llm_client,
            defer_decomposition=True,
        )
        write_mission_state(state, actual_state_path)
        decomposition = run_goal_decomposition(
            goal_text,
            mission_id=actual_mission_id,
            use_llm_decomposer=use_llm_decomposer,
            allow_llm_fallback=allow_llm_fallback,
            llm_client=llm_client,
        )
        state.checkpoint = apply_goal_decomposition_checkpoint(state.checkpoint, decomposition)
        state.status = (
            MissionStatus.FAILED
            if decomposition.status == "failed"
            else MissionStatus.RUNNING
            if execute
            else MissionStatus.PLANNED
        )
        state.failure_reason = "goal decomposition failed" if decomposition.status == "failed" else None
        state.next_action = "refresh experiment memory"
    write_mission_state(state, actual_state_path)

    while state.completed_cycles < state.max_cycles and state.status in {MissionStatus.RUNNING, MissionStatus.PLANNED}:
        step = choose_next_step(state)
        state.steps.append(step)
        state.next_action = step.kind.value
        write_mission_state(state, actual_state_path)

        if not state.execute:
            state.status = MissionStatus.PLANNED
            state.checkpoint["planned_step"] = step.model_dump(mode="json")
            write_mission_state(state, actual_state_path)
            return state

        step = execute_step(step, state, llm_client=llm_client)
        state.steps[-1] = step
        if step.status in {MissionStepStatus.COMPLETED, MissionStepStatus.FAILED, MissionStepStatus.SKIPPED} and step.kind != MissionStepKind.ASK_USER:
            state.completed_cycles += 1
        state.checkpoint["completed_cycles"] = state.completed_cycles
        state.checkpoint["last_step"] = step.model_dump(mode="json")
        record_controller_cycle(state, step)
        write_mission_state(state, actual_state_path)

        if step.kind == MissionStepKind.NOOP:
            state.status = MissionStatus.COMPLETED
            state.next_action = "mission completed"
            write_mission_state(state, actual_state_path)
            return state

        if step.kind == MissionStepKind.GENERATE_CONCLUSION and step.status == MissionStepStatus.COMPLETED:
            state.status = MissionStatus.COMPLETED
            state.next_action = "mission completed with conclusion"
            write_mission_state(state, actual_state_path)
            return state

    if state.status not in {MissionStatus.FAILED, MissionStatus.WAITING_FOR_USER}:
        state.status = MissionStatus.COMPLETED
        state.next_action = "maximum mission cycles reached"
    write_mission_state(state, actual_state_path)
    return state
