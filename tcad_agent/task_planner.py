from __future__ import annotations

import argparse
import json
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from tcad_agent.llm import LLMClient, LLMConfig
from tcad_agent.task_spec import PROJECT_ROOT, TaskSpec, parse_task_text, write_task_spec


class ChatClient(Protocol):
    config: LLMConfig

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        ...


class PlannerStatus(str, Enum):
    COMPLETED = "completed"
    FALLBACK = "fallback"
    FAILED = "failed"


class TaskPlanningResult(BaseModel):
    status: PlannerStatus
    input_text: str
    task_id: str | None = None
    model: str | None = None
    raw_response: str | None = None
    parsed_response: dict[str, Any] | None = None
    task_spec: dict[str, Any] | None = None
    validation_errors: list[str] = Field(default_factory=list)
    repairs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    fallback_used: bool = False


def write_planning_result(result: TaskPlanningResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_task_candidate(parsed: dict[str, Any]) -> dict[str, Any]:
    for key in ("task_spec", "task", "spec"):
        value = parsed.get(key)
        if isinstance(value, dict):
            return value
    return parsed


def deep_merge_allowed(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    repairs: list[str],
    prefix: str = "",
) -> dict[str, Any]:
    merged = dict(baseline)
    for key, value in candidate.items():
        if key not in baseline:
            repairs.append(f"Ignored unsupported planner field: {prefix}{key}")
            continue
        if isinstance(value, dict) and isinstance(baseline.get(key), dict):
            merged[key] = deep_merge_allowed(
                baseline[key],
                value,
                repairs,
                prefix=f"{prefix}{key}.",
            )
        else:
            merged[key] = value
    return merged


def normalize_aliases(candidate: dict[str, Any], repairs: list[str]) -> dict[str, Any]:
    normalized = dict(candidate)

    if "device_parameters" in normalized and "parameters" not in normalized:
        normalized["parameters"] = normalized.pop("device_parameters")
        repairs.append("Mapped device_parameters to parameters.")

    geometry = normalized.pop("geometry", None)
    if isinstance(geometry, dict):
        parameters = dict(normalized.get("parameters") or {})
        for key in ("length_um", "junction_um", "temperature_k"):
            if key in geometry and key not in parameters:
                parameters[key] = geometry[key]
                repairs.append(f"Mapped geometry.{key} to parameters.{key}.")
        normalized["parameters"] = parameters

    doping = normalized.pop("doping", None)
    if isinstance(doping, dict):
        parameters = dict(normalized.get("parameters") or {})
        for old, new in {
            "p": "p_doping_cm3",
            "n": "n_doping_cm3",
            "p_cm3": "p_doping_cm3",
            "n_cm3": "n_doping_cm3",
            "p_doping": "p_doping_cm3",
            "n_doping": "n_doping_cm3",
        }.items():
            if old in doping and new not in parameters:
                parameters[new] = doping[old]
                repairs.append(f"Mapped doping.{old} to parameters.{new}.")
        normalized["parameters"] = parameters

    sweep = dict(normalized.get("sweep") or {})
    for old, new in {
        "start": "start_v",
        "stop": "stop_v",
        "step": "step_v",
        "min_step": "min_step_v",
    }.items():
        if old in sweep and new not in sweep:
            sweep[new] = sweep.pop(old)
            repairs.append(f"Mapped sweep.{old} to sweep.{new}.")
    if sweep:
        normalized["sweep"] = sweep

    quality = dict(normalized.get("quality") or {})
    for old, new in {
        "min_points": "min_points",
        "max_abs_current": "max_abs_current_a",
        "current_limit": "max_abs_current_a",
        "max_convergence_failures": "max_convergence_failures",
    }.items():
        if old in quality and new not in quality:
            quality[new] = quality.pop(old)
            if old != new:
                repairs.append(f"Mapped quality.{old} to quality.{new}.")
    if quality:
        normalized["quality"] = quality

    execution = dict(normalized.get("execution") or {})
    for old, new in {
        "attempts": "max_attempts",
        "cycles": "max_cycles",
        "timeout": "timeout_seconds",
    }.items():
        if old in execution and new not in execution:
            execution[new] = execution.pop(old)
            repairs.append(f"Mapped execution.{old} to execution.{new}.")
    if execution:
        normalized["execution"] = execution

    parameters = dict(normalized.get("parameters") or {})
    for old, new in {
        "length": "length_um",
        "junction": "junction_um",
        "junction_position": "junction_um",
        "p_doping": "p_doping_cm3",
        "n_doping": "n_doping_cm3",
        "temperature": "temperature_k",
        "taun": "electron_lifetime_s",
        "taup": "hole_lifetime_s",
    }.items():
        if old in parameters and new not in parameters:
            parameters[new] = parameters.pop(old)
            repairs.append(f"Mapped parameters.{old} to parameters.{new}.")
    if parameters:
        normalized["parameters"] = parameters

    mesh = dict(normalized.get("mesh") or {})
    for old, new in {
        "contact_spacing": "contact_spacing_um",
        "junction_spacing": "junction_spacing_um",
        "contact_mesh": "contact_spacing_um",
        "junction_mesh": "junction_spacing_um",
    }.items():
        if old in mesh and new not in mesh:
            mesh[new] = mesh.pop(old)
            repairs.append(f"Mapped mesh.{old} to mesh.{new}.")
    if mesh:
        normalized["mesh"] = mesh

    return normalized


def normalize_supported_values(candidate: dict[str, Any], repairs: list[str]) -> dict[str, Any]:
    normalized = dict(candidate)
    intent = str(normalized.get("intent", "")).lower()
    if intent in {"iv", "i-v", "simulate_iv", "iv_sweep", "simulate iv"}:
        normalized["intent"] = "simulate_iv"
    elif intent:
        repairs.append(f"Unsupported intent {normalized.get('intent')!r}; defaulted to simulate_iv.")
        normalized["intent"] = "simulate_iv"

    device = str(normalized.get("device", "")).lower()
    if device in {"pn", "p-n", "pn_junction", "junction", "diode", "二极管", "pn结"}:
        normalized["device"] = "pn_junction"
    elif device:
        repairs.append(f"Unsupported device {normalized.get('device')!r}; defaulted to pn_junction.")
        normalized["device"] = "pn_junction"

    simulator = str(normalized.get("simulator", "")).lower()
    if simulator in {"devsim", "open-source devsim"}:
        normalized["simulator"] = "devsim"
    elif simulator:
        repairs.append(f"Unsupported simulator {normalized.get('simulator')!r}; defaulted to devsim.")
        normalized["simulator"] = "devsim"

    return normalized


def normalize_numeric_policy(candidate: dict[str, Any], repairs: list[str]) -> dict[str, Any]:
    normalized = dict(candidate)
    sweep = dict(normalized.get("sweep") or {})
    if {"start_v", "stop_v"} <= set(sweep) and float(sweep["stop_v"]) < float(sweep["start_v"]):
        sweep["start_v"], sweep["stop_v"] = sweep["stop_v"], sweep["start_v"]
        repairs.append("Swapped sweep.start_v and sweep.stop_v because stop_v was smaller.")
    if {"step_v", "min_step_v"} <= set(sweep) and float(sweep["min_step_v"]) > float(sweep["step_v"]):
        adjusted = max(float(sweep["step_v"]) / 4.0, 1e-6)
        sweep["min_step_v"] = adjusted
        repairs.append("Adjusted sweep.min_step_v because it exceeded sweep.step_v.")
    if sweep:
        normalized["sweep"] = sweep
    parameters = dict(normalized.get("parameters") or {})
    if {"length_um", "junction_um"} <= set(parameters) and float(parameters["junction_um"]) >= float(parameters["length_um"]):
        parameters["junction_um"] = float(parameters["length_um"]) / 2.0
        repairs.append("Adjusted parameters.junction_um to half of length_um because it was outside the device.")
    if parameters:
        normalized["parameters"] = parameters
    return normalized


def requires_mission_agent(text: str) -> bool:
    lowered = text.lower()
    keywords = [
        "mosfet",
        "nmos",
        "pmos",
        "id-vg",
        "idvg",
        "id-vd",
        "idvd",
        "output characteristic",
        "transfer characteristic",
        "moscap",
        "mos c-v",
        "mos cv",
        "schottky",
        "power mos",
        "kink",
        "mos 管",
        "输出特性",
        "转移特性",
        "固定电荷",
        "肖特基",
    ]
    return any(keyword in lowered for keyword in keywords)


def unsupported_legacy_task_result(text: str, task_id: str | None, model: str | None = None) -> TaskPlanningResult:
    message = (
        "旧版 TaskSpec v1 只支持 PN junction IV sweep。"
        "MOSFET、MOS 电容、Schottky、功率器件、收敛验证或修复流程请使用 mission agent。"
    )
    return TaskPlanningResult(
        status=PlannerStatus.FAILED,
        input_text=text,
        task_id=task_id,
        model=model,
        validation_errors=[message],
        warnings=[message],
    )


def build_messages(text: str, baseline: TaskSpec) -> tuple[str, str]:
    system = (
        "你是自主 DEVSIM agent 的 TCAD 任务规划器。"
        "请把用户的半导体仿真需求转换成简洁 JSON。"
        "当前这个旧版入口唯一可执行任务是基于 DEVSIM 的 PN junction IV sweep。"
        "如果用户需求信息不足，请做保守假设并记录。"
        "不要包含 shell 命令；assumptions 和 warnings 尽量使用中文。"
    )
    user = {
        "task": "把用户自然语言转换成 actsoft TCAD TaskSpec v1",
        "supported_schema": {
            "schema_version": "actsoft.tcad.task.v1",
            "task_id": "string",
            "title": "中文短标题",
            "intent": "simulate_iv",
            "device": "pn_junction",
            "simulator": "devsim",
            "source": {"kind": "text", "text": "原始用户输入"},
            "sweep": {
                "variable": "anode_bias",
                "start_v": "float",
                "stop_v": "float",
                "step_v": "positive float",
                "min_step_v": "positive float <= step_v",
            },
            "parameters": {
                "length_um": "positive float, default 0.1",
                "junction_um": "positive float < length_um, default length_um / 2",
                "p_doping_cm3": "positive float",
                "n_doping_cm3": "positive float",
                "temperature_k": "positive float",
                "electron_lifetime_s": "positive float",
                "hole_lifetime_s": "positive float",
            },
            "mesh": {
                "contact_spacing_um": "positive float",
                "junction_spacing_um": "positive float",
            },
            "quality": {
                "min_points": "integer >= 1",
                "max_abs_current_a": "positive float",
                "max_convergence_failures": "integer >= 0",
            },
            "execution": {
                "max_attempts": "integer >= 1",
                "max_cycles": "integer >= 1",
                "timeout_seconds": "positive float",
                "use_llm": "boolean planner handoff flag for higher-level agents",
            },
            "outputs": ["iv_sweep.csv", "iv_curve.png", "summary.json"],
            "assumptions": ["中文假设"],
            "warnings": ["中文告警"],
        },
        "baseline_from_deterministic_parser": baseline.model_dump(mode="json"),
        "user_text": text,
    }
    return system, json.dumps(user, ensure_ascii=False, indent=2)


def build_task_spec_from_planner_json(
    parsed: dict[str, Any],
    *,
    text: str,
    task_id: str | None = None,
    execution_use_llm: bool | None = None,
) -> tuple[TaskSpec, list[str]]:
    repairs: list[str] = []
    baseline = parse_task_text(text, task_id=task_id, use_llm=execution_use_llm)
    candidate = extract_task_candidate(parsed)
    candidate = normalize_aliases(candidate, repairs)
    candidate = normalize_supported_values(candidate, repairs)
    merged = deep_merge_allowed(baseline.model_dump(mode="json"), candidate, repairs)

    merged["task_id"] = task_id or merged.get("task_id") or baseline.task_id
    merged["source"] = {"kind": "text", "text": text}
    if execution_use_llm is not None:
        execution = dict(merged.get("execution") or {})
        execution["use_llm"] = execution_use_llm
        merged["execution"] = execution

    merged = normalize_numeric_policy(merged, repairs)
    spec = TaskSpec.model_validate(merged)
    if repairs:
        existing = list(spec.warnings)
        spec.warnings = existing + repairs
    return spec, repairs


def fallback_result(
    *,
    text: str,
    task_id: str | None,
    execution_use_llm: bool | None,
    reason: str,
    raw_response: str | None = None,
    parsed_response: dict[str, Any] | None = None,
    validation_errors: list[str] | None = None,
    model: str | None = None,
) -> TaskPlanningResult:
    spec = parse_task_text(text, task_id=task_id, use_llm=execution_use_llm)
    warnings = [reason, *spec.assumptions, *spec.warnings]
    return TaskPlanningResult(
        status=PlannerStatus.FALLBACK,
        input_text=text,
        task_id=spec.task_id,
        model=model,
        raw_response=raw_response,
        parsed_response=parsed_response,
        task_spec=spec.model_dump(mode="json"),
        validation_errors=validation_errors or [],
        warnings=warnings,
        fallback_used=True,
    )


def plan_task_text_with_llm(
    text: str,
    *,
    task_id: str | None = None,
    execution_use_llm: bool | None = None,
    client: ChatClient | None = None,
    allow_fallback: bool = True,
) -> TaskPlanningResult:
    if requires_mission_agent(text):
        return unsupported_legacy_task_result(text, task_id)

    baseline = parse_task_text(text, task_id=task_id, use_llm=execution_use_llm)
    chat_client = client or LLMClient()
    system, user = build_messages(text, baseline)

    try:
        raw_response = chat_client.chat(system=system, user=user, temperature=0.1)
    except Exception as exc:
        if allow_fallback:
            return fallback_result(
                text=text,
                task_id=task_id,
                execution_use_llm=execution_use_llm,
                reason=f"LLM 规划器调用失败，已使用确定性解析器：{exc}",
                model=getattr(chat_client.config, "model", None),
            )
        return TaskPlanningResult(
            status=PlannerStatus.FAILED,
            input_text=text,
            task_id=task_id,
            model=getattr(chat_client.config, "model", None),
            validation_errors=[str(exc)],
        )

    parsed = parse_json_object(raw_response)
    if parsed is None:
        if allow_fallback:
            return fallback_result(
                text=text,
                task_id=task_id,
                execution_use_llm=execution_use_llm,
                reason="LLM 规划器响应中没有 JSON 对象，已使用确定性解析器。",
                raw_response=raw_response,
                model=chat_client.config.model,
            )
        return TaskPlanningResult(
            status=PlannerStatus.FAILED,
            input_text=text,
            task_id=task_id,
            model=chat_client.config.model,
            raw_response=raw_response,
            validation_errors=["LLM 规划器响应中没有 JSON 对象。"],
        )

    try:
        spec, repairs = build_task_spec_from_planner_json(
            parsed,
            text=text,
            task_id=task_id,
            execution_use_llm=execution_use_llm,
        )
    except (ValidationError, ValueError, TypeError) as exc:
        errors = [str(exc)]
        if allow_fallback:
            return fallback_result(
                text=text,
                task_id=task_id,
                execution_use_llm=execution_use_llm,
                reason="LLM 规划器 JSON 未通过 schema 校验，已使用确定性解析器。",
                raw_response=raw_response,
                parsed_response=parsed,
                validation_errors=errors,
                model=chat_client.config.model,
            )
        return TaskPlanningResult(
            status=PlannerStatus.FAILED,
            input_text=text,
            task_id=task_id,
            model=chat_client.config.model,
            raw_response=raw_response,
            parsed_response=parsed,
            validation_errors=errors,
        )

    return TaskPlanningResult(
        status=PlannerStatus.COMPLETED,
        input_text=text,
        task_id=spec.task_id,
        model=chat_client.config.model,
        raw_response=raw_response,
        parsed_response=parsed,
        task_spec=spec.model_dump(mode="json"),
        repairs=repairs,
        warnings=[*spec.assumptions, *spec.warnings],
    )


def task_spec_from_planning_result(result: TaskPlanningResult) -> TaskSpec:
    if not result.task_spec:
        raise ValueError("Planning result does not include a task_spec.")
    return TaskSpec.model_validate(result.task_spec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a TCAD TaskSpec from natural-language text.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--task-output", type=Path, default=None)
    parser.add_argument("--no-fallback", action="store_true")
    execution_llm = parser.add_mutually_exclusive_group()
    execution_llm.add_argument("--execution-use-llm", dest="execution_use_llm", action="store_true")
    execution_llm.add_argument("--execution-no-llm", dest="execution_use_llm", action="store_false")
    parser.set_defaults(execution_use_llm=None)
    return parser.parse_args()


def default_output_path(task_id: str | None) -> Path:
    stem = task_id or "planned_task"
    return PROJECT_ROOT / "runs" / "task_plans" / stem / "task_plan_result.json"


def default_task_output_path(task_id: str | None) -> Path:
    stem = task_id or "planned_task"
    return PROJECT_ROOT / "runs" / "task_plans" / stem / "task.json"


def main() -> None:
    args = parse_args()
    result = plan_task_text_with_llm(
        args.text,
        task_id=args.task_id,
        execution_use_llm=args.execution_use_llm,
        allow_fallback=not args.no_fallback,
    )
    output = args.output or default_output_path(result.task_id or args.task_id)
    write_planning_result(result, output)

    if result.task_spec:
        task_output = args.task_output or default_task_output_path(result.task_id or args.task_id)
        write_task_spec(task_spec_from_planning_result(result), task_output)

    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.status != PlannerStatus.FAILED else 1)


if __name__ == "__main__":
    main()
