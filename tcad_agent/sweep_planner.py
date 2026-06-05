from __future__ import annotations

import json
import math
import re
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from tcad_agent.llm import LLMClient, LLMConfig
from tcad_agent.parameter_sweep import (
    ParameterSweepRequest,
    SweepAxis,
    SweepDirection,
    SweepObjective,
)
from tcad_agent.task_planner import (
    build_task_spec_from_planner_json,
    parse_json_object,
    plan_task_text_with_llm,
    task_spec_from_planning_result,
)
from tcad_agent.task_spec import TaskSpec, parse_task_text


NUMBER_RE = r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"


class ChatClient(Protocol):
    config: LLMConfig

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        ...


class SweepPlannerStatus(str, Enum):
    COMPLETED = "completed"
    FALLBACK = "fallback"
    FAILED = "failed"


class SweepPlanningResult(BaseModel):
    status: SweepPlannerStatus
    input_text: str
    sweep_id: str | None = None
    model: str | None = None
    raw_response: str | None = None
    parsed_response: dict[str, Any] | None = None
    base_task_spec: dict[str, Any] | None = None
    sweep_request: dict[str, Any] | None = None
    validation_errors: list[str] = Field(default_factory=list)
    repairs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    fallback_used: bool = False


def write_sweep_planning_result(result: SweepPlanningResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def extract_numbers(text: str) -> list[float]:
    return [float(match.group(0)) for match in re.finditer(NUMBER_RE, text, re.IGNORECASE)]


def geometric_values(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [start]
    if start > 0 and stop > 0:
        log_start = math.log10(start)
        log_stop = math.log10(stop)
        return [10 ** (log_start + (log_stop - log_start) * index / (count - 1)) for index in range(count)]
    step = (stop - start) / (count - 1)
    return [start + step * index for index in range(count)]


def normalize_numeric_values(values: list[float]) -> list[float]:
    normalized: list[float] = []
    for value in values:
        if value == 0:
            rounded = 0.0
        elif abs(value) >= 1e4 or abs(value) < 1e-3:
            rounded = float(f"{value:.12e}")
        else:
            rounded = float(f"{value:.12g}")
        if rounded not in normalized:
            normalized.append(rounded)
    return normalized


def infer_axis_path(text: str) -> str:
    lowered = text.lower()
    if re.search(r"n\s*区\s*掺杂|n\s*掺杂|n[-_ ]?doping|donor", lowered):
        return "parameters.n_doping_cm3"
    if re.search(r"温度|\btemperature\b|\btemp\b", lowered):
        return "parameters.temperature_k"
    if any(keyword in lowered for keyword in ["结位置", "junction"]):
        return "parameters.junction_um"
    if re.search(r"p\s*区\s*掺杂|p\s*掺杂|p[-_ ]?doping|acceptor", lowered):
        return "parameters.p_doping_cm3"
    return "parameters.p_doping_cm3"


def infer_axis_values(text: str, default_count: int = 3) -> list[float]:
    numbers = extract_numbers(text)
    if len(numbers) >= 3:
        return normalize_numeric_values(numbers[: min(len(numbers), 12)])
    if len(numbers) >= 2:
        return normalize_numeric_values(geometric_values(numbers[0], numbers[1], default_count))
    if len(numbers) == 1:
        value = numbers[0]
        return normalize_numeric_values([value / 10.0, value, value * 10.0])
    return [1.0e16, 1.0e17, 1.0e18]


def infer_axis_from_text(text: str) -> SweepAxis:
    axis_path = infer_axis_path(text)
    axis_region = text
    match = re.search(
        r"(?:扫描|sweep|scan)(.+?)(?:，|。|;|；|找|目标|$)",
        text,
        re.IGNORECASE,
    )
    if match:
        axis_region = match.group(1)
    count_match = re.search(r"(\d+)\s*(?:个点|点|points)", text, re.IGNORECASE)
    count = int(count_match.group(1)) if count_match else 3
    return SweepAxis(path=axis_path, values=infer_axis_values(axis_region, default_count=count))


def remove_axis_clause(text: str) -> str:
    cleaned = re.sub(
        r"(?:扫描|sweep|scan).+?(?:，|。|;|；|找|目标|$)",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" ，。;；") or text


def infer_objective(text: str) -> SweepObjective:
    lowered = text.lower()
    metric_path = "final_quality_report.metrics.final_total_current_a"
    if "最大电流" in text or "max_abs_current" in lowered:
        metric_path = "final_quality_report.metrics.max_abs_current_a"
    direction = SweepDirection.MAXIMIZE if any(keyword in lowered for keyword in ["最大", "maximize", "max "]) else SweepDirection.MINIMIZE
    if "最小" in text or "minimize" in lowered or "min " in lowered:
        direction = SweepDirection.MINIMIZE
    return SweepObjective(metric_path=metric_path, direction=direction, absolute=True)


def deterministic_sweep_plan(
    text: str,
    *,
    sweep_id: str | None = None,
    task_id: str | None = None,
    execution_use_llm: bool | None = None,
    execute: bool = False,
    max_cases: int = 100,
) -> tuple[TaskSpec, ParameterSweepRequest, list[str]]:
    actual_task_id = task_id or (f"{sweep_id}_base" if sweep_id else None)
    base_spec = parse_task_text(remove_axis_clause(text), task_id=actual_task_id, use_llm=execution_use_llm)
    axis = infer_axis_from_text(text)
    warnings: list[str] = []
    request = ParameterSweepRequest(
        sweep_id=sweep_id,
        axes=[axis],
        objective=infer_objective(text),
        execute=execute,
        use_llm=execution_use_llm,
        max_cases=max_cases,
    )
    return base_spec, request, warnings


def extract_sweep_candidate(parsed: dict[str, Any]) -> dict[str, Any]:
    for key in ("sweep_request", "parameter_sweep", "sweep"):
        value = parsed.get(key)
        if isinstance(value, dict):
            return value
    return parsed


def normalize_axis(candidate: dict[str, Any], repairs: list[str]) -> dict[str, Any]:
    normalized = dict(candidate)
    path_aliases = {
        "p_doping": "parameters.p_doping_cm3",
        "p_doping_cm3": "parameters.p_doping_cm3",
        "n_doping": "parameters.n_doping_cm3",
        "n_doping_cm3": "parameters.n_doping_cm3",
        "temperature": "parameters.temperature_k",
        "temperature_k": "parameters.temperature_k",
        "junction": "parameters.junction_um",
        "junction_um": "parameters.junction_um",
    }
    path = normalized.get("path") or normalized.get("parameter") or normalized.get("field")
    if isinstance(path, str):
        mapped = path_aliases.get(path, path)
        if mapped != path:
            repairs.append(f"Mapped axis path {path} to {mapped}.")
        normalized["path"] = mapped

    if "values" not in normalized:
        start = normalized.get("start")
        stop = normalized.get("stop")
        count = int(normalized.get("count") or normalized.get("points") or 3)
        if start is not None and stop is not None:
            normalized["values"] = normalize_numeric_values(geometric_values(float(start), float(stop), count))
            repairs.append("Expanded axis start/stop into values.")
    return normalized


def normalize_axes(raw_axes: Any, repairs: list[str]) -> list[dict[str, Any]]:
    if isinstance(raw_axes, dict):
        raw_axes = [raw_axes]
    if not isinstance(raw_axes, list):
        return []
    axes: list[dict[str, Any]] = []
    for raw_axis in raw_axes:
        if not isinstance(raw_axis, dict):
            repairs.append(f"Ignored non-object axis: {raw_axis!r}")
            continue
        axes.append(normalize_axis(raw_axis, repairs))
    return axes


def normalize_objective(raw_objective: Any, text: str, repairs: list[str]) -> dict[str, Any]:
    if not isinstance(raw_objective, dict):
        return infer_objective(text).model_dump(mode="json")
    objective = dict(raw_objective)
    direction = str(objective.get("direction", "minimize")).lower()
    if direction in {"min", "minimum", "minimize", "最小"}:
        objective["direction"] = "minimize"
    elif direction in {"max", "maximum", "maximize", "最大"}:
        objective["direction"] = "maximize"
    else:
        repairs.append(f"Unsupported objective direction {direction!r}; defaulted to minimize.")
        objective["direction"] = "minimize"
    metric = objective.get("metric") or objective.get("metric_path")
    if isinstance(metric, str):
        metric_aliases = {
            "final_current": "final_quality_report.metrics.final_total_current_a",
            "final_total_current": "final_quality_report.metrics.final_total_current_a",
            "max_abs_current": "final_quality_report.metrics.max_abs_current_a",
        }
        objective["metric_path"] = metric_aliases.get(metric, metric)
    else:
        objective["metric_path"] = infer_objective(text).metric_path
    objective["absolute"] = bool(objective.get("absolute", True))
    return objective


def build_sweep_request_from_planner_json(
    parsed: dict[str, Any],
    *,
    text: str,
    sweep_id: str | None = None,
    execute: bool = False,
    execution_use_llm: bool | None = None,
    max_cases: int = 100,
) -> tuple[ParameterSweepRequest, list[str]]:
    repairs: list[str] = []
    candidate = extract_sweep_candidate(parsed)
    axes = normalize_axes(candidate.get("axes") or candidate.get("axis"), repairs)
    if not axes:
        axes = [infer_axis_from_text(text).model_dump(mode="json")]
        repairs.append("No supported axes found; inferred one deterministic axis.")
    objective = normalize_objective(candidate.get("objective"), text, repairs)
    request_data = {
        "sweep_id": sweep_id or candidate.get("sweep_id"),
        "axes": axes,
        "objective": objective,
        "execute": execute,
        "use_llm": execution_use_llm,
        "max_cases": max_cases,
    }
    return ParameterSweepRequest.model_validate(request_data), repairs


def build_messages(text: str, base_spec: TaskSpec, baseline_request: ParameterSweepRequest) -> tuple[str, str]:
    system = (
        "你是 TCAD 参数 sweep 规划器。"
        "请把用户需求转换成 JSON。"
        "当前这个旧版入口唯一可执行的仿真任务是 DEVSIM PN junction IV sweep。"
        "sweep 轴必须修改 TaskSpec 路径，例如 parameters.p_doping_cm3、"
        "parameters.n_doping_cm3、parameters.temperature_k、parameters.junction_um、sweep.stop_v 或 mesh.junction_spacing_um。"
        "自然语言说明、warnings 和 assumptions 尽量使用中文。"
    )
    user = {
        "task": "把用户自然语言转换成 actsoft 参数 sweep v1",
        "required_response_schema": {
            "base_task_spec": "optional TaskSpec object",
            "sweep_request": {
                "sweep_id": "string or null",
                "axes": [
                    {
                        "path": "TaskSpec 路径，例如 parameters.p_doping_cm3",
                        "values": ["numbers or strings"],
                    }
                ],
                "objective": {
                    "metric_path": "e.g. final_quality_report.metrics.final_total_current_a",
                    "direction": "minimize | maximize",
                    "absolute": "boolean",
                },
            },
        },
        "baseline_base_task_spec": base_spec.model_dump(mode="json"),
        "baseline_sweep_request": baseline_request.model_dump(mode="json"),
        "user_text": text,
    }
    return system, json.dumps(user, ensure_ascii=False, indent=2)


def fallback_result(
    *,
    text: str,
    sweep_id: str | None,
    task_id: str | None,
    execution_use_llm: bool | None,
    execute: bool,
    max_cases: int,
    reason: str,
    raw_response: str | None = None,
    parsed_response: dict[str, Any] | None = None,
    validation_errors: list[str] | None = None,
    model: str | None = None,
) -> SweepPlanningResult:
    base_spec, request, warnings = deterministic_sweep_plan(
        text,
        sweep_id=sweep_id,
        task_id=task_id,
        execution_use_llm=execution_use_llm,
        execute=execute,
        max_cases=max_cases,
    )
    return SweepPlanningResult(
        status=SweepPlannerStatus.FALLBACK,
        input_text=text,
        sweep_id=request.sweep_id,
        model=model,
        raw_response=raw_response,
        parsed_response=parsed_response,
        base_task_spec=base_spec.model_dump(mode="json"),
        sweep_request=request.model_dump(mode="json"),
        validation_errors=validation_errors or [],
        warnings=[reason, *warnings],
        fallback_used=True,
    )


def plan_sweep_text_with_llm(
    text: str,
    *,
    sweep_id: str | None = None,
    task_id: str | None = None,
    execution_use_llm: bool | None = None,
    execute: bool = False,
    max_cases: int = 100,
    client: ChatClient | None = None,
    allow_fallback: bool = True,
) -> SweepPlanningResult:
    baseline_spec, baseline_request, baseline_warnings = deterministic_sweep_plan(
        text,
        sweep_id=sweep_id,
        task_id=task_id,
        execution_use_llm=execution_use_llm,
        execute=execute,
        max_cases=max_cases,
    )
    chat_client = client or LLMClient()
    system, user = build_messages(text, baseline_spec, baseline_request)
    try:
        raw_response = chat_client.chat(system=system, user=user, temperature=0.1)
    except Exception as exc:
        if allow_fallback:
            return fallback_result(
                text=text,
                sweep_id=sweep_id,
                task_id=task_id,
                execution_use_llm=execution_use_llm,
                execute=execute,
                max_cases=max_cases,
                reason=f"LLM sweep 规划器调用失败，已使用确定性解析器：{exc}",
                model=getattr(chat_client.config, "model", None),
            )
        return SweepPlanningResult(
            status=SweepPlannerStatus.FAILED,
            input_text=text,
            sweep_id=sweep_id,
            model=getattr(chat_client.config, "model", None),
            validation_errors=[str(exc)],
        )

    parsed = parse_json_object(raw_response)
    if parsed is None:
        if allow_fallback:
            return fallback_result(
                text=text,
                sweep_id=sweep_id,
                task_id=task_id,
                execution_use_llm=execution_use_llm,
                execute=execute,
                max_cases=max_cases,
                reason="LLM sweep 规划器响应中没有 JSON 对象，已使用确定性解析器。",
                raw_response=raw_response,
                model=chat_client.config.model,
            )
        return SweepPlanningResult(
            status=SweepPlannerStatus.FAILED,
            input_text=text,
            sweep_id=sweep_id,
            model=chat_client.config.model,
            raw_response=raw_response,
            validation_errors=["LLM sweep 规划器响应中没有 JSON 对象。"],
        )

    try:
        if isinstance(parsed.get("base_task_spec"), dict):
            base_spec, base_repairs = build_task_spec_from_planner_json(
                {"task_spec": parsed["base_task_spec"]},
                text=text,
                task_id=task_id or (f"{sweep_id}_base" if sweep_id else None),
                execution_use_llm=execution_use_llm,
            )
        else:
            task_result = plan_task_text_with_llm(
                text,
                task_id=task_id or (f"{sweep_id}_base" if sweep_id else None),
                execution_use_llm=execution_use_llm,
                client=client,
                allow_fallback=True,
            )
            base_spec = task_spec_from_planning_result(task_result)
            base_repairs = []
        request, repairs = build_sweep_request_from_planner_json(
            parsed,
            text=text,
            sweep_id=sweep_id,
            execute=execute,
            execution_use_llm=execution_use_llm,
            max_cases=max_cases,
        )
    except (ValidationError, ValueError, TypeError) as exc:
        errors = [str(exc)]
        if allow_fallback:
            return fallback_result(
                text=text,
                sweep_id=sweep_id,
                task_id=task_id,
                execution_use_llm=execution_use_llm,
                execute=execute,
                max_cases=max_cases,
                reason="LLM sweep 规划器 JSON 未通过校验，已使用确定性解析器。",
                raw_response=raw_response,
                parsed_response=parsed,
                validation_errors=errors,
                model=chat_client.config.model,
            )
        return SweepPlanningResult(
            status=SweepPlannerStatus.FAILED,
            input_text=text,
            sweep_id=sweep_id,
            model=chat_client.config.model,
            raw_response=raw_response,
            parsed_response=parsed,
            validation_errors=errors,
        )

    return SweepPlanningResult(
        status=SweepPlannerStatus.COMPLETED,
        input_text=text,
        sweep_id=request.sweep_id,
        model=chat_client.config.model,
        raw_response=raw_response,
        parsed_response=parsed,
        base_task_spec=base_spec.model_dump(mode="json"),
        sweep_request=request.model_dump(mode="json"),
        repairs=[*base_repairs, *repairs],
        warnings=[*baseline_warnings, *base_repairs, *repairs],
    )


def sweep_plan_from_result(result: SweepPlanningResult) -> tuple[TaskSpec, ParameterSweepRequest]:
    if not result.base_task_spec or not result.sweep_request:
        raise ValueError("Sweep planning result is missing base_task_spec or sweep_request.")
    return TaskSpec.model_validate(result.base_task_spec), ParameterSweepRequest.model_validate(result.sweep_request)
