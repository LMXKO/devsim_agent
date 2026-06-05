from __future__ import annotations

import json
import math
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, model_validator

from tcad_agent.schottky_calibration import SchottkyCalibrationRequest, run_schottky_calibration
from tcad_agent.task_spec import PROJECT_ROOT
from tcad_agent.tools.diode_breakdown import DiodeBreakdownRequest, run_diode_breakdown_sweep
from tcad_agent.tools.extended_device_sweep import ExtendedDeviceRequest, run_extended_device_sweep
from tcad_agent.tools.mos_capacitor_cv import MOSCapacitorCVRequest, run_mos_capacitor_cv_sweep
from tcad_agent.tools.mosfet_2d_id import MOSFET2DIDRequest, run_mosfet_2d_id_sweep
from tcad_agent.tools.pn_junction_iv import PNJunctionIVRequest, run_pn_junction_iv_sweep


Runner = Callable[[dict[str, Any]], dict[str, Any]]


class ToolConvergenceStatus(str, Enum):
    PLANNED = "planned"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolConvergenceQuality(str, Enum):
    PASSED = "passed"
    SUSPICIOUS = "suspicious"
    FAILED = "failed"
    PLANNED = "planned"


class ToolConvergenceRequest(BaseModel):
    convergence_id: str | None = None
    tool_name: str
    base_request: dict[str, Any]
    axis_path: str
    values: list[Any]
    metric_path: str = "quality_report.metrics.max_abs_current_a"
    relative_tolerance: float = Field(default=0.05, ge=0.0)
    execute: bool = False
    overwrite: bool = False
    convergence_root: Path = PROJECT_ROOT / "runs" / "tool_convergence"
    max_cases: int = Field(default=10, ge=2)

    @model_validator(mode="after")
    def validate_request(self) -> "ToolConvergenceRequest":
        if not self.axis_path:
            raise ValueError("axis_path is required")
        normalized = normalize_tool_convergence_payload(
            self.tool_name,
            self.base_request,
            self.axis_path,
            self.values,
            self.metric_path,
        )
        self.base_request = normalized["base_request"]
        self.axis_path = normalized["axis_path"]
        self.values = normalized["values"]
        self.metric_path = normalized["metric_path"]
        if len(self.values) < 2:
            raise ValueError("at least two convergence values are required")
        if len(self.values) > self.max_cases:
            raise ValueError(f"tool convergence would create {len(self.values)} cases, exceeding max_cases={self.max_cases}")
        return self


class ToolConvergenceState(BaseModel):
    tool_name: str = "tool_convergence"
    status: ToolConvergenceStatus
    convergence_id: str
    convergence_dir: str
    created_at: str
    updated_at: str
    execute: bool
    target_tool: str
    axis_path: str
    values: list[Any]
    metric_path: str
    relative_tolerance: float
    cases: list[dict[str, Any]] = Field(default_factory=list)
    quality_report: dict[str, Any] | None = None
    next_action: str | None = None
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_convergence_id() -> str:
    return f"toolconv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_state(state: ToolConvergenceState, path: Path) -> None:
    state.updated_at = utc_timestamp()
    write_json(path, state.model_dump(mode="json"))


def load_state(path: Path) -> ToolConvergenceState:
    return ToolConvergenceState.model_validate_json(path.read_text(encoding="utf-8"))


def result_to_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    raise TypeError(f"tool runner returned unsupported result type {type(result).__name__}")


def normalize_float_values(values: list[Any]) -> list[float]:
    normalized: list[float] = []
    for value in values:
        try:
            converted = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(converted):
            normalized.append(converted)
    return normalized


def normalize_mesh_division_values(values: list[Any]) -> list[int]:
    numeric = normalize_float_values(values)
    if not numeric:
        return [8, 12, 16]
    mapped: list[int] = []
    for value in numeric:
        if value < 4:
            candidate = int(4 + 4 * max(value, 1))
        else:
            candidate = int(round(value))
        mapped.append(max(candidate, 4))
    unique: list[int] = []
    for value in mapped:
        if value not in unique:
            unique.append(value)
    return unique if len(unique) >= 2 else [8, 12, 16]


def normalize_mosfet_convergence_payload(
    base_request: dict[str, Any],
    axis_path: str,
    values: list[Any],
    metric_path: str,
) -> dict[str, Any]:
    base = dict(base_request)
    sweep_aliases = {
        "output": "idvd",
        "output_characteristic": "idvd",
        "output_characteristics": "idvd",
        "output_curve": "idvd",
        "id_vd": "idvd",
        "id-vd": "idvd",
        "transfer": "idvg",
        "transfer_characteristic": "idvg",
        "transfer_characteristics": "idvg",
        "transfer_curve": "idvg",
        "id_vg": "idvg",
        "id-vg": "idvg",
        "all": "both",
        "both_curves": "both",
    }
    raw_sweep = str(base.get("sweep_type") or "").strip().lower().replace(" ", "_")
    if raw_sweep in sweep_aliases:
        base["sweep_type"] = sweep_aliases[raw_sweep]
    elif raw_sweep not in {"idvg", "idvd", "both"}:
        base["sweep_type"] = "idvd" if any(key in base for key in ["drain_start", "drain_stop", "drain_step"]) else "idvg"

    gate_values = base.pop("gate_values", None)
    if gate_values is not None and "idvd_gate_voltage" not in base:
        numeric_gate_values = normalize_float_values(gate_values if isinstance(gate_values, list) else [gate_values])
        if numeric_gate_values:
            base["idvd_gate_voltage"] = max(numeric_gate_values)
            base.setdefault("gate_stop", max(numeric_gate_values))
            base.setdefault("gate_start", min(numeric_gate_values))

    normalized_axis = axis_path
    normalized_values = list(values)
    if axis_path in {"mesh_refinement_level", "mesh_level", "mesh", "mesh_refinement"}:
        normalized_axis = "x_divisions"
    if normalized_axis in {"x_divisions", "silicon_y_divisions"}:
        normalized_values = normalize_mesh_division_values(values)

    metric_aliases = {
        "simulation_results.id_saturation": "quality_report.metrics.idvd_final_current_a",
        "simulation_results.saturation_current": "quality_report.metrics.idvd_final_current_a",
        "id_saturation": "quality_report.metrics.idvd_final_current_a",
        "saturation_current": "quality_report.metrics.idvd_final_current_a",
        "ion_ioff": "quality_report.metrics.ion_ioff_ratio",
        "ion_ioff_ratio": "quality_report.metrics.ion_ioff_ratio",
    }
    normalized_metric = metric_aliases.get(metric_path, metric_path)
    return {
        "base_request": base,
        "axis_path": normalized_axis,
        "values": normalized_values,
        "metric_path": normalized_metric,
    }


def normalize_tool_convergence_payload(
    tool_name: str,
    base_request: dict[str, Any],
    axis_path: str,
    values: list[Any],
    metric_path: str,
) -> dict[str, Any]:
    if tool_name == "mosfet_2d_id_sweep":
        return normalize_mosfet_convergence_payload(base_request, axis_path, values, metric_path)
    return {
        "base_request": dict(base_request),
        "axis_path": axis_path,
        "values": list(values),
        "metric_path": metric_path,
    }


def default_runner_registry() -> dict[str, Runner]:
    return {
        "pn_junction_iv_sweep": lambda request: result_to_dict(
            run_pn_junction_iv_sweep(PNJunctionIVRequest.model_validate(request))
        ),
        "mos_capacitor_cv_sweep": lambda request: result_to_dict(
            run_mos_capacitor_cv_sweep(MOSCapacitorCVRequest.model_validate(request))
        ),
        "diode_breakdown_leakage_sweep": lambda request: result_to_dict(
            run_diode_breakdown_sweep(DiodeBreakdownRequest.model_validate(request))
        ),
        "mosfet_2d_id_sweep": lambda request: result_to_dict(
            run_mosfet_2d_id_sweep(MOSFET2DIDRequest.model_validate(request))
        ),
        "extended_device_sweep": lambda request: result_to_dict(
            run_extended_device_sweep(ExtendedDeviceRequest.model_validate(request))
        ),
        "schottky_iv_calibration": lambda request: result_to_dict(
            run_schottky_calibration(SchottkyCalibrationRequest.model_validate(request))
        ),
    }


def value_at_path(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def set_value_at_path(data: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    updated = dict(data)
    current = updated
    parts = path.split(".")
    for part in parts[:-1]:
        nested = current.get(part)
        if not isinstance(nested, dict):
            nested = {}
        nested = dict(nested)
        current[part] = nested
        current = nested
    current[parts[-1]] = value
    return updated


def float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def relative_delta(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-300)


def build_case_request(request: ToolConvergenceRequest, value: Any, index: int, convergence_dir: Path) -> dict[str, Any]:
    case_request = set_value_at_path(request.base_request, request.axis_path, value)
    base_run_id = str(request.base_request.get("run_id") or request.convergence_id or "toolconv")
    case_request["run_id"] = f"{base_run_id}_case_{index:03d}"
    case_request["run_root"] = str(convergence_dir / "agent_tools")
    case_request["resume"] = False
    return case_request


def build_quality_report(state: ToolConvergenceState) -> dict[str, Any]:
    if not state.execute:
        return {
            "status": ToolConvergenceQuality.PLANNED,
            "issues": [],
            "metrics": {"cases": len(state.cases), "completed_cases": 0},
            "recommended_next_action": "执行已计划的工具收敛验证",
        }
    completed = [
        case
        for case in state.cases
        if case.get("status") == "completed" and case.get("metric_value") is not None
    ]
    issues: list[dict[str, Any]] = []
    if len(completed) < 2:
        issues.append(
            {
                "code": "too_few_completed_convergence_cases",
                "severity": "error",
                "message": "至少需要两个已完成且有指标值的工具收敛 case。",
                "evidence": {"completed_cases": len(completed)},
            }
        )
        return {
            "status": ToolConvergenceQuality.FAILED,
            "issues": issues,
            "metrics": {"cases": len(state.cases), "completed_cases": len(completed)},
            "recommended_next_action": "先重跑失败的收敛 case，再信任该结果",
        }

    left = completed[-2]
    right = completed[-1]
    left_metric = float(left["metric_value"])
    right_metric = float(right["metric_value"])
    delta = relative_delta(left_metric, right_metric)
    if delta > state.relative_tolerance:
        issues.append(
            {
                "code": "tool_not_converged",
                "severity": "warning",
                "message": "最后两个收敛 case 的指标变化超过容差。",
                "evidence": {
                    "relative_delta": delta,
                    "relative_tolerance": state.relative_tolerance,
                    "left_value": left.get("axis_value"),
                    "right_value": right.get("axis_value"),
                    "left_metric": left_metric,
                    "right_metric": right_metric,
                },
            }
        )
    failed_cases = [case for case in state.cases if case.get("status") == "failed"]
    if failed_cases:
        issues.append(
            {
                "code": "tool_convergence_case_failures",
                "severity": "warning",
                "message": "一个或多个收敛 case 失败。",
                "evidence": {"failed_cases": len(failed_cases)},
            }
        )
    quality = ToolConvergenceQuality.SUSPICIOUS if issues else ToolConvergenceQuality.PASSED
    return {
        "status": quality,
        "issues": issues,
        "metrics": {
            "cases": len(state.cases),
            "completed_cases": len(completed),
            "axis_path": state.axis_path,
            "metric_path": state.metric_path,
            "left_axis_value": left.get("axis_value"),
            "right_axis_value": right.get("axis_value"),
            "left_metric": left_metric,
            "right_metric": right_metric,
            "relative_delta": delta,
            "relative_tolerance": state.relative_tolerance,
        },
        "recommended_next_action": (
            "接受该工具/指标的收敛结果"
            if quality == ToolConvergenceQuality.PASSED
            else "扩展收敛取值、细化模型/网格，或检查失败 case"
        ),
    }


def run_tool_convergence(
    request: ToolConvergenceRequest,
    *,
    registry: dict[str, Runner] | None = None,
) -> ToolConvergenceState:
    convergence_id = request.convergence_id or default_convergence_id()
    convergence_dir = request.convergence_root / convergence_id
    state_path = convergence_dir / "state.json"
    if state_path.exists() and not request.overwrite:
        return load_state(state_path)

    convergence_dir.mkdir(parents=True, exist_ok=True)
    now = utc_timestamp()
    state = ToolConvergenceState(
        status=ToolConvergenceStatus.PLANNED,
        convergence_id=convergence_id,
        convergence_dir=str(convergence_dir),
        created_at=now,
        updated_at=now,
        execute=request.execute,
        target_tool=request.tool_name,
        axis_path=request.axis_path,
        values=request.values,
        metric_path=request.metric_path,
        relative_tolerance=request.relative_tolerance,
        next_action="execute tool convergence cases" if request.execute else "review planned convergence cases",
    )
    write_state(state, state_path)

    runners = registry or default_runner_registry()
    runner = runners.get(request.tool_name)
    if request.execute and runner is None:
        state.status = ToolConvergenceStatus.FAILED
        state.failure_reason = f"no runner registered for tool {request.tool_name}"
        state.quality_report = {
            "status": ToolConvergenceQuality.FAILED,
            "issues": [{"code": "unknown_tool", "severity": "error", "message": state.failure_reason}],
            "metrics": {},
            "recommended_next_action": "register a runner for this tool",
        }
        write_state(state, state_path)
        return state

    for index, value in enumerate(request.values, start=1):
        case_request = build_case_request(request, value, index, convergence_dir)
        case: dict[str, Any] = {
            "index": index,
            "axis_path": request.axis_path,
            "axis_value": value,
            "request": case_request,
            "status": "planned",
        }
        if request.execute and runner is not None:
            try:
                result = runner(case_request)
                case["result"] = result
                case["status"] = result.get("status") or "completed"
                case["quality_status"] = (result.get("quality_report") or {}).get("status")
                case["metric_value"] = float_or_none(value_at_path(result, request.metric_path))
                run_dir = result.get("run_dir")
                if run_dir:
                    candidate = Path(run_dir) / "state.json"
                    if candidate.exists():
                        case["state_path"] = str(candidate.resolve())
            except Exception as exc:
                case["status"] = "failed"
                case["failure_reason"] = str(exc)
        state.cases.append(case)
        write_state(state, state_path)

    state.quality_report = build_quality_report(state)
    state.status = (
        ToolConvergenceStatus.FAILED
        if state.quality_report["status"] == ToolConvergenceQuality.FAILED
        else ToolConvergenceStatus.COMPLETED
    )
    state.next_action = state.quality_report["recommended_next_action"]
    write_state(state, state_path)
    return state
