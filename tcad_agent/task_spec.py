from __future__ import annotations

import json
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from tcad_agent.tools.autonomous_loop import AutonomousLoopRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NUMBER_RE = r"([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)"


class TaskIntent(str, Enum):
    SIMULATE_IV = "simulate_iv"


class DeviceKind(str, Enum):
    PN_JUNCTION = "pn_junction"


class SimulatorKind(str, Enum):
    DEVSIM = "devsim"


class TaskSource(BaseModel):
    kind: Literal["text", "json", "cli"] = "text"
    text: str | None = None


class BiasSweepSpec(BaseModel):
    variable: Literal["anode_bias"] = "anode_bias"
    start_v: float = 0.0
    stop_v: float = 0.5
    step_v: float = Field(default=0.1, gt=0.0)
    min_step_v: float = Field(default=0.0125, gt=0.0)

    @model_validator(mode="after")
    def validate_sweep(self) -> "BiasSweepSpec":
        if self.stop_v < self.start_v:
            raise ValueError("stop_v must be greater than or equal to start_v")
        if self.min_step_v > self.step_v:
            raise ValueError("min_step_v must be less than or equal to step_v")
        return self


class QualityPolicySpec(BaseModel):
    min_points: int = Field(default=3, ge=1)
    max_abs_current_a: float = Field(default=1.0, gt=0.0)
    max_convergence_failures: int = Field(default=0, ge=0)


class ExecutionPolicySpec(BaseModel):
    max_attempts: int = Field(default=5, ge=1)
    max_cycles: int = Field(default=3, ge=1)
    timeout_seconds: float = Field(default=300.0, gt=0.0)
    use_llm: bool = False
    force_llm: bool = False
    max_log_chars: int = Field(default=4000, ge=0)


class PNJunctionParametersSpec(BaseModel):
    length_um: float = Field(default=0.1, gt=0.0)
    junction_um: float = Field(default=0.05, gt=0.0)
    p_doping_cm3: float = Field(default=1.0e18, gt=0.0)
    n_doping_cm3: float = Field(default=1.0e18, gt=0.0)
    temperature_k: float = Field(default=300.0, gt=0.0)
    electron_lifetime_s: float = Field(default=1.0e-8, gt=0.0)
    hole_lifetime_s: float = Field(default=1.0e-8, gt=0.0)

    @model_validator(mode="after")
    def validate_geometry(self) -> "PNJunctionParametersSpec":
        if self.junction_um >= self.length_um:
            raise ValueError("junction_um must be less than length_um")
        return self


class MeshSpec(BaseModel):
    contact_spacing_um: float = Field(default=0.001, gt=0.0)
    junction_spacing_um: float = Field(default=1.0e-5, gt=0.0)


class TaskSpec(BaseModel):
    schema_version: Literal["actsoft.tcad.task.v1"] = "actsoft.tcad.task.v1"
    task_id: str
    title: str
    intent: TaskIntent = TaskIntent.SIMULATE_IV
    device: DeviceKind = DeviceKind.PN_JUNCTION
    simulator: SimulatorKind = SimulatorKind.DEVSIM
    source: TaskSource = Field(default_factory=TaskSource)
    sweep: BiasSweepSpec = Field(default_factory=BiasSweepSpec)
    parameters: PNJunctionParametersSpec = Field(default_factory=PNJunctionParametersSpec)
    mesh: MeshSpec = Field(default_factory=MeshSpec)
    quality: QualityPolicySpec = Field(default_factory=QualityPolicySpec)
    execution: ExecutionPolicySpec = Field(default_factory=ExecutionPolicySpec)
    outputs: list[str] = Field(default_factory=lambda: ["iv_sweep.csv", "iv_curve.png", "summary.json"])
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: utc_timestamp())


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_task_id(prefix: str = "task") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def load_task_spec(path: Path) -> TaskSpec:
    return TaskSpec.model_validate_json(path.read_text(encoding="utf-8"))


def write_task_spec(spec: TaskSpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(spec.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def first_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def first_number_after(labels: str, text: str) -> float | None:
    return first_float(rf"(?:{labels})\s*[:=]?\s*{NUMBER_RE}", text)


def first_length_um(labels: str, text: str) -> float | None:
    match = re.search(
        rf"(?:{labels})\s*[:=]?\s*{NUMBER_RE}\s*(nm|纳米|um|µm|μm|微米)?",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    value = float(match.group(1))
    unit = (match.group(2) or "um").lower()
    if unit in {"nm", "纳米"}:
        return value / 1000.0
    return value


def all_voltage_numbers(text: str) -> list[float]:
    normalized = text.replace("－", "-")
    values = []
    for match in re.finditer(r"([-+]?\d+(?:\.\d+)?)\s*(?:v|V|伏|伏特)", normalized):
        values.append(float(match.group(1)))
    return values


def infer_title(text: str) -> str:
    stripped = " ".join(text.strip().split())
    if not stripped:
        return "PN junction IV sweep"
    return stripped[:80]


def parse_task_text(
    text: str,
    task_id: str | None = None,
    use_llm: bool | None = None,
) -> TaskSpec:
    normalized = text.strip()
    lowered = normalized.lower()
    assumptions: list[str] = []
    warnings: list[str] = []

    if not any(keyword in lowered for keyword in ["pn", "p-n", "junction", "二极管", "结"]):
        assumptions.append("No supported device was explicit; defaulted to PN junction.")

    if not any(keyword in lowered for keyword in ["iv", "i-v", "电流", "伏安"]):
        assumptions.append("No explicit IV intent was detected; defaulted to IV sweep.")

    voltages = all_voltage_numbers(normalized)
    start_v = 0.0
    stop_v = 0.5

    explicit_start = first_float(
        r"(?:start|from|起始|从)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:v|伏|伏特)?",
        normalized,
    )
    explicit_stop = first_float(
        r"(?:stop|to|end|终止|结束|到|扫到)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:v|伏|伏特)?",
        normalized,
    )
    if explicit_start is not None:
        start_v = explicit_start
    if explicit_stop is not None:
        stop_v = explicit_stop
    if explicit_start is None and explicit_stop is None:
        if len(voltages) >= 2:
            start_v = voltages[0]
            stop_v = voltages[1]
        elif len(voltages) == 1:
            stop_v = voltages[0]
            assumptions.append("Only one voltage was detected; treated it as stop voltage with start_v=0.")
        else:
            assumptions.append("No voltage range was detected; defaulted to 0 V -> 0.5 V.")
    elif explicit_stop is None:
        candidates = [value for value in voltages if value != start_v]
        if candidates:
            stop_v = candidates[0]

    step_v = first_float(
        rf"(?:step|步长|间隔)\s*[:=]?\s*{NUMBER_RE}\s*(?:v|伏|伏特)?",
        normalized,
    )
    if step_v is None:
        span = max(stop_v - start_v, 0.0)
        step_v = min(max(span / 5.0, 1e-6), 0.1) if span else 0.1
        assumptions.append(f"No bias step was detected; defaulted to {step_v:g} V.")

    min_step_v = first_float(
        rf"(?:min[-_ ]?step|最小步长)\s*[:=]?\s*{NUMBER_RE}\s*(?:v|伏|伏特)?",
        normalized,
    )
    if min_step_v is None:
        min_step_v = min(step_v / 4.0, 0.0125)

    max_attempts = first_float(
        rf"(?:max[-_ ]?attempts|attempts|最大尝试次数|重试次数)\s*[:=]?\s*{NUMBER_RE}",
        normalized,
    )
    max_cycles = first_float(
        rf"(?:max[-_ ]?cycles|cycles|最大轮数|循环次数)\s*[:=]?\s*{NUMBER_RE}",
        normalized,
    )
    timeout_seconds = first_float(
        rf"(?:timeout|超时)\s*[:=]?\s*{NUMBER_RE}\s*(?:s|秒)?",
        normalized,
    )
    max_abs_current_a = first_float(
        rf"(?:max[-_ ]?abs[-_ ]?current|current[-_ ]?limit|电流阈值|最大电流)\s*[:=]?\s*{NUMBER_RE}\s*(?:a|安)?",
        normalized,
    )
    length_um = first_length_um(r"length|device[-_ ]?length|器件长度|总长|长度", normalized)
    junction_um = first_length_um(r"junction[-_ ]?(?:position|depth)|junction[-_ ]?um|结位置|结深", normalized)
    p_doping_cm3 = first_number_after(r"p[-_ ]?doping|p区掺杂|p掺杂|acceptor(?:s)?", normalized)
    n_doping_cm3 = first_number_after(r"n[-_ ]?doping|n区掺杂|n掺杂|donor(?:s)?", normalized)
    temperature_k = first_float(rf"(?:temperature|temp|温度)\s*[:=]?\s*{NUMBER_RE}\s*(?:k|K)?", normalized)
    electron_lifetime_s = first_float(
        rf"(?:electron[-_ ]?lifetime|taun|电子寿命)\s*[:=]?\s*{NUMBER_RE}\s*(?:s|秒)?",
        normalized,
    )
    hole_lifetime_s = first_float(
        rf"(?:hole[-_ ]?lifetime|taup|空穴寿命)\s*[:=]?\s*{NUMBER_RE}\s*(?:s|秒)?",
        normalized,
    )
    contact_spacing_um = first_length_um(
        r"contact[-_ ]?spacing|contact[-_ ]?mesh|接触网格|接触间距",
        normalized,
    )
    junction_spacing_um = first_length_um(
        r"junction[-_ ]?spacing|junction[-_ ]?mesh|结网格|结区网格|结间距",
        normalized,
    )

    if min_step_v > step_v:
        warnings.append("min_step_v was larger than step_v; adjusted to step_v / 4.")
        min_step_v = max(step_v / 4.0, 1e-6)

    execution = ExecutionPolicySpec(
        max_attempts=int(max_attempts) if max_attempts is not None else 5,
        max_cycles=int(max_cycles) if max_cycles is not None else 3,
        timeout_seconds=timeout_seconds if timeout_seconds is not None else 300.0,
        use_llm=bool(use_llm) if use_llm is not None else "llm" in lowered or "大模型" in normalized,
    )
    quality = QualityPolicySpec(
        max_abs_current_a=max_abs_current_a if max_abs_current_a is not None else 1.0,
    )
    parameters = PNJunctionParametersSpec(
        length_um=length_um if length_um is not None else 0.1,
        junction_um=junction_um if junction_um is not None else 0.05,
        p_doping_cm3=p_doping_cm3 if p_doping_cm3 is not None else 1.0e18,
        n_doping_cm3=n_doping_cm3 if n_doping_cm3 is not None else 1.0e18,
        temperature_k=temperature_k if temperature_k is not None else 300.0,
        electron_lifetime_s=electron_lifetime_s if electron_lifetime_s is not None else 1.0e-8,
        hole_lifetime_s=hole_lifetime_s if hole_lifetime_s is not None else 1.0e-8,
    )
    mesh = MeshSpec(
        contact_spacing_um=contact_spacing_um if contact_spacing_um is not None else 0.001,
        junction_spacing_um=junction_spacing_um if junction_spacing_um is not None else 1.0e-5,
    )

    return TaskSpec(
        task_id=task_id or default_task_id("pn_iv"),
        title=infer_title(normalized),
        source=TaskSource(kind="text", text=normalized),
        sweep=BiasSweepSpec(
            start_v=start_v,
            stop_v=stop_v,
            step_v=step_v,
            min_step_v=min_step_v,
        ),
        parameters=parameters,
        mesh=mesh,
        quality=quality,
        execution=execution,
        assumptions=assumptions,
        warnings=warnings,
    )


def task_spec_to_loop_request(
    spec: TaskSpec,
    *,
    loop_id: str | None = None,
    loop_root: Path | None = None,
    run_root: Path | None = None,
    resume: bool = False,
    use_llm: bool | None = None,
) -> AutonomousLoopRequest:
    if spec.intent != TaskIntent.SIMULATE_IV:
        raise ValueError(f"Unsupported task intent: {spec.intent}")
    if spec.device != DeviceKind.PN_JUNCTION:
        raise ValueError(f"Unsupported device: {spec.device}")
    if spec.simulator != SimulatorKind.DEVSIM:
        raise ValueError(f"Unsupported simulator: {spec.simulator}")

    return AutonomousLoopRequest(
        task="pn_junction_iv",
        start=spec.sweep.start_v,
        stop=spec.sweep.stop_v,
        step=spec.sweep.step_v,
        min_step=spec.sweep.min_step_v,
        max_attempts=spec.execution.max_attempts,
        timeout_seconds=spec.execution.timeout_seconds,
        quality_min_points=spec.quality.min_points,
        quality_max_abs_current_a=spec.quality.max_abs_current_a,
        quality_max_convergence_failures=spec.quality.max_convergence_failures,
        length_um=spec.parameters.length_um,
        junction_um=spec.parameters.junction_um,
        p_doping_cm3=spec.parameters.p_doping_cm3,
        n_doping_cm3=spec.parameters.n_doping_cm3,
        temperature_k=spec.parameters.temperature_k,
        electron_lifetime_s=spec.parameters.electron_lifetime_s,
        hole_lifetime_s=spec.parameters.hole_lifetime_s,
        contact_spacing_um=spec.mesh.contact_spacing_um,
        junction_spacing_um=spec.mesh.junction_spacing_um,
        max_cycles=spec.execution.max_cycles,
        use_llm=spec.execution.use_llm if use_llm is None else use_llm,
        force_llm=spec.execution.force_llm,
        max_log_chars=spec.execution.max_log_chars,
        loop_id=loop_id or spec.task_id,
        loop_root=loop_root or PROJECT_ROOT / "runs" / "autonomous_loop",
        run_root=run_root or PROJECT_ROOT / "runs" / "agent_tools",
        resume=resume,
    )
