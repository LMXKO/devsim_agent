from __future__ import annotations

import argparse
import json
import shlex
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from tcad_agent.tools.llm_diagnose import is_allowed_next_tool_command
from tcad_agent.tools.pn_junction_iv import PNJunctionIVRequest, run_pn_junction_iv_sweep


class StrategyStatus(str, Enum):
    PLANNED = "planned"
    EXECUTED = "executed"
    SKIPPED = "skipped"
    FAILED = "failed"


class StrategyPlan(BaseModel):
    status: StrategyStatus
    state_path: str
    diagnosis_path: str | None = None
    output_path: str | None = None
    source_run_id: str | None = None
    execute: bool = False
    reason: str
    warnings: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    next_request: dict[str, Any] | None = None
    next_command: str | None = None
    executed_result: dict[str, Any] | None = None


ALLOWED_PN_FLAGS = {
    "--start": "start",
    "--stop": "stop",
    "--step": "step",
    "--min-step": "min_step",
    "--max-attempts": "max_attempts",
    "--timeout-seconds": "timeout_seconds",
    "--quality-min-points": "quality_min_points",
    "--quality-max-abs-current-a": "quality_max_abs_current_a",
    "--quality-max-convergence-failures": "quality_max_convergence_failures",
    "--length-um": "length_um",
    "--junction-um": "junction_um",
    "--p-doping-cm3": "p_doping_cm3",
    "--n-doping-cm3": "n_doping_cm3",
    "--temperature-k": "temperature_k",
    "--electron-lifetime-s": "electron_lifetime_s",
    "--hole-lifetime-s": "hole_lifetime_s",
    "--contact-spacing-um": "contact_spacing_um",
    "--junction-spacing-um": "junction_spacing_um",
    "--run-id": "run_id",
    "--run-root": "run_root",
}

FLOAT_FIELDS = {
    "start",
    "stop",
    "step",
    "min_step",
    "timeout_seconds",
    "quality_max_abs_current_a",
    "length_um",
    "junction_um",
    "p_doping_cm3",
    "n_doping_cm3",
    "temperature_k",
    "electron_lifetime_s",
    "hole_lifetime_s",
    "contact_spacing_um",
    "junction_spacing_um",
}

INT_FIELDS = {
    "max_attempts",
    "quality_min_points",
    "quality_max_convergence_failures",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def default_output_path(state_path: Path) -> Path:
    return state_path.parent / "strategy_plan.json"


def default_diagnosis_path(state_path: Path) -> Path:
    return state_path.parent / "llm_diagnosis.json"


def coerce_value(field: str, value: str) -> Any:
    if field in FLOAT_FIELDS:
        return float(value)
    if field in INT_FIELDS:
        return int(value)
    return value


def clean_request(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = set(PNJunctionIVRequest.model_fields.keys())
    cleaned = {key: value for key, value in raw.items() if key in allowed}
    cleaned.pop("resume", None)
    return cleaned


def normalize_followup_request(request: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    normalized = dict(request)
    step = float(normalized.get("step", 0.1))
    min_step = float(normalized.get("min_step", min(step, 0.0125)))
    if min_step > step:
        adjusted = max(step / 4.0, 1e-6)
        normalized["min_step"] = adjusted
        warnings.append(
            f"Adjusted min_step from {min_step:g} to {adjusted:g} because it exceeded step {step:g}."
        )
    length_um = float(normalized.get("length_um", 0.1))
    junction_um = float(normalized.get("junction_um", 0.05))
    if junction_um >= length_um:
        adjusted_junction = length_um / 2.0
        normalized["junction_um"] = adjusted_junction
        warnings.append(
            f"Adjusted junction_um from {junction_um:g} to {adjusted_junction:g} because it exceeded length_um {length_um:g}."
        )
    return normalized


def issue_codes(state: dict[str, Any]) -> set[str]:
    quality_report = state.get("quality_report") or {}
    return {issue.get("code") for issue in quality_report.get("issues", [])}


def next_followup_run_id(state: dict[str, Any], run_root: Path) -> str:
    source = state.get("run_id") or "run"
    root = run_root / "pn_junction_iv"
    for index in range(1, 1000):
        candidate = f"{source}_followup_{index:03d}"
        if not (root / candidate).exists():
            return candidate
    raise RuntimeError("Could not allocate a follow-up run id")


def command_from_request(request: PNJunctionIVRequest) -> str:
    return (
        "python3.11 -m tcad_agent.tools.pn_junction_iv "
        f"--run-id {shlex.quote(request.run_id or '')} "
        f"--start {request.start:g} "
        f"--stop {request.stop:g} "
        f"--step {request.step:g} "
        f"--min-step {request.min_step:g} "
        f"--max-attempts {request.max_attempts} "
        f"--timeout-seconds {request.timeout_seconds:g} "
        f"--quality-min-points {request.quality_min_points} "
        f"--quality-max-abs-current-a {request.quality_max_abs_current_a:g} "
        f"--quality-max-convergence-failures {request.quality_max_convergence_failures} "
        f"--length-um {request.length_um:g} "
        f"--junction-um {request.junction_um:g} "
        f"--p-doping-cm3 {request.p_doping_cm3:g} "
        f"--n-doping-cm3 {request.n_doping_cm3:g} "
        f"--temperature-k {request.temperature_k:g} "
        f"--electron-lifetime-s {request.electron_lifetime_s:g} "
        f"--hole-lifetime-s {request.hole_lifetime_s:g} "
        f"--contact-spacing-um {request.contact_spacing_um:g} "
        f"--junction-spacing-um {request.junction_spacing_um:g}"
    )


def request_from_allowed_command(
    command: str,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any] | None:
    if not is_allowed_next_tool_command(command):
        warnings.append("LLM next_tool_command was not on the allowed tool command list.")
        return None

    tokens = shlex.split(command)
    if len(tokens) < 3 or tokens[2] != "tcad_agent.tools.pn_junction_iv":
        warnings.append("Allowed command is not a pn_junction_iv follow-up command.")
        return None

    request = clean_request(state.get("request") or {})
    index = 3
    while index < len(tokens):
        flag = tokens[index]
        if flag == "--resume":
            warnings.append("--resume from LLM command was ignored for follow-up planning.")
            index += 1
            continue
        field = ALLOWED_PN_FLAGS.get(flag)
        if field is None:
            warnings.append(f"Unsupported LLM command argument ignored: {flag}")
            index += 1
            continue
        if index + 1 >= len(tokens):
            warnings.append(f"Missing value for LLM command argument: {flag}")
            return None
        request[field] = coerce_value(field, tokens[index + 1])
        index += 2
    return request


def deterministic_followup_request(
    state: dict[str, Any],
    warnings: list[str],
) -> tuple[dict[str, Any], str]:
    request = clean_request(state.get("request") or {})
    codes = issue_codes(state)
    reason = "deterministic follow-up from quality_report"

    start = float(request.get("start", 0.0))
    stop = float(request.get("stop", 0.5))
    step = float(request.get("step", 0.1))
    min_step = float(request.get("min_step", min(step, 0.0125)))

    if "current_exceeds_policy" in codes:
        new_stop = min(stop, max(start, 0.5))
        if new_stop <= start:
            new_stop = stop
            warnings.append("Could not narrow stop voltage because it would not exceed start.")
        else:
            stop = new_stop
            step = min(step, max((stop - start) / 5.0, 1e-6), 0.1)
            min_step = min(min_step, max(step / 4.0, 1e-6))
            reason = "narrow voltage range after physically suspicious current"

    if "too_many_convergence_failures" in codes and "current_exceeds_policy" not in codes:
        step = max(step / 2.0, min_step)
        min_step = min(min_step, max(step / 4.0, 1e-6))
        reason = "reduce initial bias step after convergence recovery"

    request["start"] = start
    request["stop"] = stop
    request["step"] = step
    request["min_step"] = min_step
    request["max_attempts"] = max(int(request.get("max_attempts", 3)), 3)
    request["timeout_seconds"] = float(request.get("timeout_seconds", 300.0))
    request["quality_min_points"] = int(request.get("quality_min_points", 3))
    request["quality_max_abs_current_a"] = float(
        request.get("quality_max_abs_current_a", 1.0)
    )
    request["quality_max_convergence_failures"] = int(
        request.get("quality_max_convergence_failures", 0)
    )
    return request, reason


def build_next_request(
    state: dict[str, Any],
    diagnosis: dict[str, Any] | None,
    warnings: list[str],
) -> tuple[PNJunctionIVRequest | None, str]:
    quality_status = ((state.get("quality_report") or {}).get("status"))
    if quality_status == "passed":
        return None, "quality_report.status is passed; no follow-up required"

    request = None
    reason = "deterministic follow-up from quality_report"
    parsed_response = (diagnosis or {}).get("parsed_response") or {}
    next_tool_command = parsed_response.get("next_tool_command")
    if next_tool_command:
        request = request_from_allowed_command(next_tool_command, state, warnings)
        reason = "follow-up request derived from whitelisted LLM next_tool_command"

    if request is None:
        request, reason = deterministic_followup_request(state, warnings)

    run_root = Path(request.get("run_root") or state.get("request", {}).get("run_root") or "runs/agent_tools")
    request = normalize_followup_request(request, warnings)
    request["run_root"] = str(run_root)
    request["run_id"] = next_followup_run_id(state, run_root)
    request["resume"] = False

    try:
        return PNJunctionIVRequest.model_validate(request), reason
    except ValidationError as exc:
        warnings.append(str(exc))
        return None, "follow-up request validation failed"


def build_strategy_plan(
    state_path: Path,
    diagnosis_path: Path | None = None,
    output_path: Path | None = None,
    execute: bool = False,
) -> StrategyPlan:
    state = load_json(state_path)
    actual_diagnosis_path = diagnosis_path or default_diagnosis_path(state_path)
    diagnosis = load_json(actual_diagnosis_path) if actual_diagnosis_path.exists() else None
    warnings: list[str] = []
    constraints = [
        "Do not execute arbitrary shell commands suggested by LLM output.",
        "Only PN junction IV follow-up requests are generated by this executor.",
        "A passed quality_report skips follow-up unless the caller changes state policy.",
    ]

    next_request, reason = build_next_request(state, diagnosis, warnings)
    actual_output = output_path or default_output_path(state_path)

    if next_request is None:
        plan = StrategyPlan(
            status=StrategyStatus.SKIPPED if "passed" in reason else StrategyStatus.FAILED,
            state_path=str(state_path),
            diagnosis_path=str(actual_diagnosis_path) if actual_diagnosis_path.exists() else None,
            output_path=str(actual_output),
            source_run_id=state.get("run_id"),
            execute=execute,
            reason=reason,
            warnings=warnings,
            constraints=constraints,
        )
        write_json(actual_output, plan.model_dump(mode="json"))
        return plan

    plan = StrategyPlan(
        status=StrategyStatus.PLANNED,
        state_path=str(state_path),
        diagnosis_path=str(actual_diagnosis_path) if actual_diagnosis_path.exists() else None,
        output_path=str(actual_output),
        source_run_id=state.get("run_id"),
        execute=execute,
        reason=reason,
        warnings=warnings,
        constraints=constraints,
        next_request=next_request.model_dump(mode="json"),
        next_command=command_from_request(next_request),
    )

    if execute:
        result = run_pn_junction_iv_sweep(next_request)
        plan.status = StrategyStatus.EXECUTED
        plan.executed_result = result

    write_json(actual_output, plan.model_dump(mode="json"))
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or execute a constrained TCAD follow-up.")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        plan = build_strategy_plan(
            state_path=args.state,
            diagnosis_path=args.diagnosis,
            output_path=args.output,
            execute=args.execute,
        )
        print(json.dumps(plan.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if plan.status != StrategyStatus.FAILED else 1)
    except Exception as exc:
        output = args.output or default_output_path(args.state)
        plan = StrategyPlan(
            status=StrategyStatus.FAILED,
            state_path=str(args.state),
            diagnosis_path=str(args.diagnosis) if args.diagnosis else None,
            output_path=str(output),
            reason=str(exc),
        )
        write_json(output, plan.model_dump(mode="json"))
        print(json.dumps(plan.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
