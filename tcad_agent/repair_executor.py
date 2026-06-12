from __future__ import annotations

import argparse
import json
import hashlib
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from tcad_agent.curve_diagnostics import compare_state_mutation_effect
from tcad_agent.physical_benchmark import BenchmarkStatus, run_physical_benchmark
from tcad_agent.repair_memory import append_repair_case_memory
from tcad_agent.repair_agent import ChatClient, RepairAgentDecision, decide_repair_action_with_agent
from tcad_agent.repair_strategy import RepairAction, RepairPlan, build_repair_plan, issue_codes, repair_request
from tcad_agent.task_spec import PROJECT_ROOT
from tcad_agent.tools.diode_breakdown import DiodeBreakdownRequest, run_diode_breakdown_sweep
from tcad_agent.tools.extended_device_sweep import ExtendedDeviceRequest, run_extended_device_sweep
from tcad_agent.tools.mos_capacitor_cv import MOSCapacitorCVRequest, run_mos_capacitor_cv_sweep
from tcad_agent.tools.mosfet_2d_id import MOSFET2DIDRequest, run_mosfet_2d_id_sweep
from tcad_agent.tools.pn_junction_iv import PNJunctionIVRequest, run_pn_junction_iv_sweep


Runner = Callable[[dict[str, Any]], dict[str, Any]]


class RepairExecutionStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    WAITING_FOR_USER = "waiting_for_user"
    FAILED = "failed"


class RepairExecutionAttempt(BaseModel):
    index: int
    action_name: str
    target_tool: str
    source_state_path: str
    request_patch: dict[str, Any]
    deck_patch: dict[str, Any] = Field(default_factory=dict)
    deck_mutations: list[dict[str, Any]] = Field(default_factory=list)
    agent_policy: dict[str, Any] | None = None
    next_request: dict[str, Any]
    status: RepairExecutionStatus
    started_at: str
    completed_at: str | None = None
    result_state_path: str | None = None
    quality_status: str | None = None
    benchmark_status: str | None = None
    benchmark_path: str | None = None
    benchmark_summary: dict[str, Any] | None = None
    mutation_effect_analysis: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    failure_reason: str | None = None


class RepairExecutionState(BaseModel):
    tool_name: str = "tcad_repair_executor"
    status: RepairExecutionStatus
    execution_id: str
    source_state_path: str
    execution_dir: str
    created_at: str
    updated_at: str
    execute: bool
    max_rounds: int
    allow_user_confirmation_actions: bool = False
    current_state_path: str
    repair_plan_path: str | None = None
    attempts: list[RepairExecutionAttempt] = Field(default_factory=list)
    final_state_path: str | None = None
    final_quality_status: str | None = None
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    next_action: str | None = None
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_execution_id() -> str:
    return f"repair_exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_state(state: RepairExecutionState, path: Path) -> None:
    state.updated_at = utc_timestamp()
    write_json(path, state.model_dump(mode="json"))


def load_state(path: Path) -> RepairExecutionState:
    return RepairExecutionState.model_validate_json(path.read_text(encoding="utf-8"))


def default_execution_root(source_state_path: Path) -> Path:
    return source_state_path.parent / "repair_execution"


def sanitize_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned or "repair"


def compact_source_run_id(value: str, *, max_chars: int = 72) -> str:
    cleaned = sanitize_id(value)
    if "_repair_" in cleaned:
        cleaned = cleaned.split("_repair_", 1)[0]
    if len(cleaned) <= max_chars:
        return cleaned
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned[: max_chars - 11]}_{digest}"


def quality_status_from_state(state: dict[str, Any]) -> str | None:
    report = state.get("quality_report") or state.get("final_quality_report") or {}
    return report.get("status")


def is_accepted_state(state: dict[str, Any]) -> bool:
    benchmark_context = state.get("benchmark_context") or {}
    benchmark_status = benchmark_context.get("status")
    if benchmark_status in {"failed", "suspicious"}:
        return False
    return state.get("status") == "completed" and quality_status_from_state(state) == "passed"


def benchmark_issue_dicts(benchmark_data: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for check in benchmark_data.get("checks") or []:
        if not isinstance(check, dict) or check.get("severity") not in {"warning", "error"}:
            continue
        issues.append(
            {
                "code": check.get("code"),
                "severity": check.get("severity"),
                "message": check.get("message"),
                "evidence": check.get("observed") or {},
            }
        )
    return issues


def severity_rank(value: str | None) -> int:
    return {"passed": 0, "completed": 0, "suspicious": 1, "failed": 2}.get(str(value or ""), 0)


def write_benchmark_augmented_state(
    state: RepairExecutionState,
    repaired_state_path: Path,
    benchmark_data: dict[str, Any],
) -> str:
    repaired = read_json(repaired_state_path)
    quality = dict(repaired.get("quality_report") or {})
    existing = [issue for issue in quality.get("issues") or [] if isinstance(issue, dict)]
    existing_codes = {str(issue.get("code")) for issue in existing if issue.get("code")}
    for issue in benchmark_issue_dicts(benchmark_data):
        code = str(issue.get("code") or "")
        if not code or code in existing_codes:
            continue
        existing.append(issue)
        existing_codes.add(code)
    benchmark_status = str(benchmark_data.get("status") or "")
    if severity_rank(benchmark_status) > severity_rank(str(quality.get("status") or "passed")):
        quality["status"] = benchmark_status
    quality["issues"] = existing
    repaired["quality_report"] = quality
    repaired["benchmark_context"] = {
        "status": benchmark_data.get("status"),
        "summary": benchmark_data.get("summary") or {},
        "benchmark_path": benchmark_data.get("benchmark_path"),
    }
    output = Path(state.execution_dir) / "benchmark_augmented" / f"{repaired_state_path.parent.name}_{repaired_state_path.stem}.json"
    write_json(output, repaired)
    return str(output)


def infer_result_state_path(result: dict[str, Any]) -> str | None:
    for key in ["state_path", "result_state_path", "source_state_path"]:
        value = result.get(key)
        if value:
            return str(value)
    run_dir = result.get("run_dir") or result.get("convergence_dir")
    if run_dir:
        candidate = Path(run_dir) / "state.json"
        if candidate.exists():
            return str(candidate.resolve())
    return None


def allocate_repair_run_id(source_state: dict[str, Any], action: RepairAction, round_index: int) -> str:
    source = source_state.get("run_id") or source_state.get("task_id") or source_state.get("convergence_id") or "run"
    compact_source = compact_source_run_id(str(source))
    return sanitize_id(f"{compact_source}_repair_{action.name}_{round_index:03d}")


def apply_patch_to_request(
    source_state: dict[str, Any],
    action: RepairAction,
    round_index: int,
) -> dict[str, Any]:
    request = repair_request(source_state)
    baseline_path = (
        (source_state.get("repair_context") or {}).get("baseline_state_path")
        or source_state.get("repair_baseline_state_path")
        or source_state.get("state_path")
    )
    request.update(action.request_patch)
    if action.deck_mutations:
        existing = request.get("tcad_deck_mutations")
        request["tcad_deck_mutations"] = existing if isinstance(existing, list) and existing else action.deck_mutations
    if action.deck_patch:
        history = request.get("deck_patch_history")
        if not isinstance(history, list):
            history = []
        patch_record = dict(action.deck_patch)
        patch_record.setdefault("action_name", action.name)
        patch_record.setdefault("source_state_path", source_state.get("state_path"))
        patch_record.setdefault("baseline_state_path", baseline_path)
        request["deck_patch_history"] = [*history, patch_record]
    request.setdefault("resume", False)
    request["repair_source_state_path"] = source_state.get("state_path")
    if baseline_path:
        request["repair_baseline_state_path"] = baseline_path
    request["run_id"] = allocate_repair_run_id(source_state, action, round_index)
    if "run_root" not in request and source_state.get("run_dir"):
        run_dir = Path(str(source_state["run_dir"]))
        # Most tool layouts are <run_root>/<tool_family>/<run_id>.
        request["run_root"] = str(run_dir.parent.parent if len(run_dir.parents) >= 2 else PROJECT_ROOT / "runs" / "agent_tools")
    elif "run_root" not in request:
        request["run_root"] = str(PROJECT_ROOT / "runs" / "agent_tools")
    return request


def annotate_source_state_path(state: dict[str, Any], source_path: Path) -> dict[str, Any]:
    if "state_path" in state:
        return state
    updated = dict(state)
    updated["state_path"] = str(source_path)
    return updated


def write_repair_lineage_augmented_state(
    *,
    source_path: Path,
    source_state: dict[str, Any],
    repaired_state_path: Path,
    action: RepairAction,
    next_request: dict[str, Any],
    attempt: RepairExecutionAttempt,
) -> None:
    repaired = read_json(repaired_state_path)
    baseline_path = Path(str(next_request.get("repair_baseline_state_path") or source_path))
    context = dict(repaired.get("repair_context") or {})
    context.update(
        {
            "schema_version": "actsoft.tcad.repair_context.v1",
            "baseline_state_path": str(baseline_path),
            "parent_state_path": str(source_path),
            "action_name": action.name,
            "deck_patch": action.deck_patch,
            "deck_mutations": action.deck_mutations,
            "attempt_index": attempt.index,
        }
    )
    if attempt.agent_policy:
        context["agent_policy"] = attempt.agent_policy
        context["agent_observation_summary"] = attempt.agent_policy.get("observation_summary")
        context["agent_hypothesis_zh"] = attempt.agent_policy.get("hypothesis_zh")
        context["agent_tool_plan"] = attempt.agent_policy.get("tool_plan") or []
        context["agent_safety_review"] = attempt.agent_policy.get("safety_review") or {}
    repaired["repair_context"] = context
    if action.deck_patch and baseline_path.exists():
        diagnostic = compare_state_mutation_effect(
            baseline_path,
            repaired_state_path,
            deck_patch=action.deck_patch,
            issue_codes=sorted(issue_codes(source_state)),
            overlay_output_path=repaired_state_path.parent / "baseline_mutation_overlay.svg",
        )
        diagnostic_data = diagnostic.model_dump(mode="json")
        repaired["mutation_effect_analysis"] = diagnostic_data
        attempt.mutation_effect_analysis = diagnostic_data
        if diagnostic.overlay_svg_path:
            summary = repaired.get("final_summary") or {}
            artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
            artifacts["baseline_mutation_overlay"] = diagnostic.overlay_svg_path
            summary["artifacts"] = artifacts
            repaired["final_summary"] = summary
        memory_path = append_repair_case_memory(
            baseline_state_path=str(baseline_path),
            mutation_state_path=str(repaired_state_path),
            action_name=action.name,
            issue_codes=sorted(issue_codes(source_state)),
            mutation_effect_analysis=diagnostic_data,
        )
        context["repair_case_memory_path"] = memory_path
        context["mutation_effect_decision"] = diagnostic.decision
        context["recommended_next_target"] = diagnostic.recommended_next_target
        context["worth_continuing_mutation"] = diagnostic.worth_continuing
    write_json(repaired_state_path, repaired)


def default_runner_registry() -> dict[str, Runner]:
    def pn_runner(request: dict[str, Any]) -> dict[str, Any]:
        return run_pn_junction_iv_sweep(PNJunctionIVRequest.model_validate(request))

    def moscap_runner(request: dict[str, Any]) -> dict[str, Any]:
        return run_mos_capacitor_cv_sweep(MOSCapacitorCVRequest.model_validate(request))

    def diode_runner(request: dict[str, Any]) -> dict[str, Any]:
        return run_diode_breakdown_sweep(DiodeBreakdownRequest.model_validate(request))

    def mosfet_runner(request: dict[str, Any]) -> dict[str, Any]:
        return run_mosfet_2d_id_sweep(MOSFET2DIDRequest.model_validate(request))

    def extended_runner(request: dict[str, Any]) -> dict[str, Any]:
        return run_extended_device_sweep(ExtendedDeviceRequest.model_validate(request)).model_dump(mode="json")

    return {
        "pn_junction_iv_sweep": pn_runner,
        "mos_capacitor_cv_sweep": moscap_runner,
        "diode_breakdown_leakage_sweep": diode_runner,
        "mosfet_2d_id_sweep": mosfet_runner,
        "extended_device_sweep": extended_runner,
    }


def is_executable_action(action: RepairAction, *, allow_user_confirmation_actions: bool) -> bool:
    if action.user_confirmation_required and not allow_user_confirmation_actions:
        return False
    if not action.target_tool:
        return False
    if not action.request_patch and action.name.endswith("_review"):
        return False
    return True


def choose_action(
    plan: RepairPlan,
    *,
    allow_user_confirmation_actions: bool,
    tried_action_names: set[str] | None = None,
) -> RepairAction | None:
    tried = tried_action_names or set()
    first_executable: RepairAction | None = None
    for action in plan.actions:
        if not is_executable_action(action, allow_user_confirmation_actions=allow_user_confirmation_actions):
            continue
        if first_executable is None:
            first_executable = action
        if action.name in tried:
            continue
        return action
    return first_executable


def create_initial_state(
    source_state_path: Path,
    execution_id: str,
    execution_dir: Path,
    execute: bool,
    max_rounds: int,
    allow_user_confirmation_actions: bool,
) -> RepairExecutionState:
    now = utc_timestamp()
    return RepairExecutionState(
        status=RepairExecutionStatus.RUNNING if execute else RepairExecutionStatus.PLANNED,
        execution_id=execution_id,
        source_state_path=str(source_state_path),
        execution_dir=str(execution_dir),
        created_at=now,
        updated_at=now,
        execute=execute,
        max_rounds=max_rounds,
        allow_user_confirmation_actions=allow_user_confirmation_actions,
        current_state_path=str(source_state_path),
        checkpoint={"completed_rounds": 0},
        next_action="build repair plan",
    )


def execute_repair_round(
    state: RepairExecutionState,
    state_path: Path,
    *,
    registry: dict[str, Runner] | None = None,
    use_agent_policy: bool = False,
    llm_client: ChatClient | None = None,
) -> RepairExecutionAttempt | None:
    source_path = Path(state.current_state_path)
    source_state = annotate_source_state_path(read_json(source_path), source_path)
    if is_accepted_state(source_state):
        state.status = RepairExecutionStatus.COMPLETED
        state.final_state_path = str(source_path)
        state.final_quality_status = quality_status_from_state(source_state)
        state.next_action = "accept repaired TCAD result"
        return None

    plan = build_repair_plan(source_path)
    state.repair_plan_path = plan.output_path
    tried_action_names = {attempt.action_name for attempt in state.attempts}
    state.checkpoint["tried_repair_actions"] = sorted(tried_action_names)
    agent_decision: RepairAgentDecision | None = None
    action: RepairAction | None = None
    if use_agent_policy:
        agent_decision = decide_repair_action_with_agent(
            source_path,
            deterministic_plan=plan,
            client=llm_client,
            allow_fallback=True,
        )
        state.checkpoint["last_repair_agent_decision"] = agent_decision.model_dump(mode="json")
        if agent_decision.action and agent_decision.action.name in tried_action_names:
            state.checkpoint["last_repair_agent_duplicate_action"] = agent_decision.action.name
        elif agent_decision.action:
            if agent_decision.action.user_confirmation_required and not state.allow_user_confirmation_actions:
                state.status = RepairExecutionStatus.WAITING_FOR_USER
                state.next_action = "wait for user confirmation before applying agent-selected repair action"
                state.checkpoint["blocked_repair_agent_decision"] = agent_decision.model_dump(mode="json")
                return None
            if is_executable_action(agent_decision.action, allow_user_confirmation_actions=state.allow_user_confirmation_actions):
                action = agent_decision.action
            else:
                state.checkpoint["last_repair_agent_inexecutable_action"] = agent_decision.action.model_dump(mode="json")
    if action is None:
        action = choose_action(
            plan,
            allow_user_confirmation_actions=state.allow_user_confirmation_actions,
            tried_action_names=tried_action_names,
        )
    if action is None:
        if any(candidate.user_confirmation_required for candidate in plan.actions):
            state.status = RepairExecutionStatus.WAITING_FOR_USER
            state.next_action = "wait for user confirmation before applying sensitive repair action"
            state.checkpoint["blocked_repair_plan_path"] = plan.output_path
        elif not plan.actions:
            state.status = RepairExecutionStatus.FAILED
            state.failure_reason = "repair plan did not contain an executable action"
            state.next_action = "inspect run manually"
        else:
            state.status = RepairExecutionStatus.FAILED
            state.failure_reason = "repair plan actions were not executable by this executor"
            state.next_action = "extend repair executor registry or inspect manually"
        return None

    target_tool = action.target_tool or str(source_state.get("tool_name"))
    runners = registry or default_runner_registry()
    runner = runners.get(target_tool)
    round_index = len(state.attempts) + 1
    next_request = apply_patch_to_request(source_state, action, round_index)
    attempt = RepairExecutionAttempt(
        index=round_index,
        action_name=action.name,
        target_tool=target_tool,
        source_state_path=str(source_path),
        request_patch=action.request_patch,
        deck_patch=action.deck_patch,
        deck_mutations=action.deck_mutations,
        agent_policy=agent_decision.model_dump(mode="json") if agent_decision else None,
        next_request=next_request,
        status=RepairExecutionStatus.PLANNED if not state.execute else RepairExecutionStatus.RUNNING,
        started_at=utc_timestamp(),
    )
    state.attempts.append(attempt)
    state.next_action = f"execute repair action {action.name}"
    write_state(state, state_path)

    if not state.execute:
        return attempt

    if runner is None:
        attempt.status = RepairExecutionStatus.FAILED
        attempt.failure_reason = f"no runner registered for target tool {target_tool}"
        attempt.completed_at = utc_timestamp()
        state.status = RepairExecutionStatus.FAILED
        state.failure_reason = attempt.failure_reason
        state.next_action = "extend repair executor runner registry"
        return attempt

    try:
        result = runner(next_request)
        attempt.result = result
        result_state_path = infer_result_state_path(result)
        attempt.result_state_path = result_state_path
        if result_state_path:
            write_repair_lineage_augmented_state(
                source_path=source_path,
                source_state=source_state,
                repaired_state_path=Path(result_state_path),
                action=action,
                next_request=next_request,
                attempt=attempt,
            )
            state.current_state_path = result_state_path
            repaired_state = read_json(Path(result_state_path))
            attempt.quality_status = quality_status_from_state(repaired_state)
            benchmark = run_physical_benchmark(Path(result_state_path))
            benchmark_data = benchmark.model_dump(mode="json")
            attempt.benchmark_status = str(benchmark.status.value if isinstance(benchmark.status, BenchmarkStatus) else benchmark.status)
            attempt.benchmark_path = benchmark.benchmark_path
            attempt.benchmark_summary = benchmark_data.get("summary") or {}
            state.checkpoint["last_repair_benchmark"] = {
                "status": attempt.benchmark_status,
                "benchmark_path": attempt.benchmark_path,
                "signoff_status": (attempt.benchmark_summary or {}).get("signoff_status"),
                "blocking_codes": (attempt.benchmark_summary or {}).get("blocking_codes") or [],
                "warning_codes": (attempt.benchmark_summary or {}).get("warning_codes") or [],
            }
            if is_accepted_state(repaired_state) and attempt.benchmark_status == "passed":
                state.status = RepairExecutionStatus.COMPLETED
                state.final_state_path = result_state_path
                state.final_quality_status = attempt.quality_status
                state.next_action = "accept repaired TCAD result with passed physical benchmark"
            else:
                if attempt.benchmark_status in {"failed", "suspicious"}:
                    state.current_state_path = write_benchmark_augmented_state(state, Path(result_state_path), benchmark_data)
                state.status = RepairExecutionStatus.RUNNING
                state.final_quality_status = attempt.benchmark_status if attempt.benchmark_status in {"failed", "suspicious"} else attempt.quality_status
                state.next_action = "build next repair plan from repaired run and benchmark issues"
        else:
            attempt.quality_status = (result.get("quality_report") or {}).get("status")
            state.status = RepairExecutionStatus.FAILED
            state.failure_reason = "repair runner did not expose a result state path"
            state.next_action = "inspect repair runner output"
        attempt.status = RepairExecutionStatus.COMPLETED
    except Exception as exc:
        attempt.status = RepairExecutionStatus.FAILED
        attempt.failure_reason = str(exc)
        state.status = RepairExecutionStatus.RUNNING
        state.failure_reason = str(exc)
        state.next_action = "classify failed repair attempt and try next repair action"
    attempt.completed_at = utc_timestamp()
    state.checkpoint.update(
        {
            "completed_rounds": len(state.attempts),
            "last_attempt": attempt.model_dump(mode="json"),
        }
    )
    return attempt


def run_repair_executor(
    source_state_path: Path,
    *,
    execution_id: str | None = None,
    execution_root: Path | None = None,
    execute: bool = False,
    resume: bool = False,
    max_rounds: int = 3,
    allow_user_confirmation_actions: bool = False,
    registry: dict[str, Runner] | None = None,
    use_agent_policy: bool = False,
    llm_client: ChatClient | None = None,
) -> RepairExecutionState:
    actual_execution_id = execution_id or default_execution_id()
    actual_root = execution_root or default_execution_root(source_state_path)
    execution_dir = actual_root / actual_execution_id
    actual_state_path = execution_dir / "repair_execution_state.json"

    if resume and actual_state_path.exists():
        state = load_state(actual_state_path)
        state.execute = execute
        state.max_rounds = max_rounds
        state.allow_user_confirmation_actions = allow_user_confirmation_actions
        state.status = RepairExecutionStatus.RUNNING if execute else RepairExecutionStatus.PLANNED
    else:
        execution_dir.mkdir(parents=True, exist_ok=True)
        state = create_initial_state(
            source_state_path,
            actual_execution_id,
            execution_dir,
            execute,
            max_rounds,
            allow_user_confirmation_actions,
        )
    write_state(state, actual_state_path)

    while len(state.attempts) < state.max_rounds and state.status in {
        RepairExecutionStatus.RUNNING,
        RepairExecutionStatus.PLANNED,
    }:
        execute_repair_round(
            state,
            actual_state_path,
            registry=registry,
            use_agent_policy=use_agent_policy,
            llm_client=llm_client,
        )
        write_state(state, actual_state_path)
        if not state.execute:
            state.status = RepairExecutionStatus.PLANNED
            state.checkpoint["planned_attempt"] = state.attempts[-1].model_dump(mode="json") if state.attempts else None
            write_state(state, actual_state_path)
            return state
        if state.status in {
            RepairExecutionStatus.COMPLETED,
            RepairExecutionStatus.WAITING_FOR_USER,
            RepairExecutionStatus.FAILED,
        }:
            write_state(state, actual_state_path)
            return state

    if state.status == RepairExecutionStatus.RUNNING:
        state.status = RepairExecutionStatus.FAILED
        state.failure_reason = f"maximum repair rounds reached: {state.max_rounds}"
        state.next_action = "inspect last repair attempt and decide whether to extend budget"
    write_state(state, actual_state_path)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute TCAD repair plans in a closed loop.")
    parser.add_argument("--state", type=Path, required=True, help="Source run state.json.")
    parser.add_argument("--execution-id", default=None)
    parser.add_argument("--execution-root", type=Path, default=None)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-user-confirmation-actions", action="store_true")
    parser.add_argument("--use-agent-policy", action="store_true", help="Let the configured LLM choose the next repair action before deterministic fallback.")
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        result = run_repair_executor(
            args.state,
            execution_id=args.execution_id,
            execution_root=args.execution_root,
            execute=args.execute,
            resume=args.resume,
            max_rounds=args.max_rounds,
            allow_user_confirmation_actions=args.allow_user_confirmation_actions,
            use_agent_policy=args.use_agent_policy,
        )
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if result.status != RepairExecutionStatus.FAILED else 1)
    except Exception as exc:
        print(json.dumps({"tool_name": "tcad_repair_executor", "status": "failed", "failure_reason": str(exc)}, indent=2, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
