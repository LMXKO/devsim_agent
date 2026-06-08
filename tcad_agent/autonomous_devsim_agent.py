from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

from pydantic import BaseModel, Field

from tcad_agent.llm import LLMClient, LLMConfig
from tcad_agent.reporting import final_artifacts, final_metrics, load_final_state
from tcad_agent.repair_executor import run_repair_executor
from tcad_agent.task_planner import parse_json_object
from tcad_agent.task_spec import PROJECT_ROOT


class ChatClient(Protocol):
    config: LLMConfig

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        ...


Runner = Callable[[dict[str, Any]], dict[str, Any]]
RepairRunner = Callable[..., Any]


class DevsimAgentStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    WAITING_FOR_USER = "waiting_for_user"
    COMPLETED = "completed"
    FAILED = "failed"


class DevsimAgentStepStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DevsimAgentActionKind(str, Enum):
    RUN_SUPERVISOR = "run_supervisor"
    RUN_TOOL = "run_tool"
    RUN_REPAIR_EXECUTOR = "run_repair_executor"
    RUN_PHYSICAL_BENCHMARK = "run_physical_benchmark"
    GENERATE_REPORT = "generate_report"
    GENERATE_DASHBOARD = "generate_dashboard"
    STOP_SUCCESS = "stop_success"
    ASK_USER = "ask_user"
    NOOP = "noop"


class AutonomousDevsimRequest(BaseModel):
    goal_text: str
    agent_id: str | None = None
    agent_root: Path = PROJECT_ROOT / "runs" / "autonomous_devsim_agent"
    execute: bool = False
    resume: bool = False
    max_steps: int = Field(default=12, ge=1)
    initial_tool_name: str | None = None
    initial_request: dict[str, Any] = Field(default_factory=dict)
    source_state_path: str | None = None
    use_llm: bool = True
    allow_llm_fallback: bool = True
    use_agent_policy: bool = True
    allow_user_confirmation_actions: bool = False
    supervisor_max_cycles: int = Field(default=3, ge=1)
    repair_max_rounds: int = Field(default=3, ge=1)
    generate_report: bool = True
    generate_dashboard: bool = True


class DevsimAgentAction(BaseModel):
    kind: DevsimAgentActionKind
    reason: str
    tool_name: str | None = None
    request: dict[str, Any] = Field(default_factory=dict)
    source_state_path: str | None = None
    user_confirmation_required: bool = False


class DevsimAgentStep(BaseModel):
    index: int
    kind: DevsimAgentActionKind
    status: DevsimAgentStepStatus
    reason: str
    started_at: str
    completed_at: str | None = None
    action: dict[str, Any]
    observation: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    result_state_path: str | None = None
    error: str | None = None


class AutonomousDevsimAgentState(BaseModel):
    tool_name: str = "autonomous_devsim_agent"
    status: DevsimAgentStatus
    agent_id: str
    agent_dir: str
    goal_text: str
    created_at: str
    updated_at: str
    execute: bool
    max_steps: int
    steps: list[DevsimAgentStep] = Field(default_factory=list)
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    latest_state_path: str | None = None
    final_state_path: str | None = None
    final_report_path: str | None = None
    final_dashboard_path: str | None = None
    next_action: str | None = None
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_agent_id() -> str:
    return f"devsim_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def state_path(agent_root: Path, agent_id: str) -> Path:
    return agent_root / agent_id / "autonomous_devsim_agent_state.json"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_state(state: AutonomousDevsimAgentState, path: Path) -> None:
    state.updated_at = utc_timestamp()
    write_json(path, state.model_dump(mode="json"))


def load_state(path: Path) -> AutonomousDevsimAgentState:
    return AutonomousDevsimAgentState.model_validate_json(path.read_text(encoding="utf-8"))


def create_initial_state(request: AutonomousDevsimRequest, agent_id: str, agent_dir: Path) -> AutonomousDevsimAgentState:
    now = utc_timestamp()
    return AutonomousDevsimAgentState(
        status=DevsimAgentStatus.RUNNING if request.execute else DevsimAgentStatus.PLANNED,
        agent_id=agent_id,
        agent_dir=str(agent_dir),
        goal_text=request.goal_text,
        created_at=now,
        updated_at=now,
        execute=request.execute,
        max_steps=request.max_steps,
        latest_state_path=request.source_state_path,
        checkpoint={
            "completed_steps": 0,
            "agent_first_policy": {
                "enabled": request.use_llm,
                "deterministic_fallback": request.allow_llm_fallback,
                "repair_agent_policy": request.use_agent_policy,
            },
        },
        next_action="choose first DEVSIM agent tool call",
    )


def prepare_state(request: AutonomousDevsimRequest) -> tuple[AutonomousDevsimAgentState, Path]:
    actual_id = request.agent_id or default_agent_id()
    actual_path = state_path(request.agent_root, actual_id)
    if request.resume:
        if not actual_path.exists():
            raise FileNotFoundError(f"Cannot resume autonomous DEVSIM agent; state does not exist: {actual_path}")
        state = load_state(actual_path)
        state.execute = request.execute
        state.max_steps = request.max_steps
        state.status = DevsimAgentStatus.RUNNING if request.execute else DevsimAgentStatus.PLANNED
        write_state(state, actual_path)
        return state, actual_path
    agent_dir = request.agent_root / actual_id
    agent_dir.mkdir(parents=True, exist_ok=True)
    state = create_initial_state(request, actual_id, agent_dir)
    write_state(state, actual_path)
    return state, actual_path


def result_to_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):
        dumped = result.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    return {"value": result}


def collect_state_paths(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {
                "state_path",
                "source_state_path",
                "result_state_path",
                "final_state_path",
                "current_state_path",
                "verified_state_path",
                "sweep_state_path",
                "optimization_state_path",
            } and item:
                paths.append(str(item))
            else:
                paths.extend(collect_state_paths(item))
    elif isinstance(value, list):
        for item in value:
            paths.extend(collect_state_paths(item))
    return paths


def infer_result_state_path(result: dict[str, Any]) -> str | None:
    for path in reversed(collect_state_paths(result)):
        if path:
            return path
    run_dir = result.get("run_dir") or result.get("convergence_dir") or result.get("supervisor_dir") or result.get("mission_dir")
    if run_dir:
        for name in ["state.json", "supervisor_state.json", "mission_state.json", "sweep_state.json", "optimization_state.json"]:
            candidate = Path(str(run_dir)) / name
            if candidate.exists():
                return str(candidate.resolve())
    return None


def issue_codes_from_state(state_data: dict[str, Any]) -> list[str]:
    quality = state_data.get("quality_report") or state_data.get("final_quality_report") or {}
    codes = []
    for issue in quality.get("issues") or []:
        if isinstance(issue, dict) and issue.get("code"):
            codes.append(str(issue["code"]))
    return codes


def observe_state(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {"state_path": None, "summary": "no TCAD state has been produced yet"}
    state_data = load_final_state(path_value)
    if not state_data:
        return {"state_path": path_value, "summary": "state file is missing or not JSON-readable"}
    quality = state_data.get("quality_report") or state_data.get("final_quality_report") or {}
    benchmark = state_data.get("benchmark_context") or {}
    return {
        "state_path": path_value,
        "tool_name": state_data.get("tool_name"),
        "status": state_data.get("status"),
        "quality_status": quality.get("status"),
        "issue_codes": issue_codes_from_state(state_data),
        "metrics": final_metrics(state_data),
        "artifacts": final_artifacts(state_data),
        "benchmark_context": benchmark,
        "repair_context": state_data.get("repair_context"),
        "mutation_effect_analysis": state_data.get("mutation_effect_analysis"),
    }


def dashboard_supported(path_value: str | None) -> bool:
    if not path_value:
        return False
    state_data = load_final_state(path_value)
    return bool(state_data and state_data.get("tool_name") in {"parameter_sweep", "adaptive_optimizer", "multidim_optimizer"})


def compact_steps(state: AutonomousDevsimAgentState) -> list[dict[str, Any]]:
    return [
        {
            "index": step.index,
            "kind": step.kind.value,
            "status": step.status.value,
            "reason": step.reason,
            "result_state_path": step.result_state_path,
            "error": step.error,
        }
        for step in state.steps[-8:]
    ]


def build_agent_context(state: AutonomousDevsimAgentState, request: AutonomousDevsimRequest) -> dict[str, Any]:
    return {
        "goal_text": state.goal_text,
        "status": state.status,
        "execute": state.execute,
        "max_steps": state.max_steps,
        "completed_steps": len(state.steps),
        "latest_observation": observe_state(state.latest_state_path),
        "checkpoint": state.checkpoint,
        "recent_steps": compact_steps(state),
        "initial_tool_name": request.initial_tool_name,
        "initial_request": request.initial_request,
        "source_state_path": request.source_state_path,
        "toolbelt": [
            {"kind": "run_supervisor", "purpose": "route a natural-language TCAD goal to an executable DEVSIM-backed tool"},
            {"kind": "run_tool", "purpose": "run one registered TCAD tool with a structured request"},
            {"kind": "run_repair_executor", "purpose": "repair a failed or suspicious state using repair agent policy and deterministic fallback"},
            {"kind": "run_physical_benchmark", "purpose": "check physics, capability boundary, convergence, and golden/measured evidence"},
            {"kind": "generate_report", "purpose": "write an engineer-readable Markdown report"},
            {"kind": "generate_dashboard", "purpose": "write an HTML dashboard with curves, metrics, and deck lineage"},
            {"kind": "stop_success", "purpose": "finish once enough evidence exists for a useful engineering conclusion"},
            {"kind": "ask_user", "purpose": "stop for confirmation on high-risk geometry/process/model changes"},
        ],
        "supported_action_kinds": [kind.value for kind in DevsimAgentActionKind],
    }


def build_agent_messages(context: dict[str, Any]) -> tuple[str, str]:
    system = (
        "你是一个长时间自主操作 DEVSIM 的 TCAD 工程 agent。"
        "你需要在已有工具能力内选择下一步工具调用，持续观察 state/log/curve/metrics/deck lineage，"
        "直到得到可用工程证据、需要用户确认、或预算耗尽。只返回 JSON。"
        "不要输出 shell 命令，不要编造工具，不要越过安全确认。"
    )
    user = {
        "task": "choose next autonomous DEVSIM agent action",
        "response_schema": {
            "action": {
                "kind": "one supported action kind",
                "reason": "中文说明证据链",
                "tool_name": "only for run_tool",
                "request": "object request for the selected tool",
                "source_state_path": "state to inspect/repair/benchmark/report",
                "user_confirmation_required": "boolean",
            },
            "observation_summary": "中文，当前看到的结果和风险",
            "hypothesis_zh": "中文，下一步背后的工程假设",
            "evidence_used": ["actual context keys used"],
        },
        "guardrails": [
            "一次只选择一个下一步 action。",
            "失败或可疑 state 优先 repair/benchmark，而不是直接报告成功。",
            "高风险 geometry/process/model patch 必须要求用户确认。",
            "compact/planned evidence 不能 stop_success 为签核结论。",
            "如果没有 state 且没有 initial tool，先 run_supervisor。",
        ],
        "context": context,
    }
    return system, json.dumps(user, ensure_ascii=False, indent=2)


def deterministic_action(state: AutonomousDevsimAgentState, request: AutonomousDevsimRequest) -> DevsimAgentAction:
    observation = observe_state(state.latest_state_path)
    quality_status = observation.get("quality_status")
    if not state.latest_state_path:
        if request.initial_tool_name:
            return DevsimAgentAction(
                kind=DevsimAgentActionKind.RUN_TOOL,
                tool_name=request.initial_tool_name,
                request=request.initial_request,
                reason="No TCAD state exists yet; run the requested initial DEVSIM tool.",
            )
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.RUN_SUPERVISOR,
            request={"goal_text": request.goal_text, "execute": True, "max_cycles": request.supervisor_max_cycles},
            reason="No TCAD state exists yet; ask the supervisor to route the goal to a supported tool.",
        )
    if quality_status in {"failed", "suspicious"}:
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.RUN_REPAIR_EXECUTOR,
            source_state_path=state.latest_state_path,
            reason=f"Latest state quality is {quality_status}; run the repair executor before reporting.",
        )
    if not state.checkpoint.get("physical_benchmark_done"):
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.RUN_PHYSICAL_BENCHMARK,
            source_state_path=state.latest_state_path,
            reason="Latest state is not failed; run physical benchmark before conclusion.",
        )
    if request.generate_report and not state.checkpoint.get("report_done"):
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.GENERATE_REPORT,
            source_state_path=state.latest_state_path,
            reason="Physical benchmark is done; generate an engineer-readable report.",
        )
    if request.generate_dashboard and not state.checkpoint.get("dashboard_done") and dashboard_supported(state.latest_state_path):
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.GENERATE_DASHBOARD,
            source_state_path=state.latest_state_path,
            reason="Generate an HTML dashboard for curve and lineage inspection.",
        )
    if request.generate_dashboard and not dashboard_supported(state.latest_state_path):
        state.checkpoint["dashboard_done"] = True
        state.checkpoint["dashboard_skipped_reason"] = "dashboard supports sweep/optimization states; latest state is a single tool run"
    return DevsimAgentAction(
        kind=DevsimAgentActionKind.STOP_SUCCESS,
        source_state_path=state.latest_state_path,
        reason="The autonomous DEVSIM loop has produced evidence, benchmark, and requested artifacts.",
    )


def normalize_agent_action(parsed: dict[str, Any], state: AutonomousDevsimAgentState) -> DevsimAgentAction | None:
    raw = parsed.get("action") if isinstance(parsed.get("action"), dict) else parsed
    if not isinstance(raw, dict):
        return None
    if raw.get("command") or raw.get("next_tool_command"):
        return None
    try:
        kind = DevsimAgentActionKind(str(raw.get("kind")))
    except Exception:
        return None
    return DevsimAgentAction(
        kind=kind,
        reason=str(raw.get("reason") or parsed.get("hypothesis_zh") or "Agent selected the next DEVSIM tool call."),
        tool_name=str(raw.get("tool_name")) if raw.get("tool_name") else None,
        request=raw.get("request") if isinstance(raw.get("request"), dict) else {},
        source_state_path=str(raw.get("source_state_path") or state.latest_state_path or "") or None,
        user_confirmation_required=bool(raw.get("user_confirmation_required") or raw.get("requires_user_confirmation")),
    )


def decide_next_action(
    state: AutonomousDevsimAgentState,
    request: AutonomousDevsimRequest,
    *,
    llm_client: ChatClient | None = None,
) -> tuple[DevsimAgentAction, dict[str, Any]]:
    fallback = deterministic_action(state, request)
    decision: dict[str, Any] = {
        "schema_version": "actsoft.tcad.autonomous_devsim_agent_decision.v1",
        "status": "fallback",
        "fallback_used": True,
        "deterministic_action": fallback.model_dump(mode="json"),
    }
    if not request.use_llm:
        return fallback, decision
    chat_client = llm_client or LLMClient()
    context = build_agent_context(state, request)
    system, user = build_agent_messages(context)
    try:
        raw = chat_client.chat(system=system, user=user, temperature=0.2)
    except Exception as exc:
        decision["failure_reason"] = str(exc)
        if request.allow_llm_fallback:
            return fallback, decision
        raise
    parsed = parse_json_object(raw)
    decision["raw_response"] = raw
    if parsed is None:
        decision["failure_reason"] = "agent did not return JSON"
        if request.allow_llm_fallback:
            return fallback, decision
        raise ValueError("autonomous DEVSIM agent did not return JSON")
    action = normalize_agent_action(parsed, state)
    decision["parsed_response"] = parsed
    if action is None:
        decision["failure_reason"] = "agent action was invalid or unsafe"
        if request.allow_llm_fallback:
            return fallback, decision
        raise ValueError("autonomous DEVSIM agent action was invalid or unsafe")
    decision.update(
        {
            "status": "completed",
            "fallback_used": False,
            "model": getattr(chat_client.config, "model", None),
            "action": action.model_dump(mode="json"),
            "observation_summary": parsed.get("observation_summary"),
            "hypothesis_zh": parsed.get("hypothesis_zh"),
            "evidence_used": parsed.get("evidence_used") or [],
        }
    )
    return action, decision


def default_runner_registry() -> dict[str, Runner]:
    from tcad_agent.run_queue import default_runner_registry as queue_registry
    from tcad_agent.dashboard import generate_experiment_dashboard

    registry = dict(queue_registry())

    def dashboard_runner(tool_request: dict[str, Any]) -> dict[str, Any]:
        source = tool_request.get("source") or tool_request.get("state") or tool_request.get("source_state_path")
        if not source:
            raise ValueError("experiment dashboard action requires source/state")
        output_path = Path(str(tool_request["output_path"])) if tool_request.get("output_path") else None
        return generate_experiment_dashboard(Path(str(source)), output_path).model_dump(mode="json")

    registry["experiment_dashboard"] = dashboard_runner
    return registry


def execute_action(
    state: AutonomousDevsimAgentState,
    request: AutonomousDevsimRequest,
    action: DevsimAgentAction,
    *,
    runner_registry: dict[str, Runner],
    repair_runner: RepairRunner = run_repair_executor,
) -> tuple[dict[str, Any], str | None]:
    if action.kind == DevsimAgentActionKind.RUN_SUPERVISOR:
        runner = runner_registry.get("supervisor")
        if runner is None:
            raise ValueError("supervisor runner is not registered")
        payload = {"goal_text": state.goal_text, "execute": True, "max_cycles": request.supervisor_max_cycles, **action.request}
        result = runner(payload)
        return result, infer_result_state_path(result)
    if action.kind == DevsimAgentActionKind.RUN_TOOL:
        if not action.tool_name:
            raise ValueError("run_tool action requires tool_name")
        if action.tool_name == "autonomous_devsim_agent":
            raise ValueError("autonomous DEVSIM agent cannot call itself as a nested tool")
        runner = runner_registry.get(action.tool_name)
        if runner is None:
            raise ValueError(f"runner is not registered: {action.tool_name}")
        result = runner(action.request)
        return result, infer_result_state_path(result)
    if action.kind == DevsimAgentActionKind.RUN_REPAIR_EXECUTOR:
        source = action.source_state_path or state.latest_state_path
        if not source:
            raise ValueError("repair action requires a source state")
        result = result_to_dict(
            repair_runner(
                Path(source),
                execute=True,
                max_rounds=request.repair_max_rounds,
                allow_user_confirmation_actions=request.allow_user_confirmation_actions,
                use_agent_policy=request.use_agent_policy,
            )
        )
        return result, infer_result_state_path(result) or result.get("final_state_path") or result.get("current_state_path")
    if action.kind == DevsimAgentActionKind.RUN_PHYSICAL_BENCHMARK:
        source = action.source_state_path or state.latest_state_path
        if not source:
            raise ValueError("benchmark action requires a source state")
        result = runner_registry["physical_benchmark"]({"source": source, **action.request})
        state.checkpoint["physical_benchmark_done"] = True
        state.checkpoint["physical_benchmark_path"] = result.get("benchmark_path")
        return result, source
    if action.kind == DevsimAgentActionKind.GENERATE_REPORT:
        source = action.source_state_path or state.latest_state_path
        if not source:
            raise ValueError("report action requires a source state")
        try:
            result = runner_registry["experiment_report"]({"source": source, **action.request})
        except Exception:
            conclusion_runner = runner_registry.get("experiment_conclusion")
            if conclusion_runner is None:
                raise
            result = conclusion_runner({"source": source, **action.request})
            result["report_fallback"] = "experiment_conclusion"
        state.final_report_path = result.get("report_path") or result.get("conclusion_path")
        state.checkpoint["report_done"] = True
        return result, source
    if action.kind == DevsimAgentActionKind.GENERATE_DASHBOARD:
        source = action.source_state_path or state.latest_state_path
        if not source:
            raise ValueError("dashboard action requires a source state")
        result = runner_registry["experiment_dashboard"]({"source": source, **action.request})
        state.final_dashboard_path = result.get("dashboard_path")
        state.checkpoint["dashboard_done"] = True
        return result, source
    if action.kind == DevsimAgentActionKind.STOP_SUCCESS:
        state.status = DevsimAgentStatus.COMPLETED
        state.final_state_path = action.source_state_path or state.latest_state_path
        state.next_action = "autonomous DEVSIM task completed"
        return {"status": "completed", "final_state_path": state.final_state_path}, state.final_state_path
    if action.kind == DevsimAgentActionKind.ASK_USER:
        state.status = DevsimAgentStatus.WAITING_FOR_USER
        state.next_action = action.reason
        return {"status": "waiting_for_user", "question": action.request.get("question") or action.reason}, state.latest_state_path
    return {"status": "skipped", "reason": action.reason}, state.latest_state_path


def append_step(
    state: AutonomousDevsimAgentState,
    action: DevsimAgentAction,
    decision: dict[str, Any],
) -> DevsimAgentStep:
    step = DevsimAgentStep(
        index=len(state.steps) + 1,
        kind=action.kind,
        status=DevsimAgentStepStatus.RUNNING if state.execute else DevsimAgentStepStatus.PLANNED,
        reason=action.reason,
        started_at=utc_timestamp(),
        action=action.model_dump(mode="json"),
        observation={
            "latest_state": observe_state(state.latest_state_path),
            "agent_decision": decision,
        },
    )
    state.steps.append(step)
    state.next_action = action.kind.value
    return step


def run_autonomous_devsim_agent(
    request: AutonomousDevsimRequest,
    *,
    runner_registry: dict[str, Runner] | None = None,
    repair_runner: RepairRunner = run_repair_executor,
    llm_client: ChatClient | None = None,
) -> AutonomousDevsimAgentState:
    state, path = prepare_state(request)
    registry = runner_registry or default_runner_registry()
    if state.status not in {DevsimAgentStatus.RUNNING, DevsimAgentStatus.PLANNED}:
        return state

    while len(state.steps) < request.max_steps and state.status in {DevsimAgentStatus.RUNNING, DevsimAgentStatus.PLANNED}:
        try:
            action, decision = decide_next_action(state, request, llm_client=llm_client)
        except Exception as exc:
            state.status = DevsimAgentStatus.FAILED
            state.failure_reason = str(exc)
            state.next_action = "inspect autonomous DEVSIM agent decision failure"
            write_state(state, path)
            return state
        state.checkpoint["last_agent_decision"] = decision
        if action.user_confirmation_required and not request.allow_user_confirmation_actions:
            state.status = DevsimAgentStatus.WAITING_FOR_USER
            state.next_action = "wait for user confirmation before executing autonomous DEVSIM action"
            state.checkpoint["blocked_action"] = action.model_dump(mode="json")
            write_state(state, path)
            return state

        step = append_step(state, action, decision)
        write_state(state, path)
        if not request.execute:
            state.status = DevsimAgentStatus.PLANNED
            state.checkpoint["planned_action"] = action.model_dump(mode="json")
            write_state(state, path)
            return state

        try:
            result, result_state_path = execute_action(
                state,
                request,
                action,
                runner_registry=registry,
                repair_runner=repair_runner,
            )
            step.result = result
            step.result_state_path = result_state_path
            if result_state_path:
                state.latest_state_path = result_state_path
            if state.status not in {DevsimAgentStatus.COMPLETED, DevsimAgentStatus.WAITING_FOR_USER}:
                state.status = DevsimAgentStatus.RUNNING
            step.status = DevsimAgentStepStatus.COMPLETED
            step.completed_at = utc_timestamp()
            state.checkpoint["completed_steps"] = len(state.steps)
            state.checkpoint["latest_state_path"] = state.latest_state_path
            write_state(state, path)
            if state.status in {DevsimAgentStatus.COMPLETED, DevsimAgentStatus.WAITING_FOR_USER}:
                return state
        except Exception as exc:
            step.status = DevsimAgentStepStatus.FAILED
            step.error = str(exc)
            step.completed_at = utc_timestamp()
            state.status = DevsimAgentStatus.FAILED
            state.failure_reason = str(exc)
            state.next_action = "inspect autonomous DEVSIM agent tool failure"
            write_state(state, path)
            return state

    if state.status == DevsimAgentStatus.RUNNING:
        state.status = DevsimAgentStatus.FAILED
        state.failure_reason = f"maximum autonomous DEVSIM steps reached: {request.max_steps}"
        state.next_action = "increase budget or inspect last agent step"
        write_state(state, path)
    return state
