from __future__ import annotations

import json
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

from pydantic import BaseModel, Field

from tcad_agent.curve_diagnostics import curve_shape_diagnostic, load_curve_rows
from tcad_agent.deck_ir import parse_devsim_deck_file, write_semantic_deck_patch_artifacts
from tcad_agent.device_templates import route_device_goal
from tcad_agent.engineering_objectives import EngineeringConstraint, EngineeringObjective
from tcad_agent.llm import LLMClient, LLMConfig
from tcad_agent.reporting import final_artifacts, final_metrics, load_final_state
from tcad_agent.repair_executor import run_repair_executor
from tcad_agent.task_planner import parse_json_object
from tcad_agent.task_spec import PROJECT_ROOT


class ChatClient(Protocol):
    config: LLMConfig

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        ...


class ToolCallClient(ChatClient, Protocol):
    def tool_call(
        self,
        system: str,
        user: str,
        tools: list[dict[str, Any]],
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        ...


Runner = Callable[[dict[str, Any]], dict[str, Any]]
RepairRunner = Callable[..., Any]


class DevsimAgentStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    WAITING_FOR_USER = "waiting_for_user"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DevsimAgentStepStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DevsimAgentActionKind(str, Enum):
    AUDIT_CAPABILITY = "audit_capability"
    RUN_SUPERVISOR = "run_supervisor"
    RUN_TOOL = "run_tool"
    RUN_REPAIR_EXECUTOR = "run_repair_executor"
    RUN_PHYSICAL_BENCHMARK = "run_physical_benchmark"
    EVALUATE_OBJECTIVES = "evaluate_objectives"
    INGEST_DECK = "ingest_deck"
    APPLY_DECK_PATCH = "apply_deck_patch"
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
    source_deck_path: str | None = None
    deck_patches: list[dict[str, Any]] = Field(default_factory=list)
    objectives: list[EngineeringObjective] = Field(default_factory=list)
    constraints: list[EngineeringConstraint] = Field(default_factory=list)
    cancel_file: Path | None = None
    heartbeat_path: Path | None = None
    use_llm: bool = True
    allow_llm_fallback: bool = True
    use_agent_policy: bool = True
    allow_user_confirmation_actions: bool = False
    supervisor_max_cycles: int = Field(default=3, ge=1)
    repair_max_rounds: int = Field(default=3, ge=1)
    generate_report: bool = True
    generate_dashboard: bool = True
    require_capability_audit: bool = False


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
    heartbeat_path: str | None = None
    cancel_file: str | None = None
    active_process: dict[str, Any] | None = None
    next_action: str | None = None
    failure_reason: str | None = None


class AgentToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    action_kind: str | None = None
    runner_name: str | None = None


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


def cancel_path_for_state(state: AutonomousDevsimAgentState) -> Path:
    return Path(state.cancel_file) if state.cancel_file else Path(state.agent_dir) / "cancel.requested"


def heartbeat_path_for_state(state: AutonomousDevsimAgentState) -> Path:
    return Path(state.heartbeat_path) if state.heartbeat_path else Path(state.agent_dir) / "heartbeat.json"


def cancel_requested(state: AutonomousDevsimAgentState) -> bool:
    return cancel_path_for_state(state).exists()


def write_heartbeat(
    state: AutonomousDevsimAgentState,
    *,
    active_action: DevsimAgentAction | None = None,
    step_index: int | None = None,
    note: str | None = None,
) -> None:
    path = heartbeat_path_for_state(state)
    payload = {
        "schema_version": "actsoft.tcad.autonomous_devsim_heartbeat.v1",
        "agent_id": state.agent_id,
        "status": state.status.value if isinstance(state.status, DevsimAgentStatus) else state.status,
        "updated_at": utc_timestamp(),
        "pid": os.getpid(),
        "step_index": step_index,
        "active_action": active_action.model_dump(mode="json") if active_action else None,
        "latest_state_path": state.latest_state_path,
        "note": note,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def create_initial_state(request: AutonomousDevsimRequest, agent_id: str, agent_dir: Path) -> AutonomousDevsimAgentState:
    now = utc_timestamp()
    cancel_file = request.cancel_file or agent_dir / "cancel.requested"
    heartbeat_path = request.heartbeat_path or agent_dir / "heartbeat.json"
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
        heartbeat_path=str(heartbeat_path),
        cancel_file=str(cancel_file),
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
        if request.cancel_file:
            state.cancel_file = str(request.cancel_file)
        if request.heartbeat_path:
            state.heartbeat_path = str(request.heartbeat_path)
        state.checkpoint["agent_state_path"] = str(actual_path)
        write_state(state, actual_path)
        return state, actual_path
    agent_dir = request.agent_root / actual_id
    agent_dir.mkdir(parents=True, exist_ok=True)
    state = create_initial_state(request, actual_id, agent_dir)
    state.checkpoint["agent_state_path"] = str(actual_path)
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


def safe_path(value: Any) -> Path | None:
    if not value:
        return None
    try:
        path = Path(str(value))
    except TypeError:
        return None
    return path if path.exists() else None


def read_text_tail(path: Path | None, *, max_chars: int = 2400) -> dict[str, Any] | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return {
        "path": str(path),
        "chars": len(text),
        "tail": text[-max_chars:],
    }


def merge_artifacts(state_data: dict[str, Any]) -> dict[str, str]:
    artifacts = dict(final_artifacts(state_data))
    for key, value in (state_data.get("artifacts") or {}).items():
        if value:
            artifacts.setdefault(str(key), str(value))
    for key in [
        "log_path",
        "stdout_path",
        "stderr_path",
        "semantic_deck_diff",
        "patched_source_deck",
        "tcad_deck_ir",
        "deck_patch_history",
    ]:
        value = state_data.get(key)
        if value:
            artifacts.setdefault(key, str(value))
    return artifacts


def curve_observations(artifacts: dict[str, str], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    threshold = metrics.get("breakdown_current_threshold_a") or metrics.get("target_current_a")
    for key, value in artifacts.items():
        lowered = key.lower()
        if not (lowered.endswith("csv") or "csv" in lowered or str(value).lower().endswith(".csv")):
            continue
        rows = load_curve_rows(value)
        if not rows:
            continue
        shape = curve_shape_diagnostic(rows, threshold_y=threshold)
        observations.append(
            {
                "artifact": key,
                "path": value,
                "shape": shape.model_dump(mode="json"),
            }
        )
    return observations[:6]


def artifact_observations(state_data: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    artifacts = merge_artifacts(state_data)
    log_keys = [key for key in artifacts if any(token in key.lower() for token in ["log", "stdout", "stderr"])]
    diff_keys = [key for key in artifacts if "diff" in key.lower() or str(artifacts[key]).lower().endswith(".diff")]
    deck_keys = [key for key in artifacts if any(token in key.lower() for token in ["deck", "ir", "patch"])]
    return {
        "curve_shapes": curve_observations(artifacts, metrics),
        "log_tails": {
            key: preview
            for key in log_keys[:4]
            if (preview := read_text_tail(safe_path(artifacts.get(key)), max_chars=1800))
        },
        "deck_diffs": {
            key: preview
            for key in diff_keys[:4]
            if (preview := read_text_tail(safe_path(artifacts.get(key)), max_chars=2200))
        },
        "deck_artifacts": {key: artifacts[key] for key in deck_keys[:8]},
    }


def observe_state(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {"state_path": None, "summary": "no TCAD state has been produced yet"}
    state_data = load_final_state(path_value)
    if not state_data:
        return {"state_path": path_value, "summary": "state file is missing or not JSON-readable"}
    quality = state_data.get("quality_report") or state_data.get("final_quality_report") or {}
    benchmark = state_data.get("benchmark_context") or {}
    metrics = final_metrics(state_data)
    artifacts = merge_artifacts(state_data)
    return {
        "state_path": path_value,
        "tool_name": state_data.get("tool_name"),
        "status": state_data.get("status"),
        "quality_status": quality.get("status"),
        "issue_codes": issue_codes_from_state(state_data),
        "metrics": metrics,
        "artifacts": artifacts,
        "artifact_observations": artifact_observations(state_data, metrics),
        "benchmark_context": benchmark,
        "repair_context": state_data.get("repair_context"),
        "mutation_effect_analysis": state_data.get("mutation_effect_analysis"),
    }


def dashboard_supported(path_value: str | None) -> bool:
    if not path_value:
        return False
    state_data = load_final_state(path_value)
    return bool(
        state_data
        and state_data.get("tool_name")
        in {"parameter_sweep", "adaptive_optimizer", "multidim_optimizer", "autonomous_devsim_agent"}
    )


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


def safe_tool_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)


def runner_tool_call_name(tool_name: str) -> str:
    return f"run_tool__{safe_tool_name(tool_name)}"


def object_schema(properties: dict[str, Any] | None = None, *, required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
        "additionalProperties": True,
    }


def build_agent_tool_specs(runner_registry: dict[str, Runner] | None = None) -> list[AgentToolSpec]:
    specs = [
        AgentToolSpec(
            name=DevsimAgentActionKind.AUDIT_CAPABILITY.value,
            action_kind=DevsimAgentActionKind.AUDIT_CAPABILITY.value,
            description="Audit whether the goal maps to executable, compact-baseline, or planned device coverage.",
            parameters=object_schema({"goal_text": {"type": "string"}}),
        ),
        AgentToolSpec(
            name=DevsimAgentActionKind.RUN_SUPERVISOR.value,
            action_kind=DevsimAgentActionKind.RUN_SUPERVISOR.value,
            description="Route a natural-language TCAD goal through the supervisor.",
            parameters=object_schema({"goal_text": {"type": "string"}, "max_cycles": {"type": "integer"}}),
        ),
        AgentToolSpec(
            name=DevsimAgentActionKind.RUN_REPAIR_EXECUTOR.value,
            action_kind=DevsimAgentActionKind.RUN_REPAIR_EXECUTOR.value,
            description="Repair a failed or suspicious TCAD state with the repair executor.",
            parameters=object_schema({"source_state_path": {"type": "string"}}),
        ),
        AgentToolSpec(
            name=DevsimAgentActionKind.RUN_PHYSICAL_BENCHMARK.value,
            action_kind=DevsimAgentActionKind.RUN_PHYSICAL_BENCHMARK.value,
            description="Run physical credibility benchmark checks on a state.",
            parameters=object_schema({"source_state_path": {"type": "string"}}),
        ),
        AgentToolSpec(
            name=DevsimAgentActionKind.EVALUATE_OBJECTIVES.value,
            action_kind=DevsimAgentActionKind.EVALUATE_OBJECTIVES.value,
            description="Evaluate objectives, constraints, Pareto front, and tradeoffs.",
            parameters=object_schema({"source_state_path": {"type": "string"}, "objectives": {"type": "array"}, "constraints": {"type": "array"}}),
        ),
        AgentToolSpec(
            name=DevsimAgentActionKind.INGEST_DECK.value,
            action_kind=DevsimAgentActionKind.INGEST_DECK.value,
            description="Parse a user-provided DEVSIM Python deck into source IR.",
            parameters=object_schema({"source_deck_path": {"type": "string"}}, required=["source_deck_path"]),
        ),
        AgentToolSpec(
            name=DevsimAgentActionKind.APPLY_DECK_PATCH.value,
            action_kind=DevsimAgentActionKind.APPLY_DECK_PATCH.value,
            description="Apply semantic patches to a user deck and emit patched source plus diff.",
            parameters=object_schema({"source_deck_path": {"type": "string"}, "deck_patches": {"type": "array"}}, required=["source_deck_path", "deck_patches"]),
        ),
        AgentToolSpec(
            name=DevsimAgentActionKind.GENERATE_REPORT.value,
            action_kind=DevsimAgentActionKind.GENERATE_REPORT.value,
            description="Generate an engineer-readable Markdown report or conclusion.",
            parameters=object_schema({"source_state_path": {"type": "string"}}),
        ),
        AgentToolSpec(
            name=DevsimAgentActionKind.GENERATE_DASHBOARD.value,
            action_kind=DevsimAgentActionKind.GENERATE_DASHBOARD.value,
            description="Generate an HTML dashboard for sweep/optimization/autonomous timeline states.",
            parameters=object_schema({"source_state_path": {"type": "string"}}),
        ),
        AgentToolSpec(
            name=DevsimAgentActionKind.STOP_SUCCESS.value,
            action_kind=DevsimAgentActionKind.STOP_SUCCESS.value,
            description="Stop after sufficient evidence exists.",
            parameters=object_schema({"source_state_path": {"type": "string"}}),
        ),
        AgentToolSpec(
            name=DevsimAgentActionKind.ASK_USER.value,
            action_kind=DevsimAgentActionKind.ASK_USER.value,
            description="Pause for user confirmation before a high-risk action.",
            parameters=object_schema({"question": {"type": "string"}}, required=["question"]),
        ),
    ]
    for tool_name in sorted((runner_registry or {}).keys()):
        if tool_name == "autonomous_devsim_agent":
            continue
        specs.append(
            AgentToolSpec(
                name=runner_tool_call_name(tool_name),
                action_kind=DevsimAgentActionKind.RUN_TOOL.value,
                runner_name=tool_name,
                description=f"Run registered TCAD tool `{tool_name}` with its request object.",
                parameters=object_schema(),
            )
        )
    return specs


def openai_tool_specs(specs: list[AgentToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in specs
    ]


def toolbelt_summary(specs: list[AgentToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "kind": spec.action_kind,
            "runner_name": spec.runner_name,
            "description": spec.description,
        }
        for spec in specs
    ]


def build_agent_context(
    state: AutonomousDevsimAgentState,
    request: AutonomousDevsimRequest,
    *,
    tool_specs: list[AgentToolSpec] | None = None,
) -> dict[str, Any]:
    specs = tool_specs or build_agent_tool_specs()
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
        "source_deck_path": request.source_deck_path,
        "deck_patches": request.deck_patches,
        "objectives": [item.model_dump(mode="json") for item in request.objectives],
        "constraints": [item.model_dump(mode="json") for item in request.constraints],
        "require_capability_audit": request.require_capability_audit,
        "toolbelt": toolbelt_summary(specs),
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
    if request.require_capability_audit and not state.checkpoint.get("capability_audit_done"):
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.AUDIT_CAPABILITY,
            request={"goal_text": request.goal_text},
            reason="Audit device template coverage before treating the goal as executable TCAD work.",
        )
    if request.source_deck_path and not state.checkpoint.get("deck_ingested"):
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.INGEST_DECK,
            request={"source_deck_path": request.source_deck_path},
            reason="A user DEVSIM deck was provided; parse it into source IR before running or patching.",
        )
    if request.source_deck_path and request.deck_patches and not state.checkpoint.get("deck_patch_done"):
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.APPLY_DECK_PATCH,
            request={"source_deck_path": request.source_deck_path, "deck_patches": request.deck_patches},
            reason="Apply requested semantic deck patches and emit a diff before executing the patched workflow.",
            user_confirmation_required=True,
        )
    if request.initial_tool_name and not state.checkpoint.get("initial_tool_done"):
        tool_request = dict(request.initial_request)
        if state.checkpoint.get("patched_source_deck"):
            tool_request.setdefault("source_deck_path", state.checkpoint["patched_source_deck"])
        elif request.source_deck_path:
            tool_request.setdefault("source_deck_path", request.source_deck_path)
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.RUN_TOOL,
            tool_name=request.initial_tool_name,
            request=tool_request,
            reason="Run the requested initial DEVSIM tool after any deck ingest or semantic patch setup.",
        )
    if not state.latest_state_path:
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
    if (request.objectives or request.constraints) and not state.checkpoint.get("objectives_done"):
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.EVALUATE_OBJECTIVES,
            source_state_path=state.latest_state_path,
            request={
                "objectives": [item.model_dump(mode="json") for item in request.objectives],
                "constraints": [item.model_dump(mode="json") for item in request.constraints],
            },
            reason="Evaluate multi-objective tradeoffs and constraints before deciding whether the run is worth continuing.",
        )
    if request.generate_report and not state.checkpoint.get("report_done"):
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.GENERATE_REPORT,
            source_state_path=state.latest_state_path,
            reason="Physical benchmark is done; generate an engineer-readable report.",
        )
    dashboard_source = state.latest_state_path if dashboard_supported(state.latest_state_path) else state.checkpoint.get("agent_state_path")
    if request.generate_dashboard and not state.checkpoint.get("dashboard_done") and dashboard_source:
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.GENERATE_DASHBOARD,
            source_state_path=str(dashboard_source),
            reason="Generate an HTML dashboard for curve, lineage, and autonomous timeline inspection.",
        )
    return DevsimAgentAction(
        kind=DevsimAgentActionKind.STOP_SUCCESS,
        source_state_path=state.latest_state_path,
        reason="The autonomous DEVSIM loop has produced evidence, benchmark, and requested artifacts.",
    )


def normalize_tool_call_payload(parsed: dict[str, Any]) -> dict[str, Any] | None:
    raw_tool = parsed.get("tool_call")
    if not isinstance(raw_tool, dict):
        tool_calls = parsed.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls and isinstance(tool_calls[0], dict):
            raw_tool = tool_calls[0]
    if not isinstance(raw_tool, dict):
        return None
    name = str(raw_tool.get("name") or raw_tool.get("function") or "")
    args = raw_tool.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    args = args if isinstance(args, dict) else {}
    if name.startswith("run_tool__"):
        return {
            "kind": DevsimAgentActionKind.RUN_TOOL.value,
            "tool_name": name.removeprefix("run_tool__"),
            "request": args,
            "reason": str(parsed.get("hypothesis_zh") or parsed.get("observation_summary") or f"Agent selected native tool call {name}."),
        }
    try:
        kind = DevsimAgentActionKind(name)
    except Exception:
        return None
    request = dict(args)
    return {
        "kind": kind.value,
        "request": request,
        "source_state_path": request.pop("source_state_path", None),
        "reason": str(parsed.get("hypothesis_zh") or parsed.get("observation_summary") or f"Agent selected native action {kind.value}."),
        "user_confirmation_required": bool(request.pop("user_confirmation_required", False)),
    }


def normalize_agent_action(parsed: dict[str, Any], state: AutonomousDevsimAgentState) -> DevsimAgentAction | None:
    raw = normalize_tool_call_payload(parsed)
    if raw is None:
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
    tool_specs: list[AgentToolSpec] | None = None,
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
    specs = tool_specs or build_agent_tool_specs()
    context = build_agent_context(state, request, tool_specs=specs)
    system, user = build_agent_messages(context)
    try:
        if hasattr(chat_client, "tool_call"):
            tool_result = getattr(chat_client, "tool_call")(
                system=system,
                user=user,
                tools=openai_tool_specs(specs),
                temperature=0.2,
            )
            raw = json.dumps(tool_result, ensure_ascii=False)
        else:
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


def process_metadata_from_result(result: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {"pid": os.getpid()}
    for key in ["pid", "process_id", "command", "returncode", "stdout_tail", "stderr_tail"]:
        if key in result:
            metadata[key] = result[key]
    return metadata


def deck_session_state_path(state: AutonomousDevsimAgentState, name: str) -> Path:
    return Path(state.agent_dir) / "deck_session" / name


def write_deck_ingest_state(state: AutonomousDevsimAgentState, source_path: Path) -> dict[str, Any]:
    ir = parse_devsim_deck_file(source_path)
    output_dir = deck_session_state_path(state, "ingest")
    output_dir.mkdir(parents=True, exist_ok=True)
    ir_path = output_dir / f"{source_path.stem}.deck_ir.json"
    session_state_path = output_dir / "deck_session_state.json"
    ir_path.write_text(json.dumps(ir.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    session_state = {
        "tool_name": "deck_session",
        "status": "completed",
        "run_id": f"{state.agent_id}_deck_ingest",
        "source_deck_path": str(source_path.resolve()),
        "tcad_deck_ir": str(ir_path.resolve()),
        "final_summary": {
            "artifacts": {
                "source_deck": str(source_path.resolve()),
                "tcad_deck_ir": str(ir_path.resolve()),
            },
            "metrics": {
                "deck_sections": len(ir.sections),
                "deck_assignments": len(ir.assignments),
                "deck_calls": len(ir.calls),
            },
        },
        "quality_report": {
            "status": "passed" if not ir.parse_warnings else "suspicious",
            "issues": [{"code": warning, "severity": "warning"} for warning in ir.parse_warnings],
            "metrics": {
                "deck_sections": len(ir.sections),
                "deck_assignments": len(ir.assignments),
                "deck_calls": len(ir.calls),
            },
        },
    }
    write_json(session_state_path, session_state)
    return {
        "status": "completed",
        "state_path": str(session_state_path.resolve()),
        "source_deck_path": str(source_path.resolve()),
        "tcad_deck_ir": str(ir_path.resolve()),
        "section_names": [section.name for section in ir.sections],
        "parse_warnings": ir.parse_warnings,
    }


def write_deck_patch_state(state: AutonomousDevsimAgentState, source_path: Path, patches: list[dict[str, Any]]) -> dict[str, Any]:
    output_dir = deck_session_state_path(state, "semantic_patch")
    result = write_semantic_deck_patch_artifacts(source_path, patches, output_dir)
    session_state_path = output_dir / "deck_session_state.json"
    session_state = {
        "tool_name": "deck_session",
        "status": "completed",
        "run_id": f"{state.agent_id}_deck_patch",
        "source_deck_path": str(source_path.resolve()),
        "patched_source_deck": result.patched_source_path,
        "semantic_deck_diff": result.diff_path,
        "tcad_deck_ir": result.ir_path,
        "deck_patch_history": patches,
        "final_summary": {
            "artifacts": {
                "source_deck": str(source_path.resolve()),
                "patched_source_deck": result.patched_source_path,
                "semantic_deck_diff": result.diff_path,
                "tcad_deck_ir": result.ir_path,
            },
            "metrics": {
                "deck_patches_applied": len(result.applied_patches),
                "deck_patches_unapplied": len(result.unapplied_patches),
            },
        },
        "quality_report": {
            "status": "passed" if not result.unapplied_patches else "suspicious",
            "issues": [{"code": "deck_patch_fallback_append", "severity": "warning"} for _ in result.unapplied_patches],
            "metrics": {
                "deck_patches_applied": len(result.applied_patches),
                "deck_patches_unapplied": len(result.unapplied_patches),
            },
        },
    }
    write_json(session_state_path, session_state)
    return {
        "status": "completed",
        "state_path": str(session_state_path.resolve()),
        "source_deck_path": str(source_path.resolve()),
        "patched_source_deck": result.patched_source_path,
        "semantic_deck_diff": result.diff_path,
        "tcad_deck_ir": result.ir_path,
        "applied_patches": result.applied_patches,
        "unapplied_patches": result.unapplied_patches,
    }


def execute_action(
    state: AutonomousDevsimAgentState,
    request: AutonomousDevsimRequest,
    action: DevsimAgentAction,
    *,
    runner_registry: dict[str, Runner],
    repair_runner: RepairRunner = run_repair_executor,
) -> tuple[dict[str, Any], str | None]:
    if action.kind == DevsimAgentActionKind.AUDIT_CAPABILITY:
        route = route_device_goal(str(action.request.get("goal_text") or state.goal_text))
        result = {
            "status": "completed",
            "tool_name": "capability_audit",
            "route": route.model_dump(mode="json"),
            "executable": route.executable,
            "runnable": route.runnable,
            "signoff_ready": route.signoff_ready,
            "capability_warnings": route.capability_warnings,
        }
        audit_path = Path(state.agent_dir) / "capability_audit.json"
        write_json(audit_path, result)
        state.checkpoint["capability_audit_done"] = True
        state.checkpoint["capability_audit_path"] = str(audit_path.resolve())
        state.checkpoint["capability_audit"] = result
        if route.template:
            state.checkpoint["coverage_work_package"] = {
                "template_id": route.template.template_id,
                "support": route.template.support.value,
                "tcad_fidelity": route.template.tcad_fidelity,
                "signoff_ready": route.signoff_ready,
                "executable_tool": route.template.executable_tool,
                "recommended_convergence": route.template.recommended_convergence,
                "signoff_workflow": route.template.signoff_workflow,
                "next_implementation_steps": route.template.next_implementation_steps,
                "missing_capabilities": route.template.missing_capabilities,
            }
        if route.template and not route.executable:
            state.checkpoint["runner_first_work_package"] = {
                "template_id": route.template.template_id,
                "support": route.template.support.value,
                "missing_capabilities": route.template.missing_capabilities,
                "next_implementation_steps": route.template.next_implementation_steps,
            }
        return result, None
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
    if action.kind == DevsimAgentActionKind.EVALUATE_OBJECTIVES:
        source = action.source_state_path or state.latest_state_path
        if not source:
            raise ValueError("objective evaluation action requires a source state")
        result = runner_registry["engineering_objectives"](
            {
                "source": source,
                "objectives": action.request.get("objectives") or [item.model_dump(mode="json") for item in request.objectives],
                "constraints": action.request.get("constraints") or [item.model_dump(mode="json") for item in request.constraints],
            }
        )
        state.checkpoint["objectives_done"] = True
        state.checkpoint["engineering_objectives_path"] = result.get("output_path")
        state.checkpoint["pareto_front"] = result.get("pareto_front") or []
        state.checkpoint["best_candidate"] = result.get("best_candidate")
        return result, source
    if action.kind == DevsimAgentActionKind.INGEST_DECK:
        raw_source = action.request.get("source_deck_path") or request.source_deck_path
        if not raw_source:
            raise ValueError("deck ingest action requires source_deck_path")
        result = write_deck_ingest_state(state, Path(str(raw_source)))
        state.checkpoint["deck_ingested"] = True
        state.checkpoint["source_deck_path"] = result.get("source_deck_path")
        state.checkpoint["tcad_deck_ir"] = result.get("tcad_deck_ir")
        return result, result.get("state_path")
    if action.kind == DevsimAgentActionKind.APPLY_DECK_PATCH:
        raw_source = action.request.get("source_deck_path") or request.source_deck_path
        patches = action.request.get("deck_patches") or request.deck_patches
        if not raw_source:
            raise ValueError("deck patch action requires source_deck_path")
        if not isinstance(patches, list) or not patches:
            raise ValueError("deck patch action requires non-empty deck_patches")
        result = write_deck_patch_state(state, Path(str(raw_source)), [item for item in patches if isinstance(item, dict)])
        state.checkpoint["deck_patch_done"] = True
        state.checkpoint["patched_source_deck"] = result.get("patched_source_deck")
        state.checkpoint["semantic_deck_diff"] = result.get("semantic_deck_diff")
        return result, result.get("state_path")
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
        return result, None
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
    tool_specs = build_agent_tool_specs(registry)
    if state.status not in {DevsimAgentStatus.RUNNING, DevsimAgentStatus.PLANNED}:
        return state
    write_heartbeat(state, note="agent prepared")

    while len(state.steps) < request.max_steps and state.status in {DevsimAgentStatus.RUNNING, DevsimAgentStatus.PLANNED}:
        if cancel_requested(state):
            state.status = DevsimAgentStatus.CANCELLED
            state.failure_reason = "cancel requested by control file"
            state.next_action = "cancelled before starting next autonomous step"
            write_state(state, path)
            write_heartbeat(state, note="cancelled before next step")
            return state
        try:
            action, decision = decide_next_action(state, request, llm_client=llm_client, tool_specs=tool_specs)
        except Exception as exc:
            state.status = DevsimAgentStatus.FAILED
            state.failure_reason = str(exc)
            state.next_action = "inspect autonomous DEVSIM agent decision failure"
            write_state(state, path)
            write_heartbeat(state, note="decision failure")
            return state
        state.checkpoint["last_agent_decision"] = decision
        if action.user_confirmation_required and not request.allow_user_confirmation_actions:
            state.status = DevsimAgentStatus.WAITING_FOR_USER
            state.next_action = "wait for user confirmation before executing autonomous DEVSIM action"
            state.checkpoint["blocked_action"] = action.model_dump(mode="json")
            write_state(state, path)
            write_heartbeat(state, active_action=action, note="waiting for user confirmation")
            return state

        step = append_step(state, action, decision)
        state.active_process = {"pid": os.getpid(), "step_index": step.index, "action": action.model_dump(mode="json"), "started_at": step.started_at}
        write_state(state, path)
        write_heartbeat(state, active_action=action, step_index=step.index, note="step started")
        if not request.execute:
            state.status = DevsimAgentStatus.PLANNED
            state.checkpoint["planned_action"] = action.model_dump(mode="json")
            state.active_process = None
            write_state(state, path)
            write_heartbeat(state, active_action=action, step_index=step.index, note="planned action recorded")
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
            state.checkpoint["last_process"] = process_metadata_from_result(result)
            step.result_state_path = result_state_path
            if result_state_path:
                state.latest_state_path = result_state_path
            if action.kind == DevsimAgentActionKind.RUN_TOOL and action.tool_name == request.initial_tool_name:
                state.checkpoint["initial_tool_done"] = True
            if state.status not in {DevsimAgentStatus.COMPLETED, DevsimAgentStatus.WAITING_FOR_USER}:
                state.status = DevsimAgentStatus.RUNNING
            step.status = DevsimAgentStepStatus.COMPLETED
            step.completed_at = utc_timestamp()
            state.checkpoint["completed_steps"] = len(state.steps)
            state.checkpoint["latest_state_path"] = state.latest_state_path
            state.active_process = None
            write_state(state, path)
            write_heartbeat(state, active_action=action, step_index=step.index, note="step completed")
            if state.status in {DevsimAgentStatus.COMPLETED, DevsimAgentStatus.WAITING_FOR_USER}:
                return state
            if cancel_requested(state):
                state.status = DevsimAgentStatus.CANCELLED
                state.failure_reason = "cancel requested after autonomous step completed"
                state.next_action = "cancelled after step boundary"
                state.active_process = None
                write_state(state, path)
                write_heartbeat(state, note="cancelled after step")
                return state
        except Exception as exc:
            step.status = DevsimAgentStepStatus.FAILED
            step.error = str(exc)
            step.completed_at = utc_timestamp()
            state.status = DevsimAgentStatus.FAILED
            state.failure_reason = str(exc)
            state.next_action = "inspect autonomous DEVSIM agent tool failure"
            state.active_process = None
            write_state(state, path)
            write_heartbeat(state, active_action=action, step_index=step.index, note="step failed")
            return state

    if state.status == DevsimAgentStatus.RUNNING:
        state.status = DevsimAgentStatus.FAILED
        state.failure_reason = f"maximum autonomous DEVSIM steps reached: {request.max_steps}"
        state.next_action = "increase budget or inspect last agent step"
        state.active_process = None
        write_state(state, path)
        write_heartbeat(state, note="max steps reached")
    return state
