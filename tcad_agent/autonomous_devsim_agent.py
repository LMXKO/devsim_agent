from __future__ import annotations

import argparse

import json
import os
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

from pydantic import BaseModel, Field

from tcad_agent.curve_diagnostics import compare_state_mutation_effect, curve_shape_diagnostic, load_curve_rows
from tcad_agent.deck_ir import parse_devsim_deck_file, write_semantic_deck_patch_artifacts
from tcad_agent.device_templates import route_device_goal
from tcad_agent.engineering_objectives import EngineeringConstraint, EngineeringObjective
from tcad_agent.agent_experiment_design import build_agent_experiment_design_plan
from tcad_agent.agent_guidance_patch import build_guidance_patch_plan, guidance_is_actionable_patch
from tcad_agent.evidence_lookup import PublicEvidenceLookupRequest, run_public_evidence_lookup
from tcad_agent.industrial_runner_registry import industrial_runner_coverage_matrix, runner_descriptors_for_template
from tcad_agent.industrial_runner_promotion import build_industrial_runner_promotion_plan
from tcad_agent.llm import LLMClient, LLMConfig
from tcad_agent.mutation_refinement import build_mutation_refinement_plan
from tcad_agent.public_sources import build_public_evidence_dossier
from tcad_agent.reporting import final_artifacts, final_metrics, load_final_state
from tcad_agent.repair_executor import run_repair_executor
from tcad_agent.sentaurus_lineage import SentaurusLineageArchiveRequest, build_sentaurus_lineage_archive
from tcad_agent.sentaurus_mutation_effect import SentaurusMutationEffectRequest, analyze_sentaurus_mutation_effect
from tcad_agent.sentaurus_patch_refiner import SentaurusPatchRefinerRequest, build_sentaurus_patch_refinement_plan
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


@contextmanager
def temporary_cancel_env(cancel_file: str | None):
    previous = os.environ.get("ACTSOFT_CANCEL_FILE")
    if cancel_file:
        os.environ["ACTSOFT_CANCEL_FILE"] = cancel_file
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("ACTSOFT_CANCEL_FILE", None)
        else:
            os.environ["ACTSOFT_CANCEL_FILE"] = previous


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
    RUN_USER_DECK = "run_user_deck"
    PLAN_MUTATION_REFINEMENT = "plan_mutation_refinement"
    PLAN_GUIDANCE_PATCH = "plan_guidance_patch"
    PLAN_SENTAURUS_PATCH = "plan_sentaurus_patch"
    PLAN_SENTAURUS_REFINEMENT = "plan_sentaurus_refinement"
    PLAN_EXPERIMENT_DESIGN = "plan_experiment_design"
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
    allow_unverified_deck_patch_execution: bool = False
    sentaurus_project_path: Path | None = None
    sentaurus_profile_path: Path | None = None
    sentaurus_request: dict[str, Any] = Field(default_factory=dict)
    objectives: list[EngineeringObjective] = Field(default_factory=list)
    constraints: list[EngineeringConstraint] = Field(default_factory=list)
    cancel_file: Path | None = None
    heartbeat_path: Path | None = None
    use_llm: bool = True
    allow_llm_fallback: bool = True
    use_agent_policy: bool = True
    enable_live_evidence_lookup: bool = False
    live_evidence_max_sources: int = Field(default=6, ge=1, le=24)
    allow_live_evidence_gaps: bool = False
    allow_user_confirmation_actions: bool = False
    supervisor_max_cycles: int = Field(default=3, ge=1)
    repair_max_rounds: int = Field(default=3, ge=1)
    max_mutation_refinements: int = Field(default=1, ge=0)
    auto_execute_mutation_refinements: bool = True
    enable_experiment_design: bool = False
    max_experiment_design_rounds: int = Field(default=1, ge=0)
    auto_execute_experiment_design: bool = True
    generate_report: bool = True
    generate_dashboard: bool = True
    require_capability_audit: bool = False
    mission_spec: dict[str, Any] = Field(default_factory=dict)
    agent_memory_context: list[dict[str, Any]] = Field(default_factory=list)
    curve_guidance: dict[str, Any] = Field(default_factory=dict)
    recovery_context: list[dict[str, Any]] = Field(default_factory=list)


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
    route = route_device_goal(request.goal_text)
    template_ids = [route.template.template_id] if route.template else []
    simulator = "sentaurus" if request.sentaurus_project_path else "devsim"
    live_lookup_result: dict[str, Any] | None = None
    if request.enable_live_evidence_lookup:
        live_lookup = run_public_evidence_lookup(
            PublicEvidenceLookupRequest(
                goal_text=request.goal_text,
                simulator=simulator,
                template_ids=template_ids,
                live=True,
                max_sources=request.live_evidence_max_sources,
                output_path=agent_dir / "public_evidence_lookup.json",
            )
        )
        live_lookup_result = live_lookup.model_dump(mode="json")
    public_evidence = build_public_evidence_dossier(
        request.goal_text,
        simulator=simulator,
        template_ids=template_ids,
        live_lookup_result=live_lookup_result,
    ).model_dump(mode="json")
    live_lookup_gate = (
        live_lookup_result.get("evidence_gate")
        if isinstance(live_lookup_result, dict) and isinstance(live_lookup_result.get("evidence_gate"), dict)
        else None
    )
    live_lookup_passed = bool(live_lookup_gate.get("passed")) if isinstance(live_lookup_gate, dict) else None
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
            "agent_control": {
                "mode": "agent_first" if request.use_llm else "deterministic_guardrail",
                "llm_selects_tools": request.use_llm,
                "deterministic_policy_role": "fallback_guardrail" if request.use_llm else "primary_controller",
                "decision_ledger_enabled": True,
            },
            "agent_first_policy": {
                "enabled": request.use_llm,
                "deterministic_fallback": request.allow_llm_fallback,
                "repair_agent_policy": request.use_agent_policy,
            },
            "public_evidence_gate_done": True,
            "public_evidence_dossier": public_evidence,
            "public_evidence_lookup": live_lookup_result,
            "public_evidence_lookup_gate_passed": live_lookup_passed,
            "mission_spec": request.mission_spec,
            "agent_memory_context": request.agent_memory_context,
            "curve_guidance": request.curve_guidance,
            "recovery_context": request.recovery_context,
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
        if request.mission_spec:
            state.checkpoint["mission_spec"] = request.mission_spec
        if request.agent_memory_context:
            state.checkpoint["agent_memory_context"] = request.agent_memory_context
        if request.curve_guidance:
            state.checkpoint["curve_guidance"] = request.curve_guidance
        if request.recovery_context:
            state.checkpoint["recovery_context"] = request.recovery_context
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
        for name in [
            "state.json",
            "supervisor_state.json",
            "mission_state.json",
            "sentaurus_state.json",
            "sweep_state.json",
            "optimization_state.json",
        ]:
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
        "sentaurus_mutation_effect_analysis": state_data.get("sentaurus_mutation_effect_analysis"),
        "sentaurus_lineage_archive": state_data.get("sentaurus_lineage_archive"),
    }


SIGNOFF_GAP_WARNING_CODES = {
    "power_mos_2d_layout_signoff_gaps",
    "power_mosfet_signoff_missing_evidence",
    "compact_baseline_not_signoff_evidence",
    "deck_signoff_convergence_evidence_missing",
    "deck_signoff_golden_evidence_missing",
    "tool_convergence_evidence_missing",
    "golden_or_measured_correlation_missing",
}


def state_needs_repair_before_signoff_planning(observation: dict[str, Any]) -> bool:
    quality_status = str(observation.get("quality_status") or "").lower()
    if quality_status == "failed":
        return True
    if quality_status != "suspicious":
        return False
    issue_codes = {str(code) for code in observation.get("issue_codes") or [] if code}
    metrics = observation.get("metrics") if isinstance(observation.get("metrics"), dict) else {}
    has_signoff_gap = bool(metrics.get("signoff_gaps")) or any(
        token in code for code in issue_codes for token in ["signoff", "golden", "convergence"]
    )
    if has_signoff_gap and (not issue_codes or issue_codes.issubset(SIGNOFF_GAP_WARNING_CODES)):
        return False
    return True


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
            name=DevsimAgentActionKind.RUN_USER_DECK.value,
            action_kind=DevsimAgentActionKind.RUN_USER_DECK.value,
            description="Execute a user-provided or patched DEVSIM Python deck directly.",
            parameters=object_schema({"deck_path": {"type": "string"}, "timeout_seconds": {"type": "number"}}, required=["deck_path"]),
        ),
        AgentToolSpec(
            name=DevsimAgentActionKind.PLAN_MUTATION_REFINEMENT.value,
            action_kind=DevsimAgentActionKind.PLAN_MUTATION_REFINEMENT.value,
            description="Turn baseline-vs-mutation curve diagnostics into the next finer deck/request patch.",
            parameters=object_schema({"source_state_path": {"type": "string"}}, required=["source_state_path"]),
        ),
        AgentToolSpec(
            name=DevsimAgentActionKind.PLAN_GUIDANCE_PATCH.value,
            action_kind=DevsimAgentActionKind.PLAN_GUIDANCE_PATCH.value,
            description="Turn curve-guidance next_patch_hint into an executable deck/request patch before the next run.",
            parameters=object_schema({"source_state_path": {"type": "string"}, "curve_guidance": {"type": "object"}}, required=["source_state_path"]),
        ),
        AgentToolSpec(
            name=DevsimAgentActionKind.PLAN_EXPERIMENT_DESIGN.value,
            action_kind=DevsimAgentActionKind.PLAN_EXPERIMENT_DESIGN.value,
            description="Generate ranked next experiments from benchmark/signoff gaps, curve diagnostics, and deck mutations.",
            parameters=object_schema({"source_state_path": {"type": "string"}}, required=["source_state_path"]),
        ),
        AgentToolSpec(
            name=DevsimAgentActionKind.GENERATE_REPORT.value,
            action_kind=DevsimAgentActionKind.GENERATE_REPORT.value,
            description="Generate an engineer-readable Markdown report or conclusion.",
            parameters=object_schema({"source_state_path": {"type": "string"}}),
        ),
        AgentToolSpec(
            name=DevsimAgentActionKind.PLAN_SENTAURUS_PATCH.value,
            action_kind=DevsimAgentActionKind.PLAN_SENTAURUS_PATCH.value,
            description="Plan verified Sentaurus semantic deck patches from the latest Sentaurus state and natural-language goal.",
            parameters=object_schema({"source_state_path": {"type": "string"}}),
        ),
        AgentToolSpec(
            name=DevsimAgentActionKind.PLAN_SENTAURUS_REFINEMENT.value,
            action_kind=DevsimAgentActionKind.PLAN_SENTAURUS_REFINEMENT.value,
            description="Use Sentaurus baseline-vs-mutation curve evidence to refine the same patch direction or switch to a better verified target.",
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
        "agent_hypothesis_tree": state.checkpoint.get("agent_hypothesis_tree"),
        "checkpoint": state.checkpoint,
        "public_evidence_dossier": state.checkpoint.get("public_evidence_dossier"),
        "industrial_runner_registry": industrial_runner_coverage_matrix(),
        "recent_steps": compact_steps(state),
        "initial_tool_name": request.initial_tool_name,
        "initial_request": request.initial_request,
        "source_state_path": request.source_state_path,
        "source_deck_path": request.source_deck_path,
        "deck_patches": request.deck_patches,
        "allow_unverified_deck_patch_execution": request.allow_unverified_deck_patch_execution,
        "sentaurus_project_path": str(request.sentaurus_project_path) if request.sentaurus_project_path else None,
        "sentaurus_profile_path": str(request.sentaurus_profile_path) if request.sentaurus_profile_path else None,
        "sentaurus_request": request.sentaurus_request,
        "objectives": [item.model_dump(mode="json") for item in request.objectives],
        "constraints": [item.model_dump(mode="json") for item in request.constraints],
        "require_capability_audit": request.require_capability_audit,
        "max_mutation_refinements": request.max_mutation_refinements,
        "auto_execute_mutation_refinements": request.auto_execute_mutation_refinements,
        "enable_experiment_design": request.enable_experiment_design,
        "max_experiment_design_rounds": request.max_experiment_design_rounds,
        "auto_execute_experiment_design": request.auto_execute_experiment_design,
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
            "hypothesis_tree_update": {
                "hypothesis_zh": "当前要验证的具体物理/数值假设",
                "expected_observation": "如果假设正确，下一轮曲线/指标/日志应出现什么",
                "stop_condition": "什么证据足以停止或转向",
                "next_alternatives": ["如果失败，下一批候选假设或实验"],
            },
            "evidence_used": ["actual context keys used"],
        },
        "guardrails": [
            "一次只选择一个下一步 action。",
            "规划 simulator-specific patch 前必须使用 public_evidence_dossier 与本地 deck/state 证据；没有证据时先补证据而不是猜。",
            "public_evidence_dossier 只能作为公开方法/来源索引，不能当成私有 PDK、商业模型或校准 deck。",
            "失败或可疑 state 优先 repair/benchmark，而不是直接报告成功。",
            "如果 mutation_effect_analysis 显示方向有效，先生成更细 refinement patch；如果 tradeoff 变坏，要求 Pareto/约束复核。",
            "如果最新 state 来自 Sentaurus，优先 plan_sentaurus_patch 生成可验证语义 patch，再决定是否执行下一轮 Sentaurus。",
            "如果 sentaurus_mutation_effect_analysis 显示 patch 有效，可以继续细化；如果出现 tradeoff，必须进行约束/Pareto 复核或等待确认。",
            "如果 benchmark/signoff evidence 有缺口，可以先 plan_experiment_design，让候选实验而不是单条规则驱动下一步。",
            "预算感知：如果 completed_steps + 1 >= max_steps，且已经有 baseline/benchmark/signoff/report 之一作为可解释证据，优先 stop_success 或 ask_user 总结剩余缺口，不要再开启新实验。",
            "每一步都维护 hypothesis_tree_update：假设、预期观察、停止条件和备选假设。",
            "高风险 geometry/process/model patch 必须要求用户确认。",
            "compact/planned evidence 不能 stop_success 为签核结论。",
            "如果没有 state 且没有 initial tool，先 run_supervisor。",
        ],
        "context": context,
    }
    return system, json.dumps(user, ensure_ascii=False, indent=2)


def action_from_experiment_candidate(candidate: dict[str, Any], latest_state_path: str | None) -> DevsimAgentAction:
    kind = str(candidate.get("action_kind") or "")
    request = dict(candidate.get("request") or {})
    request["agent_experiment_candidate_id"] = candidate.get("candidate_id")
    source_state_path = str(candidate.get("source_state_path") or latest_state_path or "") or None
    reason = str(candidate.get("reason") or "Execute the highest-ranked agent experiment design candidate.")
    requires_confirmation = bool(candidate.get("requires_user_confirmation"))
    if kind == DevsimAgentActionKind.RUN_REPAIR_EXECUTOR.value:
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.RUN_REPAIR_EXECUTOR,
            source_state_path=source_state_path,
            request=request,
            reason=reason,
            user_confirmation_required=requires_confirmation,
        )
    if kind == DevsimAgentActionKind.ASK_USER.value:
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.ASK_USER,
            source_state_path=source_state_path,
            request=request,
            reason=reason,
            user_confirmation_required=requires_confirmation,
        )
    if kind == DevsimAgentActionKind.PLAN_MUTATION_REFINEMENT.value:
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.PLAN_MUTATION_REFINEMENT,
            source_state_path=source_state_path,
            request=request,
            reason=reason,
            user_confirmation_required=requires_confirmation,
        )
    return DevsimAgentAction(
        kind=DevsimAgentActionKind.RUN_TOOL,
        tool_name=str(candidate.get("tool_name") or ""),
        source_state_path=source_state_path,
        request=request,
        reason=reason,
        user_confirmation_required=requires_confirmation,
    )


def active_curve_guidance(state: AutonomousDevsimAgentState, request: AutonomousDevsimRequest) -> dict[str, Any] | None:
    guidance = request.curve_guidance if isinstance(request.curve_guidance, dict) else {}
    if not guidance:
        guidance = state.checkpoint.get("curve_guidance") if isinstance(state.checkpoint.get("curve_guidance"), dict) else {}
    return guidance if guidance_is_actionable_patch(guidance) else None


def guidance_signature(guidance: dict[str, Any], source_state_path: str | None) -> str:
    hint = guidance.get("next_patch_hint") if isinstance(guidance.get("next_patch_hint"), dict) else {}
    parts = [
        str(guidance.get("source_state_path") or ""),
        str(guidance.get("created_at") or ""),
        str(guidance.get("recommended_action") or ""),
        str(guidance.get("recommended_target") or hint.get("target") or ""),
        str(guidance.get("recommended_direction") or hint.get("direction") or ""),
        str(guidance.get("reason") or ""),
    ]
    return "|".join(parts)


def action_from_sentaurus_patch_candidate(
    candidate: dict[str, Any],
    request: AutonomousDevsimRequest,
    state: AutonomousDevsimAgentState,
) -> DevsimAgentAction:
    tool_request = sentaurus_tool_request(request)
    existing_patches = tool_request.get("patches") if isinstance(tool_request.get("patches"), list) else []
    history = state.checkpoint.get("sentaurus_patch_history")
    lineage_patches: list[dict[str, Any]] = []
    if isinstance(history, list):
        for item in history:
            if isinstance(item, dict) and isinstance(item.get("patches"), list):
                lineage_patches.extend(patch for patch in item["patches"] if isinstance(patch, dict))
    tool_request["patches"] = [*existing_patches, *lineage_patches, *(candidate.get("patches") or [])]
    tool_request["sentaurus_patch_candidate_id"] = candidate.get("candidate_id")
    if state.latest_state_path:
        tool_request["repair_baseline_state_path"] = state.latest_state_path
    if not tool_request.get("run_id"):
        safe_id = safe_tool_name(str(candidate.get("candidate_id") or "sentaurus_patch"))[:80]
        tool_request["run_id"] = f"{request.agent_id or 'agent'}_{safe_id}"
    return DevsimAgentAction(
        kind=DevsimAgentActionKind.RUN_TOOL,
        tool_name="sentaurus_run",
        source_state_path=state.latest_state_path,
        request=tool_request,
        reason=str(candidate.get("hypothesis") or "Execute the selected verified Sentaurus semantic patch candidate."),
        user_confirmation_required=bool(candidate.get("requires_user_confirmation") or candidate.get("risk_level") == "high"),
    )


def sentaurus_tool_request(request: AutonomousDevsimRequest) -> dict[str, Any]:
    if not request.sentaurus_project_path:
        raise ValueError("sentaurus_project_path is required for Sentaurus execution")
    payload = dict(request.sentaurus_request)
    payload.setdefault("goal_text", request.goal_text)
    payload.setdefault("project_path", str(request.sentaurus_project_path))
    if request.sentaurus_profile_path:
        payload.setdefault("profile_path", str(request.sentaurus_profile_path))
    return payload


def live_evidence_lookup_gap(state: AutonomousDevsimAgentState) -> dict[str, Any] | None:
    lookup = state.checkpoint.get("public_evidence_lookup")
    if not isinstance(lookup, dict) or not lookup.get("live"):
        return None
    gate = lookup.get("evidence_gate")
    if not isinstance(gate, dict) or bool(gate.get("passed")):
        return None
    return lookup


def deterministic_action(state: AutonomousDevsimAgentState, request: AutonomousDevsimRequest) -> DevsimAgentAction:
    observation = observe_state(state.latest_state_path)
    quality_status = observation.get("quality_status")
    if request.require_capability_audit and not state.checkpoint.get("capability_audit_done"):
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.AUDIT_CAPABILITY,
            request={"goal_text": request.goal_text},
            reason="Audit device template coverage before treating the goal as executable TCAD work.",
        )
    live_gap = live_evidence_lookup_gap(state)
    if live_gap and not request.allow_live_evidence_gaps:
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.ASK_USER,
            request={
                "gate": "public_evidence_lookup",
                "question": "Live public evidence lookup did not verify a source for this operation. Confirm before the agent continues.",
                "lookup_status": live_gap.get("status"),
                "source_ids": live_gap.get("source_ids") or [],
                "failed_source_ids": live_gap.get("failed_source_ids") or [],
                "evidence_gate": live_gap.get("evidence_gate") or {},
            },
            reason="Live public evidence lookup has gaps; pause instead of inventing simulator or process semantics.",
        )
    if live_gap and request.allow_live_evidence_gaps and not state.checkpoint.get("public_evidence_gap_override"):
        state.checkpoint["public_evidence_gap_override"] = {
            "accepted": True,
            "accepted_at": utc_timestamp(),
            "lookup_status": live_gap.get("status"),
            "reason": "User allowed the agent to continue despite live public evidence gaps.",
        }
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
    if (
        state.checkpoint.get("deck_patch_done")
        and state.checkpoint.get("deck_patch_unverified")
        and not request.allow_unverified_deck_patch_execution
    ):
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.ASK_USER,
            request={
                "question": "Some deck patches were only appended as unverified fallback variables. Confirm before executing this patched deck.",
                "unverified_patches": state.checkpoint.get("deck_patch_unverified") or [],
            },
            reason="Semantic deck patch did not verify every requested edit against an existing deck binding.",
        )
    if request.initial_tool_name and not state.checkpoint.get("initial_tool_done"):
        tool_request = dict(request.initial_request)
        if request.initial_tool_name == "sentaurus_run" and request.sentaurus_project_path:
            tool_request = {**sentaurus_tool_request(request), **tool_request}
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
    if request.sentaurus_project_path and not state.checkpoint.get("sentaurus_initial_run_done"):
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.RUN_TOOL,
            tool_name="sentaurus_run",
            request=sentaurus_tool_request(request),
            reason="Sentaurus project context was provided; run the external Sentaurus baseline before repair, benchmark, or patch planning.",
        )
    if request.source_deck_path and not request.initial_tool_name and not state.checkpoint.get("user_deck_done"):
        deck_path = state.checkpoint.get("patched_source_deck") or request.source_deck_path
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.RUN_USER_DECK,
            request={"deck_path": deck_path, "run_id": f"{state.agent_id}_user_deck"},
            reason="No initial tool was requested; execute the user DEVSIM deck directly after ingest/patch setup.",
        )
    pending_refinement = state.checkpoint.get("pending_mutation_refinement")
    if (
        request.auto_execute_mutation_refinements
        and isinstance(pending_refinement, dict)
        and pending_refinement.get("next_request")
        and not pending_refinement.get("executed")
    ):
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.RUN_TOOL,
            tool_name=str(pending_refinement.get("target_tool") or observation.get("tool_name") or ""),
            request=dict(pending_refinement["next_request"]),
            reason=str(pending_refinement.get("reason") or "Execute the curve-guided mutation refinement patch."),
            user_confirmation_required=bool(pending_refinement.get("requires_user_confirmation")),
        )
    pending_guidance_patch = state.checkpoint.get("pending_guidance_patch")
    if (
        request.auto_execute_mutation_refinements
        and isinstance(pending_guidance_patch, dict)
        and pending_guidance_patch.get("next_request")
        and not pending_guidance_patch.get("executed")
    ):
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.RUN_TOOL,
            tool_name=str(pending_guidance_patch.get("target_tool") or observation.get("tool_name") or ""),
            request=dict(pending_guidance_patch["next_request"]),
            reason=str(pending_guidance_patch.get("reason") or "Execute the curve-guidance deck/request patch."),
            user_confirmation_required=bool(pending_guidance_patch.get("requires_user_confirmation")),
        )
    pending_experiment = state.checkpoint.get("pending_agent_experiment_candidate")
    if (
        request.auto_execute_experiment_design
        and isinstance(pending_experiment, dict)
        and not pending_experiment.get("executed")
    ):
        return action_from_experiment_candidate(pending_experiment, state.latest_state_path)
    pending_sentaurus_patch = state.checkpoint.get("pending_sentaurus_patch_candidate")
    if (
        request.auto_execute_experiment_design
        and request.sentaurus_project_path
        and isinstance(pending_sentaurus_patch, dict)
        and not pending_sentaurus_patch.get("executed")
    ):
        return action_from_sentaurus_patch_candidate(pending_sentaurus_patch, request, state)
    if not state.latest_state_path:
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.RUN_SUPERVISOR,
            request={"goal_text": request.goal_text, "execute": True, "max_cycles": request.supervisor_max_cycles},
            reason="No TCAD state exists yet; ask the supervisor to route the goal to a supported tool.",
        )
    guidance = active_curve_guidance(state, request)
    guidance_patch_runs = int(state.checkpoint.get("guidance_patch_runs") or 0)
    signature = guidance_signature(guidance, state.latest_state_path) if guidance else ""
    if (
        guidance
        and observation.get("tool_name") != "sentaurus_run"
        and guidance_patch_runs < request.max_mutation_refinements
        and state.checkpoint.get("guidance_patch_signature") != signature
    ):
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.PLAN_GUIDANCE_PATCH,
            source_state_path=state.latest_state_path,
            request={"curve_guidance": guidance, "guidance_signature": signature},
            reason="Curve guidance identified an actionable next deck/request patch; plan it before stopping or reporting.",
        )
    mutation_refinement_runs = int(state.checkpoint.get("mutation_refinement_runs") or 0)
    latest_analysis = observation.get("mutation_effect_analysis")
    if (
        isinstance(latest_analysis, dict)
        and latest_analysis
        and observation.get("tool_name") != "sentaurus_run"
        and mutation_refinement_runs < request.max_mutation_refinements
        and state.checkpoint.get("mutation_refinement_plan_source_path") != state.latest_state_path
    ):
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.PLAN_MUTATION_REFINEMENT,
            source_state_path=state.latest_state_path,
            reason="Latest state contains baseline-vs-mutation curve evidence; generate the next finer deck/request patch before continuing.",
        )
    sentaurus_effect = observation.get("sentaurus_mutation_effect_analysis")
    if isinstance(sentaurus_effect, dict) and observation.get("tool_name") == "sentaurus_run":
        decision = str(sentaurus_effect.get("decision") or "")
        if (
            decision == "blocked_for_pareto_review"
            and state.checkpoint.get("sentaurus_tradeoff_review_source_path") != state.latest_state_path
        ):
            if request.objectives or request.constraints:
                return DevsimAgentAction(
                    kind=DevsimAgentActionKind.EVALUATE_OBJECTIVES,
                    source_state_path=state.latest_state_path,
                    request={
                        "objectives": [item.model_dump(mode="json") for item in request.objectives],
                        "constraints": [item.model_dump(mode="json") for item in request.constraints],
                    },
                    reason="Sentaurus patch improved one direction but introduced tradeoffs; evaluate configured constraints before continuing.",
                )
            return DevsimAgentAction(
                kind=DevsimAgentActionKind.ASK_USER,
                source_state_path=state.latest_state_path,
                request={
                    "question": "Sentaurus patch introduced tradeoff regressions. Review Pareto/constraints before the agent continues.",
                    "sentaurus_mutation_effect_analysis": sentaurus_effect,
                },
                reason="Sentaurus patch introduced tradeoff regressions; pause for Pareto/constraint review.",
            )
        if (
            decision in {"reject_candidate", "switch_target"}
            and state.checkpoint.get("sentaurus_rejected_patch_source_path") != state.latest_state_path
        ):
            state.checkpoint["sentaurus_rejected_patch_source_path"] = state.latest_state_path
            state.checkpoint["sentaurus_rejected_patch_analysis"] = sentaurus_effect
        sentaurus_experiment_runs = int(state.checkpoint.get("experiment_design_runs") or 0)
        if (
            request.enable_experiment_design
            and request.sentaurus_project_path
            and decision in {"continue_refine", "switch_target", "reject_candidate"}
            and sentaurus_experiment_runs < request.max_experiment_design_rounds
            and state.checkpoint.get("sentaurus_refinement_plan_source_path") != state.latest_state_path
        ):
            return DevsimAgentAction(
                kind=DevsimAgentActionKind.PLAN_SENTAURUS_REFINEMENT,
                source_state_path=state.latest_state_path,
                request={"allow_high_risk": request.allow_user_confirmation_actions},
                reason="Latest Sentaurus patch has baseline-vs-mutation curve evidence; refine the same verified direction or switch targets before generic planning.",
            )
    sentaurus_patch_runs = int(state.checkpoint.get("sentaurus_patch_plan_runs") or 0)
    if (
        request.enable_experiment_design
        and observation.get("tool_name") == "sentaurus_run"
        and not isinstance(sentaurus_effect, dict)
        and sentaurus_patch_runs < request.max_experiment_design_rounds
        and state.checkpoint.get("sentaurus_patch_plan_source_path") != state.latest_state_path
    ):
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.PLAN_SENTAURUS_PATCH,
            source_state_path=state.latest_state_path,
            request={"allow_high_risk": request.allow_user_confirmation_actions},
            reason="Latest state is a Sentaurus run; plan verified semantic deck patches from the natural-language goal before generic repair/reporting.",
        )
    if state_needs_repair_before_signoff_planning(observation):
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
    experiment_design_runs = int(state.checkpoint.get("experiment_design_runs") or 0)
    if (
        request.enable_experiment_design
        and experiment_design_runs < request.max_experiment_design_rounds
        and state.checkpoint.get("experiment_design_source_path") != state.latest_state_path
    ):
        return DevsimAgentAction(
            kind=DevsimAgentActionKind.PLAN_EXPERIMENT_DESIGN,
            source_state_path=state.latest_state_path,
            reason="Use benchmark/signoff gaps, curve diagnostics, and deck mutations to rank the next autonomous experiment.",
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


def parse_agent_decision_json(raw: str) -> dict[str, Any] | None:
    parsed = parse_json_object(raw)
    if not isinstance(parsed, dict):
        return None
    if any(key in parsed for key in ["action", "tool_call", "tool_calls", "kind"]):
        return parsed
    content = parsed.get("content")
    if isinstance(content, str):
        nested = parse_json_object(content)
        if isinstance(nested, dict):
            return nested
    return parsed


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
    if fallback.kind == DevsimAgentActionKind.ASK_USER and fallback.request.get("gate") == "public_evidence_lookup":
        decision["status"] = "hard_public_evidence_gate"
        return fallback, decision
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
    parsed = parse_agent_decision_json(raw)
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
            "hypothesis_tree_update": parsed.get("hypothesis_tree_update") if isinstance(parsed.get("hypothesis_tree_update"), dict) else None,
            "evidence_used": parsed.get("evidence_used") or [],
        }
    )
    return action, decision


def result_verdict(result: dict[str, Any] | None, error: str | None = None) -> str:
    if error:
        return "failed"
    if not isinstance(result, dict):
        return "planned"
    quality = result.get("quality_report") if isinstance(result.get("quality_report"), dict) else {}
    status = str(result.get("status") or quality.get("status") or "").lower()
    quality_status = str(quality.get("status") or "").lower()
    if status == "failed" or quality_status == "failed" or result.get("failure_reason"):
        return "failed"
    if status == "suspicious" or quality_status == "suspicious":
        return "suspicious"
    if status in {"completed", "passed"} or quality_status == "passed":
        return "supported"
    return status or "observed"


def update_agent_hypothesis_tree(
    state: AutonomousDevsimAgentState,
    step: DevsimAgentStep,
    decision: dict[str, Any],
    *,
    result: dict[str, Any] | None = None,
    result_state_path: str | None = None,
    error: str | None = None,
) -> None:
    tree = state.checkpoint.get("agent_hypothesis_tree")
    if not isinstance(tree, dict):
        tree = {
            "schema_version": "actsoft.tcad.agent_hypothesis_tree.v1",
            "goal_text": state.goal_text,
            "nodes": [],
            "open_questions": [],
        }
    nodes = tree.get("nodes")
    if not isinstance(nodes, list):
        nodes = []
        tree["nodes"] = nodes
    parsed = decision.get("parsed_response") if isinstance(decision.get("parsed_response"), dict) else {}
    update = decision.get("hypothesis_tree_update") if isinstance(decision.get("hypothesis_tree_update"), dict) else {}
    if not update and isinstance(parsed, dict):
        update = parsed.get("hypothesis_tree_update") if isinstance(parsed.get("hypothesis_tree_update"), dict) else {}
    action = step.action or {}
    hypothesis = (
        update.get("hypothesis_zh")
        or decision.get("hypothesis_zh")
        or step.reason
        or "执行下一步以补齐 TCAD 证据。"
    )
    parent_id = update.get("parent_id") or (nodes[-1].get("id") if nodes and isinstance(nodes[-1], dict) else None)
    node = {
        "id": update.get("id") or f"h{len(nodes) + 1:03d}",
        "parent_id": parent_id,
        "step_index": step.index,
        "action_kind": action.get("kind"),
        "tool_name": action.get("tool_name"),
        "hypothesis_zh": hypothesis,
        "expected_observation": update.get("expected_observation"),
        "stop_condition": update.get("stop_condition"),
        "next_alternatives": update.get("next_alternatives") if isinstance(update.get("next_alternatives"), list) else [],
        "evidence_used": decision.get("evidence_used") or [],
        "verdict": result_verdict(result, error),
        "result_state_path": result_state_path,
        "error": error,
        "created_at": step.started_at,
        "updated_at": utc_timestamp(),
    }
    if isinstance(result, dict):
        quality = result.get("quality_report") if isinstance(result.get("quality_report"), dict) else {}
        node["observed_status"] = result.get("status")
        node["observed_quality"] = quality.get("status")
        if result.get("failure_reason"):
            node["failure_reason"] = result.get("failure_reason")
        metrics = quality.get("metrics") if isinstance(quality.get("metrics"), dict) else {}
        if metrics:
            node["metric_keys"] = list(metrics)[:8]
    nodes.append(node)
    tree["nodes"] = nodes[-40:]
    open_questions: list[str] = []
    for item in reversed(tree["nodes"]):
        if not isinstance(item, dict):
            continue
        if item.get("verdict") in {"failed", "suspicious"}:
            open_questions.extend(str(value) for value in item.get("next_alternatives") or [] if value)
        if len(open_questions) >= 5:
            break
    tree["open_questions"] = open_questions[:5]
    tree["last_hypothesis"] = node
    tree["updated_at"] = node["updated_at"]
    state.checkpoint["agent_hypothesis_tree"] = tree


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
                "deck_patches_verified": len(result.verified_patches),
                "all_patches_verified": result.all_patches_verified,
            },
        },
        "quality_report": {
            "status": "passed" if not result.unapplied_patches else "suspicious",
            "issues": [{"code": "deck_patch_fallback_append", "severity": "warning"} for _ in result.unapplied_patches],
            "metrics": {
                "deck_patches_applied": len(result.applied_patches),
                "deck_patches_unapplied": len(result.unapplied_patches),
                "deck_patches_verified": len(result.verified_patches),
                "all_patches_verified": result.all_patches_verified,
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
        "verified_patches": result.verified_patches,
        "unverified_patches": result.unverified_patches,
        "all_patches_verified": result.all_patches_verified,
    }


def augment_mutation_refinement_result(
    *,
    baseline_state_path: str | None,
    result_state_path: str | None,
    deck_patch: dict[str, Any],
) -> dict[str, Any] | None:
    if not baseline_state_path or not result_state_path:
        return None
    baseline = Path(str(baseline_state_path))
    result_path = Path(str(result_state_path))
    if not baseline.exists() or not result_path.exists():
        return None
    diagnostic = compare_state_mutation_effect(
        baseline,
        result_path,
        deck_patch=deck_patch,
        overlay_output_path=result_path.parent / "baseline_mutation_overlay.svg",
    )
    result_state = load_final_state(str(result_path))
    if not result_state:
        return diagnostic.model_dump(mode="json")
    diagnostic_data = diagnostic.model_dump(mode="json")
    result_state["mutation_effect_analysis"] = diagnostic_data
    raw_context = result_state.get("repair_context")
    context = dict(raw_context) if isinstance(raw_context, dict) else {}
    context.update(
        {
            "schema_version": "actsoft.tcad.repair_context.v1",
            "baseline_state_path": str(baseline),
            "parent_state_path": str(result_path),
            "action_name": "agent_mutation_refinement",
            "deck_patch": deck_patch,
            "mutation_effect_decision": diagnostic.decision,
            "recommended_next_target": diagnostic.recommended_next_target,
            "worth_continuing_mutation": diagnostic.worth_continuing,
        }
    )
    result_state["repair_context"] = context
    if diagnostic.overlay_svg_path:
        summary = result_state.get("final_summary") or {}
        artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
        artifacts["baseline_mutation_overlay"] = diagnostic.overlay_svg_path
        summary["artifacts"] = artifacts
        result_state["final_summary"] = summary
    write_json(result_path, result_state)
    return diagnostic_data


def augment_sentaurus_patch_result(
    *,
    baseline_state_path: str | None,
    result_state_path: str | None,
    candidate: dict[str, Any],
    goal_text: str,
) -> dict[str, Any] | None:
    if not baseline_state_path or not result_state_path:
        return None
    baseline = Path(str(baseline_state_path))
    result_path = Path(str(result_state_path))
    if not baseline.exists() or not result_path.exists():
        return None
    output_path = result_path.parent / "sentaurus_mutation_effect.json"
    analysis = analyze_sentaurus_mutation_effect(
        SentaurusMutationEffectRequest(
            baseline_state_path=baseline,
            mutation_state_path=result_path,
            candidate=candidate,
            goal_text=goal_text,
            output_path=output_path,
            overlay_output_path=result_path.parent / "sentaurus_baseline_mutation_overlay.svg",
        )
    )
    analysis_data = analysis.model_dump(mode="json")
    result_state = load_final_state(str(result_path))
    if not result_state:
        return analysis_data
    result_state["sentaurus_mutation_effect_analysis"] = analysis_data
    raw_context = result_state.get("repair_context")
    context = dict(raw_context) if isinstance(raw_context, dict) else {}
    context.update(
        {
            "schema_version": "actsoft.tcad.repair_context.v1",
            "baseline_state_path": str(baseline),
            "parent_state_path": str(result_path),
            "action_name": "sentaurus_patch_candidate",
            "sentaurus_patch_candidate_id": candidate.get("candidate_id"),
            "sentaurus_mutation_effect_decision": analysis.decision,
            "sentaurus_recommended_next_target": analysis.recommended_next_target,
            "worth_continuing_sentaurus_patch": analysis.worth_continuing,
        }
    )
    result_state["repair_context"] = context
    summary = result_state.get("final_summary") if isinstance(result_state.get("final_summary"), dict) else {}
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    artifacts["sentaurus_mutation_effect"] = str(output_path.resolve())
    if analysis.overlay_svg_path:
        artifacts["sentaurus_baseline_mutation_overlay"] = analysis.overlay_svg_path
    summary["artifacts"] = artifacts
    result_state["final_summary"] = summary
    write_json(result_path, result_state)
    archive_output_path = result_path.parent / "sentaurus_lineage_archive.json"
    archive = build_sentaurus_lineage_archive(
        SentaurusLineageArchiveRequest(
            source_state_path=result_path,
            output_path=archive_output_path,
        )
    )
    archive_data = archive.model_dump(mode="json")
    result_state = load_final_state(str(result_path)) or result_state
    result_state["sentaurus_lineage_archive"] = archive_data
    summary = result_state.get("final_summary") if isinstance(result_state.get("final_summary"), dict) else {}
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    artifacts["sentaurus_lineage_archive"] = str(archive_output_path.resolve())
    summary["artifacts"] = artifacts
    result_state["final_summary"] = summary
    write_json(result_path, result_state)
    return analysis_data


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
            "industrial_runner_coverage": route.industrial_runner_coverage,
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
                "industrial_runner_coverage": [
                    item.model_dump(mode="json") for item in runner_descriptors_for_template(route.template.template_id)
                ],
            }
            promotion_path = Path(state.agent_dir) / "runner_promotion_plan.json"
            promotion = build_industrial_runner_promotion_plan(
                state.goal_text,
                template_id=route.template.template_id,
                simulator="sentaurus" if request.sentaurus_project_path else "devsim",
                live_lookup_result=state.checkpoint.get("public_evidence_lookup") if isinstance(state.checkpoint.get("public_evidence_lookup"), dict) else None,
                output_path=promotion_path,
            )
            state.checkpoint["runner_promotion_plan_path"] = promotion.output_path
            state.checkpoint["runner_promotion_plan"] = promotion.model_dump(mode="json")
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
        tool_request = dict(action.request)
        if state.cancel_file:
            tool_request.setdefault("cancel_file", state.cancel_file)
        with temporary_cancel_env(state.cancel_file):
            result = runner(tool_request)
        result_state_path = infer_result_state_path(result)
        if tool_request.get("agent_experiment_candidate_id"):
            pending = state.checkpoint.get("pending_agent_experiment_candidate")
            if isinstance(pending, dict):
                state.checkpoint["pending_agent_experiment_candidate"] = {
                    **pending,
                    "executed": True,
                    "result_state_path": result_state_path,
                }
            state.checkpoint["executed_agent_experiment_candidates"] = int(
                state.checkpoint.get("executed_agent_experiment_candidates") or 0
            ) + 1
        if tool_request.get("mutation_refinement_id"):
            state.checkpoint["mutation_refinement_runs"] = int(state.checkpoint.get("mutation_refinement_runs") or 0) + 1
            state.checkpoint["last_mutation_refinement_id"] = tool_request.get("mutation_refinement_id")
            pending = state.checkpoint.get("pending_mutation_refinement")
            deck_patch = (pending or {}).get("deck_patch") if isinstance(pending, dict) else {}
            baseline = tool_request.get("repair_baseline_state_path") or tool_request.get("repair_source_state_path") or state.latest_state_path
            diagnostic = augment_mutation_refinement_result(
                baseline_state_path=str(baseline) if baseline else None,
                result_state_path=result_state_path,
                deck_patch=deck_patch if isinstance(deck_patch, dict) else {},
            )
            if diagnostic:
                result["mutation_effect_analysis"] = diagnostic
            state.checkpoint["pending_mutation_refinement"] = {**pending, "executed": True} if isinstance(pending, dict) else {"executed": True}
        if tool_request.get("guidance_patch_id"):
            state.checkpoint["guidance_patch_runs"] = int(state.checkpoint.get("guidance_patch_runs") or 0) + 1
            state.checkpoint["last_guidance_patch_id"] = tool_request.get("guidance_patch_id")
            pending = state.checkpoint.get("pending_guidance_patch")
            deck_patch = (pending or {}).get("deck_patch") if isinstance(pending, dict) else {}
            baseline = tool_request.get("guidance_source_state_path") or tool_request.get("repair_baseline_state_path") or state.latest_state_path
            diagnostic = augment_mutation_refinement_result(
                baseline_state_path=str(baseline) if baseline else None,
                result_state_path=result_state_path,
                deck_patch=deck_patch if isinstance(deck_patch, dict) else {},
            )
            if diagnostic:
                result["mutation_effect_analysis"] = diagnostic
                state.checkpoint["latest_guidance_mutation_effect_analysis"] = diagnostic
            state.checkpoint["pending_guidance_patch"] = {**pending, "executed": True} if isinstance(pending, dict) else {"executed": True}
        if tool_request.get("sentaurus_patch_candidate_id"):
            pending = state.checkpoint.get("pending_sentaurus_patch_candidate")
            pending_patches = pending.get("patches") if isinstance(pending, dict) and isinstance(pending.get("patches"), list) else []
            candidate_for_analysis = dict(pending) if isinstance(pending, dict) else {"candidate_id": tool_request.get("sentaurus_patch_candidate_id")}
            baseline = tool_request.get("repair_baseline_state_path") or state.latest_state_path
            analysis = augment_sentaurus_patch_result(
                baseline_state_path=str(baseline) if baseline else None,
                result_state_path=result_state_path,
                candidate=candidate_for_analysis,
                goal_text=state.goal_text,
            )
            if analysis:
                result["sentaurus_mutation_effect_analysis"] = analysis
            if isinstance(pending, dict):
                state.checkpoint["pending_sentaurus_patch_candidate"] = {
                    **pending,
                    "executed": True,
                    "result_state_path": result_state_path,
                    "sentaurus_mutation_effect_analysis": analysis,
                }
            history = state.checkpoint.get("sentaurus_patch_history")
            history_items = list(history) if isinstance(history, list) else []
            history_items.append(
                {
                    "candidate_id": tool_request.get("sentaurus_patch_candidate_id"),
                    "patches": pending_patches,
                    "result_state_path": result_state_path,
                    "sentaurus_mutation_effect_analysis": analysis,
                }
            )
            state.checkpoint["sentaurus_patch_history"] = history_items
            state.checkpoint["latest_sentaurus_mutation_effect_analysis"] = analysis
            if result_state_path:
                patched_state = load_final_state(result_state_path)
                if patched_state and patched_state.get("sentaurus_lineage_archive"):
                    state.checkpoint["latest_sentaurus_lineage_archive"] = patched_state.get("sentaurus_lineage_archive")
            state.checkpoint["executed_sentaurus_patch_candidates"] = int(
                state.checkpoint.get("executed_sentaurus_patch_candidates") or 0
            ) + 1
        return result, result_state_path
    if action.kind == DevsimAgentActionKind.RUN_REPAIR_EXECUTOR:
        source = action.source_state_path or state.latest_state_path
        if not source:
            raise ValueError("repair action requires a source state")
        with temporary_cancel_env(state.cancel_file):
            result = result_to_dict(
                repair_runner(
                    Path(source),
                    execute=True,
                    max_rounds=request.repair_max_rounds,
                    allow_user_confirmation_actions=request.allow_user_confirmation_actions,
                    use_agent_policy=request.use_agent_policy,
                )
            )
        if action.request.get("agent_experiment_candidate_id"):
            pending = state.checkpoint.get("pending_agent_experiment_candidate")
            if isinstance(pending, dict):
                state.checkpoint["pending_agent_experiment_candidate"] = {
                    **pending,
                    "executed": True,
                    "result_state_path": infer_result_state_path(result) or result.get("final_state_path") or result.get("current_state_path"),
                }
            state.checkpoint["executed_agent_experiment_candidates"] = int(
                state.checkpoint.get("executed_agent_experiment_candidates") or 0
            ) + 1
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
        state.checkpoint["engineering_objective_decision"] = result.get("decision") or {}
        if "Sentaurus patch" in action.reason:
            state.checkpoint["sentaurus_tradeoff_review_source_path"] = source
            state.checkpoint["sentaurus_tradeoff_review"] = result
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
        state.checkpoint["deck_patch_verified"] = result.get("all_patches_verified")
        state.checkpoint["deck_patch_unverified"] = result.get("unverified_patches") or []
        return result, result.get("state_path")
    if action.kind == DevsimAgentActionKind.RUN_USER_DECK:
        runner = runner_registry.get("user_deck_execution")
        if runner is None:
            raise ValueError("user_deck_execution runner is not registered")
        deck_path = action.request.get("deck_path") or state.checkpoint.get("patched_source_deck") or request.source_deck_path
        if not deck_path:
            raise ValueError("user deck execution requires deck_path")
        payload = {
            **action.request,
            "deck_path": deck_path,
            "cancel_file": state.cancel_file,
        }
        with temporary_cancel_env(state.cancel_file):
            result = runner(payload)
        state.checkpoint["user_deck_done"] = True
        state.checkpoint["user_deck_state_path"] = result.get("state_path")
        return result, infer_result_state_path(result) or result.get("state_path")
    if action.kind == DevsimAgentActionKind.PLAN_MUTATION_REFINEMENT:
        source = action.source_state_path or state.latest_state_path
        if not source:
            raise ValueError("mutation refinement action requires a source state")
        output_dir = Path(state.agent_dir) / "mutation_refinement"
        output_path = output_dir / f"mutation_refinement_{int(state.checkpoint.get('mutation_refinement_plans') or 0) + 1:03d}.json"
        plan = build_mutation_refinement_plan(Path(source), output_path=output_path)
        result = plan.model_dump(mode="json")
        state.checkpoint["mutation_refinement_plans"] = int(state.checkpoint.get("mutation_refinement_plans") or 0) + 1
        state.checkpoint["mutation_refinement_plan_path"] = result.get("output_path")
        state.checkpoint["mutation_refinement_plan_source_path"] = source
        if plan.status == "completed" and plan.next_request:
            state.checkpoint["pending_mutation_refinement"] = result
        elif plan.status == "blocked_for_pareto_review":
            state.checkpoint["blocked_mutation_refinement"] = result
        else:
            state.checkpoint["failed_mutation_refinement"] = result
        return result, source
    if action.kind == DevsimAgentActionKind.PLAN_GUIDANCE_PATCH:
        source = action.source_state_path or state.latest_state_path
        if not source:
            raise ValueError("guidance patch action requires a source state")
        guidance = action.request.get("curve_guidance") if isinstance(action.request.get("curve_guidance"), dict) else None
        if guidance is None:
            guidance = active_curve_guidance(state, request)
        if not guidance:
            raise ValueError("guidance patch action requires actionable curve_guidance")
        output_dir = Path(state.agent_dir) / "guidance_patches"
        output_path = output_dir / f"guidance_patch_{int(state.checkpoint.get('guidance_patch_plans') or 0) + 1:03d}.json"
        plan = build_guidance_patch_plan(
            Path(source),
            curve_guidance=guidance,
            goal_text=state.goal_text,
            output_path=output_path,
        )
        result = plan.model_dump(mode="json")
        state.checkpoint["guidance_patch_plans"] = int(state.checkpoint.get("guidance_patch_plans") or 0) + 1
        state.checkpoint["guidance_patch_plan_path"] = result.get("output_path")
        state.checkpoint["guidance_patch_plan_source_path"] = source
        state.checkpoint["guidance_patch_signature"] = action.request.get("guidance_signature") or guidance_signature(guidance, source)
        if plan.status == "completed" and plan.next_request:
            state.checkpoint["pending_guidance_patch"] = result
        elif plan.status == "no_action":
            state.checkpoint["skipped_guidance_patch"] = result
        else:
            state.checkpoint["failed_guidance_patch"] = result
        return result, source
    if action.kind == DevsimAgentActionKind.PLAN_SENTAURUS_PATCH:
        source = action.source_state_path or state.latest_state_path
        if not source:
            raise ValueError("Sentaurus patch planning action requires a source state")
        output_dir = Path(state.agent_dir) / "sentaurus_patch_plans"
        output_path = output_dir / f"sentaurus_patch_plan_{int(state.checkpoint.get('sentaurus_patch_plan_count') or 0) + 1:03d}.json"
        runner = runner_registry.get("sentaurus_patch_planner")
        payload = {
            "goal_text": state.goal_text,
            "source_state_path": source,
            "output_path": str(output_path),
            "allow_high_risk": bool(action.request.get("allow_high_risk", False)),
        }
        deck_files = request.sentaurus_request.get("deck_files") if isinstance(request.sentaurus_request.get("deck_files"), list) else []
        if deck_files:
            payload["deck_files"] = deck_files
        if runner is None:
            from tcad_agent.sentaurus_patch_planner import SentaurusPatchPlannerRequest, plan_sentaurus_patches

            result = plan_sentaurus_patches(SentaurusPatchPlannerRequest.model_validate(payload)).model_dump(mode="json")
        else:
            result = runner(payload)
        state.checkpoint["sentaurus_patch_plan_count"] = int(state.checkpoint.get("sentaurus_patch_plan_count") or 0) + 1
        state.checkpoint["sentaurus_patch_plan_runs"] = int(state.checkpoint.get("sentaurus_patch_plan_runs") or 0) + 1
        state.checkpoint["experiment_design_runs"] = int(state.checkpoint.get("experiment_design_runs") or 0) + 1
        state.checkpoint["sentaurus_patch_plan_path"] = result.get("output_path")
        state.checkpoint["sentaurus_patch_plan_source_path"] = source
        state.checkpoint["sentaurus_public_evidence_dossier"] = result.get("public_evidence_dossier") or state.checkpoint.get("public_evidence_dossier")
        state.checkpoint["experiment_design_source_path"] = source
        state.checkpoint["sentaurus_patch_candidates"] = result.get("candidates") or []
        selected = result.get("selected_candidate")
        if isinstance(selected, dict):
            state.checkpoint["pending_sentaurus_patch_candidate"] = selected
        elif result.get("status") == "blocked_for_user_confirmation":
            state.checkpoint["blocked_sentaurus_patch_candidates"] = result.get("candidates") or []
        else:
            state.checkpoint["sentaurus_patch_planner_exhausted"] = result
        return result, source
    if action.kind == DevsimAgentActionKind.PLAN_SENTAURUS_REFINEMENT:
        source = action.source_state_path or state.latest_state_path
        if not source:
            raise ValueError("Sentaurus refinement action requires a source state")
        output_dir = Path(state.agent_dir) / "sentaurus_patch_refinements"
        output_path = output_dir / f"sentaurus_patch_refinement_{int(state.checkpoint.get('sentaurus_refinement_plan_count') or 0) + 1:03d}.json"
        runner = runner_registry.get("sentaurus_patch_refiner")
        payload = {
            "source_state_path": source,
            "goal_text": state.goal_text,
            "output_path": str(output_path),
            "allow_high_risk": bool(action.request.get("allow_high_risk", False)),
            "use_llm": request.use_llm,
            "allow_llm_fallback": request.allow_llm_fallback,
        }
        if runner is None:
            result = build_sentaurus_patch_refinement_plan(SentaurusPatchRefinerRequest.model_validate(payload)).model_dump(mode="json")
        else:
            result = runner(payload)
        state.checkpoint["sentaurus_refinement_plan_count"] = int(state.checkpoint.get("sentaurus_refinement_plan_count") or 0) + 1
        state.checkpoint["sentaurus_refinement_runs"] = int(state.checkpoint.get("sentaurus_refinement_runs") or 0) + 1
        state.checkpoint["experiment_design_runs"] = int(state.checkpoint.get("experiment_design_runs") or 0) + 1
        state.checkpoint["sentaurus_refinement_plan_path"] = result.get("output_path")
        state.checkpoint["sentaurus_refinement_plan_source_path"] = source
        state.checkpoint["experiment_design_source_path"] = source
        state.checkpoint["sentaurus_refinement_candidates"] = result.get("candidates") or []
        selected = result.get("selected_candidate")
        if isinstance(selected, dict):
            state.checkpoint["pending_sentaurus_patch_candidate"] = selected
        elif result.get("status") == "blocked_for_user_confirmation":
            state.checkpoint["blocked_sentaurus_refinement_candidates"] = result.get("candidates") or []
        elif result.get("status") == "blocked_for_pareto_review":
            state.checkpoint["blocked_sentaurus_refinement"] = result
        else:
            state.checkpoint["sentaurus_refinement_exhausted"] = result
        return result, source
    if action.kind == DevsimAgentActionKind.PLAN_EXPERIMENT_DESIGN:
        source = action.source_state_path or state.latest_state_path
        if not source:
            raise ValueError("experiment design action requires a source state")
        output_dir = Path(state.agent_dir) / "experiment_design"
        output_path = output_dir / f"experiment_design_{int(state.checkpoint.get('experiment_design_plans') or 0) + 1:03d}.json"
        benchmark_path = state.checkpoint.get("physical_benchmark_path")
        plan = build_agent_experiment_design_plan(
            Path(source),
            benchmark_path=Path(str(benchmark_path)) if benchmark_path else None,
            output_path=output_path,
        )
        result = plan.model_dump(mode="json")
        state.checkpoint["experiment_design_plans"] = int(state.checkpoint.get("experiment_design_plans") or 0) + 1
        state.checkpoint["experiment_design_runs"] = int(state.checkpoint.get("experiment_design_runs") or 0) + 1
        state.checkpoint["experiment_design_plan_path"] = result.get("output_path")
        state.checkpoint["experiment_design_source_path"] = source
        state.checkpoint["agent_experiment_candidates"] = result.get("candidates") or []
        if plan.selected_candidate:
            state.checkpoint["pending_agent_experiment_candidate"] = plan.selected_candidate.model_dump(mode="json")
        else:
            state.checkpoint["agent_experiment_design_exhausted"] = result
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


def invalidate_state_dependent_signoff(state: AutonomousDevsimAgentState, previous_state_path: str | None, next_state_path: str | None) -> None:
    if not next_state_path or next_state_path == previous_state_path:
        return
    for key in (
        "physical_benchmark_done",
        "physical_benchmark_path",
        "objectives_done",
        "engineering_objectives_path",
        "pareto_front",
        "best_candidate",
        "report_done",
        "dashboard_done",
    ):
        state.checkpoint.pop(key, None)
    state.final_report_path = None
    state.final_dashboard_path = None
    state.checkpoint["state_dependent_signoff_invalidated"] = {
        "previous_state_path": previous_state_path,
        "next_state_path": next_state_path,
        "at": utc_timestamp(),
    }


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
        ledger = state.checkpoint.get("agent_decision_ledger")
        ledger_items = list(ledger) if isinstance(ledger, list) else []
        ledger_items.append(
            {
                "step_index": len(state.steps) + 1,
                "decided_at": utc_timestamp(),
                "decision_status": decision.get("status"),
                "fallback_used": bool(decision.get("fallback_used")),
                "action": action.model_dump(mode="json"),
                "observation_summary": decision.get("observation_summary"),
                "hypothesis_zh": decision.get("hypothesis_zh"),
            }
        )
        state.checkpoint["agent_decision_ledger"] = ledger_items[-80:]
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
            update_agent_hypothesis_tree(state, step, decision, result=None, result_state_path=state.latest_state_path)
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
            previous_state_path = state.latest_state_path
            if result_state_path:
                state.latest_state_path = result_state_path
                invalidate_state_dependent_signoff(state, previous_state_path, result_state_path)
            update_agent_hypothesis_tree(state, step, decision, result=result, result_state_path=result_state_path or state.latest_state_path)
            if action.kind == DevsimAgentActionKind.RUN_TOOL and action.tool_name == request.initial_tool_name:
                state.checkpoint["initial_tool_done"] = True
            if action.kind == DevsimAgentActionKind.RUN_TOOL and action.tool_name == "sentaurus_run" and request.sentaurus_project_path:
                state.checkpoint["sentaurus_initial_run_done"] = True
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
            update_agent_hypothesis_tree(state, step, decision, result=None, result_state_path=state.latest_state_path, error=str(exc))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the long-horizon autonomous DEVSIM agent runtime.")
    parser.add_argument("--goal", "--goal-text", dest="goal_text", required=True)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--agent-root", type=Path, default=PROJECT_ROOT / "runs" / "autonomous_devsim_agent")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--initial-tool-name", default=None)
    parser.add_argument("--initial-request-json", default=None, help="JSON object request for --initial-tool-name.")
    parser.add_argument("--source-state-path", default=None)
    parser.add_argument("--source-deck-path", default=None)
    parser.add_argument("--deck-patches-json", default=None, help="JSON list of semantic deck patches.")
    parser.add_argument("--allow-unverified-deck-patch-execution", action="store_true")
    parser.add_argument("--sentaurus-project-path", type=Path, default=None)
    parser.add_argument("--sentaurus-profile-path", type=Path, default=None)
    parser.add_argument("--sentaurus-request-json", default=None, help="JSON object passed to the sentaurus_run tool.")
    parser.add_argument("--objectives-json", default=None, help="JSON list of engineering objectives.")
    parser.add_argument("--constraints-json", default=None, help="JSON list of engineering constraints.")
    parser.add_argument("--cancel-file", type=Path, default=None)
    parser.add_argument("--heartbeat-path", type=Path, default=None)
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic policy only.")
    parser.add_argument("--no-llm-fallback", action="store_true", help="Fail if the LLM action is invalid/unavailable.")
    parser.add_argument("--no-agent-repair-policy", action="store_true", help="Disable LLM repair policy inside repair_executor.")
    parser.add_argument("--enable-live-evidence-lookup", action="store_true", help="Fetch registry public sources before planning.")
    parser.add_argument("--live-evidence-max-sources", type=int, default=6)
    parser.add_argument("--allow-live-evidence-gaps", action="store_true", help="Continue after an explicit live evidence lookup gap.")
    parser.add_argument("--allow-user-confirmation-actions", action="store_true")
    parser.add_argument("--supervisor-max-cycles", type=int, default=3)
    parser.add_argument("--repair-max-rounds", type=int, default=3)
    parser.add_argument("--max-mutation-refinements", type=int, default=1)
    parser.add_argument("--no-auto-mutation-refinement", action="store_true")
    parser.add_argument("--enable-experiment-design", action="store_true")
    parser.add_argument("--max-experiment-design-rounds", type=int, default=1)
    parser.add_argument("--no-auto-experiment-design", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--require-capability-audit", action="store_true")
    return parser.parse_args()


def request_from_args(args: argparse.Namespace) -> AutonomousDevsimRequest:
    initial_request = {}
    if args.initial_request_json:
        parsed = json.loads(args.initial_request_json)
        if not isinstance(parsed, dict):
            raise ValueError("--initial-request-json must decode to a JSON object")
        initial_request = parsed
    deck_patches = []
    if args.deck_patches_json:
        parsed = json.loads(args.deck_patches_json)
        if not isinstance(parsed, list):
            raise ValueError("--deck-patches-json must decode to a JSON list")
        deck_patches = parsed
    sentaurus_request = {}
    if args.sentaurus_request_json:
        parsed = json.loads(args.sentaurus_request_json)
        if not isinstance(parsed, dict):
            raise ValueError("--sentaurus-request-json must decode to a JSON object")
        sentaurus_request = parsed
    objectives = []
    if args.objectives_json:
        parsed = json.loads(args.objectives_json)
        if not isinstance(parsed, list):
            raise ValueError("--objectives-json must decode to a JSON list")
        objectives = parsed
    constraints = []
    if args.constraints_json:
        parsed = json.loads(args.constraints_json)
        if not isinstance(parsed, list):
            raise ValueError("--constraints-json must decode to a JSON list")
        constraints = parsed
    return AutonomousDevsimRequest(
        goal_text=args.goal_text,
        agent_id=args.agent_id,
        agent_root=args.agent_root,
        execute=args.execute,
        resume=args.resume,
        max_steps=args.max_steps,
        initial_tool_name=args.initial_tool_name,
        initial_request=initial_request,
        source_state_path=args.source_state_path,
        source_deck_path=args.source_deck_path,
        deck_patches=deck_patches,
        allow_unverified_deck_patch_execution=args.allow_unverified_deck_patch_execution,
        sentaurus_project_path=args.sentaurus_project_path,
        sentaurus_profile_path=args.sentaurus_profile_path,
        sentaurus_request=sentaurus_request,
        objectives=objectives,
        constraints=constraints,
        cancel_file=args.cancel_file,
        heartbeat_path=args.heartbeat_path,
        use_llm=not args.no_llm,
        allow_llm_fallback=not args.no_llm_fallback,
        use_agent_policy=not args.no_agent_repair_policy,
        enable_live_evidence_lookup=args.enable_live_evidence_lookup,
        live_evidence_max_sources=args.live_evidence_max_sources,
        allow_live_evidence_gaps=args.allow_live_evidence_gaps,
        allow_user_confirmation_actions=args.allow_user_confirmation_actions,
        supervisor_max_cycles=args.supervisor_max_cycles,
        repair_max_rounds=args.repair_max_rounds,
        max_mutation_refinements=args.max_mutation_refinements,
        auto_execute_mutation_refinements=not args.no_auto_mutation_refinement,
        enable_experiment_design=args.enable_experiment_design,
        max_experiment_design_rounds=args.max_experiment_design_rounds,
        auto_execute_experiment_design=not args.no_auto_experiment_design,
        generate_report=not args.no_report,
        generate_dashboard=not args.no_dashboard,
        require_capability_audit=args.require_capability_audit,
    )


def main() -> None:
    try:
        state = run_autonomous_devsim_agent(request_from_args(parse_args()))
        print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
        raise SystemExit(0 if state.status != DevsimAgentStatus.FAILED else 1)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "tool_name": "autonomous_devsim_agent",
                    "status": DevsimAgentStatus.FAILED,
                    "failure_reason": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
