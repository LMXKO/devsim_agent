from __future__ import annotations

import argparse

import json
import os
import shutil
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from tcad_agent.autonomous_devsim_agent import (
    AutonomousDevsimRequest,
    DevsimAgentActionKind,
    DevsimAgentStatus,
    run_autonomous_devsim_agent,
)
from tcad_agent.agent_cockpit import generate_agent_cockpit
from tcad_agent.agent_goal_router import AgentGoalRouteRequest, route_agent_goal
from tcad_agent.agent_soak import AgentSoakRequest, AgentSoakStatus, run_agent_soak
from tcad_agent.curve_diagnostics import compare_state_mutation_effect
from tcad_agent.curve_decision_eval import CurveDecisionEvalRequest, CurveDecisionEvalStatus, run_curve_decision_eval
from tcad_agent.experiment_index import list_records, rebuild_index
from tcad_agent.llm import LLMConfig
from tcad_agent.physical_benchmark import run_physical_benchmark
from tcad_agent.power_mosfet_signoff import PowerMOSFETSignoffRequest, run_power_mosfet_signoff
from tcad_agent.run_queue import (
    QueueStatus,
    claim_next_items,
    default_runner_registry as queue_default_runner_registry,
    enqueue_run,
    get_item,
    recover_owner_running_items,
    run_queue_daemon,
    run_queue_worker,
)
from tcad_agent.sentaurus_deck import apply_sentaurus_semantic_patch_text
from tcad_agent.task_spec import PROJECT_ROOT


class LongRunValidationStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class LongRunValidationSuite(str, Enum):
    QUEUE_SMOKE = "queue_smoke"
    AUTONOMOUS_E2E = "autonomous_e2e"
    ALL = "all"


class LongRunValidationMode(str, Enum):
    SIMULATED = "simulated"
    REAL = "real"


class LongRunValidationRequest(BaseModel):
    validation_id: str | None = None
    validation_root: Path = PROJECT_ROOT / "runs" / "long_run_validation"
    suite: LongRunValidationSuite = LongRunValidationSuite.QUEUE_SMOKE
    mode: LongRunValidationMode = LongRunValidationMode.SIMULATED
    scenario_ids: list[str] = Field(default_factory=list)
    agent_max_steps: int = Field(default=12, ge=1)
    use_llm: bool = False
    allow_llm_fallback: bool = True
    real_agent_request: dict[str, Any] = Field(default_factory=dict)
    queue_goals: list[dict[str, Any]] = Field(default_factory=list)
    poll_interval_seconds: float = 0.0
    max_idle_loops: int = 1


class LongRunScenarioResult(BaseModel):
    scenario_id: str
    title: str
    status: LongRunValidationStatus
    started_at: str
    completed_at: str
    duration_seconds: float
    assertions: list[str] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    failure_reason: str | None = None
    result_path: str | None = None


class LongRunValidationState(BaseModel):
    tool_name: str = "long_run_validation"
    status: LongRunValidationStatus
    validation_id: str
    validation_dir: str
    created_at: str
    updated_at: str
    queue_db_path: str
    queued_items: list[dict[str, Any]] = Field(default_factory=list)
    daemon_result: dict[str, Any] | None = None
    benchmark_results: list[dict[str, Any]] = Field(default_factory=list)
    scenario_results: list[dict[str, Any]] = Field(default_factory=list)
    index_summary: dict[str, Any] | None = None
    indexed_records: list[dict[str, Any]] = Field(default_factory=list)
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_validation_id() -> str:
    return f"longrun_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def default_queue_goals(validation_dir: Path) -> list[dict[str, Any]]:
    run_root = validation_dir / "agent_tools"
    return [
        {
            "queue_id": "longrun_schottky",
            "tool_name": "extended_device_sweep",
            "request": {
                "device_type": "schottky_diode",
                "fidelity": "devsim_1d",
                "evidence_level": "tcad_executable",
                "start": -0.1,
                "stop": 0.1,
                "step": 0.1,
                "run_id": "longrun_schottky",
                "run_root": str(run_root),
            },
            "priority": 10,
        },
        {
            "queue_id": "longrun_power_mosfet",
            "tool_name": "extended_device_sweep",
            "request": {
                "device_type": "power_mosfet_bv_ron",
                "fidelity": "physics_1d",
                "evidence_level": "tcad_executable",
                "run_id": "longrun_power_mosfet",
                "run_root": str(run_root),
            },
            "priority": 5,
        },
        {
            "queue_id": "longrun_bjt",
            "tool_name": "extended_device_sweep",
            "request": {
                "device_type": "bjt_gummel_output",
                "fidelity": "physics_1d",
                "evidence_level": "tcad_executable",
                "run_id": "longrun_bjt",
                "run_root": str(run_root),
            },
            "priority": 4,
        },
        {
            "queue_id": "longrun_power_mosfet_convergence",
            "tool_name": "tool_convergence",
            "request": {
                "convergence_id": "longrun_power_mosfet_convergence",
                "tool_name": "extended_device_sweep",
                "base_request": {
                    "device_type": "power_mosfet_bv_ron",
                    "fidelity": "physics_1d",
                    "evidence_level": "tcad_executable",
                    "start": 0.0,
                    "stop": -90.0,
                    "step": 5.0,
                },
                "axis_path": "power_mos_drift_region_doping_cm3",
                "values": [5.0e15, 1.0e16, 2.0e16],
                "metric_path": "quality_report.metrics.specific_on_resistance_ohm_cm2",
                "relative_tolerance": 0.25,
                "execute": True,
                "convergence_root": str(validation_dir / "tool_convergence"),
                "overwrite": True,
            },
            "priority": 3,
        },
        {
            "queue_id": "longrun_bjt_convergence",
            "tool_name": "tool_convergence",
            "request": {
                "convergence_id": "longrun_bjt_convergence",
                "tool_name": "extended_device_sweep",
                "base_request": {
                    "device_type": "bjt_gummel_output",
                    "fidelity": "physics_1d",
                    "evidence_level": "tcad_executable",
                    "start": 0.55,
                    "stop": 0.8,
                    "step": 0.025,
                },
                "axis_path": "bjt_base_width_um",
                "values": [0.15, 0.2, 0.3],
                "metric_path": "quality_report.metrics.current_gain_beta",
                "relative_tolerance": 0.3,
                "execute": True,
                "convergence_root": str(validation_dir / "tool_convergence"),
                "overwrite": True,
            },
            "priority": 2,
        },
    ]


def write_state(state: LongRunValidationState, path: Path) -> None:
    state.updated_at = utc_timestamp()
    write_json(path, state.model_dump(mode="json"))


def require(condition: bool, message: str) -> str:
    if not condition:
        raise RuntimeError(message)
    return message


def artifact(name: str, path: Path | str | None, *, kind: str = "file", description: str | None = None) -> dict[str, Any]:
    if not path:
        return {"name": name, "path": None, "kind": kind, "description": description, "exists": False}
    resolved = Path(str(path))
    return {
        "name": name,
        "path": str(resolved.resolve()) if resolved.exists() else str(resolved),
        "kind": kind,
        "description": description,
        "exists": resolved.exists(),
    }


def path_exists(value: Any) -> bool:
    if not value:
        return False
    return Path(str(value)).exists()


def runner_state_path(result: dict[str, Any], *, label: str) -> Path:
    raw_state = result.get("state_path")
    if raw_state and Path(str(raw_state)).exists():
        return Path(str(raw_state))
    raw_run_dir = result.get("run_dir")
    if raw_run_dir:
        candidate = Path(str(raw_run_dir)) / "state.json"
        if candidate.exists():
            return candidate
    raise RuntimeError(f"{label} did not emit a readable state path")


def write_curve_state(
    state_path: Path,
    *,
    tool_name: str,
    run_id: str,
    request: dict[str, Any],
    quality_status: str,
    metrics: dict[str, Any],
    csv_header: str = "voltage_v,current_a,electric_field_v_per_cm",
    csv_rows: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    curve_path = state_path.parent / "curve.csv"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    rows = csv_rows or ["0,0,0", "1,1e-9,1e5"]
    curve_path.write_text("\n".join([csv_header, *rows]) + "\n", encoding="utf-8")
    payload: dict[str, Any] = {
        "tool_name": tool_name,
        "status": "completed",
        "run_id": run_id,
        "request": request,
        "run_dir": str(state_path.parent),
        "final_summary": {"artifacts": {"csv": str(curve_path)}, "metrics": metrics},
        "quality_report": {
            "status": quality_status,
            "issues": [{"code": "long_run_validation_suspicious_state", "severity": "warning"}]
            if quality_status != "passed"
            else [],
            "metrics": metrics,
        },
    }
    if extra:
        payload.update(extra)
    write_json(state_path, payload)
    return state_path


def scenario_agent_confirmation_pause(
    scenario_dir: Path,
    request: LongRunValidationRequest,
    queue_db: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    del queue_db
    source_deck = scenario_dir / "unmatched_user_deck.py"
    source_deck.parent.mkdir(parents=True, exist_ok=True)
    source_deck.write_text("solve(type='dc')\n", encoding="utf-8")
    tool_calls: list[dict[str, Any]] = []
    state = run_autonomous_devsim_agent(
        AutonomousDevsimRequest(
            goal_text="Validate unverified semantic deck patch confirmation pause.",
            agent_id="confirmation_pause_agent",
            agent_root=scenario_dir / "agents",
            execute=True,
            use_llm=False,
            max_steps=min(max(request.agent_max_steps, 4), 8),
            source_deck_path=str(source_deck),
            deck_patches=[
                {
                    "deck_path": "geometry.field_plate_length_um",
                    "request_path": "power_mos_field_plate_length_um",
                    "value": 2.0,
                }
            ],
            allow_user_confirmation_actions=True,
            allow_unverified_deck_patch_execution=False,
            initial_tool_name="extended_device_sweep",
            generate_report=False,
            generate_dashboard=False,
        ),
        runner_registry={"extended_device_sweep": lambda tool_request: tool_calls.append(tool_request) or {"status": "completed"}},
    )
    assertions = [
        require(state.status == DevsimAgentStatus.WAITING_FOR_USER, "agent paused for user confirmation"),
        require(state.steps[-1].kind == DevsimAgentActionKind.ASK_USER, "agent converted unverified patch into ask_user action"),
        require(not tool_calls, "patched user deck was not executed before confirmation"),
        require(bool(state.checkpoint.get("deck_patch_unverified")), "unverified deck patch lineage was recorded"),
    ]
    return (
        assertions,
        [
            artifact("agent_state", Path(state.agent_dir) / "autonomous_devsim_agent_state.json"),
            artifact("heartbeat", state.heartbeat_path),
            artifact("semantic_deck_diff", state.checkpoint.get("semantic_deck_diff")),
            artifact("patched_source_deck", state.checkpoint.get("patched_source_deck")),
        ],
        {
            "agent_status": state.status,
            "step_kinds": [step.kind.value for step in state.steps],
            "unverified_patches": state.checkpoint.get("deck_patch_unverified") or [],
        },
    )


def scenario_agent_cancel_boundary(
    scenario_dir: Path,
    request: LongRunValidationRequest,
    queue_db: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    del request, queue_db
    cancel_file = scenario_dir / "agents" / "cancel_agent" / "cancel.requested"
    cancel_file.parent.mkdir(parents=True, exist_ok=True)
    cancel_file.write_text(json.dumps({"reason": "validation cancel before first step"}), encoding="utf-8")
    state = run_autonomous_devsim_agent(
        AutonomousDevsimRequest(
            goal_text="Validate cancellation before the next autonomous step.",
            agent_id="cancel_agent",
            agent_root=scenario_dir / "agents",
            execute=True,
            use_llm=False,
            cancel_file=cancel_file,
            heartbeat_path=scenario_dir / "agents" / "cancel_agent" / "heartbeat.json",
            initial_tool_name="extended_device_sweep",
            generate_report=False,
            generate_dashboard=False,
        ),
        runner_registry={"extended_device_sweep": lambda tool_request: {"status": "completed"}},
    )
    assertions = [
        require(state.status == DevsimAgentStatus.CANCELLED, "agent stopped on cancel token before work"),
        require(len(state.steps) == 0, "cancelled agent did not start a tool step"),
        require(Path(str(state.heartbeat_path)).exists(), "cancelled agent wrote heartbeat"),
    ]
    return (
        assertions,
        [
            artifact("agent_state", Path(state.agent_dir) / "autonomous_devsim_agent_state.json"),
            artifact("heartbeat", state.heartbeat_path),
            artifact("cancel_file", cancel_file),
        ],
        {"agent_status": state.status, "failure_reason": state.failure_reason},
    )


def scenario_agent_repair_report(
    scenario_dir: Path,
    request: LongRunValidationRequest,
    queue_db: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    del queue_db
    suspicious_state = write_curve_state(
        scenario_dir / "runs" / "initial_suspicious" / "state.json",
        tool_name="pn_junction_iv_sweep",
        run_id="initial_suspicious",
        request={"run_id": "initial_suspicious"},
        quality_status="suspicious",
        metrics={"leakage_current_a": 2e-6, "points": 3},
        csv_rows=["-1,2e-6,2e5", "0,1e-10,0", "1,1e-6,1e5"],
    )
    repaired_state = write_curve_state(
        scenario_dir / "runs" / "repaired_passed" / "state.json",
        tool_name="pn_junction_iv_sweep",
        run_id="repaired_passed",
        request={"run_id": "repaired_passed"},
        quality_status="passed",
        metrics={"leakage_current_a": 5e-8, "points": 3},
        csv_rows=["-1,5e-8,1.1e5", "0,1e-10,0", "1,8e-7,9e4"],
        extra={"repair_context": {"action_name": "validation_fake_repair", "baseline_state_path": str(suspicious_state)}},
    )
    calls: list[str] = []

    def fake_repair_runner(source: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append("repair")
        require(source == suspicious_state, "repair runner received suspicious state")
        require(bool(kwargs.get("use_agent_policy")), "repair runner kept agent policy enabled")
        return {
            "status": "completed",
            "final_state_path": str(repaired_state),
            "current_state_path": str(repaired_state),
            "final_quality_status": "passed",
        }

    report_path = scenario_dir / "report" / "repair_report.md"
    dashboard_path = scenario_dir / "report" / "repair_dashboard.html"

    def fake_report_runner(tool_request: dict[str, Any]) -> dict[str, Any]:
        del tool_request
        calls.append("report")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("# Repair validation\n\nCompleted.\n", encoding="utf-8")
        return {"status": "completed", "report_path": str(report_path)}

    def fake_dashboard_runner(tool_request: dict[str, Any]) -> dict[str, Any]:
        del tool_request
        calls.append("dashboard")
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        dashboard_path.write_text("<html><body>repair validation</body></html>\n", encoding="utf-8")
        return {"status": "completed", "dashboard_path": str(dashboard_path)}

    state = run_autonomous_devsim_agent(
        AutonomousDevsimRequest(
            goal_text="Validate repair, benchmark, report, and dashboard in one autonomous loop.",
            agent_id="repair_report_agent",
            agent_root=scenario_dir / "agents",
            execute=True,
            use_llm=False,
            max_steps=max(request.agent_max_steps, 6),
            initial_tool_name="pn_junction_iv_sweep",
            initial_request={"run_id": "initial_suspicious"},
        ),
        runner_registry={
            "pn_junction_iv_sweep": lambda tool_request: calls.append("tool") or {"status": "completed", "state_path": str(suspicious_state)},
            "physical_benchmark": lambda tool_request: calls.append("benchmark")
            or {"status": "completed", "benchmark_path": str(scenario_dir / "benchmark.json")},
            "experiment_report": fake_report_runner,
            "experiment_dashboard": fake_dashboard_runner,
        },
        repair_runner=fake_repair_runner,
    )
    step_kinds = [step.kind for step in state.steps]
    assertions = [
        require(state.status == DevsimAgentStatus.COMPLETED, "agent completed after repair and reporting"),
        require(DevsimAgentActionKind.RUN_REPAIR_EXECUTOR in step_kinds, "repair executor was selected from suspicious curve state"),
        require(DevsimAgentActionKind.RUN_PHYSICAL_BENCHMARK in step_kinds, "physical benchmark ran after repair"),
        require(DevsimAgentActionKind.GENERATE_REPORT in step_kinds, "engineer-readable report was generated"),
        require(Path(str(state.final_report_path)).exists(), "final report artifact exists"),
        require(Path(str(state.final_dashboard_path)).exists(), "dashboard artifact exists"),
    ]
    return (
        assertions,
        [
            artifact("agent_state", Path(state.agent_dir) / "autonomous_devsim_agent_state.json"),
            artifact("initial_state", suspicious_state),
            artifact("repaired_state", repaired_state),
            artifact("report", state.final_report_path, kind="markdown"),
            artifact("dashboard", state.final_dashboard_path, kind="html"),
        ],
        {"agent_status": state.status, "step_kinds": [kind.value for kind in step_kinds], "calls": calls},
    )


def scenario_mutation_refinement_multiround(
    scenario_dir: Path,
    request: LongRunValidationRequest,
    queue_db: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    del queue_db
    mutation = {
        "name": "field_plate_length_refine",
        "target": "field_plate",
        "request_path": "power_mos_field_plate_length_um",
        "deck_path": "geometry.field_plate_length_um",
        "values": [1.5, 2.0, 2.25, 2.375],
        "requires_user_confirmation": True,
    }
    baseline_state = write_curve_state(
        scenario_dir / "runs" / "baseline_power" / "state.json",
        tool_name="extended_device_sweep",
        run_id="baseline_power",
        request={"power_mos_field_plate_length_um": 1.5},
        quality_status="passed",
        metrics={
            "leakage_current_a": 2e-8,
            "max_electric_field_v_per_cm": 2e5,
            "specific_on_resistance_ohm_cm2": 0.05,
        },
        csv_header="drain_voltage_v,off_current_a,electric_field_v_per_cm",
        csv_rows=["0,1e-10,0", "-10,2e-8,2e5"],
    )
    mutation_state = write_curve_state(
        scenario_dir / "runs" / "mutation_power" / "state.json",
        tool_name="extended_device_sweep",
        run_id="mutation_power",
        request={
            "device_type": "power_mosfet_bv_ron",
            "fidelity": "physics_1d",
            "power_mos_field_plate_length_um": 2.0,
            "tcad_deck_mutations": [mutation],
        },
        quality_status="passed",
        metrics={
            "leakage_current_a": 1e-8,
            "max_electric_field_v_per_cm": 1.5e5,
            "specific_on_resistance_ohm_cm2": 0.05,
        },
        csv_header="drain_voltage_v,off_current_a,electric_field_v_per_cm",
        csv_rows=["0,1e-10,0", "-10,1e-8,1.5e5"],
        extra={
            "tcad_deck_mutations": [mutation],
            "repair_context": {"baseline_state_path": str(baseline_state)},
            "mutation_effect_analysis": {
                "decision": "continue_same_target",
                "worth_continuing": True,
                "recommended_next_target": "field_plate",
                "recommended_next_direction": "increase",
                "baseline_value": 1.5,
                "mutation_value": 2.0,
                "rationale": "field peak improved without Ron tradeoff",
            },
        },
    )
    tool_requests: list[dict[str, Any]] = []

    def fake_extended_device(tool_request: dict[str, Any]) -> dict[str, Any]:
        tool_requests.append(dict(tool_request))
        value = float(tool_request["power_mos_field_plate_length_um"])
        index = len(tool_requests)
        field = 1.4e5 if index == 1 else 1.32e5
        leakage = 8e-9 if index == 1 else 7e-9
        refined_state = write_curve_state(
            scenario_dir / "runs" / f"refined_power_{index}" / "state.json",
            tool_name="extended_device_sweep",
            run_id=f"refined_power_{index}",
            request=dict(tool_request),
            quality_status="passed",
            metrics={
                "leakage_current_a": leakage,
                "max_electric_field_v_per_cm": field,
                "specific_on_resistance_ohm_cm2": 0.05,
            },
            csv_header="drain_voltage_v,off_current_a,electric_field_v_per_cm",
            csv_rows=["0,1e-10,0", f"-10,{leakage},{field}"],
            extra={"tcad_deck_mutations": tool_request.get("tcad_deck_mutations") or [mutation]},
        )
        return {"status": "completed", "state_path": str(refined_state)}

    state = run_autonomous_devsim_agent(
        AutonomousDevsimRequest(
            goal_text="Validate multi-round curve-guided field-plate refinement.",
            agent_id="mutation_refinement_agent",
            agent_root=scenario_dir / "agents",
            execute=True,
            use_llm=False,
            max_steps=max(request.agent_max_steps, 7),
            source_state_path=str(mutation_state),
            max_mutation_refinements=2,
            allow_user_confirmation_actions=True,
            generate_report=False,
            generate_dashboard=False,
        ),
        runner_registry={
            "extended_device_sweep": fake_extended_device,
            "physical_benchmark": lambda tool_request: {"status": "completed", "benchmark_path": str(scenario_dir / "benchmark.json")},
        },
    )
    values = [item.get("power_mos_field_plate_length_um") for item in tool_requests]
    final_state_path = state.final_state_path or state.latest_state_path
    final_state = Path(str(final_state_path)) if final_state_path else None
    final_payload = json.loads(final_state.read_text(encoding="utf-8")) if final_state and final_state.exists() else {}
    artifacts = (final_payload.get("final_summary") or {}).get("artifacts") or {}
    assertions = [
        require(state.status == DevsimAgentStatus.COMPLETED, "agent completed after two mutation refinement rounds"),
        require(state.checkpoint.get("mutation_refinement_runs") == 2, "two mutation refinement runs were executed"),
        require(values == [2.25, 2.375], "curve evidence produced progressively finer field-plate values"),
        require(bool(artifacts.get("baseline_mutation_overlay")), "final refined state contains overlay comparison artifact"),
    ]
    return (
        assertions,
        [
            artifact("agent_state", Path(state.agent_dir) / "autonomous_devsim_agent_state.json"),
            artifact("baseline_state", baseline_state),
            artifact("starting_mutation_state", mutation_state),
            artifact("mutation_refinement_plan", state.checkpoint.get("mutation_refinement_plan_path")),
            artifact("final_refined_state", final_state),
            artifact("final_overlay", artifacts.get("baseline_mutation_overlay"), kind="svg"),
        ],
        {"agent_status": state.status, "refinement_values": values, "step_kinds": [step.kind.value for step in state.steps]},
    )


def write_sentaurus_curve(path: Path, *, leakage: float, breakdown: float, field: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "voltage_v,current_a,electric_field_v_per_cm",
                f"0,{leakage},{field * 0.05}",
                f"-50,{leakage * 5},{field * 0.45}",
                f"{breakdown},1e-6,{field}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def apply_sentaurus_patches_to_project(project_copy: Path, patches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for patch in patches:
        file_name = str(patch.get("file") or "")
        if not file_name:
            records.append({"applied": False, "verified": False, "error": "missing file", "patch": patch})
            continue
        deck_path = project_copy / file_name
        if not deck_path.exists():
            records.append({"applied": False, "verified": False, "error": f"missing deck file: {file_name}", "patch": patch})
            continue
        before = deck_path.read_text(encoding="utf-8")
        after, record, _ = apply_sentaurus_semantic_patch_text(before, patch, source_path=file_name)
        deck_path.write_text(after, encoding="utf-8")
        record["file"] = file_name
        records.append(record)
    return records


def scenario_sentaurus_autonomous_refinement(
    scenario_dir: Path,
    request: LongRunValidationRequest,
    queue_db: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    del queue_db
    fixture = PROJECT_ROOT / "tcad_agent" / "examples" / "sentaurus_fixtures" / "power_diode_bv"
    source_project = scenario_dir / "source_project"
    shutil.copytree(fixture, source_project)
    calls: list[dict[str, Any]] = []

    def lifetime_value(project_copy: Path) -> float:
        text = (project_copy / "device.cmd").read_text(encoding="utf-8")
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) == 3 and parts[0] == "set" and parts[1] == "LIFETIME_SCALE":
                return float(parts[2])
        return 1.0

    def fake_sentaurus(tool_request: dict[str, Any]) -> dict[str, Any]:
        calls.append(dict(tool_request))
        run_index = len(calls)
        run_dir = scenario_dir / "sentaurus_runs" / f"sentaurus_{run_index:03d}"
        project_copy = run_dir / "project"
        shutil.copytree(source_project, project_copy)
        patches = [patch for patch in tool_request.get("patches") or [] if isinstance(patch, dict)]
        patch_records = apply_sentaurus_patches_to_project(project_copy, patches)
        lifetime = lifetime_value(project_copy)
        leakage = 1e-9 / max(lifetime, 1.0)
        field = 8e5 * (1.0 - min((lifetime - 1.0) * 0.02, 0.04))
        breakdown = -100.0 - min((lifetime - 1.0) * 2.0, 4.0)
        curve_path = project_copy / "power_diode_bv_extract.csv"
        write_sentaurus_curve(curve_path, leakage=leakage, breakdown=breakdown, field=field)
        log_path = project_copy / "power_diode_bv_des.log"
        log_path.write_text("Sentaurus fake validation backend completed\n", encoding="utf-8")
        state_path = run_dir / "sentaurus_state.json"
        metrics = {
            "solver_backend": "sentaurus",
            "tcad_solver_invoked": True,
            "curve_points": 3,
            "curve_x_key": "voltage_v",
            "curve_y_key": "current_a",
            "curve_field_key": "electric_field_v_per_cm",
            "breakdown_current_threshold_a": 1e-6,
            "leakage_abs_current_at_target_a": leakage,
            "breakdown_voltage_at_threshold_v": breakdown,
            "max_electric_field_v_per_cm": field,
            "specific_on_resistance_ohm_cm2": 0.05,
            "sentaurus_patches_verified": sum(1 for record in patch_records if record.get("verified")),
        }
        write_json(
            state_path,
            {
                "tool_name": "sentaurus_run",
                "status": "completed",
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "project_path": str(source_project),
                "project_copy_path": str(project_copy),
                "request": {
                    "goal_text": tool_request.get("goal_text"),
                    "project_path": str(source_project),
                    "deck_files": tool_request.get("deck_files") or ["device.cmd"],
                    "patches": patches,
                },
                "quality_report": {"status": "passed", "issues": [], "metrics": metrics},
                "final_summary": {
                    "artifacts": {
                        "project_copy": str(project_copy),
                        "sentaurus_curve_csv": str(curve_path),
                        "log": str(log_path),
                    },
                    "metrics": metrics,
                    "parameters": {"deck_files": ["device.cmd"]},
                },
                "sentaurus_patch_records": patch_records,
            },
        )
        return {"status": "completed", "state_path": str(state_path)}

    benchmark_path = scenario_dir / "sentaurus_benchmark.json"

    def fake_benchmark(tool_request: dict[str, Any]) -> dict[str, Any]:
        write_json(
            benchmark_path,
            {
                "tool_name": "physical_benchmark",
                "status": "completed",
                "source_state_path": tool_request.get("source"),
                "counts": {"errors": 0, "warnings": 0},
            },
        )
        return {"status": "completed", "benchmark_path": str(benchmark_path)}

    state = run_autonomous_devsim_agent(
        AutonomousDevsimRequest(
            goal_text="Use Sentaurus contract mode to reduce reverse leakage while preserving BV/Ron/field peak.",
            agent_id="sentaurus_autonomous_refinement_agent",
            agent_root=scenario_dir / "agents",
            execute=True,
            use_llm=False,
            max_steps=max(request.agent_max_steps, 8),
            sentaurus_project_path=source_project,
            sentaurus_request={"flow": ["sdevice"], "deck_files": ["device.cmd"]},
            enable_experiment_design=True,
            max_experiment_design_rounds=2,
            generate_report=False,
            generate_dashboard=False,
        ),
        runner_registry={"sentaurus_run": fake_sentaurus, "physical_benchmark": fake_benchmark},
    )
    final_state = Path(str(state.final_state_path or state.latest_state_path))
    final_payload = json.loads(final_state.read_text(encoding="utf-8")) if final_state.exists() else {}
    artifacts = (final_payload.get("final_summary") or {}).get("artifacts") or {}
    lineage = final_payload.get("sentaurus_lineage_archive") if isinstance(final_payload.get("sentaurus_lineage_archive"), dict) else {}
    patch_values = [
        patch.get("value")
        for call in calls[1:]
        for patch in call.get("patches", [])
        if isinstance(patch, dict) and patch.get("variable") == "LIFETIME_SCALE"
    ]
    step_kinds = [step.kind for step in state.steps]
    assertions = [
        require(state.status == DevsimAgentStatus.COMPLETED, "Sentaurus autonomous contract scenario completed"),
        require(len(calls) == 3, "baseline, first patch, and refined patch Sentaurus runs executed"),
        require(DevsimAgentActionKind.PLAN_SENTAURUS_PATCH in step_kinds, "Sentaurus patch planner ran"),
        require(DevsimAgentActionKind.PLAN_SENTAURUS_REFINEMENT in step_kinds, "Sentaurus patch refiner ran from curve evidence"),
        require(patch_values[-2:] == ["2", "2.5"], "Sentaurus patch values progressed from first patch to half-step refinement"),
        require(len(lineage.get("entries") or []) == 3, "Sentaurus lineage archive contains baseline plus two patch runs"),
        require(bool(artifacts.get("sentaurus_lineage_archive")), "final Sentaurus state links lineage archive"),
        require(bool(artifacts.get("sentaurus_baseline_mutation_overlay")), "final Sentaurus state links curve overlay"),
    ]
    return (
        assertions,
        [
            artifact("agent_state", Path(state.agent_dir) / "autonomous_devsim_agent_state.json"),
            artifact("source_project", source_project, kind="directory"),
            artifact("final_sentaurus_state", final_state),
            artifact("sentaurus_patch_plan", state.checkpoint.get("sentaurus_patch_plan_path")),
            artifact("sentaurus_refinement_plan", state.checkpoint.get("sentaurus_refinement_plan_path")),
            artifact("sentaurus_lineage_archive", artifacts.get("sentaurus_lineage_archive")),
            artifact("sentaurus_overlay", artifacts.get("sentaurus_baseline_mutation_overlay"), kind="svg"),
        ],
        {
            "agent_status": state.status,
            "step_kinds": [kind.value for kind in step_kinds],
            "sentaurus_run_count": len(calls),
            "patch_values": patch_values,
            "lineage_entries": len(lineage.get("entries") or []),
            "best_lineage_entry": (lineage.get("best_entry") or {}).get("lineage_id"),
        },
    )


def scenario_natural_language_power_marathon(
    scenario_dir: Path,
    request: LongRunValidationRequest,
    queue_db: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    del queue_db
    for generated in ["agent_tools", "agents", "cockpit", "runs", "signoff", "signoff_baselines"]:
        target = scenario_dir / generated
        if target.exists():
            shutil.rmtree(target)
    goal = "AI 长时间自主操作 DEVSIM/Sentaurus 完成功率器件 BV/Ron/漏电/field peak 优化任务"
    route_path = scenario_dir / "agent_goal_route.json"
    route = route_agent_goal(
        AgentGoalRouteRequest(
            goal_text=goal,
            execute=True,
            max_steps=max(request.agent_max_steps, 8),
            run_root=scenario_dir / "route",
        ),
        output_path=route_path,
    )
    initial_request = dict(route.autonomous_request.get("initial_request") or {})
    initial_request.update(
        {
            "device_type": "power_mosfet_bv_ron",
            "fidelity": "devsim_2d_field_plate",
            "evidence_level": "tcad_executable",
            "run_id": "marathon_power_2d",
            "run_root": str(scenario_dir / "agent_tools"),
            "start": 0.0,
            "stop": -20.0,
            "step": 10.0,
            "quality_min_points": 3,
            "timeout_seconds": 180.0,
        }
    )
    agent_payload = dict(route.autonomous_request)
    agent_payload.update(
        {
            "goal_text": goal,
            "agent_id": "natural_language_power_marathon_agent",
            "agent_root": scenario_dir / "agents",
            "execute": True,
            "use_llm": False,
            "allow_llm_fallback": request.allow_llm_fallback,
            "max_steps": max(request.agent_max_steps, 8),
            "initial_tool_name": "extended_device_sweep",
            "initial_request": initial_request,
            "require_capability_audit": True,
            "enable_experiment_design": True,
            "max_experiment_design_rounds": 1,
            "auto_execute_experiment_design": True,
            "generate_report": False,
            "generate_dashboard": False,
        }
    )
    signoff_calls: list[dict[str, Any]] = []

    def fast_power_signoff(tool_request: dict[str, Any]) -> dict[str, Any]:
        payload = dict(tool_request)
        signoff_calls.append(payload)
        payload["execute"] = True
        payload["run_convergence"] = False
        payload["run_root"] = str(scenario_dir / "signoff")
        baseline_request = dict(payload.get("baseline_request") or {})
        baseline_request["run_id"] = f"{payload.get('run_id') or 'marathon_power_signoff'}_baseline_fast"
        baseline_request["run_root"] = str(scenario_dir / "signoff_baselines")
        payload["baseline_request"] = baseline_request
        state = run_power_mosfet_signoff(PowerMOSFETSignoffRequest.model_validate(payload))
        return state.model_dump(mode="json")

    registry = queue_default_runner_registry()
    registry["power_mosfet_signoff"] = fast_power_signoff
    state = run_autonomous_devsim_agent(AutonomousDevsimRequest.model_validate(agent_payload), runner_registry=registry)

    agent_state_path = Path(state.agent_dir) / "autonomous_devsim_agent_state.json"
    final_state = Path(str(state.final_state_path or state.latest_state_path))
    final_payload = json.loads(final_state.read_text(encoding="utf-8")) if final_state.exists() else {}
    signoff_gate = final_payload.get("signoff_gate") if isinstance(final_payload.get("signoff_gate"), dict) else {}
    candidates = state.checkpoint.get("agent_experiment_candidates") if isinstance(state.checkpoint.get("agent_experiment_candidates"), list) else []
    candidate_ids = [item.get("candidate_id") for item in candidates if isinstance(item, dict)]
    selected = state.checkpoint.get("pending_agent_experiment_candidate")
    selected_candidate_id = selected.get("candidate_id") if isinstance(selected, dict) else None
    tool_names = [step.action.get("tool_name") for step in state.steps if isinstance(step.action, dict)]

    cockpit_state_path = scenario_dir / "cockpit" / "marathon_cockpit_state.json"
    cockpit_payload = json.loads(agent_state_path.read_text(encoding="utf-8")) if agent_state_path.exists() else {}
    cockpit_payload["signoff_gate"] = signoff_gate
    summary = cockpit_payload.get("final_summary") if isinstance(cockpit_payload.get("final_summary"), dict) else {}
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    if final_state.exists():
        artifacts["power_mosfet_signoff_state"] = str(final_state.resolve())
    if final_payload.get("artifacts"):
        artifacts.update({f"signoff_{key}": value for key, value in final_payload.get("artifacts", {}).items()})
    summary["artifacts"] = artifacts
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    if signoff_gate:
        metrics["signoff_verdict"] = signoff_gate.get("verdict")
    summary["metrics"] = metrics
    cockpit_payload["final_summary"] = summary
    cockpit_payload["next_action"] = final_payload.get("next_action") or cockpit_payload.get("next_action")
    write_json(cockpit_state_path, cockpit_payload)
    cockpit = generate_agent_cockpit(cockpit_state_path, scenario_dir / "cockpit" / "agent_cockpit.html")

    resume_state_path = write_curve_state(
        scenario_dir / "runs" / "resume_probe" / "state.json",
        tool_name="extended_device_sweep",
        run_id="resume_probe",
        request={"run_id": "resume_probe"},
        quality_status="passed",
        metrics={"leakage_current_a": 1e-9, "curve_points": 3},
        csv_rows=["0,0,0", "-5,5e-10,5e4", "-10,1e-9,1e5"],
    )
    resume_calls: list[dict[str, Any]] = []

    def resume_runner(tool_request: dict[str, Any]) -> dict[str, Any]:
        resume_calls.append(dict(tool_request))
        return {"status": "completed", "state_path": str(resume_state_path)}

    planned = run_autonomous_devsim_agent(
        AutonomousDevsimRequest(
            goal_text="Plan a resume-boundary probe before executing.",
            agent_id="marathon_resume_probe",
            agent_root=scenario_dir / "agents",
            execute=False,
            use_llm=False,
            max_steps=3,
            initial_tool_name="extended_device_sweep",
            initial_request={"run_id": "resume_probe"},
            generate_report=False,
            generate_dashboard=False,
        ),
        runner_registry={"extended_device_sweep": resume_runner},
    )
    resumed = run_autonomous_devsim_agent(
        AutonomousDevsimRequest(
            goal_text="Plan a resume-boundary probe before executing.",
            agent_id="marathon_resume_probe",
            agent_root=scenario_dir / "agents",
            execute=True,
            resume=True,
            use_llm=False,
            max_steps=5,
            initial_tool_name="extended_device_sweep",
            initial_request={"run_id": "resume_probe"},
            generate_report=False,
            generate_dashboard=False,
        ),
        runner_registry={
            "extended_device_sweep": resume_runner,
            "physical_benchmark": lambda tool_request: {"status": "completed", "benchmark_path": str(scenario_dir / "resume_benchmark.json")},
        },
    )
    cancel_file = scenario_dir / "agents" / "marathon_cancel_probe" / "cancel.requested"
    cancel_file.parent.mkdir(parents=True, exist_ok=True)
    cancel_file.write_text("cancel\n", encoding="utf-8")
    cancelled = run_autonomous_devsim_agent(
        AutonomousDevsimRequest(
            goal_text="Cancel before starting an autonomous tool step.",
            agent_id="marathon_cancel_probe",
            agent_root=scenario_dir / "agents",
            execute=True,
            use_llm=False,
            max_steps=2,
            initial_tool_name="extended_device_sweep",
            cancel_file=cancel_file,
            generate_report=False,
            generate_dashboard=False,
        ),
        runner_registry={},
    )

    step_kinds = [step.kind for step in state.steps]
    assertions = [
        require(route.status == "matched", "natural-language agent goal routed to an executable template"),
        require(route.selected_template_id == "power_mosfet_bv_ron", "router selected the Power MOSFET/LDMOS BV/Ron template"),
        require(route.primary_tool == "autonomous_devsim_agent", "router selected the autonomous agent as primary tool"),
        require(state.status == DevsimAgentStatus.COMPLETED, "natural-language marathon agent completed"),
        require(DevsimAgentActionKind.AUDIT_CAPABILITY in step_kinds, "agent audited capability coverage before running"),
        require("extended_device_sweep" in tool_names, "agent executed the DEVSIM 2D Power MOSFET runner"),
        require("power_mosfet_signoff" in tool_names, "agent selected and executed the Power MOSFET signoff runner"),
        require("power_mosfet_2d_signoff_evidence_pack" in candidate_ids, "experiment design exposed the Power MOSFET signoff evidence candidate"),
        require(selected_candidate_id == "power_mosfet_2d_signoff_evidence_pack", "signoff evidence candidate became the selected pending candidate"),
        require(bool(signoff_calls), "signoff runner was invoked by the autonomous agent"),
        require(signoff_gate.get("verdict") == "conditional", "signoff gate remained conditional until convergence/golden evidence is supplied"),
        require(cockpit.status == "completed", "minimal cockpit was generated from marathon lineage"),
        require(planned.status == DevsimAgentStatus.PLANNED, "resume probe first persisted a planned action"),
        require(resumed.status == DevsimAgentStatus.COMPLETED, "resume probe completed from prior planned state"),
        require(cancelled.status == DevsimAgentStatus.CANCELLED, "cancel probe stopped at an agent step boundary"),
    ]
    return (
        assertions,
        [
            artifact("agent_goal_route", route_path),
            artifact("agent_state", agent_state_path),
            artifact("heartbeat", state.heartbeat_path),
            artifact("final_signoff_state", final_state),
            artifact("cockpit_state", cockpit_state_path),
            artifact("agent_cockpit", cockpit.output_path, kind="html"),
            artifact("resume_agent_state", Path(planned.agent_dir) / "autonomous_devsim_agent_state.json"),
            artifact("cancel_heartbeat", cancelled.heartbeat_path),
        ],
        {
            "route_template": route.selected_template_id,
            "route_runner": route.selected_runner_id,
            "initial_fidelity": initial_request.get("fidelity"),
            "agent_status": state.status,
            "step_kinds": [kind.value for kind in step_kinds],
            "tool_names": [name for name in tool_names if name],
            "experiment_candidate_ids": candidate_ids,
            "selected_experiment_candidate": selected_candidate_id,
            "signoff_call_count": len(signoff_calls),
            "signoff_verdict": signoff_gate.get("verdict"),
            "cockpit_sections": cockpit.sections,
            "resume_status": resumed.status,
            "resume_tool_calls": len(resume_calls),
            "cancel_status": cancelled.status,
        },
    )


def scenario_public_user_deck_acceptance(
    scenario_dir: Path,
    request: LongRunValidationRequest,
    queue_db: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    return run_public_user_deck_acceptance(
        scenario_dir,
        request,
        queue_db,
        agent_id="public_user_deck_acceptance_agent",
        use_llm=False,
        allow_llm_fallback=request.allow_llm_fallback,
        require_live_llm=False,
    )


def decision_evidence_from_steps(steps: list[Any]) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for step in steps:
        if isinstance(step, dict):
            observation = step.get("observation") if isinstance(step.get("observation"), dict) else {}
        else:
            raw_observation = getattr(step, "observation", {})
            observation = raw_observation if isinstance(raw_observation, dict) else {}
        decision = observation.get("agent_decision")
        if isinstance(decision, dict):
            decisions.append(decision)
    models = sorted({str(item.get("model")) for item in decisions if item.get("model")})
    return {
        "decision_count": len(decisions),
        "llm_decision_count": sum(1 for item in decisions if item.get("status") == "completed" and not item.get("fallback_used")),
        "fallback_count": sum(1 for item in decisions if item.get("fallback_used")),
        "raw_response_count": sum(1 for item in decisions if item.get("raw_response")),
        "models": models,
    }


def decision_evidence_from_agent_state(state: Any) -> dict[str, Any]:
    return decision_evidence_from_steps(list(getattr(state, "steps", []) or []))


def run_public_user_deck_acceptance(
    scenario_dir: Path,
    request: LongRunValidationRequest,
    queue_db: Path,
    *,
    agent_id: str,
    use_llm: bool,
    allow_llm_fallback: bool,
    require_live_llm: bool,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    del queue_db
    deck_path = PROJECT_ROOT / "tcad_agent" / "examples" / "user_deck_acceptance" / "pn_diode_acceptance_deck.py"
    if require_live_llm:
        live_llm_config_or_raise()
    previous_root = os.environ.get("ACTSOFT_USER_DECK_ACCEPTANCE_ROOT")
    os.environ["ACTSOFT_USER_DECK_ACCEPTANCE_ROOT"] = str((scenario_dir / "deck_runs").resolve())
    try:
        state = run_autonomous_devsim_agent(
            AutonomousDevsimRequest(
                goal_text="读取公开 PN diode DEVSIM deck，把 N 区掺杂调低后运行并输出验收证据",
                agent_id=agent_id,
                agent_root=scenario_dir / "agents",
                execute=True,
                use_llm=use_llm,
                allow_llm_fallback=allow_llm_fallback,
                max_steps=max(request.agent_max_steps, 6),
                source_deck_path=str(deck_path),
                deck_patches=[
                    {
                        "deck_path": "doping.n_doping_cm3",
                        "request_path": "n_doping_cm3",
                        "value": 8.0e17,
                    }
                ],
                initial_request={"run_root": str(scenario_dir / "user_deck_states")},
                allow_user_confirmation_actions=True,
                generate_report=False,
                generate_dashboard=False,
            )
        )
    finally:
        if previous_root is None:
            os.environ.pop("ACTSOFT_USER_DECK_ACCEPTANCE_ROOT", None)
        else:
            os.environ["ACTSOFT_USER_DECK_ACCEPTANCE_ROOT"] = previous_root

    final_state_path = state.final_state_path or state.latest_state_path
    final_state = Path(str(final_state_path)) if final_state_path else None
    final_payload = json.loads(final_state.read_text(encoding="utf-8")) if final_state and final_state.exists() else {}
    quality = final_payload.get("quality_report") if isinstance(final_payload.get("quality_report"), dict) else {}
    metrics = quality.get("metrics") if isinstance(quality.get("metrics"), dict) else {}
    artifacts = (final_payload.get("final_summary") or {}).get("artifacts") if isinstance(final_payload.get("final_summary"), dict) else {}
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    diff_value = state.checkpoint.get("semantic_deck_diff")
    diff_path = Path(str(diff_value)) if diff_value else None
    diff_text = diff_path.read_text(encoding="utf-8") if diff_path and diff_path.is_file() else ""
    step_kinds = [step.kind for step in state.steps]
    decision_evidence = decision_evidence_from_agent_state(state)

    assertions = [
        require(deck_path.exists(), "public DEVSIM user deck fixture exists"),
        require(
            state.status == DevsimAgentStatus.COMPLETED,
            f"public user deck acceptance agent completed (status={state.status}, failure_reason={state.failure_reason})",
        ),
        require(DevsimAgentActionKind.INGEST_DECK in step_kinds, "agent ingested the public user deck"),
        require(DevsimAgentActionKind.APPLY_DECK_PATCH in step_kinds, "agent applied a semantic deck patch"),
        require(DevsimAgentActionKind.RUN_USER_DECK in step_kinds, "agent executed the patched user deck directly"),
        require(DevsimAgentActionKind.RUN_PHYSICAL_BENCHMARK in step_kinds, "agent benchmarked the user deck result"),
        require(bool(state.checkpoint.get("deck_patch_verified")), "semantic patch was verified against an existing deck binding"),
        require(not state.checkpoint.get("deck_patch_unverified"), "no unverified fallback deck patch was needed"),
        require("-    \"n_doping_cm3\": 1.0e18" in diff_text and "+    \"n_doping_cm3\": 8e+17" in diff_text, "deck diff records the N doping change"),
        require(quality.get("status") == "passed", "patched user deck quality passed"),
        require(metrics.get("n_doping_cm3") == 8.0e17, "patched deck execution reported updated N doping"),
        require(path_exists(artifacts.get("csv")), "patched user deck emitted a CSV artifact"),
        require(path_exists(artifacts.get("plot")), "patched user deck emitted a plot artifact"),
        require(path_exists(state.checkpoint.get("physical_benchmark_path")), "physical benchmark artifact exists"),
    ]
    if require_live_llm:
        assertions.extend(
            [
                require(use_llm, "live acceptance ran with use_llm enabled"),
                require(not allow_llm_fallback, "live acceptance disabled deterministic LLM fallback"),
                require(decision_evidence["decision_count"] >= 5, "agent recorded model decisions across the full user-deck loop"),
                require(decision_evidence["llm_decision_count"] == decision_evidence["decision_count"], "every agent decision came from the LLM"),
                require(decision_evidence["fallback_count"] == 0, "no deterministic fallback was used in live LLM acceptance"),
                require(bool(decision_evidence["models"]), "LLM model name was recorded in decision evidence"),
                require(decision_evidence["raw_response_count"] == decision_evidence["decision_count"], "raw model responses were recorded for every decision"),
            ]
        )
    return (
        assertions,
        [
            artifact("agent_state", Path(state.agent_dir) / "autonomous_devsim_agent_state.json"),
            artifact("source_deck", deck_path),
            artifact("patched_deck", state.checkpoint.get("patched_source_deck")),
            artifact("semantic_deck_diff", state.checkpoint.get("semantic_deck_diff")),
            artifact("deck_ir", state.checkpoint.get("tcad_deck_ir")),
            artifact("final_user_deck_state", final_state),
            artifact("user_deck_csv", artifacts.get("csv"), kind="csv"),
            artifact("user_deck_plot", artifacts.get("plot"), kind="png"),
            artifact("physical_benchmark", state.checkpoint.get("physical_benchmark_path")),
        ],
        {
            "agent_status": state.status,
            "step_kinds": [kind.value for kind in step_kinds],
            "use_llm": use_llm,
            "allow_llm_fallback": allow_llm_fallback,
            "live_llm_required": require_live_llm,
            "llm_decision_evidence": decision_evidence,
            "deck_patch_verified": state.checkpoint.get("deck_patch_verified"),
            "updated_n_doping_cm3": metrics.get("n_doping_cm3"),
            "quality_status": quality.get("status"),
            "final_state_path": str(final_state) if final_state else None,
        },
    )


def scenario_public_user_deck_live_llm_acceptance(
    scenario_dir: Path,
    request: LongRunValidationRequest,
    queue_db: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    return run_public_user_deck_acceptance(
        scenario_dir,
        request,
        queue_db,
        agent_id="public_user_deck_live_llm_acceptance_agent",
        use_llm=True,
        allow_llm_fallback=False,
        require_live_llm=True,
    )


def live_llm_config_or_raise() -> LLMConfig:
    llm_config = LLMConfig.from_env()
    if not llm_config.base_url or not llm_config.model:
        raise RuntimeError(
            "live LLM validation requires ACTSOFT_LLM_BASE_URL and ACTSOFT_LLM_MODEL "
            "or a local runs/llm_settings.json; deterministic fallback is disabled for this scenario."
        )
    return llm_config


def run_curve_decision_eval_acceptance(
    scenario_dir: Path,
    *,
    eval_id: str,
    use_llm: bool,
    allow_llm_fallback: bool,
    require_live_llm: bool,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    if require_live_llm:
        live_llm_config_or_raise()
    eval_root = scenario_dir / "curve_decision_eval"
    if eval_root.exists():
        shutil.rmtree(eval_root)
    result = run_curve_decision_eval(
        CurveDecisionEvalRequest(
            eval_id=eval_id,
            eval_root=eval_root,
            use_llm=use_llm,
            allow_llm_fallback=allow_llm_fallback,
        )
    )
    case_summaries = [
        {
            "case_id": item.case_id,
            "status": item.status,
            "recommended_action": item.recommended_action,
            "recommended_target": item.recommended_target,
            "decision_source": item.decision_source,
            "fallback_used": item.fallback_used,
            "model": item.model,
        }
        for item in result.cases
    ]
    actions_by_case = {item.case_id: item.recommended_action for item in result.cases}
    assertions = [
        require(result.status == CurveDecisionEvalStatus.COMPLETED, "curve decision eval completed"),
        require(result.case_count >= 4, "curve decision eval covered multiple curve-decision cases"),
        require(result.passed_count == result.case_count, "all curve decision cases passed"),
        require(all(path_exists(item.overlay_svg_path) for item in result.cases), "each curve decision case emitted an overlay SVG"),
        require(actions_by_case.get("lifetime_leakage_improved") == "refine_effective_mutation", "lifetime improvement refines same direction"),
        require(actions_by_case.get("field_plate_ron_tradeoff") == "pareto_review_before_next_patch", "field-plate Ron tradeoff triggers Pareto review"),
        require(actions_by_case.get("drift_doping_ron_not_improved") == "switch_mutation_target", "ineffective drift doping switches target"),
        require(actions_by_case.get("nonmonotonic_curve_requires_repair") == "repair_curve_shape", "nonmonotonic curve triggers bias/mesh repair"),
    ]
    if require_live_llm:
        assertions.extend(
            [
                require(use_llm, "curve decision eval ran with use_llm enabled"),
                require(not allow_llm_fallback, "live curve decision eval disabled deterministic fallback"),
                require(result.llm_decision_count == result.case_count, "every curve decision came from the LLM"),
                require(result.fallback_count == 0, "no deterministic fallback was used in live curve decision eval"),
                require(result.raw_response_count == result.case_count, "raw LLM responses were recorded for every curve case"),
                require(bool(result.models), "LLM model name was recorded for curve decision eval"),
            ]
        )
    artifacts = [
        artifact("curve_decision_eval_result", result.result_path, description="Full curve decision evaluation result"),
    ]
    artifacts.extend(
        artifact(f"{item.case_id}_overlay", item.overlay_svg_path, kind="svg", description=item.title)
        for item in result.cases
    )
    return (
        assertions,
        artifacts,
        {
            "eval_status": result.status,
            "case_count": result.case_count,
            "passed_count": result.passed_count,
            "failed_count": result.failed_count,
            "use_llm": result.use_llm,
            "allow_llm_fallback": result.allow_llm_fallback,
            "llm_decision_count": result.llm_decision_count,
            "fallback_count": result.fallback_count,
            "raw_response_count": result.raw_response_count,
            "models": result.models,
            "cases": case_summaries,
            "result_path": result.result_path,
        },
    )


def scenario_public_curve_decision_eval(
    scenario_dir: Path,
    request: LongRunValidationRequest,
    queue_db: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    del request, queue_db
    return run_curve_decision_eval_acceptance(
        scenario_dir,
        eval_id="public_curve_decision_eval",
        use_llm=False,
        allow_llm_fallback=True,
        require_live_llm=False,
    )


def scenario_public_curve_decision_live_llm_eval(
    scenario_dir: Path,
    request: LongRunValidationRequest,
    queue_db: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    del request, queue_db
    return run_curve_decision_eval_acceptance(
        scenario_dir,
        eval_id="public_curve_decision_live_llm_eval",
        use_llm=True,
        allow_llm_fallback=False,
        require_live_llm=True,
    )


def scenario_public_curve_decision_live_llm_agent_loop(
    scenario_dir: Path,
    request: LongRunValidationRequest,
    queue_db: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    del queue_db
    live_llm_config_or_raise()
    source_state = write_curve_state(
        scenario_dir / "source" / "state.json",
        tool_name="extended_device_sweep",
        run_id="curve_decision_loop_source",
        request={
            "device_type": "power_mosfet_bv_ron",
            "fidelity": "devsim_2d_field_plate",
            "power_mos_drift_region_doping_cm3": 1.0e16,
        },
        quality_status="passed",
        metrics={
            "leakage_current_a": 1.0e-8,
            "breakdown_voltage_v": -80.0,
            "specific_on_resistance_ohm_cm2": 5.0e-3,
            "max_electric_field_v_per_cm": 2.8e5,
        },
        csv_header="drain_voltage_v,off_current_a,electric_field_v_per_cm",
        csv_rows=["0,1e-10,1e4", "-20,1e-8,1.8e5", "-40,1e-6,2.8e5"],
        extra={
            "mutation_effect_analysis": {
                "mutation_target": "drift_doping",
                "primary_metric": "specific_on_resistance_ohm_cm2",
                "primary_improved": True,
                "worth_continuing": True,
                "decision": "continue_same_target",
                "rationale": "Ron improved without blocking tradeoffs after the drift doping probe.",
                "recommended_next_target": "drift_doping",
                "recommended_next_direction": "decrease",
                "improved_metrics": ["specific_on_resistance_ohm_cm2"],
                "regressed_metrics": [],
                "tradeoff_violations": [],
            }
        },
    )
    refined_state = scenario_dir / "refined" / "state.json"
    tool_requests: list[dict[str, Any]] = []

    def fake_extended_device(tool_request: dict[str, Any]) -> dict[str, Any]:
        tool_requests.append(tool_request)
        write_curve_state(
            refined_state,
            tool_name="extended_device_sweep",
            run_id="curve_decision_loop_refined",
            request=tool_request,
            quality_status="passed",
            metrics={
                "leakage_current_a": 8.0e-9,
                "breakdown_voltage_v": -82.0,
                "specific_on_resistance_ohm_cm2": 4.6e-3,
                "max_electric_field_v_per_cm": 2.65e5,
            },
            csv_header="drain_voltage_v,off_current_a,electric_field_v_per_cm",
            csv_rows=["0,8e-11,9e3", "-20,8e-9,1.6e5", "-40,8e-7,2.65e5"],
        )
        return {"status": "completed", "state_path": str(refined_state)}

    state = run_autonomous_devsim_agent(
        AutonomousDevsimRequest(
            goal_text="Use the LLM curve reviewer to refine the Power MOSFET drift doping patch from mutation-effect curve evidence.",
            agent_id="public_curve_decision_live_llm_agent_loop",
            agent_root=scenario_dir / "agents",
            execute=True,
            use_llm=True,
            allow_llm_fallback=False,
            source_state_path=str(source_state),
            max_steps=max(6, request.agent_max_steps),
            max_mutation_refinements=1,
            allow_user_confirmation_actions=True,
            generate_report=False,
            generate_dashboard=False,
        ),
        runner_registry={
            "extended_device_sweep": fake_extended_device,
            "physical_benchmark": lambda tool_request: {
                "status": "completed",
                "benchmark_path": str(scenario_dir / "physical_benchmark.json"),
            },
        },
    )
    step_kinds = [step.kind for step in state.steps]
    decision_evidence = decision_evidence_from_agent_state(state)
    curve_plan = state.checkpoint.get("latest_curve_decision_plan") if isinstance(state.checkpoint.get("latest_curve_decision_plan"), dict) else {}
    assertions = [
        require(
            state.status == DevsimAgentStatus.COMPLETED,
            f"live curve-decision agent loop completed (status={state.status}, failure_reason={state.failure_reason})",
        ),
        require(DevsimAgentActionKind.PLAN_CURVE_DECISION in step_kinds, "agent planned a curve decision from mutation-effect evidence"),
        require(DevsimAgentActionKind.PLAN_GUIDANCE_PATCH in step_kinds, "agent converted curve decision into a guidance patch"),
        require(DevsimAgentActionKind.RUN_TOOL in step_kinds, "agent executed the curve-selected next tool request"),
        require(DevsimAgentActionKind.RUN_PHYSICAL_BENCHMARK in step_kinds, "agent benchmarked the refined result before stopping"),
        require(bool(tool_requests), "curve-selected runner request was executed"),
        require(bool(tool_requests[0].get("guidance_patch_id")), "executed request carried guidance patch lineage"),
        require(curve_plan.get("decision_source") == "llm", "curve decision itself came from the LLM"),
        require(not curve_plan.get("fallback_used"), "curve decision planner did not use deterministic fallback"),
        require(bool(curve_plan.get("raw_response")), "curve decision raw LLM response was recorded"),
        require(decision_evidence["decision_count"] >= 4, "agent recorded model decisions across the loop"),
        require(decision_evidence["llm_decision_count"] == decision_evidence["decision_count"], "every outer agent action came from the LLM"),
        require(decision_evidence["fallback_count"] == 0, "outer agent action selection used no fallback"),
        require(decision_evidence["raw_response_count"] == decision_evidence["decision_count"], "raw model responses were recorded for every outer agent action"),
    ]
    return (
        assertions,
        [
            artifact("agent_state", Path(state.agent_dir) / "autonomous_devsim_agent_state.json"),
            artifact("curve_decision_plan", curve_plan.get("output_path")),
            artifact("guidance_patch_plan", state.checkpoint.get("guidance_patch_plan_path")),
            artifact("refined_state", refined_state),
        ],
        {
            "agent_status": state.status,
            "step_kinds": [kind.value for kind in step_kinds],
            "decision_evidence": decision_evidence,
            "curve_decision_source": curve_plan.get("decision_source"),
            "curve_decision_action": curve_plan.get("recommended_action"),
            "curve_decision_target": curve_plan.get("recommended_target"),
            "curve_decision_fallback_used": curve_plan.get("fallback_used"),
            "executed_request_has_guidance_patch": bool(tool_requests and tool_requests[0].get("guidance_patch_id")),
        },
    )


def live_llm_devsim_soak_options(request: LongRunValidationRequest) -> dict[str, Any]:
    raw = request.real_agent_request.get("public_curve_decision_live_llm_devsim_soak")
    options = raw if isinstance(raw, dict) else {}
    return {
        "duration_hours": float(options.get("duration_hours", 0.0)),
        "max_steps": max(6, int(options.get("max_steps", max(request.agent_max_steps, 6)))),
        "step_slice": max(1, int(options.get("step_slice", 2))),
    }


def scenario_public_curve_decision_live_llm_devsim_soak(
    scenario_dir: Path,
    request: LongRunValidationRequest,
    queue_db: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    del queue_db
    live_llm_config_or_raise()
    options = live_llm_devsim_soak_options(request)
    for stale_dir in ["real_devsim_runs", "soak"]:
        target = scenario_dir / stale_dir
        if target.exists():
            shutil.rmtree(target)

    real_registry = queue_default_runner_registry()
    extended_runner = real_registry["extended_device_sweep"]
    run_root = scenario_dir / "real_devsim_runs"
    base_request = {
        "device_type": "power_mosfet_bv_ron",
        "fidelity": "devsim_2d_field_plate",
        "evidence_level": "tcad_executable",
        "run_root": str(run_root),
        "start": 0.0,
        "stop": -20.0,
        "step": 10.0,
        "quality_min_points": 3,
        "timeout_seconds": 180.0,
        "power_mos_drift_region_doping_cm3": 1.0e16,
    }
    baseline_result = extended_runner({**base_request, "run_id": "devsim_soak_baseline"})
    mutation_result = extended_runner(
        {
            **base_request,
            "run_id": "devsim_soak_mutation_high_drift_doping",
            "power_mos_drift_region_doping_cm3": 2.0e16,
        }
    )
    baseline_state = runner_state_path(baseline_result, label="baseline extended_device_sweep")
    mutation_state = runner_state_path(mutation_result, label="mutation extended_device_sweep")
    effect = compare_state_mutation_effect(
        baseline_state,
        mutation_state,
        deck_patch={
            "target": "drift_doping",
            "request_path": "power_mos_drift_region_doping_cm3",
            "baseline_value": 1.0e16,
            "value": 2.0e16,
        },
        issue_codes=["ron_high"],
        overlay_output_path=scenario_dir / "seed_baseline_mutation_overlay.svg",
    ).model_dump(mode="json")
    effect_path = scenario_dir / "seed_mutation_effect_analysis.json"
    write_json(effect_path, effect)
    mutation_payload = json.loads(mutation_state.read_text(encoding="utf-8"))
    mutation_payload["mutation_effect_analysis"] = effect
    summary = mutation_payload.get("final_summary") if isinstance(mutation_payload.get("final_summary"), dict) else {}
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    artifacts["baseline_mutation_overlay"] = effect.get("overlay_svg_path")
    artifacts["mutation_effect_analysis"] = str(effect_path.resolve())
    summary["artifacts"] = artifacts
    mutation_payload["final_summary"] = summary
    write_json(mutation_state, mutation_payload)

    soak_id = "public_curve_decision_live_llm_devsim_soak"
    soak_state = run_agent_soak(
        AgentSoakRequest(
            goal_text=(
                "Live LLM agent_soak must review real DEVSIM Power MOSFET baseline-vs-mutation curves, "
                "choose the next drift-doping patch, execute the real runner, benchmark, and stop."
            ),
            soak_id=soak_id,
            soak_root=scenario_dir / "soak",
            execute=True,
            duration_hours=options["duration_hours"],
            max_steps=options["max_steps"],
            step_slice=options["step_slice"],
            poll_interval_seconds=0.0,
            memory_path=scenario_dir / "agent_memory.jsonl",
            compile_mission_spec=False,
            enable_recovery=False,
            enable_curve_guidance=True,
            auto_execute_curve_guidance=True,
            max_curve_guided_patches=1,
            autonomous_request={
                "source_state_path": str(mutation_state),
                "use_llm": True,
                "allow_llm_fallback": False,
                "allow_user_confirmation_actions": True,
                "max_mutation_refinements": 1,
                "generate_report": False,
                "generate_dashboard": False,
            },
        )
    )

    agent_state_path = Path(str(soak_state.agent_state_path)) if soak_state.agent_state_path else None
    agent_state = None
    if agent_state_path and agent_state_path.exists():
        agent_state = json.loads(agent_state_path.read_text(encoding="utf-8"))
    agent_steps = agent_state.get("steps") if isinstance(agent_state, dict) else []
    step_kinds = [str((step.get("kind") if isinstance(step, dict) else "")) for step in agent_steps if isinstance(step, dict)]
    checkpoint = agent_state.get("checkpoint") if isinstance(agent_state, dict) and isinstance(agent_state.get("checkpoint"), dict) else {}
    curve_plan = checkpoint.get("latest_curve_decision_plan") if isinstance(checkpoint.get("latest_curve_decision_plan"), dict) else {}
    final_state = Path(str(soak_state.final_state_path)) if soak_state.final_state_path else None
    final_payload = json.loads(final_state.read_text(encoding="utf-8")) if final_state and final_state.exists() else {}
    final_request = final_payload.get("request") if isinstance(final_payload.get("request"), dict) else {}
    final_repair_context = final_payload.get("repair_context") if isinstance(final_payload.get("repair_context"), dict) else {}
    final_deck_patch = final_repair_context.get("deck_patch") if isinstance(final_repair_context.get("deck_patch"), dict) else {}
    final_metrics = ((final_payload.get("quality_report") or {}).get("metrics") or {}) if isinstance(final_payload.get("quality_report"), dict) else {}
    final_artifacts = ((final_payload.get("final_summary") or {}).get("artifacts") or {}) if isinstance(final_payload.get("final_summary"), dict) else {}
    cycle_statuses = [cycle.status for cycle in soak_state.cycles]

    assertions = [
        require(soak_state.status == AgentSoakStatus.COMPLETED, f"live LLM DEVSIM soak completed (status={soak_state.status}, failure_reason={soak_state.failure_reason})"),
        require(len(soak_state.cycles) >= 2, "agent_soak crossed at least one resume/slice boundary"),
        require("slice_exhausted" in cycle_statuses, "agent_soak recorded a slice_exhausted resume boundary"),
        require(soak_state.fallback_decisions == 0, "live LLM DEVSIM soak used no deterministic fallback"),
        require(soak_state.model_decisions >= 4, "live LLM DEVSIM soak recorded model decisions across the loop"),
        require("plan_curve_decision" in step_kinds, "soak agent planned a curve decision from real mutation evidence"),
        require("plan_guidance_patch" in step_kinds, "soak agent converted the curve decision into a guidance patch"),
        require("run_tool" in step_kinds, "soak agent executed the curve-selected real runner request"),
        require("run_physical_benchmark" in step_kinds, "soak agent benchmarked the real refined state"),
        require("stop_success" in step_kinds, "soak agent stopped after benchmark evidence"),
        require(curve_plan.get("decision_source") == "llm", "curve decision planner used the live LLM"),
        require(not curve_plan.get("fallback_used"), "curve decision planner used no fallback"),
        require(bool(curve_plan.get("raw_response")), "curve decision planner recorded raw LLM response"),
        require(bool(checkpoint.get("guidance_patch_runs")), "guidance patch execution was recorded in checkpoint"),
        require(str(final_payload.get("run_id") or "").endswith("_guidance_patch"), "final refined run id preserves guidance patch lineage"),
        require(final_request.get("run_id") == final_payload.get("run_id"), "final state request points at the guidance patch run"),
        require(final_repair_context.get("action_name") == "agent_mutation_refinement", "final refined state records agent mutation-refinement context"),
        require(final_deck_patch.get("curve_guidance_action") == "refine_effective_mutation", "final refined state records curve-guided deck patch lineage"),
        require(bool(final_metrics.get("tcad_solver_invoked")), "final refined state records TCAD solver invocation"),
        require(bool(final_metrics.get("devsim_2d_solver_invoked")), "final refined state records DEVSIM 2D solver invocation"),
        require(final_metrics.get("runner_contract_id") == "power_mosfet_bv_ron_devsim_2d_field_plate", "final refined state used the Power MOSFET DEVSIM 2D field-plate runner"),
        require(path_exists(effect.get("overlay_svg_path")), "seed baseline/mutation overlay exists"),
        require(path_exists(final_artifacts.get("baseline_mutation_overlay")), "final guidance patch overlay exists"),
    ]
    return (
        assertions,
        [
            artifact("soak_state", soak_state.state_path),
            artifact("agent_state", soak_state.agent_state_path),
            artifact("baseline_state", baseline_state),
            artifact("mutation_state_with_effect", mutation_state),
            artifact("seed_mutation_effect", effect_path),
            artifact("seed_overlay", effect.get("overlay_svg_path"), kind="svg"),
            artifact("curve_decision_plan", curve_plan.get("output_path")),
            artifact("guidance_patch_plan", checkpoint.get("guidance_patch_plan_path")),
            artifact("final_refined_state", final_state),
            artifact("final_overlay", final_artifacts.get("baseline_mutation_overlay"), kind="svg"),
            artifact("soak_cockpit", soak_state.latest_cockpit_path, kind="html"),
        ],
        {
            "soak_status": soak_state.status,
            "cycle_statuses": cycle_statuses,
            "completed_steps": soak_state.completed_steps,
            "model_decisions": soak_state.model_decisions,
            "fallback_decisions": soak_state.fallback_decisions,
            "step_kinds": step_kinds,
            "curve_decision_source": curve_plan.get("decision_source"),
            "curve_decision_action": curve_plan.get("recommended_action"),
            "curve_decision_target": curve_plan.get("recommended_target"),
            "guidance_patch_runs": checkpoint.get("guidance_patch_runs"),
            "final_run_id": final_payload.get("run_id"),
            "final_curve_guidance_action": final_deck_patch.get("curve_guidance_action"),
            "final_solver_invoked": final_metrics.get("tcad_solver_invoked"),
            "final_devsim_2d_invoked": final_metrics.get("devsim_2d_solver_invoked"),
            "final_runner_contract_id": final_metrics.get("runner_contract_id"),
            "final_state_path": str(final_state) if final_state else None,
        },
    )


def live_llm_soak_options(request: LongRunValidationRequest) -> dict[str, Any]:
    raw = request.real_agent_request.get("public_user_deck_live_llm_soak")
    options = raw if isinstance(raw, dict) else {}
    return {
        "duration_hours": float(options.get("duration_hours", 0.0)),
        "max_steps": max(6, int(options.get("max_steps", max(request.agent_max_steps, 6)))),
        "step_slice": max(1, int(options.get("step_slice", 2))),
    }


def scenario_public_user_deck_live_llm_soak(
    scenario_dir: Path,
    request: LongRunValidationRequest,
    queue_db: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    del queue_db
    live_llm_config_or_raise()
    options = live_llm_soak_options(request)
    deck_path = PROJECT_ROOT / "tcad_agent" / "examples" / "user_deck_acceptance" / "pn_diode_acceptance_deck.py"
    soak_root = scenario_dir / "soak"
    soak_id = "public_user_deck_live_llm_soak"
    for stale_dir in [soak_root / soak_id, scenario_dir / "user_deck_states", scenario_dir / "deck_runs"]:
        if stale_dir.exists():
            shutil.rmtree(stale_dir)
    previous_root = os.environ.get("ACTSOFT_USER_DECK_ACCEPTANCE_ROOT")
    os.environ["ACTSOFT_USER_DECK_ACCEPTANCE_ROOT"] = str((scenario_dir / "deck_runs").resolve())
    try:
        state = run_agent_soak(
            AgentSoakRequest(
                goal_text="长时间自主读取公开 PN diode DEVSIM deck，把 N 区掺杂调低，分片运行并持续输出验收证据",
                soak_id=soak_id,
                soak_root=soak_root,
                execute=True,
                duration_hours=options["duration_hours"],
                max_steps=options["max_steps"],
                step_slice=options["step_slice"],
                poll_interval_seconds=0.0,
                memory_path=scenario_dir / "agent_memory.jsonl",
                compile_mission_spec=False,
                enable_curve_guidance=False,
                autonomous_request={
                    "source_deck_path": str(deck_path),
                    "deck_patches": [
                        {
                            "deck_path": "doping.n_doping_cm3",
                            "request_path": "n_doping_cm3",
                            "value": 8.0e17,
                        }
                    ],
                    "initial_request": {"run_root": str(scenario_dir / "user_deck_states")},
                    "use_llm": True,
                    "allow_llm_fallback": False,
                    "allow_user_confirmation_actions": True,
                    "generate_report": True,
                    "generate_dashboard": False,
                },
            )
        )
    finally:
        if previous_root is None:
            os.environ.pop("ACTSOFT_USER_DECK_ACCEPTANCE_ROOT", None)
        else:
            os.environ["ACTSOFT_USER_DECK_ACCEPTANCE_ROOT"] = previous_root

    agent_state_path = Path(str(state.agent_state_path)) if state.agent_state_path else None
    agent_state = None
    if agent_state_path and agent_state_path.exists():
        agent_state = json.loads(agent_state_path.read_text(encoding="utf-8"))
    final_state_path = Path(str(state.final_state_path)) if state.final_state_path else None
    final_payload = json.loads(final_state_path.read_text(encoding="utf-8")) if final_state_path and final_state_path.exists() else {}
    quality = final_payload.get("quality_report") if isinstance(final_payload.get("quality_report"), dict) else {}
    metrics = quality.get("metrics") if isinstance(quality.get("metrics"), dict) else {}
    final_artifacts = (final_payload.get("final_summary") or {}).get("artifacts") if isinstance(final_payload.get("final_summary"), dict) else {}
    final_artifacts = final_artifacts if isinstance(final_artifacts, dict) else {}
    cycle_statuses = [cycle.status for cycle in state.cycles]
    step_kinds = []
    if isinstance(agent_state, dict):
        step_kinds = [str(step.get("kind")) for step in agent_state.get("steps") or [] if isinstance(step, dict)]
    decision_evidence = decision_evidence_from_steps((agent_state or {}).get("steps", []))

    assertions = [
        require(deck_path.exists(), "public DEVSIM user deck fixture exists"),
        require(state.status == AgentSoakStatus.COMPLETED, f"live LLM soak completed (status={state.status}, failure_reason={state.failure_reason})"),
        require(len(state.cycles) >= 2, "soak split the autonomous mission across multiple cycles"),
        require("slice_exhausted" in cycle_statuses, "soak hit at least one step-slice boundary before completion"),
        require(cycle_statuses[-1] == AgentSoakStatus.COMPLETED, "final soak cycle completed the agent mission"),
        require(state.model_decisions >= 6, "soak recorded live model decisions across the mission"),
        require(state.fallback_decisions == 0, "strict live LLM soak used no deterministic fallback"),
        require(decision_evidence["decision_count"] == state.model_decisions, "agent decision ledger matches soak model-decision count"),
        require(decision_evidence["llm_decision_count"] == decision_evidence["decision_count"], "every soak decision came from the LLM"),
        require(bool(decision_evidence["models"]), "LLM model name was recorded in soak decision evidence"),
        require(DevsimAgentActionKind.INGEST_DECK.value in step_kinds, "soak agent ingested the public deck"),
        require(DevsimAgentActionKind.APPLY_DECK_PATCH.value in step_kinds, "soak agent applied the semantic deck patch"),
        require(DevsimAgentActionKind.RUN_USER_DECK.value in step_kinds, "soak agent executed the patched user deck"),
        require(DevsimAgentActionKind.RUN_PHYSICAL_BENCHMARK.value in step_kinds, "soak agent benchmarked the user deck result"),
        require(
            DevsimAgentActionKind.GENERATE_REPORT.value not in step_kinds
            or step_kinds.index(DevsimAgentActionKind.RUN_PHYSICAL_BENCHMARK.value) < step_kinds.index(DevsimAgentActionKind.GENERATE_REPORT.value),
            "soak generated report only after physical benchmark",
        ),
        require(quality.get("status") == "passed", "final soak user deck quality passed"),
        require(metrics.get("n_doping_cm3") == 8.0e17, "final soak user deck reported updated N doping"),
        require(path_exists(final_artifacts.get("csv")), "final soak user deck emitted a CSV artifact"),
        require(path_exists(final_artifacts.get("plot")), "final soak user deck emitted a plot artifact"),
        require(path_exists(state.heartbeat_path), "soak heartbeat artifact exists"),
        require(path_exists(state.agent_state_path), "nested autonomous agent state artifact exists"),
        require(path_exists(state.latest_cockpit_path), "soak cockpit artifact exists"),
    ]
    return (
        assertions,
        [
            artifact("soak_state", state.state_path),
            artifact("soak_heartbeat", state.heartbeat_path),
            artifact("agent_state", state.agent_state_path),
            artifact("latest_cockpit", state.latest_cockpit_path, kind="html"),
            artifact("source_deck", deck_path),
            artifact("final_user_deck_state", final_state_path),
            artifact("user_deck_csv", final_artifacts.get("csv"), kind="csv"),
            artifact("user_deck_plot", final_artifacts.get("plot"), kind="png"),
        ],
        {
            "soak_status": state.status,
            "cycle_count": len(state.cycles),
            "cycle_statuses": cycle_statuses,
            "completed_steps": state.completed_steps,
            "model_decisions": state.model_decisions,
            "fallback_decisions": state.fallback_decisions,
            "llm_decision_evidence": decision_evidence,
            "step_kinds": step_kinds,
            "updated_n_doping_cm3": metrics.get("n_doping_cm3"),
            "quality_status": quality.get("status"),
            "final_state_path": str(final_state_path) if final_state_path else None,
        },
    )


USER_DECK_CORPUS_CASES: list[dict[str, Any]] = [
    {
        "case_id": "function_wrapped_pn",
        "deck_path": PROJECT_ROOT / "tcad_agent" / "examples" / "user_deck_corpus" / "function_wrapped_pn_deck.py",
        "patch": {"deck_path": "doping.n_doping_cm3", "request_path": "n_doping_cm3", "value": 7.5e17},
        "expected_metric": "n_doping_cm3",
        "expected_value": 7.5e17,
        "shape": "function_wrapped_config",
    },
    {
        "case_id": "imported_defaults_pn",
        "deck_path": PROJECT_ROOT / "tcad_agent" / "examples" / "user_deck_corpus" / "imported_defaults_pn_deck.py",
        "patch": {"deck_path": "geometry.length_um", "request_path": "length_um", "value": 0.14},
        "expected_metric": "length_um",
        "expected_value": 0.14,
        "shape": "package_imports_with_local_overrides",
    },
    {
        "case_id": "multi_sweep_lifetime",
        "deck_path": PROJECT_ROOT / "tcad_agent" / "examples" / "user_deck_corpus" / "multi_sweep_lifetime_deck.py",
        "patch": {"deck_path": "physics_models.electron_lifetime_s", "request_path": "electron_lifetime_s", "value": 1.0e-8},
        "expected_metric": "electron_lifetime_s",
        "expected_value": 1.0e-8,
        "shape": "multi_sweep_bias_sequence",
    },
]


def load_deck_ir_sections(path_value: str | None) -> list[str]:
    if not path_value:
        return []
    path = Path(path_value)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        str(section.get("name"))
        for section in payload.get("sections") or []
        if isinstance(section, dict) and section.get("name")
    ]


def scenario_public_user_deck_corpus_acceptance(
    scenario_dir: Path,
    request: LongRunValidationRequest,
    queue_db: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    del queue_db
    for stale_dir in [scenario_dir / "agents", scenario_dir / "deck_runs", scenario_dir / "user_deck_states"]:
        if stale_dir.exists():
            shutil.rmtree(stale_dir)
    previous_root = os.environ.get("ACTSOFT_USER_DECK_CORPUS_ROOT")
    os.environ["ACTSOFT_USER_DECK_CORPUS_ROOT"] = str((scenario_dir / "deck_runs").resolve())
    assertions: list[str] = []
    artifacts: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []
    try:
        for case in USER_DECK_CORPUS_CASES:
            case_id = str(case["case_id"])
            deck_path = Path(case["deck_path"])
            state = run_autonomous_devsim_agent(
                AutonomousDevsimRequest(
                    goal_text=f"读取真实风格公开 DEVSIM deck corpus case {case_id}，做语义 patch 后运行并输出验收证据",
                    agent_id=f"user_deck_corpus_{case_id}",
                    agent_root=scenario_dir / "agents",
                    execute=True,
                    use_llm=False,
                    allow_llm_fallback=request.allow_llm_fallback,
                    max_steps=max(request.agent_max_steps, 6),
                    source_deck_path=str(deck_path),
                    deck_patches=[case["patch"]],
                    initial_request={"run_root": str(scenario_dir / "user_deck_states" / case_id)},
                    allow_user_confirmation_actions=True,
                    generate_report=False,
                    generate_dashboard=False,
                )
            )
            final_state_path = state.final_state_path or state.latest_state_path
            final_state = Path(str(final_state_path)) if final_state_path else None
            final_payload = json.loads(final_state.read_text(encoding="utf-8")) if final_state and final_state.exists() else {}
            quality = final_payload.get("quality_report") if isinstance(final_payload.get("quality_report"), dict) else {}
            metrics = quality.get("metrics") if isinstance(quality.get("metrics"), dict) else {}
            final_artifacts = (
                (final_payload.get("final_summary") or {}).get("artifacts")
                if isinstance(final_payload.get("final_summary"), dict)
                else {}
            )
            final_artifacts = final_artifacts if isinstance(final_artifacts, dict) else {}
            sections = load_deck_ir_sections(state.checkpoint.get("tcad_deck_ir"))
            step_kinds = [step.kind for step in state.steps]
            expected_metric = str(case["expected_metric"])
            expected_value = case["expected_value"]
            assertions.extend(
                [
                    require(deck_path.exists(), f"{case_id} source deck exists"),
                    require(state.status == DevsimAgentStatus.COMPLETED, f"{case_id} agent completed"),
                    require(DevsimAgentActionKind.INGEST_DECK in step_kinds, f"{case_id} deck was ingested"),
                    require(DevsimAgentActionKind.APPLY_DECK_PATCH in step_kinds, f"{case_id} semantic patch was applied"),
                    require(DevsimAgentActionKind.RUN_USER_DECK in step_kinds, f"{case_id} patched deck was executed"),
                    require(DevsimAgentActionKind.RUN_PHYSICAL_BENCHMARK in step_kinds, f"{case_id} benchmark was executed"),
                    require(bool(state.checkpoint.get("deck_patch_verified")), f"{case_id} semantic patch verified"),
                    require(not state.checkpoint.get("deck_patch_unverified"), f"{case_id} had no unverified patches"),
                    require({"geometry", "doping", "model", "mesh", "bias"}.intersection(sections), f"{case_id} deck IR found semantic sections"),
                    require(quality.get("status") == "passed", f"{case_id} final quality passed"),
                    require(metrics.get(expected_metric) == expected_value, f"{case_id} metric {expected_metric} was patched"),
                    require(path_exists(final_artifacts.get("csv")), f"{case_id} emitted CSV artifact"),
                    require(path_exists(final_artifacts.get("plot")), f"{case_id} emitted plot artifact"),
                    require(path_exists(state.checkpoint.get("physical_benchmark_path")), f"{case_id} benchmark artifact exists"),
                ]
            )
            artifacts.extend(
                [
                    artifact(f"{case_id}_agent_state", Path(state.agent_dir) / "autonomous_devsim_agent_state.json"),
                    artifact(f"{case_id}_source_deck", deck_path),
                    artifact(f"{case_id}_patched_deck", state.checkpoint.get("patched_source_deck")),
                    artifact(f"{case_id}_semantic_deck_diff", state.checkpoint.get("semantic_deck_diff")),
                    artifact(f"{case_id}_deck_ir", state.checkpoint.get("tcad_deck_ir")),
                    artifact(f"{case_id}_final_state", final_state),
                    artifact(f"{case_id}_csv", final_artifacts.get("csv"), kind="csv"),
                    artifact(f"{case_id}_plot", final_artifacts.get("plot"), kind="png"),
                    artifact(f"{case_id}_benchmark", state.checkpoint.get("physical_benchmark_path")),
                ]
            )
            case_results.append(
                {
                    "case_id": case_id,
                    "shape": case["shape"],
                    "status": state.status,
                    "step_kinds": [kind.value for kind in step_kinds],
                    "deck_patch_verified": state.checkpoint.get("deck_patch_verified"),
                    "section_names": sections,
                    "expected_metric": expected_metric,
                    "expected_value": expected_value,
                    "observed_value": metrics.get(expected_metric),
                    "quality_status": quality.get("status"),
                    "final_state_path": str(final_state) if final_state else None,
                }
            )
    finally:
        if previous_root is None:
            os.environ.pop("ACTSOFT_USER_DECK_CORPUS_ROOT", None)
        else:
            os.environ["ACTSOFT_USER_DECK_CORPUS_ROOT"] = previous_root
    return (
        assertions,
        artifacts,
        {
            "case_count": len(case_results),
            "cases": case_results,
            "shapes": [case["shape"] for case in USER_DECK_CORPUS_CASES],
        },
    )


def scenario_queue_confirmation_resume(
    scenario_dir: Path,
    request: LongRunValidationRequest,
    queue_db: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    from tcad_agent.web_app import WebAppConfig, approve_item_confirmation

    source_deck = scenario_dir / "queued_unmatched_deck.py"
    source_deck.parent.mkdir(parents=True, exist_ok=True)
    source_deck.write_text("solve(type='dc')\n", encoding="utf-8")
    final_state = write_curve_state(
        scenario_dir / "runs" / "queued_passed" / "state.json",
        tool_name="extended_device_sweep",
        run_id="queued_passed",
        request={"run_id": "queued_passed"},
        quality_status="passed",
        metrics={"leakage_current_a": 1e-8, "max_electric_field_v_per_cm": 1.2e5},
        csv_rows=["0,1e-10,0", "1,1e-8,1.2e5"],
    )
    tool_calls: list[dict[str, Any]] = []

    def queued_agent_runner(tool_request: dict[str, Any]) -> dict[str, Any]:
        agent_state = run_autonomous_devsim_agent(
            AutonomousDevsimRequest.model_validate(tool_request),
            runner_registry={
                "extended_device_sweep": lambda nested_request: tool_calls.append(nested_request)
                or {"status": "completed", "state_path": str(final_state)},
                "physical_benchmark": lambda nested_request: {
                    "status": "completed",
                    "benchmark_path": str(scenario_dir / "queued_benchmark.json"),
                },
            },
        )
        return agent_state.model_dump(mode="json")

    queue_id = "e2e_confirmation_resume"
    enqueue_run(
        queue_db,
        queue_id=queue_id,
        tool_name="autonomous_devsim_agent",
        request={
            "goal_text": "Queued agent should pause on unverified deck patch, then resume after approval.",
            "agent_root": str(scenario_dir / "agents"),
            "agent_id": "queued_confirmation_agent",
            "execute": True,
            "resume": False,
            "use_llm": False,
            "max_steps": max(request.agent_max_steps, 7),
            "source_deck_path": str(source_deck),
            "deck_patches": [
                {
                    "deck_path": "geometry.field_plate_length_um",
                    "request_path": "power_mos_field_plate_length_um",
                    "value": 2.0,
                }
            ],
            "allow_user_confirmation_actions": True,
            "allow_unverified_deck_patch_execution": False,
            "initial_tool_name": "extended_device_sweep",
            "generate_report": False,
            "generate_dashboard": False,
        },
        priority=50,
        tags=["long_run_validation", "confirmation"],
        max_attempts=2,
    )
    first_worker = run_queue_worker(queue_db, owner="confirmation_worker", registry={"autonomous_devsim_agent": queued_agent_runner})
    paused_item = get_item(queue_db, queue_id)
    require(paused_item is not None, "paused queue item exists")
    approved = approve_item_confirmation(WebAppConfig(root=scenario_dir, queue_db_path=queue_db), queue_id)
    second_worker = run_queue_worker(queue_db, owner="confirmation_worker", registry={"autonomous_devsim_agent": queued_agent_runner})
    completed_item = get_item(queue_db, queue_id)
    require(completed_item is not None, "completed queue item exists")
    assertions = [
        require(first_worker.skipped == 1, "queue worker paused agent when it waited for user confirmation"),
        require(paused_item.status == QueueStatus.PAUSED, "queue item status became paused"),
        require(approved["status"] == QueueStatus.QUEUED, "approval resumed the paused queue item"),
        require(second_worker.completed == 1, "queue worker completed the approved agent"),
        require(completed_item.status == QueueStatus.COMPLETED, "approved queue item completed"),
        require(bool(tool_calls), "approved run executed the patched deck tool"),
        require(completed_item.request.get("allow_unverified_deck_patch_execution") is True, "approval opened unverified patch execution explicitly"),
    ]
    return (
        assertions,
        [
            artifact("agent_state", scenario_dir / "agents" / "queued_confirmation_agent" / "autonomous_devsim_agent_state.json"),
            artifact("heartbeat", scenario_dir / "agents" / "queued_confirmation_agent" / "heartbeat.json"),
            artifact("final_state", final_state),
        ],
        {
            "first_worker": first_worker.model_dump(mode="json"),
            "second_worker": second_worker.model_dump(mode="json"),
            "paused_status": paused_item.status,
            "completed_status": completed_item.status,
            "tool_calls": tool_calls,
        },
    )


def scenario_queue_interruption_recovery(
    scenario_dir: Path,
    request: LongRunValidationRequest,
    queue_db: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    del request
    queue_id = "e2e_interruption_recovery"
    result_state = write_curve_state(
        scenario_dir / "runs" / "recovered" / "state.json",
        tool_name="extended_device_sweep",
        run_id="recovered",
        request={"run_id": "recovered"},
        quality_status="passed",
        metrics={"leakage_current_a": 1e-9},
    )
    enqueue_run(
        queue_db,
        queue_id=queue_id,
        tool_name="autonomous_devsim_agent",
        request={"goal_text": "Recover a worker-owned long-run agent.", "execute": True},
        priority=40,
        tags=["long_run_validation", "recovery"],
        max_attempts=2,
    )
    claimed = claim_next_items(queue_db, owner="dead_worker", limit=1, lease_seconds=3600)
    recovery = recover_owner_running_items(queue_db, owner="dead_worker")
    recovered_item = get_item(queue_db, queue_id)
    worker = run_queue_worker(
        queue_db,
        owner="live_worker",
        registry={
            "autonomous_devsim_agent": lambda tool_request: {
                "status": "completed",
                "state_path": str(result_state),
                "agent_dir": str(scenario_dir / "agents" / "recovered_agent"),
            }
        },
    )
    completed_item = get_item(queue_db, queue_id)
    require(recovered_item is not None, "recovered queue item exists")
    require(completed_item is not None, "completed recovered queue item exists")
    assertions = [
        require(len(claimed) == 1, "interrupted worker claimed one item"),
        require(recovery["recovered"] == 1, "owner-scoped recovery requeued the interrupted item"),
        require(recovered_item.status == QueueStatus.QUEUED, "interrupted item returned to queued state"),
        require(worker.completed == 1, "live worker completed the recovered item"),
        require(completed_item.status == QueueStatus.COMPLETED, "recovered queue item completed"),
    ]
    return (
        assertions,
        [artifact("result_state", result_state)],
        {"recovery": recovery, "worker": worker.model_dump(mode="json")},
    )


def scenario_real_autonomous_agent(
    scenario_dir: Path,
    request: LongRunValidationRequest,
    queue_db: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    del queue_db
    payload = {
        "goal_text": "Run a real autonomous DEVSIM validation mission and produce durable evidence.",
        "agent_id": "real_autonomous_agent",
        "agent_root": scenario_dir / "agents",
        "execute": True,
        "use_llm": request.use_llm,
        "allow_llm_fallback": request.allow_llm_fallback,
        "max_steps": request.agent_max_steps,
        "initial_tool_name": "extended_device_sweep",
        "initial_request": {
            "device_type": "power_mosfet_bv_ron",
            "fidelity": "physics_1d",
            "evidence_level": "tcad_executable",
            "run_id": "real_long_run_power_mosfet",
            "run_root": str(scenario_dir / "real_tools"),
        },
    }
    payload.update(request.real_agent_request)
    state = run_autonomous_devsim_agent(AutonomousDevsimRequest.model_validate(payload))
    acceptable = {DevsimAgentStatus.COMPLETED, DevsimAgentStatus.WAITING_FOR_USER}
    assertions = [
        require(state.status in acceptable, "real autonomous agent completed or stopped at an explicit confirmation gate"),
        require(Path(state.agent_dir, "autonomous_devsim_agent_state.json").exists(), "real agent state artifact exists"),
        require(Path(str(state.heartbeat_path)).exists(), "real agent heartbeat exists"),
        require(bool(state.steps) or state.status == DevsimAgentStatus.WAITING_FOR_USER, "real agent produced steps or a confirmation gate"),
    ]
    return (
        assertions,
        [
            artifact("agent_state", Path(state.agent_dir) / "autonomous_devsim_agent_state.json"),
            artifact("heartbeat", state.heartbeat_path),
            artifact("final_state", state.final_state_path or state.latest_state_path),
            artifact("final_report", state.final_report_path, kind="markdown"),
            artifact("final_dashboard", state.final_dashboard_path, kind="html"),
        ],
        {"agent_status": state.status, "step_kinds": [step.kind.value for step in state.steps], "failure_reason": state.failure_reason},
    )


ScenarioRunner = Callable[[Path, LongRunValidationRequest, Path], tuple[list[str], list[dict[str, Any]], dict[str, Any]]]


SCENARIO_REGISTRY: dict[str, tuple[str, ScenarioRunner]] = {
    "agent_confirmation_pause": ("Agent pauses before executing unverified deck patch", scenario_agent_confirmation_pause),
    "agent_cancel_boundary": ("Agent observes cancel token at step boundary", scenario_agent_cancel_boundary),
    "agent_repair_report": ("Agent repairs suspicious curve and writes report", scenario_agent_repair_report),
    "mutation_refinement_multiround": ("Agent performs multi-round mutation refinement", scenario_mutation_refinement_multiround),
    "sentaurus_autonomous_refinement": ("Agent performs Sentaurus patch/effect/refinement lineage loop", scenario_sentaurus_autonomous_refinement),
    "natural_language_power_marathon": (
        "Natural-language goal drives Power MOSFET DEVSIM, signoff, cockpit, resume, and cancel",
        scenario_natural_language_power_marathon,
    ),
    "public_user_deck_acceptance": (
        "Public DEVSIM user deck is ingested, patched, executed, and benchmarked by deterministic guardrails",
        scenario_public_user_deck_acceptance,
    ),
    "public_user_deck_corpus_acceptance": (
        "Public real-style DEVSIM user deck corpus is ingested, patched, executed, and benchmarked",
        scenario_public_user_deck_corpus_acceptance,
    ),
    "public_curve_decision_eval": (
        "Public baseline-vs-mutation curves drive next patch decisions with overlay evidence",
        scenario_public_curve_decision_eval,
    ),
    "public_curve_decision_live_llm_eval": (
        "A real configured LLM chooses next patch directions from public curve overlays without fallback",
        scenario_public_curve_decision_live_llm_eval,
    ),
    "public_curve_decision_live_llm_agent_loop": (
        "A real configured LLM drives the autonomous agent from curve decision to guidance patch execution",
        scenario_public_curve_decision_live_llm_agent_loop,
    ),
    "public_curve_decision_live_llm_devsim_soak": (
        "A real configured LLM drives agent_soak across resume slices with real DEVSIM runner evidence",
        scenario_public_curve_decision_live_llm_devsim_soak,
    ),
    "public_user_deck_live_llm_acceptance": (
        "Public DEVSIM user deck is ingested, patched, executed, and benchmarked by a real configured LLM",
        scenario_public_user_deck_live_llm_acceptance,
    ),
    "public_user_deck_live_llm_soak": (
        "Live LLM user-deck mission is sliced through agent_soak with resume, heartbeat, cockpit, and no fallback",
        scenario_public_user_deck_live_llm_soak,
    ),
    "queue_confirmation_resume": ("Queue approval resumes a waiting agent", scenario_queue_confirmation_resume),
    "queue_interruption_recovery": ("Queue recovers interrupted long-run work", scenario_queue_interruption_recovery),
    "real_autonomous_agent": ("Real LLM/DEVSIM autonomous agent run", scenario_real_autonomous_agent),
}


DEFAULT_AUTONOMOUS_E2E_SCENARIOS = [
    "agent_confirmation_pause",
    "agent_cancel_boundary",
    "agent_repair_report",
    "mutation_refinement_multiround",
    "sentaurus_autonomous_refinement",
    "natural_language_power_marathon",
    "public_user_deck_acceptance",
    "public_user_deck_corpus_acceptance",
    "public_curve_decision_eval",
    "queue_confirmation_resume",
    "queue_interruption_recovery",
]


def selected_scenario_ids(request: LongRunValidationRequest) -> list[str]:
    if request.scenario_ids:
        return request.scenario_ids
    if request.suite not in {LongRunValidationSuite.AUTONOMOUS_E2E, LongRunValidationSuite.ALL}:
        return []
    selected = list(DEFAULT_AUTONOMOUS_E2E_SCENARIOS)
    if request.mode == LongRunValidationMode.REAL or request.real_agent_request:
        selected.append("real_autonomous_agent")
    return selected


def run_scenario(
    validation_dir: Path,
    queue_db: Path,
    request: LongRunValidationRequest,
    scenario_id: str,
) -> LongRunScenarioResult:
    if scenario_id not in SCENARIO_REGISTRY:
        now = utc_timestamp()
        result = LongRunScenarioResult(
            scenario_id=scenario_id,
            title="unknown scenario",
            status=LongRunValidationStatus.FAILED,
            started_at=now,
            completed_at=now,
            duration_seconds=0.0,
            failure_reason=f"unknown long-run validation scenario: {scenario_id}",
        )
        write_json(validation_dir / "scenarios" / scenario_id / "scenario_result.json", result.model_dump(mode="json"))
        return result
    title, runner = SCENARIO_REGISTRY[scenario_id]
    scenario_dir = validation_dir / "scenarios" / scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_timestamp()
    start = time.monotonic()
    try:
        assertions, artifacts, details = runner(scenario_dir, request, queue_db)
        status = LongRunValidationStatus.COMPLETED
        failure_reason = None
    except Exception as exc:
        assertions = []
        artifacts = []
        details = {}
        status = LongRunValidationStatus.FAILED
        failure_reason = str(exc)
    result = LongRunScenarioResult(
        scenario_id=scenario_id,
        title=title,
        status=status,
        started_at=started_at,
        completed_at=utc_timestamp(),
        duration_seconds=round(time.monotonic() - start, 6),
        assertions=assertions,
        artifacts=artifacts,
        details=details,
        failure_reason=failure_reason,
    )
    result_path = scenario_dir / "scenario_result.json"
    result.result_path = str(result_path.resolve())
    write_json(result_path, result.model_dump(mode="json"))
    return result


def run_queue_smoke(
    state: LongRunValidationState,
    request: LongRunValidationRequest,
    validation_dir: Path,
    queue_db: Path,
) -> None:
    queue_goals = request.queue_goals or default_queue_goals(validation_dir)
    for item in queue_goals:
        queued = enqueue_run(
            queue_db,
            queue_id=item.get("queue_id"),
            tool_name=item["tool_name"],
            request=item.get("request") or {},
            priority=int(item.get("priority") or 0),
            max_attempts=int(item.get("max_attempts") or 1),
            tags=item.get("tags") or ["long_run_validation"],
        )
        state.queued_items.append(queued.model_dump(mode="json"))

    daemon = run_queue_daemon(
        queue_db,
        owner=f"{state.validation_id}_daemon",
        concurrency=1,
        poll_interval_seconds=request.poll_interval_seconds,
        max_idle_loops=request.max_idle_loops,
    )
    state.daemon_result = daemon.model_dump(mode="json")

    completed_items = []
    for queued in state.queued_items:
        item = get_item(queue_db, queued["queue_id"])
        if item is None:
            raise RuntimeError(f"queue item disappeared: {queued['queue_id']}")
        completed_items.append(item.model_dump(mode="json"))
        if item.status != "completed":
            raise RuntimeError(f"queue item did not complete: {queued['queue_id']} status={item.status}")
        if item.result_state_path:
            benchmark = run_physical_benchmark(Path(item.result_state_path))
            state.benchmark_results.append(benchmark.model_dump(mode="json"))
    state.queued_items = completed_items

    failed_benchmarks = [item for item in state.benchmark_results if item.get("status") == "failed"]
    if failed_benchmarks:
        raise RuntimeError(f"{len(failed_benchmarks)} benchmark(s) failed")


def run_long_run_validation(request: LongRunValidationRequest) -> LongRunValidationState:
    validation_id = request.validation_id or default_validation_id()
    validation_dir = request.validation_root / validation_id
    validation_dir.mkdir(parents=True, exist_ok=True)
    state_path = validation_dir / "validation_state.json"
    queue_db = validation_dir / "run_queue.sqlite"
    now = utc_timestamp()
    state = LongRunValidationState(
        status=LongRunValidationStatus.COMPLETED,
        validation_id=validation_id,
        validation_dir=str(validation_dir),
        created_at=now,
        updated_at=now,
        queue_db_path=str(queue_db),
    )
    write_state(state, state_path)
    try:
        if request.suite in {LongRunValidationSuite.QUEUE_SMOKE, LongRunValidationSuite.ALL}:
            run_queue_smoke(state, request, validation_dir, queue_db)
            write_state(state, state_path)

        scenario_ids = selected_scenario_ids(request)
        for scenario_id in scenario_ids:
            scenario = run_scenario(validation_dir, queue_db, request, scenario_id)
            state.scenario_results.append(scenario.model_dump(mode="json"))
            write_state(state, state_path)

        failed_scenarios = [item for item in state.scenario_results if item.get("status") == LongRunValidationStatus.FAILED]
        if failed_scenarios:
            names = ", ".join(str(item.get("scenario_id")) for item in failed_scenarios)
            raise RuntimeError(f"long-run scenario(s) failed: {names}")

        index_db = validation_dir / "experiment_index.sqlite"
        state.index_summary = rebuild_index(validation_dir, index_db)
        state.indexed_records = list_records(index_db, limit=20)
        state.status = LongRunValidationStatus.COMPLETED
    except Exception as exc:
        state.status = LongRunValidationStatus.FAILED
        state.failure_reason = str(exc)
    write_state(state, state_path)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an unattended long-run TCAD agent validation.")
    parser.add_argument("--validation-id", default=None)
    parser.add_argument("--validation-root", type=Path, default=None)
    parser.add_argument(
        "--suite",
        choices=[item.value for item in LongRunValidationSuite],
        default=LongRunValidationSuite.QUEUE_SMOKE.value,
        help="Validation suite to run. queue_smoke preserves the original fast queue regression.",
    )
    parser.add_argument(
        "--mode",
        choices=[item.value for item in LongRunValidationMode],
        default=LongRunValidationMode.SIMULATED.value,
        help="simulated uses deterministic local runners; real runs the provided/default autonomous agent request with real tools.",
    )
    parser.add_argument("--scenario-id", action="append", default=None, help="Run one scenario. Repeat to run multiple scenarios.")
    parser.add_argument("--agent-max-steps", type=int, default=12)
    parser.add_argument("--use-llm", action="store_true", help="Allow the real autonomous scenario to call the configured LLM.")
    parser.add_argument("--no-llm-fallback", action="store_true", help="Fail the real autonomous scenario if LLM output is invalid/unavailable.")
    parser.add_argument("--real-agent-request-json", default=None, help="JSON object merged into the real autonomous agent request.")
    parser.add_argument("--queue-goals-json", default=None)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.0)
    parser.add_argument("--max-idle-loops", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = {
        "validation_id": args.validation_id,
        "suite": args.suite,
        "mode": args.mode,
        "scenario_ids": args.scenario_id or [],
        "agent_max_steps": args.agent_max_steps,
        "use_llm": args.use_llm,
        "allow_llm_fallback": not args.no_llm_fallback,
        "poll_interval_seconds": args.poll_interval_seconds,
        "max_idle_loops": args.max_idle_loops,
    }
    if args.validation_root is not None:
        data["validation_root"] = args.validation_root
    if args.queue_goals_json:
        goals = json.loads(args.queue_goals_json)
        if not isinstance(goals, list):
            raise ValueError("--queue-goals-json must decode to a list")
        data["queue_goals"] = goals
    if args.real_agent_request_json:
        real_request = json.loads(args.real_agent_request_json)
        if not isinstance(real_request, dict):
            raise ValueError("--real-agent-request-json must decode to an object")
        data["real_agent_request"] = real_request
    state = run_long_run_validation(LongRunValidationRequest.model_validate(data))
    print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
    raise SystemExit(0 if state.status == LongRunValidationStatus.COMPLETED else 2)


if __name__ == "__main__":
    main()
