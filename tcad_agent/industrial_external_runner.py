from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.device_templates import route_device_goal
from tcad_agent.industrial_runner_registry import runner_descriptors_for_template
from tcad_agent.sentaurus import SentaurusRunRequest, run_sentaurus
from tcad_agent.task_spec import PROJECT_ROOT


class IndustrialExternalRunnerRequest(BaseModel):
    goal_text: str
    template_id: str | None = None
    simulator: str = "sentaurus"
    project_path: Path | None = None
    profile_path: Path | None = None
    run_id: str | None = None
    run_root: Path = PROJECT_ROOT / "runs" / "industrial_external_runner"
    flow: list[str] = Field(default_factory=list)
    command_args: dict[str, list[str]] = Field(default_factory=dict)
    deck_files: list[str] = Field(default_factory=list)
    patches: list[dict[str, Any]] = Field(default_factory=list)
    reference_curve_path: Path | None = None
    timeout_seconds: float = 3600.0
    cancel_file: str | None = None
    execute: bool = True


class IndustrialExternalRunnerState(BaseModel):
    tool_name: str = "industrial_external_tcad_runner"
    schema_version: str = "actsoft.tcad.industrial_external_runner.v1"
    status: str
    run_id: str
    run_dir: str
    goal_text: str
    template_id: str | None = None
    simulator: str
    created_at: str
    updated_at: str
    request: dict[str, Any] = Field(default_factory=dict)
    runner_contract: dict[str, Any] = Field(default_factory=dict)
    delegated_state_path: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    final_summary: dict[str, Any] = Field(default_factory=dict)
    quality_report: dict[str, Any] = Field(default_factory=dict)
    next_action: str | None = None
    failure_reason: str | None = None
    state_path: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_run_id() -> str:
    return f"external_runner_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def template_id_for_request(request: IndustrialExternalRunnerRequest) -> str | None:
    if request.template_id:
        return request.template_id
    route = route_device_goal(request.goal_text)
    return route.template.template_id if route.template else None


def build_runner_contract(request: IndustrialExternalRunnerRequest, template_id: str | None) -> dict[str, Any]:
    coverage = [runner.model_dump(mode="json") for runner in runner_descriptors_for_template(template_id or "")]
    external = [runner for runner in coverage if runner.get("maturity") == "real_external"]
    return {
        "schema_version": "actsoft.tcad.industrial_external_runner_contract.v1",
        "template_id": template_id,
        "simulator": request.simulator,
        "external_runner_ids": [runner.get("runner_id") for runner in external],
        "requires_user_owned_workspace": True,
        "requires_local_or_remote_license": True,
        "repository_boundary": [
            "do_not_commit_sentaurus_software",
            "do_not_commit_license_strings",
            "do_not_commit_pdk_or_process_decks",
            "do_not_commit_commercial_model_files",
        ],
        "expected_inputs": [
            "project_path",
            "profile_path_or_inline_profile",
            "deck_files",
            "extraction_csv_or_reference_curve_when_available",
        ],
        "expected_outputs": [
            "state.json",
            "solver_logs",
            "deck_ir_artifacts",
            "curve_csv",
            "physical_benchmark",
            "lineage_or_patch_effect_analysis",
        ],
        "coverage": coverage,
    }


def waiting_state(
    request: IndustrialExternalRunnerRequest,
    *,
    run_id: str,
    run_dir: Path,
    state_path: Path,
    template_id: str | None,
    contract: dict[str, Any],
    reason: str,
) -> IndustrialExternalRunnerState:
    state = IndustrialExternalRunnerState(
        status="waiting_for_external_workspace",
        run_id=run_id,
        run_dir=str(run_dir),
        goal_text=request.goal_text,
        template_id=template_id,
        simulator=request.simulator,
        created_at=utc_timestamp(),
        updated_at=utc_timestamp(),
        request=request.model_dump(mode="json"),
        runner_contract=contract,
        artifacts={"runner_contract": str(run_dir / "runner_contract.json")},
        final_summary={
            "metrics": {
                "solver_backend": request.simulator,
                "tcad_solver_invoked": False,
                "external_workspace_required": True,
            },
            "data_provenance": {
                "commercial_assets_not_in_repository": True,
                "real_physics_requires_external_workspace": True,
            },
        },
        quality_report={
            "status": "suspicious",
            "issues": [{"code": "external_workspace_required", "severity": "warning", "message": reason}],
            "metrics": {"tcad_solver_invoked": False, "external_workspace_required": True},
            "recommended_next_action": "provide user-owned project_path and runtime profile, then rerun industrial_external_tcad_runner",
        },
        next_action="provide external TCAD workspace/profile",
        failure_reason=reason,
        state_path=str(state_path),
    )
    write_json(run_dir / "runner_contract.json", contract)
    write_json(state_path, state.model_dump(mode="json"))
    return state


def run_industrial_external_runner(request: IndustrialExternalRunnerRequest) -> IndustrialExternalRunnerState:
    run_id = request.run_id or default_run_id()
    run_dir = (request.run_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    template_id = template_id_for_request(request)
    contract = build_runner_contract(request, template_id)
    simulator = request.simulator.strip().lower()
    if simulator != "sentaurus":
        return waiting_state(
            request,
            run_id=run_id,
            run_dir=run_dir,
            state_path=state_path,
            template_id=template_id,
            contract=contract,
            reason=f"unsupported external simulator `{request.simulator}`; only Sentaurus adapter is registered",
        )
    if not request.project_path:
        return waiting_state(
            request,
            run_id=run_id,
            run_dir=run_dir,
            state_path=state_path,
            template_id=template_id,
            contract=contract,
            reason="Sentaurus project_path is required and must point to a user-owned workspace outside repository assets.",
        )

    sentaurus_state = run_sentaurus(
        SentaurusRunRequest(
            goal_text=request.goal_text,
            project_path=request.project_path,
            profile_path=request.profile_path,
            run_id=f"{run_id}_sentaurus",
            run_root=run_dir / "sentaurus",
            flow=request.flow,
            command_args=request.command_args,
            deck_files=request.deck_files,
            patches=request.patches,
            reference_curve_path=request.reference_curve_path,
            timeout_seconds=request.timeout_seconds,
            cancel_file=request.cancel_file,
            execute=request.execute,
        )
    )
    artifacts = {"runner_contract": str(run_dir / "runner_contract.json"), "delegated_sentaurus_state": sentaurus_state.state_path or ""}
    artifacts.update({str(key): str(value) for key, value in sentaurus_state.artifacts.items() if value})
    metrics = dict((sentaurus_state.final_summary.get("metrics") if isinstance(sentaurus_state.final_summary, dict) else {}) or {})
    state = IndustrialExternalRunnerState(
        status=sentaurus_state.status,
        run_id=run_id,
        run_dir=str(run_dir),
        goal_text=request.goal_text,
        template_id=template_id,
        simulator=request.simulator,
        created_at=utc_timestamp(),
        updated_at=utc_timestamp(),
        request=request.model_dump(mode="json"),
        runner_contract=contract,
        delegated_state_path=sentaurus_state.state_path,
        artifacts=artifacts,
        final_summary={
            "artifacts": artifacts,
            "metrics": metrics | {"external_runner_delegated": True, "delegated_tool": "sentaurus_run"},
            "data_provenance": {
                "commercial_assets_not_in_repository": True,
                "delegated_state_path": sentaurus_state.state_path,
            },
        },
        quality_report=sentaurus_state.quality_report,
        next_action=sentaurus_state.next_action,
        failure_reason=sentaurus_state.failure_reason,
        state_path=str(state_path),
    )
    write_json(run_dir / "runner_contract.json", contract)
    write_json(state_path, state.model_dump(mode="json"))
    return state

