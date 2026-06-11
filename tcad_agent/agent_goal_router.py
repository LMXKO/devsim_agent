from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcad_agent.device_templates import DeviceRouteResult, RouteStatus, device_templates, route_device_goal
from tcad_agent.industrial_runner_registry import (
    industrial_runner_coverage_matrix,
    preferred_runner_for_template,
    runner_descriptors_for_template,
)
from tcad_agent.task_spec import PROJECT_ROOT


class AgentGoalRouteRequest(BaseModel):
    goal_text: str
    simulator: str | None = None
    execute: bool = False
    max_steps: int = 12
    run_root: Path = PROJECT_ROOT / "runs" / "agent_goal_router"
    source_deck_path: str | None = None
    sentaurus_project_path: str | None = None
    sentaurus_profile_path: str | None = None
    reference_curve_path: str | None = None


class AgentGoalRouteResult(BaseModel):
    tool_name: str = "agent_goal_router"
    schema_version: str = "actsoft.tcad.agent_goal_router.v1"
    status: str
    goal_text: str
    normalized_goal_text: str
    simulator_strategy: str
    selected_template_id: str | None = None
    selected_runner_id: str | None = None
    primary_tool: str | None = None
    route: dict[str, Any] | None = None
    autonomous_request: dict[str, Any] = Field(default_factory=dict)
    sentaurus_request: dict[str, Any] = Field(default_factory=dict)
    evidence_plan: list[dict[str, Any]] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    industrial_runner_coverage: dict[str, Any] = Field(default_factory=dict)
    next_action: str | None = None
    output_path: str | None = None
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def text_mentions_any(text: str, tokens: list[str]) -> bool:
    lowered = text.lower()
    return any(token.lower() in lowered for token in tokens)


def is_agent_meta_goal(goal_text: str) -> bool:
    return text_mentions_any(
        goal_text,
        [
            "agent",
            "自主",
            "自动",
            "长时间",
            "长期",
            "完成任务",
            "devsim",
            "sentaurus",
            "仿真助手",
            "自主操作",
        ],
    )


def infer_device_goal(goal_text: str) -> str:
    direct = route_device_goal(goal_text)
    if direct.status == RouteStatus.MATCHED:
        return goal_text
    lowered = goal_text.lower()
    if text_mentions_any(lowered, ["功率器件", "power device", "high voltage", "高压", "耐压", "bv", "ron"]):
        return f"{goal_text} power MOSFET LDMOS BV Ron field plate drift doping leakage"
    for template in device_templates():
        if any(alias.lower() in lowered for alias in template.aliases):
            return f"{goal_text} {template.display_name}"
    return goal_text


def choose_simulator_strategy(request: AgentGoalRouteRequest, route: DeviceRouteResult | None) -> str:
    requested = (request.simulator or "").strip().lower()
    goal = request.goal_text.lower()
    if requested == "sentaurus" or ("sentaurus" in goal and "devsim" not in goal):
        return "sentaurus_external" if request.sentaurus_project_path else "sentaurus_external_workspace_required"
    if "sentaurus" in goal and "devsim" in goal:
        return "devsim_primary_sentaurus_optional"
    if route and route.runnable:
        return "devsim_primary"
    return "mission_planning"


def build_evidence_plan(route: DeviceRouteResult | None, strategy: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = [
        {
            "id": "capability_audit",
            "action": "audit_device_template_and_runner_coverage",
            "required": True,
            "reason": "先确认自然语言目标映射到真实 runner、surrogate 还是外部契约。",
        }
    ]
    if route and route.request_hint:
        steps.append(
            {
                "id": "baseline_run",
                "action": "run_primary_tcad_runner",
                "tool": route.suggested_tool,
                "request": route.request_hint,
                "required": True,
                "reason": "生成可检查的 baseline 曲线和 metrics。",
            }
        )
    if strategy.startswith("sentaurus"):
        steps.append(
            {
                "id": "external_workspace_gate",
                "action": "require_sentaurus_project_and_runtime_profile",
                "required": True,
                "reason": "商业软件、license、PDK、工艺 deck 和模型文件必须留在用户本机或远端受控 workspace。",
            }
        )
    steps.extend(
        [
            {
                "id": "physical_benchmark",
                "action": "run_physical_benchmark",
                "required": True,
                "reason": "把曲线质量、物理可信度和签核缺口显式化。",
            },
            {
                "id": "experiment_design",
                "action": "plan_next_experiment_from_curve_and_pareto",
                "required": True,
                "reason": "根据 baseline/mutation overlay、Pareto 和 signoff gaps 选择下一轮 patch。",
            },
            {
                "id": "signoff_pack",
                "action": "build_signoff_evidence_pack",
                "required": True,
                "reason": "没有 mesh convergence 或 golden/实测相关性时必须输出 conditional/blocked，而不是假装完成。",
            },
        ]
    )
    return steps


def autonomous_request_for_route(request: AgentGoalRouteRequest, route: DeviceRouteResult | None, strategy: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "goal_text": request.goal_text,
        "execute": request.execute,
        "max_steps": request.max_steps,
        "require_capability_audit": True,
        "enable_experiment_design": True,
        "max_experiment_design_rounds": 3,
        "generate_report": True,
        "generate_dashboard": True,
    }
    if route and route.suggested_tool:
        payload["initial_tool_name"] = route.suggested_tool
        payload["initial_request"] = dict(route.request_hint)
    if request.source_deck_path:
        payload["source_deck_path"] = request.source_deck_path
    if request.reference_curve_path:
        payload.setdefault("constraints", [])
        payload["initial_request"] = {**payload.get("initial_request", {}), "reference_curve_path": request.reference_curve_path}
    if strategy.startswith("sentaurus") and request.sentaurus_project_path:
        payload["sentaurus_project_path"] = request.sentaurus_project_path
        if request.sentaurus_profile_path:
            payload["sentaurus_profile_path"] = request.sentaurus_profile_path
    return payload


def sentaurus_request_for_route(request: AgentGoalRouteRequest, route: DeviceRouteResult | None) -> dict[str, Any]:
    if not request.sentaurus_project_path:
        return {}
    payload: dict[str, Any] = {
        "goal_text": request.goal_text,
        "project_path": request.sentaurus_project_path,
        "execute": request.execute,
    }
    if request.sentaurus_profile_path:
        payload["profile_path"] = request.sentaurus_profile_path
    if request.reference_curve_path:
        payload["reference_curve_path"] = request.reference_curve_path
    if route and route.template:
        payload["device_template_id"] = route.template.template_id
    return payload


def route_agent_goal(request: AgentGoalRouteRequest, output_path: Path | None = None) -> AgentGoalRouteResult:
    normalized_goal = infer_device_goal(request.goal_text) if is_agent_meta_goal(request.goal_text) else request.goal_text
    route = route_device_goal(normalized_goal)
    matched = route.status == RouteStatus.MATCHED
    strategy = choose_simulator_strategy(request, route if matched else None)
    missing_inputs: list[str] = []
    if strategy == "sentaurus_external_workspace_required":
        missing_inputs.extend(["sentaurus_project_path", "sentaurus_runtime_profile"])
    selected_runner = preferred_runner_for_template(route.template.template_id) if matched and route.template else None
    if selected_runner and selected_runner.maturity.value == "real_external" and not request.sentaurus_project_path:
        missing_inputs.append("external_tcad_project_path")

    status = "matched" if matched and not missing_inputs else "needs_input" if matched else "unmatched"
    primary_tool = "autonomous_devsim_agent" if matched else None
    if strategy == "sentaurus_external" and request.sentaurus_project_path:
        primary_tool = "sentaurus_run"
    result = AgentGoalRouteResult(
        status=status,
        goal_text=request.goal_text,
        normalized_goal_text=normalized_goal,
        simulator_strategy=strategy,
        selected_template_id=route.template.template_id if matched and route.template else None,
        selected_runner_id=selected_runner.runner_id if selected_runner else None,
        primary_tool=primary_tool,
        route=route.model_dump(mode="json") if matched else None,
        autonomous_request=autonomous_request_for_route(request, route if matched else None, strategy) if matched else {},
        sentaurus_request=sentaurus_request_for_route(request, route if matched else None),
        evidence_plan=build_evidence_plan(route if matched else None, strategy),
        missing_inputs=list(dict.fromkeys(missing_inputs)),
        industrial_runner_coverage={
            "all": industrial_runner_coverage_matrix(),
            "selected_template": [
                item.model_dump(mode="json")
                for item in (runner_descriptors_for_template(route.template.template_id) if matched and route.template else [])
            ],
        },
        next_action=(
            "run autonomous_devsim_agent with the generated request"
            if status == "matched"
            else "provide external Sentaurus workspace/profile before executing commercial TCAD"
            if missing_inputs
            else "clarify device family or simulation target"
        ),
    )
    if output_path:
        result.output_path = str(output_path.resolve())
        write_json(output_path, result.model_dump(mode="json"))
    return result

