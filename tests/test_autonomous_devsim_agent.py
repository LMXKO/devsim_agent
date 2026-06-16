from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import tcad_agent.autonomous_devsim_agent as agent_mod
from tcad_agent.autonomous_devsim_agent import (
    AutonomousDevsimAgentState,
    AutonomousDevsimRequest,
    DevsimAgentActionKind,
    DevsimAgentStatus,
    decide_next_action,
    deterministic_action,
    execute_action,
    infer_result_state_path,
    observe_state,
    run_autonomous_devsim_agent,
    state_needs_repair_before_signoff_planning,
)
from tcad_agent.evidence_lookup import PublicEvidenceLookupResult
from tcad_agent.llm import LLMConfig


class FakeAgentClient:
    config = LLMConfig(model="fake-autonomous-devsim-agent")

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        return self.response


class FakeToolCallClient(FakeAgentClient):
    def tool_call(self, system: str, user: str, tools: list[dict[str, object]], temperature: float = 0.1) -> dict[str, object]:
        self.calls.append({"system": system, "user": user, "tools": tools, "temperature": temperature})
        return json.loads(self.response)


class FakeSequenceClient(FakeAgentClient):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses[-1] if responses else "{}")
        self.responses = list(responses)

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        if self.responses:
            return self.responses.pop(0)
        return self.response


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class AutonomousDevsimAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_tool_state(self, name: str, quality_status: str) -> Path:
        run_dir = self.root / "runs" / name
        csv_path = run_dir / "curve.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("voltage_v,current_a\n0,0\n1,1e-6\n", encoding="utf-8")
        state_path = run_dir / "state.json"
        write_json(
            state_path,
            {
                "tool_name": "pn_junction_iv_sweep",
                "status": "completed",
                "run_id": name,
                "run_dir": str(run_dir),
                "final_summary": {
                    "artifacts": {"csv": str(csv_path)},
                    "metrics": {"leakage_current_a": 1e-6, "points": 2},
                },
                "quality_report": {
                    "status": quality_status,
                    "issues": [{"code": "current_not_monotonic", "severity": "warning"}] if quality_status != "passed" else [],
                    "metrics": {"leakage_current_a": 1e-6, "points": 2},
                },
            },
        )
        return state_path

    def test_infer_result_state_prefers_run_dir_state_over_nested_source_paths(self) -> None:
        old_state = self.write_tool_state("old_source", "passed")
        new_state = self.write_tool_state("new_output", "passed")

        inferred = infer_result_state_path(
            {
                "status": "completed",
                "run_dir": str(new_state.parent),
                "request": {"source_state_path": str(old_state)},
            }
        )

        self.assertEqual(inferred, str(new_state.resolve()))

    def test_llm_state_bound_action_uses_latest_state_not_stale_source(self) -> None:
        old_state = self.write_tool_state("old_benchmark_source", "passed")
        latest_state = self.write_tool_state("latest_benchmark_source", "passed")
        state = AutonomousDevsimAgentState(
            status=DevsimAgentStatus.RUNNING,
            agent_id="agent_source_guard",
            agent_dir=str(self.root / "agents" / "agent_source_guard"),
            goal_text="Benchmark the latest TCAD evidence.",
            created_at="2026-06-15T00:00:00Z",
            updated_at="2026-06-15T00:00:00Z",
            execute=True,
            max_steps=4,
            latest_state_path=str(latest_state),
            checkpoint={"public_evidence_gate_done": True},
        )
        request = AutonomousDevsimRequest(
            goal_text=state.goal_text,
            execute=True,
            use_llm=True,
            allow_llm_fallback=False,
            generate_report=False,
            generate_dashboard=False,
        )
        client = FakeAgentClient(
            json.dumps(
                {
                    "kind": "run_physical_benchmark",
                    "reason": "Benchmark the state I referenced.",
                    "source_state_path": str(old_state),
                }
            )
        )

        action, decision = decide_next_action(state, request, llm_client=client)

        self.assertEqual(action.kind, DevsimAgentActionKind.RUN_PHYSICAL_BENCHMARK)
        self.assertEqual(action.source_state_path, str(latest_state))
        self.assertFalse(decision["fallback_used"])
        self.assertEqual(decision["decision_source"], "mandatory_agent_policy")
        self.assertEqual(decision["guarded_action_source"], "physical_benchmark_gate")
        self.assertEqual(decision["llm_requested_action"]["source_state_path"], str(old_state))

    def test_sentaurus_patch_plan_policy_overrides_premature_benchmark(self) -> None:
        project = self.root / "sentaurus_project"
        project.mkdir()
        (project / "device.cmd").write_text("set LIFETIME_SCALE 1.0\n", encoding="utf-8")
        state_path = self.root / "sentaurus" / "sentaurus_state.json"
        write_json(
            state_path,
            {
                "tool_name": "sentaurus_run",
                "status": "completed",
                "run_id": "sentaurus_baseline",
                "project_copy_path": str(project),
                "request": {"deck_files": ["device.cmd"]},
                "quality_report": {
                    "status": "passed",
                    "metrics": {"solver_backend": "sentaurus", "tcad_solver_invoked": True, "curve_points": 3},
                },
                "final_summary": {"artifacts": {"project_copy": str(project)}, "metrics": {"solver_backend": "sentaurus"}},
            },
        )
        state = AutonomousDevsimAgentState(
            status=DevsimAgentStatus.RUNNING,
            agent_id="agent_sentaurus_policy",
            agent_dir=str(self.root / "agents" / "agent_sentaurus_policy"),
            goal_text="Reduce Sentaurus reverse leakage with a verified lifetime patch.",
            created_at="2026-06-15T00:00:00Z",
            updated_at="2026-06-15T00:00:00Z",
            execute=True,
            max_steps=4,
            latest_state_path=str(state_path),
            checkpoint={"sentaurus_initial_run_done": True, "public_evidence_gate_done": True},
        )
        request = AutonomousDevsimRequest(
            goal_text=state.goal_text,
            execute=True,
            use_llm=True,
            allow_llm_fallback=False,
            sentaurus_project_path=project,
            sentaurus_request={"deck_files": ["device.cmd"]},
            enable_experiment_design=True,
            max_experiment_design_rounds=1,
            generate_report=False,
            generate_dashboard=False,
        )
        client = FakeAgentClient(
            json.dumps(
                {
                    "kind": "run_physical_benchmark",
                    "reason": "Try to benchmark before patch planning.",
                    "source_state_path": str(state_path),
                }
            )
        )

        action, decision = decide_next_action(state, request, llm_client=client)

        self.assertEqual(action.kind, DevsimAgentActionKind.PLAN_SENTAURUS_PATCH)
        self.assertFalse(decision["fallback_used"])
        self.assertEqual(decision["decision_source"], "mandatory_agent_policy")
        self.assertEqual(decision["guarded_action_source"], "sentaurus_patch_plan")
        self.assertTrue(decision["queued_plan_enforced"])
        self.assertEqual(decision["llm_requested_action"]["kind"], "run_physical_benchmark")

    def test_sentaurus_schema_extension_policy_after_patch_planner_exhausted(self) -> None:
        project = self.root / "sentaurus_schema_project"
        project.mkdir()
        (project / "device.cmd").write_text("set SURFACE_RECOMB_VELOCITY 1e5\n", encoding="utf-8")
        state_path = self.root / "sentaurus_schema" / "sentaurus_state.json"
        write_json(
            state_path,
            {
                "tool_name": "sentaurus_run",
                "status": "completed",
                "run_id": "sentaurus_schema_baseline",
                "project_copy_path": str(project),
                "request": {"deck_files": ["device.cmd"]},
                "quality_report": {
                    "status": "passed",
                    "metrics": {"solver_backend": "sentaurus", "tcad_solver_invoked": True, "curve_points": 3},
                },
                "final_summary": {"artifacts": {"project_copy": str(project)}, "metrics": {"solver_backend": "sentaurus"}},
            },
        )
        state = AutonomousDevsimAgentState(
            status=DevsimAgentStatus.RUNNING,
            agent_id="agent_schema_policy",
            agent_dir=str(self.root / "agents" / "agent_schema_policy"),
            goal_text="Reduce Sentaurus reverse leakage by tuning surface recombination velocity.",
            created_at="2026-06-15T00:00:00Z",
            updated_at="2026-06-15T00:00:00Z",
            execute=True,
            max_steps=4,
            latest_state_path=str(state_path),
            checkpoint={
                "sentaurus_initial_run_done": True,
                "sentaurus_patch_plan_source_path": str(state_path),
                "sentaurus_patch_planner_exhausted": {
                    "status": "no_actionable_candidates",
                    "source_state_path": str(state_path),
                },
            },
        )
        request = AutonomousDevsimRequest(
            goal_text=state.goal_text,
            execute=True,
            use_llm=True,
            allow_llm_fallback=False,
            sentaurus_project_path=project,
            sentaurus_request={"deck_files": ["device.cmd"]},
            enable_experiment_design=True,
            max_experiment_design_rounds=1,
            generate_report=False,
            generate_dashboard=False,
        )
        client = FakeAgentClient(
            json.dumps(
                {
                    "kind": "run_physical_benchmark",
                    "reason": "Try to benchmark before schema extension.",
                    "source_state_path": str(state_path),
                }
            )
        )

        action, decision = decide_next_action(state, request, llm_client=client)

        self.assertEqual(action.kind, DevsimAgentActionKind.PLAN_MUTATION_SCHEMA_EXTENSION)
        self.assertFalse(decision["fallback_used"])
        self.assertEqual(decision["guarded_action_source"], "mutation_schema_extension")
        self.assertTrue(decision["queued_plan_enforced"])

    def test_execute_mutation_schema_extension_records_checkpoint_package(self) -> None:
        project = self.root / "sentaurus_schema_exec_project"
        project.mkdir()
        (project / "device.cmd").write_text("set SURFACE_RECOMB_VELOCITY 1e5\n", encoding="utf-8")
        state_path = self.root / "sentaurus_schema_exec" / "sentaurus_state.json"
        write_json(
            state_path,
            {
                "tool_name": "sentaurus_run",
                "status": "completed",
                "run_id": "sentaurus_schema_exec",
                "project_copy_path": str(project),
                "request": {"deck_files": ["device.cmd"]},
                "quality_report": {
                    "status": "passed",
                    "metrics": {"solver_backend": "sentaurus", "tcad_solver_invoked": True, "curve_points": 3},
                },
                "final_summary": {"artifacts": {"project_copy": str(project)}, "metrics": {"solver_backend": "sentaurus"}},
            },
        )
        state = AutonomousDevsimAgentState(
            status=DevsimAgentStatus.RUNNING,
            agent_id="agent_schema_exec",
            agent_dir=str(self.root / "agents" / "agent_schema_exec"),
            goal_text="Reduce Sentaurus reverse leakage by tuning surface recombination velocity.",
            created_at="2026-06-15T00:00:00Z",
            updated_at="2026-06-15T00:00:00Z",
            execute=True,
            max_steps=4,
            latest_state_path=str(state_path),
            checkpoint={},
        )
        request = AutonomousDevsimRequest(
            goal_text=state.goal_text,
            execute=True,
            use_llm=False,
            sentaurus_project_path=project,
            sentaurus_request={"deck_files": ["device.cmd"]},
            enable_experiment_design=True,
            generate_report=False,
            generate_dashboard=False,
        )

        result, result_state_path = execute_action(
            state,
            request,
            agent_mod.DevsimAgentAction(
                kind=DevsimAgentActionKind.PLAN_MUTATION_SCHEMA_EXTENSION,
                reason="Build schema extension package.",
                source_state_path=str(state_path),
            ),
            runner_registry={},
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result_state_path, str(state_path))
        self.assertEqual(state.checkpoint["mutation_schema_extension_runs"], 1)
        self.assertTrue(Path(state.checkpoint["mutation_schema_extension_path"]).exists())
        self.assertTrue(state.checkpoint["pending_mutation_schema_extension_candidate"]["ready_for_review"])

        state.checkpoint["sentaurus_initial_run_done"] = True
        next_action = deterministic_action(state, request)
        self.assertEqual(next_action.kind, DevsimAgentActionKind.PLAN_MUTATION_SCHEMA_PROMOTION)

        promotion_result, promotion_state_path = execute_action(
            state,
            request,
            next_action,
            runner_registry={},
        )

        self.assertEqual(promotion_result["status"], "ready_for_confirmation")
        self.assertEqual(promotion_state_path, str(state_path))
        self.assertEqual(state.checkpoint["mutation_schema_promotion_runs"], 1)
        self.assertTrue(Path(state.checkpoint["mutation_schema_promotion_path"]).exists())
        self.assertTrue(Path(state.checkpoint["latest_mutation_schema_promotion"]["artifacts"]["mutation_vocabulary_patch"]).exists())

        review_action = deterministic_action(state, request)
        self.assertEqual(review_action.kind, DevsimAgentActionKind.ASK_USER)
        self.assertIn("Mutation schema promotion is ready", review_action.request["question"])

    def test_plan_only_records_next_tool_action(self) -> None:
        request = AutonomousDevsimRequest(
            goal_text="Run PN IV autonomously",
            agent_id="agent_plan",
            agent_root=self.root / "agents",
            execute=False,
            use_llm=False,
            initial_tool_name="pn_junction_iv_sweep",
            initial_request={"run_id": "first"},
        )

        state = run_autonomous_devsim_agent(request, runner_registry={})

        self.assertEqual(state.status, DevsimAgentStatus.PLANNED)
        self.assertEqual(state.steps[0].kind, DevsimAgentActionKind.RUN_TOOL)
        self.assertEqual(state.checkpoint["planned_action"]["tool_name"], "pn_junction_iv_sweep")
        self.assertTrue(state.checkpoint["public_evidence_gate_done"])
        self.assertIn("public_evidence_dossier", state.checkpoint)
        self.assertTrue((self.root / "agents" / "agent_plan" / "autonomous_devsim_agent_state.json").exists())

    def test_execute_runs_tool_repairs_benchmarks_and_writes_conclusion(self) -> None:
        suspicious_state = self.write_tool_state("first_bad", "suspicious")
        repaired_state = self.write_tool_state("first_repaired", "passed")
        calls: list[str] = []

        def fake_tool(request: dict[str, object]) -> dict[str, object]:
            calls.append("tool")
            return {"status": "completed", "state_path": str(suspicious_state)}

        def fake_repair_runner(source: Path, **kwargs: object) -> dict[str, object]:
            calls.append("repair")
            self.assertEqual(source, suspicious_state)
            self.assertTrue(kwargs["use_agent_policy"])
            return {
                "status": "completed",
                "final_state_path": str(repaired_state),
                "current_state_path": str(repaired_state),
                "final_quality_status": "passed",
            }

        def fake_benchmark(request: dict[str, object]) -> dict[str, object]:
            calls.append("benchmark")
            self.assertEqual(request["source"], str(repaired_state))
            return {"status": "completed", "benchmark_path": str(self.root / "benchmark.json")}

        def fake_report(request: dict[str, object]) -> dict[str, object]:
            calls.append("report")
            raise ValueError("single run state is not a sweep report")

        def fake_conclusion(request: dict[str, object]) -> dict[str, object]:
            calls.append("conclusion")
            return {"status": "completed", "conclusion_path": str(self.root / "conclusion.md")}

        def fake_dashboard(request: dict[str, object]) -> dict[str, object]:
            calls.append("dashboard")
            self.assertTrue(str(request["source"]).endswith("autonomous_devsim_agent_state.json"))
            return {"status": "completed", "dashboard_path": str(self.root / "dashboard.html")}

        request = AutonomousDevsimRequest(
            goal_text="Run PN IV, repair if suspicious, then conclude",
            agent_id="agent_exec",
            agent_root=self.root / "agents",
            execute=True,
            use_llm=False,
            max_steps=6,
            initial_tool_name="pn_junction_iv_sweep",
            initial_request={"run_id": "first"},
        )

        state = run_autonomous_devsim_agent(
            request,
            runner_registry={
                "pn_junction_iv_sweep": fake_tool,
                "physical_benchmark": fake_benchmark,
                "experiment_report": fake_report,
                "experiment_conclusion": fake_conclusion,
                "experiment_dashboard": fake_dashboard,
            },
            repair_runner=fake_repair_runner,
        )

        self.assertEqual(state.status, DevsimAgentStatus.COMPLETED)
        self.assertEqual(
            [step.kind for step in state.steps],
            [
                DevsimAgentActionKind.RUN_TOOL,
                DevsimAgentActionKind.RUN_REPAIR_EXECUTOR,
                DevsimAgentActionKind.RUN_PHYSICAL_BENCHMARK,
                DevsimAgentActionKind.GENERATE_REPORT,
                DevsimAgentActionKind.GENERATE_DASHBOARD,
                DevsimAgentActionKind.STOP_SUCCESS,
            ],
        )
        self.assertEqual(calls, ["tool", "repair", "benchmark", "report", "conclusion", "dashboard"])
        self.assertEqual(state.final_state_path, str(repaired_state))
        self.assertEqual(state.final_report_path, str(self.root / "conclusion.md"))
        self.assertEqual(state.final_dashboard_path, str(self.root / "dashboard.html"))

    def test_llm_agent_can_select_first_tool_call(self) -> None:
        client = FakeAgentClient(
            json.dumps(
                {
                    "action": {
                        "kind": "run_tool",
                        "reason": "先跑 MOS C-V 获取基线。",
                        "tool_name": "mos_capacitor_cv_sweep",
                        "request": {"run_id": "mos_cv_first"},
                    },
                    "observation_summary": "还没有 state。",
                    "hypothesis_zh": "先建立 baseline。",
                    "hypothesis_tree_update": {
                        "hypothesis_zh": "MOS C-V baseline 可以暴露氧化层/固定电荷的一阶异常。",
                        "expected_observation": "得到 C-V 曲线和 Cox 附近 sanity check。",
                        "stop_condition": "quality passed 且 benchmark 没有硬错误。",
                        "next_alternatives": ["如果 C 明显超 Cox，优先检查 oxide thickness。"],
                    },
                    "evidence_used": ["goal_text", "toolbelt"],
                }
            )
        )
        request = AutonomousDevsimRequest(
            goal_text="自主完成 MOS C-V",
            agent_id="agent_llm",
            agent_root=self.root / "agents",
            execute=False,
            use_llm=True,
        )

        state = run_autonomous_devsim_agent(request, runner_registry={}, llm_client=client)

        self.assertEqual(state.status, DevsimAgentStatus.PLANNED)
        self.assertEqual(state.steps[0].kind, DevsimAgentActionKind.RUN_TOOL)
        self.assertEqual(state.steps[0].action["tool_name"], "mos_capacitor_cv_sweep")
        self.assertFalse(state.checkpoint["last_agent_decision"]["fallback_used"])
        tree = state.checkpoint["agent_hypothesis_tree"]
        self.assertEqual(tree["nodes"][0]["hypothesis_zh"], "MOS C-V baseline 可以暴露氧化层/固定电荷的一阶异常。")
        self.assertEqual(tree["nodes"][0]["verdict"], "planned")
        self.assertEqual(len(client.calls), 1)

    def test_agent_cannot_call_itself_as_nested_tool(self) -> None:
        client = FakeAgentClient(
            json.dumps(
                {
                    "action": {
                        "kind": "run_tool",
                        "reason": "不要嵌套启动自己。",
                        "tool_name": "autonomous_devsim_agent",
                        "request": {"goal_text": "nested"},
                    }
                }
            )
        )
        request = AutonomousDevsimRequest(
            goal_text="自主完成 PN IV",
            agent_id="agent_no_self_call",
            agent_root=self.root / "agents",
            execute=True,
            use_llm=True,
            max_steps=1,
        )

        state = run_autonomous_devsim_agent(
            request,
            runner_registry={"autonomous_devsim_agent": lambda request: {"status": "completed"}},
            llm_client=client,
        )

        self.assertEqual(state.status, DevsimAgentStatus.FAILED)
        self.assertIn("cannot call itself", state.failure_reason)

    def test_cancel_file_stops_before_next_step_and_writes_heartbeat(self) -> None:
        cancel_file = self.root / "agents" / "agent_cancel" / "cancel.requested"
        cancel_file.parent.mkdir(parents=True, exist_ok=True)
        cancel_file.write_text("cancel", encoding="utf-8")
        heartbeat = self.root / "agents" / "agent_cancel" / "heartbeat.json"
        request = AutonomousDevsimRequest(
            goal_text="Cancel before running",
            agent_id="agent_cancel",
            agent_root=self.root / "agents",
            execute=True,
            use_llm=False,
            cancel_file=cancel_file,
            heartbeat_path=heartbeat,
            initial_tool_name="pn_junction_iv_sweep",
        )

        state = run_autonomous_devsim_agent(
            request,
            runner_registry={"pn_junction_iv_sweep": lambda request: {"status": "completed"}},
        )

        self.assertEqual(state.status, DevsimAgentStatus.CANCELLED)
        self.assertEqual(state.steps, [])
        self.assertTrue(heartbeat.exists())
        self.assertEqual(json.loads(heartbeat.read_text(encoding="utf-8"))["status"], "cancelled")

    def test_observe_state_reads_curve_log_and_deck_diff(self) -> None:
        run_dir = self.root / "runs" / "observe"
        csv_path = run_dir / "curve.csv"
        log_path = run_dir / "run.log"
        diff_path = run_dir / "deck.diff"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("voltage_v,current_a\n-1,1e-8\n0,1e-10\n1,1e-6\n", encoding="utf-8")
        log_path.write_text("start\nsolve completed\n", encoding="utf-8")
        diff_path.write_text("- oxide = 50\n+ oxide = 45\n", encoding="utf-8")
        state_path = run_dir / "state.json"
        write_json(
            state_path,
            {
                "tool_name": "pn_junction_iv_sweep",
                "status": "completed",
                "final_summary": {"artifacts": {"csv": str(csv_path), "log": str(log_path), "semantic_deck_diff": str(diff_path)}},
                "quality_report": {"status": "passed", "metrics": {"breakdown_current_threshold_a": 1e-6}},
            },
        )

        observed = observe_state(str(state_path))

        self.assertTrue(observed["artifact_observations"]["curve_shapes"])
        self.assertIn("solve completed", observed["artifact_observations"]["log_tails"]["log"]["tail"])
        self.assertIn("oxide", observed["artifact_observations"]["deck_diffs"]["semantic_deck_diff"]["tail"])

    def test_native_tool_call_selects_registered_runner(self) -> None:
        client = FakeToolCallClient(
            json.dumps(
                {
                    "tool_call": {
                        "name": "run_tool__mos_capacitor_cv_sweep",
                        "arguments": {"run_id": "native_tool"},
                    },
                    "observation_summary": "native tool call",
                }
            )
        )
        request = AutonomousDevsimRequest(
            goal_text="Run MOS C-V with native tool call",
            agent_id="agent_native_tool",
            agent_root=self.root / "agents",
            execute=False,
            use_llm=True,
        )

        state = run_autonomous_devsim_agent(
            request,
            runner_registry={"mos_capacitor_cv_sweep": lambda request: {"status": "completed"}},
            llm_client=client,
        )

        self.assertEqual(state.steps[0].kind, DevsimAgentActionKind.RUN_TOOL)
        self.assertEqual(state.steps[0].action["tool_name"], "mos_capacitor_cv_sweep")
        self.assertIn("tools", client.calls[0])

    def test_compatible_content_json_tool_result_selects_action(self) -> None:
        client = FakeToolCallClient(
            json.dumps(
                {
                    "content": """```json
{
  "action": {
    "kind": "run_tool",
    "tool_name": "mos_capacitor_cv_sweep",
    "request": {"run_id": "compatible_content"},
    "reason": "兼容模式把 JSON 放进 content 代码块。"
  },
  "observation_summary": "content JSON",
  "hypothesis_zh": "仍应作为模型决策执行。"
}
```"""
                }
            )
        )
        request = AutonomousDevsimRequest(
            goal_text="Run MOS C-V with compatible content JSON",
            agent_id="agent_compatible_content",
            agent_root=self.root / "agents",
            execute=False,
            use_llm=True,
        )

        state = run_autonomous_devsim_agent(
            request,
            runner_registry={"mos_capacitor_cv_sweep": lambda request: {"status": "completed"}},
            llm_client=client,
        )

        self.assertEqual(state.steps[0].kind, DevsimAgentActionKind.RUN_TOOL)
        self.assertEqual(state.steps[0].action["tool_name"], "mos_capacitor_cv_sweep")
        self.assertFalse(state.checkpoint["last_agent_decision"]["fallback_used"])
        self.assertEqual(state.checkpoint["last_agent_decision"]["status"], "completed")

    def test_agent_context_exposes_industrial_runner_registry_and_2d_power_alias(self) -> None:
        client = FakeToolCallClient(
            json.dumps(
                {
                    "tool_call": {
                        "name": "run_tool__power_mosfet_bv_ron_2d_runner",
                        "arguments": {"run_id": "agent_power_2d"},
                    },
                    "observation_summary": "选择 Power MOSFET 2D 场板 runner。",
                    "hypothesis_zh": "2D field-plate runner 比 1D baseline 更适合观察场峰位置。",
                }
            )
        )
        request = AutonomousDevsimRequest(
            goal_text="自主优化 Power MOSFET BV Ron field plate",
            agent_id="agent_power_2d_toolbelt",
            agent_root=self.root / "agents",
            execute=False,
            use_llm=True,
        )

        state = run_autonomous_devsim_agent(
            request,
            runner_registry={"power_mosfet_bv_ron_2d_runner": lambda request: {"status": "completed"}},
            llm_client=client,
        )

        tool_names = {tool["function"]["name"] for tool in client.calls[0]["tools"]}
        self.assertIn("run_tool__power_mosfet_bv_ron_2d_runner", tool_names)
        prompt = json.loads(client.calls[0]["user"])
        registry = prompt["context"]["industrial_runner_registry"]
        power_runners = registry["by_template"]["power_mosfet_bv_ron"]
        self.assertIn("power_mosfet_bv_ron_devsim_2d_field_plate", {item["runner_id"] for item in power_runners})
        self.assertEqual(state.steps[0].action["tool_name"], "power_mosfet_bv_ron_2d_runner")
        self.assertFalse(state.checkpoint["agent_decision_ledger"][0]["fallback_used"])
        self.assertEqual(state.checkpoint["agent_control"]["mode"], "agent_first")

    def test_deck_ingest_patch_then_initial_tool(self) -> None:
        source_deck = self.root / "user_deck.py"
        source_deck.write_text("oxide_thickness_nm = 50\nsolve_voltage = 1.0\n", encoding="utf-8")
        passed_state = self.write_tool_state("deck_tool_passed", "passed")
        tool_requests: list[dict[str, object]] = []

        def fake_tool(request: dict[str, object]) -> dict[str, object]:
            tool_requests.append(request)
            return {"status": "completed", "state_path": str(passed_state)}

        request = AutonomousDevsimRequest(
            goal_text="Patch deck then run",
            agent_id="agent_deck",
            agent_root=self.root / "agents",
            execute=True,
            use_llm=False,
            max_steps=5,
            source_deck_path=str(source_deck),
            deck_patches=[{"deck_path": "geometry.oxide_thickness_nm", "request_path": "oxide_thickness_nm", "value": 45}],
            allow_user_confirmation_actions=True,
            initial_tool_name="pn_junction_iv_sweep",
            generate_report=False,
            generate_dashboard=False,
        )

        state = run_autonomous_devsim_agent(
            request,
            runner_registry={
                "pn_junction_iv_sweep": fake_tool,
                "physical_benchmark": lambda request: {"status": "completed", "benchmark_path": str(self.root / "benchmark.json")},
            },
        )

        self.assertEqual(state.status, DevsimAgentStatus.COMPLETED)
        self.assertEqual(
            [step.kind for step in state.steps],
            [
                DevsimAgentActionKind.INGEST_DECK,
                DevsimAgentActionKind.APPLY_DECK_PATCH,
                DevsimAgentActionKind.RUN_TOOL,
                DevsimAgentActionKind.RUN_PHYSICAL_BENCHMARK,
                DevsimAgentActionKind.STOP_SUCCESS,
            ],
        )
        self.assertTrue(Path(state.checkpoint["patched_source_deck"]).exists())
        self.assertEqual(tool_requests[0]["source_deck_path"], state.checkpoint["patched_source_deck"])

    def test_user_deck_executes_directly_when_no_initial_tool(self) -> None:
        source_deck = self.root / "direct_user_deck.py"
        source_deck.write_text("print('{\"deck\":\"ran\"}')\n", encoding="utf-8")

        request = AutonomousDevsimRequest(
            goal_text="Run user deck directly",
            agent_id="agent_user_deck",
            agent_root=self.root / "agents",
            execute=True,
            use_llm=False,
            max_steps=2,
            source_deck_path=str(source_deck),
            generate_report=False,
            generate_dashboard=False,
        )

        state = run_autonomous_devsim_agent(request)

        self.assertEqual(state.status, DevsimAgentStatus.FAILED)
        self.assertIn(DevsimAgentActionKind.RUN_USER_DECK, [step.kind for step in state.steps])
        deck_step = next(step for step in state.steps if step.kind == DevsimAgentActionKind.RUN_USER_DECK)
        self.assertEqual(deck_step.result["status"], "completed")
        self.assertTrue(Path(deck_step.result["state_path"]).exists())
        self.assertEqual(deck_step.result["reported_stdout_json"], {"deck": "ran"})

    def test_unverified_deck_patch_pauses_before_execution(self) -> None:
        source_deck = self.root / "unmatched_user_deck.py"
        source_deck.write_text("solve(type='dc')\n", encoding="utf-8")
        tool_calls: list[dict[str, object]] = []

        request = AutonomousDevsimRequest(
            goal_text="Patch unmatched deck symbol then run",
            agent_id="agent_unverified_patch",
            agent_root=self.root / "agents",
            execute=True,
            use_llm=False,
            max_steps=3,
            source_deck_path=str(source_deck),
            deck_patches=[{"deck_path": "geometry.field_plate_length_um", "request_path": "power_mos_field_plate_length_um", "value": 2.0}],
            allow_user_confirmation_actions=True,
            initial_tool_name="extended_device_sweep",
            generate_report=False,
            generate_dashboard=False,
        )

        state = run_autonomous_devsim_agent(
            request,
            runner_registry={"extended_device_sweep": lambda request: tool_calls.append(request) or {"status": "completed"}},
        )

        self.assertEqual(state.status, DevsimAgentStatus.WAITING_FOR_USER)
        self.assertEqual(tool_calls, [])
        self.assertFalse(state.checkpoint["deck_patch_verified"])
        self.assertTrue(state.checkpoint["deck_patch_unverified"])
        self.assertEqual(state.steps[-1].kind, DevsimAgentActionKind.ASK_USER)

    def test_live_evidence_gap_pauses_before_tool_execution(self) -> None:
        original_lookup = agent_mod.run_public_evidence_lookup

        def fake_lookup(request: object) -> PublicEvidenceLookupResult:
            return PublicEvidenceLookupResult(
                status="completed_with_lookup_gaps",
                goal_text="Sentaurus operation with missing public evidence",
                simulator="sentaurus",
                live=True,
                source_ids=["sentaurus_quasistationary_training"],
                failed_source_ids=["sentaurus_quasistationary_training"],
                evidence_gate={
                    "gate": "live_public_evidence_lookup",
                    "mode": "live_fetch",
                    "passed": False,
                    "source_count": 1,
                    "verified_count": 0,
                    "failed_count": 1,
                },
            )

        tool_calls: list[dict[str, object]] = []
        agent_mod.run_public_evidence_lookup = fake_lookup
        try:
            request = AutonomousDevsimRequest(
                goal_text="Sentaurus operation with missing public evidence",
                agent_id="agent_live_gap",
                agent_root=self.root / "agents",
                execute=True,
                use_llm=True,
                enable_live_evidence_lookup=True,
                initial_tool_name="pn_junction_iv_sweep",
                generate_report=False,
                generate_dashboard=False,
            )

            state = agent_mod.run_autonomous_devsim_agent(
                request,
                runner_registry={"pn_junction_iv_sweep": lambda request: tool_calls.append(request) or {"status": "completed"}},
                llm_client=FakeAgentClient('{"action": {"kind": "run_tool", "tool_name": "pn_junction_iv_sweep"}}'),
            )
        finally:
            agent_mod.run_public_evidence_lookup = original_lookup

        self.assertEqual(state.status, DevsimAgentStatus.WAITING_FOR_USER)
        self.assertEqual(tool_calls, [])
        self.assertEqual(state.steps[-1].kind, DevsimAgentActionKind.ASK_USER)
        self.assertEqual(state.steps[-1].action["request"]["gate"], "public_evidence_lookup")
        self.assertFalse(state.checkpoint["public_evidence_lookup_gate_passed"])

    def test_capability_audit_records_coverage_work_package(self) -> None:
        request = AutonomousDevsimRequest(
            goal_text="GaN HEMT current collapse transient signoff",
            agent_id="agent_capability",
            agent_root=self.root / "agents",
            execute=True,
            use_llm=False,
            max_steps=1,
            require_capability_audit=True,
            generate_report=False,
            generate_dashboard=False,
        )

        state = run_autonomous_devsim_agent(request, runner_registry={})

        self.assertEqual(state.status, DevsimAgentStatus.FAILED)
        self.assertTrue(Path(state.checkpoint["capability_audit_path"]).exists())
        self.assertIn("capability_audit", state.checkpoint)
        self.assertIn("coverage_work_package", state.checkpoint)
        self.assertEqual(state.checkpoint["coverage_work_package"]["template_id"], "gan_hemt_id_bv")
        self.assertTrue(state.checkpoint["coverage_work_package"]["industrial_runner_coverage"])
        maturities = {runner["maturity"] for runner in state.checkpoint["coverage_work_package"]["industrial_runner_coverage"]}
        self.assertIn("real_external", maturities)
        self.assertIn("physics_surrogate", maturities)
        self.assertIn("runner_promotion_plan", state.checkpoint)
        self.assertTrue(Path(state.checkpoint["runner_promotion_plan_path"]).exists())
        self.assertEqual(state.checkpoint["runner_promotion_plan"]["template_id"], "gan_hemt_id_bv")
        self.assertIn("runner_contract", [stage["stage_id"] for stage in state.checkpoint["runner_promotion_plan"]["stages"]])

    def test_objective_evaluation_runs_before_success_when_requested(self) -> None:
        passed_state = self.write_tool_state("objective_passed", "passed")
        calls: list[str] = []

        request = AutonomousDevsimRequest(
            goal_text="Run and evaluate leakage objective",
            agent_id="agent_objective",
            agent_root=self.root / "agents",
            execute=True,
            use_llm=False,
            max_steps=4,
            initial_tool_name="pn_junction_iv_sweep",
            objectives=[{"metric_path": "leakage_current_a", "direction": "minimize"}],
            generate_report=False,
            generate_dashboard=False,
        )

        state = run_autonomous_devsim_agent(
            request,
            runner_registry={
                "pn_junction_iv_sweep": lambda request: {"status": "completed", "state_path": str(passed_state)},
                "physical_benchmark": lambda request: calls.append("benchmark") or {"status": "completed", "benchmark_path": str(self.root / "benchmark.json")},
                "engineering_objectives": lambda request: calls.append("objectives") or {
                    "status": "completed",
                    "output_path": str(self.root / "objectives.json"),
                    "best_candidate": {"candidate_id": "objective_passed"},
                    "pareto_front": [{"candidate_id": "objective_passed"}],
                },
            },
        )

        self.assertEqual(state.status, DevsimAgentStatus.COMPLETED)
        self.assertEqual(calls, ["benchmark", "objectives"])
        self.assertEqual(state.checkpoint["engineering_objectives_path"], str(self.root / "objectives.json"))
        self.assertEqual(state.checkpoint["pareto_front"][0]["candidate_id"], "objective_passed")

    def test_signoff_gap_suspicious_state_continues_to_planning(self) -> None:
        observation = {
            "quality_status": "suspicious",
            "issue_codes": ["power_mos_2d_layout_signoff_gaps"],
            "metrics": {"signoff_gaps": ["mesh_convergence", "golden_or_measured_correlation"]},
        }

        self.assertFalse(state_needs_repair_before_signoff_planning(observation))
        self.assertTrue(
            state_needs_repair_before_signoff_planning(
                {"quality_status": "suspicious", "issue_codes": ["current_not_monotonic"], "metrics": {}}
            )
        )

    def test_agent_plans_and_executes_curve_guided_mutation_refinement(self) -> None:
        baseline_dir = self.root / "runs" / "baseline_power"
        baseline_csv = baseline_dir / "curve.csv"
        baseline_csv.parent.mkdir(parents=True, exist_ok=True)
        baseline_csv.write_text("drain_voltage_v,off_current_a,electric_field_v_per_cm\n0,1e-10,0\n-10,2e-8,2e5\n", encoding="utf-8")
        baseline_state = baseline_dir / "state.json"
        write_json(
            baseline_state,
            {
                "tool_name": "extended_device_sweep",
                "status": "completed",
                "run_id": "baseline_power",
                "request": {"power_mos_field_plate_length_um": 1.5},
                "final_summary": {
                    "artifacts": {"csv": str(baseline_csv)},
                    "metrics": {
                        "leakage_current_a": 2e-8,
                        "max_electric_field_v_per_cm": 2e5,
                        "specific_on_resistance_ohm_cm2": 0.05,
                    },
                },
                "quality_report": {"status": "passed", "metrics": {"leakage_current_a": 2e-8}},
            },
        )
        mutation_dir = self.root / "runs" / "mutation_power"
        mutation_csv = mutation_dir / "curve.csv"
        mutation_csv.parent.mkdir(parents=True, exist_ok=True)
        mutation_csv.write_text("drain_voltage_v,off_current_a,electric_field_v_per_cm\n0,1e-10,0\n-10,1e-8,1.5e5\n", encoding="utf-8")
        mutation_state = mutation_dir / "state.json"
        mutation = {
            "name": "field_plate_length_refine",
            "target": "field_plate",
            "request_path": "power_mos_field_plate_length_um",
            "deck_path": "geometry.field_plate_length_um",
            "values": [1.5, 2.0, 2.25],
            "requires_user_confirmation": True,
        }
        write_json(
            mutation_state,
            {
                "tool_name": "extended_device_sweep",
                "status": "completed",
                "run_id": "mutation_power",
                "request": {
                    "device_type": "power_mosfet_bv_ron",
                    "fidelity": "physics_1d",
                    "power_mos_field_plate_length_um": 2.0,
                    "tcad_deck_mutations": [mutation],
                },
                "tcad_deck_mutations": [mutation],
                "final_summary": {
                    "artifacts": {"csv": str(mutation_csv)},
                    "metrics": {
                        "leakage_current_a": 1e-8,
                        "max_electric_field_v_per_cm": 1.5e5,
                        "specific_on_resistance_ohm_cm2": 0.05,
                    },
                },
                "quality_report": {"status": "passed", "metrics": {"leakage_current_a": 1e-8}},
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
        refined_dir = self.root / "runs" / "refined_power"
        refined_state = refined_dir / "state.json"
        refined_csv = refined_dir / "curve.csv"
        tool_requests: list[dict[str, object]] = []

        def fake_extended_device(request: dict[str, object]) -> dict[str, object]:
            tool_requests.append(request)
            refined_csv.parent.mkdir(parents=True, exist_ok=True)
            refined_csv.write_text("drain_voltage_v,off_current_a,electric_field_v_per_cm\n0,1e-10,0\n-10,8e-9,1.4e5\n", encoding="utf-8")
            write_json(
                refined_state,
                {
                    "tool_name": "extended_device_sweep",
                    "status": "completed",
                    "run_id": "refined_power",
                    "request": request,
                    "final_summary": {
                        "artifacts": {"csv": str(refined_csv)},
                        "metrics": {
                            "leakage_current_a": 8e-9,
                            "max_electric_field_v_per_cm": 1.4e5,
                            "specific_on_resistance_ohm_cm2": 0.05,
                        },
                    },
                    "quality_report": {"status": "passed", "metrics": {"leakage_current_a": 8e-9}},
                },
            )
            return {"status": "completed", "state_path": str(refined_state)}

        request = AutonomousDevsimRequest(
            goal_text="Refine power MOSFET mutation from curve evidence",
            agent_id="agent_refine",
            agent_root=self.root / "agents",
            execute=True,
            use_llm=False,
            max_steps=4,
            source_state_path=str(mutation_state),
            max_mutation_refinements=1,
            allow_user_confirmation_actions=True,
            generate_report=False,
            generate_dashboard=False,
        )

        state = run_autonomous_devsim_agent(
            request,
            runner_registry={
                "extended_device_sweep": fake_extended_device,
                "physical_benchmark": lambda request: {"status": "completed", "benchmark_path": str(self.root / "benchmark.json")},
            },
        )

        self.assertEqual(state.status, DevsimAgentStatus.COMPLETED)
        self.assertIn(DevsimAgentActionKind.PLAN_MUTATION_REFINEMENT, [step.kind for step in state.steps])
        self.assertEqual(tool_requests[0]["power_mos_field_plate_length_um"], 2.25)
        self.assertIn("mutation_refinement_id", tool_requests[0])
        self.assertEqual(state.checkpoint["mutation_refinement_runs"], 1)
        self.assertTrue(Path(state.checkpoint["mutation_refinement_plan_path"]).exists())
        refined = json.loads(refined_state.read_text(encoding="utf-8"))
        self.assertIn("mutation_effect_analysis", refined)
        self.assertIn("baseline_mutation_overlay", refined["final_summary"]["artifacts"])

    def test_agent_executes_curve_guidance_patch_without_prior_mutation_effect(self) -> None:
        source_dir = self.root / "runs" / "guidance_source"
        source_csv = source_dir / "curve.csv"
        source_csv.parent.mkdir(parents=True, exist_ok=True)
        source_csv.write_text(
            "drain_voltage_v,off_current_a,electric_field_v_per_cm\n0,1e-10,1e4\n-10,1e-8,2e5\n-20,1e-6,5e5\n",
            encoding="utf-8",
        )
        source_state = source_dir / "state.json"
        write_json(
            source_state,
            {
                "tool_name": "extended_device_sweep",
                "status": "completed",
                "run_id": "guidance_source",
                "request": {
                    "device_type": "power_mosfet_bv_ron",
                    "fidelity": "devsim_2d_field_plate",
                    "power_mos_drift_region_doping_cm3": 1.0e16,
                },
                "final_summary": {
                    "artifacts": {"csv": str(source_csv)},
                    "metrics": {
                        "leakage_current_a": 1e-8,
                        "breakdown_voltage_v": -20,
                        "specific_on_resistance_ohm_cm2": 0.05,
                    },
                },
                "quality_report": {"status": "passed", "metrics": {"leakage_current_a": 1e-8}},
            },
        )
        refined_dir = self.root / "runs" / "guidance_refined"
        refined_csv = refined_dir / "curve.csv"
        refined_state = refined_dir / "state.json"
        tool_requests: list[dict[str, object]] = []

        def fake_extended_device(request: dict[str, object]) -> dict[str, object]:
            tool_requests.append(request)
            refined_csv.parent.mkdir(parents=True, exist_ok=True)
            refined_csv.write_text(
                "drain_voltage_v,off_current_a,electric_field_v_per_cm\n0,1e-10,1e4\n-10,8e-9,1.8e5\n-20,8e-7,4.2e5\n",
                encoding="utf-8",
            )
            write_json(
                refined_state,
                {
                    "tool_name": "extended_device_sweep",
                    "status": "completed",
                    "run_id": "guidance_refined",
                    "request": request,
                    "final_summary": {
                        "artifacts": {"csv": str(refined_csv)},
                        "metrics": {
                            "leakage_current_a": 8e-9,
                            "breakdown_voltage_v": -22,
                            "specific_on_resistance_ohm_cm2": 0.052,
                        },
                    },
                    "quality_report": {"status": "passed", "metrics": {"leakage_current_a": 8e-9}},
                },
            )
            return {"status": "completed", "state_path": str(refined_state)}

        request = AutonomousDevsimRequest(
            goal_text="Use curve guidance to continue Power MOSFET optimization",
            agent_id="agent_guidance_patch",
            agent_root=self.root / "agents",
            execute=True,
            use_llm=False,
            max_steps=5,
            source_state_path=str(source_state),
            curve_guidance={
                "recommended_action": "improve_tradeoff",
                "recommended_target": "drift_doping",
                "recommended_direction": "decrease",
                "reason": "Ron/BV tradeoff needs a drift doping probe.",
                "next_patch_hint": {"target": "drift_doping", "direction": "decrease"},
            },
            max_mutation_refinements=1,
            generate_report=False,
            generate_dashboard=False,
        )

        state = run_autonomous_devsim_agent(
            request,
            runner_registry={
                "extended_device_sweep": fake_extended_device,
                "physical_benchmark": lambda request: {"status": "completed", "benchmark_path": str(self.root / "benchmark.json")},
            },
        )

        self.assertEqual(state.status, DevsimAgentStatus.COMPLETED)
        self.assertIn(DevsimAgentActionKind.PLAN_GUIDANCE_PATCH, [step.kind for step in state.steps])
        self.assertEqual(len(tool_requests), 1)
        self.assertIn("guidance_patch_id", tool_requests[0])
        self.assertLess(tool_requests[0]["power_mos_drift_region_doping_cm3"], 1.0e16)
        self.assertEqual(state.checkpoint["guidance_patch_runs"], 1)
        refined = json.loads(refined_state.read_text(encoding="utf-8"))
        self.assertIn("mutation_effect_analysis", refined)
        self.assertIn("baseline_mutation_overlay", refined["final_summary"]["artifacts"])

    def test_agent_uses_llm_curve_decision_before_guidance_patch(self) -> None:
        source_dir = self.root / "runs" / "curve_decision_source"
        source_csv = source_dir / "curve.csv"
        source_csv.parent.mkdir(parents=True, exist_ok=True)
        source_csv.write_text(
            "drain_voltage_v,off_current_a,electric_field_v_per_cm\n0,1e-10,1e4\n-10,1e-8,2e5\n-20,1e-6,5e5\n",
            encoding="utf-8",
        )
        source_state = source_dir / "state.json"
        write_json(
            source_state,
            {
                "tool_name": "extended_device_sweep",
                "status": "completed",
                "run_id": "curve_decision_source",
                "request": {
                    "device_type": "power_mosfet_bv_ron",
                    "fidelity": "devsim_2d_field_plate",
                    "power_mos_drift_region_doping_cm3": 1.0e16,
                },
                "final_summary": {
                    "artifacts": {"csv": str(source_csv)},
                    "metrics": {
                        "leakage_current_a": 1e-8,
                        "breakdown_voltage_v": -20,
                        "specific_on_resistance_ohm_cm2": 0.05,
                    },
                },
                "quality_report": {"status": "passed", "metrics": {"leakage_current_a": 1e-8}},
                "mutation_effect_analysis": {
                    "mutation_target": "drift_doping",
                    "primary_metric": "specific_on_resistance_ohm_cm2",
                    "primary_improved": True,
                    "worth_continuing": True,
                    "decision": "continue_same_target",
                    "rationale": "Ron improved without blocking tradeoffs.",
                    "recommended_next_target": "drift_doping",
                    "recommended_next_direction": "decrease",
                    "improved_metrics": ["specific_on_resistance_ohm_cm2"],
                    "regressed_metrics": [],
                    "tradeoff_violations": [],
                },
            },
        )
        refined_dir = self.root / "runs" / "curve_decision_refined"
        refined_csv = refined_dir / "curve.csv"
        refined_state = refined_dir / "state.json"
        tool_requests: list[dict[str, object]] = []

        def fake_extended_device(request: dict[str, object]) -> dict[str, object]:
            tool_requests.append(request)
            refined_csv.parent.mkdir(parents=True, exist_ok=True)
            refined_csv.write_text(
                "drain_voltage_v,off_current_a,electric_field_v_per_cm\n0,1e-10,1e4\n-10,7e-9,1.7e5\n-20,7e-7,4.0e5\n",
                encoding="utf-8",
            )
            write_json(
                refined_state,
                {
                    "tool_name": "extended_device_sweep",
                    "status": "completed",
                    "run_id": "curve_decision_refined",
                    "request": request,
                    "final_summary": {
                        "artifacts": {"csv": str(refined_csv)},
                        "metrics": {
                            "leakage_current_a": 7e-9,
                            "breakdown_voltage_v": -24,
                            "specific_on_resistance_ohm_cm2": 0.046,
                        },
                    },
                    "quality_report": {"status": "passed", "metrics": {"leakage_current_a": 7e-9}},
                },
            )
            return {"status": "completed", "state_path": str(refined_state)}

        client = FakeAgentClient(
            json.dumps(
                {
                    "recommended_action": "refine_effective_mutation",
                    "recommended_target": "drift_doping",
                    "recommended_direction": "decrease",
                    "rationale": "The mutation improved Ron without tradeoffs, so refine drift doping with a smaller step.",
                    "evidence_used": ["mutation_effect_analysis", "metric_deltas", "curve_shape"],
                }
            )
        )
        request = AutonomousDevsimRequest(
            goal_text="Use the LLM curve reviewer to continue Power MOSFET drift doping optimization",
            agent_id="agent_curve_decision",
            agent_root=self.root / "agents",
            execute=True,
            use_llm=True,
            allow_llm_fallback=True,
            max_steps=6,
            source_state_path=str(source_state),
            max_mutation_refinements=1,
            generate_report=False,
            generate_dashboard=False,
        )

        state = run_autonomous_devsim_agent(
            request,
            runner_registry={
                "extended_device_sweep": fake_extended_device,
                "physical_benchmark": lambda request: {"status": "completed", "benchmark_path": str(self.root / "benchmark.json")},
            },
            llm_client=client,
        )

        self.assertEqual(state.status, DevsimAgentStatus.COMPLETED)
        self.assertIn(DevsimAgentActionKind.PLAN_CURVE_DECISION, [step.kind for step in state.steps])
        self.assertIn(DevsimAgentActionKind.PLAN_GUIDANCE_PATCH, [step.kind for step in state.steps])
        self.assertEqual(len(tool_requests), 1)
        self.assertIn("guidance_patch_id", tool_requests[0])
        self.assertLess(tool_requests[0]["power_mos_drift_region_doping_cm3"], 1.0e16)
        plan = state.checkpoint["latest_curve_decision_plan"]
        self.assertEqual(plan["decision_source"], "llm")
        self.assertEqual(plan["recommended_action"], "refine_effective_mutation")
        self.assertTrue(state.checkpoint["pending_curve_decision_plan"]["executed"])
        self.assertEqual(state.checkpoint["curve_decision_runs"], 1)

    def test_resume_enforces_queued_llm_curve_decision_before_benchmark(self) -> None:
        source_dir = self.root / "runs" / "queued_curve_decision_source"
        source_csv = source_dir / "curve.csv"
        source_csv.parent.mkdir(parents=True, exist_ok=True)
        source_csv.write_text(
            "drain_voltage_v,off_current_a,electric_field_v_per_cm\n0,1e-10,1e4\n-10,1e-8,2e5\n-20,1e-6,5e5\n",
            encoding="utf-8",
        )
        source_state = source_dir / "state.json"
        write_json(
            source_state,
            {
                "tool_name": "extended_device_sweep",
                "status": "completed",
                "run_id": "queued_curve_decision_source",
                "request": {
                    "device_type": "power_mosfet_bv_ron",
                    "fidelity": "devsim_2d_field_plate",
                    "power_mos_drift_region_doping_cm3": 1.0e16,
                    "tcad_deck_mutations": [
                        {
                            "name": "drift_doping",
                            "target": "drift_doping",
                            "operation": "set",
                            "request_path": "power_mos_drift_region_doping_cm3",
                            "values": [7.5e15, 1.25e16],
                        }
                    ],
                },
                "final_summary": {
                    "artifacts": {"csv": str(source_csv)},
                    "metrics": {
                        "leakage_current_a": 1e-8,
                        "breakdown_voltage_v": -20,
                        "specific_on_resistance_ohm_cm2": 0.05,
                    },
                },
                "quality_report": {"status": "passed", "metrics": {"leakage_current_a": 1e-8}},
                "mutation_effect_analysis": {
                    "mutation_target": "drift_doping",
                    "primary_metric": "specific_on_resistance_ohm_cm2",
                    "primary_improved": True,
                    "worth_continuing": True,
                    "decision": "continue_same_target",
                    "rationale": "Ron improved without blocking tradeoffs.",
                    "recommended_next_target": "drift_doping",
                    "recommended_next_direction": "increase",
                    "improved_metrics": ["specific_on_resistance_ohm_cm2"],
                    "regressed_metrics": [],
                    "tradeoff_violations": [],
                },
            },
        )
        first_client = FakeSequenceClient(
            [
                json.dumps(
                    {
                        "kind": "plan_curve_decision",
                        "reason": "Ask the curve reviewer to choose the next patch.",
                        "request": {"use_llm": True, "allow_llm_fallback": False},
                    }
                ),
                json.dumps(
                    {
                        "recommended_action": "refine_effective_mutation",
                        "recommended_target": "drift_doping",
                        "recommended_direction": "increase",
                        "rationale": "The mutation improved Ron without tradeoffs, so refine drift doping.",
                        "evidence_used": ["mutation_effect_analysis", "metric_deltas", "curve_shape"],
                    }
                ),
            ]
        )
        first_request = AutonomousDevsimRequest(
            goal_text="Resume the LLM curve-decision loop without skipping the queued patch.",
            agent_id="agent_curve_decision_resume",
            agent_root=self.root / "agents",
            execute=True,
            use_llm=True,
            allow_llm_fallback=False,
            max_steps=1,
            source_state_path=str(source_state),
            max_mutation_refinements=1,
            generate_report=False,
            generate_dashboard=False,
        )

        first_state = run_autonomous_devsim_agent(first_request, llm_client=first_client)

        self.assertEqual(first_state.steps[-1].kind, DevsimAgentActionKind.PLAN_CURVE_DECISION)
        self.assertFalse(first_state.checkpoint["pending_curve_decision_plan"].get("executed", False))

        resume_client = FakeSequenceClient(
            [
                json.dumps(
                    {
                        "kind": "run_physical_benchmark",
                        "reason": "Try to benchmark immediately.",
                        "source_state_path": str(source_state),
                    }
                )
            ]
        )
        resume_state = run_autonomous_devsim_agent(
            first_request.model_copy(update={"resume": True, "max_steps": 2}),
            llm_client=resume_client,
        )

        self.assertEqual(resume_state.steps[-1].kind, DevsimAgentActionKind.PLAN_GUIDANCE_PATCH)
        decision = resume_state.steps[-1].observation["agent_decision"]
        self.assertFalse(decision["fallback_used"])
        self.assertEqual(decision["decision_source"], "queued_llm_plan")
        self.assertTrue(decision["queued_plan_enforced"])
        self.assertEqual(decision["llm_requested_action"]["kind"], "run_physical_benchmark")
        self.assertTrue(resume_state.checkpoint["pending_curve_decision_plan"]["executed"])

    def test_sentaurus_effect_triggers_refinement_before_generic_patch_planning(self) -> None:
        project = self.root / "sentaurus_project"
        project.mkdir()
        (project / "device.cmd").write_text("set LIFETIME_SCALE 2.0\n", encoding="utf-8")
        sentaurus_state = self.root / "sentaurus" / "sentaurus_state.json"
        write_json(
            sentaurus_state,
            {
                "tool_name": "sentaurus_run",
                "status": "completed",
                "run_id": "sentaurus_patch",
                "project_copy_path": str(project),
                "request": {"goal_text": "Reduce leakage.", "deck_files": ["device.cmd"]},
                "quality_report": {"status": "passed", "metrics": {"leakage_abs_current_at_target_a": 4e-10}},
                "final_summary": {
                    "artifacts": {"project_copy": str(project)},
                    "parameters": {"deck_files": ["device.cmd"]},
                    "metrics": {"solver_backend": "sentaurus"},
                },
                "sentaurus_mutation_effect_analysis": {
                    "decision": "continue_refine",
                    "candidate_id": "device.cmd:lifetime:LIFETIME_SCALE",
                    "candidate": {
                        "candidate_id": "device.cmd:lifetime:LIFETIME_SCALE",
                        "patches": [{"file": "device.cmd", "operation": "sentaurus_set_variable", "variable": "LIFETIME_SCALE", "value": "2"}],
                        "validation_records": [
                            {
                                "file": "device.cmd",
                                "operation": "sentaurus_set_variable",
                                "variable": "LIFETIME_SCALE",
                                "verified": True,
                                "old_value": "1.0",
                                "value": "2",
                            }
                        ],
                    },
                },
            },
        )
        request = AutonomousDevsimRequest(
            goal_text="Reduce leakage.",
            agent_id="agent_sentaurus_refine",
            agent_root=self.root / "agents",
            execute=True,
            use_llm=False,
            sentaurus_project_path=project,
            enable_experiment_design=True,
            max_experiment_design_rounds=2,
            generate_report=False,
            generate_dashboard=False,
        )
        state = AutonomousDevsimAgentState(
            status=DevsimAgentStatus.RUNNING,
            agent_id="agent_sentaurus_refine",
            agent_dir=str(self.root / "agents" / "agent_sentaurus_refine"),
            goal_text=request.goal_text,
            created_at="2026-06-10T00:00:00Z",
            updated_at="2026-06-10T00:00:00Z",
            execute=True,
            max_steps=4,
            latest_state_path=str(sentaurus_state),
            checkpoint={"sentaurus_initial_run_done": True, "experiment_design_runs": 1},
        )

        action = deterministic_action(state, request)

        self.assertEqual(action.kind, DevsimAgentActionKind.PLAN_SENTAURUS_REFINEMENT)
        self.assertEqual(action.source_state_path, str(sentaurus_state))


if __name__ == "__main__":
    unittest.main()
