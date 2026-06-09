from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.autonomous_devsim_agent import (
    AutonomousDevsimRequest,
    DevsimAgentActionKind,
    DevsimAgentStatus,
    observe_state,
    run_autonomous_devsim_agent,
)
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


if __name__ == "__main__":
    unittest.main()
