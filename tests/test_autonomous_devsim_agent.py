from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.autonomous_devsim_agent import (
    AutonomousDevsimRequest,
    DevsimAgentActionKind,
    DevsimAgentStatus,
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
                DevsimAgentActionKind.STOP_SUCCESS,
            ],
        )
        self.assertEqual(calls, ["tool", "repair", "benchmark", "report", "conclusion"])
        self.assertEqual(state.final_state_path, str(repaired_state))
        self.assertEqual(state.final_report_path, str(self.root / "conclusion.md"))
        self.assertEqual(state.checkpoint["dashboard_skipped_reason"], "dashboard supports sweep/optimization states; latest state is a single tool run")

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


if __name__ == "__main__":
    unittest.main()
