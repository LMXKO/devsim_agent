from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.agent_experiment_design import build_agent_experiment_design_plan
from tcad_agent.autonomous_devsim_agent import (
    AutonomousDevsimRequest,
    DevsimAgentActionKind,
    DevsimAgentStatus,
    run_autonomous_devsim_agent,
)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_power_state(path: Path, *, measured_curve_path: str | None = None) -> Path:
    curve = path.parent / "curve.csv"
    curve.parent.mkdir(parents=True, exist_ok=True)
    curve.write_text("drain_voltage_v,off_current_a,electric_field_v_per_cm\n0,1e-10,0\n-10,1e-8,1.5e5\n", encoding="utf-8")
    request = {
        "device_type": "power_mosfet_bv_ron",
        "fidelity": "physics_1d",
        "evidence_level": "tcad_executable",
        "power_mos_junction_mesh_spacing_um": 0.01,
        "power_mos_field_plate_length_um": 1.5,
    }
    if measured_curve_path:
        request["measured_curve_path"] = measured_curve_path
    write_json(
        path,
        {
            "tool_name": "extended_device_sweep",
            "status": "completed",
            "run_id": path.parent.name,
            "request": request,
            "final_summary": {
                "artifacts": {"csv": str(curve)},
                "metrics": {
                    "device_type": "power_mosfet_bv_ron",
                    "fidelity": "physics_1d",
                    "evidence_level": "tcad_executable",
                    "leakage_current_a": 1e-8,
                    "max_electric_field_v_per_cm": 1.5e5,
                    "specific_on_resistance_ohm_cm2": 0.05,
                },
            },
            "quality_report": {
                "status": "passed",
                "metrics": {
                    "device_type": "power_mosfet_bv_ron",
                    "fidelity": "physics_1d",
                    "evidence_level": "tcad_executable",
                    "leakage_current_a": 1e-8,
                    "max_electric_field_v_per_cm": 1.5e5,
                    "specific_on_resistance_ohm_cm2": 0.05,
                },
            },
        },
    )
    return path


class AgentExperimentDesignTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_builds_convergence_and_golden_candidates_from_signoff_gaps(self) -> None:
        reference = self.root / "reference.csv"
        reference.write_text("drain_voltage_v,off_current_a\n0,1e-10\n-10,1e-8\n", encoding="utf-8")
        state_path = write_power_state(self.root / "power" / "state.json", measured_curve_path=str(reference))

        plan = build_agent_experiment_design_plan(state_path, output_path=self.root / "plan.json")
        candidates = {candidate.candidate_id: candidate for candidate in plan.candidates}

        self.assertEqual(plan.status, "completed")
        self.assertIn("collect_convergence_evidence", candidates)
        self.assertIn("collect_golden_measured_correlation", candidates)
        self.assertEqual(candidates["collect_convergence_evidence"].tool_name, "tool_convergence")
        self.assertEqual(candidates["collect_convergence_evidence"].request["axis_path"], "power_mos_junction_mesh_spacing_um")
        self.assertEqual(candidates["collect_golden_measured_correlation"].tool_name, "golden_curve_comparison")
        self.assertTrue(Path(plan.output_path).exists())

    def test_autonomous_agent_executes_highest_ranked_design_candidate(self) -> None:
        state_path = write_power_state(self.root / "power_agent" / "state.json")
        convergence_state = self.root / "convergence" / "state.json"
        tool_requests: list[dict[str, object]] = []
        benchmark_sources: list[str] = []

        def fake_tool_convergence(request: dict[str, object]) -> dict[str, object]:
            tool_requests.append(request)
            write_json(
                convergence_state,
                {
                    "tool_name": "tool_convergence",
                    "status": "completed",
                    "quality_report": {
                        "status": "passed",
                        "metrics": {
                            "relative_delta": 0.02,
                            "axis_path": request["axis_path"],
                            "metric_path": request["metric_path"],
                        },
                    },
                    "final_summary": {"metrics": {"relative_delta": 0.02}, "artifacts": {}},
                },
            )
            return {"status": "completed", "state_path": str(convergence_state)}

        def fake_physical_benchmark(request: dict[str, object]) -> dict[str, object]:
            benchmark_sources.append(str(request["source"]))
            return {"status": "completed", "benchmark_path": str(self.root / "missing_benchmark.json")}

        request = AutonomousDevsimRequest(
            goal_text="Run signoff promotion experiment design",
            agent_id="agent_design",
            agent_root=self.root / "agents",
            execute=True,
            use_llm=False,
            max_steps=5,
            source_state_path=str(state_path),
            enable_experiment_design=True,
            max_experiment_design_rounds=1,
            generate_report=False,
            generate_dashboard=False,
        )

        state = run_autonomous_devsim_agent(
            request,
            runner_registry={
                "physical_benchmark": fake_physical_benchmark,
                "tool_convergence": fake_tool_convergence,
            },
        )

        self.assertEqual(state.status, DevsimAgentStatus.COMPLETED)
        self.assertIn(DevsimAgentActionKind.PLAN_EXPERIMENT_DESIGN, [step.kind for step in state.steps])
        self.assertEqual(tool_requests[0]["agent_experiment_candidate_id"], "collect_convergence_evidence")
        self.assertEqual(tool_requests[0]["tool_name"], "extended_device_sweep")
        self.assertEqual(benchmark_sources, [str(state_path), str(convergence_state)])
        self.assertEqual(state.checkpoint["executed_agent_experiment_candidates"], 1)
        self.assertTrue(Path(state.checkpoint["experiment_design_plan_path"]).exists())


if __name__ == "__main__":
    unittest.main()
