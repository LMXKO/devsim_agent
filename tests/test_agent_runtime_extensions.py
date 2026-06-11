from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.agent_curve_guidance import build_agent_curve_guidance
from tcad_agent.agent_memory import append_agent_memory_from_soak, retrieve_agent_memory
from tcad_agent.agent_recovery import build_recovery_decision
from tcad_agent.agent_soak_daemon import AgentSoakDaemonRequest, run_agent_soak_daemon
from tcad_agent.mission_spec_compiler import compile_mission_spec
from tcad_agent.run_queue import get_item


def write_curve_state(path: Path) -> Path:
    csv_path = path.parent / "curve.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "voltage_v,current_a,electric_field_v_per_cm\n0,1e-12,1e4\n-10,1e-9,2e5\n-20,1e-6,5e5\n",
        encoding="utf-8",
    )
    path.write_text(
        json.dumps(
            {
                "tool_name": "extended_device_sweep",
                "status": "completed",
                "final_summary": {
                    "artifacts": {"csv": str(csv_path)},
                    "metrics": {
                        "leakage_current_a": 1e-9,
                        "breakdown_voltage_v": -20,
                        "max_electric_field_v_per_cm": 5e5,
                    },
                },
                "quality_report": {"status": "passed", "metrics": {"points": 3}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


class AgentRuntimeExtensionTest(unittest.TestCase):
    def test_compile_mission_spec_expands_power_device_task(self) -> None:
        spec = compile_mission_spec("优化 power MOSFET BV/Ron/leakage 和 field peak")

        self.assertEqual(spec.selected_tool, "extended_device_sweep")
        self.assertEqual(spec.intent["device_family"], "power_mosfet")
        mutation_names = {item["name"] for item in spec.allowed_mutations}
        self.assertIn("field_plate", mutation_names)
        self.assertIn("drift_doping", mutation_names)
        self.assertIn("region_specific_lifetime", mutation_names)
        self.assertIn("curve_shape_diagnostic", spec.validation_plan)

    def test_agent_memory_appends_and_retrieves_relevant_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.jsonl"
            append_agent_memory_from_soak(
                {
                    "soak_id": "mem_power",
                    "status": "completed",
                    "request": {"goal_text": "优化 power MOSFET BV leakage"},
                    "completed_steps": 5,
                    "model_decisions": 5,
                    "fallback_decisions": 0,
                    "final_state_path": "/tmp/state.json",
                    "curve_guidance": {"recommended_action": "reduce_field_peak", "recommended_target": "field_plate"},
                },
                mission_spec=compile_mission_spec("优化 power MOSFET BV leakage").model_dump(mode="json"),
                memory_path=memory_path,
            )

            matches = retrieve_agent_memory("继续优化 LDMOS power MOSFET BV", memory_path=memory_path)

        self.assertEqual(matches[0]["soak_id"], "mem_power")
        self.assertEqual(matches[0]["outcome"], "useful")

    def test_recovery_classifies_llm_transport_and_convergence(self) -> None:
        llm = build_recovery_decision(
            failure_reason="Connection error.",
            agent_status="failed",
            completed_steps=0,
            autonomous_request={"allow_llm_fallback": False},
            recovery_events=[],
            max_attempts=2,
        )
        convergence = build_recovery_decision(
            failure_reason="Newton convergence failed",
            agent_status="failed",
            completed_steps=1,
            autonomous_request={"allow_llm_fallback": True},
            recovery_events=[],
            max_attempts=2,
        )

        self.assertEqual(llm.family, "llm_transport")
        self.assertTrue(llm.should_retry)
        self.assertFalse(llm.request_patch)
        self.assertEqual(convergence.family, "simulator_convergence")
        self.assertTrue(convergence.request_patch["enable_experiment_design"])

    def test_curve_guidance_reads_curve_shape_and_recommends_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = write_curve_state(Path(tmp) / "state.json")

            guidance = build_agent_curve_guidance(
                goal_text="优化 power MOSFET field peak 和 BV",
                source_state_path=str(state_path),
            )

        self.assertEqual(guidance.status, "completed")
        self.assertEqual(guidance.recommended_target, "field_plate")
        self.assertIsNotNone(guidance.shape)
        self.assertEqual(guidance.shape["field_peak_value"], 5e5)

    def test_curve_guidance_uses_mutation_effect_before_goal_heuristic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = write_curve_state(Path(tmp) / "state.json")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["mutation_effect_analysis"] = {
                "mutation_target": "field_plate",
                "primary_metric": "max_electric_field_v_per_cm",
                "primary_improved": True,
                "worth_continuing": True,
                "decision": "continue_same_target",
                "rationale": "field peak dropped and Ron stayed inside tolerance",
                "recommended_next_target": "field_plate",
                "recommended_next_direction": "increase",
                "improved_metrics": ["max_electric_field_v_per_cm"],
                "regressed_metrics": [],
                "tradeoff_violations": [],
            }
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

            guidance = build_agent_curve_guidance(
                goal_text="继续看 power MOSFET 下一轮 deck patch",
                source_state_path=str(state_path),
            )

        self.assertEqual(guidance.recommended_action, "refine_effective_mutation")
        self.assertEqual(guidance.recommended_target, "field_plate")
        self.assertIn("baseline_vs_mutation_effect", guidance.decision_basis)
        self.assertTrue(guidance.mutation_effect["worth_continuing"])

    def test_agent_soak_daemon_enqueues_and_runs_agent_soak_item(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "queue.sqlite"

            state = run_agent_soak_daemon(
                AgentSoakDaemonRequest(
                    daemon_id="daemon_unit",
                    queue_id="daemon_queue",
                    goal_text="Run a short daemon soak",
                    queue_db_path=db,
                    daemon_root=root / "daemon",
                    execute=False,
                    max_loops=1,
                    max_idle_loops=1,
                    poll_interval_seconds=0,
                    autonomous_request={"use_llm": False},
                ),
                registry={"agent_soak": lambda request: {"status": "completed", "state_path": str(root / "soak_state.json")}},
            )
            item = get_item(db, "daemon_queue")

        self.assertEqual(state.status, "completed")
        self.assertEqual(item.tool_name, "agent_soak")
        self.assertEqual(item.status.value, "completed")


if __name__ == "__main__":
    unittest.main()
