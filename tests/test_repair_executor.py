from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from tcad_agent.llm import LLMConfig
from tcad_agent.repair_executor import RepairExecutionStatus, run_repair_executor


class FakeAgentClient:
    config = LLMConfig(model="fake-repair-agent")

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        return self.response


class RepairExecutorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run_root = self.root / "agent_tools"
        self.old_repair_memory_path = os.environ.get("ACTSOFT_REPAIR_MEMORY_PATH")
        os.environ["ACTSOFT_REPAIR_MEMORY_PATH"] = str(self.root / "repair_case_memory.jsonl")

    def tearDown(self) -> None:
        if self.old_repair_memory_path is None:
            os.environ.pop("ACTSOFT_REPAIR_MEMORY_PATH", None)
        else:
            os.environ["ACTSOFT_REPAIR_MEMORY_PATH"] = self.old_repair_memory_path
        self.tmp.cleanup()

    def write_source_state(self, *, issue_code: str = "current_not_monotonic", failure_class: str | None = None) -> Path:
        run_dir = self.run_root / "pn_junction_iv" / "source_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "tool_name": "pn_junction_iv_sweep",
            "status": "completed" if failure_class is None else "failed",
            "run_id": "source_run",
            "run_dir": str(run_dir),
            "request": {
                "start": 0.0,
                "stop": 1.0,
                "step": 0.5,
                "min_step": 0.125,
                "max_attempts": 2,
                "run_root": str(self.run_root),
            },
            "attempts": [{"failure_class": failure_class}] if failure_class else [],
            "quality_report": {
                "status": "suspicious",
                "issues": [{"code": issue_code, "severity": "warning"}],
                "metrics": {},
            },
        }
        state_path = run_dir / "state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        return state_path

    def write_mosfet_convergence_state(self) -> Path:
        run_dir = self.run_root / "mosfet_2d_id" / "mos_bad"
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "tool_name": "mosfet_2d_id_sweep",
            "status": "failed",
            "run_id": "mos_bad",
            "run_dir": str(run_dir),
            "request": {
                "sweep_type": "idvd",
                "gate_start": 0.8,
                "gate_stop": 1.2,
                "gate_step": 0.2,
                "min_gate_step": 0.05,
                "drain_start": 0.0,
                "drain_stop": 1.2,
                "drain_step": 0.05,
                "min_drain_step": 0.0125,
                "idvd_gate_voltage": 1.2,
                "impact_ionization_model": "selberherr",
                "x_divisions": 8,
                "silicon_y_divisions": 3,
                "run_root": str(self.run_root),
            },
            "attempts": [{"failure_class": "convergence", "failure_reason": "DEVSIM solver did not converge."}],
            "quality_report": {
                "status": "failed",
                "issues": [{"code": "too_many_convergence_failures", "severity": "error"}],
                "metrics": {},
            },
        }
        state_path = run_dir / "state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        return state_path

    def write_mosfet_schema_state(self) -> Path:
        run_dir = self.run_root / "mosfet_2d_id" / "schema_bad"
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "tool_name": "mosfet_2d_id_sweep",
            "status": "failed",
            "run_id": "schema_bad",
            "run_dir": str(run_dir),
            "request": {
                "sweep_type": "output_characteristic",
                "gate_values": [0.8, 1.0, 1.2],
                "drain_start": 0.0,
                "drain_stop": 1.2,
                "drain_step": 0.1,
                "run_root": str(self.run_root),
            },
            "attempts": [{"failure_class": "validation", "failure_reason": "sweep_type output_characteristic invalid"}],
            "quality_report": {"status": "failed", "issues": []},
        }
        state_path = run_dir / "state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        return state_path

    def write_tool_convergence_failed_case_state(self) -> Path:
        run_dir = self.root / "tool_convergence" / "conv_bad"
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "tool_name": "tool_convergence",
            "status": "failed",
            "convergence_id": "conv_bad",
            "convergence_dir": str(run_dir),
            "target_tool": "mosfet_2d_id_sweep",
            "axis_path": "x_divisions",
            "values": [8, 12, 16],
            "cases": [
                {
                    "index": 1,
                    "status": "failed",
                    "failure_reason": "DEVSIM solver did not converge.",
                    "request": {
                        "sweep_type": "idvd",
                        "drain_start": 0.0,
                        "drain_stop": 1.2,
                        "drain_step": 0.1,
                        "min_drain_step": 0.025,
                        "gate_step": 0.2,
                        "min_gate_step": 0.05,
                        "run_root": str(self.run_root),
                    },
                }
            ],
            "quality_report": {
                "status": "failed",
                "issues": [{"code": "too_few_completed_convergence_cases", "severity": "error"}],
                "metrics": {"cases": 3, "completed_cases": 0},
            },
        }
        state_path = run_dir / "state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        return state_path

    def passing_runner(self, request: dict[str, object]) -> dict[str, object]:
        run_dir = Path(str(request["run_root"])) / "pn_junction_iv" / str(request["run_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "tool_name": "pn_junction_iv_sweep",
            "status": "completed",
            "run_id": request["run_id"],
            "run_dir": str(run_dir),
            "request": request,
            "attempts": [],
            "quality_report": {"status": "passed", "issues": [], "metrics": {}},
        }
        (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
        return state

    def suspicious_runner(self, request: dict[str, object]) -> dict[str, object]:
        run_dir = Path(str(request["run_root"])) / "pn_junction_iv" / str(request["run_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "tool_name": "pn_junction_iv_sweep",
            "status": "completed",
            "run_id": request["run_id"],
            "run_dir": str(run_dir),
            "request": request,
            "attempts": [],
            "quality_report": {
                "status": "suspicious",
                "issues": [{"code": "current_not_monotonic", "severity": "warning"}],
                "metrics": {},
            },
        }
        (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
        return state

    def mosfet_runner_passes_only_after_model_staging(self, request: dict[str, object]) -> dict[str, object]:
        run_dir = Path(str(request["run_root"])) / "mosfet_2d_id" / str(request["run_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        staged_model = (
            request.get("impact_ionization_model") == "none"
            and request.get("deferred_impact_ionization_model") == "selberherr"
        )
        state = {
            "tool_name": "mosfet_2d_id_sweep",
            "status": "completed" if staged_model else "failed",
            "run_id": request["run_id"],
            "run_dir": str(run_dir),
            "request": request,
            "attempts": [] if staged_model else [{"failure_class": "convergence"}],
            "quality_report": {
                "status": "passed" if staged_model else "failed",
                "issues": [] if staged_model else [{"code": "too_many_convergence_failures", "severity": "error"}],
                "metrics": {},
            },
        }
        (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
        return state

    def mosfet_passing_runner(self, request: dict[str, object]) -> dict[str, object]:
        run_dir = Path(str(request["run_root"])) / "mosfet_2d_id" / str(request["run_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "tool_name": "mosfet_2d_id_sweep",
            "status": "completed",
            "run_id": request["run_id"],
            "run_dir": str(run_dir),
            "request": request,
            "attempts": [],
            "quality_report": {"status": "passed", "issues": [], "metrics": {}},
        }
        (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
        return state

    def moscap_physical_failure_runner(self, request: dict[str, object]) -> dict[str, object]:
        run_dir = Path(str(request["run_root"])) / "mos_capacitor_cv" / str(request["run_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "tool_name": "mos_capacitor_cv_sweep",
            "status": "completed",
            "run_id": request["run_id"],
            "run_dir": str(run_dir),
            "request": {"oxide_thickness_nm": 5.0, **request},
            "quality_report": {
                "status": "passed",
                "issues": [],
                "metrics": {
                    "min_capacitance_f_per_cm2": 1e-8,
                    "max_capacitance_f_per_cm2": 2e-6,
                    "final_capacitance_f_per_cm2": 1e-6,
                },
            },
        }
        (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
        return state

    def extended_field_plate_runner(self, request: dict[str, object]) -> dict[str, object]:
        run_dir = Path(str(request["run_root"])) / "extended_devices" / "power_mosfet_bv_ron" / str(request["run_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        csv_path = run_dir / "sweep.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "drain_voltage_v,off_current_a,electric_field_v_per_cm",
                    "0,1e-10,0",
                    "-10,2e-10,2e5",
                    "-20,3e-10,8e5",
                ]
            ),
            encoding="utf-8",
        )
        state = {
            "tool_name": "extended_device_sweep",
            "status": "completed",
            "run_id": request["run_id"],
            "run_dir": str(run_dir),
            "request": request,
            "final_summary": {
                "artifacts": {"csv": str(csv_path)},
                "metrics": {
                    "leakage_current_a": 3e-10,
                    "max_electric_field_v_per_cm": 8e5,
                    "breakdown_voltage_v": -90.0,
                    "specific_on_resistance_ohm_cm2": 0.05,
                },
            },
            "quality_report": {
                "status": "suspicious",
                "issues": [{"code": "power_mos_field_exceeds_critical_margin", "severity": "warning"}],
                "metrics": {
                    "leakage_current_a": 3e-10,
                    "max_electric_field_v_per_cm": 8e5,
                    "breakdown_voltage_v": -90.0,
                    "specific_on_resistance_ohm_cm2": 0.05,
                },
            },
        }
        (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
        return state

    def test_plan_only_builds_next_repair_request(self) -> None:
        source = self.write_source_state()

        state = run_repair_executor(source, execution_id="repair_plan", execute=False)

        self.assertEqual(state.status, RepairExecutionStatus.PLANNED)
        self.assertEqual(len(state.attempts), 1)
        self.assertEqual(state.attempts[0].action_name, "local_bias_step_refinement")
        self.assertEqual(state.attempts[0].next_request["step"], 0.25)
        self.assertTrue((source.parent / "repair_execution" / "repair_plan" / "repair_execution_state.json").exists())

    def test_agent_policy_can_choose_next_repair_action(self) -> None:
        source = self.write_source_state()
        client = FakeAgentClient(
            json.dumps(
                {
                    "action": {
                        "name": "agent_local_curve_probe",
                        "priority": 140,
                        "reason": "曲线非单调但不是收敛失败，先缩小局部 step 取得更高信息密度。",
                        "target_tool": "pn_junction_iv_sweep",
                        "request_patch": {"step": 0.2, "agent_policy": "curve_shape_probe"},
                        "checklist": ["compare refined curve with baseline", "check monotonicity count"],
                        "expected_effect": "用更细 bias 分辨真实拐点和数值跳变。",
                    },
                    "observation_summary": "I-V 曲线出现非单调，现有点数不足以区分数值跳变和真实 kink。",
                    "hypothesis_zh": "当前更像 bias 步长造成的数值假象，而不是先改物理模型。",
                    "tool_plan": [
                        {
                            "tool": "curve_diagnostics.overlay",
                            "why": "对比 refined curve 与 baseline",
                            "expected_evidence": "非单调段是否随步长缩小而消失",
                        }
                    ],
                    "safety_review": {"risk_level": "low", "requires_user_confirmation": False},
                    "evidence_used": ["quality_issues", "metrics", "deterministic_fallback_actions"],
                }
            )
        )

        state = run_repair_executor(
            source,
            execution_id="repair_agent_plan",
            execute=False,
            use_agent_policy=True,
            llm_client=client,
        )

        self.assertEqual(state.status, RepairExecutionStatus.PLANNED)
        self.assertEqual(state.attempts[0].action_name, "agent_local_curve_probe")
        self.assertEqual(state.attempts[0].next_request["step"], 0.2)
        self.assertEqual(state.attempts[0].agent_policy["status"], "completed")
        self.assertEqual(state.attempts[0].agent_policy["hypothesis_zh"], "当前更像 bias 步长造成的数值假象，而不是先改物理模型。")
        self.assertTrue(
            any(
                "Agent tool plan: curve_diagnostics.overlay" in item
                for item in state.attempts[0].agent_policy["action"]["checklist"]
            )
        )
        self.assertEqual(len(client.calls), 1)

    def test_agent_policy_falls_back_when_response_is_invalid(self) -> None:
        source = self.write_source_state()
        client = FakeAgentClient("not json")

        state = run_repair_executor(
            source,
            execution_id="repair_agent_fallback",
            execute=False,
            use_agent_policy=True,
            llm_client=client,
        )

        self.assertEqual(state.status, RepairExecutionStatus.PLANNED)
        self.assertEqual(state.attempts[0].action_name, "local_bias_step_refinement")
        self.assertTrue(state.attempts[0].agent_policy["fallback_used"])

    def test_agent_policy_requires_confirmation_for_high_risk_deck_patch(self) -> None:
        source = self.write_source_state()
        client = FakeAgentClient(
            json.dumps(
                {
                    "action": {
                        "name": "agent_guard_ring_geometry_patch",
                        "priority": 150,
                        "reason": "agent wants to try a termination geometry patch",
                        "target_tool": "pn_junction_iv_sweep",
                        "deck_patch": {
                            "operation": "set",
                            "request_path": "guard_ring_spacing_um",
                            "deck_path": "geometry.guard_ring_spacing_um",
                            "value": 1.2,
                            "target": "guard_ring",
                        },
                        "expected_effect": "may reduce edge field crowding",
                    },
                    "safety_review": {"risk_level": "high", "requires_user_confirmation": True},
                }
            )
        )

        state = run_repair_executor(
            source,
            execution_id="repair_agent_guard_ring_wait",
            execute=True,
            use_agent_policy=True,
            llm_client=client,
            registry={"pn_junction_iv_sweep": self.passing_runner},
        )

        self.assertEqual(state.status, RepairExecutionStatus.WAITING_FOR_USER)
        self.assertEqual(state.attempts, [])
        blocked = state.checkpoint["blocked_repair_agent_decision"]
        self.assertTrue(blocked["action"]["user_confirmation_required"])
        self.assertIn("high-risk", " ".join(blocked["warnings"]))

    def test_execute_runs_repair_and_accepts_passed_result(self) -> None:
        source = self.write_source_state()

        state = run_repair_executor(
            source,
            execution_id="repair_exec",
            execute=True,
            registry={"pn_junction_iv_sweep": self.passing_runner},
        )

        self.assertEqual(state.status, RepairExecutionStatus.COMPLETED)
        self.assertEqual(state.final_quality_status, "passed")
        self.assertEqual(len(state.attempts), 1)
        self.assertTrue(Path(state.final_state_path).exists())

    def test_sensitive_repair_waits_for_user_by_default(self) -> None:
        source = self.write_source_state(issue_code="junction_not_inside_device")

        state = run_repair_executor(
            source,
            execution_id="repair_wait",
            execute=True,
            registry={"pn_junction_iv_sweep": self.passing_runner},
        )

        self.assertEqual(state.status, RepairExecutionStatus.WAITING_FOR_USER)
        self.assertEqual(state.attempts, [])
        self.assertIn("blocked_repair_plan_path", state.checkpoint)

    def test_max_rounds_fails_if_quality_never_passes(self) -> None:
        source = self.write_source_state()

        state = run_repair_executor(
            source,
            execution_id="repair_budget",
            execute=True,
            max_rounds=2,
            registry={"pn_junction_iv_sweep": self.suspicious_runner},
        )

        self.assertEqual(state.status, RepairExecutionStatus.FAILED)
        self.assertIn("maximum repair rounds", state.failure_reason)
        self.assertEqual(len(state.attempts), 2)
        self.assertEqual(state.attempts[1].next_request["run_id"].count("_repair_"), 1)
        self.assertLess(len(str(state.attempts[1].next_request["run_id"])), 120)

    def test_execute_tries_next_repair_strategy_after_failed_attempt(self) -> None:
        source = self.write_mosfet_convergence_state()

        state = run_repair_executor(
            source,
            execution_id="repair_model_staging",
            execute=True,
            max_rounds=3,
            registry={"mosfet_2d_id_sweep": self.mosfet_runner_passes_only_after_model_staging},
        )

        self.assertEqual(state.status, RepairExecutionStatus.COMPLETED)
        self.assertEqual(
            [attempt.action_name for attempt in state.attempts],
            ["continuation_bias_ramp", "reuse_last_successful_initial_solution", "model_switch_staging"],
        )
        self.assertTrue(state.attempts[1].next_request["resume"])
        self.assertEqual(state.attempts[2].next_request["impact_ionization_model"], "none")
        self.assertEqual(state.attempts[2].next_request["deferred_impact_ionization_model"], "selberherr")

    def test_tool_convergence_failed_case_executes_target_tool_retry(self) -> None:
        source = self.write_tool_convergence_failed_case_state()

        state = run_repair_executor(
            source,
            execution_id="repair_toolconv_case",
            execute=True,
            max_rounds=2,
            registry={"mosfet_2d_id_sweep": self.mosfet_passing_runner},
        )

        self.assertEqual(state.status, RepairExecutionStatus.COMPLETED)
        self.assertEqual(state.attempts[0].target_tool, "mosfet_2d_id_sweep")
        self.assertEqual(state.attempts[0].action_name, "rerun_failed_convergence_cases_with_safe_bias")
        self.assertLess(state.attempts[0].next_request["drain_step"], 0.1)
        self.assertTrue(state.attempts[0].next_request["resume"])

    def test_schema_alias_repair_executes_normalized_request(self) -> None:
        source = self.write_mosfet_schema_state()

        state = run_repair_executor(
            source,
            execution_id="repair_schema",
            execute=True,
            registry={"mosfet_2d_id_sweep": self.mosfet_passing_runner},
        )

        self.assertEqual(state.status, RepairExecutionStatus.COMPLETED)
        self.assertEqual(state.attempts[0].action_name, "schema_field_alias_normalization")
        self.assertEqual(state.attempts[0].next_request["sweep_type"], "idvd")
        self.assertEqual(state.attempts[0].next_request["idvd_gate_voltage"], 1.2)

    def test_repair_runs_physical_benchmark_and_augments_failed_evidence(self) -> None:
        run_dir = self.run_root / "mos_capacitor_cv" / "moscap_bad"
        run_dir.mkdir(parents=True, exist_ok=True)
        source_state = {
            "tool_name": "mos_capacitor_cv_sweep",
            "status": "completed",
            "run_id": "moscap_bad",
            "run_dir": str(run_dir),
            "request": {"start": -1.0, "stop": 1.0, "step": 0.5, "run_root": str(self.run_root)},
            "quality_report": {
                "status": "suspicious",
                "issues": [{"code": "moscap_cv_dynamic_range_too_low", "severity": "warning"}],
                "metrics": {},
            },
        }
        source = run_dir / "state.json"
        source.write_text(json.dumps(source_state), encoding="utf-8")

        state = run_repair_executor(
            source,
            execution_id="repair_benchmark_gate",
            execute=True,
            max_rounds=1,
            registry={"mos_capacitor_cv_sweep": self.moscap_physical_failure_runner},
        )

        self.assertEqual(state.status, RepairExecutionStatus.FAILED)
        self.assertEqual(state.attempts[0].benchmark_status, "failed")
        self.assertIn("last_repair_benchmark", state.checkpoint)
        self.assertTrue(Path(state.current_state_path).exists())
        augmented = json.loads(Path(state.current_state_path).read_text(encoding="utf-8"))
        codes = {issue["code"] for issue in augmented["quality_report"]["issues"]}
        self.assertIn("moscap_capacitance_exceeds_cox", codes)

    def test_deck_mutation_attempt_records_curve_effect_analysis(self) -> None:
        run_dir = self.run_root / "extended_devices" / "power_mosfet_bv_ron" / "power_source"
        run_dir.mkdir(parents=True, exist_ok=True)
        csv_path = run_dir / "sweep.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "drain_voltage_v,off_current_a,electric_field_v_per_cm",
                    "0,1e-10,0",
                    "-10,5e-10,4e5",
                    "-20,8e-10,1e6",
                ]
            ),
            encoding="utf-8",
        )
        source_state = {
            "tool_name": "extended_device_sweep",
            "status": "completed",
            "run_id": "power_source",
            "run_dir": str(run_dir),
            "request": {
                "device_type": "power_mosfet_bv_ron",
                "fidelity": "physics_1d",
                "power_mos_field_plate_length_um": 1.5,
                "run_root": str(self.run_root),
                "tcad_deck_mutations": [
                    {
                        "name": "sweep_field_plate_length",
                        "target": "field_plate",
                        "request_path": "power_mos_field_plate_length_um",
                        "deck_path": "geometry.field_plate_length_um",
                        "values": [2.1, 1.5, 0.9],
                        "reason": "vary field plate",
                    }
                ],
            },
            "final_summary": {
                "artifacts": {"csv": str(csv_path)},
                "metrics": {
                    "leakage_current_a": 8e-10,
                    "max_electric_field_v_per_cm": 1e6,
                    "breakdown_voltage_v": -90.0,
                    "specific_on_resistance_ohm_cm2": 0.05,
                },
            },
            "quality_report": {
                "status": "suspicious",
                "issues": [{"code": "power_mos_field_exceeds_critical_margin", "severity": "warning"}],
                "metrics": {
                    "leakage_current_a": 8e-10,
                    "max_electric_field_v_per_cm": 1e6,
                    "breakdown_voltage_v": -90.0,
                    "specific_on_resistance_ohm_cm2": 0.05,
                },
            },
        }
        source = run_dir / "state.json"
        source.write_text(json.dumps(source_state), encoding="utf-8")

        client = FakeAgentClient(
            json.dumps(
                {
                    "action": {
                        "name": "agent_refine_field_plate",
                        "priority": 155,
                        "reason": "baseline field peak is high; field plate mutation is the most direct low-risk termination probe.",
                        "target_tool": "extended_device_sweep",
                        "request_patch": {"power_mos_field_plate_length_um": 2.1},
                        "deck_patch": {
                            "operation": "set",
                            "request_path": "power_mos_field_plate_length_um",
                            "deck_path": "geometry.field_plate_length_um",
                            "value": 2.1,
                            "target": "field_plate",
                            "agent_rationale": "extend field plate and compare field/leakage/BV overlay before touching doping.",
                        },
                        "checklist": ["compare overlay", "check Ron/BV tradeoff"],
                        "expected_effect": "field peak and leakage should improve; Ron should stay bounded.",
                    },
                    "observation_summary": "baseline leakage and field are both high around the reverse sweep endpoint.",
                    "hypothesis_zh": "termination field crowding dominates, so field plate length is a better first lever than lifetime.",
                    "tool_plan": [
                        {
                            "tool": "curve_diagnostics.overlay",
                            "why": "overlay baseline and mutation",
                            "expected_evidence": "field peak and leakage decrease without BV/Ron violation",
                        }
                    ],
                    "safety_review": {
                        "risk_level": "medium",
                        "requires_user_confirmation": False,
                        "constraints_checked": ["BV", "Ron", "field", "leakage"],
                    },
                    "evidence_used": ["mutation_effect_analysis", "quality_issues", "tcad_deck_mutations"],
                }
            )
        )

        state = run_repair_executor(
            source,
            execution_id="repair_curve_effect",
            execute=True,
            max_rounds=1,
            allow_user_confirmation_actions=True,
            use_agent_policy=True,
            llm_client=client,
            registry={"extended_device_sweep": self.extended_field_plate_runner},
        )

        self.assertEqual(len(state.attempts), 1)
        analysis = state.attempts[0].mutation_effect_analysis
        self.assertIsNotNone(analysis)
        self.assertTrue(analysis["primary_improved"])
        self.assertEqual(analysis["decision"], "continue_same_target")
        self.assertTrue(Path(analysis["overlay_svg_path"]).exists())
        repaired = json.loads(Path(state.attempts[0].result_state_path).read_text(encoding="utf-8"))
        self.assertIn("baseline_mutation_overlay", repaired["final_summary"]["artifacts"])
        self.assertEqual(repaired["repair_context"]["agent_hypothesis_zh"], "termination field crowding dominates, so field plate length is a better first lever than lifetime.")
        self.assertTrue(Path(repaired["repair_context"]["repair_case_memory_path"]).exists())
        self.assertEqual(state.attempts[0].agent_policy["tool_plan"][0]["tool"], "curve_diagnostics.overlay")


if __name__ == "__main__":
    unittest.main()
