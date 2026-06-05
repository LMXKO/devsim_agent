from __future__ import annotations

import json
import unittest

from tcad_agent.llm import LLMConfig
from tcad_agent.task_planner import (
    PlannerStatus,
    build_task_spec_from_planner_json,
    parse_json_object,
    plan_task_text_with_llm,
    task_spec_from_planning_result,
)


class FakeClient:
    config = LLMConfig(model="fake-planner")

    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, str | float]] = []

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class TaskPlannerTest(unittest.TestCase):
    def test_parse_json_object_extracts_wrapped_json(self) -> None:
        parsed = parse_json_object('prefix {"task_id":"x"} suffix')

        self.assertEqual(parsed, {"task_id": "x"})

    def test_build_task_spec_repairs_aliases_and_min_step(self) -> None:
        spec, repairs = build_task_spec_from_planner_json(
            {
                "task_spec": {
                    "task_id": "model_id",
                    "title": "PN IV",
                    "intent": "iv_sweep",
                    "device": "diode",
                    "simulator": "DEVSIM",
                    "sweep": {"start": 0.0, "stop": 5.0, "step": 0.2, "min_step": 1.0},
                    "execution": {"attempts": 3, "cycles": 4},
                    "extra": "ignored",
                }
            },
            text="做 PN IV 到 5V",
            task_id="forced_id",
            execution_use_llm=False,
        )

        self.assertEqual(spec.task_id, "forced_id")
        self.assertEqual(spec.sweep.stop_v, 5.0)
        self.assertEqual(spec.sweep.step_v, 0.2)
        self.assertEqual(spec.sweep.min_step_v, 0.05)
        self.assertEqual(spec.execution.max_attempts, 3)
        self.assertEqual(spec.execution.max_cycles, 4)
        self.assertFalse(spec.execution.use_llm)
        self.assertTrue(any("Mapped sweep.step" in item for item in repairs))
        self.assertTrue(any("Ignored unsupported planner field" in item for item in repairs))

    def test_build_task_spec_repairs_parameter_aliases(self) -> None:
        spec, repairs = build_task_spec_from_planner_json(
            {
                "task_spec": {
                    "geometry": {"length_um": 0.2, "junction_um": 0.3},
                    "doping": {"p": 1e17, "n": 2e17},
                    "parameters": {"temperature": 350},
                    "mesh": {"contact_mesh": 0.002, "junction_mesh": 0.00002},
                }
            },
            text="PN IV 到 0.5V",
            task_id="param_alias",
            execution_use_llm=False,
        )

        self.assertEqual(spec.parameters.length_um, 0.2)
        self.assertEqual(spec.parameters.junction_um, 0.1)
        self.assertEqual(spec.parameters.p_doping_cm3, 1e17)
        self.assertEqual(spec.parameters.n_doping_cm3, 2e17)
        self.assertEqual(spec.parameters.temperature_k, 350)
        self.assertEqual(spec.mesh.contact_spacing_um, 0.002)
        self.assertEqual(spec.mesh.junction_spacing_um, 0.00002)
        self.assertTrue(any("Mapped doping.p" in item for item in repairs))
        self.assertTrue(any("Adjusted parameters.junction_um" in item for item in repairs))

    def test_llm_planner_success_returns_task_spec(self) -> None:
        response = json.dumps(
            {
                "task_spec": {
                    "title": "PN junction IV",
                    "sweep": {"start_v": 0.0, "stop_v": 5.0, "step_v": 5.0, "min_step_v": 1.25},
                    "execution": {"max_attempts": 3, "max_cycles": 3},
                }
            }
        )
        client = FakeClient(response)

        result = plan_task_text_with_llm(
            "PN junction IV from 0 to 5 V step 5 V",
            task_id="llm_task",
            execution_use_llm=False,
            client=client,
        )
        spec = task_spec_from_planning_result(result)

        self.assertEqual(result.status, PlannerStatus.COMPLETED)
        self.assertEqual(result.model, "fake-planner")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(spec.task_id, "llm_task")
        self.assertEqual(spec.sweep.stop_v, 5.0)

    def test_llm_planner_falls_back_on_invalid_json(self) -> None:
        client = FakeClient("not json")

        result = plan_task_text_with_llm(
            "PN IV to 0.5 V step 0.1 V",
            task_id="fallback_task",
            execution_use_llm=False,
            client=client,
        )
        spec = task_spec_from_planning_result(result)

        self.assertEqual(result.status, PlannerStatus.FALLBACK)
        self.assertTrue(result.fallback_used)
        self.assertEqual(spec.task_id, "fallback_task")
        self.assertEqual(spec.sweep.stop_v, 0.5)

    def test_legacy_task_planner_refuses_to_misroute_mosfet_to_pn_iv(self) -> None:
        client = FakeClient(Exception("should not be called"))

        result = plan_task_text_with_llm(
            "做 2D NMOS output characteristic，Vd 0 到 1.2V，并检查 kink",
            task_id="mosfet_task",
            execution_use_llm=False,
            client=client,
        )

        self.assertEqual(result.status, PlannerStatus.FAILED)
        self.assertIsNone(result.task_spec)
        self.assertIn("mission agent", result.validation_errors[0])
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
