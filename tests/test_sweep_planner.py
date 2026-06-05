from __future__ import annotations

import json
import unittest

from tcad_agent.llm import LLMConfig
from tcad_agent.sweep_planner import (
    SweepPlannerStatus,
    build_sweep_request_from_planner_json,
    deterministic_sweep_plan,
    plan_sweep_text_with_llm,
    sweep_plan_from_result,
)


class FakeClient:
    config = LLMConfig(model="fake-sweep-planner")

    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, str | float]] = []

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class SweepPlannerTest(unittest.TestCase):
    def test_deterministic_sweep_plan_infers_p_doping_axis(self) -> None:
        base_spec, request, warnings = deterministic_sweep_plan(
            "PN IV 从 0V 扫到 0.2V 步长 0.1V，扫描 P 区掺杂 1e16 到 1e18，找电流最小 max_attempts 3",
            sweep_id="det_sweep",
            execution_use_llm=False,
        )

        self.assertEqual(base_spec.sweep.stop_v, 0.2)
        self.assertEqual(request.sweep_id, "det_sweep")
        self.assertEqual(request.axes[0].path, "parameters.p_doping_cm3")
        self.assertEqual(request.axes[0].values, [1e16, 1e17, 1e18])
        self.assertEqual(request.objective.direction, "minimize")
        self.assertEqual(warnings, [])

    def test_build_sweep_request_repairs_axis_aliases(self) -> None:
        request, repairs = build_sweep_request_from_planner_json(
            {
                "sweep_request": {
                    "axes": [{"parameter": "p_doping", "start": 1e16, "stop": 1e18, "count": 3}],
                    "objective": {"metric": "max_abs_current", "direction": "min"},
                }
            },
            text="扫描 P 掺杂",
            sweep_id="alias_sweep",
        )

        self.assertEqual(request.axes[0].path, "parameters.p_doping_cm3")
        self.assertEqual(request.axes[0].values, [1e16, 1e17, 1e18])
        self.assertEqual(request.objective.metric_path, "final_quality_report.metrics.max_abs_current_a")
        self.assertTrue(any("Mapped axis path" in item for item in repairs))
        self.assertTrue(any("Expanded axis start/stop" in item for item in repairs))

    def test_llm_sweep_planner_success(self) -> None:
        response = json.dumps(
            {
                "base_task_spec": {
                    "task_id": "model_base",
                    "title": "PN IV",
                    "sweep": {"start_v": 0.0, "stop_v": 0.2, "step_v": 0.1, "min_step_v": 0.025},
                },
                "sweep_request": {
                    "axes": [{"path": "parameters.p_doping_cm3", "values": [1e16, 1e17]}],
                    "objective": {
                        "metric_path": "final_quality_report.metrics.final_total_current_a",
                        "direction": "minimize",
                        "absolute": True,
                    },
                },
            }
        )
        client = FakeClient(response)

        result = plan_sweep_text_with_llm(
            "扫描 P 掺杂",
            sweep_id="llm_sweep",
            execution_use_llm=False,
            client=client,
        )
        base_spec, request = sweep_plan_from_result(result)

        self.assertEqual(result.status, SweepPlannerStatus.COMPLETED)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(base_spec.task_id, "llm_sweep_base")
        self.assertEqual(request.sweep_id, "llm_sweep")
        self.assertEqual(request.axes[0].values, [1e16, 1e17])

    def test_llm_sweep_planner_falls_back_on_invalid_json(self) -> None:
        client = FakeClient("not json")

        result = plan_sweep_text_with_llm(
            "扫描 P 区掺杂 1e16 到 1e18，找电流最小",
            sweep_id="fallback_sweep",
            execution_use_llm=False,
            client=client,
        )
        _, request = sweep_plan_from_result(result)

        self.assertEqual(result.status, SweepPlannerStatus.FALLBACK)
        self.assertTrue(result.fallback_used)
        self.assertEqual(request.axes[0].values, [1e16, 1e17, 1e18])


if __name__ == "__main__":
    unittest.main()
