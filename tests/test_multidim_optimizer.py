from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from tcad_agent.multidim_optimizer import (
    AxisScale,
    MultiDimAxis,
    MultiDimOptimizationRequest,
    MultiDimOptimizationStatus,
    run_multidim_optimization,
)
from tcad_agent.task_spec import TaskSpec, parse_task_text
from tcad_agent.tools.task_runner import TaskRunState, TaskRunStatus


class MultiDimCountingRunner:
    def __init__(self) -> None:
        self.task_ids: list[str] = []

    def __call__(self, spec: TaskSpec, **kwargs: object) -> TaskRunState:
        self.task_ids.append(spec.task_id)
        root = Path(kwargs["task_root"])
        run_dir = root / spec.task_id
        execute = bool(kwargs.get("execute"))
        doping_score = abs(math.log10(spec.parameters.p_doping_cm3) - 17.0)
        junction_score = abs(spec.parameters.junction_um - 0.05) * 100.0
        objective = doping_score + junction_score
        return TaskRunState(
            status=TaskRunStatus.COMPLETED if execute else TaskRunStatus.PLANNED,
            task_id=spec.task_id,
            task_path=str(run_dir / "task.json"),
            task_run_dir=str(run_dir),
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            execute=execute,
            execution_request={},
            execution_state_path=str(run_dir / "state.json"),
            execution_result={},
            final_state_path=str(run_dir / "final_state.json") if execute else None,
            final_quality_report={
                "status": "passed",
                "metrics": {"final_total_current_a": objective},
            }
            if execute
            else None,
        )


class MultiDimOptimizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def base_spec(self) -> TaskSpec:
        return parse_task_text("PN IV 0 to 0.2 V step 0.1 V", task_id="base")

    def request(self, *, max_rounds: int = 2, execute: bool = True) -> MultiDimOptimizationRequest:
        return MultiDimOptimizationRequest(
            optimize_id="multi_unit",
            optimize_root=self.root,
            axes=[
                MultiDimAxis(
                    path="parameters.p_doping_cm3",
                    min_value=1e16,
                    max_value=1e18,
                    scale=AxisScale.LOG,
                    initial_points=3,
                    max_new_points_per_round=2,
                ),
                MultiDimAxis(
                    path="parameters.junction_um",
                    min_value=0.04,
                    max_value=0.06,
                    scale=AxisScale.LINEAR,
                    initial_points=3,
                    max_new_points_per_round=2,
                ),
            ],
            max_rounds=max_rounds,
            max_cases=20,
            max_cases_per_round=9,
            execute=execute,
        )

    def test_execute_refines_two_axes_around_best(self) -> None:
        runner = MultiDimCountingRunner()

        state = run_multidim_optimization(self.base_spec(), self.request(), task_runner=runner)

        self.assertEqual(state.status, MultiDimOptimizationStatus.COMPLETED)
        self.assertEqual(len(state.rounds), 2)
        self.assertEqual(len(state.observations), 17)
        self.assertEqual(len(runner.task_ids), 17)
        self.assertEqual(state.best_observation["objective_value"], 0.0)
        self.assertAlmostEqual(state.best_observation["values"]["parameters.p_doping_cm3"], 1e17)
        self.assertAlmostEqual(state.best_observation["values"]["parameters.junction_um"], 0.05)
        second_round_values = state.rounds[1].candidate_values
        self.assertIn(
            {
                "parameters.p_doping_cm3": math.sqrt(1e16 * 1e17),
                "parameters.junction_um": 0.05,
            },
            second_round_values,
        )
        self.assertTrue((self.root / "multi_unit" / "optimization_state.json").exists())
        self.assertTrue((self.root / "multi_unit" / "rounds" / "multi_unit_round_001" / "summary.csv").exists())

    def test_dry_run_plans_first_round_only(self) -> None:
        runner = MultiDimCountingRunner()

        state = run_multidim_optimization(
            self.base_spec(),
            self.request(max_rounds=3, execute=False),
            task_runner=runner,
        )

        self.assertEqual(state.status, MultiDimOptimizationStatus.PLANNED)
        self.assertEqual(len(state.rounds), 1)
        self.assertEqual(len(state.observations), 9)
        self.assertIsNone(state.best_observation)
        self.assertEqual(state.next_action, "execute planned multi-dimensional optimization round")

    def test_resume_adds_only_missing_rounds(self) -> None:
        runner = MultiDimCountingRunner()

        first = run_multidim_optimization(
            self.base_spec(),
            self.request(max_rounds=1),
            task_runner=runner,
        )
        self.assertEqual(first.status, MultiDimOptimizationStatus.COMPLETED)
        self.assertEqual(len(runner.task_ids), 9)

        second = run_multidim_optimization(
            self.base_spec(),
            self.request(max_rounds=2),
            task_runner=runner,
        )

        self.assertEqual(second.status, MultiDimOptimizationStatus.COMPLETED)
        self.assertEqual(len(second.rounds), 2)
        self.assertEqual(len(second.observations), 17)
        self.assertEqual(len(runner.task_ids), 17)


if __name__ == "__main__":
    unittest.main()
