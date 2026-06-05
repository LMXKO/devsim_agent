from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from tcad_agent.adaptive_optimizer import (
    AdaptiveAxis,
    AdaptiveOptimizationRequest,
    AxisScale,
    OptimizationStatus,
    run_adaptive_optimization,
)
from tcad_agent.task_spec import TaskSpec, parse_task_text
from tcad_agent.tools.task_runner import TaskRunState, TaskRunStatus


class CountingRunner:
    def __init__(self) -> None:
        self.task_ids: list[str] = []

    def __call__(self, spec: TaskSpec, **kwargs: object) -> TaskRunState:
        self.task_ids.append(spec.task_id)
        root = Path(kwargs["task_root"])
        run_dir = root / spec.task_id
        execute = bool(kwargs.get("execute"))
        value = spec.parameters.p_doping_cm3
        objective = abs(math.log10(value) - 17.0)
        return TaskRunState(
            status=TaskRunStatus.COMPLETED if execute else TaskRunStatus.PLANNED,
            task_id=spec.task_id,
            task_path=str(run_dir / "task.json"),
            task_run_dir=str(run_dir),
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            execute=execute,
            loop_request={},
            loop_state_path=str(run_dir / "loop_state.json"),
            loop_result={},
            final_state_path=str(run_dir / "final_state.json") if execute else None,
            final_quality_report={
                "status": "passed",
                "metrics": {"final_total_current_a": objective},
            }
            if execute
            else None,
        )


class AdaptiveOptimizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def base_spec(self) -> TaskSpec:
        return parse_task_text("PN IV 0 to 0.2 V step 0.1 V", task_id="base")

    def request(self, *, max_rounds: int = 2, execute: bool = True) -> AdaptiveOptimizationRequest:
        return AdaptiveOptimizationRequest(
            optimize_id="opt_unit",
            optimize_root=self.root,
            axis=AdaptiveAxis(
                path="parameters.p_doping_cm3",
                min_value=1e16,
                max_value=1e18,
                scale=AxisScale.LOG,
                initial_points=3,
                max_new_points_per_round=2,
            ),
            max_rounds=max_rounds,
            execute=execute,
        )

    def test_execute_refines_around_best_log_point(self) -> None:
        runner = CountingRunner()

        state = run_adaptive_optimization(self.base_spec(), self.request(), task_runner=runner)

        self.assertEqual(state.status, OptimizationStatus.COMPLETED)
        self.assertEqual(len(state.rounds), 2)
        self.assertEqual(len(state.observations), 5)
        self.assertEqual(len(runner.task_ids), 5)
        self.assertAlmostEqual(state.best_observation["value"], 1e17)
        self.assertEqual(state.best_observation["objective_value"], 0.0)
        self.assertAlmostEqual(state.rounds[1].values[0], math.sqrt(1e16 * 1e17))
        self.assertAlmostEqual(state.rounds[1].values[1], math.sqrt(1e17 * 1e18))
        self.assertTrue((self.root / "opt_unit" / "optimization_state.json").exists())

    def test_dry_run_plans_first_round_only(self) -> None:
        runner = CountingRunner()

        state = run_adaptive_optimization(
            self.base_spec(),
            self.request(max_rounds=3, execute=False),
            task_runner=runner,
        )

        self.assertEqual(state.status, OptimizationStatus.PLANNED)
        self.assertEqual(len(state.rounds), 1)
        self.assertEqual(len(state.observations), 3)
        self.assertIsNone(state.best_observation)
        self.assertEqual(state.next_action, "execute planned optimization round")

    def test_resume_adds_only_missing_rounds(self) -> None:
        runner = CountingRunner()

        first = run_adaptive_optimization(
            self.base_spec(),
            self.request(max_rounds=1),
            task_runner=runner,
        )
        self.assertEqual(first.status, OptimizationStatus.COMPLETED)
        self.assertEqual(len(runner.task_ids), 3)

        second = run_adaptive_optimization(
            self.base_spec(),
            self.request(max_rounds=2),
            task_runner=runner,
        )

        self.assertEqual(second.status, OptimizationStatus.COMPLETED)
        self.assertEqual(len(second.rounds), 2)
        self.assertEqual(len(second.observations), 5)
        self.assertEqual(len(runner.task_ids), 5)


if __name__ == "__main__":
    unittest.main()
