from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tcad_agent.parameter_sweep import (
    ParameterSweepRequest,
    SweepAxis,
    SweepStatus,
    parse_axis_spec,
    run_parameter_sweep,
)
from tcad_agent.task_spec import TaskSpec, parse_task_text
from tcad_agent.tools.task_runner import TaskRunState, TaskRunStatus


def fake_completed_task_runner(spec: TaskSpec, **kwargs: object) -> TaskRunState:
    value = spec.parameters.p_doping_cm3 / 1.0e18
    root = Path(kwargs["task_root"])
    run_dir = root / spec.task_id
    return TaskRunState(
        status=TaskRunStatus.COMPLETED,
        task_id=spec.task_id,
        task_path=str(run_dir / "task.json"),
        task_run_dir=str(run_dir),
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        execute=bool(kwargs.get("execute")),
        loop_request={},
        loop_state_path=str(run_dir / "loop_state.json"),
        loop_result={},
        final_state_path=str(run_dir / "final_state.json"),
        final_quality_report={
            "status": "passed",
            "metrics": {"final_total_current_a": value},
        },
    )


class ParameterSweepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def base_spec(self) -> TaskSpec:
        return parse_task_text("PN IV 0 to 0.2 V step 0.1 V", task_id="base")

    def test_parse_axis_spec(self) -> None:
        axis = parse_axis_spec("parameters.p_doping_cm3=1e16,1e17,1e18")

        self.assertEqual(axis.path, "parameters.p_doping_cm3")
        self.assertEqual(axis.values, [1e16, 1e17, 1e18])

    def test_dry_run_creates_planned_cases(self) -> None:
        state = run_parameter_sweep(
            self.base_spec(),
            ParameterSweepRequest(
                sweep_id="dry_sweep",
                sweep_root=self.root,
                axes=[SweepAxis(path="parameters.p_doping_cm3", values=[1e16, 1e17])],
                execute=False,
            ),
        )

        self.assertEqual(state.status, SweepStatus.PLANNED)
        self.assertEqual(len(state.cases), 2)
        self.assertTrue((self.root / "dry_sweep" / "base_task.json").exists())
        self.assertTrue((self.root / "dry_sweep" / "summary.csv").exists())

    def test_execute_selects_best_minimum_objective(self) -> None:
        state = run_parameter_sweep(
            self.base_spec(),
            ParameterSweepRequest(
                sweep_id="exec_sweep",
                sweep_root=self.root,
                axes=[SweepAxis(path="parameters.p_doping_cm3", values=[1e18, 1e16, 1e17])],
                execute=True,
            ),
            task_runner=fake_completed_task_runner,
        )

        self.assertEqual(state.status, SweepStatus.COMPLETED)
        self.assertEqual(len(state.cases), 3)
        self.assertEqual(state.best_case_index, 2)
        self.assertEqual(state.best_case["values"]["parameters.p_doping_cm3"], 1e16)
        self.assertEqual(state.best_case["objective_value"], 0.01)


if __name__ == "__main__":
    unittest.main()
