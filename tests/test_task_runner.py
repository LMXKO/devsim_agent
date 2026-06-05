from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tcad_agent.task_planner import PlannerStatus, TaskPlanningResult
from tcad_agent.task_spec import parse_task_text
from tcad_agent.tools.autonomous_loop import AutonomousLoopRequest
from tcad_agent.tools.task_runner import TaskRunStatus, run_task


class FakeLoopRunner:
    def __init__(self, status: str = "completed") -> None:
        self.status = status
        self.calls: list[AutonomousLoopRequest] = []

    def __call__(self, request: AutonomousLoopRequest) -> dict[str, object]:
        self.calls.append(request)
        return {
            "status": self.status,
            "loop_id": request.loop_id,
            "final_state_path": "/tmp/final_state.json" if self.status == "completed" else None,
            "final_quality_report": {"status": "passed"} if self.status == "completed" else None,
            "failure_reason": None if self.status == "completed" else "loop failed",
        }


class TaskRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def spec(self):
        return parse_task_text(
            "PN junction IV from 0 to 5 V step 5 V min_step 1.25 V",
            task_id="task_unit",
            use_llm=False,
        )

    def test_dry_run_writes_task_and_planned_state(self) -> None:
        state = run_task(
            self.spec(),
            task_root=self.root / "tasks",
            loop_root=self.root / "loops",
            run_root=self.root / "agent_tools",
            execute=False,
        )

        self.assertEqual(state.status, TaskRunStatus.PLANNED)
        self.assertTrue((self.root / "tasks" / "task_unit" / "task.json").exists())
        self.assertTrue((self.root / "tasks" / "task_unit" / "task_run_state.json").exists())
        self.assertEqual(state.loop_request["stop"], 5.0)
        self.assertEqual(state.loop_state_path, str(self.root / "loops" / "task_unit" / "loop_state.json"))

    def test_execute_runs_loop_and_marks_completed(self) -> None:
        runner = FakeLoopRunner()

        state = run_task(
            self.spec(),
            task_root=self.root / "tasks",
            loop_root=self.root / "loops",
            run_root=self.root / "agent_tools",
            execute=True,
            loop_runner=runner,
        )

        self.assertEqual(state.status, TaskRunStatus.COMPLETED)
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(runner.calls[0].loop_id, "task_unit")
        self.assertEqual(state.final_quality_report, {"status": "passed"})

    def test_execute_propagates_loop_failure(self) -> None:
        runner = FakeLoopRunner(status="failed")

        state = run_task(
            self.spec(),
            task_root=self.root / "tasks",
            loop_root=self.root / "loops",
            run_root=self.root / "agent_tools",
            execute=True,
            loop_runner=runner,
        )

        self.assertEqual(state.status, TaskRunStatus.FAILED)
        self.assertEqual(state.failure_reason, "loop failed")

    def test_writes_planning_result_when_available(self) -> None:
        spec = self.spec()
        planning_result = TaskPlanningResult(
            status=PlannerStatus.COMPLETED,
            input_text="PN IV",
            task_id=spec.task_id,
            task_spec=spec.model_dump(mode="json"),
        )

        state = run_task(
            spec,
            task_root=self.root / "tasks",
            loop_root=self.root / "loops",
            run_root=self.root / "agent_tools",
            execute=False,
            planner="llm",
            planning_result=planning_result,
        )

        self.assertEqual(state.planner, "llm")
        self.assertEqual(state.planner_status, PlannerStatus.COMPLETED)
        self.assertTrue((self.root / "tasks" / "task_unit" / "task_plan_result.json").exists())


if __name__ == "__main__":
    unittest.main()
