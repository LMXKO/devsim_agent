from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tcad_agent.mesh_convergence import (
    MeshConvergenceRequest,
    MeshConvergenceStatus,
    run_mesh_convergence,
)
from tcad_agent.task_spec import TaskSpec, parse_task_text
from tcad_agent.tools.task_runner import TaskRunState, TaskRunStatus


class MeshRunner:
    def __init__(self, scale: float) -> None:
        self.scale = scale
        self.task_ids: list[str] = []

    def __call__(self, spec: TaskSpec, **kwargs: object) -> TaskRunState:
        self.task_ids.append(spec.task_id)
        root = Path(kwargs["task_root"])
        run_dir = root / spec.task_id
        execute = bool(kwargs.get("execute"))
        objective = 1.0 + spec.mesh.junction_spacing_um * self.scale
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


class MeshConvergenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def base_spec(self) -> TaskSpec:
        return parse_task_text("PN IV 0 to 0.2 V step 0.1 V", task_id="mesh_base")

    def request(self, *, execute: bool = True, tolerance: float = 0.01) -> MeshConvergenceRequest:
        return MeshConvergenceRequest(
            convergence_id="mesh_unit",
            convergence_root=self.root,
            axis_path="mesh.junction_spacing_um",
            values=[2e-5, 1e-5, 5e-6],
            relative_tolerance=tolerance,
            execute=execute,
        )

    def test_passes_when_finest_mesh_delta_is_small(self) -> None:
        state = run_mesh_convergence(
            self.base_spec(),
            self.request(),
            task_runner=MeshRunner(scale=1000.0),
        )

        self.assertEqual(state.status, MeshConvergenceStatus.COMPLETED)
        self.assertEqual(state.quality_report["status"], "passed")
        self.assertLess(state.quality_report["metrics"]["relative_delta"], 0.01)
        self.assertEqual(len(state.cases), 3)
        self.assertTrue((self.root / "mesh_unit" / "state.json").exists())

    def test_marks_large_mesh_delta_suspicious(self) -> None:
        state = run_mesh_convergence(
            self.base_spec(),
            self.request(tolerance=0.001),
            task_runner=MeshRunner(scale=10000.0),
        )

        self.assertEqual(state.status, MeshConvergenceStatus.COMPLETED)
        self.assertEqual(state.quality_report["status"], "suspicious")
        self.assertIn("mesh_not_converged", {issue["code"] for issue in state.quality_report["issues"]})

    def test_dry_run_plans_sweep(self) -> None:
        state = run_mesh_convergence(
            self.base_spec(),
            self.request(execute=False),
            task_runner=MeshRunner(scale=1000.0),
        )

        self.assertEqual(state.status, MeshConvergenceStatus.PLANNED)
        self.assertEqual(state.quality_report["status"], "planned")
        self.assertEqual(len(state.cases), 3)


if __name__ == "__main__":
    unittest.main()
