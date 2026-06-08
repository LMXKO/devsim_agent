from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tcad_agent.long_run_validation import (
    LongRunValidationRequest,
    LongRunValidationStatus,
    run_long_run_validation,
)


class LongRunValidationTest(unittest.TestCase):
    def test_runs_queue_daemon_benchmarks_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = run_long_run_validation(
                LongRunValidationRequest(validation_id="unit_longrun", validation_root=Path(tmp))
            )

        self.assertEqual(state.status, LongRunValidationStatus.COMPLETED)
        self.assertEqual(state.daemon_result["completed"], 5)
        self.assertEqual(len(state.benchmark_results), 5)
        self.assertGreaterEqual(state.index_summary["records_indexed"], 2)
        self.assertTrue(all(item["status"] == "completed" for item in state.queued_items))
        self.assertIn("longrun_power_mosfet_convergence", {item["queue_id"] for item in state.queued_items})
        self.assertIn("longrun_bjt_convergence", {item["queue_id"] for item in state.queued_items})


if __name__ == "__main__":
    unittest.main()
