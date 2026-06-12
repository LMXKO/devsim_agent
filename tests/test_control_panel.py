from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tcad_agent.control_panel import collect_control_panel_data
from tcad_agent.long_run_validation import LongRunValidationRequest, run_long_run_validation


class ControlPanelTest(unittest.TestCase):
    def test_collects_dashboard_data_from_runs_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_long_run_validation(LongRunValidationRequest(validation_id="panel_longrun", validation_root=root))
            data = collect_control_panel_data(root)

        self.assertGreaterEqual(data["counts"]["experiment_records"], 2)
        self.assertGreaterEqual(data["counts"]["benchmarks"], 2)
        self.assertEqual(data["counts"]["validations"], 1)
        self.assertIn("llm_status", data)


if __name__ == "__main__":
    unittest.main()
