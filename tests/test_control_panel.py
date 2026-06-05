from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tcad_agent.control_panel import ControlPanelStatus, generate_control_panel
from tcad_agent.long_run_validation import LongRunValidationRequest, run_long_run_validation


class ControlPanelTest(unittest.TestCase):
    def test_generates_static_panel_from_runs_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_long_run_validation(LongRunValidationRequest(validation_id="panel_longrun", validation_root=root))

            result = generate_control_panel(root, output_dir=root / "panel")
            data = json.loads(Path(result.data_path).read_text(encoding="utf-8"))
            html = Path(result.html_path).read_text(encoding="utf-8")

        self.assertEqual(result.status, ControlPanelStatus.COMPLETED)
        self.assertGreaterEqual(data["counts"]["experiment_records"], 2)
        self.assertGreaterEqual(data["counts"]["benchmarks"], 2)
        self.assertIn("TCAD Agent Control Panel", html)
        self.assertIn("Run Queue", html)


if __name__ == "__main__":
    unittest.main()
