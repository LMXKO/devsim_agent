from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

from tcad_agent.process_control import run_cancellable


class ProcessControlTest(unittest.TestCase):
    def test_run_cancellable_terminates_when_cancel_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cancel_file = Path(tmp) / "cancel.requested"
            cancel_file.write_text("cancel", encoding="utf-8")
            started = time.monotonic()

            completed = run_cancellable(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                capture_output=True,
                text=True,
                timeout=60,
                cancel_file=cancel_file,
                poll_interval_seconds=0.05,
            )

        self.assertLess(time.monotonic() - started, 5)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("ACTSOFT_CANCELLED", completed.stderr)


if __name__ == "__main__":
    unittest.main()
