from __future__ import annotations

import unittest

from tcad_agent.examples.pn_junction.run import voltage_targets


class PNJunctionRunnerTest(unittest.TestCase):
    def test_voltage_targets_increase(self) -> None:
        self.assertEqual(voltage_targets(0.0, 1.0, 0.5), [0.0, 0.5, 1.0])

    def test_voltage_targets_decrease_for_reverse_sweep(self) -> None:
        self.assertEqual(voltage_targets(0.0, -1.0, 0.5), [0.0, -0.5, -1.0])


if __name__ == "__main__":
    unittest.main()
