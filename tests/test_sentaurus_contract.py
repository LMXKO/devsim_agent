from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tcad_agent.sentaurus_contract import (
    default_fixture_root,
    discover_contract_projects,
    validate_fixture_corpus,
    validate_sentaurus_contract,
)


class SentaurusContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_fixture_corpus_validates_offline_deck_contracts(self) -> None:
        projects = discover_contract_projects(default_fixture_root())
        self.assertGreaterEqual(len(projects), 3)

        results = validate_fixture_corpus(default_fixture_root(), output_root=self.root / "contract_runs")

        self.assertTrue(results)
        self.assertTrue(all(result.status == "passed" for result in results))
        self.assertIn("power_diode_bv", {result.case_id for result in results})
        self.assertIn("mosfet_idvg", {result.case_id for result in results})
        self.assertIn("mixed_mode_transient", {result.case_id for result in results})

    def test_fake_backend_e2e_validates_runner_interface_without_real_sentaurus(self) -> None:
        project = default_fixture_root() / "power_diode_bv"

        result = validate_sentaurus_contract(
            project,
            run_fake_e2e=True,
            output_root=self.root / "fake_e2e",
            report_path=self.root / "contract_report.json",
        )

        self.assertEqual(result.status, "passed")
        self.assertTrue(result.sentaurus_state_path)
        self.assertTrue(Path(result.sentaurus_state_path).exists())
        self.assertTrue(Path(result.report_path).exists())
        codes = {check.code for check in result.checks}
        self.assertIn("sentaurus_contract_fake_backend_completed", codes)
        self.assertIn("sentaurus_contract_curve_columns_present", codes)


if __name__ == "__main__":
    unittest.main()

