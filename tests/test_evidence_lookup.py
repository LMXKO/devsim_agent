from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tcad_agent.evidence_lookup import PublicEvidenceLookupRequest, run_public_evidence_lookup


class PublicEvidenceLookupTest(unittest.TestCase):
    def test_live_lookup_verifies_public_methodology_with_fake_fetcher(self) -> None:
        def fake_fetcher(url: str, timeout: float) -> tuple[int, str]:
            self.assertGreater(timeout, 0)
            return (
                200,
                """
                <html><title>Sentaurus Device Training</title>
                <body>
                File Electrode Physics Math Solve Plot sections are used.
                Quasistationary sweeps use InitialStep MaxStep MinStep and Goal.
                Plot output includes ElectricField and ImpactIonization.
                </body></html>
                """,
            )

        result = run_public_evidence_lookup(
            PublicEvidenceLookupRequest(
                goal_text="Sentaurus LDMOS BV field peak quasistationary sweep",
                simulator="sentaurus",
                source_ids=["sentaurus_quasistationary_training"],
                live=True,
            ),
            fetcher=fake_fetcher,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.verified_source_ids, ["sentaurus_quasistationary_training"])
        finding = result.findings[0]
        self.assertEqual(finding.status, "verified")
        self.assertIn("sentaurus_quasistationary_step_control", finding.methodology_claims)
        self.assertIn("sentaurus_plot_field_outputs", finding.methodology_claims)
        self.assertTrue(result.evidence_gate["passed"])

    def test_registry_only_lookup_is_deterministic_and_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "lookup.json"
            result = run_public_evidence_lookup(
                PublicEvidenceLookupRequest(
                    goal_text="DEVSIM MOS capacitor C-V",
                    simulator="devsim",
                    template_ids=["mos_capacitor_cv"],
                    live=False,
                    output_path=output,
                )
            )

            self.assertEqual(result.status, "completed")
            self.assertTrue(output.exists())
            self.assertTrue(result.findings)
            self.assertFalse(result.findings[0].live_checked)


if __name__ == "__main__":
    unittest.main()
