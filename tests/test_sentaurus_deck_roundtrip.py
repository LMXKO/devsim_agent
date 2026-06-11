from __future__ import annotations

import unittest

from tcad_agent.sentaurus_deck import apply_sentaurus_semantic_patch_text


class SentaurusDeckRoundTripTest(unittest.TestCase):
    def test_semantic_patch_records_roundtrip_and_lineage(self) -> None:
        source = "\n".join(
            [
                "set LIFETIME_SCALE 1",
                "Physics {",
                "  Mobility( DopingDep )",
                "  Recombination( SRH )",
                "}",
                "",
            ]
        )

        after, record, ir = apply_sentaurus_semantic_patch_text(
            source,
            {"operation": "sentaurus_set_variable", "variable": "LIFETIME_SCALE", "value": "2"},
            source_path="device.cmd",
        )

        self.assertIn("set LIFETIME_SCALE 2", after)
        self.assertTrue(record["round_trip_verified"])
        self.assertEqual(record["patch_lineage"][0]["parameter"], "LIFETIME_SCALE")
        self.assertIn("Physics", record["patched_section_index"])
        self.assertGreaterEqual(len(ir.sections), 1)


if __name__ == "__main__":
    unittest.main()

