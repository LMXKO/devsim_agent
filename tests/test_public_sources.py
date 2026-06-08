from __future__ import annotations

import unittest

from tcad_agent.public_sources import (
    get_public_tcad_category,
    get_public_tcad_source,
    list_public_tcad_categories,
    public_categories_for_template,
    public_sources_for_template,
    validate_public_tcad_registry,
)


class PublicTCADSourcesTest(unittest.TestCase):
    def test_registry_has_the_seven_public_template_categories(self) -> None:
        categories = list_public_tcad_categories()
        ids = {category["category_id"] for category in categories}

        self.assertEqual(
            ids,
            {
                "mosfet_id_dibl",
                "diode_sbd_breakdown",
                "ldmos_igbt_power",
                "gan_algan_hemt",
                "bjt_gummel_output",
                "finfet_soi_variability",
                "moscap_capacitance",
            },
        )
        self.assertEqual(validate_public_tcad_registry(), [])

    def test_category_sources_resolve_to_public_references(self) -> None:
        category = get_public_tcad_category("mosfet_id_dibl")
        self.assertIsNotNone(category)

        source_ids = set(category.source_ids)
        self.assertIn("devsim_3dmos", source_ids)
        self.assertIn("sentaurus_quasistationary_training", source_ids)
        for source_id in source_ids:
            source = get_public_tcad_source(source_id)
            self.assertIsNotNone(source)
            self.assertTrue(source.url.startswith("https://"))

    def test_template_lookup_returns_sources_and_categories(self) -> None:
        categories = public_categories_for_template("gan_hemt_id_bv")
        sources = public_sources_for_template("gan_hemt_id_bv")

        self.assertEqual([category["category_id"] for category in categories], ["gan_algan_hemt"])
        self.assertGreaterEqual(len(sources), 3)
        self.assertIn("genius_tcad_open", {source["source_id"] for source in sources})


if __name__ == "__main__":
    unittest.main()
