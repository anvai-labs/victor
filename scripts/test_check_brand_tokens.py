#!/usr/bin/env python3
"""Regression tests for check_brand_tokens.py (shared verbatim across Anvai docs sites).

Each case mutates a temporary copy of this repo's own mkdocs.yml and shared
stylesheets, so the tests exercise the real configuration. Run directly:
python scripts/test_check_brand_tokens.py
"""

import importlib.util
import re
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_brand_tokens", ROOT / "scripts" / "check_brand_tokens.py"
)
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)


class BrandCheckerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        match = re.search(r"^docs_dir:\s*(\S+)", mkdocs, re.MULTILINE)
        docs_dir = match.group(1) if match else "docs"
        shutil.copytree(ROOT / docs_dir / "stylesheets", self.tmp / docs_dir / "stylesheets")
        self.mkdocs = self.tmp / "mkdocs.yml"
        self.mkdocs.write_text(mkdocs, encoding="utf-8")
        self.stylesheets = self.tmp / docs_dir / "stylesheets"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def edit(self, old, new, count=1):
        text = self.mkdocs.read_text(encoding="utf-8")
        self.assertIn(old, text)
        self.mkdocs.write_text(text.replace(old, new, count), encoding="utf-8")

    def failures(self):
        return checker.check(self.tmp)

    def test_repo_configuration_passes(self):
        self.assertEqual(self.failures(), [])

    def test_dropping_tokens_css_fails(self):
        self.edit("  - stylesheets/tokens.css\n", "")
        self.assertTrue(any("extra_css" in f for f in self.failures()))

    def test_commented_entry_fails(self):
        self.edit("  - stylesheets/tokens.css\n", "  # - stylesheets/tokens.css\n")
        self.assertTrue(any("extra_css" in f for f in self.failures()))

    def test_reversed_order_fails(self):
        self.edit(
            "  - stylesheets/tokens.css\n  - stylesheets/brand.css\n",
            "  - stylesheets/brand.css\n  - stylesheets/tokens.css\n",
        )
        self.assertTrue(any("extra_css" in f for f in self.failures()))

    def test_later_duplicate_key_wins_like_yaml(self):
        text = self.mkdocs.read_text(encoding="utf-8")
        self.mkdocs.write_text(text + "\nextra_css:\n  - other.css\n", encoding="utf-8")
        self.assertTrue(any("extra_css" in f for f in self.failures()))

    def test_flow_style_list_passes(self):
        self.edit(
            "extra_css:\n  - stylesheets/tokens.css\n  - stylesheets/brand.css\n",
            "extra_css: [stylesheets/tokens.css, stylesheets/brand.css]\n",
        )
        self.assertEqual(self.failures(), [])

    def test_material_font_loading_fails(self):
        self.edit("  font: false\n", "  font:\n    text: Roboto\n")
        self.assertTrue(any("theme.font" in f for f in self.failures()))

    def test_each_palette_entry_needs_custom_primary_and_accent(self):
        original = self.mkdocs.read_text(encoding="utf-8")
        for key, value in (("primary", "indigo"), ("accent", "amber")):
            needle = f"{key}: custom"
            for index in range(original.count(needle)):
                with self.subTest(key=key, entry=index):
                    at = -1
                    for _ in range(index + 1):
                        at = original.index(needle, at + 1)
                    mutated = original[:at] + f"{key}: {value}" + original[at + len(needle) :]
                    self.mkdocs.write_text(mutated, encoding="utf-8")
                    self.assertTrue(any(needle in f for f in self.failures()))
        self.mkdocs.write_text(original, encoding="utf-8")
        self.assertEqual(self.failures(), [])

    def test_edited_shared_file_fails(self):
        with (self.stylesheets / "brand.css").open("a", encoding="utf-8") as handle:
            handle.write("/* local edit */\n")
        self.assertTrue(any("brand.css drifted" in f for f in self.failures()))

    def test_missing_license_fails(self):
        (self.stylesheets / "fonts" / "Inter-OFL.txt").unlink()
        self.assertTrue(any("Inter-OFL.txt is missing" in f for f in self.failures()))


if __name__ == "__main__":
    unittest.main()
