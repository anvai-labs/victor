"""The plugin-CCG golden corpus: the Option-B parity oracle.

Pins the enhanced builder's per-file statement output (shape + enrichment)
across six languages so the native statement-walk port cannot drift. When
the native path lands, its differential test reads the SAME samples/ and
goldens as pure data — semantics stay owned here, the engine stays generic.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
GOLDENS_DIR = Path(__file__).resolve().parent / "goldens"
sys.path.insert(0, str(REPO / "verticals" / "victor-coding"))

from victor_coding.codebase.ccg_builder import PluginBackedCCGBuilder  # noqa: E402

sys.path.insert(0, str(GOLDENS_DIR))
from golden_support import scope_index_of, statement_signature  # noqa: E402

GOLDENS = json.loads((GOLDENS_DIR / "ccg_plugin_goldens.json").read_text(encoding="utf-8"))
SAMPLES = GOLDENS_DIR / "samples"

FILE_BY_LANGUAGE = {
    "python": "sample.py",
    "rust": "sample.rs",
    "typescript": "sample.ts",
    "go": "sample.go",
    "java": "Sample.java",
    "ruby": "sample.rb",
}


def _build(builder, file_name, tmp_path):
    import shutil

    target = tmp_path / file_name
    shutil.copy(SAMPLES / file_name, target)
    return asyncio.run(builder.build_ccg_for_file(target))


LANGUAGES = [lang for lang, golden in GOLDENS.items() if golden.get("supported")]


@pytest.mark.parametrize("language", LANGUAGES or list(FILE_BY_LANGUAGE))
def test_plugin_ccg_matches_golden(language, tmp_path):
    golden = GOLDENS[language]
    if not golden.get("supported"):
        pytest.skip(f"{language} unsupported in this environment")
    builder = PluginBackedCCGBuilder()
    nodes, _edges = _build(builder, FILE_BY_LANGUAGE[language], tmp_path)
    scope_index = scope_index_of(nodes)
    live = [statement_signature(node, scope_index) for node in nodes]
    expected = golden["statements"]
    assert live == expected, (
        f"{language}: plugin CCG drifted from the golden corpus "
        f"({len(live)} vs {len(expected)} statements). If the change is "
        "intentional, regenerate via "
        "verticals/victor-coding/tests/codebase/goldens/generate_ccg_goldens.py"
    )
