"""Regenerate the plugin-CCG golden corpus (the Option-B parity oracle).

The golden corpus pins the enhanced (victor-coding) CCG builder's per-file
output — statement shape AND enrichment — so the native statement-walk port
(``extract_plugin_statements_batch`` in the root engine, parameterized by
these same semantics as data) has an in-repo oracle that cannot drift.

Run from the repo root after an INTENTIONAL semantic change::

    python verticals/victor-coding/tests/codebase/goldens/generate_ccg_goldens.py

The companion test (``test_ccg_goldens.py``) fails whenever live output
diverges; regenerating is the acceptance step for deliberate changes.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
GOLDENS = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "verticals" / "victor-coding"))

from victor_coding.codebase.ccg_builder import PluginBackedCCGBuilder  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_support import scope_index_of, statement_signature  # noqa: E402

# The corpus inputs are the hand-authored files under samples/ (dense by
# design: defs, nesting, conditions, loops, try/except, assignments, calls,
# returns/yields, visibility markers, underscore-private names, same-line
# statement chains). This generator reads them, runs the live plugin walk,
# and records the output as the golden.
SAMPLES: dict[str, str] = {
    "python": "sample.py",
    "rust": "sample.rs",
    "typescript": "sample.ts",
    "go": "sample.go",
    "java": "Sample.java",
    "ruby": "sample.rb",
}


def _hint_records(language: str, file_name: str, builder) -> list[list]:
    """The raw symbol-hint set for a sample, as data.

    Hints ride the native batch as inputs (phase 1 computes them Python-side,
    ~2.7 ms); the differential needs the exact (start_byte, end_byte,
    node_type) -> (name, symbol_type) mapping the live walk consumed.
    """

    plugin = builder._get_plugin(language)  # noqa: SLF001 - same package family
    parser_language = builder._parser_language_for_plugin(plugin, language, Path(file_name))
    parser = builder._get_parser(parser_language)
    source = (GOLDENS / "samples" / file_name).read_bytes()
    tree = parser.parse(source)
    hints = builder._extract_symbol_hints(tree, parser, plugin)  # noqa: SLF001
    return [
        [start_byte, end_byte, node_type, hint.name, hint.symbol_type]
        for (start_byte, end_byte, node_type), hint in sorted(hints.items(), key=repr)
    ]


def build_golden(language: str, file_name: str) -> dict:
    builder = PluginBackedCCGBuilder()
    if not builder.supports_language(language):
        return {"language": language, "supported": False}
    source = (GOLDENS / "samples" / file_name).read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / file_name
        target.write_text(source, encoding="utf-8")
        nodes, _edges = asyncio.run(builder.build_ccg_for_file(target))
    # scope ids embed the file path (storage identity); index-normalize so
    # goldens are path-independent. Walk order defines the index.
    scope_index = scope_index_of(nodes)
    return {
        "language": language,
        "supported": True,
        "statements": [statement_signature(node, scope_index) for node in nodes],
        "hints": _hint_records(language, file_name, builder),
    }


def export_tables() -> dict:
    """The language-syntax tables as data — the native engine's parameters."""
    from victor_coding.codebase.ccg_builder import native_tables

    tables = native_tables()
    return {
        key: (sorted(value) if isinstance(value, set) else value) for key, value in tables.items()
    }


def main() -> int:
    goldens = {}
    for language, file_name in SAMPLES.items():
        golden = build_golden(language, file_name)
        goldens[language] = golden
        count = len(golden.get("statements", []))
        print(f"{language}: {count if golden['supported'] else 'unsupported'} statements")
    (GOLDENS / "ccg_plugin_goldens.json").write_text(
        json.dumps(goldens, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    (GOLDENS / "plugin_tables.json").write_text(
        json.dumps(export_tables(), indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {GOLDENS / 'ccg_plugin_goldens.json'} + plugin_tables.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
