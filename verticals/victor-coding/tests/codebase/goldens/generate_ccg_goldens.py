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
    print(f"wrote {GOLDENS / 'ccg_plugin_goldens.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
