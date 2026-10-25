"""Native plugin-shape statement extraction — golden differential parity.

The oracle is the golden corpus at
``verticals/victor-coding/tests/codebase/goldens/`` (the plugin walk's
pinned output, six languages, shape AND enrichment). This test reads the
samples, goldens, hint records and language-syntax tables as PURE DATA —
no root-to-vertical import — and requires the native walk to reproduce
every statement for the languages with a native grammar. The plugin's
Python walk stays the typed reference implementation and production
fallback; this slice is not yet wired into the consult path (phase 2
flips it behind the builder-resolution knob).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from victor.processing.native.plugin_ccg import (
    PLUGIN_NATIVE_LANGUAGES,
    extract_plugin_statements,
    plugin_statements_available,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
GOLDENS_DIR = REPO_ROOT / "verticals" / "victor-coding" / "tests" / "codebase" / "goldens"

FILE_BY_LANGUAGE = {
    "python": "sample.py",
    "rust": "sample.rs",
    "typescript": "sample.ts",
    "go": "sample.go",
    "java": "Sample.java",
    "ruby": "sample.rb",
}

# Grammars present in the Rust crate today; the others are golden-pinned
# Python-side until their crate lands (the builtin slice grew the same way).
PARITY_LANGUAGES = [lang for lang in ("python", "typescript") if lang in PLUGIN_NATIVE_LANGUAGES]


def _load_corpus():
    goldens = json.loads((GOLDENS_DIR / "ccg_plugin_goldens.json").read_text(encoding="utf-8"))
    tables = json.loads((GOLDENS_DIR / "plugin_tables.json").read_text(encoding="utf-8"))
    return goldens, tables


def _record_to_signature(record, scope_index):
    return {
        "statement_type": record.statement_type,
        "ast_kind": record.node_type,
        "name": record.name,
        "line": record.start_line,
        "end_line": record.end_line,
        "visibility": record.visibility,
        "scope": scope_index.get(record.scope_id, -1),
        "symbol_name": record.symbol_name,
        "symbol_type": record.symbol_type,
        "definitions": list(record.definitions),
        "uses": list(record.uses),
        "calls": list(record.calls),
        "content_preview": record.content_preview,
    }


@pytest.mark.parametrize("language", PARITY_LANGUAGES or ["python"])
def test_native_plugin_walk_matches_golden(language):
    if not plugin_statements_available(language):
        pytest.skip("native extension not built")
    goldens, tables = _load_corpus()
    golden = goldens[language]
    if not golden.get("supported"):
        pytest.skip(f"{language} unsupported by the plugin in this environment")

    sample = (GOLDENS_DIR / "samples" / FILE_BY_LANGUAGE[language]).read_bytes()
    hints = [tuple(h) for h in golden["hints"]]
    results = extract_plugin_statements(
        [(language, f"/golden/{FILE_BY_LANGUAGE[language]}", sample)],
        tables,
        [hints],
    )
    assert results is not None, "native path must serve a supported language"
    records = results[0]

    scope_index = {record.node_id: i for i, record in enumerate(records)}
    live = [_record_to_signature(record, scope_index) for record in records]
    expected = golden["statements"]

    if live != expected:
        # Pin the FIRST divergence, not just counts — porting bugs are
        # cheapest to find at the exact statement.
        for i, (got, want) in enumerate(zip(live, expected)):
            if got != want:
                pytest.fail(f"{language}: statement {i} diverged\n got: {got}\nwant: {want}")
        pytest.fail(
            f"{language}: statement count diverged: {len(live)} native vs "
            f"{len(expected)} golden"
        )


def test_languages_without_grammar_return_none():
    """rust/go/java/ruby have goldens but no native grammar yet: the
    wrapper must decline (None), never guess."""
    goldens, tables = _load_corpus()
    for language in ("rust", "go", "java", "ruby"):
        if not goldens.get(language, {}).get("supported"):
            continue
        sample = (GOLDENS_DIR / "samples" / FILE_BY_LANGUAGE.get(language, "")).read_bytes()
        hints = [tuple(h) for h in goldens[language]["hints"]]
        assert (
            extract_plugin_statements([(language, "/x", sample)], tables, [hints]) is None
        ), language


def test_malformed_tables_decline():
    assert extract_plugin_statements([("python", "/x", b"pass")], {"classify": {}}, [[]]) is None


@pytest.mark.skipif(
    not plugin_statements_available("python"),
    reason="empty-batch shape needs the extension (None means unavailable)",
)
def test_empty_batch_returns_empty():
    assert extract_plugin_statements([], {}, []) == []


def test_unavailable_extension_declines(monkeypatch):
    """Pure-Python installs: every call declines (None), never raises —
    the bounded-failure contract the future consult path relies on."""
    import victor.processing.native.plugin_ccg as mod

    monkeypatch.setattr(mod, "_NATIVE_AVAILABLE", False, raising=False)
    assert extract_plugin_statements([("python", "/x", b"pass")], {"classify": {}}, [[]]) is None
    assert extract_plugin_statements([], {}, []) is None


def test_tables_export_is_fresh():
    """The JSON snapshot must match the live victor-coding module — the
    export cannot drift from its single source of truth."""
    sys_path_append = str(REPO_ROOT / "verticals" / "victor-coding")
    import sys

    if sys_path_append not in sys.path:
        sys.path.insert(0, sys_path_append)
    try:
        import victor_coding.codebase.ccg_builder as m
    except ImportError:
        pytest.skip("victor-coding not installed")

    _, tables = _load_corpus()
    assert tables["classify"] == dict(sorted(m.NODE_TYPE_TO_STATEMENT_TYPE.items()))
    assert set(tables["identifier_types"]) == m.IDENTIFIER_NODE_TYPES
    assert set(tables["skip_descent"]) == m.SKIP_IDENTIFIER_DESCENT_TYPES
    assert set(tables["scope_types"]) == m.SCOPE_STATEMENT_TYPES
    assert set(tables["definition_types"]) == m.DEFINITION_STATEMENT_TYPES
    assert set(tables["keywords"]) == m.BROAD_KEYWORDS
