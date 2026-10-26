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
# All grammars in the crate now: python+typescript (engine phase 1) and
# rust+go+java+ruby (this increment). The goldens pin them all.
PARITY_LANGUAGES = [
    lang
    for lang in ("python", "typescript", "rust", "go", "java", "ruby")
    if lang in PLUGIN_NATIVE_LANGUAGES
]


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
        # The golden stores the metadata view (25-item caps); the native
        # records carry FULL sets — capping is the metadata layer's job.
        "definitions": list(record.definitions)[:25],
        "uses": list(record.uses)[:25],
        "calls": list(record.calls)[:25],
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
    """Languages with goldens but NO native grammar must decline (None),
    never guess. As grammars land this set shrinks — currently only
    languages outside the plugin's CCG scope (e.g. sql, markdown)."""
    assert extract_plugin_statements([("sql", "/x", b"SELECT 1")], _valid_tables(), [[]]) is None


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


def _valid_tables():
    return {
        "classify": {"function_definition": "function_def"},
        "identifier_types": ["identifier"],
        "skip_descent": ["comment"],
        "scope_types": ["function_def"],
        "definition_types": ["function_def"],
        "keywords": ["def"],
        "function_def": "function_def",
        "class_def": "class_def",
        "call_type": "call",
        "def_fields": ["left"],
        "name_fields": ["name"],
        "param_fields": ["parameters"],
        "call_fields": ["function"],
    }


def test_tables_kwargs_normalizes_lists_to_sets():
    from victor.processing.native.plugin_ccg import _tables_kwargs

    kwargs = _tables_kwargs(_valid_tables())
    assert kwargs is not None
    assert kwargs["identifier_types"] == {"identifier"}
    assert kwargs["skip_descent"] == {"comment"}
    assert kwargs["classify"] == {"function_definition": "function_def"}


def test_tables_kwargs_rejects_incomplete_shape():
    from victor.processing.native.plugin_ccg import _tables_kwargs

    tables = _valid_tables()
    del tables["keywords"]
    assert _tables_kwargs(tables) is None


def test_stubbed_native_happy_path(monkeypatch):
    """With a stubbed extension present, the wrapper unpacks tables,
    forwards files and hints, and returns the FFI result verbatim."""
    import victor.processing.native.plugin_ccg as mod

    captured = {}

    class _Stub:
        @staticmethod
        def extract_plugin_statements_batch(files, **kwargs):
            captured["files"] = files
            captured["kwargs"] = kwargs
            return [["record"]]

    monkeypatch.setattr(mod, "_NATIVE_AVAILABLE", True)
    monkeypatch.setattr(mod, "_native", _Stub)
    result = mod.extract_plugin_statements([("python", "/x.py", b"pass")], _valid_tables(), [[]])
    assert result == [["record"]]
    assert captured["files"] == [("python", "/x.py", b"pass")]
    assert captured["kwargs"]["keywords"] == {"def"}
    assert captured["kwargs"]["hints_by_file"] == [[]]


def test_stubbed_native_ffi_failure_degrades_to_none(monkeypatch):
    import victor.processing.native.plugin_ccg as mod

    class _Boom:
        @staticmethod
        def extract_plugin_statements_batch(*args, **kwargs):
            raise RuntimeError("ffi blew up")

    monkeypatch.setattr(mod, "_NATIVE_AVAILABLE", True)
    monkeypatch.setattr(mod, "_native", _Boom)
    assert (
        mod.extract_plugin_statements([("python", "/x.py", b"pass")], _valid_tables(), [[]]) is None
    )


def test_consult_site_injects_the_extractor(tmp_path, monkeypatch):
    """The root consult site must hand the enhanced builder the native
    extractor (duck-typed, no cross-package import) before calling it."""
    import asyncio

    from victor.core.indexing.ccg_builder import CodeContextGraphBuilder

    injected = {}

    class _Enhanced:
        supports_language = staticmethod(lambda lang: lang == "python")

        async def build_ccg_for_file(self, path, language=None):
            injected["extractor"] = getattr(self, "_native_extractor", None)
            return [], []

    builder = CodeContextGraphBuilder.__new__(CodeContextGraphBuilder)
    builder._builder_resolution = "auto"
    builder._resolve_enhanced_builder = lambda lang: _Enhanced()
    builder._enhanced_builder_supports_language = lambda b, lang: True

    target = tmp_path / "x.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    asyncio.run(builder.build_ccg_for_file(target))

    from victor.processing.native.plugin_ccg import extract_plugin_statements

    assert injected["extractor"] is extract_plugin_statements
