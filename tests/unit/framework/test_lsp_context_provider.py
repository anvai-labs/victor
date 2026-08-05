# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for the FEP-0019 Phase 3 LSP context provider.

Covers ``build_context`` (victor/framework/lsp_context.py): rendering symbols +
errors, throttling by signature, degradation, budget truncation, caps, and
severity filtering.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from victor.framework.lsp_context import build_context


def _diag(severity: int, message: str, line: int = 10):
    return SimpleNamespace(
        severity=severity,
        message=message,
        range=SimpleNamespace(start=SimpleNamespace(line=line)),
    )


def _sym(name: str, kind: int, detail=None, children=None):
    return SimpleNamespace(name=name, kind=kind, detail=detail, children=children or [])


class _MockLSP:
    """Duck-typed LSP: async get_document_symbols + sync get_diagnostics."""

    def __init__(self, symbols=None, diagnostics=None):
        self._symbols = symbols or {}
        self._diags = diagnostics or {}

    async def get_document_symbols(self, file_path: str):
        return self._symbols.get(file_path, [])

    def get_diagnostics(self, file_path: str):
        return self._diags.get(file_path, [])


@pytest.fixture
def _patch_modified():
    """Patch workspace_files_modified to return a controlled list."""

    def factory(files):
        async def fake(workspace, timeout=15):
            return list(files)

        return patch("victor.framework.lsp_context.workspace_files_modified", fake)

    return factory


@pytest.mark.asyncio
async def test_no_lsp_returns_none(_patch_modified):
    with _patch_modified(["a.py"]):
        block, sig = await build_context(None, Path("/tmp"))
    assert block is None and sig is None


@pytest.mark.asyncio
async def test_no_workspace_returns_none(_patch_modified):
    with _patch_modified(["a.py"]):
        block, sig = await build_context(_MockLSP(), None)
    assert block is None and sig is None


@pytest.mark.asyncio
async def test_no_modified_files(tmp_path, _patch_modified):
    with _patch_modified([]):
        block, sig = await build_context(_MockLSP(), tmp_path)
    assert block is None and sig is None


@pytest.mark.asyncio
async def test_non_source_files_filtered(tmp_path, _patch_modified):
    (tmp_path / "README.md").write_text("x")
    (tmp_path / "data.json").write_text("{}")
    with _patch_modified(["README.md", "data.json"]):
        block, sig = await build_context(_MockLSP(), tmp_path)
    assert block is None and sig is None


@pytest.mark.asyncio
async def test_renders_symbols_and_errors(tmp_path, _patch_modified):
    f = tmp_path / "src.py"
    f.write_text("x")
    lsp = _MockLSP(
        symbols={str(f): [_sym("SessionManager", 5, "(store)"), _sym("validate", 6)]},
        diagnostics={str(f): [_diag(1, "undefined name 'X'", line=41)]},
    )
    with _patch_modified(["src.py"]):
        block, sig = await build_context(lsp, tmp_path)
    assert block is not None and sig is not None
    assert "src.py" in block
    assert "class SessionManager" in block
    assert "method validate" in block
    assert "! L42: undefined name 'X'" in block


@pytest.mark.asyncio
async def test_throttles_on_unchanged_signature(tmp_path, _patch_modified):
    f = tmp_path / "src.py"
    f.write_text("x")
    lsp = _MockLSP(symbols={str(f): [_sym("Foo", 5)]})
    with _patch_modified(["src.py"]):
        block1, sig = await build_context(lsp, tmp_path)
        block2, sig2 = await build_context(lsp, tmp_path, last_signature=sig)
    assert block1 is not None
    assert block2 is None  # unchanged → skip
    assert sig == sig2


@pytest.mark.asyncio
async def test_reinjects_after_change(tmp_path, _patch_modified):
    f = tmp_path / "src.py"
    f.write_text("x")
    lsp = _MockLSP(symbols={str(f): [_sym("Foo", 5)]})
    with _patch_modified(["src.py"]):
        block1, sig1 = await build_context(lsp, tmp_path)
        lsp._symbols[str(f)] = [_sym("Foo", 5), _sym("Bar", 12)]  # changed
        block2, sig2 = await build_context(lsp, tmp_path, last_signature=sig1)
    assert block1 is not None and block2 is not None
    assert sig1 != sig2


@pytest.mark.asyncio
async def test_warnings_excluded(tmp_path, _patch_modified):
    f = tmp_path / "src.py"
    f.write_text("x")
    lsp = _MockLSP(diagnostics={str(f): [_diag(2, "unused import", line=1)]})
    with _patch_modified(["src.py"]):
        block, sig = await build_context(lsp, tmp_path)
    # warning-only file yields nothing to inject
    assert block is None and sig is None


@pytest.mark.asyncio
async def test_budget_truncation(tmp_path, _patch_modified):
    f = tmp_path / "src.py"
    f.write_text("x")
    lsp = _MockLSP(symbols={str(f): [_sym(f"name_{i}", 12) for i in range(20)]})
    with _patch_modified(["src.py"]):
        block, _ = await build_context(lsp, tmp_path, char_budget=80)
    assert block is not None
    assert block.endswith("…")
    assert len(block) <= 80


@pytest.mark.asyncio
async def test_max_files_cap(tmp_path, _patch_modified):
    for n in range(5):
        (tmp_path / f"f{n}.py").write_text("x")
    lsp = _MockLSP()
    # every file returns one symbol
    for n in range(5):
        p = str(tmp_path / f"f{n}.py")
        lsp._symbols[p] = [_sym(f"S{n}", 5)]
    with _patch_modified([f"f{n}.py" for n in range(5)]):
        block, _ = await build_context(lsp, tmp_path, max_files=2)
    # only 2 files appear (header line + symbols)
    file_headers = [l for l in block.splitlines() if l.endswith(".py")]
    assert len(file_headers) == 2


@pytest.mark.asyncio
async def test_lsp_exception_is_swallowed(tmp_path, _patch_modified):
    class RaisingLSP:
        async def get_document_symbols(self, fp):
            raise RuntimeError("boom")

        def get_diagnostics(self, fp):
            return []

    (tmp_path / "src.py").write_text("x")
    with _patch_modified(["src.py"]):
        block, sig = await build_context(RaisingLSP(), tmp_path)
    # exception caught per-file → empty → nothing to inject
    assert block is None and sig is None


@pytest.mark.asyncio
async def test_dropped_file_with_no_symbols_or_errors(tmp_path, _patch_modified):
    (tmp_path / "src.py").write_text("x")
    lsp = _MockLSP()  # no symbols, no diagnostics
    with _patch_modified(["src.py"]):
        block, sig = await build_context(lsp, tmp_path)
    assert block is None and sig is None
