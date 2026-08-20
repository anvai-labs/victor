# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the framework LSP adapter (victor-coding → FEP-0019 activation).

The adapter bridges victor-coding's ``LSPConnectionPool`` (display dicts +
``victor_contracts`` types) to the LSP shape the FEP-0019 framework runtime
duck-types, so the diagnostic middleware + symbol context provider activate.
Uses a mock pool (no live language server).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import pytest

from victor_contracts.lsp_runtime import (
    Diagnostic,
    DiagnosticSeverity,
    DocumentSymbol,
    Position,
    Range,
    SymbolKind,
)


def _ds(name: str, kind: SymbolKind, children=None, detail=None) -> DocumentSymbol:
    r = Range(start=Position(line=1, character=0), end=Position(line=9, character=0))
    sel = Range(start=Position(line=1, character=6), end=Position(line=1, character=6))
    return DocumentSymbol(
        name=name,
        kind=kind,
        range=r,
        selection_range=sel,
        detail=detail,
        children=children or [],
    )


class _MockPool:
    """Stand-in for LSPConnectionPool returning controlled data."""

    def __init__(self, symbols=None):
        self._symbols = symbols or []
        self.calls: List[tuple] = []

    async def open_document(self, file_path: str, text=None) -> bool:
        self.calls.append(("open_document", file_path))
        return True

    async def update_document(self, file_path: str, text: str) -> bool:
        self.calls.append(("update_document", file_path))
        return True

    def close_document(self, file_path: str) -> None:
        self.calls.append(("close_document", file_path))

    def get_diagnostics(self, file_path: str):
        # Match the pool's real shape: severity as STRING, line 1-indexed.
        return [
            {
                "line": 41,
                "character": 5,
                "message": "undefined name 'foo'",
                "severity": "error",
                "source": "pyright",
                "code": None,
            },
            {
                "line": 8,
                "character": 0,
                "message": "unused import",
                "severity": "warning",
                "source": "ruff",
                "code": "F401",
            },
        ]

    async def get_document_symbols(self, file_path: str):
        self.calls.append(("get_document_symbols", file_path))
        return self._symbols


def _adapter(pool=None):
    from victor_coding.lsp.framework_adapter import FrameworkLSPAdapter

    return FrameworkLSPAdapter(pool or _MockPool())


class TestDiagnosticsConversion:
    def test_severity_name_to_enum_and_line_zero_indexed(self):
        adapter = _adapter()
        diags = adapter.get_diagnostics("a.py")
        assert len(diags) == 2
        assert all(isinstance(d, Diagnostic) for d in diags)
        # "error" -> ERROR(1), line 41 (1-indexed) -> 40
        assert diags[0].severity == DiagnosticSeverity.ERROR
        assert int(diags[0].severity) == 1
        assert diags[0].range.start.line == 40
        assert diags[0].range.start.character == 5
        assert diags[0].message == "undefined name 'foo'"
        assert diags[0].source == "pyright"
        # "warning" -> WARNING(2)
        assert diags[1].severity == DiagnosticSeverity.WARNING
        assert int(diags[1].severity) == 2
        assert diags[1].code == "F401"

    def test_unknown_severity_defaults_to_error(self):
        pool = _MockPool()
        pool.get_diagnostics = lambda fp: [
            {"line": 1, "character": 0, "message": "x", "severity": "bogus"}
        ]
        diags = _adapter(pool).get_diagnostics("a.py")
        assert diags[0].severity == DiagnosticSeverity.ERROR


class TestDocumentSymbols:
    @pytest.mark.asyncio
    async def test_symbols_returned_as_contracts_types(self):
        pool = _MockPool(
            symbols=[
                _ds(
                    "SessionManager",
                    SymbolKind.CLASS,
                    detail="(store)",
                    children=[
                        _ds("__init__", SymbolKind.CONSTRUCTOR),
                        _ds("validate", SymbolKind.METHOD),
                    ],
                ),
                _ds("TOP_LEVEL_CONST", SymbolKind.CONSTANT),
            ]
        )
        symbols = await _adapter(pool).get_document_symbols("a.py")
        assert len(symbols) == 2
        assert all(isinstance(s, DocumentSymbol) for s in symbols)
        # kind is an IntEnum — the framework reads it as int.
        assert symbols[0].name == "SessionManager"
        assert int(symbols[0].kind) == int(SymbolKind.CLASS)
        assert symbols[0].detail == "(store)"
        assert len(symbols[0].children) == 2
        assert int(symbols[0].children[0].kind) == int(SymbolKind.CONSTRUCTOR)

    @pytest.mark.asyncio
    async def test_empty_symbols(self):
        symbols = await _adapter().get_document_symbols("a.py")
        assert symbols == []


class TestPassthrough:
    @pytest.mark.asyncio
    async def test_open_and_update_delegate_to_pool(self):
        pool = _MockPool()
        adapter = _adapter(pool)
        assert await adapter.open_document("a.py", "x = 1") is True
        assert await adapter.update_document("a.py", "x = 2") is True
        adapter.close_document("a.py")
        names = [c[0] for c in pool.calls]
        assert names == ["open_document", "update_document", "close_document"]


class TestGetLspWiring:
    def test_coding_assistant_get_lsp_returns_adapter(self):
        from victor_coding.assistant import CodingAssistant
        from victor_coding.lsp.framework_adapter import FrameworkLSPAdapter

        cap = CodingAssistant.get_lsp()
        assert isinstance(cap, FrameworkLSPAdapter)


class TestEndToEndActivation:
    """The real CodingAssistant flows through the framework's apply_lsp."""

    def test_apply_lsp_wires_coding_assistant_capability(self):
        from victor.framework.step_handlers import FrameworkStepHandler

        from victor_coding.assistant import CodingAssistant
        from victor_coding.lsp.framework_adapter import FrameworkLSPAdapter

        class _Recorder:
            def __init__(self) -> None:
                self.set_calls: list = []

            def set_lsp(self, capability) -> None:
                self.set_calls.append(capability)

        orch = _Recorder()
        context = SimpleNamespace()
        result = SimpleNamespace(add_info=lambda *a, **k: None, add_warning=lambda *a, **k: None)

        FrameworkStepHandler().apply_lsp(orch, CodingAssistant, context, result)

        assert len(orch.set_calls) == 1
        assert isinstance(orch.set_calls[0], FrameworkLSPAdapter)
