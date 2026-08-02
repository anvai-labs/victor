# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regression guard: vertical ``get_lsp()`` wires to the orchestrator (FEP-0019).

The framework defines ``VerticalBase.get_lsp()`` and ``AgentOrchestrator.set_lsp``
(and the ``"lsp" -> "set_lsp"`` capability mapping) but, before this guard, NO
step handler ever read ``get_lsp()`` and wired it — so even a vertical that
provided a live LSP implementation never reached the orchestrator, leaving the
entire FEP-0019 chain (middleware + symbol context) dormant.

These tests pin the wiring so a regression (dropping the ``apply_lsp`` step) is
caught in CI. They use plain stub orchestrators (not bare ``MagicMock``) to avoid
the ``runtime_checkable Protocol`` isinstance short-circuit in
``invoke_capability``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from victor.core.verticals.base import VerticalBase
from victor.framework.lsp_context import build_context
from victor.framework.lsp_middleware import LSPDiagnosticMiddleware

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _SetLspRecorder:
    """Stub orchestrator that records set_lsp calls (plain class, not MagicMock)."""

    def __init__(self) -> None:
        self.set_lsp_calls: List[Any] = []

    def set_lsp(self, capability: Any) -> None:
        self.set_lsp_calls.append(capability)


class _RecordingChain:
    """Minimal middleware chain that records add() calls."""

    def __init__(self) -> None:
        self.added: List[Any] = []

    def add(self, middleware: Any) -> None:
        middleware.get_priority()  # mirror real chain contract
        self.added.append(middleware)


class _RealSetLspOrchestrator:
    """Orchestrator stub whose set_lsp delegates to the real registration helper.

    Mirrors AgentOrchestrator.set_lsp so the full FEP-0019 chain can be verified
    end-to-end without bootstrapping the whole orchestrator.
    """

    def __init__(self) -> None:
        self._lsp: Any = None
        self._lsp_middleware: Any = None
        self._lsp_feedback_mode = "errors"
        self._middleware_chain = _RecordingChain()

    def set_lsp(self, capability: Any) -> None:
        from victor.framework.lsp_middleware import register_lsp_on_chain

        self._lsp = capability
        register_lsp_on_chain(self, self._middleware_chain)


class _MockLSPImpl:
    """Duck-typed LSP implementation a vertical might provide."""

    def __init__(self, symbols=None, diagnostics=None):
        self._symbols = symbols or {}
        self._diags = diagnostics or {}

    async def get_document_symbols(self, file_path: str):
        return self._symbols.get(file_path, [])

    def get_diagnostics(self, file_path: str):
        return self._diags.get(file_path, [])


def _sym(name, kind=5):
    return SimpleNamespace(name=name, kind=kind, detail=None, children=[])


def _diag(severity=1, message="err", line=10):
    return SimpleNamespace(
        severity=severity,
        message=message,
        range=SimpleNamespace(start=SimpleNamespace(line=line)),
    )


class MockVerticalWithLsp(VerticalBase):
    """Vertical that provides a live LSP implementation."""

    name = "test_lsp_vertical"
    _lsp_singleton: Any = None

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "Test"

    @classmethod
    def get_lsp(cls) -> Any:
        return cls._lsp_singleton


# ---------------------------------------------------------------------------
# Wiring unit tests (TDD core)
# ---------------------------------------------------------------------------


def _ctx_and_result():
    context = MagicMock()
    result = MagicMock()
    result.add_info = MagicMock()
    result.add_warning = MagicMock()
    return context, result


class TestApplyLspWiring:
    """apply_lsp reads vertical.get_lsp() and wires it to the orchestrator."""

    def test_apply_lsp_wires_vertical_lsp_to_set_lsp(self):
        from victor.framework.step_handlers import FrameworkStepHandler

        lsp = _MockLSPImpl()
        MockVerticalWithLsp._lsp_singleton = lsp
        try:
            orch = _SetLspRecorder()
            context, result = _ctx_and_result()
            handler = FrameworkStepHandler()

            handler.apply_lsp(orch, MockVerticalWithLsp, context, result)

            assert orch.set_lsp_calls == [lsp]
        finally:
            MockVerticalWithLsp._lsp_singleton = None

    def test_apply_lsp_skips_when_vertical_has_no_lsp(self):
        from victor.framework.step_handlers import FrameworkStepHandler

        orch = _SetLspRecorder()
        context, result = _ctx_and_result()
        handler = FrameworkStepHandler()

        # VerticalBase.get_lsp returns None by default.
        handler.apply_lsp(orch, _PlainVertical, context, result)

        assert orch.set_lsp_calls == []

    def test_do_apply_invokes_apply_lsp(self):
        """The full _do_apply pipeline must wire LSP (regression guard)."""
        from victor.framework.step_handlers import FrameworkStepHandler

        lsp = _MockLSPImpl()
        MockVerticalWithLsp._lsp_singleton = lsp
        try:
            orch = _SetLspRecorder()
            context, result = _ctx_and_result()
            handler = FrameworkStepHandler()

            handler._do_apply(orch, MockVerticalWithLsp, context, result)

            assert orch.set_lsp_calls == [lsp]
        finally:
            MockVerticalWithLsp._lsp_singleton = None


class _PlainVertical(VerticalBase):
    name = "plain_vertical"

    @classmethod
    def get_tools(cls):
        return ["read"]

    @classmethod
    def get_system_prompt(cls):
        return "Test"


# ---------------------------------------------------------------------------
# End-to-end chain verification (vertical.get_lsp -> middleware + symbols)
# ---------------------------------------------------------------------------


class TestLspEndToEndChain:
    """When a vertical provides LSP, the full FEP-0019 chain activates."""

    def test_vertical_lsp_reaches_middleware_and_context_provider(self, tmp_path):
        from victor.framework.step_handlers import FrameworkStepHandler

        f = tmp_path / "src.py"
        f.write_text("x")
        lsp = _MockLSPImpl(
            symbols={str(f): [_sym("SessionManager", 5)]},
            diagnostics={str(f): [_diag(1, "undefined 'x'", line=41)]},
        )
        MockVerticalWithLsp._lsp_singleton = lsp
        try:
            orch = _RealSetLspOrchestrator()
            context, result = _ctx_and_result()
            handler = FrameworkStepHandler()

            handler.apply_lsp(orch, MockVerticalWithLsp, context, result)

            # 1. The capability reached the orchestrator.
            assert orch._lsp is lsp
            # 2. Phase 2: the diagnostic middleware auto-registered on the chain.
            assert len(orch._middleware_chain.added) == 1
            assert isinstance(orch._middleware_chain.added[0], LSPDiagnosticMiddleware)

            # 3. Phase 3: the context provider yields symbols from the live impl.
            async def fake_modified(workspace, timeout=15):
                return ["src.py"]

            with patch(
                "victor.framework.lsp_context.workspace_files_modified",
                fake_modified,
            ):
                block, _ = asyncio_run(build_context(orch._lsp, tmp_path))
            assert block is not None
            assert "class SessionManager" in block
            assert "! L42:" in block
        finally:
            MockVerticalWithLsp._lsp_singleton = None


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)
