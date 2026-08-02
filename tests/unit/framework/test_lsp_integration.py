# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for FEP-0019 LSP-integrated verification.

Covers:
- ``LSPVerifier`` (DECIDE gate) — diagnostics → VerificationResult, degradation.
- ``LSPDiagnosticMiddleware`` (ACT phase) — same-turn diagnostic injection.
- ``AgentOrchestrator.set_lsp`` auto-registration of the middleware.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from victor.framework.lsp_middleware import LSPDiagnosticMiddleware
from victor.framework.verification import VerificationResult
from victor.framework.verifiers import LSPVerifier

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _diag(severity: int, message: str, line: int = 10):
    """Build a duck-typed LSP diagnostic."""
    return SimpleNamespace(
        severity=severity,
        message=message,
        range=SimpleNamespace(start=SimpleNamespace(line=line)),
    )


class _MockLSP:
    """Minimal LSP capability double: async update_document + sync get_diagnostics."""

    def __init__(self, diagnostics_map: dict | None = None):
        self._diags = diagnostics_map or {}
        self.updated: list = []

    async def update_document(self, file_path: str, content: str) -> bool:
        self.updated.append((file_path, content))
        return True

    def get_diagnostics(self, file_path: str):
        return self._diags.get(file_path, [])


class _RaisingLSP(_MockLSP):
    def get_diagnostics(self, file_path: str):
        raise RuntimeError("LSP server crashed")


# ---------------------------------------------------------------------------
# LSPVerifier
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lsp_verifier_no_capability_is_vacuous_pass():
    """No LSP capability → vacuous VerificationResult, no crash."""
    verifier = LSPVerifier(lsp_capability=None)
    result = await verifier.verify(workspace=Path("/tmp"))
    assert isinstance(result, VerificationResult)
    assert result.total == 0
    assert "not available" in result.feedback.lower()


@pytest.mark.asyncio
async def test_lsp_verifier_no_edited_files():
    """Capability present, no workspace → 'no edited files'."""
    verifier = LSPVerifier(lsp_capability=_MockLSP())
    result = await verifier.verify(workspace=None)
    assert result.total == 0
    assert "no edited files" in result.feedback.lower()


@pytest.mark.asyncio
async def test_lsp_verifier_clean(monkeypatch):
    """Edited files with zero errors → VERIFIED."""

    async def _fake_modified(workspace, timeout=15):
        return ["a.py", "b.py"]

    monkeypatch.setattr("victor.framework.verifiers.workspace_files_modified", _fake_modified)
    lsp = _MockLSP()  # no diagnostics → clean
    verifier = LSPVerifier(lsp_capability=lsp)
    result = await verifier.verify(workspace=Path("/tmp"))
    assert result.is_verified
    assert result.passed == result.total == 1
    assert "VERIFIED" in result.feedback


@pytest.mark.asyncio
async def test_lsp_verifier_reports_errors(monkeypatch):
    """Severity-1 errors → not verified, feedback lists them."""

    async def _fake_modified(workspace, timeout=15):
        return ["a.py"]

    monkeypatch.setattr("victor.framework.verifiers.workspace_files_modified", _fake_modified)
    lsp = _MockLSP(
        {
            "a.py": [
                _diag(1, "undefined name 'foo'", line=41),
                _diag(1, "expected str, got int", line=58),
            ]
        }
    )
    verifier = LSPVerifier(lsp_capability=lsp)
    result = await verifier.verify(workspace=Path("/tmp"))
    assert not result.is_verified
    assert result.passed == 0
    assert result.total == 2
    assert "undefined name 'foo'" in result.feedback
    assert "expected str, got int" in result.feedback


@pytest.mark.asyncio
async def test_lsp_verifier_warnings_excluded_unless_requested(monkeypatch):
    """Severity-2 warnings are ignored by default, counted when include_warnings."""

    async def _fake_modified(workspace, timeout=15):
        return ["a.py"]

    monkeypatch.setattr("victor.framework.verifiers.workspace_files_modified", _fake_modified)
    lsp = _MockLSP({"a.py": [_diag(2, "unused import 'os'", line=1)]})

    # Default: warnings ignored → clean.
    clean = await LSPVerifier(lsp_capability=lsp).verify(workspace=Path("/tmp"))
    assert clean.is_verified

    # include_warnings → warning counts as a failure.
    strict = await LSPVerifier(lsp_capability=lsp, include_warnings=True).verify(
        workspace=Path("/tmp")
    )
    assert not strict.is_verified
    assert strict.total == 1


@pytest.mark.asyncio
async def test_lsp_verifier_survives_lsp_exception(monkeypatch):
    """A crashing LSP server doesn't fail verification — file is skipped."""

    async def _fake_modified(workspace, timeout=15):
        return ["a.py"]

    monkeypatch.setattr("victor.framework.verifiers.workspace_files_modified", _fake_modified)
    verifier = LSPVerifier(lsp_capability=_RaisingLSP())
    result = await verifier.verify(workspace=Path("/tmp"))
    # Exception caught per-file → clean result.
    assert result.is_verified


# ---------------------------------------------------------------------------
# LSPDiagnosticMiddleware
# ---------------------------------------------------------------------------


def test_middleware_protocol_shape():
    """Middleware declares LOW priority and the file-modifying tool set."""
    mw = LSPDiagnosticMiddleware()
    assert mw.get_priority().name == "LOW"
    tools = mw.get_applicable_tools()
    assert tools is not None
    assert "edit" in tools and "write" in tools
    assert "read" not in tools


@pytest.mark.asyncio
async def test_middleware_no_lsp_is_noop():
    """No LSP capability → after_tool_call returns None (no diagnostic fetch)."""
    mw = LSPDiagnosticMiddleware(lsp_capability=None)
    out = await mw.after_tool_call("edit", {"path": "a.py"}, {"output": "ok"}, True)
    assert out is None


@pytest.mark.asyncio
async def test_middleware_non_applicable_tool_skipped():
    """Non-file tools (read) are skipped even with LSP present."""
    mw = LSPDiagnosticMiddleware(lsp_capability=_MockLSP({"a.py": [_diag(1, "x")]}))
    out = await mw.after_tool_call("read", {"path": "a.py"}, {"output": "..."}, True)
    assert out is None


@pytest.mark.asyncio
async def test_middleware_failed_tool_skipped():
    """An unsuccessful tool call is not diagnosed."""
    mw = LSPDiagnosticMiddleware(lsp_capability=_MockLSP({"a.py": [_diag(1, "x")]}))
    out = await mw.after_tool_call("edit", {"path": "a.py"}, {"output": "..."}, False)
    assert out is None


@pytest.mark.asyncio
async def test_middleware_injects_into_dict_result(tmp_path):
    """A dict result gets diagnostics appended to output + lsp_diagnostics key."""
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    lsp = _MockLSP({str(f): [_diag(1, "undefined name 'foo'", line=41)]})
    mw = LSPDiagnosticMiddleware(lsp_capability=lsp, debounce_ms=0)
    out = await mw.after_tool_call("edit", {"path": str(f)}, {"output": "wrote"}, True)
    assert out is not None
    assert "undefined name 'foo'" in out["output"]
    assert out["lsp_diagnostics"]
    # update_document was pushed the new content.
    assert lsp.updated and lsp.updated[0][0] == str(f)


@pytest.mark.asyncio
async def test_middleware_injects_into_str_result(tmp_path):
    """A string result gets diagnostics appended as text."""
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    lsp = _MockLSP({str(f): [_diag(1, "type error", line=5)]})
    mw = LSPDiagnosticMiddleware(lsp_capability=lsp, debounce_ms=0)
    out = await mw.after_tool_call("write", {"file_path": str(f)}, "done", True)
    assert isinstance(out, str)
    assert "type error" in out


@pytest.mark.asyncio
async def test_middleware_mode_all_includes_warnings(tmp_path):
    """mode='all' surfaces severity-2 warnings; default 'errors' hides them."""
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    lsp = _MockLSP({str(f): [_diag(2, "unused import", line=1)]})

    errors_only = LSPDiagnosticMiddleware(lsp_capability=lsp, mode="errors", debounce_ms=0)
    out = await errors_only.after_tool_call("edit", {"path": str(f)}, {"output": ""}, True)
    assert out is None  # warning filtered → nothing to append

    all_mode = LSPDiagnosticMiddleware(lsp_capability=lsp, mode="all", debounce_ms=0)
    out2 = await all_mode.after_tool_call("edit", {"path": str(f)}, {"output": ""}, True)
    assert out2 is not None and "unused import" in out2["output"]


@pytest.mark.asyncio
async def test_middleware_failure_is_silent(tmp_path):
    """A crashing LSP server never breaks the tool result."""
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    mw = LSPDiagnosticMiddleware(lsp_capability=_RaisingLSP(), debounce_ms=0)
    out = await mw.after_tool_call("edit", {"path": str(f)}, {"output": "wrote"}, True)
    assert out is None  # swallowed → original result preserved


@pytest.mark.asyncio
async def test_middleware_no_path_argument_skipped():
    """Missing path/file_path arg → no diagnostic fetch."""
    mw = LSPDiagnosticMiddleware(lsp_capability=_MockLSP(), debounce_ms=0)
    out = await mw.after_tool_call("edit", {}, {"output": "wrote"}, True)
    assert out is None


@pytest.mark.asyncio
async def test_middleware_before_is_noop():
    """before_tool_call always proceeds (diagnostics are post-edit only)."""
    from victor.core.vertical_types import MiddlewareResult

    mw = LSPDiagnosticMiddleware(lsp_capability=_MockLSP())
    result = await mw.before_tool_call("edit", {"path": "a.py"})
    assert isinstance(result, MiddlewareResult)
    assert result.proceed is True


def test_middleware_lsp_property_setter():
    """The lsp property can be bound/rebound after construction."""
    mw = LSPDiagnosticMiddleware()
    assert mw.lsp is None
    cap = _MockLSP()
    mw.lsp = cap
    assert mw.lsp is cap


# ---------------------------------------------------------------------------
# Orchestrator wiring (set_lsp auto-registers the middleware)
# ---------------------------------------------------------------------------


class _FakeChain:
    """Records add() calls; mirrors MiddlewareChain.add usage."""

    def __init__(self):
        self.added: list = []

    def add(self, middleware):
        # Real chain calls get_priority(); exercise it to catch contract drift.
        middleware.get_priority()
        self.added.append(middleware)


def test_orchestrator_set_lsp_registers_middleware():
    """set_lsp(capability) registers LSPDiagnosticMiddleware on the chain once."""
    from victor.agent.orchestrator import AgentOrchestrator

    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    chain = _FakeChain()
    orch._middleware_chain = chain

    cap = _MockLSP()
    orch.set_lsp(cap)

    assert len(chain.added) == 1
    mw = chain.added[0]
    assert isinstance(mw, LSPDiagnosticMiddleware)
    assert mw.lsp is cap
    # Re-calling set_lsp does NOT double-register (idempotent) — rebinds only.
    orch.set_lsp(_MockLSP())
    assert len(chain.added) == 1


def test_orchestrator_set_lsp_none_does_not_register():
    """set_lsp(None) clears the capability without touching the chain."""
    from victor.agent.orchestrator import AgentOrchestrator

    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    chain = _FakeChain()
    orch._middleware_chain = chain
    orch.set_lsp(None)
    assert chain.added == []


def test_orchestrator_set_middleware_chain_back_registers():
    """If LSP is set before the chain exists, the chain-set path registers it."""
    from victor.agent.orchestrator import AgentOrchestrator

    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch.set_lsp(_MockLSP())  # chain not yet present → deferred
    assert getattr(orch, "_lsp_middleware", None) is None  # not registered yet

    chain = _FakeChain()
    orch.set_middleware_chain(chain)  # now registers
    assert len(chain.added) == 1


def test_orchestrator_set_lsp_feedback_mode_propagates():
    """set_lsp_feedback_mode updates the registered middleware + remembers it."""
    from victor.agent.orchestrator import AgentOrchestrator

    from victor.framework.lsp_middleware import (
        register_lsp_on_chain,
        set_lsp_feedback_mode,
    )

    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    chain = _FakeChain()
    orch._middleware_chain = chain
    orch.set_lsp(_MockLSP())
    mw = chain.added[0]
    assert mw._mode == "errors"

    set_lsp_feedback_mode(orch, "all")
    assert mw._mode == "all"

    # A later re-registration honors the remembered mode.
    orch._lsp_middleware = None
    chain.added.clear()
    register_lsp_on_chain(orch, chain)
    assert chain.added[0]._mode == "all"
