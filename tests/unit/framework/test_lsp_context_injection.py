# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for the FEP-0019 Phase 3 LSP context injection in AgenticLoop.

Covers ``_maybe_inject_lsp_context`` and ``_resolve_workspace``: opt-in gating,
no-LSP / no-workspace no-ops, inject-once + signature throttle, and the
workspace fallback chain. Uses ``AgenticLoop.__new__`` to build a bare instance
(no full bootstrap).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from victor.framework.agentic_loop import AgenticLoop
from victor.framework.lsp_context import build_context  # noqa: F401 (sanity import)


def _diag(severity=1, message="err", line=10):
    return SimpleNamespace(
        severity=severity,
        message=message,
        range=SimpleNamespace(start=SimpleNamespace(line=line)),
    )


def _sym(name, kind=5):
    return SimpleNamespace(name=name, kind=kind, detail=None, children=[])


class _MockLSP:
    def __init__(self, symbols=None, diagnostics=None):
        self._symbols = symbols or {}
        self._diags = diagnostics or {}

    async def get_document_symbols(self, fp):
        return self._symbols.get(fp, [])

    def get_diagnostics(self, fp):
        return self._diags.get(fp, [])


class _RecordingChatCtx:
    """Minimal chat context that records add_message calls."""

    def __init__(self):
        self.messages = []

    def add_message(self, role, content, metadata=None, **kwargs):
        self.messages.append({"role": role, "content": content, "metadata": metadata})


def _make_loop(*, lsp=None, chat_ctx=None, enabled=True, workspace=None, project_root=None):
    """Build a bare AgenticLoop with just the attrs the injector touches."""
    loop = AgenticLoop.__new__(AgenticLoop)
    loop._lsp_context_enabled = enabled
    loop._last_lsp_signature = None

    turn_executor = SimpleNamespace()
    turn_executor._chat_context = chat_ctx
    loop.turn_executor = turn_executor

    project_context = None
    if project_root is not None:
        project_context = SimpleNamespace(root_path=project_root)
    loop.orchestrator = SimpleNamespace(lsp=lsp, project_context=project_context)

    # state workspace is set by the caller; default None
    loop._state_workspace = workspace
    return loop


@pytest.mark.asyncio
async def test_disabled_does_not_inject(tmp_path):
    chat = _RecordingChatCtx()
    loop = _make_loop(lsp=_MockLSP(), chat_ctx=chat, enabled=False, workspace=tmp_path)
    await loop._maybe_inject_lsp_context({"workspace": str(tmp_path)})
    assert chat.messages == []


@pytest.mark.asyncio
async def test_no_lsp_does_not_inject(tmp_path):
    chat = _RecordingChatCtx()
    loop = _make_loop(lsp=None, chat_ctx=chat, enabled=True, workspace=tmp_path)
    await loop._maybe_inject_lsp_context({"workspace": str(tmp_path)})
    assert chat.messages == []


@pytest.mark.asyncio
async def test_no_workspace_does_not_inject(monkeypatch):
    chat = _RecordingChatCtx()
    loop = _make_loop(lsp=_MockLSP(), chat_ctx=chat, enabled=True)  # no project_root
    # Force the final get_project_paths() fallback to fail too.
    monkeypatch.setattr(
        "victor.config.settings.get_project_paths",
        lambda: (_ for _ in ()).throw(RuntimeError("no paths")),
    )
    await loop._maybe_inject_lsp_context({})  # no workspace key
    assert chat.messages == []


@pytest.mark.asyncio
async def test_injects_once_then_throttles(tmp_path):
    f = tmp_path / "src.py"
    f.write_text("x")
    chat = _RecordingChatCtx()
    lsp = _MockLSP(
        symbols={str(f): [_sym("Foo")]},
        diagnostics={str(f): [_diag(1, "undefined 'x'", line=41)]},
    )
    loop = _make_loop(lsp=lsp, chat_ctx=chat, enabled=True, workspace=tmp_path)

    async def fake_modified(workspace, timeout=15):
        return ["src.py"]

    with patch("victor.framework.lsp_context.workspace_files_modified", fake_modified):
        await loop._maybe_inject_lsp_context({"workspace": str(tmp_path)})
        assert len(chat.messages) == 1
        assert "class Foo" in chat.messages[0]["content"]
        assert "! L42:" in chat.messages[0]["content"]
        assert chat.messages[0]["role"] == "user"
        assert loop._last_lsp_signature is not None

        # Second call: identical content → throttled (no new message).
        await loop._maybe_inject_lsp_context({"workspace": str(tmp_path)})
        assert len(chat.messages) == 1


@pytest.mark.asyncio
async def test_resolve_workspace_state_first(tmp_path):
    loop = _make_loop(project_root=Path("/elsewhere"))
    resolved = loop._resolve_workspace({"workspace": str(tmp_path)})
    assert resolved == Path(str(tmp_path))


@pytest.mark.asyncio
async def test_resolve_workspace_project_context_fallback(tmp_path):
    loop = _make_loop(project_root=tmp_path)
    resolved = loop._resolve_workspace({})
    assert resolved == tmp_path


@pytest.mark.asyncio
async def test_resolve_workspace_working_dir_key(tmp_path):
    loop = _make_loop()  # no project_root
    resolved = loop._resolve_workspace({"working_dir": str(tmp_path)})
    assert resolved == Path(str(tmp_path))


@pytest.mark.asyncio
async def test_resolve_workspace_all_fail(monkeypatch):
    loop = _make_loop()  # no project_root, no state workspace
    monkeypatch.setattr(
        "victor.config.settings.get_project_paths",
        lambda: (_ for _ in ()).throw(RuntimeError("no paths")),
    )
    assert loop._resolve_workspace({}) is None


@pytest.mark.asyncio
async def test_lsp_gather_exception_is_swallowed(tmp_path):
    """A crashing LSP server never injects and never raises."""
    f = tmp_path / "src.py"
    f.write_text("x")
    chat = _RecordingChatCtx()

    class RaisingLSP:
        async def get_document_symbols(self, fp):
            raise RuntimeError("boom")

        def get_diagnostics(self, fp):
            return []

    loop = _make_loop(lsp=RaisingLSP(), chat_ctx=chat, enabled=True, workspace=tmp_path)

    async def fake_modified(workspace, timeout=15):
        return ["src.py"]

    with patch("victor.framework.lsp_context.workspace_files_modified", fake_modified):
        await loop._maybe_inject_lsp_context({"workspace": str(tmp_path)})
    assert chat.messages == []
