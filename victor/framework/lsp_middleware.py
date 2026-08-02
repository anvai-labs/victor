# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""LSP diagnostic middleware for real-time code feedback (FEP-0019).

Implements ``MiddlewareProtocol`` (``victor.core.verticals.protocols``) so it
slots into the existing tool ``MiddlewareChain``. After every file-modifying
tool call (edit/write) it pushes the new content into the LSP server, briefly
waits for the server to publish updated diagnostics, then appends any errors to
the tool result — the agent sees ``Line 42: undefined name 'foo'`` in the same
turn. This is the generate→diagnose→fix loop, far cheaper than claiming done
and re-running the whole suite.

Gracefully degrades: when LSP is unavailable (victor-coding not installed) the
middleware is a no-op — ``after_tool_call`` returns ``None`` and the edit result
passes through unchanged.

Wiring is automatic: ``AgentOrchestrator.set_lsp()`` registers this middleware
on the chain whenever a vertical provides an LSP capability. No manual setup.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Set

from victor.core.vertical_types import MiddlewarePriority, MiddlewareResult

logger = logging.getLogger(__name__)

# Tools that modify files and should trigger LSP diagnostics.
_FILE_MODIFYING_TOOLS: Set[str] = frozenset(
    {"edit", "write", "file_editor", "file_write", "lsp_write_enhancer"}
)


def register_lsp_on_chain(owner: Any, chain: Any) -> None:
    """Register the LSP diagnostic middleware on ``chain``, binding ``owner._lsp``.

    Idempotent: creates the middleware once (stored on ``owner._lsp_middleware``)
    and re-binds the capability on every call. Reads ``owner._lsp_feedback_mode``
    for the ``mode`` and syncs an existing middleware to it. Safe to call from
    either ``set_lsp`` or ``set_middleware_chain`` — in either order — so the
    orchestrator itself stays a thin delegator (FEP-0019).

    Args:
        owner: The orchestrator-like object (needs ``_lsp`` and, optionally,
            ``_lsp_feedback_mode`` / ``_lsp_middleware`` attributes).
        chain: The tool ``MiddlewareChain`` (must expose ``add``).
    """
    try:
        mode = getattr(owner, "_lsp_feedback_mode", "errors")
        middleware = getattr(owner, "_lsp_middleware", None)
        if middleware is None:
            middleware = LSPDiagnosticMiddleware(mode=mode)
            owner._lsp_middleware = middleware
            chain.add(middleware)
        else:
            middleware._mode = mode
        middleware.lsp = getattr(owner, "_lsp", None)
    except Exception:
        logger.debug("LSP middleware registration deferred", exc_info=True)


def set_lsp_feedback_mode(owner: Any, mode: str) -> None:
    """Set the LSP diagnostics feedback mode on ``owner`` (FEP-0019).

    ``"errors"`` (default) reports only severity-1 diagnostics; ``"all"``
    includes warnings; ``"none"`` disables feedback. Stored on the owner and
    applied to an already-registered middleware immediately; remembered for any
    later (re)registration.
    """
    owner._lsp_feedback_mode = mode
    middleware = getattr(owner, "_lsp_middleware", None)
    if middleware is not None:
        middleware._mode = mode


class LSPDiagnosticMiddleware:
    """Inject LSP diagnostics after file-modifying tool calls.

    After an edit/write tool succeeds:
    1. Push the new file content into the LSP server (``update_document``).
    2. Wait briefly (~``debounce_ms``) for the server to publish diagnostics.
    3. Pull diagnostics for the edited file (``get_diagnostics``).
    4. Append errors to the tool result so the agent sees them inline.

    Design:
    - **Non-blocking**: any LSP failure is swallowed. The tool result is
      returned unchanged — never delays the agent.
    - **Scoped**: ``get_applicable_tools()`` limits it to file-modifying tools.
    - **Configurable**: ``mode="errors"`` (severity=1 only, default) or
      ``mode="all"`` (include severity=2 warnings).
    """

    def __init__(
        self,
        lsp_capability: Any = None,
        workspace: Optional[str] = None,
        mode: str = "errors",
        debounce_ms: int = 100,
    ):
        self._lsp = lsp_capability
        self._workspace = workspace
        self._mode = mode
        self._debounce_s = debounce_ms / 1000.0

    @property
    def lsp(self) -> Any:
        """The bound LSP capability (or None)."""
        return self._lsp

    @lsp.setter
    def lsp(self, capability: Any) -> None:
        """Bind (or rebind) an LSP capability."""
        self._lsp = capability

    # -- MiddlewareProtocol -------------------------------------------------

    def get_priority(self) -> MiddlewarePriority:
        """Run after most other middleware in the after-phase."""
        return MiddlewarePriority.LOW

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Only file-modifying tools need post-edit diagnostics."""
        return set(_FILE_MODIFYING_TOOLS)

    async def before_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> MiddlewareResult:
        """No-op (diagnostics are post-edit only)."""
        return MiddlewareResult()

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """Inject LSP diagnostics after a file-modifying tool succeeds."""
        if not success or self._lsp is None:
            return None

        # Defense-in-depth: the chain already filters via get_applicable_tools(),
        # but self-guard so direct calls (or a None-applicable chain) behave
        # correctly — never diagnose a non-file-modifying tool.
        applicable = self.get_applicable_tools()
        if applicable is not None and tool_name not in applicable:
            return None

        file_path = arguments.get("path") or arguments.get("file_path")
        if not file_path:
            return None

        diagnostics = await self._fetch_diagnostics(str(file_path))
        if not diagnostics:
            return None

        diagnostic_text = self._format_diagnostics(diagnostics)
        if not diagnostic_text:
            return None
        return self._append_to_result(result, diagnostic_text)

    # -- internals ----------------------------------------------------------

    async def _fetch_diagnostics(self, file_path: str) -> list:
        """Push the new content into the LSP server and pull diagnostics."""
        try:
            content = self._read_file(file_path)
            if content is not None:
                await self._safe_call(self._lsp.update_document(file_path, content))
                # Give the server a moment to publish updated diagnostics.
                await asyncio.sleep(self._debounce_s)
            diags = self._safe_call_sync(self._lsp.get_diagnostics, file_path)
            return diags or []
        except Exception:
            logger.debug("LSP diagnostics fetch failed for %s", file_path, exc_info=True)
            return []

    def _format_diagnostics(self, diagnostics: list) -> str:
        """Format diagnostics into a human-readable block for the agent."""
        errors: list[str] = []
        warnings: list[str] = []
        for diag in diagnostics:
            severity = getattr(diag, "severity", 1)
            msg = getattr(diag, "message", str(diag))
            rng = getattr(diag, "range", None)
            start = getattr(rng, "start", None) if rng else None
            line = getattr(start, "line", "?") if start else "?"
            entry = f"  Line {line}: {msg}"
            if severity == 1:
                errors.append(entry)
            elif severity == 2:
                warnings.append(entry)

        parts: list[str] = []
        if errors:
            parts.append(f"⚠ LSP errors ({len(errors)}):\n" + "\n".join(errors))
        if self._mode == "all" and warnings:
            parts.append(f"LSP warnings ({len(warnings)}):\n" + "\n".join(warnings))
        return "\n".join(parts)

    @staticmethod
    def _read_file(file_path: str) -> Optional[str]:
        """Read the current file content (for the LSP document update)."""
        try:
            from pathlib import Path

            return Path(file_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

    @staticmethod
    def _append_to_result(result: Any, diagnostic_text: str) -> Any:
        """Append diagnostic text to the tool result, preserving its shape."""
        if isinstance(result, dict):
            existing = result.get("output", "") or result.get("result", "") or ""
            result["output"] = f"{existing}\n\n{diagnostic_text}" if existing else diagnostic_text
            if "lsp_diagnostics" not in result:
                result["lsp_diagnostics"] = diagnostic_text
            return result
        if isinstance(result, str):
            return f"{result}\n\n{diagnostic_text}"
        return result

    async def _safe_call(self, coro: Any) -> None:
        """Await a coroutine, swallowing exceptions."""
        try:
            await coro
        except Exception:
            pass

    @staticmethod
    def _safe_call_sync(fn: Any, *args: Any) -> Any:
        """Call a sync function, swallowing exceptions."""
        try:
            return fn(*args)
        except Exception:
            return []
