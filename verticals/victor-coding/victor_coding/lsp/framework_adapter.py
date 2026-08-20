# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Framework LSP adapter — activates FEP-0019 in victor-coding.

Bridges victor-coding's :class:`LSPConnectionPool` to the LSP shape the FEP-0019
framework runtime consumes (``open_document`` / ``update_document`` /
``get_diagnostics`` / ``get_document_symbols``). The framework duck-types
``orchestrator.lsp`` (its tests use plain namespaces), so this adapter returns
``victor_contracts.lsp_runtime`` types — the same types victor-coding already
uses — and is returned directly from :meth:`CodingAssistant.get_lsp`. No
``victor.framework`` imports: verticals depend on ``victor_contracts`` only
(the monorepo extractability boundary).

Activates:
* Phase 1 — ``LSPVerifier`` (post-COMPLETE diagnostics gate)
* Phase 2 — ``LSPDiagnosticMiddleware`` (same-turn post-edit diagnostics)
* Phase 3 — ``LSPContextProvider`` (proactive document-symbol context)

The framework's ``FrameworkStepHandler.apply_lsp`` routes the object returned by
``get_lsp`` to ``orchestrator.set_lsp`` during integration.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from victor_contracts.lsp_runtime import (
    Diagnostic,
    DiagnosticSeverity,
    DocumentSymbol,
    Position,
    Range,
)

logger = logging.getLogger(__name__)

# The pool renders severity as a display name; map back to LSP severity enums.
# https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#diagnosticSeverity
_SEVERITY_NAME_TO_ENUM = {
    "error": DiagnosticSeverity.ERROR,
    "warning": DiagnosticSeverity.WARNING,
    "info": DiagnosticSeverity.INFORMATION,
    "hint": DiagnosticSeverity.HINT,
}


class FrameworkLSPAdapter:
    """Adapt ``LSPConnectionPool`` to the FEP-0019 LSP runtime contract.

    Implements the methods the FEP-0019 runtime consumes:
    ``open_document``, ``update_document``, ``close_document``,
    ``get_diagnostics``, ``get_document_symbols``.
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    # --- document lifecycle -------------------------------------------------

    async def open_document(self, file_path: str, content: Optional[str] = None) -> bool:
        try:
            return bool(await self._pool.open_document(file_path, content))
        except Exception:
            logger.debug("LSP open_document failed for %s", file_path, exc_info=True)
            return False

    async def update_document(self, file_path: str, content: str) -> bool:
        try:
            return bool(await self._pool.update_document(file_path, content))
        except Exception:
            logger.debug("LSP update_document failed for %s", file_path, exc_info=True)
            return False

    def close_document(self, file_path: str) -> None:
        try:
            self._pool.close_document(file_path)
        except Exception:
            logger.debug("LSP close_document failed for %s", file_path, exc_info=True)

    # --- diagnostics (Phase 1 + 2) ------------------------------------------

    def get_diagnostics(self, file_path: str) -> List[Diagnostic]:
        """Convert the pool's display dicts to ``Diagnostic`` objects.

        The pool returns ``[{line (1-indexed), character, message, severity
        (name), source, code}]``; the framework reads ``severity`` (int-like),
        ``message`` and ``range.start.line`` (0-indexed).
        """
        try:
            raw = self._pool.get_diagnostics(file_path) or []
        except Exception:
            logger.debug("LSP get_diagnostics failed for %s", file_path, exc_info=True)
            return []
        out: List[Diagnostic] = []
        for d in raw:
            line = max(0, int(d.get("line") or 1) - 1)  # pool is 1-indexed
            character = int(d.get("character") or 0)
            rng = Range(
                start=Position(line=line, character=character),
                end=Position(line=line, character=character),
            )
            code = d.get("code")
            out.append(
                Diagnostic(
                    range=rng,
                    message=str(d.get("message") or ""),
                    severity=_SEVERITY_NAME_TO_ENUM.get(
                        str(d.get("severity")).lower(), DiagnosticSeverity.ERROR
                    ),
                    source=d.get("source"),
                    code=None if code is None else str(code),
                )
            )
        return out

    # --- document symbols (Phase 3) ----------------------------------------

    async def get_document_symbols(self, file_path: str) -> List[DocumentSymbol]:
        """Return the pool's document-symbol tree as-is (victor_contracts types).

        The framework reads ``name`` / ``kind`` (int-like) / ``range`` /
        ``children`` / ``detail`` via duck-typing, all present on
        ``DocumentSymbol``.
        """
        try:
            symbols = await self._pool.get_document_symbols(file_path)
        except Exception:
            logger.debug("LSP get_document_symbols failed for %s", file_path, exc_info=True)
            return []
        return list(symbols or [])
