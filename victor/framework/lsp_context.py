# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""LSP context provider for proactive generation guidance (FEP-0019 Phase 3).

Before the agent generates/edits code, this builds a compact, position-free
context block: the **live document symbols (signatures/types/API surface)** +
**current diagnostics** of the file(s) it is editing. The agentic loop injects
this as a per-turn user message so code matches the existing codebase on the
*first* try — fewer edit→diagnose→fix cycles.

Position-free: uses file-level LSP queries only (``get_document_symbols`` +
``get_diagnostics``), so no cursor/position tracking is required. Target files
are resolved via ``workspace_files_modified()`` (git diff vs HEAD).

Throttled by a content signature: when the symbol/error set for the active
files is unchanged since the last injection, ``build_context`` returns
``(None, signature)`` so the loop skips re-injecting identical context.

Every step is None-safe: no LSP capability, no impl, no modified files, or no
errors/symbols all yield ``(None, signature)`` — the loop simply skips. Zero
behavior change when LSP is unavailable (victor-coding not installed).
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

from victor.framework.workspace import workspace_files_modified

logger = logging.getLogger(__name__)

# Source-code extensions worth pulling LSP symbols for (excludes data/config/build).
_SOURCE_EXTENSIONS = frozenset(
    {
        ".py",
        ".js",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".jsx",
        ".rs",
        ".go",
        ".java",
        ".kt",
        ".kts",
        ".scala",
        ".sc",
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hh",
        ".hpp",
        ".hxx",
        ".cs",
        ".rb",
        ".swift",
        ".lua",
        ".php",
        ".vue",
        ".svelte",
        ".dart",
        ".ex",
        ".exs",
        ".clj",
        ".lisp",
        ".el",
    }
)

# LSP SymbolKind int → short label (subset that matters for code generation).
# https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#symbolKind
_SYMBOL_KIND_LABELS = {
    1: "",  # File
    2: "module",  # Module
    3: "namespace",  # Namespace
    4: "package",  # Package
    5: "class",  # Class
    6: "method",  # Method
    7: "property",  # Property
    8: "field",  # Field
    9: "constructor",  # Constructor
    10: "enum",  # Enum
    11: "interface",  # Interface
    12: "function",  # Function
    13: "variable",  # Variable
    14: "constant",  # Constant
    23: "struct",  # Struct
    24: "event",  # Event
    26: "type",  # TypeParameter
}

# Per-file caps (keep the block compact + budgeted).
_MAX_SYMBOLS_PER_FILE = 12
_MAX_CHILDREN_PER_SYMBOL = 5
_MAX_DIAGNOSTICS_PER_FILE = 8


async def build_context(
    lsp: Any,
    workspace: Optional[Path],
    *,
    max_files: int = 2,
    char_budget: int = 2500,
    last_signature: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Build a compact LSP context block for the files being edited.

    Args:
        lsp: An LSP capability/service (duck-typed; needs ``get_document_symbols``
            and ``get_diagnostics``). None → ``(None, None)``.
        workspace: Workspace root for resolving modified files + absolute paths.
            None → ``(None, None)``.
        max_files: Max number of files to include.
        char_budget: Soft cap on rendered block size.
        last_signature: Signature from the previous injection; if the current
            symbol/error set matches, returns ``(None, signature)`` (throttle).

    Returns:
        ``(block, signature)``. ``block`` is None when there is nothing new to
        inject (no LSP, no modified source files, no symbols/errors, or
        unchanged since ``last_signature``). ``signature`` is the current
        content signature (or None when nothing was gathered).
    """
    if lsp is None or workspace is None:
        return None, None

    try:
        modified = await workspace_files_modified(workspace)
    except Exception:
        logger.debug("workspace_files_modified failed", exc_info=True)
        return None, None

    files = _select_files(modified, workspace, max_files)
    if not files:
        return None, None

    gathered = await _gather(lsp, files)
    # Drop files that yielded neither symbols nor errors.
    gathered = {rel: (syms, diags) for rel, (syms, diags) in gathered.items() if syms or diags}
    if not gathered:
        return None, None

    signature = _signature(gathered)
    if last_signature is not None and signature == last_signature:
        return None, signature

    block = _render(gathered, char_budget)
    if not block:
        return None, signature
    return block, signature


def _select_files(modified: List[str], workspace: Path, max_files: int) -> List[Tuple[str, Path]]:
    """Pick the most-recently-modified source files, as (relpath, abspath)."""
    candidates: List[Tuple[float, str, Path]] = []
    for rel in modified:
        rel = rel.strip()
        if not rel or os.path.isabs(rel):
            continue
        if Path(rel).suffix.lower() not in _SOURCE_EXTENSIONS:
            continue
        abs_path = workspace / rel
        try:
            if not abs_path.is_file():
                continue
            mtime = abs_path.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, rel, abs_path))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [(rel, abs_path) for _, rel, abs_path in candidates[:max_files]]


async def _gather(lsp: Any, files: List[Tuple[str, Path]]) -> dict:
    """Pull symbols + diagnostics for each file. Swallows per-file errors."""
    gathered: dict = {}
    for rel, abs_path in files:
        symbols: list = []
        diagnostics: list = []
        try:
            get_symbols = getattr(lsp, "get_document_symbols", None)
            if get_symbols is not None:
                result = get_symbols(str(abs_path))
                symbols = list(await result) if _is_awaitable(result) else list(result)
        except Exception:
            logger.debug("get_document_symbols failed for %s", rel, exc_info=True)
            symbols = []
        try:
            get_diags = getattr(lsp, "get_diagnostics", None)
            if get_diags is not None:
                result = get_diags(str(abs_path))
                diagnostics = list(await result) if _is_awaitable(result) else list(result)
        except Exception:
            logger.debug("get_diagnostics failed for %s", rel, exc_info=True)
            diagnostics = []
        # Only severity-1 (errors) are rendered; filter once here so the
        # keep-filter, signature, and renderer all agree.
        diagnostics = [d for d in diagnostics if getattr(d, "severity", 1) == 1]
        gathered[rel] = (symbols, diagnostics)
    return gathered


def _is_awaitable(value: Any) -> bool:
    import inspect

    return inspect.isawaitable(value)


def _signature(gathered: dict) -> str:
    """Stable signature of the gathered symbol/error set (for throttling)."""
    parts: list = []
    for rel in sorted(gathered):
        symbols, diagnostics = gathered[rel]
        names = sorted(getattr(s, "name", "") for s in symbols)
        err_count = sum(1 for d in diagnostics if getattr(d, "severity", 1) == 1)
        parts.append(f"{rel}|{','.join(names)}|{err_count}")
    return hashlib.md5("\n".join(parts).encode("utf-8")).hexdigest()


def _render(gathered: dict, char_budget: int) -> str:
    """Render the gathered data into a compact, budgeted context block.

    Per-file caps (``_MAX_SYMBOLS_PER_FILE`` etc.) bound the content; the char
    budget is a clean final safety truncation.
    """
    lines: list = ["[LSP context — live signatures + errors for files you are editing]"]
    for rel in sorted(gathered):
        symbols, diagnostics = gathered[rel]
        lines.append(rel)
        lines.extend(_render_symbols(symbols))
        lines.extend(_render_diagnostics(diagnostics))
    if len(lines) <= 1:
        return ""
    block = "\n".join(lines)
    if len(block) > char_budget:
        block = block[: char_budget - 1].rstrip() + "…"
    return block


def _render_symbols(symbols: list) -> List[str]:
    """Render top-level symbols + one level of children."""
    out: list = []
    for sym in symbols[:_MAX_SYMBOLS_PER_FILE]:
        out.append(_format_symbol(sym, indent=1))
        children = getattr(sym, "children", None) or []
        for child in children[:_MAX_CHILDREN_PER_SYMBOL]:
            out.append(_format_symbol(child, indent=2))
    if len(symbols) > _MAX_SYMBOLS_PER_FILE:
        out.append(f"  …({len(symbols) - _MAX_SYMBOLS_PER_FILE} more)")
    return out


def _format_symbol(symbol: Any, indent: int) -> str:
    """Format one symbol as e.g. '  class SessionManager'."""
    pad = "  " * indent
    kind = _SYMBOL_KIND_LABELS.get(getattr(symbol, "kind", 0), "")
    name = getattr(symbol, "name", "?")
    detail = getattr(symbol, "detail", None)
    label = f"{kind} {name}".strip()
    if detail:
        detail = detail.strip().replace("\n", " ")
        if detail and len(detail) <= 80:
            label = f"{label}  — {detail}"
        elif detail:
            label = f"{label}  — {detail[:77]}…"
    return f"{pad}{label}"


def _render_diagnostics(diagnostics: list) -> List[str]:
    """Render severity-1 errors as '! Lnn: message'."""
    out: list = []
    errors = [d for d in diagnostics if getattr(d, "severity", 1) == 1]
    for diag in errors[:_MAX_DIAGNOSTICS_PER_FILE]:
        rng = getattr(diag, "range", None)
        start = getattr(rng, "start", None) if rng else None
        line = getattr(start, "line", None) if start else None
        msg = (getattr(diag, "message", "") or "").strip().replace("\n", " ")
        where = f"L{line + 1}" if isinstance(line, int) else "?"
        out.append(f"  ! {where}: {msg}")
    if len(errors) > _MAX_DIAGNOSTICS_PER_FILE:
        out.append(f"  ! …({len(errors) - _MAX_DIAGNOSTICS_PER_FILE} more errors)")
    return out
