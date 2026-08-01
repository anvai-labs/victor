"""Size-capping / body-split — the discipline ProximaDB's ``code.py`` lacked.

A symbol within budget yields one chunk. An oversized symbol is split into
line-aligned, overlapping sub-chunks (LlamaIndex ``CodeSplitter`` style: respect
structure, but never exceed ``max_chunk_chars``). Sub-chunks share the parent
``symbol_id`` and carry hierarchical, deterministic ``chunk_id``s (Victor's pattern).
"""

from __future__ import annotations

from .config import ChunkConfig
from .model import CodeChunk, CodeSymbol


def _base_metadata(symbol: CodeSymbol) -> dict:
    return {
        "symbol_id": symbol.id,
        "symbol_type": symbol.symbol_type.name,
        "fully_qualified_name": symbol.fully_qualified_name,
        "simple_name": symbol.simple_name,
        "language": symbol.language,
        "file_path": symbol.location.file_path,
        "start_line": symbol.location.start_line,
        "end_line": symbol.location.end_line,
        "signature": symbol.signature,
        "documentation": symbol.documentation,
        "modifiers": list(symbol.modifiers),
        "scope_chain": list(symbol.scope_chain),
        "return_type": symbol.return_type,
        "complexity": symbol.complexity,
    }


def chunks_for_symbol(symbol: CodeSymbol, config: ChunkConfig) -> list[CodeChunk]:
    """Project one symbol into one or more size-capped chunks."""

    source = symbol.source_code
    line_count = symbol.location.end_line - symbol.location.start_line + 1
    fits = len(source) <= config.max_chunk_chars
    small = line_count <= config.large_symbol_threshold_lines

    if fits or small:
        # Whole symbol as a single chunk. If a *small* symbol is still over the char
        # budget (rare: dense one-liners), we still cap it below.
        if fits:
            meta = _base_metadata(symbol)
            meta["chunk_index"] = 0
            meta["chunk_total"] = 1
            return [
                CodeChunk(
                    chunk_id=f"{symbol.id}#0",
                    text=source,
                    symbol_id=symbol.id,
                    start_pos=symbol.location.byte_offset,
                    end_pos=symbol.location.byte_offset + len(source.encode("utf-8")),
                    metadata=meta,
                )
            ]

    return _body_split(symbol, config)


def _body_split(symbol: CodeSymbol, config: ChunkConfig) -> list[CodeChunk]:
    """Split an oversized symbol body into overlapping, line-aligned sub-chunks."""

    lines = symbol.source_code.splitlines(keepends=True)
    max_chars = config.max_chunk_chars
    overlap_chars = config.chunk_overlap_chars

    windows: list[tuple[int, str, bool]] = []  # (start_line_offset, text, is_line_split)
    cur: list[str] = []
    cur_len = 0
    cur_start = 0
    i = 0
    while i < len(lines):
        ln = lines[i]
        # A single line longer than the budget (degenerate minified case) is split
        # into overlapping char windows — every char lands in some chunk, none are
        # silently dropped (the old code truncated to ``ln[:max_chars]``), and it
        # never rides inside a line window where it would blow the budget.
        # ChunkConfig guarantees overlap < max, so step >= 1.
        if len(ln) > max_chars:
            if cur:
                windows.append((cur_start, "".join(cur), False))
                cur = []
                cur_len = 0
            step = max(1, max_chars - overlap_chars)
            for off in range(0, len(ln), step):
                windows.append((i, ln[off : off + max_chars], True))
                if off + max_chars >= len(ln):
                    break
            i += 1
            cur_start = i
            continue
        if cur_len + len(ln) > max_chars and cur:
            windows.append((cur_start, "".join(cur), False))
            # Build overlap tail by walking back from the end of the current window.
            tail: list[str] = []
            tail_len = 0
            j = i - 1
            while j >= cur_start and tail_len + len(lines[j]) <= overlap_chars:
                tail.insert(0, lines[j])
                tail_len += len(lines[j])
                j -= 1
            cur = list(tail)
            cur_len = tail_len
            cur_start = j + 1
        cur.append(ln)
        cur_len += len(ln)
        i += 1
    if cur:
        windows.append((cur_start, "".join(cur), False))

    total = len(windows)
    out: list[CodeChunk] = []
    base_line = symbol.location.start_line
    for idx, (line_off, text, is_line_split) in enumerate(windows):
        meta = _base_metadata(symbol)
        meta["chunk_index"] = idx
        meta["chunk_total"] = total
        meta["is_body_split"] = True
        meta["start_line"] = base_line + line_off
        # The last line of THIS window, not the whole symbol's end_line.
        spanned = text.count("\n") + (0 if text.endswith("\n") else 1)
        meta["end_line"] = base_line + line_off + max(spanned - 1, 0)
        if is_line_split:
            meta["is_line_split"] = True
        out.append(
            CodeChunk(
                chunk_id=f"{symbol.id}#body#{idx}",
                text=text,
                symbol_id=symbol.id,
                start_pos=symbol.location.byte_offset,
                end_pos=symbol.location.byte_offset + len(symbol.source_code.encode("utf-8")),
                metadata=meta,
            )
        )
    return out
