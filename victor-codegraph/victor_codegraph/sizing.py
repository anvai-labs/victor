"""Size-capping / body-split — the discipline ProximaDB's ``code.py`` lacked.

A symbol within budget yields one chunk. An oversized symbol is split into
line-aligned, overlapping sub-chunks (LlamaIndex ``CodeSplitter`` style: respect
structure, but never exceed ``max_chunk_chars``). Sub-chunks share the parent
``symbol_id`` and carry hierarchical, deterministic ``chunk_id``s (Victor's pattern).
"""

from __future__ import annotations

from .config import ChunkConfig
from .model import CodeChunk, CodeSymbol


def split_text_windows(text: str, config: ChunkConfig) -> list[tuple[int, int]]:
    """Return overlapping, newline-preferred windows satisfying the hard budget."""

    if not text:
        return []
    windows: list[tuple[int, int]] = []
    start = 0
    size = len(text)
    while start < size:
        hard_end = min(size, start + config.max_chunk_chars)
        end = hard_end

        if (
            config.token_counter is not None
            and config.estimate_tokens(text[start:end]) > config.max_chunk_tokens
        ):
            lo, hi = start + 1, end
            best = start
            while lo <= hi:
                mid = (lo + hi) // 2
                if config.estimate_tokens(text[start:mid]) <= config.max_chunk_tokens:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best == start:
                raise ValueError("token counter cannot fit even one character in the chunk budget")
            end = best

        if end < size:
            newline = text.rfind("\n", start, end)
            if newline >= start:
                candidate = newline + 1
                if candidate > start:
                    end = candidate

        windows.append((start, end))
        if end >= size:
            break

        if config.chunk_overlap_chars:
            floor = max(start + 1, end - config.chunk_overlap_chars)
            newline = text.find("\n", floor, end)
            next_start = newline + 1 if newline >= 0 else floor
        else:
            next_start = end
        start = max(start + 1, min(next_start, end))
    return windows


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
    fits = len(source) <= config.max_chunk_chars and (
        config.token_counter is None or config.estimate_tokens(source) <= config.max_chunk_tokens
    )
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

    source = symbol.source_code
    windows = split_text_windows(source, config)
    total = len(windows)
    out: list[CodeChunk] = []
    base_line = symbol.location.start_line
    for idx, (char_start, char_end) in enumerate(windows):
        text = source[char_start:char_end]
        line_off = source.count("\n", 0, char_start)
        line_start_char = source.rfind("\n", 0, char_start) + 1
        is_line_split = char_start != line_start_char or (
            char_end < len(source) and source[char_end - 1 : char_end] != "\n"
        )
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
                start_pos=symbol.location.byte_offset + len(source[:char_start].encode("utf-8")),
                end_pos=symbol.location.byte_offset + len(source[:char_end].encode("utf-8")),
                metadata=meta,
            )
        )
    return out
