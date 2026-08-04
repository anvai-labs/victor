"""Soft adapter from Victor chunking contracts to victor-codegraph v2."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectedCodeChunk:
    """Consumer-neutral projection of a canonical v2 code chunk."""

    content: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    chunk_type: str
    symbol_name: str | None = None
    parent_symbol: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


_LANGUAGE_ALIASES = {
    "csharp": "c_sharp",
    "cs": "c_sharp",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "rs": "rust",
}


def _byte_to_char_offset(content_bytes: bytes, offset: int) -> int:
    bounded = max(0, min(int(offset), len(content_bytes)))
    return len(content_bytes[:bounded].decode("utf-8", errors="ignore"))


def chunk_with_codegraph(
    content: str,
    *,
    file_path: str,
    language: str | None,
    max_chunk_size: int,
    chunk_overlap: int,
    token_counter: Callable[[str], int] | None = None,
    chars_per_token: float = 1.0,
) -> list[ProjectedCodeChunk]:
    """Return canonical chunks with one bounded raw-text fallback.

    ``max_chunk_size`` and ``chunk_overlap`` are interpreted in the units of
    ``token_counter`` when supplied, otherwise as characters.
    """
    try:
        import victor_codegraph as codegraph

        config = codegraph.ChunkConfig(
            max_chunk_tokens=max(1, int(max_chunk_size)),
            chunk_overlap_tokens=max(0, int(chunk_overlap)),
            chars_per_token=chars_per_token,
            token_counter=token_counter,
        )
        canonical = codegraph.chunk(
            content,
            language=_LANGUAGE_ALIASES.get(language or "", language),
            file_path=file_path,
            config=config,
        )
    except Exception as exc:
        logger.debug("victor-codegraph chunk planner unavailable for %s: %s", file_path, exc)
        return _fallback_chunks(
            content,
            file_path=file_path,
            language=language,
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            token_counter=token_counter,
            chars_per_token=chars_per_token,
        )

    source_bytes = content.encode("utf-8")
    projected: list[ProjectedCodeChunk] = []
    for item in canonical:
        metadata = dict(item.metadata)
        symbol_type = str(metadata.get("symbol_type") or "module").lower()
        strategy = str(metadata.get("strategy") or "symbol")
        if strategy in {"module_context", "sliding_window"}:
            symbol_type = "module"
        scope_chain = list(metadata.get("scope_chain") or [])
        projected.append(
            ProjectedCodeChunk(
                content=item.text,
                start_char=_byte_to_char_offset(source_bytes, item.start_pos),
                end_char=_byte_to_char_offset(source_bytes, item.end_pos),
                start_line=int(metadata.get("start_line") or 1),
                end_line=int(metadata.get("end_line") or metadata.get("start_line") or 1),
                chunk_type=symbol_type,
                symbol_name=metadata.get("simple_name") or metadata.get("fully_qualified_name"),
                parent_symbol=scope_chain[-1] if scope_chain else None,
                metadata={
                    **metadata,
                    "chunk_id": item.chunk_id,
                    "symbol_id": item.symbol_id,
                    "codegraph_strategy": strategy,
                    "strategy": "victor_codegraph_v2",
                },
            )
        )
    return projected


def _fallback_chunks(
    content: str,
    *,
    file_path: str,
    language: str | None,
    max_chunk_size: int,
    chunk_overlap: int,
    token_counter: Callable[[str], int] | None,
    chars_per_token: float,
) -> list[ProjectedCodeChunk]:
    """Deterministic newline-preferred windows for dependency failure only."""
    if not content:
        return []
    hard_chars = max(1, int(max_chunk_size * chars_per_token))
    overlap_chars = min(max(0, int(chunk_overlap * chars_per_token)), hard_chars - 1)
    chunks: list[ProjectedCodeChunk] = []
    start = 0
    while start < len(content):
        end = min(len(content), start + hard_chars)
        if token_counter is not None and token_counter(content[start:end]) > max_chunk_size:
            lo, hi = start + 1, end
            best = start
            while lo <= hi:
                mid = (lo + hi) // 2
                if token_counter(content[start:mid]) <= max_chunk_size:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            end = max(start + 1, best)
        if end < len(content):
            newline = content.rfind("\n", start, end)
            if newline >= start:
                end = newline + 1
        text = content[start:end]
        start_line = content.count("\n", 0, start) + 1
        end_line = start_line + text.count("\n") - (1 if text.endswith("\n") else 0)
        chunks.append(
            ProjectedCodeChunk(
                content=text,
                start_char=start,
                end_char=end,
                start_line=start_line,
                end_line=max(start_line, end_line),
                chunk_type="module",
                metadata={
                    "chunk_id": f"{file_path}#fallback#{len(chunks)}",
                    "symbol_id": f"{file_path}#fallback#{len(chunks)}",
                    "file_path": file_path,
                    "language": language or "text",
                    "strategy": "plain_code_fallback",
                },
            )
        )
        if end >= len(content):
            break
        start = max(start + 1, end - overlap_chars)
    return chunks


__all__ = ["ProjectedCodeChunk", "chunk_with_codegraph"]
