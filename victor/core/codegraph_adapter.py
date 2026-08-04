"""One soft boundary from Victor contracts to victor-codegraph v2."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
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


@dataclass(frozen=True)
class CodeImport:
    """Small structured projection of a canonical Python import statement."""

    module: str
    names: tuple[str, ...] = ()
    alias: str | None = None
    is_from_import: bool = False


_LANGUAGE_ALIASES = {
    "csharp": "c_sharp",
    "cs": "c_sharp",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "rs": "rust",
}


def normalize_code_language(language: str | None) -> str | None:
    """Normalize consumer aliases to the canonical package vocabulary."""
    return _LANGUAGE_ALIASES.get(language or "", language)


def parse_code(
    content: str,
    *,
    file_path: str = "<unknown>",
    language: str | None = None,
    repo_root: str | Path | None = None,
) -> Any | None:
    """Parse through victor-codegraph, returning ``None`` at the soft boundary."""
    try:
        import victor_codegraph as codegraph

        return codegraph.parse(
            content,
            language=normalize_code_language(language),
            file_path=file_path,
            repo_root=repo_root,
        )
    except Exception as exc:
        logger.debug("victor-codegraph parser unavailable for %s: %s", file_path, exc)
        return None


def codegraph_available() -> bool:
    """Return whether the canonical parser package can be imported."""
    try:
        import victor_codegraph  # noqa: F401
    except ImportError:
        return False
    return True


def codegraph_file_id(file_path: str, language: str) -> str | None:
    """Return the canonical v2 file identity when the package is available."""
    try:
        import victor_codegraph as codegraph

        return codegraph.stable_symbol_key(
            language,
            codegraph.CodeSymbolType.FILE,
            file_path,
            None,
            file_path,
        )
    except Exception as exc:
        logger.debug("victor-codegraph file identity unavailable for %s: %s", file_path, exc)
        return None


def project_python_imports(statements: list[str]) -> list[CodeImport]:
    """Project CodeGraph's canonical Python import text without reparsing source."""
    projected: list[CodeImport] = []
    for statement in statements:
        text = statement.strip()
        if text.startswith("from ") and " import " in text:
            module, imported = text[5:].split(" import ", 1)
            names = tuple(
                item.strip().split(" as ", 1)[0]
                for item in imported.strip("() ").split(",")
                if item.strip()
            )
            projected.append(CodeImport(module=module.strip(), names=names, is_from_import=True))
            continue
        if text.startswith("import "):
            for imported in text[7:].split(","):
                parts = imported.strip().split(" as ", 1)
                if parts[0]:
                    projected.append(
                        CodeImport(
                            module=parts[0],
                            alias=parts[1] if len(parts) == 2 else None,
                        )
                    )
    return projected


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
            language=normalize_code_language(language),
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


__all__ = [
    "CodeImport",
    "ProjectedCodeChunk",
    "chunk_with_codegraph",
    "codegraph_available",
    "codegraph_file_id",
    "normalize_code_language",
    "parse_code",
    "project_python_imports",
]
