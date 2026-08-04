"""Public entrypoints: ``parse`` (symbols+relations) and ``chunk`` (size-capped).

Fallback chain (Victor's posture): python-ast -> tree-sitter -> sliding-window. A parse
never hard-fails; an unknown or grammar-less language degrades to line-window chunks.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from .config import ChunkConfig
from .languages import detect_language
from .model import (
    CapabilityTier,
    CodeChunk,
    CodeRelationType,
    CodeSymbolType,
    ParseDiagnostic,
    ParsedCode,
    ParseStatus,
    content_hash,
    stable_symbol_key,
)
from .python_parser import parse_python
from .sizing import chunks_for_symbol, split_text_windows
from .treesitter_parser import GrammarUnavailable, parse_treesitter


def normalize_file_path(file_path: str, repo_root: str | os.PathLike[str] | None = None) -> str:
    """Return the canonical repo-relative POSIX path used by identity."""

    if file_path == "<unknown>":
        return file_path
    path = Path(file_path)
    if repo_root is not None:
        root = Path(repo_root)
        if root.is_file():
            return path.name
        try:
            path = path.resolve().relative_to(root.resolve())
        except (OSError, ValueError):
            path = Path(os.path.relpath(path, root))
    normalized = path.as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _finalize_identity(parsed: ParsedCode, content: str) -> ParsedCode:
    """Promote parser-local line IDs to the v2 structural identity contract."""

    id_map: dict[str, str] = {}
    seen: dict[str, str] = {}
    for symbol in parsed.symbols:
        legacy = symbol.legacy_id or symbol.id
        stable = stable_symbol_key(
            symbol.language,
            symbol.symbol_type,
            symbol.fully_qualified_name,
            symbol.signature,
            symbol.location.file_path,
        )
        other = seen.get(stable)
        if other is not None and other != legacy:
            parsed.status = ParseStatus.PARTIAL
            parsed.diagnostics.append(
                ParseDiagnostic(
                    code="identity_collision",
                    message=f"multiple symbols share structural key {stable}",
                    severity="error",
                    line=symbol.location.start_line,
                )
            )
        seen[stable] = legacy
        id_map[legacy] = stable
        symbol.legacy_id = legacy
        symbol.id = stable
        symbol.identity_version = "v2"

    for relation in parsed.relations:
        relation.from_symbol_id = id_map.get(relation.from_symbol_id, relation.from_symbol_id)
        relation.to_symbol_id = id_map.get(relation.to_symbol_id, relation.to_symbol_id)
    parsed.source_code = content
    return parsed


def parse(
    content: str,
    language: str | None = None,
    file_path: str = "<unknown>",
    *,
    repo_root: str | os.PathLike[str] | None = None,
    config: ChunkConfig | None = None,
) -> ParsedCode:
    """Parse source into symbols + relations, falling back gracefully."""

    config = config or ChunkConfig()
    file_path = normalize_file_path(file_path, repo_root)
    language = language or detect_language(file_path)

    if language == "python":
        try:
            parsed = _finalize_identity(parse_python(content, file_path), content)
            parsed.capability_tier = CapabilityTier.FULL
            if not config.extract_relations:
                parsed.relations = []
            return parsed
        except SyntaxError as exc:
            return ParsedCode(
                file_path=file_path,
                language="python",
                content_hash=content_hash(content),
                status=ParseStatus.FALLBACK,
                capability_tier=CapabilityTier.FALLBACK,
                diagnostics=[
                    ParseDiagnostic(
                        code="syntax_error",
                        message=str(exc),
                        line=exc.lineno,
                        column=exc.offset,
                    )
                ],
                source_code=content,
            )
        except Exception as exc:
            return ParsedCode(
                file_path=file_path,
                language="python",
                content_hash=content_hash(content),
                status=ParseStatus.ERROR,
                capability_tier=CapabilityTier.FALLBACK,
                diagnostics=[
                    ParseDiagnostic(code="parser_error", message=f"{type(exc).__name__}: {exc}")
                ],
                source_code=content,
            )

    if language is not None and language != "python":
        try:
            parsed = _finalize_identity(parse_treesitter(content, file_path, language), content)
            if not config.extract_relations:
                parsed.relations = []
            return parsed
        except GrammarUnavailable as exc:
            return ParsedCode(
                file_path=file_path,
                language=language,
                content_hash=content_hash(content),
                status=ParseStatus.FALLBACK,
                capability_tier=CapabilityTier.FALLBACK,
                diagnostics=[ParseDiagnostic(code="grammar_unavailable", message=str(exc))],
                source_code=content,
            )
        except Exception as exc:
            return ParsedCode(
                file_path=file_path,
                language=language,
                content_hash=content_hash(content),
                status=ParseStatus.ERROR,
                capability_tier=CapabilityTier.FALLBACK,
                diagnostics=[
                    ParseDiagnostic(code="parser_error", message=f"{type(exc).__name__}: {exc}")
                ],
                source_code=content,
            )

    # Last resort: no symbols (caller's chunk() will sliding-window the raw text).
    return ParsedCode(
        file_path=file_path,
        language=language or "text",
        symbols=[],
        relations=[],
        imports=[],
        content_hash=content_hash(content),
        status=ParseStatus.FALLBACK,
        capability_tier=CapabilityTier.FALLBACK,
        diagnostics=[
            ParseDiagnostic(
                code="unsupported_language",
                message=f"no parser frontend for language {language or 'text'}",
            )
        ],
        source_code=content,
    )


def _sliding_window(
    content: str,
    file_path: str,
    language: str,
    config: ChunkConfig,
    *,
    base_byte: int = 0,
    base_line: int = 1,
    strategy: str = "sliding_window",
) -> list[CodeChunk]:
    """Universal fallback when no symbols were extracted."""

    if not content:
        return []
    out: list[CodeChunk] = []
    for idx, (start, end) in enumerate(split_text_windows(content, config)):
        text = content[start:end]
        start_line = base_line + content.count("\n", 0, start)
        end_line = start_line + text.count("\n") - (1 if text.endswith("\n") else 0)
        start_pos = base_byte + len(content[:start].encode("utf-8"))
        out.append(
            CodeChunk(
                chunk_id=f"{file_path}#window#{base_byte}#{idx}",
                text=text,
                symbol_id=f"{file_path}#window#{idx}",
                start_pos=start_pos,
                end_pos=start_pos + len(text.encode("utf-8")),
                metadata={
                    "file_path": file_path,
                    "language": language,
                    "chunk_index": idx,
                    "start_line": start_line,
                    "end_line": max(start_line, end_line),
                    "strategy": strategy,
                },
            )
        )
    return out


def chunk(
    content: str,
    language: str | None = None,
    file_path: str = "<unknown>",
    config: ChunkConfig | None = None,
) -> list[CodeChunk]:
    """Parse + project into size-capped, embeddable chunks."""

    config = config or ChunkConfig()
    detected = language or detect_language(file_path)
    if config.languages is not None and detected not in set(config.languages):
        return []
    basename = Path(file_path).name.lower()
    if not config.include_tests and (
        basename.startswith("test_")
        or "_test." in basename
        or ".test." in basename
        or ".spec." in basename
    ):
        return []
    parsed = parse(content, language, file_path, config=config)

    if not parsed.symbols:
        return _sliding_window(content, file_path, parsed.language, config)

    out: list[CodeChunk] = []
    container_ids = {
        relation.from_symbol_id
        for relation in parsed.relations
        if relation.relation_type == CodeRelationType.CONTAINS
    }

    def container_summary(symbol):
        source = symbol.source_code
        if symbol.symbol_type not in {
            CodeSymbolType.CLASS,
            CodeSymbolType.INTERFACE,
            CodeSymbolType.TRAIT,
            CodeSymbolType.STRUCT,
            CodeSymbolType.ENUM,
        }:
            return symbol
        candidates = [pos for pos in (source.find(":\n"), source.find("{")) if pos >= 0]
        if not candidates:
            return symbol
        end = min(candidates) + 1
        summary = source[:end]
        return replace(
            symbol,
            source_code=summary,
            location=replace(
                symbol.location,
                end_line=symbol.location.start_line + summary.count("\n"),
                byte_length=len(summary.encode("utf-8")),
            ),
        )

    for sym in parsed.symbols:
        if not config.include_private and "private" in sym.modifiers:
            continue
        projected = container_summary(sym) if sym.id in container_ids else sym
        out.extend(chunks_for_symbol(projected, config))

    # Preserve module-level imports, declarations, initialization, and executable code.
    # Top-level definitions are removed from this context because their leaf symbols are
    # emitted separately; container summaries avoid duplicating member bodies.
    source_bytes = content.encode("utf-8")
    intervals = sorted(
        (
            max(0, symbol.location.byte_offset),
            min(len(source_bytes), symbol.location.byte_offset + symbol.location.byte_length),
        )
        for symbol in parsed.symbols
        if not symbol.scope_chain and symbol.location.byte_length > 0
    )
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    cursor = 0
    for start, end in [*merged, (len(source_bytes), len(source_bytes))]:
        if start > cursor:
            gap_bytes = source_bytes[cursor:start]
            gap = gap_bytes.decode("utf-8", errors="replace")
            if gap.strip():
                prefix = source_bytes[:cursor].decode("utf-8", errors="replace")
                out.extend(
                    _sliding_window(
                        gap,
                        file_path,
                        parsed.language,
                        config,
                        base_byte=cursor,
                        base_line=prefix.count("\n") + 1,
                        strategy="module_context",
                    )
                )
        cursor = max(cursor, end)
    return sorted(out, key=lambda item: (item.start_pos, item.end_pos, item.chunk_id))
