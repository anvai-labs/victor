"""Repo-walking convenience — iterate source files and chunk/parse a whole tree.

Every consumer (Victor codebase indexing, AnvaiOps code-graph-sync) needs the same
loop: walk a repository, skip noise directories, and chunk/parse each source file
whose extension maps to a known language. This puts that loop in one place.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from .config import ChunkConfig
from .languages import detect_language
from .model import CodeChunk, ParsedCode
from .parser import chunk as _chunk
from .parser import parse as _parse

# Directories never worth indexing (VCS, caches, vendored deps, build output).
DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
        "target",
        ".idea",
        ".gradle",
        ".next",
        "site-packages",
        "vendor",
    }
)


def iter_source_files(
    root: str | os.PathLike[str],
    *,
    languages: list[str] | None = None,
    exclude_dirs: frozenset[str] | set[str] = DEFAULT_EXCLUDE_DIRS,
    follow_symlinks: bool = False,
) -> Iterator[Path]:
    """Yield files under ``root`` whose extension maps to a known language.

    Skips ``exclude_dirs`` and dot-directories. If ``root`` is itself a file, yields it
    (when its extension is recognized). ``languages`` restricts to those language names.
    """
    root_path = Path(root)
    if root_path.is_file():
        if detect_language(str(root_path)) is not None:
            yield root_path
        return
    allowed = set(languages) if languages else None
    excl = set(exclude_dirs)
    for dirpath, dirnames, filenames in os.walk(root_path, followlinks=follow_symlinks):
        # Prune in place so os.walk does not descend into excluded/hidden dirs.
        dirnames[:] = [d for d in dirnames if d not in excl and not d.startswith(".")]
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            lang = detect_language(str(p))
            if lang is None:
                continue
            if allowed is not None and lang not in allowed:
                continue
            yield p


def read_source_text(
    path: str | os.PathLike[str],
    *,
    encoding: str = "utf-8",
    encoding_fallback: bool = True,
) -> str | None:
    """Read a source file, degrading through encodings instead of skipping it.

    Decode order: BOM-aware UTF-8 (``utf-8-sig`` when a BOM is present), then
    ``encoding``, then ``latin-1`` (a total decode — cannot fail) when
    ``encoding_fallback`` is enabled. This matches the package's "a parse never
    hard-fails" posture: the extension allowlist already keeps binaries out, and
    a legacy-encoded file is far more useful mojibake-free-enough than absent.

    Every hashing/parsing entry point in this package MUST read through this
    function so ``content_hash`` stays consistent across them.

    Returns ``None`` only when the file cannot be read at all (``OSError``) or
    decoding fails with ``encoding_fallback`` disabled.
    """
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError:
        return None
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig", errors="replace")
    try:
        return data.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        if not encoding_fallback:
            return None
        return data.decode("latin-1")


def parse_path(
    path: str | os.PathLike[str],
    *,
    encoding: str = "utf-8",
    encoding_fallback: bool = True,
) -> ParsedCode | None:
    """Parse a single file into symbols + relations. Returns ``None`` if unreadable."""
    p = Path(path)
    content = read_source_text(p, encoding=encoding, encoding_fallback=encoding_fallback)
    if content is None:
        return None
    return _parse(content, file_path=str(p))


def chunk_path(
    path: str | os.PathLike[str],
    config: ChunkConfig | None = None,
    *,
    encoding: str = "utf-8",
    encoding_fallback: bool = True,
) -> list[CodeChunk]:
    """Read + chunk a single file. Returns an empty list if it can't be read."""
    p = Path(path)
    content = read_source_text(p, encoding=encoding, encoding_fallback=encoding_fallback)
    if content is None:
        return []
    return _chunk(content, file_path=str(p), config=config)


def chunk_repo(
    root: str | os.PathLike[str],
    config: ChunkConfig | None = None,
    *,
    languages: list[str] | None = None,
) -> Iterator[CodeChunk]:
    """Walk ``root`` and yield chunks for every source file (streaming, low-memory)."""
    for p in iter_source_files(root, languages=languages):
        yield from chunk_path(p, config)
