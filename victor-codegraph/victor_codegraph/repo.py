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


def parse_path(path: str | os.PathLike[str], *, encoding: str = "utf-8") -> ParsedCode | None:
    """Parse a single file into symbols + relations. Returns ``None`` if unreadable."""
    p = Path(path)
    try:
        content = p.read_text(encoding=encoding)
    except (OSError, UnicodeDecodeError):
        return None
    return _parse(content, file_path=str(p))


def chunk_path(
    path: str | os.PathLike[str],
    config: ChunkConfig | None = None,
    *,
    encoding: str = "utf-8",
) -> list[CodeChunk]:
    """Read + chunk a single file. Returns an empty list if it can't be read."""
    p = Path(path)
    try:
        content = p.read_text(encoding=encoding)
    except (OSError, UnicodeDecodeError):
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
