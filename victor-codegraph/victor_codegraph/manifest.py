"""Shared incremental-parse contract: content-hash manifests.

``ParsedCode.content_hash`` existed from day one but nothing consumed it — every
consumer (Victor's graph pipeline, the ProximaDB SDK, AnvaiOps) invented its own
change detection. This module is the one shared contract:

- a *manifest* is a plain ``dict[str, str]`` mapping repo-relative POSIX paths to
  content hashes (``model.content_hash``). It is deliberately just a dict so any
  consumer can persist it however it likes (JSON blob, DB rows, object store) —
  this package stays stdlib-only and does no persistence itself.
- :func:`build_manifest` produces the current manifest for a tree.
- :func:`diff_manifest` classifies added / changed / removed / unchanged paths.
- :func:`iter_changed_files` streams only the files whose hash differs from a
  previous manifest — the read used for hashing is the same read parsing needs,
  and the hash equals ``parse_path(...).content_hash``, so consumers can store
  the yielded hash directly after indexing the file.

mtime fast-paths remain the consumer's optimization layer (e.g. Victor's
``graph_file_mtime`` table): hashing requires reading the file, which parsing
needs anyway, but a consumer that trusts mtimes can skip calling this at all.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .model import content_hash
from .repo import DEFAULT_EXCLUDE_DIRS, iter_source_files, read_source_text


@dataclass(frozen=True)
class ManifestDiff:
    """Classification of paths between a current and a previous manifest."""

    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)


def _manifest_key(path: Path, root: Path) -> str:
    """Repo-relative POSIX key — portable across machines and OSes."""
    if path == root:
        return path.name  # single-file root
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return Path(os.path.relpath(path, root)).as_posix()


def build_manifest(
    root: str | os.PathLike[str],
    *,
    languages: list[str] | None = None,
    exclude_dirs: frozenset[str] | set[str] = DEFAULT_EXCLUDE_DIRS,
) -> dict[str, str]:
    """Map every recognized source file under ``root`` to its content hash."""
    root_path = Path(root)
    manifest: dict[str, str] = {}
    for p in iter_source_files(root_path, languages=languages, exclude_dirs=exclude_dirs):
        text = read_source_text(p)
        if text is None:
            continue
        manifest[_manifest_key(p, root_path)] = content_hash(text)
    return manifest


def diff_manifest(current: dict[str, str], previous: dict[str, str]) -> ManifestDiff:
    """Classify paths by comparing two manifests (pure, no filesystem access)."""
    added: list[str] = []
    changed: list[str] = []
    unchanged: list[str] = []
    for path, digest in sorted(current.items()):
        old = previous.get(path)
        if old is None:
            added.append(path)
        elif old != digest:
            changed.append(path)
        else:
            unchanged.append(path)
    removed = sorted(set(previous) - set(current))
    return ManifestDiff(added=added, changed=changed, removed=removed, unchanged=unchanged)


def iter_changed_files(
    root: str | os.PathLike[str],
    previous: dict[str, str] | None,
    *,
    languages: list[str] | None = None,
    exclude_dirs: frozenset[str] | set[str] = DEFAULT_EXCLUDE_DIRS,
) -> Iterator[tuple[Path, str]]:
    """Yield ``(path, content_hash)`` for files absent from or changed vs ``previous``.

    ``previous=None`` yields every source file (full index). The yielded hash
    equals ``parse_path(path).content_hash``, so consumers can persist it into
    their next manifest without re-reading the file.
    """
    prev = previous or {}
    root_path = Path(root)
    for p in iter_source_files(root_path, languages=languages, exclude_dirs=exclude_dirs):
        text = read_source_text(p)
        if text is None:
            continue
        digest = content_hash(text)
        if prev.get(_manifest_key(p, root_path)) != digest:
            yield p, digest
