# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Merkle projection over the flat content-hash manifest.

The flat manifest (``manifest.py``) is the single source of truth; this
module derives a per-directory rolling digest tree from it so incremental
consumers can prune work by subtree (root equality is the O(1) no-op
short-circuit). Pure and stdlib-only: no filesystem access, no persistence
— consumers may persist :func:`serialize_tree` output however they like.

The Merkle root is intentionally INDEPENDENT of
``repository_index.generation_id``: different construction (hierarchical
directory rolling vs whole-repo sorted-manifest digest), different
purpose (subtree pruning vs snapshot identity).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Mapping


class MerkleFormatError(ValueError):
    """A serialized Merkle tree is corrupt or has the wrong shape."""


def _digest(parts: list[str]) -> str:
    hasher = hashlib.blake2b()
    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


@dataclass(frozen=True)
class MerkleTree:
    """Directory-level rolling digests derived from a flat manifest.

    ``nodes`` maps directory prefixes (``""`` for the root, ``"src/"``,
    ``"src/core/"``) to their subtree digest; ``leaves`` maps file paths to
    their manifest content hashes. ``chunks`` (optional) carries the
    chunk-level projection: ``path -> {chunk_id -> chunk digest}`` — the
    tree then extends from directory to file to CONSTITUENT CHUNK, so
    chunk-relevant identity in the graph is represented at leaf
    granularity and re-embedding work can be chunk-precise.
    """

    root: str
    nodes: dict[str, str] = field(default_factory=dict)
    leaves: dict[str, str] = field(default_factory=dict)
    leaf_count: int = 0
    chunks: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class MerkleDiff:
    """File-level work set plus the minimal changed-directory frontier.

    ``changed_dirs`` lists the DEEPEST directories whose subtree digest
    differs — a directory is listed only when no child directory also
    differs (ancestors are implied by any frontier entry). When chunk
    digests are present on both trees, ``changed_chunks`` carries the
    chunk-precise work set as ``(file, chunk_id)`` pairs — a symbol-body
    edit lists only THAT symbol's chunks, not every chunk in the file.
    """

    changed_dirs: list[str] = field(default_factory=list)
    added_files: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    removed_files: list[str] = field(default_factory=list)
    changed_chunks: list[tuple[str, str]] = field(default_factory=list)


def _dir_of(path: str) -> str:
    return path.rsplit("/", 1)[0] + "/" if "/" in path else ""


def _name_of(path: str) -> str:
    return path.rsplit("/", 1)[1] if "/" in path else path


def build_merkle_tree(
    manifest: Mapping[str, str],
    chunk_digests: Mapping[str, Mapping[str, str]] | None = None,
) -> MerkleTree:
    """Build the directory digest tree from a flat ``path -> content_hash`` map.

    Directory digests roll bottom-up (deepest directory first): each
    directory's digest is blake2b over its canonically sorted
    ``name:digest`` entries — files contribute their manifest hash, child
    directories contribute their subtree digest. Deterministic for a given
    manifest regardless of insertion order.

    ``chunk_digests`` optionally extends the leaves to the file's
    CONSTITUENT CHUNKS: ``path -> {chunk_id -> chunk digest}`` (deterministic
    ids from ``sizing``, e.g. ``{symbol.id}#body#{idx}``). When present, a
    file's digest rolls its sorted chunk digests instead of the raw content
    hash — the tree then represents chunk-relevant graph identity at leaf
    granularity. ``leaves`` still records the manifest content hash as the
    integrity check that chunk coverage is complete for that file.
    """
    if not manifest:
        empty = _digest([])
        return MerkleTree(root=empty, nodes={"": empty}, leaves={}, leaf_count=0)
    chunks_in = dict(chunk_digests) if chunk_digests else {}

    # Group file digests by directory; collect every ancestor directory.
    files_by_dir: dict[str, dict[str, str]] = {}
    dirs: set[str] = set()
    leaves: dict[str, str] = {}
    file_digests: dict[str, str] = {}
    for path, digest in manifest.items():
        leaves[path] = digest
        chunk_map = chunks_in.get(path)
        if chunk_map:
            file_digests[path] = _digest(
                [f"{cid}:{cdigest}" for cid, cdigest in sorted(chunk_map.items())]
            )
        else:
            file_digests[path] = digest
        directory = _dir_of(path)
        files_by_dir.setdefault(directory, {})[_name_of(path)] = file_digests[path]
        dirs.add(directory)
        dirs.add("")  # the root always participates in the tree
        parts = [p for p in directory.split("/") if p]
        for depth in range(1, len(parts)):
            dirs.add("/".join(parts[:depth]) + "/")

    dir_digests: dict[str, str] = {}
    # Deepest first, so a child directory's digest always exists when its
    # parent is rolled.
    for directory in sorted(dirs, key=lambda d: (-d.count("/"), d)):
        entries: list[tuple[str, str]] = list(files_by_dir.get(directory, {}).items())
        prefix_len = len(directory)
        for child_dir, child_digest in dir_digests.items():
            if child_dir.startswith(directory) and "/" not in child_dir[prefix_len:-1]:
                entries.append((child_dir[prefix_len:-1] + "/", child_digest))
        dir_digests[directory] = _digest([f"{name}:{digest}" for name, digest in sorted(entries)])

    root = dir_digests[""]
    nodes = {"": root, **{d: h for d, h in dir_digests.items() if d}}
    return MerkleTree(
        root=root,
        nodes=nodes,
        leaves=leaves,
        leaf_count=len(manifest),
        chunks={p: dict(sorted(c.items())) for p, c in sorted(chunks_in.items()) if c},
    )


def diff_merkle(current: MerkleTree, previous: MerkleTree) -> MerkleDiff:
    """Diff two trees: file-level changes plus the deepest changed dirs.

    Root equality short-circuits to an empty diff (the consumer no-op).
    """
    if current.root == previous.root:
        return MerkleDiff()

    old_leaves = previous.leaves
    new_leaves = current.leaves
    added = sorted(set(new_leaves) - set(old_leaves))
    removed = sorted(set(old_leaves) - set(new_leaves))
    changed = sorted(p for p in set(new_leaves) & set(old_leaves) if new_leaves[p] != old_leaves[p])

    differing = {
        directory
        for directory, digest in current.nodes.items()
        if previous.nodes.get(directory) != digest
    }
    frontier = sorted(
        d
        for d in differing
        if not any(o != d and o.startswith(d) and o in differing for o in differing)
    )
    changed_chunks: list[tuple[str, str]] = []
    if current.chunks and previous.chunks:
        for path in set(current.chunks) | set(previous.chunks):
            old_chunks = previous.chunks.get(path, {})
            new_chunks = current.chunks.get(path, {})
            for chunk_id in sorted(set(old_chunks) | set(new_chunks)):
                if new_chunks.get(chunk_id) != old_chunks.get(chunk_id):
                    changed_chunks.append((path, chunk_id))
    return MerkleDiff(
        changed_dirs=frontier,
        added_files=added,
        changed_files=changed,
        removed_files=removed,
        changed_chunks=changed_chunks,
    )


def changed_subtrees(current: MerkleTree, previous: MerkleTree) -> frozenset[str]:
    """Directory prefixes needing rework (the diff frontier).

    Consumers use ``current.root == previous.root`` as the O(1) no-op
    check before calling this — root equality implies an empty frontier.
    """
    if current.root == previous.root:
        return frozenset()
    return frozenset(diff_merkle(current, previous).changed_dirs)


def serialize_tree(tree: MerkleTree) -> str:
    """Canonical sorted-JSON serialization for consumer persistence."""
    return json.dumps(
        {
            "version": 1,
            "root": tree.root,
            "leaf_count": tree.leaf_count,
            "nodes": dict(sorted(tree.nodes.items())),
            "leaves": dict(sorted(tree.leaves.items())),
            "chunks": {p: dict(sorted(c.items())) for p, c in sorted(tree.chunks.items())},
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def deserialize_tree(text: str) -> MerkleTree:
    """Restore a serialized tree; raises MerkleFormatError on corruption."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MerkleFormatError(f"corrupt Merkle tree JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("root"), str):
        raise MerkleFormatError("Merkle tree payload has the wrong shape")
    nodes = payload.get("nodes")
    leaves = payload.get("leaves")
    if not isinstance(nodes, dict) or not isinstance(leaves, dict):
        raise MerkleFormatError("Merkle tree payload has the wrong shape")
    chunks_raw = payload.get("chunks", {})
    if not isinstance(chunks_raw, dict):
        raise MerkleFormatError("Merkle tree payload has the wrong shape")
    chunks = {
        str(path): {str(cid): str(cdigest) for cid, cdigest in c.items()}
        for path, c in chunks_raw.items()
        if isinstance(c, dict)
    }
    return MerkleTree(
        root=payload["root"],
        nodes={str(k): str(v) for k, v in nodes.items()},
        leaves={str(k): str(v) for k, v in leaves.items()},
        leaf_count=int(payload.get("leaf_count", len(leaves))),
        chunks=chunks,
    )


__all__ = [
    "MerkleTree",
    "MerkleDiff",
    "MerkleFormatError",
    "build_merkle_tree",
    "diff_merkle",
    "changed_subtrees",
    "serialize_tree",
    "deserialize_tree",
]
