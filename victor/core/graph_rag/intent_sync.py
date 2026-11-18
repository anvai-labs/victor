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

"""Merkle-driven incremental embedding of the intent tier (FEP-0049 phase 3).

Runs AFTER graph publication in ``GraphIndexingPipeline`` — never inside
the sqlite replacement. Work set = flat-manifest diff (authoritative);
the Merkle projection supplies the O(1) root-equality shortcut and the
report granularity. Per-file ordering: chunk -> embed -> delete stale
chunk ids -> upsert vectors -> commit the sync marker (crash between
vector and marker re-embeds identical vectors next run — safe).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


@dataclass
class IntentSyncReport:
    """What the incremental embedding pass actually did (observable)."""

    scanned_files: int = 0
    reembedded_files: int = 0
    deleted_files: int = 0
    chunks_embedded: int = 0
    skipped_by_root_shortcut: bool = False
    skipped_dirs: list[str] = field(default_factory=list)
    degraded_reason: Optional[str] = None
    profile_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned_files": self.scanned_files,
            "reembedded_files": self.reembedded_files,
            "deleted_files": self.deleted_files,
            "chunks_embedded": self.chunks_embedded,
            "skipped_by_root_shortcut": self.skipped_by_root_shortcut,
            "skipped_dirs": self.skipped_dirs,
            "degraded_reason": self.degraded_reason,
            "profile_id": self.profile_id,
        }


async def sync_intent_tier(
    root: Path,
    graph_store: Any,
    *,
    current_manifest: Mapping[str, str],
    intent_store: Any,
    profile_id: str,
    encoder: Any = None,
    force: bool = False,
    _marker_dropout: bool = False,
) -> Optional[IntentSyncReport]:
    """Bring the intent-tier chunk embeddings level with ``current_manifest``.

    Returns ``None`` when the caller's flag gate kept the encoder
    unconstructed (no model-load cost when disabled). ``encoder`` follows
    the package encoder shape (``encode`` batch + ``profile_id``);
    ``intent_store`` is an :class:`IntentIndexStore` (or compatible).
    """
    if encoder is None:
        return None
    encoder = _LazyEncoder(encoder)

    from victor.storage.graph.intent_index import IntentChunkRecord
    from victor_codegraph import build_merkle_tree, diff_manifest

    report = IntentSyncReport(profile_id=profile_id)
    previous: dict[str, str] = dict(intent_store.synced_manifest())

    # Merkle projection for the shortcut + frontier granularity; the flat
    # diff stays the authoritative work set.
    current_tree = build_merkle_tree(current_manifest)
    previous_tree = build_merkle_tree(previous)
    if not force and current_tree.root == previous_tree.root:
        report.skipped_by_root_shortcut = True
        report.scanned_files = len(current_manifest)
        return report

    diff = diff_manifest(current_manifest, previous)

    # Profile change: everything stored under another profile is stale —
    # full rebuild before the incremental pass.
    stale = intent_store.stale_files(profile_id)
    if stale:
        await intent_store.clear()

    removed = set(diff.removed) | (set(previous) - set(current_manifest) - set(diff.added))
    work = sorted(set(diff.added) | set(diff.changed))
    if force:
        work = sorted(set(current_manifest))
        removed = set(previous)

    report.scanned_files = len(current_manifest)
    report.deleted_files = len(removed)
    for path in sorted(removed):
        await intent_store.delete_file(path)

    source_hashes = {p: current_manifest[p] for p in work}
    for rel_path in work:
        try:
            from victor_codegraph import chunk_path, embed_chunks

            chunks = chunk_path(root / rel_path, repo_root=root)
            if not chunks:
                # Nothing chunkable: still commit the marker so the file
                # does not re-appear in every future work set.
                pass
            else:
                docs = embed_chunks(
                    chunks,
                    encoder.encode,
                    _chunk_profile(),
                    source_hashes=source_hashes,
                )
                # Delete-then-upsert: chunk index shifts when a body grows.
                await intent_store.delete_file(rel_path)
                records = [
                    IntentChunkRecord(
                        chunk_id=d.chunk_id,
                        # node_id IS the package symbol id (ADR-044 seam:
                        # codegraph_repository node_id=s.id).
                        symbol_oid=d.symbol_id,
                        symbol_id=d.symbol_id,
                        file=rel_path,
                        content=chunk.text,
                        input_hash=d.input_hash,
                        source_hash=d.source_hash,
                    )
                    for d, chunk in zip(docs, chunks, strict=True)
                ]
                await intent_store.upsert_chunks(records, [list(d.values) for d in docs])
                report.chunks_embedded += len(docs)
            report.reembedded_files += 1
        except Exception as exc:
            logger.warning("Intent-tier embed failed for %s: %s", rel_path, exc)
            report.degraded_reason = f"{rel_path}: {exc}"
            continue
        if not _marker_dropout:
            await intent_store.record_synced([(rel_path, current_manifest[rel_path], profile_id)])

    return report


class _LazyEncoder:
    """Defers encoder construction until the first real encode call — a
    root-shortcut sync never pays the model-load cost."""

    def __init__(self, factory_or_encoder: Any) -> None:
        self._factory = factory_or_encoder
        self._inner: Any = None
        self.profile_id = ""

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._inner is None:
            inner = (
                self._factory()
                if callable(self._factory) and not hasattr(self._factory, "encode")
                else self._factory
            )
            self._inner = inner
            self.profile_id = getattr(inner, "profile_id", "") or getattr(
                getattr(inner, "profile", None), "identity", ""
            )
        return self._inner.encode(texts)


def _chunk_profile() -> Any:
    from victor_codegraph import CHUNK_SOURCE_PROFILE

    return CHUNK_SOURCE_PROFILE


__all__ = ["IntentSyncReport", "sync_intent_tier"]
