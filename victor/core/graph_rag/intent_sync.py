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
    oversized_skipped: list[str] = field(default_factory=list)
    failed_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned_files": self.scanned_files,
            "reembedded_files": self.reembedded_files,
            "deleted_files": self.deleted_files,
            "chunks_embedded": self.chunks_embedded,
            "skipped_by_root_shortcut": self.skipped_by_root_shortcut,
            "skipped_dirs": self.skipped_dirs,
            "oversized_skipped": self.oversized_skipped,
            "failed_files": self.failed_files,
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
    # Capability gate FIRST: stores without intent-state support (duckdb,
    # memory, proxima) would silently re-embed the full corpus on every run
    # -- markers never persist, so nothing short-circuits.
    if getattr(graph_store, "record_intent_synced", None) is None:
        return IntentSyncReport(
            profile_id=profile_id,
            scanned_files=len(current_manifest),
            degraded_reason=(
                "graph store does not implement intent-state tracking "
                "(sqlite-backed stores support the intent tier)"
            ),
        )

    if encoder is None:
        return None
    encoder = _LazyEncoder(encoder)

    from victor.storage.graph.intent_index import IntentChunkRecord
    from victor_codegraph import build_merkle_tree, diff_manifest

    report = IntentSyncReport(profile_id=profile_id)
    previous: dict[str, str] = dict(intent_store.synced_manifest())

    # Profile-drift check BEFORE the root shortcut: a model/contract upgrade
    # with NO source edits must still rebuild (the shortcut would otherwise
    # serve stale-profile vectors forever).
    stale = intent_store.stale_files(profile_id)

    # Merkle projection for the shortcut + frontier granularity; the flat
    # diff stays the authoritative work set.
    current_tree = build_merkle_tree(current_manifest)
    previous_tree = build_merkle_tree(previous)
    if not force and not stale and current_tree.root == previous_tree.root:
        report.skipped_by_root_shortcut = True
        report.scanned_files = len(current_manifest)
        return report

    diff = diff_manifest(current_manifest, previous)

    # F9: diff_manifest defines removed = previous - current exactly.
    removed = set(diff.removed)
    work = sorted(set(diff.added) | set(diff.changed))
    if stale or force:
        # Full rebuild: clear() wipes ALL vectors and markers, so the work
        # set must cover every current file -- not just the diff.
        await intent_store.clear()
        work = sorted(current_manifest)
        removed = set(previous)
        if stale:
            report.degraded_reason = "profile change: full intent-tier rebuild"

    report.scanned_files = len(current_manifest)
    report.deleted_files = len(removed)
    for path in sorted(removed):
        await intent_store.delete_file(path)

    import asyncio

    from victor.storage.graph.intent_index import _CHUNKS_PER_FILE_CEILING

    source_hashes = {p: current_manifest[p] for p in work}
    failed_files: list[str] = []
    for rel_path in work:
        try:
            from victor_codegraph import chunk_path, embed_chunks

            # CPU-bound parse + inference off the event loop (the TUI,
            # HTTP API and watch daemon share it).
            chunks = await asyncio.to_thread(chunk_path, root / rel_path, repo_root=root)
            if len(chunks) > _CHUNKS_PER_FILE_CEILING:
                report.oversized_skipped.append(rel_path)
                logger.warning(
                    "Intent tier skipped %s: %d chunks exceed the per-file ceiling %d",
                    rel_path,
                    len(chunks),
                    _CHUNKS_PER_FILE_CEILING,
                )
                continue
            if not chunks:
                # Nothing chunkable: still commit the marker so the file
                # does not re-appear in every future work set.
                pass
            else:
                docs = await asyncio.to_thread(
                    embed_chunks,
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
            failed_files.append(rel_path)
            logger.warning("Intent-tier embed failed for %s: %s", rel_path, exc)
            report.degraded_reason = f"{rel_path}: {exc}"
            continue
        if not _marker_dropout:
            await intent_store.record_synced([(rel_path, current_manifest[rel_path], profile_id)])

    # Systemic-failure escalation: every file failed -> the run is NOT a
    # healthy partial; say so loudly (a missing sentence-transformers install
    # used to loop N import-error retries per indexing run, invisibly).
    if work and report.reembedded_files == 0:
        report.failed_files = failed_files
        report.degraded_reason = (
            f"systemic: all {len(work)} work files failed; first error: "
            f"{report.degraded_reason}"
        )
        logger.error("Intent-tier sync failed systemically: %s", report.degraded_reason)

    return report


class _LazyEncoder:
    """Defers encoder construction until the first real encode call — a
    root-shortcut sync never pays the model-load cost."""

    def __init__(self, factory_or_encoder: Any) -> None:
        self._factory = factory_or_encoder
        self._inner: Any = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._inner is None:
            # Marker identity comes from the profile_id argument, never from
            # the encoder: a caller passing a divergent encoder is flagged
            # by the store's profile metadata check, not hidden here.
            self._inner = (
                self._factory()
                if callable(self._factory) and not hasattr(self._factory, "encode")
                else self._factory
            )
        return self._inner.encode(texts)


def _chunk_profile() -> Any:
    from victor_codegraph import CHUNK_SOURCE_PROFILE

    return CHUNK_SOURCE_PROFILE


__all__ = ["IntentSyncReport", "sync_intent_tier"]
