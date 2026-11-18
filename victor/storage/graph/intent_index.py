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

"""Intent-tier embedding index (FEP-0049): chunk-source vectors anchored
to graph symbols by stable oid.

Storage split: chunk VECTORS live in a dedicated LanceDB table
(``code_chunks``) through the shared embedding-provider registry; the
per-file SYNC state lives in the graph SQLite DB (``intent_file_state``)
so it participates in WAL and the repo's existing ``.victor`` layout.
The correlation key is ``symbol_oid`` (ADR-044 stable oid) carried in
vector metadata — uniform across sqlite/duckdb/lancedb and
proxima-by-oid backends. The legacy per-node embedding column on the
node table is never written by this module (guarded by test).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

logger = logging.getLogger(__name__)

INTENT_TABLE = "code_chunks"

_CHUNKS_PER_FILE_CEILING = 5000


@dataclass(frozen=True)
class IntentChunkRecord:
    """One chunk's identity for the vector row + sync journal."""

    chunk_id: str  # deterministic: f"{symbol.id}#body#{idx}"
    symbol_oid: str  # ADR-044 stable oid — the graph correlation key
    symbol_id: str
    file: str  # repo-relative POSIX path
    content: str  # chunk source text
    input_hash: str  # sha256 of chunk text (staleness contract)
    source_hash: str  # manifest hash of the file (verbatim)


class _VectorProvider(Protocol):
    """The slice of the embedding-provider surface the intent tier uses."""

    async def index_embedded_documents(self, documents: list[dict[str, Any]]) -> None: ...

    async def delete_by_file(self, file_path: str) -> int: ...

    async def clear_index(self) -> None: ...

    async def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Chunk-level intent search: provider self-embeds the query.

        Returns raw provider hits (score + metadata incl. ``symbol_id`` /
        ``symbol_oid`` / ``chunk_id``) for the caller to aggregate per
        symbol. Empty when the provider/table has no vectors.
        """
        hits = await self._provider.search_similar(query, limit=limit)
        results = []
        for hit in hits:
            results.append(
                {
                    "score": float(getattr(hit, "score", 0.0)),
                    "chunk_id": str(getattr(hit, "id", "")),
                    "content": str(getattr(hit, "content", "")),
                    "metadata": dict(getattr(hit, "metadata", {}) or {}),
                }
            )
        return results

    async def close(self) -> None: ...


class IntentIndexStore:
    """Chunk-level intent embedding index.

    ``provider`` is an embedding-registry provider configured with
    ``table_name=INTENT_TABLE``; ``graph_store`` is any object exposing the
    intent-state methods (``SqliteGraphStore`` implements them; tests use
    in-memory fakes).
    """

    def __init__(self, provider: _VectorProvider, graph_store: Any) -> None:
        self._provider = provider
        self._graph_store = graph_store

    async def upsert_chunks(
        self, records: Sequence[IntentChunkRecord], vectors: Sequence[Sequence[float]]
    ) -> int:
        """Upsert chunk vectors (idempotent by chunk_id) — BEFORE the sync
        marker commit, per the FEP-0049 crash-ordering contract."""
        if len(records) != len(vectors):
            raise ValueError("record/vector count mismatch")
        if not records:
            return 0
        documents = [
            {
                "id": record.chunk_id,
                "vector": list(vector),
                "content": record.content,
                "metadata": {
                    "symbol_oid": record.symbol_oid,
                    "symbol_id": record.symbol_id,
                    "file_path": record.file,
                    "input_hash": record.input_hash,
                    "source_hash": record.source_hash,
                },
            }
            for record, vector in zip(records, vectors, strict=True)
        ]
        await self._provider.index_embedded_documents(documents)
        return len(documents)

    async def delete_file(self, file: str) -> int:
        """Remove one file's vectors and its sync marker."""
        deleted = await self._provider.delete_by_file(file)
        gs = self._graph_store
        deleter = getattr(gs, "delete_intent_state_for_file", None)
        if deleter is not None:
            deleter(file)
        return int(deleted or 0)

    async def clear(self) -> None:
        """Drop all vectors and sync markers (profile-change rebuild)."""
        await self._provider.clear_index()
        clearer = getattr(self._graph_store, "clear_intent_state", None)
        if clearer is not None:
            clearer()

    async def record_synced(self, rows: Iterable[tuple[str, str, str]]) -> None:
        """Commit sync markers (file, source_hash, profile_id) — LAST."""
        recorder = getattr(self._graph_store, "record_intent_synced", None)
        if recorder is None:
            return
        for file, source_hash, profile_id in rows:
            recorder(file, source_hash, profile_id)

    def synced_manifest(self) -> dict[str, str]:
        """The previous manifest for the next incremental Merkle walk."""
        getter = getattr(self._graph_store, "intent_synced_manifest", None)
        return dict(getter()) if getter is not None else {}

    def stale_files(self, profile_id: str) -> list[str]:
        """Files committed under a different profile (full re-embed set)."""
        getter = getattr(self._graph_store, "intent_stale_files", None)
        return list(getter(profile_id)) if getter is not None else []

    async def close(self) -> None:
        closer = getattr(self._provider, "close", None)
        if closer is not None:
            await closer()


__all__ = [
    "INTENT_TABLE",
    "_CHUNKS_PER_FILE_CEILING",
    "IntentChunkRecord",
    "IntentIndexStore",
]
