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

"""SQLite-backed intent-tier chunk store (FEP-0049 consolidation).

Replaces the LanceDB ``code_chunks`` table with a SQLite ``intent_chunks``
table in the same project.db as the graph (vectors as float32
little-endian BLOBs; one row per deterministic chunk_id). Search is
brute-force cosine over a numpy matrix cached across queries and
invalidated by a write-version counter (measured: 0.1 ms @ 14K rows,
3.4 ms @ 100K rows) — no ANN index, so no lancedb/pyarrow dependency on
the intent tier's required path.

The per-file sync journal stays in ``intent_file_state`` (zero-chunk
files need their "nothing to embed" tombstone, and the marker-commits-
LAST crash contract is expressed through it). ``profile_id`` per row
detects mixed-profile corruption at search time; zero-norm vectors are
excluded from the matrix at load so they can never yield NaN similarity.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from victor.core.schema import Tables

logger = logging.getLogger(__name__)

_INTENT_CHUNKS_TABLE = Tables.INTENT_CHUNKS
_VECTOR_DTYPE = np.dtype("<f4")  # explicit little-endian float32 (portable BLOBs)


def _serialize_vector(values: Any) -> bytes:
    arr = np.asarray(values, dtype=_VECTOR_DTYPE)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(
            f"intent chunk vector must be a non-empty 1-D sequence, got shape {arr.shape}"
        )
    return arr.tobytes()


def _deserialize_vector(blob: bytes) -> Any:
    if len(blob) % _VECTOR_DTYPE.itemsize:
        raise ValueError(f"corrupt intent vector blob: {len(blob)} bytes is not a float32 multiple")
    return np.frombuffer(blob, dtype=_VECTOR_DTYPE)


class SqliteIntentChunkStore:
    """The ``_VectorProvider`` protocol slice backed by the ``intent_chunks``
    SQLite table.

    ``graph_store`` is a ``SqliteGraphStore`` (its dedicated connection and
    transaction discipline are reused; same-package access to ``_connect``).
    The query encoder is injected lazily via :meth:`set_encoder_factory` —
    an encoder that is never configured means search returns no hits, never
    a model load.
    """

    def __init__(self, graph_store: Any) -> None:
        self._graph_store = graph_store
        self._encoder_factory: Optional[Callable[[], Any]] = None
        self._encoder: Any = None
        # Matrix cache, invalidated by the write version.
        self._matrix: Any = None  # (N, D) float32
        self._matrix_ids: List[str] = []
        self._matrix_meta: List[Dict[str, str]] = []
        self._write_version: int = 0
        self._cache_version: int = -1

    # -- configuration ----------------------------------------------------

    def set_encoder_factory(self, factory: Callable[[], Any]) -> None:
        """Inject the query encoder factory (called at most once, on first search)."""
        self._encoder_factory = factory

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            if self._encoder_factory is None:
                return None
            self._encoder = self._encoder_factory()
        return self._encoder

    # -- write path --------------------------------------------------------

    async def index_embedded_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Upsert chunk vectors (idempotent by deterministic chunk_id)."""
        with self._graph_store._connect() as conn:
            for doc in documents:
                blob = _serialize_vector(doc["vector"])
                meta = doc.get("metadata") or {}
                conn.execute(
                    f"""
                    INSERT INTO {_INTENT_CHUNKS_TABLE}
                        (chunk_id, symbol_oid, symbol_id, file, content,
                         input_hash, source_hash, profile_id, vector)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        symbol_oid=excluded.symbol_oid,
                        symbol_id=excluded.symbol_id,
                        file=excluded.file,
                        content=excluded.content,
                        input_hash=excluded.input_hash,
                        source_hash=excluded.source_hash,
                        profile_id=excluded.profile_id,
                        vector=excluded.vector
                    """,
                    (
                        doc["id"],
                        meta.get("symbol_oid", ""),
                        meta.get("symbol_id", ""),
                        meta.get("file_path", ""),
                        doc.get("content", ""),
                        meta.get("input_hash", ""),
                        meta.get("source_hash", ""),
                        meta.get("profile_id", ""),
                        blob,
                    ),
                )
        self._bump_version()

    async def delete_by_file(self, file_path: str) -> int:
        with self._graph_store._connect() as conn:
            cursor = conn.execute(
                f"DELETE FROM {_INTENT_CHUNKS_TABLE} WHERE file = ?", (file_path,)
            )
            deleted = cursor.rowcount
        self._bump_version()
        return int(deleted or 0)

    async def clear_index(self) -> None:
        with self._graph_store._connect() as conn:
            conn.execute(f"DELETE FROM {_INTENT_CHUNKS_TABLE}")
        self._bump_version()

    # -- read path ----------------------------------------------------------

    async def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Brute-force cosine search over the cached numpy matrix.

        Returns ``[]`` (never an error) when the table is empty or no
        encoder is configured; raises ``ValueError`` on mixed-profile
        corruption (stored rows from more than one embedding profile).
        """
        matrix, ids, meta = self._load_matrix()
        if matrix is None:
            return []
        encoder = self._get_encoder()
        if encoder is None:
            return []
        query_vec = np.asarray(encoder.encode_query(query), dtype=_VECTOR_DTYPE)
        if query_vec.shape != (matrix.shape[1],):
            raise ValueError(
                f"query vector dimension {query_vec.shape[0]} does not match "
                f"stored chunk dimension {matrix.shape[1]}"
            )
        qnorm = float(np.linalg.norm(query_vec))
        if qnorm == 0.0:
            return []
        sims = (matrix @ query_vec) / qnorm  # rows are pre-normalized at load
        k = min(limit, sims.shape[0])
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        results: List[Dict[str, Any]] = []
        for i in top:
            if sims[i] <= 0.0:
                break  # sorted descending: everything after is worse
            m = meta[i]
            results.append(
                {
                    "score": float(sims[i]),
                    "chunk_id": ids[i],
                    "content": m["content"],
                    "metadata": {
                        "symbol_oid": m["symbol_oid"],
                        "symbol_id": m["symbol_id"],
                        "file_path": m["file"],
                        "input_hash": m["input_hash"],
                        "source_hash": m["source_hash"],
                        "profile_id": m["profile_id"],
                    },
                }
            )
        return results

    def _load_matrix(self) -> Tuple[Any, List[str], List[Dict[str, str]]]:
        """Return the cached (matrix, ids, metadata), rebuilding on write.

        Zero-norm rows are EXCLUDED at load (never produce NaN similarity);
        rows from more than one profile_id raise ValueError (mixed-profile
        corruption is a rebuild condition, not a ranking oddity).
        """
        if self._cache_version == self._write_version and self._matrix is not None:
            return self._matrix, self._matrix_ids, self._matrix_meta
        with self._graph_store._connect() as conn:
            rows = conn.execute(f"""
                SELECT chunk_id, symbol_oid, symbol_id, file, content,
                       input_hash, source_hash, profile_id, vector
                FROM {_INTENT_CHUNKS_TABLE} ORDER BY chunk_id
                """).fetchall()
        if not rows:
            self._matrix = None
            self._matrix_ids = []
            self._matrix_meta = []
            self._cache_version = self._write_version
            return self._matrix, self._matrix_ids, self._matrix_meta
        ids = [str(r[0]) for r in rows]
        meta = [
            {
                "symbol_oid": str(r[1]),
                "symbol_id": str(r[2]),
                "file": str(r[3]),
                "content": str(r[4]),
                "input_hash": str(r[5]),
                "source_hash": str(r[6]),
                "profile_id": str(r[7]),
            }
            for r in rows
        ]
        profile_ids = {m["profile_id"] for m in meta}
        if len(profile_ids) > 1:
            raise ValueError(
                f"intent chunk table holds {len(profile_ids)} distinct embedding "
                f"profiles ({sorted(profile_ids)[:3]}...); rebuild required"
            )
        vectors = [_deserialize_vector(bytes(r[8])) for r in rows]
        matrix = np.stack(vectors)
        norms = np.linalg.norm(matrix, axis=1)
        keep = norms > 0.0
        if not bool(keep.all()):
            logger.warning(
                "Excluding %d zero-norm intent chunk vectors from search", int((~keep).sum())
            )
            matrix = matrix[keep]
            ids = [i for i, k in zip(ids, keep) if k]
            meta = [m for m, k in zip(meta, keep) if k]
        matrix = matrix / norms[keep][:, None]
        if matrix.dtype != _VECTOR_DTYPE:
            matrix = matrix.astype(_VECTOR_DTYPE)
        self._matrix = matrix
        self._matrix_ids = ids
        self._matrix_meta = meta
        self._cache_version = self._write_version
        return self._matrix, self._matrix_ids, self._matrix_meta

    # -- lifecycle ----------------------------------------------------------

    async def close(self) -> None:
        self._matrix = None
        self._matrix_ids = []
        self._matrix_meta = []
        self._cache_version = -1

    def _bump_version(self) -> None:
        self._write_version += 1


__all__ = ["SqliteIntentChunkStore"]
