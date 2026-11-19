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

"""SQLite-backed intent chunk store tests (FEP-0049 consolidation).

Covers the ``SqliteIntentChunkStore`` provider slice — round-trip ranking,
idempotent upsert, delete isolation, cache invalidation, profile/zero-norm
corruption guards — and the ``IntentIndexStore.search`` read path that
delegates to the provider (the shipped-without-read-path bug).
"""

import numpy as np
import pytest

from victor.storage.graph.intent_index import IntentChunkRecord, IntentIndexStore
from victor.storage.graph.intent_sqlite_store import (
    SqliteIntentChunkStore,
    _deserialize_vector,
    _serialize_vector,
)

PROFILE_A = "profile-aaa"
PROFILE_B = "profile-bbb"


class FakeEncoder:
    """Deterministic package-encoder stand-in (encode_query + profile)."""

    def __init__(self, profile_id=PROFILE_A, dim=3):
        self.profile_id = profile_id
        self.dim = dim
        self.queries: list[str] = []
        self.calls = 0

    def encode_query(self, query):
        self.queries.append(query)
        self.calls += 1
        table = {
            "payments": np.array([1.0, 0.0, 0.0]),
            "refund": np.array([0.0, 1.0, 0.0]),
            "logging": np.array([0.0, 0.0, 1.0]),
        }
        return table.get(query, [0.0] * self.dim)


@pytest.fixture
def graph_store(tmp_path):
    from victor.storage.graph.sqlite_store import SqliteGraphStore

    return SqliteGraphStore(project_path=tmp_path)


@pytest.fixture
def store(graph_store):
    s = SqliteIntentChunkStore(graph_store)
    s.set_encoder_factory(FakeEncoder)
    return s


def _doc(chunk_id, file, vector, profile=PROFILE_A, content="def x(): ..."):
    return {
        "id": chunk_id,
        "vector": vector,
        "content": content,
        "metadata": {
            "symbol_oid": f"oid-{chunk_id}",
            "symbol_id": f"sym-{chunk_id}",
            "file_path": file,
            "input_hash": "i" * 64,
            "source_hash": "s" * 64,
            "profile_id": profile,
        },
    }


class TestRoundTrip:
    @pytest.mark.asyncio
    async def test_search_ranks_by_cosine_similarity(self, store):
        await store.index_embedded_documents(
            [
                _doc("payments#0", "src/pay.py", [1.0, 0.1, 0.0]),
                _doc("refund#0", "src/refund.py", [0.1, 1.0, 0.0]),
                _doc("logging#0", "src/log.py", [0.0, 0.0, 1.0]),
            ]
        )
        hits = await store.search("payments", limit=2)
        assert [h["chunk_id"] for h in hits] == ["payments#0", "refund#0"]
        assert hits[0]["score"] > hits[1]["score"] > 0
        assert hits[0]["metadata"]["file_path"] == "src/pay.py"
        assert hits[0]["content"] == "def x(): ..."

    @pytest.mark.asyncio
    async def test_empty_table_and_unconfigured_encoder_return_empty(self, store):
        assert await store.search("payments") == []
        no_encoder = SqliteIntentChunkStore(store._graph_store)
        await no_encoder.index_embedded_documents([_doc("a#0", "f.py", [1.0, 0.0, 0.0])])
        # No encoder factory configured: search is empty, never a model load.
        assert await no_encoder.search("payments") == []

    @pytest.mark.asyncio
    async def test_encoder_constructed_lazily_and_exactly_once(self, store):
        calls = []

        def factory():
            calls.append(1)
            return FakeEncoder()

        store.set_encoder_factory(factory)
        await store.index_embedded_documents([_doc("a#0", "f.py", [1.0, 0.0, 0.0])])
        assert calls == []  # indexing must not touch the encoder
        await store.search("payments")
        await store.search("refund")
        assert len(calls) == 1  # cached across searches

    @pytest.mark.asyncio
    async def test_query_vector_dimension_mismatch_raises(self, store):
        await store.index_embedded_documents([_doc("a#0", "f.py", [1.0, 0.0, 0.0])])
        store.set_encoder_factory(lambda: FakeEncoder(dim=4))
        with pytest.raises(ValueError, match="dimension"):
            await store.search("anything")


class TestWriteSemantics:
    @pytest.mark.asyncio
    async def test_upsert_is_idempotent_by_chunk_id(self, store):
        doc = _doc("a#0", "f.py", [1.0, 0.0, 0.0])
        await store.index_embedded_documents([doc])
        await store.index_embedded_documents([doc])
        hits = await store.search("payments", limit=10)
        assert len(hits) == 1

    @pytest.mark.asyncio
    async def test_upsert_replaces_changed_vector(self, store):
        await store.index_embedded_documents([_doc("a#0", "f.py", [1.0, 0.0, 0.0])])
        await store.index_embedded_documents([_doc("a#0", "f.py", [0.0, 0.0, 1.0])])
        hits = await store.search("logging", limit=5)
        assert hits and hits[0]["chunk_id"] == "a#0"

    @pytest.mark.asyncio
    async def test_delete_by_file_is_isolated(self, store):
        await store.index_embedded_documents(
            [_doc("a#0", "f.py", [1.0, 0.0, 0.0]), _doc("b#0", "g.py", [1.0, 0.0, 0.0])]
        )
        deleted = await store.delete_by_file("f.py")
        assert deleted == 1
        remaining = await store.search("payments", limit=10)
        assert [h["chunk_id"] for h in remaining] == ["b#0"]

    @pytest.mark.asyncio
    async def test_clear_index_wipes_all(self, store):
        await store.index_embedded_documents(
            [_doc("a#0", "f.py", [1.0, 0.0, 0.0]), _doc("b#0", "g.py", [1.0, 0.0, 0.0])]
        )
        await store.clear_index()
        assert await store.search("payments") == []

    @pytest.mark.asyncio
    async def test_cache_rebuilds_after_write(self, store):
        await store.index_embedded_documents([_doc("a#0", "f.py", [1.0, 0.0, 0.0])])
        assert await store.search("payments") != []
        await store.index_embedded_documents([_doc("b#0", "g.py", [1.0, 0.0, 0.0])])
        hits = await store.search("payments", limit=10)
        assert {h["chunk_id"] for h in hits} == {"a#0", "b#0"}

    @pytest.mark.asyncio
    async def test_invalid_vector_shapes_are_rejected(self, store):
        with pytest.raises(ValueError, match="1-D"):
            await store.index_embedded_documents([_doc("a#0", "f.py", [])])
        with pytest.raises(ValueError, match="1-D"):
            await store.index_embedded_documents([_doc("a#0", "f.py", [[1.0, 0.0]])])


class TestCorruptionGuards:
    @pytest.mark.asyncio
    async def test_mixed_profile_rows_raise_on_search(self, store):
        await store.index_embedded_documents(
            [
                _doc("a#0", "f.py", [1.0, 0.0, 0.0], profile=PROFILE_A),
                _doc("b#0", "g.py", [0.0, 1.0, 0.0], profile=PROFILE_B),
            ]
        )
        with pytest.raises(ValueError, match="profiles"):
            await store.search("payments")

    @pytest.mark.asyncio
    async def test_zero_norm_vectors_excluded_from_matrix(self, store):
        await store.index_embedded_documents(
            [
                _doc("good#0", "f.py", [1.0, 0.0, 0.0]),
                _doc("zero#0", "f.py", [0.0, 0.0, 0.0]),
            ]
        )
        hits = await store.search("payments", limit=10)
        assert [h["chunk_id"] for h in hits] == ["good#0"]

    def test_corrupt_blob_fails_closed(self):
        with pytest.raises(ValueError, match="float32 multiple"):
            _deserialize_vector(b"\x01\x02\x03")
        with pytest.raises(ValueError, match="1-D"):
            _serialize_vector([])
        with pytest.raises(ValueError, match="1-D"):
            _serialize_vector([[1.0, 0.0]])
        assert _serialize_vector([1.0, 2.0, 3.0, 4.0]) == (
            np.array([1.0, 2.0, 3.0, 4.0], dtype="<f4").tobytes()
        )


class TestIntentIndexSearchReadPath:
    """The shipped-without-read-path bug: IntentIndexStore.search must
    delegate to the provider instead of AttributeError-ing."""

    @pytest.mark.asyncio
    async def test_search_delegates_to_provider(self):
        provider_hits = [
            {"score": 0.9, "chunk_id": "a#0", "content": "x", "metadata": {"file_path": "f.py"}}
        ]

        class Provider:
            async def search(self, query, limit=20):
                assert query == "payments" and limit == 3
                return provider_hits

        index = IntentIndexStore(Provider(), graph_store=None)
        assert await index.search("payments", limit=3) == provider_hits

    @pytest.mark.asyncio
    async def test_provider_without_search_returns_empty(self):
        class LegacyProvider:
            async def index_embedded_documents(self, documents): ...

        index = IntentIndexStore(LegacyProvider(), graph_store=None)
        assert await index.search("payments") == []

    @pytest.mark.asyncio
    async def test_end_to_end_sqlite_store_via_intent_index(self, graph_store):
        store = SqliteIntentChunkStore(graph_store)
        store.set_encoder_factory(FakeEncoder)
        index = IntentIndexStore(store, graph_store)
        records = [
            IntentChunkRecord(
                chunk_id="pay#0",
                symbol_oid="oid-pay",
                symbol_id="sym-pay",
                file="src/pay.py",
                content="charge",
                input_hash="i" * 64,
                source_hash="s" * 64,
            ),
            IntentChunkRecord(
                chunk_id="log#0",
                symbol_oid="oid-log",
                symbol_id="sym-log",
                file="src/log.py",
                content="logging",
                input_hash="i" * 64,
                source_hash="s" * 64,
            ),
        ]
        vectors = [[1.0, 0.1, 0.0], [0.0, 0.0, 1.0]]
        assert await index.upsert_chunks(records, vectors) == 2
        # Orthogonal (score exactly 0) hits stay below the positive-score
        # floor: unrelated chunks never surface as noise.
        hits = await index.search("payments", limit=2)
        assert [h["chunk_id"] for h in hits] == ["pay#0"]
        logging_hits = await index.search("logging", limit=2)
        assert [h["chunk_id"] for h in logging_hits] == ["log#0"]
        # The intent index search returns provider-shaped hits for per-symbol
        # aggregation; symbol ids ride the metadata.
        assert hits[0]["metadata"]["symbol_oid"] == "oid-pay"


class TestProviderSelection:
    """build_intent_index must route sqlite-backed stores to the co-located
    SQLite provider and everything else to the legacy registry path —
    a duckdb-shaped store with ``_connect`` must NOT be mistaken for one."""

    def test_sqlite_backed_store_gets_sqlite_provider(self, graph_store):
        from victor.core.graph_rag.seeding import build_intent_index

        index = build_intent_index(graph_store)
        assert index is not None
        from victor.storage.graph.intent_sqlite_store import SqliteIntentChunkStore

        assert isinstance(index._provider, SqliteIntentChunkStore)

    def test_duckdb_shaped_store_never_gets_sqlite_provider(self):
        """DuckDBGraphStore defines ``_connect`` but has no intent-state
        support — routing it to the SQLite store would fail at write time
        on a missing intent_chunks table."""
        from victor.core.graph_rag.seeding import build_intent_index

        class DuckDBShapedStore:
            def _connect(self): ...

        assert build_intent_index(DuckDBShapedStore()) is None

    def test_sync_fake_store_keeps_legacy_path(self):
        """Intent-sync test fakes carry ``record_intent_synced`` without a
        sqlite connection — they must keep the legacy registry path (None
        without ``_build_embedding_config``), not the SQLite store."""
        from victor.core.graph_rag.seeding import build_intent_index

        class SyncFakeStore:
            def record_intent_synced(self, file, source_hash, profile_id): ...

        assert build_intent_index(SyncFakeStore()) is None

    def test_unavailable_store_returns_none_not_raise(self):
        from victor.core.graph_rag.seeding import build_intent_index

        assert build_intent_index(None) is None
