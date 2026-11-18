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

"""Intent-tier index store tests (FEP-0049 phase 2): idempotent upserts,
per-file deletion, sync-journal semantics, embedding_ref guard."""

import pytest

from victor.storage.graph.intent_index import (
    IntentChunkRecord,
    IntentIndexStore,
)

PROFILE = "profile-abc"


def _record(chunk_id="s1#body#0", file="src/a.py", oid="oid1", text="def a(): ..."):
    return IntentChunkRecord(
        chunk_id=chunk_id,
        symbol_oid=oid,
        symbol_id="s1",
        file=file,
        content=text,
        input_hash="i" * 64,
        source_hash="s" * 64,
    )


class FakeProvider:
    """In-memory stand-in for the LanceDB provider slice."""

    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.cleared = False

    async def index_embedded_documents(self, documents):
        for doc in documents:
            self.docs[doc["id"]] = doc

    async def delete_by_file(self, file_path):
        doomed = [k for k, d in self.docs.items() if d["metadata"]["file_path"] == file_path]
        for k in doomed:
            del self.docs[k]
        return len(doomed)

    async def clear_index(self):
        self.docs.clear()
        self.cleared = True

    async def close(self):
        pass


class FakeGraphStore:
    """In-memory intent_file_state."""

    def __init__(self):
        self.state: dict[str, tuple[str, str]] = {}

    def record_intent_synced(self, file, source_hash, profile_id):
        self.state[file] = (source_hash, profile_id)

    def intent_synced_manifest(self):
        return {f: h for f, (h, _) in self.state.items()}

    def intent_stale_files(self, profile_id):
        return [f for f, (_, p) in self.state.items() if p != profile_id]

    def clear_intent_state(self):
        self.state.clear()

    def delete_intent_state_for_file(self, file):
        self.state.pop(file, None)


@pytest.fixture
def store():
    provider = FakeProvider()
    graph = FakeGraphStore()
    return IntentIndexStore(provider, graph), provider, graph


class TestUpsert:
    async def test_upsert_idempotent_by_chunk_id(self, store):
        s, provider, _ = store
        n = await s.upsert_chunks([_record()], [[1.0] * 384])
        assert n == 1
        await s.upsert_chunks([_record()], [[2.0] * 384])
        assert len(provider.docs) == 1
        assert provider.docs["s1#body#0"]["vector"] == [2.0] * 384

    async def test_count_mismatch_rejected(self, store):
        s, provider, _ = store
        with pytest.raises(ValueError, match="mismatch"):
            await s.upsert_chunks([_record()], [])
        assert not provider.docs

    async def test_metadata_carries_oid_correlation(self, store):
        s, provider, _ = store
        await s.upsert_chunks([_record(oid="oid-77")], [[1.0] * 384])
        meta = provider.docs["s1#body#0"]["metadata"]
        assert meta["symbol_oid"] == "oid-77"
        assert meta["file_path"] == "src/a.py"


class TestDeleteAndClear:
    async def test_delete_file_removes_vectors_and_marker(self, store):
        s, provider, graph = store
        await s.upsert_chunks(
            [_record(), _record(chunk_id="s1#body#1")], [[1.0] * 384, [1.0] * 384]
        )
        await s.record_synced([("src/a.py", "s" * 64, PROFILE)])
        removed = await s.delete_file("src/a.py")
        assert removed == 2
        assert not provider.docs
        assert "src/a.py" not in graph.state

    async def test_clear_drops_vectors_and_state(self, store):
        s, provider, graph = store
        await s.upsert_chunks([_record()], [[1.0] * 384])
        await s.record_synced([("src/a.py", "s" * 64, PROFILE)])
        await s.clear()
        assert provider.cleared
        assert not provider.docs
        assert not graph.state


class TestSyncJournal:
    async def test_synced_manifest_roundtrip(self, store):
        s, _, graph = store
        await s.record_synced([("src/a.py", "h1", PROFILE), ("src/b.py", "h2", PROFILE)])
        manifest = s.synced_manifest()
        assert manifest == {"src/a.py": "h1", "src/b.py": "h2"}

    async def test_stale_files_detects_profile_change(self, store):
        s, _, _ = store
        await s.record_synced([("src/a.py", "h1", "old-profile")])
        assert s.stale_files(PROFILE) == ["src/a.py"]
        await s.record_synced([("src/a.py", "h1", PROFILE)])
        assert s.stale_files(PROFILE) == []


class TestEmbeddingRefGuard:
    def test_intent_module_never_references_embedding_ref(self):
        """FEP-0049/D2: the intent path must not write graph_node.embedding_ref
        (Proxima backend retires it; correlation is by oid)."""
        import inspect

        import victor.storage.graph.intent_index as module

        source = inspect.getsource(module)
        assert "embedding_ref" not in source


class TestGraphStoreIntegration:
    def test_sqlite_store_implements_intent_state_contract(self, tmp_path):
        """SqliteGraphStore persists intent_file_state through WAL schema."""
        from victor.storage.graph.sqlite_store import SqliteGraphStore

        store = SqliteGraphStore(project_path=tmp_path)
        store.record_intent_synced("src/a.py", "h1", PROFILE)
        store.record_intent_synced("src/b.py", "h2", "other")
        assert store.intent_synced_manifest() == {"src/a.py": "h1", "src/b.py": "h2"}
        assert store.intent_stale_files(PROFILE) == ["src/b.py"]
        store.delete_intent_state_for_file("src/b.py")
        assert store.intent_synced_manifest() == {"src/a.py": "h1"}
        store.clear_intent_state()
        assert store.intent_synced_manifest() == {}
