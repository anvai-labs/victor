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

"""Merkle-driven intent-tier incremental sync tests (FEP-0049 phase 3).

Fake encoder + fake provider: the assertions are about the WORK SET
(what got re-embedded), crash ordering, and degradation — not model
quality.
"""

from pathlib import Path


from victor.core.graph_rag.intent_sync import sync_intent_tier


class FakeProvider:
    def __init__(self):
        self.docs: dict[str, dict] = {}

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

    async def close(self):
        pass


class FakeGraphStore:
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


class FakeEncoder:
    """Package-encoder shape: records every encoded text (the work-set probe)."""

    profile_id = "prof-1"

    def __init__(self, fail_before_commit: bool = False):
        self.encoded_texts: list[str] = []
        self.fail_before_commit = fail_before_commit

    def encode(self, texts):
        self.encoded_texts.extend(texts)
        return [[1.0] + [0.0] * 383 for _ in texts]

    def encode_query(self, query):
        return self.encode([query])[0]


def _write(tmp_path: Path, name: str, body: str) -> str:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return str(path)


def _manifest_for(root: Path) -> dict[str, str]:
    from victor_codegraph import build_manifest

    manifest = build_manifest(root)
    return {p: h for p, h in manifest.items()}


def _runner(tmp_path, provider, graph, encoder, force=False):

    current = _manifest_for(tmp_path)

    async def run():
        return await sync_intent_tier(
            tmp_path,
            graph,
            current_manifest=current,
            intent_store=IntentIndexStoreStub(provider, graph),
            profile_id=FakeEncoder.profile_id,
            encoder=encoder,
            force=force,
        )

    import asyncio

    return asyncio.run(run())


class IntentIndexStoreStub:
    """Real IntentIndexStore over the fakes (production construction path
    minus the LanceDB dependency)."""

    def __init__(self, provider, graph):
        from victor.storage.graph.intent_index import IntentIndexStore

        self._inner = IntentIndexStore(provider, graph)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TestIntentSync:
    def test_first_run_embeds_all_files(self, tmp_path):
        _write(tmp_path, "a.py", "def alpha():\n    return 1\n")
        _write(tmp_path, "sub/b.py", "def beta():\n    return 2\n")
        provider, graph, encoder = FakeProvider(), FakeGraphStore(), FakeEncoder()
        report = _runner(tmp_path, provider, graph, encoder)
        assert report is not None
        assert report.reembedded_files == 2
        assert report.chunks_embedded >= 2
        assert len(encoder.encoded_texts) == report.chunks_embedded
        assert graph.intent_synced_manifest().keys() == {p for p in _manifest_for(tmp_path)}

    def test_root_shortcut_skips_all_work(self, tmp_path):
        _write(tmp_path, "a.py", "def alpha():\n    return 1\n")
        provider, graph, encoder = FakeProvider(), FakeGraphStore(), FakeEncoder()
        _runner(tmp_path, provider, graph, encoder)
        calls_before = len(encoder.encoded_texts)
        report = _runner(tmp_path, provider, graph, encoder)
        assert report.skipped_by_root_shortcut is True
        assert report.chunks_embedded == 0
        assert len(encoder.encoded_texts) == calls_before

    def test_deep_edit_reembeds_only_changed_file(self, tmp_path):
        _write(tmp_path, "a/b/deep.py", "def deep():\n    return 1\n")
        _write(tmp_path, "a/other.py", "def other():\n    return 2\n")
        provider, graph, encoder = FakeProvider(), FakeGraphStore(), FakeEncoder()
        _runner(tmp_path, provider, graph, encoder)
        calls_before = len(encoder.encoded_texts)
        _write(tmp_path, "a/b/deep.py", "def deep():\n    return 99\n")
        report = _runner(tmp_path, provider, graph, encoder)
        assert report.reembedded_files == 1
        new_texts = encoder.encoded_texts[calls_before:]
        assert all("99" in t or "deep" in t for t in new_texts)
        assert not any("other" in t for t in new_texts)

    def test_removed_file_loses_vectors_and_marker(self, tmp_path):
        _write(tmp_path, "doomed.py", "def doomed():\n    return 1\n")
        provider, graph, encoder = FakeProvider(), FakeGraphStore(), FakeEncoder()
        _runner(tmp_path, provider, graph, encoder)
        assert provider.docs
        (tmp_path / "doomed.py").unlink()
        report = _runner(tmp_path, provider, graph, encoder)
        assert report.deleted_files == 1
        assert not provider.docs
        assert "doomed.py" not in graph.intent_synced_manifest()

    def test_force_full_reembed(self, tmp_path):
        _write(tmp_path, "a.py", "def alpha():\n    return 1\n")
        provider, graph, encoder = FakeProvider(), FakeGraphStore(), FakeEncoder()
        _runner(tmp_path, provider, graph, encoder)
        calls_before = len(encoder.encoded_texts)
        report = _runner(tmp_path, provider, graph, encoder, force=True)
        assert report.reembedded_files == 1
        assert len(encoder.encoded_texts) > calls_before

    def test_crash_between_vectors_and_marker_converges(self, tmp_path):
        """Vectors stored but the marker never commits -> the next run
        re-embeds the same file and the store stays duplicate-free."""
        _write(tmp_path, "a.py", "def alpha():\n    return 1\n")
        provider, graph, encoder = FakeProvider(), FakeGraphStore(), FakeEncoder()

        current = _manifest_for(tmp_path)

        import asyncio
        from victor.storage.graph.intent_index import IntentIndexStore

        async def crashy_run():
            store = IntentIndexStore(provider, graph)
            return await sync_intent_tier(
                tmp_path,
                graph,
                current_manifest=current,
                intent_store=store,
                profile_id=FakeEncoder.profile_id,
                encoder=encoder,
                _marker_dropout=True,
            )

        asyncio.run(crashy_run())
        assert provider.docs  # vectors landed
        assert not graph.intent_synced_manifest()  # marker dropped
        report = _runner(tmp_path, provider, graph, encoder)
        assert report.reembedded_files == 1
        # Idempotent chunk_id upserts: no duplicates after convergence.
        assert len(provider.docs) == report.chunks_embedded

    def test_flag_off_semantics_via_none(self, tmp_path):
        _write(tmp_path, "a.py", "def alpha():\n    return 1\n")
        provider, graph, encoder = FakeProvider(), FakeGraphStore(), None
        current = _manifest_for(tmp_path)

        import asyncio

        report = asyncio.run(
            sync_intent_tier(
                tmp_path,
                graph,
                current_manifest=current,
                intent_store=IntentIndexStoreStub(provider, graph),
                profile_id=FakeEncoder.profile_id,
                encoder=None,
            )
        )
        assert report is None
        assert not provider.docs
