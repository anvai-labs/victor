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

"""Intent-tier reads through the REGISTERED search tool (Codex findings 7+21).

The tool-level factory shim used to hand ``intent_seed_scores`` an ASYNC
factory; the seeding core calls factories synchronously, so with the intent
flag on every explicit-intent (and hybrid intent-leg) query died with
"'coroutine' object has no attribute 'search'" — reported as an unavailable
leg before the index was ever touched, while hybrid could silently hide the
missing contribution.

These tests drive the registered ``graph_semantic_search`` tool end to end
over one indexed SQLite fixture (real graph store + real intent chunk table
+ production ``build_intent_index`` construction path; only the embedding
MODEL is a deterministic stand-in) and require a known symbol to contribute
a score in BOTH modes.
"""

from __future__ import annotations

import hashlib
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.config.settings import Settings
from victor.storage.graph.intent_sqlite_store import SqliteIntentChunkStore
from victor.storage.graph.protocol import GraphEdge, GraphNode
from victor.storage.graph.sqlite_store import SqliteGraphStore
from victor.tools import graph_query_tool
from victor.tools.graph_query_tool import graph_semantic_search

QUERY = "rate limit parser"

_NODES = [
    GraphNode(node_id="n_alpha", type="function", name="alpha_parser", file="src/a.py", line=1),
    GraphNode(node_id="n_beta", type="function", name="beta_writer", file="src/b.py", line=1),
    GraphNode(node_id="n_gamma", type="function", name="gamma_commit", file="src/c.py", line=1),
]

_EDGES = [
    GraphEdge(src="n_alpha", dst="n_beta", type="CALLS"),
    GraphEdge(src="n_beta", dst="n_gamma", type="CALLS"),
]

_DIM = 384


class _HashEncoder:
    """Deterministic token-bag encoder (the model is NOT under test)."""

    profile_id = "test-profile"

    def encode(self, texts):
        return [self._one(t) for t in texts]

    def encode_query(self, query):
        return self._one(query)

    @staticmethod
    def _one(text):
        vec = [0.0] * _DIM
        for token in str(text).lower().split():
            digest = int(hashlib.sha256(token.encode()).hexdigest(), 16)
            vec[digest % _DIM] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]


def _chunk_doc(symbol_id: str, content: str) -> dict:
    return {
        "id": f"{symbol_id}#body#0",
        "vector": _HashEncoder._one(content),
        "content": content,
        "metadata": {
            "symbol_oid": f"oid-{symbol_id}",
            "symbol_id": symbol_id,
            "file_path": "src/a.py" if symbol_id == "n_alpha" else "src/b.py",
            "input_hash": "i" * 64,
            "source_hash": "s" * 64,
            "profile_id": _HashEncoder.profile_id,
        },
    }


async def _indexed_workspace(tmp_path) -> SqliteGraphStore:
    """One indexed fixture: graph nodes/edges + intent chunk vectors."""
    store = SqliteGraphStore(project_path=tmp_path)
    await store.initialize()
    await store.upsert_nodes(_NODES)
    await store.upsert_edges(_EDGES)
    provider = SqliteIntentChunkStore(store)
    await provider.index_embedded_documents(
        [
            _chunk_doc("n_alpha", "rate limit the parser requests"),
            _chunk_doc("n_beta", "beta writes config files"),
        ]
    )
    return store


def _enabled_flags():
    manager = MagicMock()
    manager.is_enabled.return_value = True
    return manager


def _settings(**kwargs) -> Settings:
    import os

    for key in list(os.environ):
        if key.startswith("VICTOR_") and "INTENT" in key.upper():
            os.environ.pop(key)
    return Settings(**kwargs)


@pytest.fixture
async def indexed_root(tmp_path):
    """The indexed fixture root: graph nodes/edges + intent chunk vectors."""
    store = await _indexed_workspace(tmp_path)
    yield tmp_path
    await store.close()


@pytest.fixture
def intent_encoder(monkeypatch):
    """The production construction path stays; only the model is faked."""
    from victor.core.graph_rag import seeding

    monkeypatch.setattr(seeding, "chunk_encoder_factory", lambda: _HashEncoder)


# ── the registered tool + indexed fixture, both modes ────────────────────────


async def test_registered_tool_intent_mode_scores_known_symbol(indexed_root, intent_encoder):
    """mode="intent" with the flag on: the indexed symbol contributes a score.

    Regression: the async factory shim made this return empty scores with a
    coroutine attribute error before touching the index.
    """
    from victor.tools.metadata_registry import (
        get_tool_metadata,
        register_tool_metadata,
    )

    # Registered: the tool the framework actually dispatches (the .Tool
    # instance the shared registry stores and executes). The session
    # autouse fixture resets the metadata registry between tests, so
    # (re-)register the decorated tool the way first import does.
    if get_tool_metadata("graph_semantic_search") is None:
        register_tool_metadata(graph_semantic_search.Tool)
    assert get_tool_metadata("graph_semantic_search") is not None
    assert graph_semantic_search.Tool.name == "graph_semantic_search"

    with (
        patch(
            "victor.core.feature_flags.get_feature_flag_manager",
            return_value=_enabled_flags(),
        ),
        patch(
            "victor.config.settings.get_settings",
            return_value=_settings(search={"graph": {"codebase_intent_indexing_enabled": True}}),
        ),
    ):
        out = await graph_semantic_search.Tool.execute(
            {}, query=QUERY, path=str(indexed_root), mode="intent", max_results=10
        )

    assert out.success, out.error
    payload = out.output
    assert "error" not in payload, payload.get("error")
    assert payload["metadata"]["mode"] == "intent"
    assert payload["metadata"]["seed_count"] >= 1, payload["metadata"]
    # The seeded note must be absent: the leg was AVAILABLE and contributed.
    assert "note" not in payload["metadata"], payload["metadata"].get("note")
    # A known symbol contributes a score through the intent tier.
    scored = {r["name"]: r["relevance_score"] for r in payload["results"]}
    assert scored.get("alpha_parser", 0.0) > 0.0, scored


async def test_registered_tool_hybrid_mode_fuses_intent_leg(indexed_root, intent_encoder):
    """mode="hybrid" with the flag on (codebase mirror spelling): the intent
    leg contributes even for a node the vector/keyword legs miss."""
    from types import SimpleNamespace as NS

    vector_provider = MagicMock()
    vector_provider.search_similar = AsyncMock(
        return_value=[NS(metadata={"node_id": "n_beta"}, content="beta", score=0.9)]
    )

    with (
        patch(
            "victor.core.feature_flags.get_feature_flag_manager",
            return_value=_enabled_flags(),
        ),
        # Finding 21: the flag reaches the runtime through the codebase-group
        # mirror spelling, not only the canonical field.
        patch(
            "victor.config.settings.get_settings",
            return_value=_settings(codebase={"codebase_intent_indexing_enabled": True}),
        ),
        patch(
            "victor.tools.graph_query_tool._vector_provider_for",
            return_value=vector_provider,
        ),
    ):
        out = await graph_semantic_search.Tool.execute(
            {}, query=QUERY, path=str(indexed_root), mode="hybrid", max_results=10
        )

    assert out.success, out.error
    payload = out.output
    assert "error" not in payload, payload.get("error")
    names = {r["name"] for r in payload["results"]}
    # n_beta came from the vector leg; n_alpha ONLY the intent leg reaches.
    assert "beta_writer" in names, names
    scored = {r["name"]: r["relevance_score"] for r in payload["results"]}
    assert scored.get("alpha_parser", 0.0) > 0.0, scored
    # The seeding report must show the fused legs, not a hidden miss.
    assert payload["metadata"]["seeding"]["fused_seeds"] >= 2, payload["metadata"]


# ── the factory contract (negative controls) ─────────────────────────────────


def test_intent_factory_shim_is_synchronous():
    """The core seeding contract: factories are called synchronously.

    Pins the defect: an async factory returned a coroutine that the helper
    then called .search on.
    """
    assert inspect.iscoroutinefunction(graph_query_tool._intent_provider_for) is False


async def test_missing_intent_index_degrades_with_note_not_coroutine(intent_encoder):
    """No index -> clean 'unavailable leg' note; never a coroutine error."""
    retriever = MagicMock()
    from victor.core.graph_rag.retrieval import RetrievalResult

    retriever.expand_from_seeds = AsyncMock(
        return_value=RetrievalResult(nodes=[], edges=[], subgraphs=[], query="q")
    )
    store = SimpleNamespace()  # no _connect / record_intent_synced -> no provider
    store.initialize = AsyncMock()

    with (
        patch(
            "victor.core.feature_flags.get_feature_flag_manager",
            return_value=_enabled_flags(),
        ),
        patch(
            "victor.config.settings.get_settings",
            return_value=_settings(search={"graph": {"codebase_intent_indexing_enabled": True}}),
        ),
        patch("victor.storage.graph.create_graph_store", return_value=store),
        patch("victor.core.graph_rag.MultiHopRetriever", return_value=retriever),
    ):
        out = await graph_semantic_search.Tool.execute(
            {}, query="q", path=".", mode="intent", max_results=5
        )

    assert out.success
    note = out.output["metadata"].get("note", "")
    assert "no intent index provider" in note, note
    # The old failure surfaced as "'coroutine' object has no attribute 'search'".
    assert "coroutine" not in note, note


async def test_flag_off_never_touches_the_index(indexed_root):
    """Flag off (explicit): the intent leg is skipped without constructing
    the provider — the encoder factory must never fire."""
    from victor.core.graph_rag import seeding

    def _must_not_build():
        raise AssertionError("intent index constructed while the flag was off")

    retriever = MagicMock()
    from victor.core.graph_rag.retrieval import RetrievalResult

    retriever.expand_from_seeds = AsyncMock(
        return_value=RetrievalResult(nodes=[], edges=[], subgraphs=[], query=QUERY)
    )

    with (
        patch(
            "victor.core.feature_flags.get_feature_flag_manager",
            return_value=_enabled_flags(),
        ),
        patch(
            "victor.config.settings.get_settings",
            return_value=_settings(search={"graph": {"codebase_intent_indexing_enabled": False}}),
        ),
        patch.object(seeding, "chunk_encoder_factory", _must_not_build),
        patch("victor.core.graph_rag.MultiHopRetriever", return_value=retriever),
    ):
        out = await graph_semantic_search.Tool.execute(
            {}, query=QUERY, path=str(indexed_root), mode="intent", max_results=5
        )

    assert out.success
    assert "intent leg disabled" in out.output["metadata"].get("note", "")
