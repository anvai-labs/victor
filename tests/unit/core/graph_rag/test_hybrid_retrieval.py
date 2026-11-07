"""Public hybrid retrieval and read-only graph stores.

Retrieval used to seed only from the store's keyword search, so a caller that
asked for "semantic" retrieval never touched the indexed embeddings, and a
query-only process still wrote into the project database (schema refresh,
project metadata, persisted query cache). These tests pin the public contract:

- ``RetrievalConfig(mode="hybrid")`` fuses keyword and vector seeds and says,
  in ``result.metadata["seeding"]``, whether the vector leg contributed;
- ``create_graph_store(..., read_only=True)`` never changes database content
  (SQLite may create empty ``-wal``/``-shm`` coordination files, which a WAL
  reader requires; see SqliteGraphStore).
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.core.graph_rag import MultiHopRetriever, RetrievalConfig, SeedReport
from victor.storage.graph import create_graph_store
from victor.storage.graph.protocol import GraphEdge, GraphNode
from victor.storage.graph.sqlite_store import SqliteGraphStore


def _node(node_id: str, name: str) -> GraphNode:
    return GraphNode(node_id=node_id, type="function", name=name, file=f"src/{name}.py", line=1)


async def _build_index(root: Path) -> None:
    store = SqliteGraphStore(project_path=root)
    await store.initialize()
    await store.upsert_nodes(
        [
            _node("n_alpha", "alpha_parser"),
            _node("n_beta", "beta_writer"),
            _node("n_gamma", "gamma_commit"),
        ]
    )
    await store.upsert_edges(
        [
            GraphEdge(src="n_alpha", dst="n_beta", type="CALLS"),
            GraphEdge(src="n_beta", dst="n_gamma", type="CALLS"),
        ]
    )
    await store.close()


def _db_bytes(root: Path) -> dict:
    """Digest of the database and its write-ahead log (the durable content).

    ``-shm`` is SQLite's shared-memory wal-index, which readers update by
    design; it holds no database content and is not compared.
    """
    victor_dir = root / ".victor"
    return {
        name: hashlib.sha256((victor_dir / name).read_bytes()).hexdigest()
        for name in ("project.db", "project.db-wal")
        if (victor_dir / name).exists()
    }


@pytest.fixture
async def index_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    await _build_index(root)
    return root


# ── read-only stores ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_only_store_queries_without_changing_the_database(index_root):
    before = _db_bytes(index_root)
    store = create_graph_store("sqlite", index_root, read_only=True)
    assert store.read_only is True
    await store.initialize()
    found = await store.search_symbols("alpha_parser", limit=5)
    assert [n.node_id for n in found] == ["n_alpha"]
    config = RetrievalConfig(seed_count=3, top_k=5, max_hops=2, direction="both")
    result = await MultiHopRetriever(store, config).retrieve("alpha_parser", config)
    assert "n_alpha" in {n.node_id for n in result.nodes}
    await store.close()
    assert _db_bytes(index_root) == before


@pytest.mark.asyncio
async def test_read_only_over_a_checkpointed_index_leaves_content_and_no_wal_data(
    index_root, tmp_path
):
    # A published index is fully checkpointed: project.db alone, no -wal file.
    # Reading it must leave project.db byte-identical; SQLite may create its
    # coordination files, which must stay empty.
    pinned = tmp_path / "pinned"
    (pinned / ".victor").mkdir(parents=True)
    with sqlite3.connect(index_root / ".victor" / "project.db") as src:
        with sqlite3.connect(pinned / ".victor" / "project.db") as dst:
            src.backup(dst)
    with sqlite3.connect(pinned / ".victor" / "project.db") as conn:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    for extra in ("project.db-wal", "project.db-shm"):
        (pinned / ".victor" / extra).unlink(missing_ok=True)
    before = hashlib.sha256((pinned / ".victor" / "project.db").read_bytes()).hexdigest()

    store = create_graph_store("sqlite", pinned, read_only=True)
    await store.initialize()
    assert [n.node_id for n in await store.search_symbols("beta_writer", limit=3)] == ["n_beta"]
    await store.close()

    after = hashlib.sha256((pinned / ".victor" / "project.db").read_bytes()).hexdigest()
    assert after == before
    wal = pinned / ".victor" / "project.db-wal"
    assert not wal.exists() or wal.stat().st_size == 0


@pytest.mark.asyncio
async def test_an_ordinary_open_does_write_which_is_why_read_only_exists(index_root):
    # Control for the test above: the read-write open path refreshes project
    # metadata and the schema, so the byte comparison can detect a write.
    before = _db_bytes(index_root)
    store = SqliteGraphStore(project_path=index_root)
    await store.initialize()
    await store.close()
    assert _db_bytes(index_root) != before


@pytest.mark.asyncio
async def test_read_only_store_refuses_writes(index_root):
    store = SqliteGraphStore(project_path=index_root, read_only=True)
    await store.initialize()
    with pytest.raises(sqlite3.OperationalError):
        await store.upsert_nodes([_node("n_new", "delta")])
    await store.close()


@pytest.mark.asyncio
async def test_read_only_connection_has_both_independent_write_guards(index_root):
    # mode=ro and query_only each refuse writes alone; pin both so neither can
    # be dropped unnoticed behind the other.
    store = SqliteGraphStore(project_path=index_root, read_only=True)
    conn = store._connect()
    assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
    rows = conn.execute("PRAGMA database_list").fetchall()
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("PRAGMA query_only=OFF")
        conn.execute("CREATE TABLE probe (x)")
    assert rows[0][2].endswith("project.db")
    await store.close()


def test_read_only_store_requires_an_existing_database(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        SqliteGraphStore(project_path=empty, read_only=True)
    assert not (empty / ".victor").exists()  # nothing created


def test_backends_without_a_read_only_mode_refuse_it(tmp_path):
    with pytest.raises(ValueError, match="no read-only mode"):
        create_graph_store("memory", tmp_path, read_only=True)


@pytest.mark.asyncio
async def test_read_only_store_bypasses_the_query_cache(index_root):
    config = RetrievalConfig(seed_count=3, top_k=5, direction="both")
    cache = MagicMock()
    cache.get.return_value = None
    with patch("victor.core.graph_rag.query_cache.get_graph_query_cache", return_value=cache):
        ro = SqliteGraphStore(project_path=index_root, read_only=True)
        await MultiHopRetriever(ro, config).retrieve("alpha_parser", config)
        await ro.close()
        cache.get.assert_not_called()
        cache.put.assert_not_called()

        rw = SqliteGraphStore(project_path=index_root)  # control: cache is used
        await MultiHopRetriever(rw, config).retrieve("alpha_parser", config)
        await rw.close()
        cache.get.assert_called()
        cache.put.assert_called()


# ── hybrid retrieval through the public retriever ───────────────────────────


def _hit(node_id: str, score: float):
    return SimpleNamespace(metadata={"node_id": node_id}, content=node_id, score=score)


def _provider(search_similar):
    provider = MagicMock()
    provider.search_similar = search_similar
    return provider


async def _hybrid(index_root, provider):
    store = SqliteGraphStore(project_path=index_root, read_only=True)
    config = RetrievalConfig(seed_count=3, top_k=5, max_hops=1, direction="both", mode="hybrid")
    with patch("victor.core.graph_rag.seeding.vector_provider_for", return_value=provider):
        result = await MultiHopRetriever(store, config).retrieve("alpha_parser", config)
    await store.close()
    return result


@pytest.mark.asyncio
async def test_hybrid_mode_fuses_vector_seeds_and_says_so(index_root):
    # The vector leg finds n_gamma, which keyword search for "alpha_parser"
    # cannot; it must reach the seeds and the report must name both legs.
    provider = _provider(AsyncMock(return_value=[_hit("n_gamma", 0.9)]))
    result = await _hybrid(index_root, provider)
    seeding = result.metadata["seeding"]
    assert seeding["mode"] == "hybrid"
    assert seeding["semantic_leg"] == "used"
    assert seeding["keyword_seeds"] >= 1 and seeding["semantic_seeds"] == 1
    assert "n_gamma" in result.seed_nodes and "n_alpha" in result.seed_nodes
    provider.search_similar.assert_awaited_once()


@pytest.mark.parametrize(
    "provider, leg, note_fragment",
    [
        (None, "unavailable", "no vector store provider"),
        (
            _provider(AsyncMock(side_effect=ValueError("query encoder profile does not match"))),
            "unavailable",
            "query encoder profile does not match",
        ),
        (_provider(AsyncMock(return_value=[])), "empty", "victor graph index --embeddings"),
    ],
    ids=["no-provider", "encoder-mismatch", "no-vectors"],
)
@pytest.mark.asyncio
async def test_hybrid_mode_reports_why_the_vector_leg_did_not_contribute(
    index_root, provider, leg, note_fragment
):
    result = await _hybrid(index_root, provider)
    seeding = result.metadata["seeding"]
    assert seeding["semantic_leg"] == leg
    assert seeding["semantic_seeds"] == 0
    assert note_fragment in seeding["note"]
    assert result.seed_nodes == ["n_alpha"]  # keyword seeds still answer


@pytest.mark.asyncio
async def test_structural_mode_reports_no_vector_leg(index_root):
    store = SqliteGraphStore(project_path=index_root, read_only=True)
    config = RetrievalConfig(seed_count=3, top_k=5, mode="structural")
    result = await MultiHopRetriever(store, config).retrieve("alpha_parser", config)
    await store.close()
    assert result.metadata["seeding"]["semantic_leg"] == "not-requested"
    assert result.seed_nodes == ["n_alpha"]


@pytest.mark.asyncio
async def test_seeded_modes_are_not_served_from_a_cache_that_drops_their_report(index_root):
    cache = MagicMock()
    with patch("victor.core.graph_rag.query_cache.get_graph_query_cache", return_value=cache):
        provider = _provider(AsyncMock(return_value=[]))
        store = SqliteGraphStore(project_path=index_root)
        config = RetrievalConfig(seed_count=3, top_k=5, mode="hybrid")
        with patch("victor.core.graph_rag.seeding.vector_provider_for", return_value=provider):
            result = await MultiHopRetriever(store, config).retrieve("alpha_parser", config)
        await store.close()
    cache.get.assert_not_called()
    cache.put.assert_not_called()
    assert "seeding" in result.metadata


def test_seed_report_is_a_plain_record():
    report = SeedReport("hybrid", 2, 1, 3, "used")
    assert report.to_dict() == {
        "mode": "hybrid",
        "keyword_seeds": 2,
        "semantic_seeds": 1,
        "fused_seeds": 3,
        "semantic_leg": "used",
        "note": None,
    }
