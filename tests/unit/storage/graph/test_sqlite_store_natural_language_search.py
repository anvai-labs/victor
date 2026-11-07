"""Natural-language keyword search on the sqlite graph store.

FTS5 requires every term of a query to match, so a question such as "where
are snapshot manifests parsed" matched nothing on a real Apache Iceberg index.
``search_symbols(match="any")`` matches any meaningful term instead; the
default stays exact for symbol lookups.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from victor.storage.graph.protocol import GraphNode
from victor.storage.graph.sqlite_store import SqliteGraphStore, natural_language_terms


@pytest.fixture
async def store(tmp_path: Path):
    graph = SqliteGraphStore(project_path=tmp_path)
    await graph.initialize()
    await graph.upsert_nodes(
        [
            GraphNode(
                node_id="n_parse",
                type="function",
                name="read_manifest",
                file="src/manifests.py",
                line=1,
                docstring="Parse the snapshot manifest list.",
            ),
            GraphNode(
                node_id="n_cafe",
                type="function",
                name="brew",
                file="src/cafe.py",
                line=1,
                docstring="Serves a café order.",
            ),
            GraphNode(
                node_id="n_other", type="function", name="unrelated", file="src/x.py", line=1
            ),
        ]
    )
    yield graph
    await graph.close()


def test_terms_drop_stopwords_short_words_and_repeats():
    assert natural_language_terms("Where are the snapshot manifests, and the snapshot list?") == [
        "snapshot",
        "manifests",
        "list",
    ]
    assert natural_language_terms("how is it a to do") == []
    assert len(natural_language_terms(" ".join(f"term{i}" for i in range(40)))) == 16


@pytest.mark.asyncio
async def test_a_question_matches_nothing_by_default_but_any_term_finds_it(store):
    question = "where is the snapshot manifest parsed"
    assert await store.search_symbols(question, limit=5) == []
    found = await store.search_symbols(question, limit=5, match="any")
    assert [n.node_id for n in found] == ["n_parse"]


@pytest.mark.asyncio
async def test_exact_lookups_are_unchanged(store):
    assert [n.node_id for n in await store.search_symbols("read_manifest", limit=5)] == ["n_parse"]


@pytest.mark.asyncio
async def test_a_question_of_only_stopwords_matches_nothing(store):
    assert await store.search_symbols("how is it done", limit=5, match="any") == []


@pytest.mark.asyncio
async def test_unknown_match_modes_are_refused(store):
    with pytest.raises(ValueError, match="match must be"):
        await store.search_symbols("snapshot", match="some")


def test_terms_follow_the_fts5_tokenizer_for_accents_and_scripts():
    assert natural_language_terms("café latte") == ["cafe", "latte"]
    assert natural_language_terms("größe xyz") == ["große", "xyz"]
    assert natural_language_terms("日本語 thing") == ["日本語", "thing"]


@pytest.mark.asyncio
async def test_accented_questions_match_the_folded_index(store):
    found = await store.search_symbols("where is café ordering", limit=5, match="any")
    assert [n.node_id for n in found] == ["n_cafe"]


@pytest.mark.asyncio
async def test_any_term_matches_without_fts5_through_per_term_like(store):
    import sqlite3

    conn = store._connect()
    conn.execute("DROP TABLE graph_node_fts")
    conn.commit()
    found = await store.search_symbols("snapshot zzzunmatched", limit=5, match="any")
    assert [n.node_id for n in found] == ["n_parse"]
    single = await store.search_symbols("manifest", limit=5, match="any")
    assert [n.node_id for n in single] == ["n_parse"]
    assert isinstance(conn, sqlite3.Connection)
