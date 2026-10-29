"""Range resolution must not depend on graph-store insertion order."""

from itertools import permutations
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from victor.storage.graph.protocol import GraphNode
from victor.tools.graph_query_tool import _resolve_target as tool_target
from victor.ui.commands.graph import _resolve_target as cli_target


def node(identifier, start, end, file="src/service.ts"):
    return GraphNode(identifier, "method", identifier, file, start, end)


@pytest.mark.parametrize("resolve", [tool_target, cli_target])
async def test_nested_constructor_is_selected_independent_of_store_order(resolve):
    # Representative class/constructor ranges from the pinned AgentBrowser index.
    nodes = [node("class", 791, 6715), node("constructor", 910, 1059)]
    for ordered in permutations(nodes):
        store = SimpleNamespace(
            get_node_by_id=AsyncMock(return_value=None),
            get_nodes_by_file=AsyncMock(return_value=list(ordered)),
            find_nodes=AsyncMock(return_value=[]),
        )
        for line in (910, 1000, 1059):
            assert await resolve(f"src/service.ts:{line}", store) == "constructor"
        assert await resolve("src/service.ts:1060", store) == "class"


@pytest.mark.parametrize("resolve", [tool_target, cli_target])
async def test_ambiguous_or_invalid_ranges_do_not_guess_or_fall_back_to_name(resolve):
    for nodes in (
        [node("a", 10, 20), node("b", 10, 20)],
        [node("bad", 20, 10), node("missing", None, 30), node("bool", True, 20)],
        [],
    ):
        store = SimpleNamespace(
            get_node_by_id=AsyncMock(return_value=None),
            get_nodes_by_file=AsyncMock(return_value=nodes),
            find_nodes=AsyncMock(return_value=[node("unrelated-name", 1, 1)]),
        )
        assert await resolve("src/service.ts:15", store) is None
        store.find_nodes.assert_not_called()


@pytest.mark.parametrize("resolve", [tool_target, cli_target])
async def test_exact_node_id_disambiguates_overlapping_ranges(resolve):
    store = SimpleNamespace(get_node_by_id=AsyncMock(return_value=node("chosen", 10, 20)))
    assert await resolve("chosen", store) == "chosen"


@pytest.mark.parametrize("resolve", [tool_target, cli_target])
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_store_owned_absolute_relative_path_aliases(resolve, backend, tmp_path, monkeypatch):
    from victor.storage.graph.memory_store import MemoryGraphStore
    from victor.storage.graph.sqlite_store import SqliteGraphStore

    monkeypatch.chdir(tmp_path)
    store = MemoryGraphStore() if backend == "memory" else SqliteGraphStore(tmp_path)
    await store.initialize()
    try:
        path = str((tmp_path / "src/service.ts").resolve())
        await store.upsert_nodes(
            [node("class", 791, 6715, path), node("constructor", 910, 1059, path)]
        )
        assert await resolve("src/service.ts:1000", store) == "constructor"
        assert await resolve(f"{path}:1000", store) == "constructor"
    finally:
        await store.close()


@pytest.mark.parametrize("resolve", [tool_target, cli_target])
async def test_declaration_without_end_line_matches_only_its_known_line(resolve):
    store = SimpleNamespace(
        get_node_by_id=AsyncMock(return_value=None),
        get_nodes_by_file=AsyncMock(return_value=[node("declaration", 10, None)]),
    )
    assert await resolve("src/service.ts:10", store) == "declaration"
    assert await resolve("src/service.ts:11", store) is None
