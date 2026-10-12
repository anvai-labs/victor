"""Graph indexing must reject incomplete installs before touching an index."""

from unittest.mock import AsyncMock, patch
from types import SimpleNamespace
from importlib import import_module

import pytest

from rich.console import Console

from victor.ui.commands import graph


@pytest.mark.parametrize("force", [False, True])
async def test_missing_graph_dependencies_preserve_existing_index(tmp_path, force):
    with (
        patch(
            "victor.core.graph_rag.dependencies.require_graph_dependencies",
            side_effect=RuntimeError("Install victor-ai[graph]: tree_sitter_rust unavailable"),
        ),
        patch.object(graph, "create_graph_store") as create,
    ):
        assert not await graph._index_async(str(tmp_path), True, force)
    create.assert_not_called()


@pytest.mark.parametrize(
    "error_count,partial_count,omitted_count", [(0, 0, 0), (0, 1, 3), (12, 1, 3)]
)
async def test_graph_index_reports_coverage_and_failure_details(
    tmp_path, error_count, partial_count, omitted_count
):
    from victor.core.graph_rag.indexing import GraphIndexStats

    store = AsyncMock()
    output = Console(record=True, width=160)
    stats = GraphIndexStats(
        files_processed=2,
        files_unchanged=1,
        files_partial=partial_count,
        nodes_created=9,
        edges_created=12,
        persisted_nodes=7,
        persisted_edges=8,
        error_count=error_count,
        errors=[f"src/[module]{i}.py: parse failed" for i in range(error_count)],
        repository_relations_dropped_unmaterialized=omitted_count,
    )
    with (
        patch.object(graph, "console", output),
        patch("victor.core.graph_rag.dependencies.require_graph_dependencies"),
        patch.object(graph, "create_graph_store", return_value=store),
        patch("victor.core.graph_rag.GraphIndexingPipeline") as pipeline,
        patch(
            "victor.core.graph_rag.indexing.run_indexing_with_lock",
            new_callable=AsyncMock,
            return_value=stats,
        ),
    ):
        assert await graph._index_async(str(tmp_path), False, False) is (error_count == 0)
    store.initialize.assert_awaited_once()
    pipeline.assert_called_once()
    rendered = output.export_text()
    for label in (
        f"Partial parses this pass: {partial_count}",
        "Node upserts attempted: 9",
        "Edge upserts attempted: 12",
        "Persisted nodes: 7",
        "Persisted edges: 8",
        "Unchanged files: 1",
        "Existing file diagnostics are retained",
    ):
        assert label in rendered
    assert "Nodes created:" not in rendered and "Edges created:" not in rendered
    if omitted_count:
        assert f"Relations omitted (unmaterialized endpoints): {omitted_count}" in rendered
    assert ("✓ Indexing pass complete" in rendered) is not bool(
        error_count or partial_count or omitted_count
    )
    if error_count:
        assert "Indexing incomplete" in rendered
        assert "Errors: 12" in rendered
        assert "src/[module]0.py: parse failed" in rendered
        assert "src/[module]10.py" not in rendered
        assert "2 additional errors" in rendered
    elif partial_count or omitted_count:
        assert "Indexing finished with coverage gaps" in rendered


@pytest.mark.parametrize("missing", ["victor_codegraph", "tree_sitter_rust"])
def test_preflight_names_missing_requirement_and_install_extra(missing):
    from victor.core.graph_rag.dependencies import require_graph_dependencies

    def load(name):
        if name == missing:
            raise ModuleNotFoundError(name)
        if name == "victor_codegraph":
            return SimpleNamespace(
                parse=lambda *a, **kw: SimpleNamespace(status=SimpleNamespace(value="success"))
            )
        return import_module(name)

    with patch("victor.core.graph_rag.dependencies.import_module", side_effect=load):
        with pytest.raises(RuntimeError, match=missing) as error:
            require_graph_dependencies()
    assert 'pip install "victor-ai[graph]"' in str(error.value)
