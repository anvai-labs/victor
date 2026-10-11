"""Graph indexing must reject incomplete installs before touching an index."""

from unittest.mock import AsyncMock, patch
from types import SimpleNamespace
from importlib import import_module

import pytest

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


async def test_complete_graph_install_reaches_indexing(tmp_path):
    from victor.core.graph_rag.indexing import GraphIndexStats

    store = AsyncMock()
    with (
        patch("victor.core.graph_rag.dependencies.require_graph_dependencies"),
        patch.object(graph, "create_graph_store", return_value=store),
        patch("victor.core.graph_rag.GraphIndexingPipeline") as pipeline,
        patch(
            "victor.core.graph_rag.indexing.run_indexing_with_lock",
            new_callable=AsyncMock,
            return_value=GraphIndexStats(),
        ),
    ):
        assert await graph._index_async(str(tmp_path), False, False)
    store.initialize.assert_awaited_once()
    pipeline.assert_called_once()


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
