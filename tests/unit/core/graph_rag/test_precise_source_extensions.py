"""Every extension declared by a compiler provider must survive source discovery."""

import pytest

from victor.core.graph_rag.config import GraphIndexConfig
from victor.core.graph_rag.indexing import GraphIndexingPipeline
from victor.core.graph_rag.precise_providers import (
    all_providers,
    extensions_for_provider,
    providers_for_files,
)


@pytest.mark.parametrize(
    "provider,extension",
    [
        (provider.name, extension)
        for provider in all_providers()
        for extension in sorted(extensions_for_provider(provider.name))
    ],
)
async def test_every_registered_extension_reaches_automatic_dispatch(tmp_path, provider, extension):
    source = tmp_path / ("sample" + extension)
    source.write_text("fixture\n")
    pipeline = object.__new__(GraphIndexingPipeline)
    pipeline.config = GraphIndexConfig(
        root_path=tmp_path, enable_ccg=False, enable_embeddings=False
    )
    files = await pipeline._discover_files(tmp_path)
    assert files == [source]
    assert set(providers_for_files([path.relative_to(tmp_path).as_posix() for path in files])) == {
        provider
    }


async def test_unregistered_unknown_file_is_not_forced_into_discovery(tmp_path):
    (tmp_path / "sample.unknown").write_text("fixture\n")
    pipeline = object.__new__(GraphIndexingPipeline)
    pipeline.config = GraphIndexConfig(
        root_path=tmp_path, enable_ccg=False, enable_embeddings=False
    )
    assert await pipeline._discover_files(tmp_path) == []
    assert providers_for_files(["sample.unknown"]) == {}
