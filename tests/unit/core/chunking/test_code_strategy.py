from __future__ import annotations

from pathlib import Path

import pytest

from victor.core.chunking.base import ChunkingConfig
from victor.core.chunking.registry import ChunkingRegistry
from victor.core.chunking.strategies.code import CodeChunkingStrategy


def test_registry_routes_source_and_language_to_codegraph(tmp_path: Path) -> None:
    pytest.importorskip("victor_codegraph")
    source = tmp_path / "sample.py"
    content = (
        "import os\n\n"
        "def alpha(value):\n"
        "    return value + 1\n\n"
        "def beta(value):\n"
        "    return alpha(value)\n"
    )
    strategy = ChunkingRegistry(
        ChunkingConfig(chunk_size=48, chunk_overlap=8, min_chunk_size=0, max_chunk_size=64)
    )

    chunks = strategy.chunk(content, source=str(source))

    symbol_names = {chunk.metadata.get("symbol_name") for chunk in chunks}
    assert {"alpha", "beta"} <= symbol_names
    assert all(chunk.length <= 64 for chunk in chunks)
    assert all(chunk.metadata["strategy"] == "victor_codegraph_v2" for chunk in chunks)
    assert all(chunk.metadata["file_path"] == str(source) for chunk in chunks)


def test_code_strategy_preserves_legacy_fallback(monkeypatch) -> None:
    codegraph = pytest.importorskip("victor_codegraph")
    strategy = CodeChunkingStrategy(
        ChunkingConfig(chunk_size=64, chunk_overlap=0, min_chunk_size=0, max_chunk_size=80)
    )
    monkeypatch.setattr(
        codegraph, "chunk", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError())
    )

    chunks = strategy.chunk("def fallback():\n    return 1\n", language="python")

    assert len(chunks) == 1
    assert chunks[0].metadata["symbol_name"] is None
    assert chunks[0].metadata["strategy"] == "plain_code_fallback"
