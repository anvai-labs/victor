"""Tests for ``victor.framework.ingestion.models``."""

from __future__ import annotations

import pytest
from victor_contracts import chunking as sdk_chunking

from victor.framework import ingestion
from victor.framework.ingestion.models import Chunk, ChunkingConfig, DocumentType, SourceContent


def test_chunking_config_is_the_sdk_class():
    assert ChunkingConfig is sdk_chunking.ChunkingConfig
    assert ingestion.ChunkingConfig is sdk_chunking.ChunkingConfig


def test_chunking_config_defaults_unchanged():
    cfg = ChunkingConfig()
    assert (cfg.chunk_size, cfg.chunk_overlap, cfg.min_chunk_size, cfg.max_chunk_size) == (
        1344,
        134,
        200,
        2000,
    )
    assert cfg.respect_sentence_boundaries and cfg.respect_paragraph_boundaries and cfg.code_aware


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"min_chunk_size": -1},
        {"chunk_size": 100, "chunk_overlap": 100},
        {"chunk_overlap": -1},
        {"chunk_overlap": 50, "min_chunk_size": 10},
    ],
)
def test_chunking_config_rejects_non_terminating_settings(kwargs):
    with pytest.raises(ValueError):
        ChunkingConfig(**kwargs)


def test_chunk_length_and_dict():
    chunk = Chunk(content="héllo", start_char=3, end_char=8, chunk_index=2, metadata={"a": 1})
    assert chunk.length == 5
    assert chunk.to_dict() == {
        "content": "héllo",
        "start_char": 3,
        "end_char": 8,
        "chunk_index": 2,
        "doc_type": "text",
        "metadata": {"a": 1},
    }


def test_chunk_metadata_is_not_shared():
    first, second = Chunk("a", 0, 1), Chunk("b", 1, 2)
    first.metadata["k"] = "v"
    assert second.metadata == {}


def test_source_content_defaults_to_auto():
    source = SourceContent(content="x", source="file.txt")
    assert source.doc_type is DocumentType.AUTO
    assert source.metadata == {}
