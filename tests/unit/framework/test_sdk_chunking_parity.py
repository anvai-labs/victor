"""Parity between the SDK chunking core and the framework ``BaseChunker``.

``victor_contracts.chunking.TextChunker`` carries the text/markdown/code
strategies that ``victor.framework.ingestion.chunker.BaseChunker`` still
implements locally until it delegates to the SDK. This differential test
fails if either copy drifts before then.
"""

from __future__ import annotations

import random

import pytest
from victor_contracts.chunking import ChunkingConfig as SdkChunkingConfig
from victor_contracts.chunking import TextChunker

from victor.framework.ingestion.chunker import BaseChunker
from victor.framework.ingestion.models import ChunkingConfig

_PIECES = ["alpha", "beta.", "gamma!", "delta?", "\n", "\n\n", "# Head", "## Sub", "def f():"]
_PIECES += ["class C:", "fn g()", "  ", "x = 1", "ünï", "```"]


def _corpus(seed: int) -> str:
    rng = random.Random(seed)
    lines = []
    for _ in range(rng.randint(0, 200)):
        piece = " ".join(rng.choice(_PIECES) for _ in range(rng.randint(1, 8)))
        lines.append(piece)
    return "\n".join(lines)


def _config(seed: int) -> dict[str, int | bool]:
    rng = random.Random(seed * 7 + 1)
    size = rng.randint(40, 600)
    min_size = rng.randint(0, 60)
    return {
        "chunk_size": size,
        "chunk_overlap": rng.randint(0, min(size - 1, min_size)),
        "min_chunk_size": min_size,
        "max_chunk_size": rng.randint(50, 600),
        "respect_sentence_boundaries": rng.random() < 0.7,
    }


def test_config_fields_and_defaults_match() -> None:
    assert vars(ChunkingConfig()) == vars(SdkChunkingConfig())


# Hand-built edge: the only sentence break sits just inside the 100-char
# look-back window, which random corpora (dense with breaks) rarely isolate.
_EDGE_CONTENT = "A" * 20 + ". " + "b" * 200 + ". " + "c" * 40
_EDGE_CONFIG: dict[str, int | bool] = {
    "chunk_size": 120,
    "chunk_overlap": 0,
    "min_chunk_size": 10,
    "max_chunk_size": 200,
    "respect_sentence_boundaries": True,
}


def _case(seed: int) -> tuple[str, dict[str, int | bool]]:
    if seed < 0:
        return _EDGE_CONTENT, _EDGE_CONFIG
    return _corpus(seed), _config(seed)


@pytest.mark.parametrize("seed", [-1, *range(60)])
def test_strategies_match_sdk_core(seed: int) -> None:
    content, cfg = _case(seed)
    host = BaseChunker(ChunkingConfig(**cfg))
    sdk = TextChunker(SdkChunkingConfig(**cfg))
    assert host._chunk_text(content) == [tuple(s) for s in sdk.chunk_text(content)]
    assert host._chunk_markdown(content) == [tuple(s) for s in sdk.chunk_markdown(content)]
    assert host._chunk_code(content) == [tuple(s) for s in sdk.chunk_code(content)]
    assert host.estimate_chunks(content) == sdk.estimate_chunks(content)
