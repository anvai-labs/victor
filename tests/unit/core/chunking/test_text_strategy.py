"""Text chunking strategy: overlap guard and loop termination.

``ChunkingConfig`` now rejects ``chunk_overlap >= chunk_size`` (the SDK's
``victor_contracts.chunking`` guard, ported into the core copy), and the text
strategy stops once its window stops advancing instead of re-reading the final
window forever. Together those close the infinite-loop class where the loop
step ``chunk_end - chunk_overlap`` lands on the same position on every
iteration.
"""

from __future__ import annotations

import pytest

from victor.core.chunking.base import ChunkingConfig
from victor.core.chunking.registry import ChunkingRegistry
from victor.core.chunking.strategies.text import TextChunkingStrategy


def _prose(sentences: int = 120) -> str:
    """Deterministic sentence-separated corpus (~5k characters)."""
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    out = []
    for i in range(sentences):
        n = 4 + (i * 3) % 7
        body = " ".join(words[(i + k) % len(words)] for k in range(n))
        out.append(body.capitalize() + (".", "!", "?")[i % 3])
    return " ".join(out)


class TestOverlapGuard:
    """The ported ``0 <= chunk_overlap < chunk_size`` validation."""

    @pytest.mark.parametrize(
        ("chunk_size", "chunk_overlap"),
        [
            (200, 300),  # the reported hang: overlap beyond the window
            (200, 200),  # overlap == chunk_size: the window never moves
            (1024, 2048),
            (200, -1),  # negative overlap was already rejected
        ],
    )
    def test_overlap_outside_bounds_is_rejected(self, chunk_size: int, chunk_overlap: int) -> None:
        with pytest.raises(ValueError, match=r"0 <= chunk_overlap < chunk_size"):
            ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def test_rejection_message_names_both_values(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            ChunkingConfig(chunk_size=200, chunk_overlap=300)

        assert "chunk_overlap=300" in str(excinfo.value)
        assert "chunk_size=200" in str(excinfo.value)

    @pytest.mark.parametrize("chunk_overlap", [0, 199])
    def test_boundary_overlaps_are_accepted(self, chunk_overlap: int) -> None:
        config = ChunkingConfig(chunk_size=200, chunk_overlap=chunk_overlap)

        assert config.chunk_overlap == chunk_overlap

    def test_existing_validation_is_unchanged(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ChunkingConfig(chunk_size=0)
        with pytest.raises(ValueError, match="min_chunk_size must be non-negative"):
            ChunkingConfig(min_chunk_size=-1)
        with pytest.raises(ValueError, match="max_chunk_size must be >= chunk_size"):
            ChunkingConfig(max_chunk_size=512)


class TestLoopTermination:
    """Every accepted config must return, not hang."""

    @pytest.mark.timeout(10)
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},  # pure defaults (1024/128/100) hung on any content
            {"chunk_size": 1024, "chunk_overlap": 100},
            {"chunk_size": 200, "chunk_overlap": 199, "min_chunk_size": 0},
            {"chunk_size": 200, "chunk_overlap": 199, "min_chunk_size": 100},
        ],
        ids=["defaults", "docstring-config", "overlap-max", "overlap-max-min100"],
    )
    def test_chunk_returns_for_previously_hanging_configs(self, kwargs: dict) -> None:
        chunks = TextChunkingStrategy(ChunkingConfig(**kwargs)).chunk(_prose())

        assert isinstance(chunks, list)

    @pytest.mark.timeout(10)
    def test_default_strategy_chunks_short_content(self) -> None:
        """The default config hung even on a single-line string."""
        chunks = TextChunkingStrategy().chunk("Short sentence. Another one here.")

        assert chunks == []  # below min_chunk_size, but it returns

    @pytest.mark.timeout(10)
    def test_registry_default_path_terminates_for_plain_text(self) -> None:
        """``chunk_document`` with no config used to spin forever on text."""
        chunks = ChunkingRegistry().chunk(_prose(40), doc_type="text")

        assert len(chunks) > 0

    @pytest.mark.timeout(10)
    def test_markdown_fallback_terminates_with_default_config(self) -> None:
        """Header-less markdown delegates to the text strategy."""
        registry = ChunkingRegistry()

        chunks = registry.chunk(_prose(40), doc_type="markdown")

        assert len(chunks) > 0

    @pytest.mark.timeout(10)
    @pytest.mark.parametrize("chunk_overlap", [0, 199])
    def test_boundary_overlaps_chunk_correctly(self, chunk_overlap: int) -> None:
        config = ChunkingConfig(chunk_size=200, chunk_overlap=chunk_overlap, min_chunk_size=0)
        chunks = TextChunkingStrategy(config).chunk(_prose())

        assert len(chunks) > 1
        # Windows advance in document order and stay inside the content.
        starts = [chunk.start_char for chunk in chunks]
        assert starts == sorted(starts)
        assert starts[0] == 0
        assert all(0 <= chunk.start_char < chunk.end_char <= len(_prose()) for chunk in chunks)


class TestWorkingConfigsAreUnchanged:
    """Output for configs that always terminated is byte-for-byte pinned.

    Golden spans were produced by the pre-fix implementation; the termination
    fix must not alter them.
    """

    @pytest.mark.timeout(10)
    def test_golden_spans_for_working_config(self) -> None:
        corpus = _prose()
        strategy = TextChunkingStrategy(ChunkingConfig(chunk_size=1024, chunk_overlap=100))

        chunks = strategy.chunk(corpus)

        assert [(c.start_char, c.end_char) for c in chunks] == [
            (0, 1108),
            (1008, 2087),
            (1987, 3099),
            (2999, 4108),
            (4008, 4932),
        ]

    @pytest.mark.timeout(10)
    def test_golden_span_count_for_no_overlap(self) -> None:
        strategy = TextChunkingStrategy(
            ChunkingConfig(chunk_size=200, chunk_overlap=0, min_chunk_size=0)
        )

        chunks = strategy.chunk(_prose())

        assert len(chunks) == 18
        # No overlap: windows are contiguous.
        assert all(chunks[i].end_char == chunks[i + 1].start_char for i in range(len(chunks) - 1))
