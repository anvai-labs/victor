"""Tests for the shared edge-decider core."""

from __future__ import annotations

import pytest

from victor.agent.edge_decision_cache import (
    CachedEdgeDecider,
    extract_labeled_float,
    extract_labeled_text,
)


@pytest.mark.parametrize(
    "response,expected",
    [
        ("CONFIDENCE: 0.42", 0.42),
        ("confidence:0.9 trailing", 0.9),
        ("CONFIDENCE: 1.2.3", 0.8),
        ("nothing here", 0.8),
    ],
)
def test_extract_labeled_float(response: str, expected: float) -> None:
    assert extract_labeled_float(response, "CONFIDENCE", 0.8) == expected


def test_extract_labeled_text_strips_and_defaults() -> None:
    assert extract_labeled_text("REASONING:   short answer  \nNEXT: x", "REASONING", "d") == (
        "short answer"
    )
    assert extract_labeled_text("no label", "REASONING", "d") == "d"


class _Decider(CachedEdgeDecider[str]):
    _STATS_TOTAL_KEY = "total_things"
    _LOG_NAME = "Decider"


async def test_decide_caches_edge_results_under_custom_stats_key() -> None:
    decider = _Decider()
    calls = []

    async def edge(start: float) -> str:
        calls.append(start)
        return "edge"

    assert await decider._decide("q", edge, lambda start: "heuristic") == "edge"
    assert await decider._decide("q", edge, lambda start: "heuristic") == "edge"
    assert len(calls) == 1
    assert decider.get_statistics() == {
        "total_things": 2,
        "cache_hits": 1,
        "cache_hit_rate": 0.5,
        "cache_size": 1,
    }


async def test_decide_falls_back_without_caching_on_edge_failure() -> None:
    decider = _Decider()

    async def edge(start: float) -> str:
        raise RuntimeError("down")

    assert await decider._decide("q", edge, lambda start: "heuristic") == "heuristic"
    assert decider.get_statistics()["cache_size"] == 0


async def test_disabled_or_uncached_decider_skips_cache() -> None:
    async def edge(start: float) -> str:
        return "edge"

    disabled = _Decider(enabled=False)
    assert await disabled._decide("q", edge, lambda start: "heuristic") == "heuristic"

    uncached = _Decider(use_cache=False)
    await uncached._decide("q", edge, lambda start: "heuristic")
    assert uncached.get_statistics()["cache_size"] == 0
    uncached.reset_statistics()
    assert uncached.get_statistics()["total_things"] == 0
