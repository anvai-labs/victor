"""Behavioural tests for ComplexityEstimator (edge model + cache + stats)."""

from __future__ import annotations

from unittest.mock import patch as _patch

from victor.agent.complexity_estimator import ComplexityBand, ComplexityEstimator

_DECIDE = "victor.framework.agentic_loop.decide_sync"


def patch(target, *args, **kwargs):
    # The edge-model import target does not exist in victor.framework.agentic_loop
    # (the production path always falls back to heuristics); inject it so the
    # designed edge/cache path stays characterised.
    if target == _DECIDE:
        kwargs.setdefault("create", True)
    return _patch(target, *args, **kwargs)


_RESPONSE = "COMPLEXITY: 0.65\nBAND: complex\nCONFIDENCE: 0.9\nREASONING: many steps  "


async def test_edge_estimate_parses_labelled_fields() -> None:
    estimator = ComplexityEstimator()
    with patch(_DECIDE, return_value=_RESPONSE):
        estimate = await estimator.estimate("refactor the scheduler")
    assert estimate.score == 0.65
    # Latent bug pinned: the band label is upper-cased before ComplexityBand()
    # lookup, whose values are lower-case, so parsing always falls back.
    assert estimate.band is ComplexityBand.MODERATE
    assert estimate.confidence == 0.9
    assert estimate.reasoning == "many steps"


async def test_missing_or_malformed_fields_use_defaults() -> None:
    estimator = ComplexityEstimator(use_cache=False)
    with patch(_DECIDE, return_value="COMPLEXITY: 1.2.3\nBAND: huge\nCONFIDENCE: x"):
        estimate = await estimator.estimate("q")
    assert estimate.score == 0.5
    assert estimate.band is ComplexityBand.MODERATE
    assert estimate.confidence == 0.8
    assert estimate.reasoning == "No reasoning provided"


async def test_cache_hit_counts_and_statistics() -> None:
    estimator = ComplexityEstimator()
    with patch(_DECIDE, return_value=_RESPONSE) as decide:
        first = await estimator.estimate("same")
        second = await estimator.estimate("same")
    assert second is first
    assert decide.call_count == 1
    assert estimator.get_statistics() == {
        "total_estimates": 2,
        "cache_hits": 1,
        "cache_hit_rate": 0.5,
        "cache_size": 1,
    }
    estimator.reset_statistics()
    assert estimator.get_statistics() == {
        "total_estimates": 0,
        "cache_hits": 0,
        "cache_hit_rate": 0.0,
        "cache_size": 0,
    }


async def test_expired_cache_entry_is_evicted() -> None:
    estimator = ComplexityEstimator(cache_ttl=10)
    with patch(_DECIDE, return_value=_RESPONSE) as decide:
        with patch("time.time", return_value=1000.0):
            await estimator.estimate("q")
        with patch("time.time", return_value=1010.0):
            await estimator.estimate("q")
    assert decide.call_count == 2
    assert estimator.get_statistics()["cache_hits"] == 0


async def test_edge_failure_and_disabled_fall_back_to_heuristic() -> None:
    estimator = ComplexityEstimator()
    with patch(_DECIDE, side_effect=RuntimeError("down")):
        failed = await estimator.estimate("debug the error")
    assert failed.reasoning == "Heuristic-based estimation"
    assert failed.confidence == 0.7
    assert estimator.get_statistics()["cache_size"] == 0

    disabled = ComplexityEstimator(enabled=False)
    with patch(_DECIDE) as decide:
        estimate = await disabled.estimate("design the system architecture")
    decide.assert_not_called()
    assert estimate.score == 0.6
    assert disabled.get_statistics()["total_estimates"] == 1


async def test_without_decision_hook_production_path_is_heuristic() -> None:
    # Pins today's behaviour: the edge hook import fails, so estimates are
    # heuristic and never cached.
    estimator = ComplexityEstimator()
    estimate = await estimator.estimate("list files")
    assert estimate.reasoning == "Heuristic-based estimation"
    assert estimator.get_statistics()["cache_size"] == 0
