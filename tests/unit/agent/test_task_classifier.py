"""Behavioural tests for TaskClassifier (edge model + cache + stats)."""

from __future__ import annotations

from unittest.mock import patch as _patch

from victor.agent.task_classifier import TaskClassifier

_DECIDE = "victor.framework.agentic_loop.decide_sync"


def patch(target, *args, **kwargs):
    # The edge-model import target does not exist in victor.framework.agentic_loop
    # (the production path always falls back to heuristics); inject it so the
    # designed edge/cache path stays characterised.
    if target == _DECIDE:
        kwargs.setdefault("create", True)
    return _patch(target, *args, **kwargs)


_TYPES = ["debug", "edit", "search"]
_RESPONSE = (
    "TASK_TYPE: Debug\nCONFIDENCE: 0.85\nALTERNATIVES: edit:0.4, bogus:0.9, search: 0.1\n"
    "REASONING: stack trace present"
)


async def test_edge_classification_parses_labelled_fields() -> None:
    classifier = TaskClassifier()
    with patch(_DECIDE, return_value=_RESPONSE):
        result = await classifier.classify("why does this crash", valid_types=_TYPES)
    assert result.task_type == "debug"
    assert result.confidence == 0.85
    assert result.alternatives == {"edit": 0.4, "search": 0.1}
    assert result.reasoning == "stack trace present"


async def test_invalid_type_and_missing_fields_use_defaults() -> None:
    classifier = TaskClassifier(use_cache=False)
    with patch(_DECIDE, return_value="TASK_TYPE: dance\nALTERNATIVES: edit:x"):
        result = await classifier.classify("q", valid_types=_TYPES)
    assert result.task_type == "unknown"
    assert result.confidence == 0.8
    assert result.alternatives == {}
    assert result.reasoning == "No reasoning provided"


async def test_cache_hit_counts_and_statistics() -> None:
    classifier = TaskClassifier()
    with patch(_DECIDE, return_value=_RESPONSE) as decide:
        first = await classifier.classify("same", valid_types=_TYPES)
        second = await classifier.classify("same", valid_types=_TYPES)
    assert second is first
    assert decide.call_count == 1
    assert classifier.get_statistics() == {
        "total_classifications": 2,
        "cache_hits": 1,
        "cache_hit_rate": 0.5,
        "cache_size": 1,
    }
    classifier.reset_statistics()
    assert classifier.get_statistics()["total_classifications"] == 0
    assert classifier.get_statistics()["cache_size"] == 0


async def test_expired_cache_entry_is_evicted() -> None:
    classifier = TaskClassifier(cache_ttl=10)
    with patch(_DECIDE, return_value=_RESPONSE) as decide:
        with patch("time.time", return_value=1000.0):
            await classifier.classify("q", valid_types=_TYPES)
        with patch("time.time", return_value=1010.0):
            await classifier.classify("q", valid_types=_TYPES)
    assert decide.call_count == 2


async def test_edge_failure_and_disabled_fall_back_to_heuristic() -> None:
    classifier = TaskClassifier()
    with patch(_DECIDE, side_effect=RuntimeError("down")):
        failed = await classifier.classify("fix the bug")
    assert (failed.task_type, failed.reasoning) == ("debug", "Heuristic-based classification")
    assert classifier.get_statistics()["cache_size"] == 0

    disabled = TaskClassifier(enabled=False)
    with patch(_DECIDE) as decide:
        result = await disabled.classify("search for usages")
    decide.assert_not_called()
    assert result.task_type == "search"
    assert disabled.get_statistics()["total_classifications"] == 1


async def test_without_decision_hook_production_path_is_heuristic() -> None:
    # Pins today's behaviour: the edge hook import fails, so classifications
    # are heuristic and never cached.
    classifier = TaskClassifier()
    result = await classifier.classify("list files")
    assert result.reasoning == "Heuristic-based classification"
    assert classifier.get_statistics()["cache_size"] == 0
