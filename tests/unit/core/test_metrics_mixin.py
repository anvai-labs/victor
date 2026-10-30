"""Tests for victor.core.metrics_mixin.AdhocMetricsMixin."""

from typing import Any, Dict

import pytest

from victor.core.metrics_mixin import AdhocMetricsMixin
from victor.core.protocols import MetricsCollectorProtocol


class _Collector(AdhocMetricsMixin):
    """Minimal collector that never initialises the ad-hoc store explicitly."""

    def get_snapshot(self) -> Dict[str, Any]:
        return dict(self._adhoc_metrics)


class _EarlyUseCollector(AdhocMetricsMixin):
    """Records a metric from ``__init__`` before any other state exists."""

    def __init__(self) -> None:
        self.record_metric("boot", 1)
        self.ready = True


def test_storage_exists_without_explicit_init() -> None:
    collector = _Collector()
    assert collector._adhoc_metrics == {}
    # Same dict on repeated access (created once, then reused).
    assert collector._adhoc_metrics is collector._adhoc_metrics


def test_storage_usable_during_init() -> None:
    collector = _EarlyUseCollector()
    assert collector._adhoc_metrics == {"boot": 1.0}


def test_untagged_key_is_bare_name() -> None:
    collector = _Collector()
    collector.record_metric("latency_ms", 12.5)
    assert collector.get_snapshot() == {"latency_ms": 12.5}


def test_tagged_key_sorts_tags_by_name() -> None:
    collector = _Collector()
    collector.record_metric("requests", 3, zone="eu", method="GET", code="200")
    assert collector.get_snapshot() == {"requests{code=200,method=GET,zone=eu}": 3.0}


def test_tagged_and_untagged_keys_are_distinct() -> None:
    collector = _Collector()
    collector.record_metric("hits", 1)
    collector.record_metric("hits", 2, tier="a")
    assert collector.get_snapshot() == {"hits": 1.0, "hits{tier=a}": 2.0}


def test_value_is_coerced_to_float() -> None:
    collector = _Collector()
    collector.record_metric("count", 7)
    value = collector.get_snapshot()["count"]
    assert value == 7.0
    assert type(value) is float


def test_non_numeric_value_raises() -> None:
    collector = _Collector()
    with pytest.raises((TypeError, ValueError)):
        collector.record_metric("bad", "not-a-number")  # type: ignore[arg-type]
    assert collector.get_snapshot() == {}


def test_later_observation_overwrites_same_key() -> None:
    collector = _Collector()
    collector.record_metric("gauge", 1, k="v")
    collector.record_metric("gauge", 5, k="v")
    assert collector.get_snapshot() == {"gauge{k=v}": 5.0}


def test_instances_do_not_share_storage() -> None:
    first = _Collector()
    second = _Collector()
    first.record_metric("only_first", 1)
    assert second.get_snapshot() == {}
    assert first._adhoc_metrics is not second._adhoc_metrics


def test_subclass_with_snapshot_satisfies_protocol() -> None:
    assert isinstance(_Collector(), MetricsCollectorProtocol)
