"""Tests for the periodic metrics reporter base."""

from __future__ import annotations

import threading

import pytest

from victor.core.events.periodic import PeriodicMetricsReporter


class _Reporter(PeriodicMetricsReporter):
    _THREAD_NAME = "test-periodic-reporter"

    def __init__(self, fail_first: bool = False) -> None:
        super().__init__(interval_seconds=0.01, topic="t", source="s", reset_after_emit=False)
        self.calls = 0
        self.fail_first = fail_first
        self.second_call = threading.Event()

    def _emit(self) -> None:
        self.calls += 1
        if self.calls >= 2:
            self.second_call.set()
        if self.fail_first and self.calls == 1:
            raise RuntimeError("boom")


def test_rejects_non_positive_interval() -> None:
    with pytest.raises(ValueError):
        PeriodicMetricsReporter(interval_seconds=0, topic="t", source="s", reset_after_emit=False)


def test_start_is_idempotent_and_stop_joins() -> None:
    reporter = _Reporter()
    reporter.start()
    first_thread = reporter._thread
    reporter.start()
    assert reporter._thread is first_thread
    assert reporter.is_running
    assert first_thread is not None and first_thread.name == "test-periodic-reporter"
    assert first_thread.daemon
    reporter.stop()
    assert not reporter.is_running
    reporter.stop()  # stopping twice is a no-op


def test_emit_errors_do_not_kill_the_loop() -> None:
    reporter = _Reporter(fail_first=True)
    reporter.start()
    try:
        assert reporter.second_call.wait(timeout=2.0)
    finally:
        reporter.stop()


def test_base_emit_is_abstract() -> None:
    reporter = PeriodicMetricsReporter(
        interval_seconds=1, topic="t", source="s", reset_after_emit=False
    )
    with pytest.raises(NotImplementedError):
        reporter._emit()
