"""Unit tests for the pure PromptQueue (FIFO + listener)."""

from __future__ import annotations

from typing import List

from victor.ui.tui.prompt_queue import PromptQueue


def test_fifo_order_peek_and_consume() -> None:
    queue = PromptQueue()
    assert not queue
    assert queue.peek() is None
    assert queue.consume() is None

    queue.enqueue("first")
    queue.enqueue("second")
    queue.enqueue("third")
    assert len(queue) == 3
    assert queue.peek() == "first"  # peek does not remove
    assert len(queue) == 3

    assert queue.consume() == "first"
    assert queue.consume() == "second"
    assert queue.consume() == "third"
    assert not queue


def test_enqueue_ignores_whitespace_only_input() -> None:
    queue = PromptQueue()
    assert queue.enqueue("") is False
    assert queue.enqueue("   ") is False
    assert queue.enqueue("\t\n") is False
    assert not queue
    assert queue.enqueue("  real  ") is True
    assert queue.snapshot() == ("  real  ",)  # stored verbatim


def test_clear_returns_dropped_count_and_empties() -> None:
    queue = PromptQueue()
    assert queue.clear() == 0  # empty clear is a no-op
    queue.enqueue("a")
    queue.enqueue("b")
    assert queue.clear() == 2
    assert not queue
    assert queue.clear() == 0


def test_snapshot_is_oldest_first_and_defensive() -> None:
    queue = PromptQueue()
    queue.enqueue("one")
    queue.enqueue("two")
    snap = queue.snapshot()
    assert snap == ("one", "two")
    queue.enqueue("three")
    assert snap == ("one", "two")  # snapshot unaffected by later mutation
    assert queue.snapshot() == ("one", "two", "three")


def test_on_change_fires_on_enqueue_consume_and_nonempty_clear() -> None:
    events: List[int] = []
    queue = PromptQueue(on_change=lambda q: events.append(len(q)))

    queue.enqueue("a")
    queue.enqueue("b")
    assert events == [1, 2]

    queue.consume()
    assert events == [1, 2, 1]

    queue.clear()
    assert events == [1, 2, 1, 0]

    # No-ops must not fire: empty-input enqueue, consume on empty, empty clear.
    queue.enqueue("  ")
    queue.consume()
    queue.clear()
    assert events == [1, 2, 1, 0]


def test_on_change_is_optional() -> None:
    queue = PromptQueue()
    queue.enqueue("a")  # must not raise without a listener
    assert queue.consume() == "a"
