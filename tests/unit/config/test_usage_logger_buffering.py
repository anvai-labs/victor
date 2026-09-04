import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from victor.observability.analytics.enhanced_logger import EnhancedUsageLogger


@pytest.fixture
def log_file(tmp_path: Path) -> Path:
    return tmp_path / "usage_log.jsonl"


def _events(log_file: Path):
    if not log_file.exists():
        return []
    return [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]


class TestBufferedWrites:
    """Co-design review item 14: buffered mode defers writes off the hot path."""

    def test_buffered_writes_nothing_below_threshold(self, log_file: Path):
        logger = EnhancedUsageLogger(
            log_file=log_file, enabled=True, buffered=True, batch_size=100, flush_interval=999.0
        )

        for i in range(5):
            logger.log_event("tool_call", {"i": i})

        assert not log_file.exists() or log_file.read_text() == ""

    def test_buffered_flushes_at_batch_size(self, log_file: Path):
        logger = EnhancedUsageLogger(
            log_file=log_file, enabled=True, buffered=True, batch_size=3, flush_interval=999.0
        )

        logger.log_event("tool_call", {"i": 1})
        logger.log_event("tool_call", {"i": 2})
        assert not log_file.exists() or log_file.read_text() == ""

        logger.log_event("tool_call", {"i": 3})
        assert len(_events(log_file)) == 3

    def test_buffered_flushes_at_time_interval(self, log_file: Path):
        logger = EnhancedUsageLogger(
            log_file=log_file, enabled=True, buffered=True, batch_size=100, flush_interval=0.05
        )

        logger.log_event("tool_call", {"i": 1})
        assert not log_file.exists() or log_file.read_text() == ""

        time.sleep(0.1)
        logger.log_event("tool_call", {"i": 2})
        assert len(_events(log_file)) == 2

    def test_flush_writes_pending_events(self, log_file: Path):
        logger = EnhancedUsageLogger(
            log_file=log_file, enabled=True, buffered=True, batch_size=100, flush_interval=999.0
        )
        logger.log_event("tool_call", {"i": 1})
        logger.log_event("tool_call", {"i": 2})

        logger.flush()

        events = _events(log_file)
        assert [e["data"]["i"] for e in events] == [1, 2]

    def test_flush_with_nothing_buffered_is_a_noop(self, log_file: Path):
        logger = EnhancedUsageLogger(
            log_file=log_file, enabled=True, buffered=True, batch_size=100, flush_interval=999.0
        )
        logger.flush()
        assert not log_file.exists() or log_file.read_text() == ""

    def test_close_flushes_pending_events(self, log_file: Path):
        logger = EnhancedUsageLogger(
            log_file=log_file, enabled=True, buffered=True, batch_size=100, flush_interval=999.0
        )
        logger.log_event("tool_call", {"i": 1})

        logger.close()

        assert len(_events(log_file)) == 1

    def test_close_on_unbuffered_logger_is_a_noop(self, log_file: Path):
        """close() must be safe to call unconditionally during shutdown,
        even for a logger that was never put into buffered mode."""
        logger = EnhancedUsageLogger(log_file=log_file, enabled=True, buffered=False)
        logger.log_event("tool_call", {"i": 1})
        assert len(_events(log_file)) == 1

        logger.close()  # must not raise, must not duplicate the line

        assert len(_events(log_file)) == 1

    def test_bounded_buffer_drops_oldest_and_counts(self, log_file: Path):
        logger = EnhancedUsageLogger(
            log_file=log_file,
            enabled=True,
            buffered=True,
            batch_size=1_000_000,  # never auto-flush by size in this test
            flush_interval=999.0,
            max_buffer_size=3,
        )

        for i in range(5):
            logger.log_event("tool_call", {"i": i})

        assert logger.dropped_event_count == 2
        logger.flush()
        events = _events(log_file)
        # Oldest two (i=0, i=1) were dropped; newest three survive in order.
        assert [e["data"]["i"] for e in events] == [2, 3, 4]

    def test_buffered_event_captures_timestamp_at_log_time_not_flush_time(self, log_file: Path):
        """Event-time facts (timestamp, run_kind) must reflect when
        log_event() was called, not when the deferred flush happens."""
        logger = EnhancedUsageLogger(
            log_file=log_file, enabled=True, buffered=True, batch_size=100, flush_interval=999.0
        )

        logger.log_event("tool_call", {"i": 1})
        time.sleep(0.05)
        logger.flush()

        event = _events(log_file)[0]
        # Recorded timestamp must predate the flush by roughly the sleep,
        # not be stamped "now" at flush time.
        logged_at = time.time()
        from datetime import datetime

        event_time = datetime.fromisoformat(event["timestamp"]).timestamp()
        assert logged_at - event_time >= 0.04

    def test_one_bad_record_does_not_lose_the_rest_of_the_batch(self, log_file: Path):
        """A serialization failure for one buffered event must not drop
        its batch-mates — matches the sync path's per-event isolation."""
        logger = EnhancedUsageLogger(
            log_file=log_file, enabled=True, buffered=True, batch_size=100, flush_interval=999.0
        )
        logger.log_event("tool_call", {"i": 1})
        logger.log_event("tool_call", {"i": 2})
        logger.log_event("tool_call", {"i": 3})

        call_count = {"n": 0}
        original = logger._sanitize_log_data

        def _flaky(data):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("boom")
            return original(data)

        with patch.object(logger, "_sanitize_log_data", side_effect=_flaky):
            logger.flush()

        events = _events(log_file)
        assert [e["data"]["i"] for e in events] == [1, 3]

    def test_disabled_logger_never_buffers(self, log_file: Path):
        logger = EnhancedUsageLogger(log_file=log_file, enabled=False, buffered=True)
        logger.log_event("tool_call", {"i": 1})
        assert logger._buffer == []
