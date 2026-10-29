"""Tests for semantic event sampling filter."""

import time


from victor.observability.analytics.sampling_filter import (
    SamplingPolicy,
    SemanticSamplingFilter,
)

# ---------------------------------------------------------------------------
# SamplingPolicy defaults
# ---------------------------------------------------------------------------


class TestSamplingPolicy:
    def test_defaults(self):
        policy = SamplingPolicy()
        assert "tool_call" in policy.always_pass
        assert "error" in policy.always_pass
        assert "content" in policy.sampled_types
        assert "progress" in policy.deduped_types
        assert policy.sample_rate == 10
        assert policy.dedup_window_seconds == 5.0


# ---------------------------------------------------------------------------
# SemanticSamplingFilter
# ---------------------------------------------------------------------------


class TestSemanticSamplingFilter:
    def test_high_value_events_always_pass(self):
        f = SemanticSamplingFilter()
        for et in [
            "tool_call",
            "tool_result",
            "error",
            "recovery",
            "session_start",
            "session_end",
            "user_prompt",
            "assistant_response",
        ]:
            assert f.should_emit(et, {}) is True

    def test_content_events_sampled(self):
        policy = SamplingPolicy(sample_rate=5)
        f = SemanticSamplingFilter(policy)

        results = [f.should_emit("content", {}) for _ in range(20)]
        # Every 5th should pass
        passed = sum(results)
        assert passed == 4  # positions 5, 10, 15, 20

    def test_content_sample_rate_one_passes_all(self):
        policy = SamplingPolicy(sample_rate=1)
        f = SemanticSamplingFilter(policy)
        results = [f.should_emit("content", {}) for _ in range(10)]
        assert all(results)

    def test_progress_events_deduplicated(self):
        policy = SamplingPolicy(dedup_window_seconds=0.1)
        f = SemanticSamplingFilter(policy)

        # First progress event passes
        assert f.should_emit("progress", {}) is True
        # Immediate second is dropped
        assert f.should_emit("progress", {}) is False
        # After window passes, next one passes
        time.sleep(0.15)
        assert f.should_emit("progress", {}) is True

    def test_milestone_events_deduplicated(self):
        policy = SamplingPolicy(dedup_window_seconds=0.05)
        f = SemanticSamplingFilter(policy)

        assert f.should_emit("milestone", {}) is True
        assert f.should_emit("milestone", {}) is False

    def test_unknown_event_types_pass(self):
        f = SemanticSamplingFilter()
        assert f.should_emit("custom_event", {}) is True
        assert f.should_emit("something_new", {"key": "val"}) is True

    def test_stats_tracking(self):
        policy = SamplingPolicy(sample_rate=2)
        f = SemanticSamplingFilter(policy)

        f.should_emit("tool_call", {})  # pass
        f.should_emit("content", {})  # drop (1st, need 2nd)
        f.should_emit("content", {})  # pass (2nd)
        f.should_emit("content", {})  # drop (3rd)

        stats = f.get_stats()
        assert stats["events_passed"] == 2
        assert stats["events_dropped"] == 2
        assert stats["total_events"] == 4

    def test_reset_clears_state(self):
        f = SemanticSamplingFilter()
        f.should_emit("content", {})
        f.should_emit("tool_call", {})
        f.reset()
        stats = f.get_stats()
        assert stats["events_passed"] == 0
        assert stats["events_dropped"] == 0


# ---------------------------------------------------------------------------
# Integration with UsageLogger
# ---------------------------------------------------------------------------


class TestUsageLoggerWithSampling:
    def test_logger_accepts_sampling_filter(self, tmp_path):
        from victor.observability.analytics.logger import UsageLogger

        f = SemanticSamplingFilter()
        logger = UsageLogger(
            log_file=tmp_path / "test.jsonl",
            enabled=True,
            sampling_filter=f,
        )
        assert logger._sampling_filter is f

    def test_logger_filters_content_events(self, tmp_path):
        import json

        from victor.observability.analytics.logger import UsageLogger

        policy = SamplingPolicy(sample_rate=5)
        f = SemanticSamplingFilter(policy)
        log_file = tmp_path / "test.jsonl"
        logger = UsageLogger(log_file=log_file, enabled=True, sampling_filter=f)

        # Emit 20 content events
        for i in range(20):
            logger.log_event("content", {"chunk": i})

        # Also emit a high-value event
        logger.log_event("tool_call", {"tool": "read"})

        lines = log_file.read_text().strip().split("\n")
        event_types = [json.loads(line)["event_type"] for line in lines]

        # Should have 4 content events (1 in 5) + 1 tool_call
        assert event_types.count("content") == 4
        assert event_types.count("tool_call") == 1

    def test_logger_works_without_filter(self, tmp_path):
        from victor.observability.analytics.logger import UsageLogger

        logger = UsageLogger(log_file=tmp_path / "test.jsonl", enabled=True)

        # Should work normally without filter
        logger.log_event("content", {"chunk": 0})
        logger.log_event("tool_call", {"tool": "read"})

        lines = (tmp_path / "test.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
