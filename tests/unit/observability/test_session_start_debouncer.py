# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for session start event debouncing."""

from __future__ import annotations

import time

import pytest

from victor.observability.debouncing import (
    DebounceConfig,
    SessionStartDebouncer,
    WindowType,
)


class TestSessionStartDebouncer:
    """Test suite for SessionStartDebouncer."""

    def test_time_based_deduplication(self):
        """Test that events within time window are deduplicated."""
        config = DebounceConfig(
            window_type=WindowType.TIME_BASED,
            window_seconds=5,
            max_events_per_window=1,  # Only 1 event allowed for strict deduplication
        )
        debouncer = SessionStartDebouncer(config)

        metadata = {"session_id": "test-123", "provider": "anthropic"}

        # First event should emit
        assert debouncer.should_emit("test-123", metadata) is True
        debouncer.record("test-123", metadata)

        # Immediate duplicate should be blocked (at limit with duplicate metadata)
        assert debouncer.should_emit("test-123", metadata) is False

        # Another attempt should also be blocked (within window)
        assert debouncer.should_emit("test-123", metadata) is False

    def test_burst_limiting(self):
        """Test that burst limit is enforced."""
        config = DebounceConfig(
            window_type=WindowType.COUNT_BASED,
            max_events_per_window=2,
        )
        debouncer = SessionStartDebouncer(config)

        metadata = {"session_id": "test-456"}

        # Allow first 2 events
        assert debouncer.should_emit("test-456", metadata) is True
        debouncer.record("test-456", metadata)

        assert debouncer.should_emit("test-456", metadata) is True
        debouncer.record("test-456", metadata)

        # Third event should be blocked
        assert debouncer.should_emit("test-456", metadata) is False

    def test_metadata_fingerprinting(self):
        """Test that different metadata creates different event keys."""
        config = DebounceConfig(
            window_type=WindowType.TIME_BASED,
            window_seconds=5,
            max_events_per_window=2,  # Allow 2 events to test different metadata
            enable_metadata_fingerprinting=True,
        )
        debouncer = SessionStartDebouncer(config)

        metadata1 = {"session_id": "test-789", "provider": "anthropic"}
        metadata2 = {"session_id": "test-789", "provider": "openai"}

        # Different providers should not deduplicate
        assert debouncer.should_emit("test-789", metadata1) is True
        debouncer.record("test-789", metadata1)

        assert debouncer.should_emit("test-789", metadata2) is True
        debouncer.record("test-789", metadata2)

        # At limit now, and same provider should deduplicate
        assert debouncer.should_emit("test-789", metadata1) is False

    def test_metadata_deduplication_disabled(self):
        """Test behavior when metadata fingerprinting is disabled."""
        config = DebounceConfig(
            window_type=WindowType.TIME_BASED,
            window_seconds=5,
            enable_metadata_fingerprinting=False,
        )
        debouncer = SessionStartDebouncer(config)

        metadata1 = {"session_id": "test-abc", "provider": "anthropic"}
        metadata2 = {"session_id": "test-abc", "provider": "openai"}

        # Without fingerprinting, should deduplicate by session_id only
        assert debouncer.should_emit("test-abc", metadata1) is True
        debouncer.record("test-abc", metadata1)

        assert debouncer.should_emit("test-abc", metadata2) is False

    def test_get_stats(self):
        """Test statistics tracking."""
        config = DebounceConfig(
            window_type=WindowType.TIME_BASED,
            window_seconds=5,
            max_events_per_window=1,  # Only 1 event allowed
        )
        debouncer = SessionStartDebouncer(config)

        metadata = {"session_id": "test-stats"}

        # First event
        assert debouncer.should_emit("test-stats", metadata) is True
        debouncer.record("test-stats", metadata)

        # Try to emit again (should be debounced at limit with duplicate metadata)
        assert debouncer.should_emit("test-stats", metadata) is False

        stats = debouncer.get_stats()
        assert stats["total_checks"] == 2
        assert stats["emitted"] == 1
        assert stats["debounced"] == 1
        assert stats["active_sessions"] == 1

    def test_reset(self):
        """Test that reset clears all state."""
        config = DebounceConfig(
            window_type=WindowType.TIME_BASED,
            window_seconds=5,
        )
        debouncer = SessionStartDebouncer(config)

        metadata = {"session_id": "test-reset"}

        # Add some state
        assert debouncer.should_emit("test-reset", metadata) is True
        debouncer.record("test-reset", metadata)

        assert debouncer.get_stats()["active_sessions"] == 1

        # Reset
        debouncer.reset()

        # State should be cleared
        assert debouncer.get_stats()["active_sessions"] == 0
        assert debouncer.get_stats()["total_checks"] == 0

        # Should be able to emit again
        assert debouncer.should_emit("test-reset", metadata) is True

    def test_window_expiry(self):
        """Test that events outside time window are not deduplicated."""
        config = DebounceConfig(
            window_type=WindowType.TIME_BASED,
            window_seconds=1,  # Short window for testing
            max_events_per_window=3,
        )
        debouncer = SessionStartDebouncer(config)

        metadata = {"session_id": "test-window"}

        # First event
        assert debouncer.should_emit("test-window", metadata) is True
        debouncer.record("test-window", metadata)

        # Wait for window to expire
        time.sleep(1.5)

        # Should be able to emit again after window expires
        assert debouncer.should_emit("test-window", metadata) is True

    def test_different_sessions_independent(self):
        """Test that different sessions are tracked independently."""
        config = DebounceConfig(
            window_type=WindowType.TIME_BASED,
            window_seconds=5,
            max_events_per_window=2,
        )
        debouncer = SessionStartDebouncer(config)

        metadata1 = {"session_id": "session-1"}
        metadata2 = {"session_id": "session-2"}

        # Both sessions should be able to emit
        assert debouncer.should_emit("session-1", metadata1) is True
        debouncer.record("session-1", metadata1)

        assert debouncer.should_emit("session-2", metadata2) is True
        debouncer.record("session-2", metadata2)

        # Both should have their own limits
        assert debouncer.should_emit("session-1", metadata1) is True
        debouncer.record("session-1", metadata1)

        # session-1 is now at limit, but session-2 should still work
        assert debouncer.should_emit("session-1", metadata1) is False
        assert debouncer.should_emit("session-2", metadata2) is True

    def test_count_based_mode(self):
        """Test count-based debouncing mode."""
        config = DebounceConfig(
            window_type=WindowType.COUNT_BASED,
            max_events_per_window=2,
        )
        debouncer = SessionStartDebouncer(config)

        metadata = {"session_id": "test-count"}

        # Should allow first 2 events
        assert debouncer.should_emit("test-count", metadata) is True
        debouncer.record("test-count", metadata)

        assert debouncer.should_emit("test-count", metadata) is True
        debouncer.record("test-count", metadata)

        # Third should be blocked
        assert debouncer.should_emit("test-count", metadata) is False

    def test_default_config(self):
        """Test that default config works correctly."""
        # Create with default config
        debouncer = SessionStartDebouncer()

        metadata = {"session_id": "test-default"}

        # Should work with default settings
        assert debouncer.should_emit("test-default", metadata) is True
        debouncer.record("test-default", metadata)

    def test_event_key_computation(self):
        """Test that event keys are computed correctly."""
        config = DebounceConfig(
            enable_metadata_fingerprinting=True,
        )
        debouncer = SessionStartDebouncer(config)

        # Same metadata should produce same key
        metadata1 = {"session_id": "test-key", "provider": "anthropic"}
        metadata2 = {"session_id": "test-key", "provider": "anthropic"}

        key1 = debouncer._compute_event_key("test-key", metadata1)
        key2 = debouncer._compute_event_key("test-key", metadata2)

        assert key1 == key2

        # Different metadata should produce different key
        metadata3 = {"session_id": "test-key", "provider": "openai"}
        key3 = debouncer._compute_event_key("test-key", metadata3)

        assert key1 != key3

    def test_metadata_hash_computation(self):
        """Test that metadata hashes are consistent."""
        debouncer = SessionStartDebouncer()

        metadata = {
            "provider": "anthropic",
            "model": "claude-3-5-sonnet",
            "mode": "chat",
        }

        hash1 = debouncer._compute_metadata_hash(metadata)
        hash2 = debouncer._compute_metadata_hash(metadata)

        # Hash should be consistent
        assert hash1 == hash2

        # Different metadata should produce different hash
        different_metadata = {"provider": "openai", "model": "gpt-4", "mode": "chat"}
        hash3 = debouncer._compute_metadata_hash(different_metadata)

        assert hash1 != hash3

    def test_thread_safety(self):
        """Test basic thread safety (smoke test)."""
        import threading

        config = DebounceConfig(
            window_type=WindowType.TIME_BASED,
            window_seconds=5,
            max_events_per_window=10,
        )
        debouncer = SessionStartDebouncer(config)

        results = []
        errors = []

        def emit_event(session_id: str):
            try:
                metadata = {"session_id": session_id}
                if debouncer.should_emit(session_id, metadata):
                    debouncer.record(session_id, metadata)
                    results.append(session_id)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=emit_event, args=(f"session-{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should not have any errors
        assert len(errors) == 0
        # All 5 sessions should have emitted
        assert len(results) == 5

    def test_empty_metadata(self):
        """Test behavior with empty metadata."""
        config = DebounceConfig(
            window_type=WindowType.TIME_BASED,
            window_seconds=5,
            max_events_per_window=1,  # Only 1 event allowed
        )
        debouncer = SessionStartDebouncer(config)

        # Empty metadata should work
        assert debouncer.should_emit("test-empty", {}) is True
        debouncer.record("test-empty", {})

        # And should deduplicate (at limit with duplicate metadata)
        assert debouncer.should_emit("test-empty", {}) is False
