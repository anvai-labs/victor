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

"""Tests for StreamingController."""

import pytest
import time

from victor.agent.streaming_controller import (
    StreamingController,
    StreamingControllerConfig,
    StreamingSession,
)


class TestStreamingSession:
    """Tests for StreamingSession dataclass."""

    def test_is_active_new_session(self):
        """Test new session is active."""
        session = StreamingSession(
            session_id="test",
            model="gpt-4",
            provider="openai",
            start_time=time.time(),
        )
        assert session.is_active is True

    def test_is_active_completed_session(self):
        """Test completed session is not active."""
        session = StreamingSession(
            session_id="test",
            model="gpt-4",
            provider="openai",
            start_time=time.time(),
            end_time=time.time(),
        )
        assert session.is_active is False

    def test_is_active_cancelled_session(self):
        """Test cancelled session is not active."""
        session = StreamingSession(
            session_id="test",
            model="gpt-4",
            provider="openai",
            start_time=time.time(),
            cancelled=True,
        )
        assert session.is_active is False

    def test_duration(self):
        """Test duration calculation."""
        start = time.time()
        session = StreamingSession(
            session_id="test",
            model="gpt-4",
            provider="openai",
            start_time=start,
            end_time=start + 5.0,
        )
        assert session.duration == pytest.approx(5.0, abs=0.01)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        session = StreamingSession(
            session_id="test-123",
            model="gpt-4",
            provider="openai",
            start_time=1000.0,
            end_time=1005.0,
        )
        data = session.to_dict()

        assert data["session_id"] == "test-123"
        assert data["model"] == "gpt-4"
        assert data["provider"] == "openai"
        assert data["duration"] == pytest.approx(5.0, abs=0.01)


class TestStreamingControllerConfig:
    """Tests for StreamingControllerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StreamingControllerConfig()
        assert config.max_history == 100
        assert config.enable_metrics_collection is True
        assert config.default_timeout == 300.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = StreamingControllerConfig(max_history=50, default_timeout=60.0)
        assert config.max_history == 50
        assert config.default_timeout == 60.0


class TestStreamingController:
    """Tests for StreamingController class."""

    @pytest.fixture
    def controller(self):
        """Create a streaming controller for testing."""
        return StreamingController()

    def test_init(self, controller):
        """Test controller initialization."""
        assert controller.is_streaming is False
        assert controller.current_session is None

    def test_start_session(self, controller):
        """Test starting a streaming session."""
        session = controller.start_session("gpt-4", "openai")

        assert session is not None
        assert session.model == "gpt-4"
        assert session.provider == "openai"
        assert session.is_active is True
        assert controller.is_streaming is True

    def test_start_session_ends_previous(self, controller):
        """Test that starting a new session ends the previous one."""
        session1 = controller.start_session("gpt-4", "openai")
        session2 = controller.start_session("claude-3", "anthropic")

        assert controller.current_session == session2
        assert session1.is_active is False
        assert session2.is_active is True

    def test_record_first_token(self, controller):
        """Test recording first token time."""
        controller.start_session("gpt-4", "openai")
        controller.record_first_token()

        assert controller.current_metrics is not None
        assert controller.current_metrics.first_token_time is not None

    def test_record_first_token_only_once(self, controller):
        """Test that first token time is only recorded once."""
        controller.start_session("gpt-4", "openai")
        controller.record_first_token()
        first_time = controller.current_metrics.first_token_time

        time.sleep(0.01)
        controller.record_first_token()

        assert controller.current_metrics.first_token_time == first_time

    def test_record_chunk(self, controller):
        """Test recording chunks."""
        controller.start_session("gpt-4", "openai")
        controller.record_chunk(content_length=100)
        controller.record_chunk(content_length=50)

        assert controller.current_metrics.total_chunks == 2
        assert controller.current_metrics.total_content_length == 150

    def test_record_tool_call(self, controller):
        """Test recording tool calls."""
        controller.start_session("gpt-4", "openai")
        controller.record_tool_call()
        controller.record_tool_call()

        assert controller.current_metrics.tool_calls_count == 2

    def test_complete_session(self, controller):
        """Test completing a session."""
        controller.start_session("gpt-4", "openai")
        controller.record_chunk(content_length=100)

        session = controller.complete_session()

        assert session is not None
        assert session.is_active is False
        assert session.end_time is not None
        assert session.metrics is not None
        assert controller.is_streaming is False

    def test_complete_session_no_active(self, controller):
        """Test completing when no active session."""
        result = controller.complete_session()
        assert result is None

    def test_cancel_session(self, controller):
        """Test cancelling a session."""
        controller.start_session("gpt-4", "openai")

        result = controller.cancel_session()

        assert result is True
        assert controller.is_cancellation_requested() is True

    def test_cancel_session_no_active(self, controller):
        """Test cancelling when no active session."""
        result = controller.cancel_session()
        assert result is False

    def test_session_history(self, controller):
        """Test session history tracking."""
        controller.start_session("gpt-4", "openai")
        controller.complete_session()

        controller.start_session("claude-3", "anthropic")
        controller.complete_session()

        history = controller.get_session_history(limit=10)

        assert len(history) == 2
        # Most recent first
        assert history[0]["provider"] == "anthropic"
        assert history[1]["provider"] == "openai"

    def test_session_history_limit(self, controller):
        """Test session history respects limit."""
        for i in range(5):
            controller.start_session(f"model-{i}", "provider")
            controller.complete_session()

        history = controller.get_session_history(limit=3)
        assert len(history) == 3

    def test_max_history_enforced(self):
        """Test that max history is enforced."""
        config = StreamingControllerConfig(max_history=3)
        controller = StreamingController(config=config)

        for i in range(5):
            controller.start_session(f"model-{i}", "provider")
            controller.complete_session()

        # Only 3 sessions should be kept
        assert len(controller._session_history) == 3

    def test_get_current_session_info(self, controller):
        """Test getting current session info."""
        controller.start_session("gpt-4", "openai")
        controller.record_chunk(content_length=100)

        info = controller.get_current_session_info()

        assert info is not None
        assert info["model"] == "gpt-4"
        assert info["provider"] == "openai"
        assert info["is_active"] is True
        assert info["chunks_received"] == 1
        assert info["content_length"] == 100

    def test_get_current_session_info_no_session(self, controller):
        """Test getting current session info when none active."""
        info = controller.get_current_session_info()
        assert info is None

    def test_reset(self, controller):
        """Test resetting controller."""
        controller.start_session("gpt-4", "openai")
        controller.complete_session()
        controller.start_session("claude-3", "anthropic")

        controller.reset()

        assert controller.is_streaming is False
        assert len(controller._session_history) == 0

    def test_get_statistics(self, controller):
        """Test getting statistics."""
        controller.start_session("gpt-4", "openai")
        controller.complete_session()

        controller.start_session("claude-3", "anthropic")
        controller.cancel_session()
        controller.complete_session()

        stats = controller.get_statistics()

        assert stats["total_sessions"] == 2
        assert stats["completed"] == 1
        assert stats["cancelled"] == 1
        assert stats["is_streaming"] is False

    def test_on_session_complete_callback(self):
        """Test on_session_complete callback."""
        completed_sessions = []

        def on_complete(session):
            completed_sessions.append(session)

        controller = StreamingController(on_session_complete=on_complete)
        controller.start_session("gpt-4", "openai")
        controller.complete_session()

        assert len(completed_sessions) == 1
        assert completed_sessions[0].model == "gpt-4"

    def test_thread_safety(self, controller):
        """Test basic thread safety of is_streaming check."""
        import threading

        results = []

        def check_streaming():
            for _ in range(100):
                results.append(controller.is_streaming)

        controller.start_session("gpt-4", "openai")

        threads = [threading.Thread(target=check_streaming) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All checks should succeed without error
        assert len(results) == 500
