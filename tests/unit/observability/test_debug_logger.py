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

"""Tests for debug logging utilities."""

import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from victor.agent.debug_logger import (
    TRACE,
    ConversationStats,
    DebugLogger,
    configure_debug_logging,
    configure_logging_levels,
    get_debug_logger,
    trace,
    NOISY_LOGGERS,
)
from victor.providers.base import Message

# =============================================================================
# TRACE LEVEL TESTS
# =============================================================================


class TestTraceLevel:
    """Tests for custom TRACE level."""

    def test_trace_level_value(self):
        """Test TRACE level has correct value."""
        assert TRACE == 5
        assert TRACE < logging.DEBUG

    def test_trace_level_named(self):
        """Test TRACE level is named correctly."""
        assert logging.getLevelName(TRACE) == "TRACE"

    def test_trace_method_exists(self):
        """Test trace method added to Logger."""
        test_logger = logging.getLogger("test_trace")
        assert hasattr(test_logger, "trace")

    def test_trace_method_calls_log(self):
        """Test trace method calls _log."""
        test_logger = logging.getLogger("test_trace_call")
        test_logger.setLevel(TRACE)

        with patch.object(test_logger, "_log") as mock_log:
            test_logger.trace("test message")
            mock_log.assert_called_once()
            args = mock_log.call_args
            assert args[0][0] == TRACE
            assert args[0][1] == "test message"

    def test_trace_method_respects_level(self):
        """Test trace method respects log level."""
        test_logger = logging.getLogger("test_trace_level")
        test_logger.setLevel(logging.DEBUG)  # Higher than TRACE

        with patch.object(test_logger, "_log") as mock_log:
            test_logger.trace("test message")
            mock_log.assert_not_called()


# =============================================================================
# CONFIGURE LOGGING LEVELS TESTS
# =============================================================================


class TestConfigureLoggingLevels:
    """Tests for configure_logging_levels function."""

    def test_configure_info_level(self):
        """Test configuring INFO level."""
        configure_logging_levels("INFO")
        victor_logger = logging.getLogger("victor")
        assert victor_logger.level == logging.INFO

    def test_configure_debug_level(self):
        """Test configuring DEBUG level."""
        configure_logging_levels("DEBUG")
        victor_logger = logging.getLogger("victor")
        assert victor_logger.level == logging.DEBUG

    def test_configure_trace_level(self):
        """Test configuring TRACE level."""
        configure_logging_levels("TRACE")
        victor_logger = logging.getLogger("victor")
        assert victor_logger.level == TRACE

    def test_configure_case_insensitive(self):
        """Test level name is case insensitive."""
        configure_logging_levels("info")
        victor_logger = logging.getLogger("victor")
        assert victor_logger.level == logging.INFO

    def test_silences_noisy_loggers(self):
        """Test noisy loggers are silenced."""
        configure_logging_levels("DEBUG")

        for logger_name in NOISY_LOGGERS:
            noisy = logging.getLogger(logger_name)
            assert noisy.level >= logging.WARNING


# =============================================================================
# CONVERSATION STATS TESTS
# =============================================================================


class TestConversationStats:
    """Tests for ConversationStats dataclass."""

    def test_default_values(self):
        """Test default stat values."""
        stats = ConversationStats()
        assert stats.total_messages == 0
        assert stats.total_chars == 0
        assert stats.user_messages == 0
        assert stats.assistant_messages == 0
        assert stats.tool_messages == 0
        assert stats.tool_calls_made == 0
        assert stats.iterations == 0
        assert stats.start_time > 0

    def test_elapsed_seconds(self):
        """Test elapsed_seconds calculation."""
        stats = ConversationStats()
        time.sleep(0.1)
        elapsed = stats.elapsed_seconds
        assert elapsed >= 0.1

    def test_summary_format(self):
        """Test summary format."""
        stats = ConversationStats()
        stats.total_messages = 10
        stats.total_chars = 5000
        stats.tool_calls_made = 3
        stats.iterations = 2

        summary = stats.summary()

        assert "msgs=10" in summary
        assert "5,000 chars" in summary
        assert "tools=3" in summary
        assert "iter=2" in summary
        assert "s" in summary  # Seconds indicator


# =============================================================================
# DEBUG LOGGER TESTS
# =============================================================================


class TestDebugLogger:
    """Tests for DebugLogger class."""

    @pytest.fixture
    def debug_logger(self):
        """Create a debug logger."""
        return DebugLogger(name="test.debug")

    @pytest.fixture
    def disabled_logger(self):
        """Create a disabled debug logger."""
        return DebugLogger(name="test.disabled", enabled=False)

    def test_init_default(self, debug_logger):
        """Test default initialization."""
        assert debug_logger.max_preview == 80
        assert debug_logger.enabled is True
        assert debug_logger._last_iteration == 0

    def test_init_custom(self):
        """Test custom initialization."""
        logger = DebugLogger(name="test", max_preview=50, enabled=False)
        assert logger.max_preview == 50
        assert logger.enabled is False

    def test_reset(self, debug_logger):
        """Test state reset."""
        debug_logger._last_iteration = 5
        debug_logger.stats.tool_calls_made = 10

        debug_logger.reset()

        assert debug_logger._last_iteration == 0
        assert debug_logger.stats.tool_calls_made == 0


class TestTruncate:
    """Tests for _truncate method."""

    @pytest.fixture
    def debug_logger(self):
        return DebugLogger(max_preview=20)

    def test_short_text_unchanged(self, debug_logger):
        """Test short text not truncated."""
        result = debug_logger._truncate("short")
        assert result == "short"

    def test_long_text_truncated(self, debug_logger):
        """Test long text is truncated."""
        long_text = "a" * 50
        result = debug_logger._truncate(long_text)
        assert len(result) == 23  # 20 + "..."
        assert result.endswith("...")

    def test_newlines_removed(self, debug_logger):
        """Test newlines are replaced with spaces."""
        text = "line1\nline2\nline3"
        result = debug_logger._truncate(text)
        assert "\n" not in result
        assert " " in result

    def test_whitespace_stripped(self, debug_logger):
        """Test whitespace is stripped."""
        text = "  text with spaces  "
        result = debug_logger._truncate(text)
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_custom_max_len(self, debug_logger):
        """Test custom max length."""
        result = debug_logger._truncate("a" * 50, max_len=10)
        assert len(result) == 13  # 10 + "..."


class TestLogIterationStart:
    """Tests for log_iteration_start method."""

    @pytest.fixture
    def debug_logger(self):
        return DebugLogger(name="test.iter")

    def test_logs_iteration(self, debug_logger):
        """Test iteration start is logged."""
        with patch.object(debug_logger.logger, "info") as mock_info:
            debug_logger.log_iteration_start(1)

            mock_info.assert_called_once()
            assert "ITER 1" in mock_info.call_args[0][0]

    def test_updates_last_iteration(self, debug_logger):
        """Test last iteration is updated."""
        debug_logger.log_iteration_start(1)
        assert debug_logger._last_iteration == 1

    def test_updates_stats(self, debug_logger):
        """Test stats iterations are updated."""
        debug_logger.log_iteration_start(3)
        assert debug_logger.stats.iterations == 3

    def test_skips_duplicate_iteration(self, debug_logger):
        """Test duplicate iteration is skipped."""
        debug_logger.log_iteration_start(1)
        with patch.object(debug_logger.logger, "info") as mock_info:
            debug_logger.log_iteration_start(1)
            mock_info.assert_not_called()

    def test_skips_when_disabled(self, debug_logger):
        """Test logging skipped when disabled."""
        debug_logger.enabled = False
        with patch.object(debug_logger.logger, "info") as mock_info:
            debug_logger.log_iteration_start(1)
            mock_info.assert_not_called()


class TestLogIterationEnd:
    """Tests for log_iteration_end method."""

    @pytest.fixture
    def debug_logger(self):
        return DebugLogger(name="test.iter_end")

    def test_logs_summary(self, debug_logger):
        """Test iteration end logs summary."""
        with patch.object(debug_logger.logger, "info") as mock_info:
            debug_logger.log_iteration_end(1)
            mock_info.assert_called_once()

    def test_shows_tools_indicator(self, debug_logger):
        """Test shows tools indicator when has tool calls."""
        with patch.object(debug_logger.logger, "info") as mock_info:
            debug_logger.log_iteration_end(1, has_tool_calls=True)
            message = mock_info.call_args[0][0]
            # Check for either emoji (→) or text (->) version of arrow
            assert "→ tools" in message or "-> tools" in message

    def test_shows_done_indicator(self, debug_logger):
        """Test shows done indicator when no tool calls."""
        with patch.object(debug_logger.logger, "info") as mock_info:
            debug_logger.log_iteration_end(1, has_tool_calls=False)
            message = mock_info.call_args[0][0]
            # Check for either emoji (→) or text (->) version of arrow
            assert "→ done" in message or "-> done" in message


class TestLogToolCall:
    """Tests for log_tool_call method."""

    @pytest.fixture
    def debug_logger(self):
        return DebugLogger(name="test.tool")

    def test_logs_tool_name(self, debug_logger):
        """Test tool name is logged."""
        with patch.object(debug_logger.logger, "info") as mock_info:
            debug_logger.log_tool_call("read_file", {"path": "test.py"}, 1)
            assert "read_file" in mock_info.call_args[0][0]

    def test_increments_tool_calls(self, debug_logger):
        """Test tool calls are incremented."""
        debug_logger.log_tool_call("tool1", {}, 1)
        debug_logger.log_tool_call("tool2", {}, 1)
        assert debug_logger.stats.tool_calls_made == 2

    def test_truncates_long_args(self, debug_logger):
        """Long args are truncated to a size-prefixed head…tail preview.

        ``_format_value`` renders ``(N chars): <first 60>...<last 30>`` for long
        values, which only shortens inputs well above ~106 chars; use a clearly
        long value so truncation is actually exercised.
        """
        long_value = "a" * 1000
        with patch.object(debug_logger.logger, "info") as mock_info:
            debug_logger.log_tool_call("test", {"long": long_value}, 1)
            call_str = mock_info.call_args[0][0]
            assert len(call_str) < len(long_value)
            assert "..." in call_str

    def test_handles_many_args(self, debug_logger):
        """Test many args shows +N more."""
        args = {f"arg{i}": f"val{i}" for i in range(5)}
        with patch.object(debug_logger.logger, "info") as mock_info:
            debug_logger.log_tool_call("test", args, 1)
            assert "+2 more" in mock_info.call_args[0][0]


class TestLogToolResult:
    """Tests for log_tool_result method."""

    @pytest.fixture
    def debug_logger(self):
        return DebugLogger(name="test.result")

    def test_logs_success(self, debug_logger):
        """Test success result logged."""
        with patch.object(debug_logger.logger, "info") as mock_info:
            debug_logger.log_tool_result("test", True, "output", 100.0)
            call_str = mock_info.call_args[0][0]
            # Accept both emoji (✓) and text (+/√) versions
            assert "✓" in call_str or "√" in call_str or "+" in call_str
            assert "test" in call_str

    def test_logs_failure(self, debug_logger):
        """Test failure result logged."""
        with patch.object(debug_logger.logger, "info") as mock_info:
            debug_logger.log_tool_result("test", False, "error", 50.0)
            call_str = mock_info.call_args[0][0]
            # Accept both emoji (✗) and text (x) versions
            assert "✗" in call_str or "x" in call_str

    def test_logs_output_size(self, debug_logger):
        """Test output size is logged."""
        output = "a" * 1000
        with patch.object(debug_logger.logger, "info") as mock_info:
            debug_logger.log_tool_result("test", True, output, 50.0)
            assert "1,000 chars" in mock_info.call_args[0][0]

    def test_logs_elapsed_time(self, debug_logger):
        """Test elapsed time is logged."""
        with patch.object(debug_logger.logger, "info") as mock_info:
            debug_logger.log_tool_result("test", True, "out", 123.4)
            assert "123ms" in mock_info.call_args[0][0]

    def test_handles_empty_output(self, debug_logger):
        """Test empty output handled."""
        with patch.object(debug_logger.logger, "info") as mock_info:
            debug_logger.log_tool_result("test", True, "", 50.0)
            assert "empty" in mock_info.call_args[0][0]


class TestLogModelResponse:
    """Tests for log_model_response method."""

    @pytest.fixture
    def debug_logger(self):
        return DebugLogger(name="test.model")

    def test_logs_at_debug_level(self, debug_logger):
        """Test logged at debug level."""
        with patch.object(debug_logger.logger, "debug") as mock_debug:
            debug_logger.log_model_response("response", False, 1)
            mock_debug.assert_called_once()

    def test_shows_content_length(self, debug_logger):
        """Test content length is shown."""
        content = "a" * 100
        with patch.object(debug_logger.logger, "debug") as mock_debug:
            debug_logger.log_model_response(content, False, 1)
            assert "100 chars" in mock_debug.call_args[0][0]

    def test_shows_tools_indicator(self, debug_logger):
        """Test shows +tools when has tool calls."""
        with patch.object(debug_logger.logger, "debug") as mock_debug:
            debug_logger.log_model_response("content", True, 1)
            assert "+tools" in mock_debug.call_args[0][0]


class TestLogNewMessages:
    """Tests for log_new_messages method."""

    @pytest.fixture
    def debug_logger(self):
        return DebugLogger(name="test.messages")

    def test_updates_stats(self, debug_logger):
        """Test stats are updated."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="tool", content="tool output"),
        ]

        debug_logger.log_new_messages(messages)

        assert debug_logger.stats.total_messages == 3
        assert debug_logger.stats.user_messages == 1
        assert debug_logger.stats.assistant_messages == 1
        assert debug_logger.stats.tool_messages == 1

    def test_calculates_total_chars(self, debug_logger):
        """Test total chars calculated."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="World"),
        ]

        debug_logger.log_new_messages(messages)

        assert debug_logger.stats.total_chars == 10


class TestLogLimits:
    """Tests for log_limits method."""

    @pytest.fixture
    def debug_logger(self):
        return DebugLogger(name="test.limits")

    def test_logs_at_debug_level(self, debug_logger):
        """Test logged at debug level."""
        with patch.object(debug_logger.logger, "debug") as mock_debug:
            debug_logger.log_limits(10, 5, 20, 10, False)
            mock_debug.assert_called_once()

    def test_calculates_percentages(self, debug_logger):
        """Test percentages are calculated."""
        with patch.object(debug_logger.logger, "debug") as mock_debug:
            debug_logger.log_limits(10, 5, 20, 10, False)
            call_str = mock_debug.call_args[0][0]
            assert "50%" in call_str  # 5/10 = 50%

    def test_shows_mode(self, debug_logger):
        """Test mode is shown."""
        with patch.object(debug_logger.logger, "debug") as mock_debug:
            debug_logger.log_limits(10, 5, 20, 10, True)
            assert "analysis" in mock_debug.call_args[0][0]

        with patch.object(debug_logger.logger, "debug") as mock_debug:
            debug_logger.log_limits(10, 5, 20, 10, False)
            assert "standard" in mock_debug.call_args[0][0]


class TestLogContextSize:
    """Tests for log_context_size method."""

    @pytest.fixture
    def debug_logger(self):
        return DebugLogger(name="test.context")

    def test_warns_very_large_context(self, debug_logger):
        """Test warning for very large context."""
        # Set explicit context limits for predictable test behavior
        debug_logger.max_context_chars = 200_000
        debug_logger.max_context_tokens = 50_000
        debug_logger.warn_threshold_chars = 150_000  # 75%
        debug_logger.info_threshold_chars = 100_000  # 50%

        with patch.object(debug_logger.logger, "warning") as mock_warn:
            debug_logger.log_context_size(150_001, 37_500)
            mock_warn.assert_called_once()
            assert "Large context" in mock_warn.call_args[0][0]
            assert "75.0% of capacity" in mock_warn.call_args[0][0]

    def test_info_medium_context(self, debug_logger):
        """Test info for medium context."""
        # Set explicit context limits for predictable test behavior
        debug_logger.max_context_chars = 200_000
        debug_logger.max_context_tokens = 50_000
        debug_logger.warn_threshold_chars = 150_000  # 75%
        debug_logger.info_threshold_chars = 100_000  # 50%

        with patch.object(debug_logger.logger, "info") as mock_info:
            debug_logger.log_context_size(100_001, 25_000)
            mock_info.assert_called_once()
            assert "Context" in mock_info.call_args[0][0]
            assert "50.0% of capacity" in mock_info.call_args[0][0]

    def test_no_log_small_context(self, debug_logger):
        """Test no log for small context."""
        # Set explicit context limits for predictable test behavior
        debug_logger.max_context_chars = 200_000
        debug_logger.max_context_tokens = 50_000
        debug_logger.warn_threshold_chars = 150_000  # 75%
        debug_logger.info_threshold_chars = 100_000  # 50%

        with patch.object(debug_logger.logger, "warning") as mock_warn:
            with patch.object(debug_logger.logger, "info") as mock_info:
                debug_logger.log_context_size(10_000, 2_500)
                mock_warn.assert_not_called()
                mock_info.assert_not_called()


class TestLogConversationSummary:
    """Tests for log_conversation_summary method."""

    @pytest.fixture
    def debug_logger(self):
        return DebugLogger(name="test.summary")

    def test_logs_summary(self, debug_logger):
        """Test summary is logged."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]

        with patch.object(debug_logger.logger, "info") as mock_info:
            debug_logger.log_conversation_summary(messages)
            mock_info.assert_called_once()
            call_str = mock_info.call_args[0][0]
            assert "SUMMARY" in call_str
            assert "Messages:" in call_str


# =============================================================================
# GLOBAL FUNCTIONS TESTS
# =============================================================================


class TestGlobalFunctions:
    """Tests for global logger functions."""

    def test_get_debug_logger_returns_instance(self):
        """Test get_debug_logger returns an instance."""
        logger = get_debug_logger()
        assert isinstance(logger, DebugLogger)

    def test_get_debug_logger_singleton(self):
        """Test get_debug_logger returns same instance."""
        logger1 = get_debug_logger()
        logger2 = get_debug_logger()
        assert logger1 is logger2

    def test_configure_debug_logging(self):
        """Test configure_debug_logging returns configured logger."""
        logger = configure_debug_logging(
            enabled=True,
            max_preview=100,
            log_level="DEBUG",
            max_context_chars=500_000,
            max_context_tokens=125_000,
        )

        assert isinstance(logger, DebugLogger)
        assert logger.enabled is True
        assert logger.max_preview == 100
        assert logger.max_context_chars == 500_000
        assert logger.max_context_tokens == 125_000
        assert logger.warn_threshold_chars == 375_000  # 75% of 500K
        assert logger.info_threshold_chars == 250_000  # 50% of 500K

    def test_configure_debug_logging_disabled(self):
        """Test configure_debug_logging with disabled."""
        logger = configure_debug_logging(enabled=False)
        assert logger.enabled is False


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_disabled_logger_no_op(self):
        """Test disabled logger is no-op."""
        logger = DebugLogger(enabled=False)

        # All methods should be no-op when disabled
        with patch.object(logger.logger, "info") as mock_info:
            with patch.object(logger.logger, "debug") as mock_debug:
                logger.log_iteration_start(1)
                logger.log_iteration_end(1)
                logger.log_tool_call("test", {}, 1)
                logger.log_tool_result("test", True, "out", 10)
                logger.log_model_response("resp", False, 1)
                logger.log_limits(10, 5, 20, 10, False)
                logger.log_context_size(100000, 25000)
                logger.log_conversation_summary([])

                mock_info.assert_not_called()
                mock_debug.assert_not_called()

    def test_empty_args_dict(self):
        """Test empty args dict."""
        logger = DebugLogger()
        with patch.object(logger.logger, "info") as mock_info:
            logger.log_tool_call("test", {}, 1)
            assert "test()" in mock_info.call_args[0][0]

    def test_none_output(self):
        """Test None output in log_tool_result."""
        logger = DebugLogger()
        with patch.object(logger.logger, "info") as mock_info:
            logger.log_tool_result("test", True, None, 50)
            assert "empty" in mock_info.call_args[0][0]

    def test_zero_budget(self):
        """Test zero budget in log_limits."""
        logger = DebugLogger()
        with patch.object(logger.logger, "debug") as mock_debug:
            # Should not raise division by zero
            logger.log_limits(0, 0, 0, 0, False)
            mock_debug.assert_called_once()

    def test_newlines_in_model_response(self):
        """Test newlines are handled in model response."""
        logger = DebugLogger()
        with patch.object(logger.logger, "debug") as mock_debug:
            logger.log_model_response("line1\nline2\nline3", False, 1)
            call_str = mock_debug.call_args[0][0]
            # Newlines should be replaced
            assert "line1" in call_str

    def test_unicode_in_log(self):
        """Test unicode content is handled."""
        logger = DebugLogger()
        with patch.object(logger.logger, "info") as mock_info:
            logger.log_tool_result("test", True, "日本語テスト", 50)
            mock_info.assert_called_once()
