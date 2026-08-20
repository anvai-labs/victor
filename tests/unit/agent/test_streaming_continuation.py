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

"""Tests for continuation action handling edge cases."""

from unittest.mock import MagicMock

import pytest

from victor.agent.conversation.history_metadata import build_internal_history_metadata
from victor.agent.conversation.types import MessageSource
from victor.agent.streaming.context import StreamingChatContext
from victor.agent.streaming.continuation import ContinuationHandler


@pytest.fixture
def continuation_handler():
    """Create a continuation handler with mocked dependencies."""
    return ContinuationHandler(
        message_adder=MagicMock(),
        chunk_generator=MagicMock(),
        sanitizer=MagicMock(),
        settings=MagicMock(),
    )


class TestContinuationSkipRules:
    """Tests for skip-continuation handling."""

    @pytest.mark.asyncio
    async def test_force_tool_execution_bypasses_skip_continuation(self, continuation_handler):
        """Forced tool recovery must still run even if completion was tentatively forced."""
        stream_ctx = StreamingChatContext(user_message="test", skip_continuation=True)

        result = await continuation_handler.handle_action(
            {
                "action": "force_tool_execution",
                "message": "Make the actual tool call now.",
                "mentioned_tools": ["read_file"],
            },
            stream_ctx,
            full_content="",
        )

        assert result.should_skip_rest is True
        continuation_handler._message_adder.add_message.assert_called_once_with(
            "user",
            "Make the actual tool call now.",
            metadata=build_internal_history_metadata(
                "force_tool_execution", source=MessageSource.AGENT_CONTINUATION
            ),
        )

    @pytest.mark.asyncio
    async def test_finish_still_respects_skip_continuation(self, continuation_handler):
        """Non-recovery actions should still be skipped when completion was forced."""
        stream_ctx = StreamingChatContext(user_message="test", skip_continuation=True)

        result = await continuation_handler.handle_action(
            {"action": "finish"},
            stream_ctx,
            full_content="",
        )

        assert result.should_return is True
        assert result.should_continue_loop is False
        continuation_handler._message_adder.add_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_prompt_tool_call_marks_internal_history_metadata(self, continuation_handler):
        stream_ctx = StreamingChatContext(user_message="test")

        result = await continuation_handler.handle_action(
            {
                "action": "prompt_tool_call",
                "message": "Continue. Use appropriate tools if needed.",
            },
            stream_ctx,
            full_content="",
        )

        assert result.should_skip_rest is True
        continuation_handler._message_adder.add_message.assert_called_once_with(
            "user",
            "Continue. Use appropriate tools if needed.",
            metadata=build_internal_history_metadata(
                "prompt_tool_call", source=MessageSource.AGENT_CONTINUATION
            ),
        )
