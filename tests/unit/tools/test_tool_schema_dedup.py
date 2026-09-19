"""TDD tests for tool selection caching in the streaming executor."""

from types import SimpleNamespace
import pytest
from unittest.mock import AsyncMock, MagicMock

from victor.agent.services.chat_planning import ChatPlanning
from victor.agent.services.chat_stream_executor import StreamingChatExecutor


def _pipeline(select_tools_for_turn):
    pipeline = StreamingChatExecutor.__new__(StreamingChatExecutor)
    pipeline._runtime_owner = SimpleNamespace(
        services=SimpleNamespace(
            planning=ChatPlanning(
                selection=SimpleNamespace(select_tools_for_turn=select_tools_for_turn)
            )
        )
    )
    pipeline._last_tool_context = None
    pipeline._last_tools = None
    return pipeline


class TestToolSelectionCaching:

    @pytest.mark.asyncio
    async def test_same_context_reuses_tools(self):
        selection = AsyncMock(return_value=[MagicMock(name="read"), MagicMock(name="write")])
        pipeline = _pipeline(selection)
        tools1 = await pipeline._get_tools_cached("fix the bug", None)
        assert selection.call_count == 1
        tools2 = await pipeline._get_tools_cached("fix the bug", None)
        assert selection.call_count == 1
        assert tools1 is tools2

    @pytest.mark.asyncio
    async def test_different_context_invalidates(self):
        selection = AsyncMock(return_value=[MagicMock(name="read")])
        pipeline = _pipeline(selection)
        await pipeline._get_tools_cached("fix the bug", None)
        assert selection.call_count == 1
        await pipeline._get_tools_cached("add a feature", None)
        assert selection.call_count == 2

    @pytest.mark.asyncio
    async def test_none_tools_not_cached(self):
        selection = AsyncMock(return_value=None)
        pipeline = _pipeline(selection)
        tools = await pipeline._get_tools_cached("hello", None)
        assert tools is None
        assert selection.call_count == 1
        await pipeline._get_tools_cached("hello", None)
        assert selection.call_count == 2
