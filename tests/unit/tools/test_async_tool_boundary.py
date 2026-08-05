# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0

"""Tests verifying sync @tool functions are offloaded to threads.

FunctionTool.execute() should use asyncio.to_thread() for sync functions
to avoid blocking the event loop during CPU-bound operations.
"""

from __future__ import annotations

import inspect

from victor.tools.decorators import tool


def _get_tool_instance(fn):
    """Create a @tool wrapper and return the FunctionTool instance."""
    wrapper = tool(fn)
    return wrapper.Tool


class TestSyncToolOffloading:
    """Verify sync tool functions run via asyncio.to_thread()."""

    async def test_sync_tool_uses_to_thread(self):
        """Sync function inside FunctionTool.execute() calls asyncio.to_thread."""

        def my_sync_tool() -> str:
            """A sync tool."""
            return "sync_result"

        tool_instance = _get_tool_instance(my_sync_tool)
        src = inspect.getsource(type(tool_instance).execute)
        assert "to_thread" in src, "execute() should use asyncio.to_thread for sync fns"

    async def test_sync_tool_result_returned_correctly(self):
        """Sync tool via to_thread preserves return value."""

        def return_tool() -> str:
            """Returns a string."""
            return "hello_world"

        tool_instance = _get_tool_instance(return_tool)
        result = await tool_instance.execute(_exec_ctx={})
        assert result.success
        assert result.output == "hello_world"

    async def test_async_tool_awaited_directly(self):
        """Async function should be awaited directly, not via to_thread."""

        async def async_tool() -> str:
            """An async tool."""
            return "async_result"

        tool_instance = _get_tool_instance(async_tool)
        result = await tool_instance.execute(_exec_ctx={})
        assert result.success
        assert result.output == "async_result"

    async def test_sync_tool_exception_propagated(self):
        """Exceptions from sync tool surface through to_thread."""

        def failing_tool() -> str:
            """A tool that fails."""
            raise ValueError("deliberate failure")

        tool_instance = _get_tool_instance(failing_tool)
        result = await tool_instance.execute(_exec_ctx={})
        assert not result.success
        assert "deliberate failure" in (result.error or "")
