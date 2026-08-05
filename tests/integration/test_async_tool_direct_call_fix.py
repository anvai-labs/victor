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

"""
Integration test for async tool direct call fix.

This test verifies that the @tool decorator properly handles async functions,
returning an async wrapper that can be awaited directly.

Background: The exploration_coordinator was importing ls directly and calling
it with `await ls(...)`, which was returning coroutine objects instead of
actual results. The fix makes the decorator return an async wrapper for async
functions, which properly awaits the underlying function.
"""

import inspect
import pytest

from victor.tools.decorators import tool


class TestAsyncToolDirectCall:
    """Test that async tools can be called directly with await."""

    @pytest.mark.asyncio
    async def test_async_tool_returns_actual_result_not_coroutine(self):
        """Verify that calling an async tool directly returns the actual result, not a coroutine object."""

        @tool
        async def my_async_tool(value: int) -> int:
            """A simple async tool for testing."""
            return value * 2

        # Call the tool directly (as exploration_coordinator does)
        result = await my_async_tool(value=5)

        # Verify we get the actual result, not a coroutine object
        assert result == 10
        assert not inspect.iscoroutine(result), "Result should not be a coroutine"

    @pytest.mark.asyncio
    async def test_async_tool_with_await_in_direct_call(self):
        """Verify that `await tool(...)` works correctly for async tools."""

        @tool
        async def async_calculator(a: int, b: int) -> dict:
            """Async calculator tool."""
            await asyncio.sleep(0)  # Simulate async operation
            return {"sum": a + b, "product": a * b}

        # This is how exploration_coordinator calls ls
        result = await async_calculator(a=3, b=4)

        # Verify structure
        assert isinstance(result, dict)
        assert result["sum"] == 7
        assert result["product"] == 12
        assert not inspect.iscoroutine(result)

    @pytest.mark.asyncio
    async def test_sync_tool_still_works(self):
        """Verify that sync tools still work correctly after the fix."""

        @tool
        def sync_tool(name: str) -> str:
            """A simple sync tool."""
            return f"Hello, {name}!"

        # Sync tools should not be awaited
        result = sync_tool(name="World")

        assert result == "Hello, World!"
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_tool_wrapper_is_coroutine_function(self):
        """Verify that the wrapper for async tools is a coroutine function."""

        @tool
        async def async_tool() -> str:
            """Async tool."""
            return "async result"

        # The wrapper should be a coroutine function
        assert inspect.iscoroutinefunction(async_tool), "Wrapper should be async for async tools"

    @pytest.mark.asyncio
    async def test_sync_tool_wrapper_is_not_coroutine_function(self):
        """Verify that the wrapper for sync tools is NOT a coroutine function."""

        @tool
        def sync_tool() -> str:
            """Sync tool."""
            return "sync result"

        # The wrapper should NOT be a coroutine function
        assert not inspect.iscoroutinefunction(sync_tool), "Wrapper should be sync for sync tools"

    @pytest.mark.asyncio
    async def test_actual_ls_tool_can_be_called_directly(self):
        """Verify that the actual ls tool from filesystem can be called directly."""

        from victor.tools.filesystem import ls

        # The ls tool should be a coroutine function
        assert inspect.iscoroutinefunction(ls), "ls tool should be async"

        # Call it directly (as exploration_coordinator does)
        result = await ls(path="victor/tools", depth=1)

        # Verify we get actual results, not a coroutine
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert not inspect.iscoroutine(result), "Result should not be a coroutine"

        # Verify structure - ls returns a dict with 'items' key
        assert "items" in result, "Result should have 'items' key"
        assert "count" in result, "Result should have 'count' key"

        # Verify items have structure
        if len(result["items"]) > 0:
            assert "name" in result["items"][0], "Items should have 'name' field"

    @pytest.mark.asyncio
    async def test_direct_call_does_not_stringify_coroutine(self):
        """Verify that direct calls don't stringify coroutine objects.

        This was the original bug: str(coroutine_object) produces
        "<coroutine object ls at 0x...>" which is 75 characters.
        """

        @tool
        async def async_tool() -> str:
            """Async tool that returns a meaningful result."""
            return "This is the actual result"

        result = await async_tool()

        # The result should be the actual return value, NOT a stringified coroutine
        assert result == "This is the actual result"
        assert not result.startswith(
            "<coroutine object"
        ), "Result should not be stringified coroutine"

        # Stringified coroutine would be ~75 chars like "<coroutine object async_tool at 0x...>"
        assert len(result) > 75 or len(result) < 50, "Result length doesn't match coroutine pattern"


# Import asyncio for the test
import asyncio
