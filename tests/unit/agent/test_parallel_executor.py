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

"""Tests for ParallelToolExecutor."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.parallel_executor import (
    ParallelExecutionConfig,
    ParallelToolExecutor,
    ToolCategory,
    TOOL_CATEGORIES,
    create_parallel_executor,
)
from victor.agent.tool_executor import ToolExecutionResult


class TestToolCategories:
    """Tests for tool categorization."""

    def test_read_tools_categorized(self):
        """Test that read-only tools are properly categorized."""
        assert TOOL_CATEGORIES.get("read") == ToolCategory.READ_ONLY
        assert TOOL_CATEGORIES.get("ls") == ToolCategory.READ_ONLY
        assert TOOL_CATEGORIES.get("code_search") == ToolCategory.READ_ONLY

    def test_write_tools_categorized(self):
        """Test that write tools are properly categorized."""
        assert TOOL_CATEGORIES.get("write") == ToolCategory.WRITE
        assert TOOL_CATEGORIES.get("edit") == ToolCategory.WRITE
        assert TOOL_CATEGORIES.get("shell") == ToolCategory.WRITE

    def test_network_tools_categorized(self):
        """Test that network tools are properly categorized."""
        assert TOOL_CATEGORIES.get("web_search") == ToolCategory.NETWORK
        assert TOOL_CATEGORIES.get("web_fetch") == ToolCategory.NETWORK


class TestParallelExecutionConfig:
    """Tests for ParallelExecutionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ParallelExecutionConfig()
        assert config.max_concurrent == 5
        assert config.enable_parallel is True
        assert config.parallelize_reads is True
        assert config.timeout_per_tool == 60.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = ParallelExecutionConfig(
            max_concurrent=10,
            enable_parallel=False,
            timeout_per_tool=30.0,
        )
        assert config.max_concurrent == 10
        assert config.enable_parallel is False
        assert config.timeout_per_tool == 30.0


class TestParallelToolExecutor:
    """Tests for ParallelToolExecutor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tool_executor = MagicMock()
        self.mock_tool_executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="test_tool",
                success=True,
                result="test result",
                execution_time=0.1,
            )
        )

    def test_get_category_known_tool(self):
        """Test category lookup for known tools."""
        executor = ParallelToolExecutor(self.mock_tool_executor)
        assert executor._get_category("read") == ToolCategory.READ_ONLY
        assert executor._get_category("write") == ToolCategory.WRITE

    def test_get_category_alias_tool(self):
        """Legacy aliases should still resolve to canonical categories."""
        executor = ParallelToolExecutor(self.mock_tool_executor)
        assert executor._get_category("read_file") == ToolCategory.READ_ONLY
        assert executor._get_category("execute_bash") == ToolCategory.WRITE
        assert executor._get_category("notebook_edit") == ToolCategory.WRITE

    def test_get_category_unknown_tool(self):
        """Test category defaults to COMPUTE for unknown tools."""
        executor = ParallelToolExecutor(self.mock_tool_executor)
        assert executor._get_category("unknown_tool") == ToolCategory.COMPUTE

    def test_can_parallelize_single_tool(self):
        """Test that single tool calls don't parallelize."""
        executor = ParallelToolExecutor(self.mock_tool_executor)
        tool_calls = [{"name": "read_file", "arguments": {}}]
        assert executor._can_parallelize(tool_calls) is False

    def test_can_parallelize_read_tools(self):
        """Test that multiple read tools can parallelize."""
        executor = ParallelToolExecutor(self.mock_tool_executor)
        tool_calls = [
            {"name": "read_file", "arguments": {"path": "/a.py"}},
            {"name": "read_file", "arguments": {"path": "/b.py"}},
        ]
        assert executor._can_parallelize(tool_calls) is True

    def test_can_parallelize_with_writes_to_different_files(self):
        """Test that writes to different files CAN parallelize via dependency graph.

        Per-file write parallelism allows writes targeting distinct files
        to run concurrently. The dependency graph ensures serialization
        only for same-file conflicts.
        """
        executor = ParallelToolExecutor(self.mock_tool_executor)
        tool_calls = [
            {"name": "read_file", "arguments": {}},
            {"name": "write_file", "arguments": {}},
        ]
        # With per-file parallelism, _can_parallelize returns True
        # (safety is enforced by the dependency graph in execute_parallel)
        assert executor._can_parallelize(tool_calls) is True

    def test_cannot_parallelize_when_disabled(self):
        """Test that parallelization respects enable flag."""
        config = ParallelExecutionConfig(enable_parallel=False)
        executor = ParallelToolExecutor(self.mock_tool_executor, config=config)
        tool_calls = [
            {"name": "read_file", "arguments": {}},
            {"name": "read_file", "arguments": {}},
        ]
        assert executor._can_parallelize(tool_calls) is False

    @pytest.mark.asyncio
    async def test_execute_parallel_empty_calls(self):
        """Test parallel execution with no tool calls."""
        executor = ParallelToolExecutor(self.mock_tool_executor)
        result = await executor.execute_parallel([])
        assert result.completed_count == 0
        assert result.failed_count == 0

    @pytest.mark.asyncio
    async def test_execute_parallel_single_tool(self):
        """Test parallel execution with single tool (falls back to sequential)."""
        executor = ParallelToolExecutor(self.mock_tool_executor)
        tool_calls = [{"name": "read_file", "arguments": {"path": "/test.py"}}]

        result = await executor.execute_parallel(tool_calls)

        assert result.completed_count == 1
        assert result.failed_count == 0
        assert len(result.results) == 1

    @pytest.mark.asyncio
    async def test_execute_parallel_multiple_reads(self):
        """Test parallel execution with multiple read-only tools."""
        executor = ParallelToolExecutor(self.mock_tool_executor)
        tool_calls = [
            {"name": "read_file", "arguments": {"path": "/a.py"}},
            {"name": "read_file", "arguments": {"path": "/b.py"}},
            {"name": "list_directory", "arguments": {"path": "/"}},
        ]

        result = await executor.execute_parallel(tool_calls)

        assert result.completed_count == 3
        assert result.failed_count == 0
        assert len(result.results) == 3

    @pytest.mark.asyncio
    async def test_execute_parallel_with_failure(self):
        """Test parallel execution handles failures."""
        self.mock_tool_executor.execute = AsyncMock(
            side_effect=[
                ToolExecutionResult(
                    tool_name="read_file",
                    success=True,
                    result="ok",
                    execution_time=0.1,
                ),
                ToolExecutionResult(
                    tool_name="read_file",
                    success=False,
                    result=None,
                    error="File not found",
                    execution_time=0.1,
                ),
            ]
        )

        executor = ParallelToolExecutor(self.mock_tool_executor)
        tool_calls = [
            {"name": "read_file", "arguments": {"path": "/exists.py"}},
            {"name": "read_file", "arguments": {"path": "/missing.py"}},
        ]

        result = await executor.execute_parallel(tool_calls)

        assert result.completed_count == 1
        assert result.failed_count == 1
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_progress_callback(self):
        """Test that progress callback is invoked."""
        callback_calls = []

        def callback(tool_name: str, status: str, success: bool):
            callback_calls.append((tool_name, status, success))

        executor = ParallelToolExecutor(self.mock_tool_executor, progress_callback=callback)
        tool_calls = [{"name": "read_file", "arguments": {"path": "/test.py"}}]

        await executor.execute_parallel(tool_calls)

        assert len(callback_calls) == 2
        assert callback_calls[0] == ("read_file", "started", True)
        assert callback_calls[1] == ("read_file", "completed", True)


class TestCreateParallelExecutor:
    """Tests for the factory function."""

    def test_create_parallel_executor(self):
        """Test factory creates executor with correct config."""
        mock_tool_executor = MagicMock()
        executor = create_parallel_executor(
            tool_executor=mock_tool_executor,
            max_concurrent=10,
            enable=True,
        )

        assert isinstance(executor, ParallelToolExecutor)
        assert executor.config.max_concurrent == 10
        assert executor.config.enable_parallel is True

    def test_create_parallel_executor_disabled(self):
        """Test factory respects enable flag."""
        mock_tool_executor = MagicMock()
        executor = create_parallel_executor(
            tool_executor=mock_tool_executor,
            enable=False,
        )

        assert executor.config.enable_parallel is False
