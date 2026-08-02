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

"""Unit tests for async tool executor."""

from __future__ import annotations

import asyncio
import pytest

from victor.agent.tool_execution.async_executor import (
    AsyncToolExecutor,
    ExecutionResult,
    ExecutionConfig,
)
from victor.agent.tool_execution.categorization import (
    ToolCategory,
    ToolCallSpec,
    categorize_tool_call,
    extract_files_from_args,
)


class TestToolCategorization:
    """Test suite for tool categorization."""

    def test_categorize_read_tools(self):
        """Test that read-only tools are categorized correctly."""
        read_tools = [
            "read_file",
            "list_directory",
            "code_search",
            "semantic_code_search",
            "grep_search",
            "plan_files",
            "git",
            "directory_tree",
            "file_info",
        ]

        for tool_name in read_tools:
            category = categorize_tool_call(tool_name, {})
            assert category == ToolCategory.READ_ONLY, f"{tool_name} should be READ_ONLY"

    def test_categorize_write_tools(self):
        """Test that write tools are categorized correctly."""
        write_tools = [
            "write_file",
            "edit_files",
            "execute_bash",
            "docker",
            "patch",
            "create_file",
            "notebook_edit",
        ]

        for tool_name in write_tools:
            category = categorize_tool_call(tool_name, {})
            assert category == ToolCategory.WRITE, f"{tool_name} should be WRITE"

    def test_categorize_network_tools(self):
        """Test that network tools are categorized correctly."""
        network_tools = [
            "web_search",
            "web_fetch",
            "http_request",
            "fetch",
        ]

        for tool_name in network_tools:
            category = categorize_tool_call(tool_name, {})
            assert category == ToolCategory.NETWORK, f"{tool_name} should be NETWORK"

    def test_categorize_unknown_tool(self):
        """Test that unknown tools default to COMPUTE."""
        category = categorize_tool_call("unknown_tool", {})
        assert category == ToolCategory.COMPUTE

    def test_extract_files_from_single_arg(self):
        """Test extracting files from single string argument."""
        args = {"file_path": "/path/to/file.py"}
        files = extract_files_from_args(args)
        assert files == ["/path/to/file.py"]

    def test_extract_files_from_list_arg(self):
        """Test extracting files from list argument."""
        args = {"files": ["/path/to/file1.py", "/path/to/file2.py"]}
        files = extract_files_from_args(args)
        assert files == ["/path/to/file1.py", "/path/to/file2.py"]

    def test_extract_files_from_multiple_args(self):
        """Test extracting files from multiple arguments."""
        args = {
            "file_path": "/path/to/file1.py",
            "path": "/path/to/file2.py",
        }
        files = extract_files_from_args(args)
        assert set(files) == {"/path/to/file1.py", "/path/to/file2.py"}

    def test_extract_files_no_files(self):
        """Test extracting files when no files present."""
        args = {"query": "test", "limit": 10}
        files = extract_files_from_args(args)
        assert files == []


class TestAsyncToolExecutor:
    """Test suite for AsyncToolExecutor."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        return AsyncToolExecutor(
            config=ExecutionConfig(
                max_concurrent=5,
                enable_write_parallelization=True,
                enable_priority_scheduling=True,
                default_timeout=30.0,
            )
        )

    @pytest.mark.asyncio
    async def test_execute_single_tool(self, executor):
        """Test executing a single tool call."""

        async def mock_executor(call: ToolCallSpec) -> str:
            return f"Result for {call.name}"

        spec = ToolCallSpec(
            name="test_tool",
            arguments={"arg1": "value1"},
            call_id="test-1",
            category=ToolCategory.READ_ONLY,
        )

        result = await executor.execute_tool_call(spec, mock_executor)

        assert result.call_id == "test-1"
        assert result.success is True
        assert result.result == "Result for test_tool"
        assert result.error is None
        assert result.duration_ms > 0
        assert result.parallelizable is True

    @pytest.mark.asyncio
    async def test_execute_parallel_read_tools(self, executor):
        """Test parallel execution of read-only tools."""

        async def mock_executor(call: ToolCallSpec) -> str:
            # Simulate some work
            await asyncio.sleep(0.1)
            return f"Result for {call.name}"

        specs = [
            ToolCallSpec(
                name=f"read_file_{i}",
                arguments={"file_path": f"/path/to/file{i}.py"},
                call_id=f"read-{i}",
                category=ToolCategory.READ_ONLY,
            )
            for i in range(5)
        ]

        results = await executor.execute_tool_calls(specs, mock_executor)

        assert len(results) == 5
        assert all(r.success for r in results)
        assert all(r.parallelizable for r in results)

    @pytest.mark.asyncio
    async def test_execute_sequential_write_tools(self, executor):
        """Test sequential execution of write tools to same file."""

        execution_order = []
        lock = asyncio.Lock()

        async def mock_executor(call: ToolCallSpec) -> str:
            # Track execution order
            async with lock:
                execution_order.append(call.call_id)
            await asyncio.sleep(0.05)
            return f"Result for {call.name}"

        specs = [
            ToolCallSpec(
                name="write_file",
                arguments={"file_path": "/path/to/file.py"},
                call_id=f"write-{i}",
                category=ToolCategory.WRITE,
            )
            for i in range(3)
        ]

        results = await executor.execute_tool_calls(specs, mock_executor)

        assert len(results) == 3
        assert all(r.success for r in results)
        # Check sequential execution
        assert execution_order == ["write-0", "write-1", "write-2"]

    @pytest.mark.asyncio
    async def test_execute_with_dependencies(self, executor):
        """Test execution with explicit dependencies."""

        execution_order = []
        lock = asyncio.Lock()

        async def mock_executor(call: ToolCallSpec) -> str:
            async with lock:
                execution_order.append(call.call_id)
            await asyncio.sleep(0.05)
            return f"Result for {call.name}"

        specs = [
            ToolCallSpec(
                name="read_file_1",
                arguments={"file_path": "/path/to/file1.py"},
                call_id="read-1",
                category=ToolCategory.READ_ONLY,
                dependencies=set(),  # No dependencies
            ),
            ToolCallSpec(
                name="read_file_2",
                arguments={"file_path": "/path/to/file2.py"},
                call_id="read-2",
                category=ToolCategory.READ_ONLY,
                dependencies={"read-1"},  # Depends on read-1
            ),
            ToolCallSpec(
                name="write_file",
                arguments={"file_path": "/path/to/file3.py"},
                call_id="write-1",
                category=ToolCategory.WRITE,
                dependencies={"read-2"},  # Depends on read-2
            ),
        ]

        results = await executor.execute_tool_calls(specs, mock_executor)

        assert len(results) == 3
        assert all(r.success for r in results)
        # Check dependency order
        assert execution_order.index("read-1") < execution_order.index("read-2")
        assert execution_order.index("read-2") < execution_order.index("write-1")

    @pytest.mark.asyncio
    async def test_execute_with_priority_scheduling(self, executor):
        """Test priority-based scheduling."""

        execution_order = []
        lock = asyncio.Lock()

        async def mock_executor(call: ToolCallSpec) -> str:
            async with lock:
                execution_order.append(call.call_id)
            await asyncio.sleep(0.05)
            return f"Result for {call.name}"

        specs = [
            ToolCallSpec(
                name="low_priority",
                arguments={},
                call_id="low",
                category=ToolCategory.READ_ONLY,
                priority=1,
            ),
            ToolCallSpec(
                name="high_priority",
                arguments={},
                call_id="high",
                category=ToolCategory.READ_ONLY,
                priority=10,
            ),
            ToolCallSpec(
                name="medium_priority",
                arguments={},
                call_id="medium",
                category=ToolCategory.READ_ONLY,
                priority=5,
            ),
        ]

        results = await executor.execute_tool_calls(specs, mock_executor)

        assert len(results) == 3
        assert all(r.success for r in results)
        # High priority should execute first
        assert execution_order[0] == "high"

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self, executor):
        """Test timeout handling."""

        async def slow_executor(call: ToolCallSpec) -> str:
            await asyncio.sleep(5)  # Sleep longer than timeout
            return "Should not reach here"

        spec = ToolCallSpec(
            name="slow_tool",
            arguments={},
            call_id="slow-1",
            category=ToolCategory.COMPUTE,
            timeout=0.1,  # 100ms timeout
        )

        result = await executor.execute_tool_call(spec, slow_executor)

        assert result.success is False
        assert "Timeout" in result.error
        assert result.result is None

    @pytest.mark.asyncio
    async def test_execute_with_error(self, executor):
        """Test error handling."""

        async def failing_executor(call: ToolCallSpec) -> str:
            raise ValueError("Test error")

        spec = ToolCallSpec(
            name="failing_tool",
            arguments={},
            call_id="fail-1",
            category=ToolCategory.COMPUTE,
        )

        result = await executor.execute_tool_call(spec, failing_executor)

        assert result.success is False
        assert "Test error" in result.error
        assert result.result is None

    @pytest.mark.asyncio
    async def test_concurrent_limit(self, executor):
        """Test concurrency limit enforcement."""

        active_count = 0
        max_active = 0
        lock = asyncio.Lock()

        async def mock_executor(call: ToolCallSpec) -> str:
            nonlocal active_count, max_active

            async with lock:
                active_count += 1
                max_active = max(max_active, active_count)

            await asyncio.sleep(0.1)

            async with lock:
                active_count -= 1

            return f"Result for {call.name}"

        # Create 10 tool calls
        specs = [
            ToolCallSpec(
                name=f"tool_{i}",
                arguments={},
                call_id=f"tool-{i}",
                category=ToolCategory.READ_ONLY,
            )
            for i in range(10)
        ]

        results = await executor.execute_tool_calls(specs, mock_executor)

        assert len(results) == 10
        assert all(r.success for r in results)
        # Should not exceed max_concurrent of 5
        assert max_active <= 5

    @pytest.mark.asyncio
    async def test_file_locking_for_writes(self, executor):
        """Test file locking for concurrent writes."""

        access_count = {}
        lock = asyncio.Lock()

        async def mock_executor(call: ToolCallSpec) -> str:
            file_path = call.arguments["file_path"]

            async with lock:
                access_count[file_path] = access_count.get(file_path, 0) + 1
                current_count = access_count[file_path]

            # Verify no concurrent access to same file
            await asyncio.sleep(0.1)

            async with lock:
                # Count should have decreased
                assert access_count[file_path] <= current_count

            async with lock:
                access_count[file_path] -= 1

            return f"Result for {call.name}"

        specs = [
            ToolCallSpec(
                name="write_file",
                arguments={"file_path": "/path/to/file.py"},
                call_id=f"write-{i}",
                category=ToolCategory.WRITE,
            )
            for i in range(3)
        ]

        results = await executor.execute_tool_calls(specs, mock_executor)

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_get_stats(self, executor):
        """Test execution statistics."""

        async def mock_executor(call: ToolCallSpec) -> str:
            return f"Result for {call.name}"

        specs = [
            ToolCallSpec(
                name="read_tool",
                arguments={},
                call_id="read-1",
                category=ToolCategory.READ_ONLY,
            ),
            ToolCallSpec(
                name="write_tool",
                arguments={},
                call_id="write-1",
                category=ToolCategory.WRITE,
            ),
        ]

        await executor.execute_tool_calls(specs, mock_executor)

        stats = executor.get_stats()

        assert stats["total_executions"] == 2
        assert stats["parallel_executions"] > 0
        assert stats["sequential_executions"] > 0
        assert stats["total_duration_ms"] > 0
        assert 0.0 <= stats["parallel_ratio"] <= 1.0

    @pytest.mark.asyncio
    async def test_reset_stats(self, executor):
        """Test statistics reset."""

        async def mock_executor(call: ToolCallSpec) -> str:
            return f"Result for {call.name}"

        specs = [
            ToolCallSpec(
                name="test_tool",
                arguments={},
                call_id="test-1",
                category=ToolCategory.READ_ONLY,
            )
        ]

        await executor.execute_tool_calls(specs, mock_executor)
        assert executor.get_stats()["total_executions"] > 0

        executor.reset_stats()
        stats = executor.get_stats()

        assert stats["total_executions"] == 0
        assert stats["parallel_executions"] == 0
        assert stats["sequential_executions"] == 0
        assert stats["total_duration_ms"] == 0.0

    @pytest.mark.asyncio
    async def test_build_dependency_graph_with_file_writes(self, executor):
        """Test dependency graph building with file write ordering."""

        specs = [
            ToolCallSpec(
                name="write_file",
                arguments={"file_path": "/path/to/file.py"},
                call_id="write-1",
                category=ToolCategory.WRITE,
            ),
            ToolCallSpec(
                name="write_file",
                arguments={"file_path": "/path/to/file.py"},
                call_id="write-2",
                category=ToolCategory.WRITE,
            ),
            ToolCallSpec(
                name="read_file",
                arguments={"file_path": "/path/to/file.py"},
                call_id="read-1",
                category=ToolCategory.READ_ONLY,
            ),
        ]

        graph = executor._build_dependency_graph(specs)

        # Second write should depend on first write
        assert "write-1" in graph["write-2"]
        # Read should depend on both writes
        assert "write-1" in graph["read-1"]
        assert "write-2" in graph["read-1"]

    @pytest.mark.asyncio
    async def test_get_execution_batches_topological_sort(self, executor):
        """Test topological sort for execution batches."""

        specs = [
            ToolCallSpec(
                name="tool_a",
                arguments={},
                call_id="a",
                category=ToolCategory.READ_ONLY,
                dependencies=set(),
            ),
            ToolCallSpec(
                name="tool_b",
                arguments={},
                call_id="b",
                category=ToolCategory.READ_ONLY,
                dependencies={"a"},
            ),
            ToolCallSpec(
                name="tool_c",
                arguments={},
                call_id="c",
                category=ToolCategory.READ_ONLY,
                dependencies={"a"},
            ),
            ToolCallSpec(
                name="tool_d",
                arguments={},
                call_id="d",
                category=ToolCategory.READ_ONLY,
                dependencies={"b", "c"},
            ),
        ]

        batches = executor._get_execution_batches(executor._build_dependency_graph(specs), specs)

        # Should have 3 batches: [a], [b, c], [d]
        assert len(batches) == 3

        # First batch: a
        assert len(batches[0]) == 1
        assert batches[0][0].call_id == "a"

        # Second batch: b and c (can execute in parallel)
        assert len(batches[1]) == 2
        batch_1_ids = {c.call_id for c in batches[1]}
        assert batch_1_ids == {"b", "c"}

        # Third batch: d
        assert len(batches[2]) == 1
        assert batches[2][0].call_id == "d"

    @pytest.mark.asyncio
    async def test_config_disable_write_parallelization(self):
        """Test disabling write parallelization."""

        executor = AsyncToolExecutor(
            config=ExecutionConfig(
                enable_write_parallelization=False,  # Disable
            )
        )

        write_count = [0]

        async def mock_executor(call: ToolCallSpec) -> str:
            write_count[0] += 1
            await asyncio.sleep(0.05)
            return f"Result for {call.name}"

        specs = [
            ToolCallSpec(
                name="write_file",
                arguments={"file_path": f"/path/to/file{i}.py"},
                call_id=f"write-{i}",
                category=ToolCategory.WRITE,
            )
            for i in range(3)
        ]

        results = await executor.execute_tool_calls(specs, mock_executor)

        assert len(results) == 3
        assert all(r.success for r in results)
        # With write parallelization disabled, should still execute sequentially
        # due to file dependencies
        assert write_count[0] == 3

    @pytest.mark.asyncio
    async def test_config_disable_priority_scheduling(self):
        """Test disabling priority-based scheduling."""

        executor = AsyncToolExecutor(
            config=ExecutionConfig(
                enable_priority_scheduling=False,  # Disable
            )
        )

        execution_order = []
        lock = asyncio.Lock()

        async def mock_executor(call: ToolCallSpec) -> str:
            async with lock:
                execution_order.append(call.call_id)
            await asyncio.sleep(0.05)
            return f"Result for {call.name}"

        specs = [
            ToolCallSpec(
                name="low",
                arguments={},
                call_id="low",
                category=ToolCategory.READ_ONLY,
                priority=1,
            ),
            ToolCallSpec(
                name="high",
                arguments={},
                call_id="high",
                category=ToolCategory.READ_ONLY,
                priority=10,
            ),
        ]

        await executor.execute_tool_calls(specs, mock_executor)

        # Without priority scheduling, order may vary
        # but both should execute
        assert len(execution_order) == 2
        assert set(execution_order) == {"low", "high"}

    @pytest.mark.asyncio
    async def test_embedding_intensive_concurrency_limit(self):
        """Test that embedding-intensive tools have lower concurrency limit."""

        executor = AsyncToolExecutor(
            config=ExecutionConfig(
                max_concurrent=10,  # General limit
                max_embedding_concurrent=2,  # Lower limit for embedding tools (safer)
                embedding_intensive_tools={"code_search"},
            )
        )

        active_count = 0
        max_active = 0
        lock = asyncio.Lock()

        async def mock_executor(call: ToolCallSpec) -> str:
            nonlocal active_count, max_active

            async with lock:
                active_count += 1
                max_active = max(max_active, active_count)

            await asyncio.sleep(0.1)

            async with lock:
                active_count -= 1

            return f"Result for {call.name}"

        # Create 8 code_search tool calls (embedding-intensive)
        specs = [
            ToolCallSpec(
                name="code_search",
                arguments={"query": f"test query {i}"},
                call_id=f"search-{i}",
                category=ToolCategory.READ_ONLY,
            )
            for i in range(8)
        ]

        results = await executor.execute_tool_calls(specs, mock_executor)

        assert len(results) == 8
        assert all(r.success for r in results)
        # Should not exceed max_embedding_concurrent of 2
        assert max_active <= 2

        # Verify stats tracked correctly
        stats = executor.get_stats()
        assert stats["embedding_intensive_executions"] == 8

    @pytest.mark.asyncio
    async def test_embedding_intensive_mixed_with_regular_tools(self):
        """Test mixing embedding-intensive and regular tools."""

        executor = AsyncToolExecutor(
            config=ExecutionConfig(
                max_concurrent=10,
                max_embedding_concurrent=2,  # Very low limit for test
                embedding_intensive_tools={"code_search"},
            )
        )

        active_count = 0
        max_active = 0
        lock = asyncio.Lock()

        async def mock_executor(call: ToolCallSpec) -> str:
            nonlocal active_count, max_active

            async with lock:
                active_count += 1
                max_active = max(max_active, active_count)

            await asyncio.sleep(0.1)

            async with lock:
                active_count -= 1

            return f"Result for {call.name}"

        # Mix of embedding-intensive and regular tools
        specs = [
            # 4 code_search (embedding-intensive, max 2 concurrent)
            ToolCallSpec(
                name="code_search",
                arguments={"query": f"search {i}"},
                call_id=f"search-{i}",
                category=ToolCategory.READ_ONLY,
            )
            for i in range(4)
        ] + [
            # 4 regular read tools (max 10 concurrent)
            ToolCallSpec(
                name="read_file",
                arguments={"file_path": f"/path/to/file{i}.py"},
                call_id=f"read-{i}",
                category=ToolCategory.READ_ONLY,
            )
            for i in range(4)
        ]

        results = await executor.execute_tool_calls(specs, mock_executor)

        assert len(results) == 8
        assert all(r.success for r in results)
        # Overall concurrency should be limited by general semaphore
        # but code_search should be further limited
        assert max_active <= 10

        # Verify stats
        stats = executor.get_stats()
        assert stats["embedding_intensive_executions"] == 4
        assert stats["total_executions"] == 8
