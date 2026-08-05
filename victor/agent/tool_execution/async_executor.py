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

"""Async-first tool executor with dependency-aware parallelization.

This module provides an async-first execution model that:
- Automatically parallelizes read-only operations
- Supports write operations with file locking
- Implements dependency graph execution
- Provides priority-based scheduling

Design Patterns:
- Strategy Pattern: Pluggable execution strategies
- Repository Pattern: File lock management
- Command Pattern: Tool execution commands
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

from victor.agent.tool_execution.categorization import (
    ToolCategory,
    ToolCallSpec,
    ExecutionPriority,
    extract_files_from_args,
    categorize_tool_call,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AsyncToolExecutor",
    "ExecutionResult",
    "ExecutionConfig",
]


@dataclass
class ExecutionResult:
    """Result of async tool execution.

    Attributes:
        call_id: Tool call identifier
        success: Whether execution succeeded
        result: Tool return value
        error: Optional error message
        duration_ms: Execution time in milliseconds
        parallelizable: Whether this was executed in parallel
    """

    call_id: str
    success: bool
    result: Any
    error: Optional[str]
    duration_ms: float
    parallelizable: bool


@dataclass
class ExecutionConfig:
    """Configuration for async tool execution.

    Attributes:
        max_concurrent: Maximum concurrent tool executions
        enable_write_parallelization: Enable parallelization with file locking
        enable_priority_scheduling: Enable priority-based scheduling
        default_timeout: Default timeout per tool in seconds
        embedding_intensive_tools: Set of tool names that are embedding-intensive
        max_embedding_concurrent: Maximum concurrent embedding-intensive tool executions
    """

    max_concurrent: int = 10
    enable_write_parallelization: bool = True
    enable_priority_scheduling: bool = True
    # Increased from 30 to 60 seconds for semantic search operations
    default_timeout: float = 60.0
    # Tools that use embedding models and need lower concurrency limits
    embedding_intensive_tools: Set[str] = field(default_factory=lambda: {"code_search"})
    # Limit concurrent embedding operations to prevent resource exhaustion
    # Reduced from 4 to 2 for safer headroom (5+ causes failures)
    max_embedding_concurrent: int = 2


class AsyncToolExecutor:
    """
    Async-first tool executor with advanced parallelization.

    Features:
    - Automatic categorization of tool calls
    - Dependency-aware parallel execution
    - File locking for concurrent writes
    - Priority-based scheduling
    - Graceful degradation on errors

    Example:
        >>> executor = AsyncToolExecutor()
        >>>
        >>> # Execute tool calls with automatic parallelization
        >>> results = await executor.execute_tool_calls(tool_calls, executor_func)
        >>>
        >>> # Results include timing and parallelization info
        >>> for result in results:
        ...     print(f"{result.call_id}: {result.duration_ms}ms")
    """

    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        executor_func: Optional[callable] = None,
    ):
        """
        Initialize the async tool executor.

        Args:
            config: Execution configuration
            executor_func: Async function to execute individual tools
        """
        self.config = config or ExecutionConfig()
        self._executor_func = executor_func

        # Concurrency control
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

        # Embedding-intensive tool semaphore (lower limit to prevent resource exhaustion)
        self._embedding_semaphore = asyncio.Semaphore(self.config.max_embedding_concurrent)

        # File locking for concurrent writes
        self._file_locks: Dict[str, asyncio.Lock] = {}
        self._lock_lock = asyncio.Lock()

        # Execution statistics
        self._stats = {
            "total_executions": 0,
            "parallel_executions": 0,
            "sequential_executions": 0,
            "file_locks_acquired": 0,
            "embedding_intensive_executions": 0,
            "total_duration_ms": 0.0,
        }

    async def execute_tool_calls(
        self,
        tool_calls: List[ToolCallSpec],
        executor_func: callable,
    ) -> List[ExecutionResult]:
        """
        Execute multiple tool calls with automatic parallelization.

        Args:
            tool_calls: List of tool call specifications
            executor_func: Async function to execute individual tools

        Returns:
            List of execution results
        """
        start_time = time.time()

        # Build dependency graph
        graph = self._build_dependency_graph(tool_calls)

        # Execute in topological order with parallelization
        results = []
        for batch in self._get_execution_batches(graph, tool_calls):
            batch_results = await self._execute_batch(batch, executor_func)
            results.extend(batch_results)

        # Update statistics
        duration = (time.time() - start_time) * 1000
        self._stats["total_executions"] = len(tool_calls)
        self._stats["total_duration_ms"] = duration

        return results

    async def execute_tool_call(
        self,
        call: ToolCallSpec,
        executor_func: callable,
    ) -> ExecutionResult:
        """
        Execute a single tool call with locking if needed.

        Args:
            call: Tool call specification
            executor_func: Async function to execute the tool

        Returns:
            Execution result
        """
        start_time = time.time()

        # Check if this is an embedding-intensive tool
        is_embedding_intensive = call.name in self.config.embedding_intensive_tools
        if is_embedding_intensive:
            self._stats["embedding_intensive_executions"] += 1

        # Acquire semaphore for concurrency control
        # For embedding-intensive tools, also acquire the embedding semaphore
        if is_embedding_intensive:
            # Acquire both semaphores (nested context managers)
            async with self._embedding_semaphore, self._semaphore:
                return await self._execute_with_locking(call, executor_func, start_time)
        else:
            async with self._semaphore:
                return await self._execute_with_locking(call, executor_func, start_time)

    async def _execute_with_locking(
        self,
        call: ToolCallSpec,
        executor_func: callable,
        start_time: float,
    ) -> ExecutionResult:
        """Execute tool call with file locking if needed."""
        # Acquire file lock for write operations
        file_lock = None
        if call.category == ToolCategory.WRITE and self.config.enable_write_parallelization:
            files = extract_files_from_args(call.arguments)
            if files:
                file_lock = await self._get_file_lock(files[0])

        try:
            if file_lock:
                async with file_lock:
                    result = await asyncio.wait_for(
                        executor_func(call),
                        timeout=call.timeout,
                    )
            else:
                result = await asyncio.wait_for(
                    executor_func(call),
                    timeout=call.timeout,
                )

            duration = (time.time() - start_time) * 1000
            return ExecutionResult(
                call_id=call.call_id,
                success=True,
                result=result,
                error=None,
                duration_ms=duration,
                parallelizable=True,
            )

        except asyncio.TimeoutError:
            duration = (time.time() - start_time) * 1000
            return ExecutionResult(
                call_id=call.call_id,
                success=False,
                result=None,
                error=f"Timeout after {call.timeout}s",
                duration_ms=duration,
                parallelizable=True,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ExecutionResult(
                call_id=call.call_id,
                success=False,
                result=None,
                error=str(e),
                duration_ms=duration,
                parallelizable=True,
            )

    async def _execute_batch(
        self,
        batch: List[ToolCallSpec],
        executor_func: callable,
    ) -> List[ExecutionResult]:
        """Execute a batch of tool calls in parallel."""
        tasks = [self._execute_single_tool(call, executor_func) for call in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        execution_results = []
        for call, result in zip(batch, results):
            if isinstance(result, Exception):
                execution_results.append(
                    ExecutionResult(
                        call_id=call.call_id,
                        success=False,
                        result=None,
                        error=str(result),
                        duration_ms=0,
                        parallelizable=True,
                    )
                )
            elif isinstance(result, ExecutionResult):
                execution_results.append(result)

        return execution_results

    async def _execute_single_tool(
        self,
        call: ToolCallSpec,
        executor_func: callable,
    ) -> ExecutionResult:
        """Execute a single tool call."""
        # Check if this can be parallelized
        parallelizable = call.category in (
            ToolCategory.READ_ONLY,
            ToolCategory.NETWORK,
        )

        if not parallelizable:
            self._stats["sequential_executions"] += 1
        else:
            self._stats["parallel_executions"] += 1

        # Execute the tool
        return await self.execute_tool_call(call, executor_func)

    def _build_dependency_graph(
        self,
        tool_calls: List[ToolCallSpec],
    ) -> Dict[str, Set[str]]:
        """Build dependency graph from tool calls.

        Returns:
            Dictionary mapping call_id to set of dependency call_ids
        """
        graph = {call.call_id: set() for call in tool_calls}

        # Build file-based dependencies
        file_writers: Dict[str, List[str]] = {}
        file_readers: Dict[str, Set[str]] = {}

        for call in tool_calls:
            if call.category == ToolCategory.WRITE:
                files = extract_files_from_args(call.arguments)
                for file in files:
                    file_writers.setdefault(file, []).append(call.call_id)

            elif call.category == ToolCategory.READ_ONLY:
                files = extract_files_from_args(call.arguments)
                for file in files:
                    file_readers.setdefault(file, set()).add(call.call_id)

        # Add dependencies: reads wait for writes
        for file, readers in file_readers.items():
            if file in file_writers:
                for reader_id in readers:
                    for writer_id in file_writers[file]:
                        graph[reader_id].add(writer_id)

        # Add dependencies: later writes to same file wait for earlier writes
        for file, writer_ids in file_writers.items():
            for i in range(1, len(writer_ids)):
                # Later write depends on earlier write
                graph[writer_ids[i]].add(writer_ids[i - 1])

        # Add explicit dependencies from ToolCallSpec
        for call in tool_calls:
            graph[call.call_id].update(call.dependencies)

        return graph

    def _get_execution_batches(
        self,
        graph: Dict[str, Set[str]],
        tool_calls: List[ToolCallSpec],
    ) -> List[List[ToolCallSpec]]:
        """
        Get execution batches using topological sort.

        Returns:
            List of batches, where each batch can be executed in parallel
        """
        # Kahn's algorithm for topological sort
        # graph maps: call_id -> dependencies (what this call depends on)
        # We need: call_id -> dependents (what calls depend on this)
        in_degree = {call.call_id: 0 for call in tool_calls}
        call_map: Dict[str, ToolCallSpec] = {call.call_id: call for call in tool_calls}

        # Calculate initial in-degrees from dependency graph
        for call_id, deps in graph.items():
            in_degree[call_id] = len(deps)

        # Build reverse adjacency list: call_id -> dependents
        dependents: Dict[str, Set[str]] = {call.call_id: set() for call in tool_calls}
        for call_id, deps in graph.items():
            for dep_id in deps:
                dependents[dep_id].add(call_id)

        # Process in batches
        batches = []
        queue = [call_id for call_id, degree in in_degree.items() if degree == 0]

        while queue:
            # Sort by priority if enabled
            if self.config.enable_priority_scheduling:
                queue.sort(key=lambda cid: call_map[cid].priority, reverse=True)

            # Current batch can execute in parallel
            batch = [call_map[cid] for cid in queue]
            batches.append(batch)

            # Update in-degrees for next batch
            next_queue = []
            for call_id in queue:
                # For all nodes that depend on this node
                for dependent_id in dependents.get(call_id, set()):
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        next_queue.append(dependent_id)

            queue = next_queue

        return batches

    async def _get_file_lock(self, file_path: str) -> asyncio.Lock:
        """Get or create a file lock.

        Args:
            file_path: Path to lock

        Returns:
            Async lock for the file
        """
        async with self._lock_lock:
            if file_path not in self._file_locks:
                self._file_locks[file_path] = asyncio.Lock()
                self._stats["file_locks_acquired"] += 1

            return self._file_locks[file_path]

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            **self._stats,
            "active_locks": len(self._file_locks),
            "parallel_ratio": (
                self._stats["parallel_executions"] / self._stats["total_executions"]
                if self._stats["total_executions"] > 0
                else 0.0
            ),
        }

    def reset_stats(self) -> None:
        """Reset statistics (useful for testing)."""
        self._stats = {
            "total_executions": 0,
            "parallel_executions": 0,
            "sequential_executions": 0,
            "file_locks_acquired": 0,
            "embedding_intensive_executions": 0,
            "total_duration_ms": 0.0,
        }
