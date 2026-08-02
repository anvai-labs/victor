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

"""Parallel tool execution for improved performance.

This module provides parallel execution of independent tool calls
while respecting dependencies and resource constraints.

Design Principles:
- Reuses existing ToolExecutor infrastructure
- Respects tool dependencies (write ops wait for reads)
- Configurable concurrency limits
- Progress callbacks for streaming updates
- Graceful error handling with partial results
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from victor.agent.action_authorizer import (
    build_write_tool_set,
    normalize_tool_name_for_policy,
)
from victor.agent.tool_executor import ToolExecutionResult, ToolExecutor


def _get_embedding_config():
    """Get embedding-intensive tool configuration from settings.

    Returns:
        Tuple of (max_embedding_concurrent, embedding_intensive_tools)
    """
    try:
        from victor.config.tool_settings import get_tool_settings

        settings = get_tool_settings()
        return (
            settings.max_embedding_concurrent,
            settings.embedding_intensive_tools,
        )
    except Exception:
        # Fallback if settings not available
        return 2, {"code_search"}


logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    READ_ONLY = "read_only"
    WRITE = "write"
    COMPUTE = "compute"
    NETWORK = "network"


# Optimized tool categorization using frozenset for O(1) lookups
READ_TOOLS = frozenset(
    {
        "read",
        "ls",
        "code_search",
        "semantic_code_search",
        "grep_search",
        "plan",
        "git",
    }
)
WRITE_TOOLS = build_write_tool_set(
    "docker",
    "create_file",
    "delete_file",
    "rename_file",
    "notebook_edit",
)
NETWORK_TOOLS = frozenset({"web_search", "web_fetch", "http"})

# TOOL_CATEGORIES dictionary for backward compatibility with tests
TOOL_CATEGORIES: Dict[str, ToolCategory] = {
    **dict.fromkeys(READ_TOOLS, ToolCategory.READ_ONLY),
    **dict.fromkeys(WRITE_TOOLS, ToolCategory.WRITE),
    **dict.fromkeys(NETWORK_TOOLS, ToolCategory.NETWORK),
}


def _canonical_tool_name(tool_name: str) -> str:
    """Normalize aliases to the canonical runtime tool name."""
    return normalize_tool_name_for_policy(tool_name)


def _extract_paths_from_arguments(tool_name: str, arguments: Dict[str, Any]) -> List[str]:
    """Extract file paths from canonical tool arguments for dependency tracking."""
    paths: List[str] = []
    direct_path = arguments.get("path") or arguments.get("file_path")
    if isinstance(direct_path, str) and direct_path:
        paths.append(direct_path)

    if tool_name == "edit":
        ops = arguments.get("ops") or arguments.get("edits") or arguments.get("files") or []
        if isinstance(ops, list):
            for op in ops:
                if not isinstance(op, dict):
                    continue
                path = op.get("path")
                new_path = op.get("new_path")
                if isinstance(path, str) and path:
                    paths.append(path)
                if isinstance(new_path, str) and new_path:
                    paths.append(new_path)

    return list(dict.fromkeys(paths))


@dataclass
class ParallelExecutionConfig:
    max_concurrent: int = 5
    enable_parallel: bool = True
    parallelize_reads: bool = True
    timeout_per_tool: float = 60.0
    # Embedding-intensive tool concurrency limits
    # Loaded from ToolSettings to support environment variable configuration
    max_embedding_concurrent: int = 2
    embedding_intensive_tools: Set[str] = field(default_factory=lambda: {"code_search"})


@dataclass
class ParallelExecutionResult:
    results: List[ToolExecutionResult] = field(default_factory=list)
    total_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    completed_count: int = 0
    failed_count: int = 0
    parallel_speedup: float = 1.0  # Speedup ratio vs sequential execution


class ParallelToolExecutor:
    def __init__(
        self,
        tool_executor: ToolExecutor,
        config: Optional[ParallelExecutionConfig] = None,
        progress_callback: Optional[Callable[[str, str, bool], None]] = None,
    ):
        self.executor = tool_executor
        # Load config with embedding settings from ToolSettings if not provided
        if config is None:
            max_emb, emb_tools = _get_embedding_config()
            self.config = ParallelExecutionConfig(
                max_embedding_concurrent=max_emb,
                embedding_intensive_tools=emb_tools,
            )
        else:
            self.config = config
        self.progress_callback = progress_callback

    def _get_category(self, tool_name: str) -> ToolCategory:
        canonical_tool_name = _canonical_tool_name(tool_name)
        if canonical_tool_name in READ_TOOLS:
            return ToolCategory.READ_ONLY
        elif canonical_tool_name in WRITE_TOOLS:
            return ToolCategory.WRITE
        elif canonical_tool_name in NETWORK_TOOLS:
            return ToolCategory.NETWORK
        return ToolCategory.COMPUTE

    def _has_write_tools(self, tool_calls: List[Dict[str, Any]]) -> bool:
        return any(_canonical_tool_name(tc.get("name", "")) in WRITE_TOOLS for tc in tool_calls)

    def _can_parallelize(self, tool_calls: List[Dict[str, Any]]) -> bool:
        """Check if the given tool calls can be parallelized.

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            True if calls can be parallelized, False otherwise
        """
        # Cannot parallelize if disabled
        if not self.config.enable_parallel:
            return False
        # Cannot parallelize single or empty calls
        if len(tool_calls) <= 1:
            return False
        # Writes to different files CAN parallelize via dependency graph.
        # Only serialize when multiple writes target the SAME file or when
        # a write has no extractable file path (unknown target).
        return True

    def _extract_file_dependencies(self, tool_calls: List[Dict[str, Any]]) -> Dict[int, Set[int]]:
        """Build a per-file dependency graph for tool calls.

        Dependency rules:
        - A read on path P depends on all prior writes to P
        - A write on path P depends on all prior writes to P (serialization)
        - A write with no extractable path depends on ALL prior writes
          (conservative: unknown target could conflict with anything)
        """
        dependencies: Dict[int, Set[int]] = {i: set() for i in range(len(tool_calls))}
        path_writers: Dict[str, List[int]] = {}  # path -> [writer indices]
        unresolved_writers: List[int] = []  # writes with no extractable path

        for i, tc in enumerate(tool_calls):
            name = tc.get("name", "")
            args = tc.get("arguments", {})
            if isinstance(args, str):
                import json as _json

                try:
                    args = _json.loads(args)
                except Exception:
                    args = {}
            canonical_name = _canonical_tool_name(name)
            paths = _extract_paths_from_arguments(canonical_name, args)

            if canonical_name in WRITE_TOOLS:
                if paths:
                    # Write→write on same file: serialize
                    for path in paths:
                        if path in path_writers:
                            dependencies[i].update(path_writers[path])
                    dependencies[i].update(unresolved_writers)
                    for path in paths:
                        path_writers.setdefault(path, []).append(i)
                else:
                    # Unknown write target: depends on ALL prior writes
                    for prev_writers in path_writers.values():
                        dependencies[i].update(prev_writers)
                    dependencies[i].update(unresolved_writers)
                    unresolved_writers.append(i)
                # All unresolved writes also depend on this one
                # (handled by future iterations looking at unresolved_writers)
            elif canonical_name in READ_TOOLS:
                # Read depends on prior writes to same path
                for path in paths:
                    if path in path_writers:
                        dependencies[i].update(path_writers[path])
                # Read also depends on unresolved writes (conservative)
                dependencies[i].update(unresolved_writers)

        return dependencies

    async def execute_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> ParallelExecutionResult:
        start_time = time.time()
        result = ParallelExecutionResult()

        if not tool_calls:
            return result

        context = context or {}

        # Sequential fallback only when parallelism is disabled or single call
        if not self.config.enable_parallel or len(tool_calls) <= 1:
            return await self._execute_sequential(tool_calls, context, result, start_time)

        # Parallel execution with dependency handling
        dependencies = self._extract_file_dependencies(tool_calls)
        pending = set(range(len(tool_calls)))
        completed = set()
        results_by_index = {}

        while pending:
            ready = [i for i in pending if not (dependencies[i] - completed)]

            if not ready:
                # Handle deadlock
                for i in pending:
                    results_by_index[i] = self._create_error_result(
                        tool_calls[i].get("name", "unknown"), "Dependency deadlock"
                    )
                    result.failed_count += 1
                break

            # Separate embedding-intensive tools from regular tools
            embedding_ready = []
            regular_ready = []
            for i in ready:
                tool_name = tool_calls[i].get("name", "")
                if tool_name in self.config.embedding_intensive_tools:
                    embedding_ready.append(i)
                else:
                    regular_ready.append(i)

            # Build batch respecting both concurrency limits
            # Count currently running embedding-intensive and regular tools
            running_embedding = sum(
                1
                for i in pending
                if i not in ready
                and tool_calls[i].get("name", "") in self.config.embedding_intensive_tools
            )
            running_regular = sum(
                1
                for i in pending
                if i not in ready
                and tool_calls[i].get("name", "") not in self.config.embedding_intensive_tools
            )

            # Calculate how many of each type we can add
            available_embedding_slots = max(
                0, self.config.max_embedding_concurrent - running_embedding
            )
            available_regular_slots = max(
                0, self.config.max_concurrent - running_embedding - running_regular
            )

            # Select tools for batch
            batch = []
            batch.extend(embedding_ready[:available_embedding_slots])
            batch.extend(regular_ready[: available_regular_slots - len(batch)])

            # If batch is empty but we have ready tasks, we're at capacity
            if not batch and ready:
                # Wait for current tasks to complete before adding more
                await asyncio.sleep(0.01)
                continue

            tasks = [self._execute_single(tool_calls[i], context) for i in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for idx, res in zip(batch, batch_results, strict=False):
                if isinstance(res, Exception):
                    exec_result = self._create_error_result(
                        tool_calls[idx].get("name", "unknown"), str(res)
                    )
                    result.failed_count += 1
                else:
                    exec_result = res
                    if exec_result.success:
                        result.completed_count += 1
                    else:
                        result.failed_count += 1
                        if exec_result.error:
                            result.errors.append(exec_result.error)

                results_by_index[idx] = exec_result
                completed.add(idx)
                pending.discard(idx)

        # Collect results in order
        result.results = [
            results_by_index[i] for i in range(len(tool_calls)) if i in results_by_index
        ]
        result.total_time = time.time() - start_time

        return result

    async def _execute_sequential(
        self,
        tool_calls: List[Dict[str, Any]],
        context: Dict[str, Any],
        result: ParallelExecutionResult,
        start_time: float,
    ) -> ParallelExecutionResult:
        for tc in tool_calls:
            exec_result = await self._execute_single(tc, context)
            result.results.append(exec_result)
            if exec_result.success:
                result.completed_count += 1
            else:
                result.failed_count += 1
                if exec_result.error:
                    result.errors.append(exec_result.error)

        result.total_time = time.time() - start_time
        return result

    def _create_error_result(self, tool_name: str, error: str) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_name=tool_name,
            success=False,
            result=None,
            error=error,
        )

    async def _execute_single(
        self, tool_call: Dict[str, Any], context: Dict[str, Any]
    ) -> ToolExecutionResult:
        tool_name = tool_call.get("name", "unknown")
        arguments = tool_call.get("arguments", {})

        if self.progress_callback:
            self.progress_callback(tool_name, "started", True)

        try:
            result = await asyncio.wait_for(
                self.executor.execute(tool_name=tool_name, arguments=arguments, context=context),
                timeout=self.config.timeout_per_tool,
            )

            if self.progress_callback:
                self.progress_callback(tool_name, "completed", result.success)

            return result

        except asyncio.TimeoutError:
            error_msg = f"Tool '{tool_name}' timed out after {self.config.timeout_per_tool}s"
            if self.progress_callback:
                self.progress_callback(tool_name, "timeout", False)
            return self._create_error_result(tool_name, error_msg)

        except Exception as e:
            if self.progress_callback:
                self.progress_callback(tool_name, "error", False)
            return self._create_error_result(tool_name, str(e))


def create_parallel_executor(
    tool_executor: ToolExecutor,
    max_concurrent: int = 5,
    enable: bool = True,
    progress_callback: Optional[Callable[[str, str, bool], None]] = None,
) -> ParallelToolExecutor:
    # Load embedding settings from ToolSettings
    max_emb, emb_tools = _get_embedding_config()
    config = ParallelExecutionConfig(
        max_concurrent=max_concurrent,
        enable_parallel=enable,
        max_embedding_concurrent=max_emb,
        embedding_intensive_tools=emb_tools,
    )
    return ParallelToolExecutor(tool_executor, config, progress_callback)
