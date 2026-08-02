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

"""High-level harness integration using framework APIs.

This module provides the integration layer between the evaluation harness
and the high-level BenchmarkAgent. It replaces the low-level agent_callback
pattern with a framework-aligned approach.

Usage:
    from victor.benchmark.harness_integration import (
        create_agent_callback,
        HighLevelEvaluationRunner,
    )

    # Create callback for existing harness
    callback = await create_agent_callback(
        provider="anthropic",
        profile="benchmark",
    )

    # Or use the high-level runner
    runner = HighLevelEvaluationRunner(
        provider="anthropic",
        config=evaluation_config,
    )
    results = await runner.run_all(tasks)
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from victor.benchmark.agent import BenchmarkAgent, BenchmarkAgentConfig, ExecutionTrace
from victor.evaluation.protocol import (
    BenchmarkTask,
    EvaluationConfig,
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger(__name__)


async def create_agent_callback(
    provider: str = "anthropic",
    model: Optional[str] = None,
    profile: Optional[str] = None,
    timeout: float = 300.0,
    workspace_manager: Optional[Any] = None,
) -> Callable[[BenchmarkTask], Any]:
    """Create an agent callback function for the evaluation harness.

    This creates a callback compatible with EvaluationHarness.run_evaluation()
    but using the high-level BenchmarkAgent internally.

    Args:
        provider: LLM provider name
        model: Optional model override
        profile: Optional Victor profile
        timeout: Timeout per task in seconds
        workspace_manager: Optional SWEBenchWorkspaceManager for repo setup

    Returns:
        Async callback function: (BenchmarkTask) -> Dict[str, Any]
    """
    config = BenchmarkAgentConfig(
        provider=provider,
        model=model,
        timeout_per_task=timeout,
    )

    # Create agent once (reused across tasks)
    if profile:
        agent = await BenchmarkAgent.from_profile(profile, model, config)
    else:
        agent = await BenchmarkAgent.create(provider, model, config)

    async def agent_callback(task: BenchmarkTask) -> Dict[str, Any]:
        """Execute a single benchmark task.

        Returns dict with code and metrics for harness compatibility.
        """
        # Setup workspace if manager provided
        workspace_path = None
        if workspace_manager:
            workspace_path = await _setup_workspace(task, workspace_manager)

        # Execute task
        trace = await agent.execute_task(task, workspace_path)

        # Reset for next task
        await agent.reset()

        # Return in format expected by harness
        return {
            "code": trace.generated_code,
            "tokens_input": trace.tokens_input,
            "tokens_output": trace.tokens_output,
            "tokens_used": trace.tokens_used,
            "cached_tokens": trace.cached_tokens,
            "reasoning_tokens": trace.reasoning_tokens,
            "cost_usd_micros": trace.cost_usd_micros,
            "tool_calls": len(trace.tool_calls),
            "turns": trace.turns,
            "task_report": trace.task_report,
            "cache_hit_rate": trace.cache_hit_rate,
            "tool_schema_tokens": trace.tool_schema_tokens,
            "compaction_saved_tokens": trace.compaction_saved_tokens,
            "compaction_messages_removed": trace.compaction_messages_removed,
            "metadata": trace.build_result_metadata(),
        }

    # Store agent reference for cleanup
    agent_callback._agent = agent
    agent_callback._partial_data = None  # For timeout handling

    return agent_callback


async def _setup_workspace(
    task: BenchmarkTask,
    workspace_manager: Any,
) -> Optional[Path]:
    """Setup workspace for a benchmark task.

    Handles SWE-bench style repo setup with caching.
    """
    try:
        cached_repo = workspace_manager.get_cached_repo_path(task)
        if cached_repo and workspace_manager.is_repo_indexed(task):
            work_dir = cached_repo
            logger.debug(f"Using indexed repo: {cached_repo.name}")
        else:
            logger.debug("Setting up repo on-the-fly")
            await workspace_manager.setup_repo_with_indexes(task)
            work_dir = workspace_manager.get_cached_repo_path(task)

        # Checkout specific commit if needed
        if task.base_commit and work_dir:
            checkout_proc = await asyncio.create_subprocess_exec(
                "git",
                "checkout",
                task.base_commit,
                cwd=work_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await checkout_proc.communicate()

        return work_dir

    except Exception as e:
        logger.warning(f"Workspace setup failed: {e}")
        return None


class HighLevelEvaluationRunner:
    """High-level evaluation runner using framework APIs.

    This class provides an alternative to using EvaluationHarness directly,
    offering a cleaner interface for running benchmark evaluations.

    Example:
        runner = HighLevelEvaluationRunner(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
        )

        async with runner:
            results = await runner.run_all(tasks)
            print(f"Pass rate: {runner.pass_rate:.1%}")
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        profile: Optional[str] = None,
        config: Optional[EvaluationConfig] = None,
    ):
        """Initialize the runner.

        Args:
            provider: LLM provider name
            model: Optional model override
            profile: Optional Victor profile
            config: Optional EvaluationConfig for advanced settings
        """
        self.provider = provider
        self.model = model
        self.profile = profile
        self.config = config
        self._agent: Optional[BenchmarkAgent] = None
        self._results: List[TaskResult] = []

    async def __aenter__(self) -> "HighLevelEvaluationRunner":
        """Context manager entry - create agent."""
        await self._ensure_agent()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup agent."""
        await self.close()

    async def _ensure_agent(self) -> BenchmarkAgent:
        """Ensure agent is created."""
        if not self._agent:
            agent_config = BenchmarkAgentConfig(
                provider=self.provider,
                model=self.model,
                timeout_per_task=self.config.timeout_per_task if self.config else 300.0,
                max_turns=self.config.max_turns if self.config else 15,
            )

            if self.profile:
                self._agent = await BenchmarkAgent.from_profile(
                    self.profile, self.model, agent_config
                )
            else:
                self._agent = await BenchmarkAgent.create(self.provider, self.model, agent_config)

        return self._agent

    async def run_task(
        self,
        task: BenchmarkTask,
        workspace_path: Optional[Path] = None,
    ) -> TaskResult:
        """Run a single benchmark task.

        Args:
            task: The benchmark task
            workspace_path: Optional workspace directory

        Returns:
            TaskResult with execution results
        """
        agent = await self._ensure_agent()

        trace = await agent.execute_task(task, workspace_path)

        # Convert trace to TaskResult
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.PASSED if trace.success else TaskStatus.FAILED,
            generated_code=trace.generated_code,
            tokens_used=trace.tokens_used,
            tokens_input=trace.tokens_input,
            tokens_output=trace.tokens_output,
            cached_tokens=trace.cached_tokens,
            reasoning_tokens=trace.reasoning_tokens,
            cost_usd_micros=trace.cost_usd_micros,
            tool_calls=len(trace.tool_calls),
            turns=trace.turns,
            duration_seconds=trace.duration_seconds,
            error_message=trace.error,
            metadata=trace.build_result_metadata(),
        )

        self._results.append(result)
        await agent.reset()

        return result

    async def run_all(
        self,
        tasks: List[BenchmarkTask],
        workspace_manager: Optional[Any] = None,
        progress_callback: Optional[Callable[[int, int, TaskResult], None]] = None,
    ) -> List[TaskResult]:
        """Run all benchmark tasks.

        Args:
            tasks: List of benchmark tasks
            workspace_manager: Optional workspace manager for repo setup
            progress_callback: Optional callback for progress updates

        Returns:
            List of TaskResult for all tasks
        """
        results = []

        for i, task in enumerate(tasks):
            logger.info(f"Running task {i + 1}/{len(tasks)}: {task.task_id}")

            # Setup workspace if needed
            workspace_path = None
            if workspace_manager:
                workspace_path = await _setup_workspace(task, workspace_manager)

            # Run task
            result = await self.run_task(task, workspace_path)
            results.append(result)

            # Progress callback
            if progress_callback:
                progress_callback(i, len(tasks), result)

        return results

    async def close(self) -> None:
        """Clean up resources."""
        if self._agent:
            await self._agent.close()
            self._agent = None

    @property
    def results(self) -> List[TaskResult]:
        """Get all collected results."""
        return self._results

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate from results."""
        if not self._results:
            return 0.0
        passed = sum(1 for r in self._results if r.status == TaskStatus.PASSED)
        return passed / len(self._results)

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens used."""
        return sum(r.tokens_used for r in self._results)

    @property
    def total_tool_calls(self) -> int:
        """Calculate total tool calls."""
        return sum(r.tool_calls for r in self._results)

    def get_metrics(self) -> Dict[str, Any]:
        """Get summary metrics for all results."""
        if not self._results:
            return {}

        passed = sum(1 for r in self._results if r.status == TaskStatus.PASSED)
        failed = sum(1 for r in self._results if r.status == TaskStatus.FAILED)
        errors = sum(1 for r in self._results if r.status == TaskStatus.ERROR)
        timeouts = sum(1 for r in self._results if r.status == TaskStatus.TIMEOUT)

        return {
            "total_tasks": len(self._results),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "timeouts": timeouts,
            "pass_rate": self.pass_rate,
            "total_tokens": self.total_tokens,
            "total_tool_calls": self.total_tool_calls,
            "avg_tokens_per_task": self.total_tokens / len(self._results),
            "avg_tool_calls_per_task": self.total_tool_calls / len(self._results),
        }
