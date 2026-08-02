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

"""High-level BenchmarkAgent using framework APIs.

This module provides a high-level agent for benchmark evaluation that:
- Uses Agent.create() instead of raw orchestrator
- Leverages framework observability for metrics collection
- Integrates with the BenchmarkVertical for domain-specific configuration
- Provides a clean API for benchmark harness integration

This replaces the low-level VictorAgentAdapter with a framework-aligned approach.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from victor.benchmark.assistant import BenchmarkVertical
from victor.benchmark.task_bridge import (
    benchmark_task_to_framework_task,
    build_benchmark_prompt,
)
from victor.evaluation.team_feedback import summarize_team_feedback
from victor.evaluation.protocol import BenchmarkTask

logger = logging.getLogger(__name__)

_EDIT_TOOL_NAMES = {
    "edit",
    "write",
    "patch",
    "file_editor",
    "todo_write",
    "scaffold",
    "refactor",
}


def _coalesce_value(*values: Any) -> Any:
    """Return the first value that is not ``None``."""
    for value in values:
        if value is not None:
            return value
    return None


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce a value to int without raising."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce a value to float without raising."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class BenchmarkAgentConfig:
    """Configuration for BenchmarkAgent."""

    provider: str = "anthropic"
    model: Optional[str] = None
    timeout_per_task: float = 300.0
    timeout_per_turn: float = 60.0
    max_turns: int = 15
    tool_budget: int = 30
    temperature: float = 0.2
    enable_thinking: bool = True
    enable_observability: bool = True
    # Workflow execution options
    use_workflow: bool = False  # If True, use YAML workflow instead of direct Agent.run()
    workflow_name: Optional[str] = None  # Workflow to use (auto-detected if None)


@dataclass
class ExecutionTrace:
    """Trace of benchmark task execution.

    Collects metrics during execution for reporting.
    """

    task_id: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    # Token usage
    tokens_input: int = 0
    tokens_output: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd_micros: int = 0
    cache_hit_rate: float = 0.0
    tool_schema_tokens: int = 0
    compaction_saved_tokens: int = 0
    compaction_messages_removed: int = 0

    # Execution metrics
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    time_to_first_tool_call_seconds: Optional[float] = None
    time_to_first_edit_seconds: Optional[float] = None
    first_edit_tool_name: Optional[str] = None

    # Results
    generated_code: str = ""
    generated_patch: Optional[str] = None
    error: Optional[str] = None
    success: bool = False
    task_report: Optional[Dict[str, Any]] = None
    team_feedback_summary: Optional[Dict[str, Any]] = None

    @property
    def tokens_used(self) -> int:
        return self.tokens_input + self.tokens_output

    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tokens_used": self.tokens_used,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd_micros": self.cost_usd_micros,
            "cache_hit_rate": self.cache_hit_rate,
            "tool_schema_tokens": self.tool_schema_tokens,
            "compaction_saved_tokens": self.compaction_saved_tokens,
            "compaction_messages_removed": self.compaction_messages_removed,
            "tool_calls": len(self.tool_calls),
            "turns": self.turns,
            "time_to_first_tool_call_seconds": self.time_to_first_tool_call_seconds,
            "time_to_first_edit_seconds": self.time_to_first_edit_seconds,
            "first_edit_tool_name": self.first_edit_tool_name,
            "generated_code": self.generated_code[:500] if self.generated_code else "",
            "error": self.error,
            "success": self.success,
            "task_report": dict(self.task_report) if self.task_report else None,
            "team_feedback_summary": (
                dict(self.team_feedback_summary) if self.team_feedback_summary else None
            ),
        }

    def build_result_metadata(self) -> Dict[str, Any]:
        """Build evaluation-harness metadata from the benchmark trace."""
        metadata: Dict[str, Any] = {}
        if self.task_report:
            metadata["task_report"] = dict(self.task_report)
        if self.team_feedback_summary:
            metadata["team_feedback_summary"] = dict(self.team_feedback_summary)
        if self.cache_hit_rate:
            metadata["cache_hit_rate"] = self.cache_hit_rate
        if self.tool_schema_tokens:
            metadata["tool_schema_tokens"] = self.tool_schema_tokens
        if self.compaction_saved_tokens:
            metadata["compaction_saved_tokens"] = self.compaction_saved_tokens
        if self.compaction_messages_removed:
            metadata["compaction_messages_removed"] = self.compaction_messages_removed
        if self.time_to_first_tool_call_seconds is not None:
            metadata["time_to_first_tool_call_seconds"] = self.time_to_first_tool_call_seconds
        if self.time_to_first_edit_seconds is not None:
            metadata["time_to_first_edit_seconds"] = self.time_to_first_edit_seconds
        if self.first_edit_tool_name:
            metadata["first_edit_tool_name"] = self.first_edit_tool_name
        return metadata


class BenchmarkAgent:
    """High-level agent for benchmark task execution.

    Uses the framework's Agent API instead of raw orchestrator,
    providing proper integration with vertical system, observability,
    and resource management.

    Example:
        agent = await BenchmarkAgent.create(provider="anthropic")
        trace = await agent.execute_task(benchmark_task, workspace_path)
        print(f"Tokens used: {trace.tokens_used}")
        print(f"Tool calls: {len(trace.tool_calls)}")
    """

    def __init__(
        self,
        agent: Any,  # victor.framework.Agent
        config: BenchmarkAgentConfig,
    ):
        """Initialize with a framework Agent instance.

        Use BenchmarkAgent.create() instead of direct construction.
        """
        self._agent = agent
        self._config = config
        self._current_trace: Optional[ExecutionTrace] = None
        self._event_unsubscribe: Optional[Callable] = None

    @classmethod
    async def create(
        cls,
        provider: str = "anthropic",
        model: Optional[str] = None,
        config: Optional[BenchmarkAgentConfig] = None,
        profile: Optional[str] = None,
    ) -> "BenchmarkAgent":
        """Create a BenchmarkAgent using framework Agent.create().

        This is the recommended way to create a BenchmarkAgent.
        It properly initializes the framework Agent with BenchmarkVertical.

        Args:
            provider: LLM provider name
            model: Optional model override
            config: Optional configuration
            profile: Optional Victor profile name

        Returns:
            Configured BenchmarkAgent instance
        """
        from victor.framework import Agent

        config = config or BenchmarkAgentConfig(provider=provider, model=model)

        # Create framework agent with BenchmarkVertical
        agent = await Agent.create(
            provider=provider,
            model=model,
            vertical=BenchmarkVertical,
            temperature=config.temperature,
            thinking=config.enable_thinking,
            enable_observability=config.enable_observability,
            profile=profile,
        )

        return cls(agent, config)

    @classmethod
    async def from_profile(
        cls,
        profile: str = "default",
        model_override: Optional[str] = None,
        config: Optional[BenchmarkAgentConfig] = None,
    ) -> "BenchmarkAgent":
        """Create BenchmarkAgent from a Victor profile.

        Args:
            profile: Profile name from ~/.victor/config.yaml
            model_override: Optional model to override profile setting
            config: Optional configuration

        Returns:
            Configured BenchmarkAgent instance
        """
        from victor.framework import Agent

        config = config or BenchmarkAgentConfig()

        # Create from profile - Agent.create handles profile loading
        agent = await Agent.create(
            profile=profile,
            model=model_override,
            vertical=BenchmarkVertical,
            temperature=config.temperature,
            thinking=config.enable_thinking,
            enable_observability=config.enable_observability,
        )

        return cls(agent, config)

    async def execute_task(
        self,
        task: BenchmarkTask,
        workspace_path: Optional[Path] = None,
        use_workflow: Optional[bool] = None,
        workflow_name: Optional[str] = None,
    ) -> ExecutionTrace:
        """Execute a benchmark task and return execution trace.

        This is the main entry point for benchmark evaluation.
        It converts the BenchmarkTask to a framework Task and executes it
        using either Agent.run() or a YAML workflow.

        Args:
            task: The benchmark task to execute
            workspace_path: Optional workspace directory
            use_workflow: Override config.use_workflow for this task
            workflow_name: Override config.workflow_name for this task

        Returns:
            ExecutionTrace with metrics and generated code
        """
        # Determine execution mode
        should_use_workflow = (
            use_workflow if use_workflow is not None else self._config.use_workflow
        )
        wf_name = workflow_name or self._config.workflow_name

        if should_use_workflow:
            return await self._execute_with_workflow(task, workspace_path, wf_name)
        else:
            return await self._execute_direct(task, workspace_path)

    async def _execute_direct(
        self,
        task: BenchmarkTask,
        workspace_path: Optional[Path] = None,
    ) -> ExecutionTrace:
        """Execute task using direct Agent.run() call.

        This is the original execution path, providing maximum flexibility
        but not using the structured YAML workflow.
        """
        trace = ExecutionTrace(task_id=task.task_id, start_time=time.time())
        self._current_trace = trace

        try:
            # Subscribe to events for metrics collection
            self._subscribe_to_events(trace)

            # Set workspace if provided
            if workspace_path:
                orchestrator = self._agent.get_orchestrator()
                if hasattr(orchestrator, "set_workspace"):
                    orchestrator.set_workspace(workspace_path)

            # Convert benchmark task to framework task
            framework_task = benchmark_task_to_framework_task(task)

            # Build enriched prompt
            prompt = build_benchmark_prompt(task, str(workspace_path) if workspace_path else None)

            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    self._agent.run(prompt, context=framework_task.context),
                    timeout=self._config.timeout_per_task,
                )

                # Extract results
                trace.generated_code = result.content or ""
                trace.success = result.success
                trace.error = result.error

                # Extract tool calls from result
                if result.tool_calls:
                    self._merge_result_tool_calls(trace, result.tool_calls)

                # Extract token usage from metadata
                if result.metadata:
                    self._apply_result_metadata_to_trace(trace, result.metadata)

            except asyncio.TimeoutError:
                trace.error = f"Task timed out after {self._config.timeout_per_task}s"
                trace.success = False
                logger.warning(f"Task {task.task_id} timed out")

        except Exception as e:
            trace.error = str(e)
            trace.success = False
            logger.exception(f"Error executing task {task.task_id}")

        finally:
            self._capture_orchestrator_task_report(trace)
            trace.end_time = time.time()
            self._unsubscribe_from_events()
            self._current_trace = None

        return trace

    async def _execute_with_workflow(
        self,
        task: BenchmarkTask,
        workspace_path: Optional[Path] = None,
        workflow_name: Optional[str] = None,
    ) -> ExecutionTrace:
        """Execute task using YAML workflow system.

        Uses BenchmarkWorkflowProvider to execute structured workflows
        with proper stage management, conditions, and transforms.

        Args:
            task: The benchmark task to execute
            workspace_path: Optional workspace directory
            workflow_name: Specific workflow to use (auto-detected if None)

        Returns:
            ExecutionTrace with metrics and generated code
        """
        from victor.benchmark.workflows import BenchmarkWorkflowProvider
        from victor.workflows.streaming import WorkflowEventType

        trace = ExecutionTrace(task_id=task.task_id, start_time=time.time())
        self._current_trace = trace

        try:
            # Subscribe to events for metrics collection
            self._subscribe_to_events(trace)

            # Set workspace if provided
            orchestrator = self._agent.get_orchestrator()
            if workspace_path and hasattr(orchestrator, "set_workspace"):
                orchestrator.set_workspace(workspace_path)

            # Get workflow provider
            provider = BenchmarkWorkflowProvider()

            # Auto-detect workflow if not specified
            if not workflow_name:
                workflow_name = self._detect_workflow_for_task(task, provider)

            logger.info(f"Executing task {task.task_id} with workflow: {workflow_name}")

            # Build workflow context from task
            context = self._build_workflow_context(task, workspace_path)

            # Execute workflow with streaming for progress tracking
            # Uses the new UnifiedWorkflowCompiler-based streaming API
            final_context: Dict[str, Any] = {}
            try:

                async def execute_workflow():
                    nonlocal final_context
                    # Use new unified compiler streaming API
                    async for node_id, state in provider.stream_compiled_workflow(
                        workflow_name, context
                    ):
                        # Each yield represents a completed node
                        logger.debug(f"Completed node: {node_id}")
                        trace.turns += 1
                        # Track tool calls from state if available
                        if "_tool_calls" in state:
                            for tool_call in state.get("_tool_calls", []):
                                self._record_tool_call(
                                    trace,
                                    tool_call.get("name", "unknown"),
                                )
                        # Keep latest state as final context
                        final_context = state

                await asyncio.wait_for(
                    execute_workflow(),
                    timeout=self._config.timeout_per_task,
                )

                # Extract results from workflow context
                trace.generated_code = final_context.get("solution_code", "")
                trace.generated_patch = final_context.get("generated_patch")
                trace.success = final_context.get("status") == "completed"

                # Extract metrics if available
                if "test_results" in final_context:
                    test_results = final_context["test_results"]
                    trace.success = test_results.get("pass_rate", 0) >= 0.95

            except asyncio.TimeoutError:
                trace.error = f"Workflow timed out after {self._config.timeout_per_task}s"
                trace.success = False
                logger.warning(f"Task {task.task_id} workflow timed out")

        except Exception as e:
            trace.error = str(e)
            trace.success = False
            logger.exception(f"Error executing workflow for task {task.task_id}")

        finally:
            self._capture_orchestrator_task_report(trace)
            trace.end_time = time.time()
            self._unsubscribe_from_events()
            self._current_trace = None

        return trace

    def _detect_workflow_for_task(
        self,
        task: BenchmarkTask,
        provider: Any,
    ) -> str:
        """Auto-detect appropriate workflow based on task characteristics.

        Args:
            task: The benchmark task
            provider: BenchmarkWorkflowProvider instance

        Returns:
            Workflow name to use
        """
        # Check task type hints
        task_type = getattr(task, "task_type", None) or getattr(task, "benchmark", None)

        if task_type:
            workflow = provider.get_workflow_for_task_type(str(task_type))
            if workflow:
                return workflow

        # Infer from task attributes
        if hasattr(task, "repo") and task.repo:
            # SWE-bench style task
            return "swe_bench"
        elif hasattr(task, "test_code") and task.test_code:
            # Code generation with tests
            return "code_generation"
        else:
            # Default to code generation
            return "code_generation"

    def _build_workflow_context(
        self,
        task: BenchmarkTask,
        workspace_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Build workflow context from benchmark task.

        Args:
            task: The benchmark task
            workspace_path: Optional workspace directory

        Returns:
            Context dict for workflow execution
        """
        context: Dict[str, Any] = {
            "task_id": task.task_id,
            "problem_id": task.task_id,
            "task_description": task.description or task.prompt,
            "prompt": task.prompt,
            "language": getattr(task, "language", "python"),
            "workspace_path": str(workspace_path) if workspace_path else None,
            # Test configuration
            "test_command": getattr(task, "test_command", "pytest"),
            "test_timeout": min(self._config.timeout_per_task / 2, 180),
            # Iteration limits
            "max_iterations": 5,
            "fix_iterations": 0,
        }

        # SWE-bench specific fields
        if hasattr(task, "repo") and task.repo:
            context["repo"] = task.repo
            context["base_commit"] = getattr(task, "base_commit", None)
            context["issue_text"] = getattr(task, "issue_text", task.prompt)
            context["issue_summary"] = task.description or task.prompt[:200]

        # Code generation specific fields
        if hasattr(task, "test_code") and task.test_code:
            context["test_cases"] = task.test_code
            context["context_code"] = getattr(task, "context_code", "")

        return context

    def _subscribe_to_events(self, trace: ExecutionTrace) -> None:
        """Subscribe to framework events for metrics collection."""
        try:
            from victor.framework import EventType

            def on_event(event: Any) -> None:
                """Handle framework events."""
                if not self._current_trace:
                    return

                event_type = getattr(event, "type", None)

                if event_type == EventType.TOOL_CALL:
                    self._record_tool_call(
                        self._current_trace,
                        getattr(event, "tool_name", "unknown"),
                    )
                elif event_type == EventType.CONTENT:
                    # Track turns based on content events
                    pass

            self._event_unsubscribe = self._agent.subscribe_to_events("*", on_event)

        except Exception as e:
            logger.debug(f"Could not subscribe to events: {e}")

    def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from framework events."""
        if self._event_unsubscribe:
            try:
                self._event_unsubscribe()
            except Exception:
                pass
            self._event_unsubscribe = None

    async def reset(self) -> None:
        """Reset agent state for next task."""
        await self._agent.reset()

    async def close(self) -> None:
        """Clean up agent resources."""
        self._unsubscribe_from_events()
        await self._agent.close()

    async def __aenter__(self) -> "BenchmarkAgent":
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        await self.close()

    @property
    def agent(self) -> Any:
        """Access the underlying framework Agent."""
        return self._agent

    @property
    def config(self) -> BenchmarkAgentConfig:
        """Get agent configuration."""
        return self._config

    def _apply_result_metadata_to_trace(
        self,
        trace: ExecutionTrace,
        metadata: Dict[str, Any],
    ) -> None:
        """Project framework task-result metadata into benchmark trace fields."""
        task_report = (
            metadata.get("task_report") if isinstance(metadata.get("task_report"), dict) else {}
        )
        usage = metadata.get("usage") if isinstance(metadata.get("usage"), dict) else {}
        prompt_details = (
            usage.get("prompt_tokens_details", {})
            if isinstance(usage.get("prompt_tokens_details", {}), dict)
            else {}
        )
        completion_details = (
            usage.get("completion_tokens_details", {})
            if isinstance(usage.get("completion_tokens_details", {}), dict)
            else {}
        )

        trace.tokens_input = _safe_int(
            _coalesce_value(
                metadata.get("tokens_input"),
                task_report.get("api_prompt_tokens") if task_report else None,
                usage.get("prompt_tokens"),
            ),
            default=trace.tokens_input,
        )
        trace.tokens_output = _safe_int(
            _coalesce_value(
                metadata.get("tokens_output"),
                task_report.get("api_completion_tokens") if task_report else None,
                usage.get("completion_tokens"),
            ),
            default=trace.tokens_output,
        )
        trace.turns = _safe_int(
            _coalesce_value(
                metadata.get("turns"),
                task_report.get("request_count") if task_report else None,
            ),
            default=trace.turns,
        )
        trace.cached_tokens = _safe_int(
            _coalesce_value(
                metadata.get("cached_tokens"),
                task_report.get("cache_read_tokens") if task_report else None,
                usage.get("cached_tokens"),
                prompt_details.get("cached_tokens"),
            ),
            default=trace.cached_tokens,
        )
        trace.reasoning_tokens = _safe_int(
            _coalesce_value(
                metadata.get("reasoning_tokens"),
                usage.get("reasoning_tokens"),
                completion_details.get("reasoning_tokens"),
            ),
            default=trace.reasoning_tokens,
        )
        trace.cost_usd_micros = _safe_int(
            _coalesce_value(
                metadata.get("cost_usd_micros"),
                (
                    round(_safe_float(task_report.get("total_cost_usd")) * 1_000_000)
                    if task_report and task_report.get("total_cost_usd") is not None
                    else None
                ),
                usage.get("cost_usd_micros"),
                usage.get("cost_in_usd_ticks"),
            ),
            default=trace.cost_usd_micros,
        )
        trace.cache_hit_rate = _safe_float(
            _coalesce_value(
                metadata.get("cache_hit_rate"),
                task_report.get("cache_hit_rate") if task_report else None,
            ),
            default=trace.cache_hit_rate,
        )
        trace.tool_schema_tokens = _safe_int(
            _coalesce_value(
                metadata.get("tool_schema_tokens"),
                task_report.get("tool_schema_tokens") if task_report else None,
            ),
            default=trace.tool_schema_tokens,
        )
        trace.compaction_saved_tokens = _safe_int(
            _coalesce_value(
                metadata.get("compaction_saved_tokens"),
                task_report.get("compaction_saved_tokens") if task_report else None,
            ),
            default=trace.compaction_saved_tokens,
        )
        trace.compaction_messages_removed = _safe_int(
            _coalesce_value(
                metadata.get("compaction_messages_removed"),
                task_report.get("compaction_messages_removed") if task_report else None,
            ),
            default=trace.compaction_messages_removed,
        )

        if task_report:
            trace.task_report = dict(task_report)
        team_feedback_summary = summarize_team_feedback({"metadata": metadata})
        if team_feedback_summary:
            trace.team_feedback_summary = dict(team_feedback_summary)

    def _record_tool_call(
        self,
        trace: ExecutionTrace,
        tool_name: Any,
        *,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a tool call and benchmark timing milestones."""
        event_time = time.time() if timestamp is None else float(timestamp)
        normalized_name = str(tool_name or "unknown")
        trace.tool_calls.append({"name": normalized_name, "timestamp": event_time})
        if trace.time_to_first_tool_call_seconds is None:
            trace.time_to_first_tool_call_seconds = max(0.0, event_time - trace.start_time)
        if (
            trace.time_to_first_edit_seconds is None
            and normalized_name.strip().lower() in _EDIT_TOOL_NAMES
        ):
            trace.time_to_first_edit_seconds = max(0.0, event_time - trace.start_time)
            trace.first_edit_tool_name = normalized_name

    def _merge_result_tool_calls(
        self,
        trace: ExecutionTrace,
        tool_calls: List[Dict[str, Any]],
    ) -> None:
        """Merge structured result tool calls without discarding event timestamps."""
        normalized_calls = [
            dict(tool_call) if isinstance(tool_call, dict) else {"name": str(tool_call)}
            for tool_call in list(tool_calls or [])
        ]
        if not trace.tool_calls:
            trace.tool_calls = normalized_calls
            return
        for index, tool_call in enumerate(normalized_calls):
            if index < len(trace.tool_calls):
                enriched = dict(trace.tool_calls[index])
                for key, value in tool_call.items():
                    enriched.setdefault(key, value)
                trace.tool_calls[index] = enriched
            else:
                trace.tool_calls.append(tool_call)

    def _capture_orchestrator_task_report(self, trace: ExecutionTrace) -> None:
        """Capture the canonical task report even when the framework result omitted it."""
        if trace.task_report is not None:
            return

        try:
            orchestrator = self._agent.get_orchestrator()
        except Exception:
            return

        getter = getattr(orchestrator, "get_last_task_report", None)
        if not callable(getter):
            return

        try:
            task_report = getter()
        except Exception:
            return

        if isinstance(task_report, dict):
            self._apply_result_metadata_to_trace(trace, {"task_report": task_report})


async def create_benchmark_agent(
    provider: str = "anthropic",
    model: Optional[str] = None,
    profile: Optional[str] = None,
    **kwargs,
) -> BenchmarkAgent:
    """Factory function for creating BenchmarkAgent.

    Convenience function that wraps BenchmarkAgent.create().

    Args:
        provider: LLM provider name
        model: Optional model name
        profile: Optional Victor profile
        **kwargs: Additional config options

    Returns:
        Configured BenchmarkAgent
    """
    config = BenchmarkAgentConfig(provider=provider, model=model, **kwargs)

    if profile:
        return await BenchmarkAgent.from_profile(profile, model, config)
    else:
        return await BenchmarkAgent.create(provider, model, config)
