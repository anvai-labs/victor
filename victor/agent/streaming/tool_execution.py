# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Tool execution handler for streaming chat.

This module provides the ToolExecutionHandler class which encapsulates
the tool execution logic that used to live in ``ChatCoordinator._stream_chat_impl``.
The canonical orchestration now happens via ``StreamingChatExecutor``.

The handler manages:
- Budget warnings and exhaustion
- Progress checking and force completion
- Tool call filtering and truncation
- Tool execution and result generation
- Reminder injection

Design Pattern: Command + Facade
================================
The handler acts as a facade over the recovery runtime, chunk generator,
and tool executor, providing a single entry point for tool execution.

Usage:
    handler = ToolExecutionHandler(
        recovery_runtime=self._recovery_service or self._recovery_coordinator,
        chunk_generator=self._chunk_generator,
        ...
    )

    result = await handler.execute_tools(
        stream_ctx=stream_ctx,
        tool_calls=tool_calls,
        user_message=user_message,
        full_content=full_content,
    )

    for chunk in result.chunks:
        yield chunk

    if result.should_return:
        return
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
)

from victor.agent.conversation.history_metadata import build_internal_history_metadata
from victor.agent.services.protocols.streaming_runtime import (
    StreamingChunkRuntimeProtocol,
    StreamingMessageAdderProtocol,
    StreamingReminderRuntimeProtocol,
    StreamingTrackerRuntimeProtocol,
    ToolExecutionRecoveryRuntimeProtocol,
)
from victor.agent.streaming.context import StreamingChatContext
from victor.providers.base import StreamChunk
from victor.tools.core_tool_aliases import canonicalize_core_tool_name
from victor.tools.decorators import resolve_tool_name

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)

TERMINAL_SKIP_OUTCOME_KINDS = frozenset(
    {
        "budget_exhausted",
        "invalid_tool_name",
        "repeated_failure",
        "permission_denied",
        "tool_unavailable",
    }
)


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class ToolExecutionResult:
    """Result of tool execution phase.

    Attributes:
        chunks: List of chunks to yield to the stream.
        should_return: Whether the main loop should return (exit).
        tool_results: Results from tool execution.
        tool_calls_executed: Number of tool calls that were executed.
        last_tool_name: Name of the last executed tool (for reminder tracking).
    """

    chunks: List[StreamChunk] = field(default_factory=list)
    should_return: bool = False
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls_executed: int = 0
    last_tool_name: Optional[str] = None

    def add_chunk(self, chunk: StreamChunk) -> None:
        """Add a chunk to yield."""
        self.chunks.append(chunk)

    def add_chunks(self, chunks: List[StreamChunk]) -> None:
        """Add multiple chunks to yield."""
        self.chunks.extend(chunks)


# =============================================================================
# Tool Execution Handler
# =============================================================================


class ToolExecutionHandler:
    """Handler for tool execution in streaming chat.

    This class encapsulates the tool execution logic extracted from
    _stream_chat_impl, providing better testability and separation of concerns.

    The handler manages:
    - Budget checks and warnings
    - Progress tracking and force completion
    - Tool call filtering and truncation
    - Actual tool execution
    - Result chunk generation
    - Reminder injection

    Example:
        handler = ToolExecutionHandler(...)
        result = await handler.execute_tools(stream_ctx, tool_calls, ...)

        for chunk in result.chunks:
            yield chunk
        if result.should_return:
            return
    """

    def __init__(
        self,
        recovery_runtime: ToolExecutionRecoveryRuntimeProtocol,
        chunk_generator: StreamingChunkRuntimeProtocol,
        message_adder: StreamingMessageAdderProtocol,
        reminder_manager: StreamingReminderRuntimeProtocol,
        unified_tracker: StreamingTrackerRuntimeProtocol,
        settings: "Settings",
        recovery_context_factory: Callable[[StreamingChatContext], Any],
        check_progress_with_handler: Callable[[StreamingChatContext], None],
        handle_force_completion_with_handler: Callable[
            [StreamingChatContext], Optional[StreamChunk]
        ],
        handle_budget_exhausted: Callable[[StreamingChatContext], AsyncIterator[StreamChunk]],
        handle_force_final_response: Callable[[StreamingChatContext], AsyncIterator[StreamChunk]],
        execute_tool_calls: Callable[[List[Dict]], Any],
        get_tool_status_message: Callable[[str, Dict], str],
        set_tool_budget_limit: Optional[Callable[[int], int]] = None,
        observed_files: Optional[Set[str]] = None,
    ):
        """Initialize the tool execution handler.

        Args:
            recovery_runtime: Runtime port for recovery actions.
            chunk_generator: Generator for stream chunks.
            message_adder: Object that can add messages to conversation.
            reminder_manager: Manager for context reminders.
            unified_tracker: Unified task tracker.
            settings: Application settings.
            recovery_context_factory: Factory for recovery context.
            check_progress_with_handler: Callback for progress checking.
            handle_force_completion_with_handler: Callback for force completion.
            handle_budget_exhausted: Async generator for budget exhausted handling.
            handle_force_final_response: Async generator for force final response.
            execute_tool_calls: Callback to execute tool calls.
            get_tool_status_message: Function to generate tool status messages.
            set_tool_budget_limit: Callback to update the canonical turn budget.
            observed_files: Set of observed files (for reminder tracking).
        """
        self._recovery_runtime = recovery_runtime
        self._chunk_generator = chunk_generator
        self._message_adder = message_adder
        self._reminder_manager = reminder_manager
        self._unified_tracker = unified_tracker
        self._settings = settings
        self._recovery_context_factory = recovery_context_factory
        self._check_progress_with_handler = check_progress_with_handler
        self._handle_force_completion_with_handler = handle_force_completion_with_handler
        self._handle_budget_exhausted = handle_budget_exhausted
        self._handle_force_final_response = handle_force_final_response
        self._set_tool_budget_limit = set_tool_budget_limit
        self._execute_tool_calls_callback = execute_tool_calls
        self._get_tool_status_message = get_tool_status_message
        self._observed_files = observed_files or set()

    def update_observed_files(self, files: Set[str]) -> None:
        """Update the set of observed files."""
        self._observed_files = files

    def _record_omitted_tool_call_response(
        self,
        result: ToolExecutionResult,
        tool_call: Dict[str, Any],
        *,
        reason: str,
        outcome_kind: str,
        block_source: str,
    ) -> None:
        """Persist a synthetic tool response for a call omitted before execution."""
        tool_name = tool_call.get("name", "tool")
        tool_args = tool_call.get("arguments", {}) or {}
        tool_call_id = tool_call.get("id")
        content = (
            f"Tool call skipped for '{tool_name}': {reason} "
            "Use a different approach or continue with the available context."
        )

        if tool_call_id:
            try:
                self._message_adder.add_message(
                    "tool",
                    content,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                    persist_synchronously=True,
                )
            except TypeError:
                self._message_adder.add_message(
                    "tool",
                    content,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                )
            except Exception:
                logger.exception(
                    "Failed to persist synthetic tool response for %s (tool_call_id=%s)",
                    tool_name,
                    tool_call_id,
                )

        result.tool_results.append(
            {
                "name": tool_name,
                "success": False,
                "elapsed": 0.0,
                "args": tool_args,
                "error": reason,
                "result": content,
                "full_result": content,
                "follow_up_suggestions": None,
                "was_pruned": False,
                "tool_call_id": tool_call_id,
                "content": content,
                "skipped": True,
                "outcome_kind": outcome_kind,
                "block_source": block_source,
                "retryable": False,
                "user_message": reason,
            }
        )

    async def execute_tools(
        self,
        stream_ctx: StreamingChatContext,
        tool_calls: Optional[List[Dict[str, Any]]],
        user_message: str,
        full_content: str,
        tool_calls_used: int,
        tool_budget: int,
    ) -> ToolExecutionResult:
        """Execute tools and return results.

        This is the main entry point that handles the complete tool execution
        phase including budget checks, filtering, execution, and result generation.

        Args:
            stream_ctx: The streaming context.
            tool_calls: List of tool calls to execute.
            user_message: The original user message.
            full_content: Full content from model response.
            tool_calls_used: Current count of tool calls used.
            tool_budget: Maximum tool call budget.

        Returns:
            ToolExecutionResult with chunks and control flags.
        """
        result = ToolExecutionResult()
        async for _chunk in self.execute_tools_streaming(
            stream_ctx=stream_ctx,
            tool_calls=tool_calls,
            user_message=user_message,
            full_content=full_content,
            tool_calls_used=tool_calls_used,
            tool_budget=tool_budget,
            result=result,
        ):
            pass
        return result

    async def execute_tools_streaming(
        self,
        stream_ctx: StreamingChatContext,
        tool_calls: Optional[List[Dict[str, Any]]],
        user_message: str,
        full_content: str,
        tool_calls_used: int,
        tool_budget: int,
        result: Optional[ToolExecutionResult] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Execute tools while yielding UI-visible chunks as soon as they exist.

        The returned stream mutates ``result`` in-place so callers can yield
        progress immediately and still inspect the final accounting state after
        the generator is exhausted.
        """
        result = result or ToolExecutionResult()

        # Sync tool tracking to context
        stream_ctx.tool_calls_used = tool_calls_used
        stream_ctx.tool_budget = tool_budget

        remaining = stream_ctx.get_remaining_budget()

        logger.debug(
            f"Entering tool execution: tool_calls={len(tool_calls) if tool_calls else 0}, "
            f"tool_calls_used={tool_calls_used}/{tool_budget}"
        )

        # Check budget warning
        budget_warning = await self._check_budget_warning(stream_ctx)
        if budget_warning:
            result.add_chunk(budget_warning)
            yield budget_warning

        # Keep progress signals current before deciding whether to grant budget relief.
        stream_ctx.unique_resources = self._unified_tracker.unique_resources
        relief_chunk = self._maybe_extend_budget_for_progress(
            stream_ctx,
            requested_tool_calls=len(tool_calls or []),
            current_budget=tool_budget,
        )
        if relief_chunk is not None:
            result.add_chunk(relief_chunk)
            yield relief_chunk
            remaining = stream_ctx.get_remaining_budget()

        # Check budget exhausted
        if remaining <= 0:
            if tool_calls:
                for pending_call in tool_calls:
                    self._record_omitted_tool_call_response(
                        result,
                        pending_call,
                        reason="Skipped because the remaining tool budget for this turn was exhausted.",
                        outcome_kind="budget_exhausted",
                        block_source="tool_budget",
                    )
            logger.warning(
                "Tool budget exhausted before executing %s queued tool call(s); "
                "turn budget=%s used=%s",
                len(tool_calls) if tool_calls else 0,
                tool_budget,
                tool_calls_used,
            )
            exhausted_result = await self._handle_budget_exhausted_phase(stream_ctx)
            if not exhausted_result.chunks:
                exhausted_result.add_chunk(
                    StreamChunk(
                        content=(
                            f"[tool] Tool budget reached ({stream_ctx.tool_budget}); "
                            f"skipped {len(tool_calls) if tool_calls else 0} queued tool call(s).\n"
                        )
                    )
                )
                exhausted_result.add_chunk(
                    StreamChunk(
                        content=(
                            "Unable to continue tool execution in this turn. Start a follow-up "
                            "turn or increase the tool budget if more tool work is required.\n"
                        ),
                        is_final=True,
                    )
                )
            result.add_chunks(exhausted_result.chunks)
            result.should_return = True
            for chunk in exhausted_result.chunks:
                yield chunk
            return

        # Sync unique_resources and check progress
        stream_ctx.unique_resources = self._unified_tracker.unique_resources
        self._check_progress_with_handler(stream_ctx)

        # Check force completion
        force_result = await self._check_force_completion(stream_ctx)
        if force_result:
            result.add_chunks(force_result.chunks)
            result.should_return = force_result.should_return
            for chunk in force_result.chunks:
                yield chunk
            if result.should_return:
                return

        # Filter and truncate tool calls
        pre_filter_chunk_count = len(result.chunks)
        tool_calls = await self._filter_and_truncate_tools(stream_ctx, tool_calls, result)
        for chunk in result.chunks[pre_filter_chunk_count:]:
            yield chunk

        # Execute tools if any remain
        if tool_calls:
            async for chunk in self._execute_filtered_tool_calls_streaming(
                stream_ctx, tool_calls, result
            ):
                yield chunk

        # Update context for next iteration
        stream_ctx.update_context_message(full_content or user_message)

    async def _check_budget_warning(
        self, stream_ctx: StreamingChatContext
    ) -> Optional[StreamChunk]:
        """Check if budget warning should be shown."""
        recovery_ctx = self._recovery_context_factory(stream_ctx)
        warning_threshold = getattr(self._settings, "tool_call_budget_warning_threshold", 250)
        warning_pct = getattr(self._settings, "tool_call_budget_warning_pct", 0.8)
        warning_remaining = getattr(self._settings, "tool_call_budget_warning_remaining", 5)
        budget_warning = self._recovery_runtime.check_tool_budget(
            recovery_ctx,
            warning_threshold,
            warning_pct,
            warning_remaining,
        )
        if inspect.isawaitable(budget_warning):
            return await budget_warning
        return budget_warning

    def _maybe_extend_budget_for_progress(
        self,
        stream_ctx: StreamingChatContext,
        *,
        requested_tool_calls: int,
        current_budget: int,
    ) -> Optional[StreamChunk]:
        """Grant a bounded one-time budget extension for in-progress action tasks."""
        if requested_tool_calls <= 0:
            return None
        if not getattr(self._settings, "tool_budget_progress_relief_enabled", True):
            return None
        max_uses = max(0, int(getattr(self._settings, "tool_budget_progress_relief_max_uses", 1)))
        if getattr(stream_ctx, "budget_relief_uses", 0) >= max_uses:
            return None
        if not stream_ctx.is_action_task:
            return None
        if stream_ctx.force_completion:
            return None

        remaining = stream_ctx.get_remaining_budget()
        if remaining >= requested_tool_calls:
            return None

        executed_tool_names = {name.lower() for name in stream_ctx.executed_tool_names}
        has_progress_signal = bool(executed_tool_names & {"edit", "write", "shell", "test"})
        has_progress_signal = has_progress_signal or len(stream_ctx.unique_resources) >= 3
        has_progress_signal = has_progress_signal or len(self._observed_files) >= 3
        if not has_progress_signal:
            return None

        relief_amount = max(
            1, int(getattr(self._settings, "tool_budget_progress_relief_amount", 10))
        )
        missing_calls = requested_tool_calls - remaining
        relief_amount = min(relief_amount, missing_calls)
        if relief_amount <= 0:
            return None

        target_budget = max(current_budget, stream_ctx.tool_budget) + relief_amount
        effective_budget = target_budget
        if callable(self._set_tool_budget_limit):
            try:
                effective_budget = int(self._set_tool_budget_limit(target_budget))
            except Exception:
                logger.exception("Failed to apply progress-based tool budget relief")
                return None

        if effective_budget <= stream_ctx.tool_budget:
            return None

        stream_ctx.tool_budget = effective_budget
        stream_ctx.budget_relief_uses += 1
        stream_ctx.budget_warning_shown = False
        logger.info(
            "Granted progress-based tool budget relief: +%s call(s) -> %s total "
            "(requested=%s remaining_before=%s)",
            effective_budget - current_budget,
            effective_budget,
            requested_tool_calls,
            remaining,
        )
        return StreamChunk(
            content=(
                f"[tool] Progress detected; extending tool budget to {effective_budget} calls "
                f"for this turn.\n"
            )
        )

    async def _handle_budget_exhausted_phase(
        self, stream_ctx: StreamingChatContext
    ) -> ToolExecutionResult:
        """Handle budget exhausted condition."""
        result = ToolExecutionResult(should_return=True)
        async for chunk in self._handle_budget_exhausted(stream_ctx):
            result.add_chunk(chunk)
        return result

    async def _check_force_completion(
        self, stream_ctx: StreamingChatContext
    ) -> Optional[ToolExecutionResult]:
        """Check if force completion should be triggered."""
        force_chunk = self._handle_force_completion_with_handler(stream_ctx)
        if force_chunk:
            result = ToolExecutionResult(should_return=True)
            result.add_chunk(force_chunk)
            async for final_chunk in self._handle_force_final_response(stream_ctx):
                result.add_chunk(final_chunk)
            return result
        return None

    async def _filter_and_truncate_tools(
        self,
        stream_ctx: StreamingChatContext,
        tool_calls: Optional[List[Dict[str, Any]]],
        result: ToolExecutionResult,
    ) -> List[Dict[str, Any]]:
        """Filter and truncate tool calls based on budget and blocking rules."""
        if not tool_calls:
            return []

        original_calls = list(tool_calls)

        # Truncate to remaining budget
        recovery_ctx = self._recovery_context_factory(stream_ctx)
        remaining = stream_ctx.get_remaining_budget()
        tool_calls, was_truncated = self._recovery_runtime.truncate_tool_calls(
            recovery_ctx,
            tool_calls,
            remaining,
        )
        if was_truncated and len(tool_calls) < len(original_calls):
            omitted_calls = original_calls[len(tool_calls) :]
            result.add_chunk(
                StreamChunk(
                    content=(
                        f"\n[loop] Tool budget limited this turn; skipped "
                        f"{len(omitted_calls)} queued tool call(s).\n"
                    )
                )
            )
            for omitted_call in omitted_calls:
                self._record_omitted_tool_call_response(
                    result,
                    omitted_call,
                    reason="Skipped because the remaining tool budget for this turn was exhausted.",
                    outcome_kind="budget_exhausted",
                    block_source="tool_budget",
                )

        # Filter blocked tool calls
        filtered_calls, blocked_chunks, blocked_count = (
            self._recovery_runtime.filter_blocked_tool_calls(recovery_ctx, tool_calls)
        )
        result.add_chunks(blocked_chunks)
        filtered_call_ids = {id(call) for call in filtered_calls}
        blocked_calls = [call for call in tool_calls if id(call) not in filtered_call_ids]
        for blocked_call in blocked_calls:
            self._record_omitted_tool_call_response(
                result,
                blocked_call,
                reason="Blocked by runtime safeguards after repeated non-progressing attempts.",
                outcome_kind="tool_blocked",
                block_source="runtime_guard",
            )

        # Check blocked threshold
        all_blocked = blocked_count > 0 and not filtered_calls
        recovery_ctx = self._recovery_context_factory(stream_ctx)
        threshold_result = self._recovery_runtime.check_blocked_threshold(
            recovery_ctx,
            all_blocked,
        )
        if threshold_result:
            chunk, should_clear = threshold_result
            result.add_chunk(chunk)
            if should_clear:
                filtered_calls = []

        return filtered_calls

    async def _execute_filtered_tool_calls(
        self,
        stream_ctx: StreamingChatContext,
        tool_calls: List[Dict[str, Any]],
        result: ToolExecutionResult,
    ) -> None:
        """Execute tool calls and generate result chunks."""
        last_tool_name = self._add_tool_start_chunks(tool_calls, result)
        await self._execute_tool_call_batch(stream_ctx, tool_calls, result, last_tool_name)

    async def _execute_filtered_tool_calls_streaming(
        self,
        stream_ctx: StreamingChatContext,
        tool_calls: List[Dict[str, Any]],
        result: ToolExecutionResult,
    ) -> AsyncIterator[StreamChunk]:
        """Execute tool calls and yield start chunks before awaiting results."""
        pre_start_chunk_count = len(result.chunks)
        last_tool_name = self._add_tool_start_chunks(tool_calls, result)
        for chunk in result.chunks[pre_start_chunk_count:]:
            yield chunk

        pre_result_chunk_count = len(result.chunks)
        await self._execute_tool_call_batch(stream_ctx, tool_calls, result, last_tool_name)
        for chunk in result.chunks[pre_result_chunk_count:]:
            yield chunk

    def _add_tool_start_chunks(
        self,
        tool_calls: List[Dict[str, Any]],
        result: ToolExecutionResult,
    ) -> Optional[str]:
        """Add tool-start chunks and return the last tool name in the batch."""
        last_tool_name = None
        batch_total = len(tool_calls)
        execution_mode = "parallel_batch" if batch_total > 1 else "single"
        for batch_index, tool_call in enumerate(tool_calls, start=1):
            tool_name = tool_call.get("name", "tool")
            tool_args = tool_call.get("arguments", {})
            try:
                status_msg = self._get_tool_status_message(tool_name, tool_args)
            except Exception:
                # A cosmetic status message must never abort tool execution.
                logger.warning(
                    "Tool status message generation failed for '%s'",
                    tool_name,
                    exc_info=True,
                )
                status_msg = f"Running {tool_name}..."
            chunk = self._chunk_generator.generate_tool_start_chunk(
                tool_name,
                tool_args,
                status_msg,
                tool_call_id=tool_call.get("id"),
            )
            metadata = chunk.metadata if isinstance(chunk.metadata, dict) else None
            tool_start = metadata.get("tool_start") if metadata else None
            if isinstance(tool_start, dict):
                tool_start.update(
                    {
                        "batch_index": batch_index,
                        "batch_total": batch_total,
                        "execution_mode": execution_mode,
                    }
                )
            result.add_chunk(chunk)
            last_tool_name = tool_name
        return last_tool_name

    async def _execute_tool_call_batch(
        self,
        stream_ctx: StreamingChatContext,
        tool_calls: List[Dict[str, Any]],
        result: ToolExecutionResult,
        last_tool_name: Optional[str],
    ) -> None:
        """Execute a filtered tool batch and append result/reminder chunks."""
        # Execute all tool calls
        tool_results = await self._execute_tool_calls_callback(tool_calls)
        result.tool_results.extend(tool_results)
        result.tool_calls_executed = len(tool_calls)
        result.last_tool_name = last_tool_name
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            canonical_name = canonicalize_core_tool_name(resolve_tool_name(tool_name))
            stream_ctx.record_executed_tool_name(canonical_name)

        # Generate result chunks
        for tool_result in tool_results:
            tool_name = tool_result.get("name", "tool")
            for chunk in self._chunk_generator.generate_tool_result_chunks(tool_result):
                result.add_chunk(chunk)

        if self._should_force_completion_after_terminal_skips(tool_results):
            stream_ctx.force_completion = True
            self._message_adder.add_message(
                "user",
                (
                    "[SYSTEM: Tool execution is no longer making progress. "
                    "Do not request more blocked tools; summarize the current findings or explain "
                    "the blocker directly.]"
                ),
                metadata=build_internal_history_metadata("force_completion"),
            )

        # NOTE: We no longer add a "Thinking..." status chunk here because:
        # 1. The main pipeline iteration logic already handles status updates.
        # 2. Providers like z.ai/Anthropic yield their own reasoning/thinking content.
        # 3. Adding it here often causes duplication in the UI (see handler.py:278).

        # Update reminder manager
        self._reminder_manager.update_state(
            observed_files=self._observed_files,
            executed_tool=last_tool_name,
            tool_calls=stream_ctx.tool_calls_used + len(tool_calls),
        )

        # Get and inject consolidated reminder
        reminder = self._reminder_manager.get_consolidated_reminder()
        if reminder:
            self._message_adder.add_message(
                "user",
                f"[SYSTEM-REMINDER: {reminder}]",
                metadata=build_internal_history_metadata("system_reminder"),
            )

    @staticmethod
    def _should_force_completion_after_terminal_skips(
        tool_results: List[Dict[str, Any]],
    ) -> bool:
        """Force completion when every tool call ended in a terminal skip state."""
        if not tool_results:
            return False
        terminal_skips = [
            result
            for result in tool_results
            if result.get("skipped")
            and not result.get("success")
            and result.get("outcome_kind") in TERMINAL_SKIP_OUTCOME_KINDS
        ]
        return len(terminal_skips) == len(tool_results)


# =============================================================================
# Factory Function
# =============================================================================


def _noop_check_progress(stream_ctx: StreamingChatContext) -> None:
    """No-op fallback for progress checking."""
    pass


def _noop_force_completion(stream_ctx: StreamingChatContext) -> Optional[StreamChunk]:
    """No-op fallback for force completion (returns None = no force)."""
    return None


async def _noop_async_generator(stream_ctx: StreamingChatContext):
    """No-op async generator fallback that yields nothing."""
    return
    yield  # Make it an async generator


async def _default_budget_exhausted_generator(stream_ctx: StreamingChatContext):
    """Fallback generator for budget exhaustion when no runtime hook is provided."""
    yield StreamChunk(
        content=(
            f"[tool] Tool budget reached ({stream_ctx.tool_budget}); "
            "skipping remaining tool calls.\n"
        )
    )
    yield StreamChunk(
        content=(
            "Unable to continue tool execution in this turn. Start a follow-up turn or "
            "increase the tool budget if more tool work is required.\n"
        ),
        is_final=True,
    )


def create_tool_execution_handler(
    orchestrator: "AgentOrchestrator",
) -> ToolExecutionHandler:
    """Factory function to create a ToolExecutionHandler from an orchestrator.

    Args:
        orchestrator: The AgentOrchestrator instance.

    Returns:
        Configured ToolExecutionHandler.
    """
    from victor.agent.orchestrator_utils import get_tool_status_message

    recovery_context_factory = orchestrator.create_recovery_context

    def _set_budget_limit(budget: int) -> int:
        unified_tracker = getattr(orchestrator, "unified_tracker", None)
        if getattr(unified_tracker, "_sticky_user_budget", False):
            return int(
                getattr(
                    unified_tracker,
                    "tool_budget",
                    getattr(orchestrator, "tool_budget", budget),
                )
            )

        effective_budget = int(budget)
        if unified_tracker is not None and hasattr(unified_tracker, "set_tool_budget"):
            unified_tracker.set_tool_budget(effective_budget)
            effective_budget = int(getattr(unified_tracker, "tool_budget", effective_budget))

        orchestrator.tool_budget = effective_budget

        task_coordinator = getattr(orchestrator, "task_coordinator", None)
        if task_coordinator is not None and hasattr(task_coordinator, "tool_budget"):
            try:
                task_coordinator.tool_budget = effective_budget
            except Exception:
                logger.debug("Failed to sync task coordinator tool budget", exc_info=True)

        tool_service = getattr(orchestrator, "_tool_service", None)
        if tool_service is not None and hasattr(tool_service, "set_tool_budget"):
            try:
                tool_service.set_tool_budget(effective_budget)
            except Exception:
                logger.debug("Failed to sync tool service budget", exc_info=True)

        return effective_budget

    return ToolExecutionHandler(
        recovery_runtime=(
            getattr(orchestrator, "_recovery_service", None) or orchestrator._recovery_coordinator
        ),
        chunk_generator=orchestrator._chunk_generator,
        message_adder=orchestrator,
        reminder_manager=orchestrator.reminder_manager,
        unified_tracker=orchestrator.unified_tracker,
        settings=orchestrator.settings,
        recovery_context_factory=recovery_context_factory,
        check_progress_with_handler=getattr(
            orchestrator, "_check_progress_with_handler", _noop_check_progress
        ),
        handle_force_completion_with_handler=getattr(
            orchestrator,
            "_handle_force_completion_with_handler",
            _noop_force_completion,
        ),
        handle_budget_exhausted=getattr(
            orchestrator,
            "_handle_budget_exhausted",
            _default_budget_exhausted_generator,
        ),
        handle_force_final_response=getattr(
            orchestrator, "_handle_force_final_response", _noop_async_generator
        ),
        set_tool_budget_limit=_set_budget_limit,
        execute_tool_calls=orchestrator.execute_tool_calls,
        get_tool_status_message=get_tool_status_message,
        observed_files=(set(orchestrator.observed_files) if orchestrator.observed_files else set()),
    )
