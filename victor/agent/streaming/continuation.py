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

"""Continuation decision handler for streaming chat.

This module provides the ContinuationHandler class which encapsulates
the continuation decision logic that previously lived inside the legacy
``ChatCoordinator._stream_chat_impl`` and is now invoked through the
``StreamingChatExecutor``.

The handler processes ContinuationStrategy action results and applies them
to the streaming context, yielding appropriate chunks and returning control
signals for the main loop.

Design Pattern: Command + Strategy
==================================
The handler implements a command-style pattern where each continuation action
(prompt_tool_call, request_summary, finish, etc.) maps to a dedicated handler
method. The ContinuationStrategy determines WHAT action to take, and this
handler determines HOW to execute it.

Usage:
    handler = ContinuationHandler(
        orchestrator=self,
        chunk_generator=self._chunk_generator,
        sanitizer=self.sanitizer,
    )

    result = await handler.handle_continuation_action(
        action_result=action_result,
        stream_ctx=stream_ctx,
        full_content=full_content,
    )

    async for chunk in result.chunks:
        yield chunk

    if result.should_return:
        return
    if result.should_skip_rest:
        continue
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
)

from victor.agent.continuation_strategy import (
    ContinuationActionType,
    coerce_continuation_action,
)
from victor.agent.continuation_contract import ContinuationDirective
from victor.agent.conversation.history_metadata import build_internal_history_metadata
from victor.agent.services.protocols.streaming_runtime import (
    StreamingChunkRuntimeProtocol,
    StreamingMessageAdderProtocol,
    StreamingProviderRuntimeProtocol,
    StreamingSanitizerRuntimeProtocol,
)
from victor.agent.streaming.context import StreamingChatContext
from victor.providers.base import StreamChunk

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.streaming.handler import StreamingChatHandler
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


def _internal_prompt_metadata(kind: str) -> dict[str, Any]:
    """Build metadata for internal user-role continuation prompts."""
    from victor.agent.conversation.types import MessageSource

    return build_internal_history_metadata(kind, source=MessageSource.AGENT_CONTINUATION)


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class ContinuationResult:
    """Result of handling a continuation action.

    Attributes:
        chunks: List of chunks to yield to the stream.
        should_return: Whether the main loop should return (exit).
        should_skip_rest: Whether to skip remaining loop body and continue.
        should_continue_loop: Whether to continue the main while loop.
        state_updates: Dict of state updates to apply to the orchestrator.
    """

    chunks: List[StreamChunk] = field(default_factory=list)
    should_return: bool = False
    should_skip_rest: bool = False
    should_continue_loop: bool = True
    state_updates: Dict[str, Any] = field(default_factory=dict)

    def add_chunk(self, chunk: StreamChunk) -> None:
        """Add a chunk to yield."""
        self.chunks.append(chunk)


# =============================================================================
# Continuation Handler
# =============================================================================


class ContinuationHandler:
    """Handler for continuation decision execution.

    This class encapsulates the logic for executing continuation actions
    determined by ContinuationStrategy. It extracts ~300 lines of if/elif
    chain from _stream_chat_impl into focused, testable methods.

    Each action type has a dedicated handler method:
    - _handle_continue_asking_input
    - _handle_return_to_user
    - _handle_prompt_tool_call
    - _handle_continue_with_synthesis_hint
    - _handle_request_summary
    - _handle_request_completion
    - _handle_execute_extracted_tool
    - _handle_force_tool_execution
    - _handle_finish

    Example:
        handler = ContinuationHandler(message_adder=orchestrator, ...)
        result = await handler.handle_action(action_result, stream_ctx, content)

        for chunk in result.chunks:
            yield chunk
        if result.should_return:
            return
    """

    def __init__(
        self,
        message_adder: StreamingMessageAdderProtocol,
        chunk_generator: StreamingChunkRuntimeProtocol,
        sanitizer: StreamingSanitizerRuntimeProtocol,
        settings: "Settings",
        provider: Optional[StreamingProviderRuntimeProtocol] = None,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        messages_getter: Optional[Callable[[], List[Any]]] = None,
        unified_tracker: Optional[Any] = None,
        finalize_metrics: Optional[Callable[[Dict], Any]] = None,
        record_outcome: Optional[Callable[..., None]] = None,
        execute_extracted_tool: Optional[Callable[..., AsyncIterator[StreamChunk]]] = None,
    ):
        """Initialize the continuation handler.

        Args:
            message_adder: Object that can add messages to conversation.
            chunk_generator: Generator for stream chunks.
            sanitizer: Content sanitizer.
            settings: Application settings.
            provider: Optional LLM provider for forced responses.
            model: Model name for provider calls.
            temperature: Temperature for provider calls.
            max_tokens: Max tokens for provider calls.
            messages_getter: Callable to get current messages list.
            unified_tracker: Task tracker for turn increment.
            finalize_metrics: Callable to finalize stream metrics.
            record_outcome: Callable to record Q-learning outcome.
            execute_extracted_tool: Callable for extracted tool execution.
        """
        self._message_adder = message_adder
        self._chunk_generator = chunk_generator
        self._sanitizer = sanitizer
        self._settings = settings
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._messages_getter = messages_getter
        self._unified_tracker = unified_tracker
        self._finalize_metrics = finalize_metrics
        self._record_outcome = record_outcome
        self._execute_extracted_tool = execute_extracted_tool

        # State tracking (mirrors orchestrator state)
        self._cumulative_prompt_interventions = 0
        self._summary_request_count = 0

    async def handle_action(
        self,
        action_result: ContinuationDirective,
        stream_ctx: StreamingChatContext,
        full_content: str,
    ) -> ContinuationResult:
        """Handle a continuation action from ContinuationStrategy.

        This is the main entry point that dispatches to specific handlers
        based on the action type.

        Args:
            action_result: The result from ContinuationStrategy.determine_continuation_action()
            stream_ctx: The streaming context.
            full_content: The full response content from the model.

        Returns:
            ContinuationResult with chunks and control flags.
        """
        # P0 FIX: Check forced_stop signal from unified_tracker FIRST
        # This prevents continuation strategy from overriding task completion detector
        if self._unified_tracker:
            forced_stop = getattr(self._unified_tracker, "_progress", None)
            if forced_stop and hasattr(forced_stop, "forced_stop"):
                stop_reason = getattr(forced_stop, "forced_stop", None)
                if stop_reason:
                    logger.info(f"Early exit: forced_stop signal detected (reason={stop_reason})")
                    return ContinuationResult(
                        chunks=[],
                        should_return=True,
                        should_continue_loop=False,
                    )

        if not isinstance(action_result, ContinuationDirective):
            legacy_payload = dict(action_result) if isinstance(action_result, dict) else {}
            action_result = ContinuationDirective.from_legacy(
                action=legacy_payload.get("action", ContinuationActionType.FINISH),
                reason=legacy_payload.get("reason", "legacy_action_result"),
                message=legacy_payload.get("message"),
                updates=legacy_payload.get("updates"),
                extracted_call=legacy_payload.get("extracted_call"),
                mentioned_tools=legacy_payload.get("mentioned_tools"),
                set_final_summary_requested=bool(
                    legacy_payload.get("set_final_summary_requested", False)
                ),
                set_max_prompts_summary_requested=bool(
                    legacy_payload.get("set_max_prompts_summary_requested", False)
                ),
            )

        # Apply state updates from action result
        self._apply_state_updates(action_result)
        action = coerce_continuation_action(action_result.get("action"))

        # AGENTIC LOOP FIX: Check if continuation should be skipped due to forced completion
        # This prevents continuation strategy from overriding task completion detector
        # Recovery actions are allowed even with skip_continuation set:
        # - EXECUTE_EXTRACTED_TOOL: Allow extracted tool execution to complete
        # - FORCE_TOOL_EXECUTION: Allow giving up and forcing completion (via continuation handler)
        # - REQUEST_SUMMARY: Allow summary request when giving up on force tool execution
        recovery_actions = {
            ContinuationActionType.EXECUTE_EXTRACTED_TOOL,
            ContinuationActionType.FORCE_TOOL_EXECUTION,
            ContinuationActionType.REQUEST_SUMMARY,
        }
        if getattr(stream_ctx, "skip_continuation", False) and action not in recovery_actions:
            logger.info("Continuation skipped: skip_continuation flag set (completion was forced)")
            return ContinuationResult(
                chunks=[],
                should_return=True,
                should_continue_loop=False,
            )

        # Dispatch to specific handler
        handlers = {
            ContinuationActionType.CONTINUE_ASKING_INPUT: self._handle_continue_asking_input,
            ContinuationActionType.RETURN_TO_USER: self._handle_return_to_user,
            ContinuationActionType.PROMPT_TOOL_CALL: self._handle_prompt_tool_call,
            ContinuationActionType.CONTINUE_WITH_SYNTHESIS_HINT: (
                self._handle_continue_with_synthesis_hint
            ),
            ContinuationActionType.REQUEST_SUMMARY: self._handle_request_summary,
            ContinuationActionType.REQUEST_COMPLETION: self._handle_request_completion,
            ContinuationActionType.EXECUTE_EXTRACTED_TOOL: self._handle_execute_extracted_tool,
            ContinuationActionType.FORCE_TOOL_EXECUTION: self._handle_force_tool_execution,
            ContinuationActionType.FINISH: self._handle_finish,
        }

        handler = handlers.get(action, self._handle_finish)
        return await handler(action_result, stream_ctx, full_content)

    def _apply_state_updates(self, action_result: ContinuationDirective) -> Dict[str, Any]:
        """Apply state updates from action result.

        Args:
            action_result: The action result with updates.

        Returns:
            Dict of updates that were applied.
        """
        updates = action_result.state_patch.to_updates_dict()
        applied = {}

        if "cumulative_prompt_interventions" in updates:
            # Note: This is tracked internally but also returned for orchestrator sync
            applied["cumulative_prompt_interventions"] = updates["cumulative_prompt_interventions"]

        if action_result.state_patch.final_summary_requested:
            applied["final_summary_requested"] = True

        if action_result.state_patch.max_prompts_summary_requested:
            applied["max_prompts_summary_requested"] = True

        return applied

    async def _handle_continue_asking_input(
        self,
        action_result: ContinuationDirective,
        stream_ctx: StreamingChatContext,
        full_content: str,
    ) -> ContinuationResult:
        """Handle continue_asking_input action.

        Adds a follow-up message and continues the loop.
        """
        result = ContinuationResult(should_skip_rest=True)
        message = action_result.get("message", "")
        if message:
            self._message_adder.add_message(
                "user",
                message,
                metadata=_internal_prompt_metadata("continue_asking_input"),
            )
        return result

    async def _handle_return_to_user(
        self,
        action_result: ContinuationDirective,
        stream_ctx: StreamingChatContext,
        full_content: str,
    ) -> ContinuationResult:
        """Handle return_to_user action.

        Content is already yielded by the pipeline before continuation handling.
        """
        result = ContinuationResult(should_return=True)
        result.add_chunk(self._chunk_generator.generate_final_marker_chunk())
        return result

    async def _handle_prompt_tool_call(
        self,
        action_result: ContinuationDirective,
        stream_ctx: StreamingChatContext,
        full_content: str,
    ) -> ContinuationResult:
        """Handle prompt_tool_call action.

        Prompts the model to make a tool call.
        """
        result = ContinuationResult(should_skip_rest=True)
        message = action_result.get("message", "")
        if message:
            self._message_adder.add_message(
                "user",
                message,
                metadata=_internal_prompt_metadata("prompt_tool_call"),
            )

        # Increment turn in tracker
        if self._unified_tracker:
            self._unified_tracker.increment_turn()

        # Track intervention
        self._cumulative_prompt_interventions += 1
        result.state_updates["cumulative_prompt_interventions"] = (
            self._cumulative_prompt_interventions
        )

        return result

    async def _handle_continue_with_synthesis_hint(
        self,
        action_result: ContinuationDirective,
        stream_ctx: StreamingChatContext,
        full_content: str,
    ) -> ContinuationResult:
        """Handle continue_with_synthesis_hint action.

        Gentle nudge to synthesize when all required files are read.
        """
        result = ContinuationResult(should_skip_rest=True)
        message = action_result.get("message", "")
        if message:
            self._message_adder.add_message(
                "user",
                message,
                metadata=_internal_prompt_metadata("synthesis_hint"),
            )

        # Update synthesis nudge count in tracker
        updates = action_result.get("updates", {})
        if "synthesis_nudge_count" in updates and self._unified_tracker:
            if hasattr(self._unified_tracker, "synthesis_nudge_count"):
                self._unified_tracker.synthesis_nudge_count = updates["synthesis_nudge_count"]

        return result

    async def _handle_request_summary(
        self,
        action_result: ContinuationDirective,
        stream_ctx: StreamingChatContext,
        full_content: str,
    ) -> ContinuationResult:
        """Handle request_summary action.

        Requests a summary from the model, with forced response on second request.
        """
        # Check if this is a repeated request
        if self._summary_request_count >= 1:
            return await self._force_summary_response(stream_ctx)

        # First summary request
        self._summary_request_count += 1
        result = ContinuationResult(should_skip_rest=True)
        message = action_result.get("message", "")
        if message:
            self._message_adder.add_message(
                "user",
                message,
                metadata=_internal_prompt_metadata("request_summary"),
            )

        return result

    async def _force_summary_response(
        self,
        stream_ctx: StreamingChatContext,
    ) -> ContinuationResult:
        """Force a summary response when model ignores summary request.

        Disables tools and makes a direct provider call.
        """
        result = ContinuationResult(should_return=True)
        logger.warning(
            "Model ignored previous summary request - forcing final response with tools disabled"
        )

        if not self._provider or not self._messages_getter:
            # Can't force without provider
            result.add_chunk(self._chunk_generator.generate_final_marker_chunk())
            return result

        try:
            # Import Message from providers.base (not core.types)
            from victor.providers.base import Message

            messages = self._messages_getter() + [
                Message(
                    role="user",
                    content="CRITICAL: Provide your FINAL ANALYSIS NOW. "
                    "Do NOT mention any more tools or files. "
                    "Summarize what you found from the tool calls you already executed.",
                )
            ]

            response = await self._provider.chat(
                messages=messages,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                tools=None,  # Disable tools to force text response
            )

            if response and response.content:
                sanitized = self._sanitizer.sanitize(response.content)
                if sanitized:
                    from victor.agent.conversation.types import (
                        MESSAGE_SOURCE_METADATA_KEY,
                        MessageSource,
                    )

                    self._message_adder.add_message(
                        "assistant",
                        sanitized,
                        metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_RESPONSE.value},
                    )
                    result.add_chunk(self._chunk_generator.generate_content_chunk(sanitized))

            # Finalize metrics
            if self._finalize_metrics:
                final_metrics = self._finalize_metrics(stream_ctx.cumulative_usage)
                elapsed_time = (
                    final_metrics.total_duration
                    if final_metrics
                    else time.time() - stream_ctx.start_time
                )
                cost_str = None
                if self._settings.analytics.show_cost_metrics and final_metrics:
                    cost_str = final_metrics.format_cost()
                metrics_line = self._chunk_generator.format_completion_metrics(
                    stream_ctx, elapsed_time, cost_str
                )
                result.add_chunk(self._chunk_generator.generate_metrics_chunk(metrics_line))

            result.add_chunk(self._chunk_generator.generate_final_marker_chunk())

        except Exception as e:
            logger.warning(f"Error forcing final response: {e}")
            result.add_chunk(self._chunk_generator.generate_final_marker_chunk())

        return result

    async def _handle_request_completion(
        self,
        action_result: ContinuationDirective,
        stream_ctx: StreamingChatContext,
        full_content: str,
    ) -> ContinuationResult:
        """Handle request_completion action.

        Requests the model to complete its response.
        """
        result = ContinuationResult(should_skip_rest=True)
        message = action_result.get("message", "")
        if message:
            self._message_adder.add_message(
                "user",
                message,
                metadata=_internal_prompt_metadata("request_completion"),
            )
        return result

    async def _handle_execute_extracted_tool(
        self,
        action_result: ContinuationDirective,
        stream_ctx: StreamingChatContext,
        full_content: str,
    ) -> ContinuationResult:
        """Handle execute_extracted_tool action.

        Executes a tool call extracted from the model's text response.
        """
        result = ContinuationResult(should_skip_rest=True)
        extracted_call = action_result.get("extracted_call")

        if extracted_call and self._execute_extracted_tool:
            logger.info(
                f"Executing extracted tool call: {extracted_call.tool_name} "
                f"(confidence: {extracted_call.confidence:.2f})"
            )
            # Execute and collect chunks
            async for chunk in self._execute_extracted_tool(stream_ctx, extracted_call):
                result.add_chunk(chunk)

        return result

    async def _handle_force_tool_execution(
        self,
        action_result: ContinuationDirective,
        stream_ctx: StreamingChatContext,
        full_content: str,
    ) -> ContinuationResult:
        """Handle force_tool_execution action.

        Forces tool execution when model mentions tools but doesn't call them.

        P0 FIX: Properly exits loop after 2 failed attempts (down from 3).
        """
        result = ContinuationResult(should_skip_rest=True)
        mentioned_tools = action_result.get("mentioned_tools", [])
        force_message = action_result.get("message", "")

        # Track force attempt
        attempt_count = stream_ctx.record_force_tool_attempt()

        # P0 FIX: Give up after 2 attempts (down from 3) for faster exit
        if attempt_count >= 2:
            # Give up and force completion
            logger.warning(
                "Giving up on forced tool execution after %d attempts - forcing completion",
                attempt_count,
            )
            self._message_adder.add_message(
                "user",
                "You are unable to make tool calls. Please provide your response "
                "NOW based on what you know. Do not mention any tools.",
                metadata=_internal_prompt_metadata("force_tool_execution_give_up"),
            )
            stream_ctx.reset_force_tool_attempts()
            # P0 FIX: Force completion to prevent further looping
            stream_ctx.force_completion = True
            stream_ctx.skip_continuation = True
            result.should_skip_rest = False  # Allow the rest of the loop to finalize
            result.should_continue_loop = True  # Continue to process the final response
        elif force_message:
            self._message_adder.add_message(
                "user",
                force_message,
                metadata=_internal_prompt_metadata("force_tool_execution"),
            )
        else:
            # Default message
            tools_str = ", ".join(mentioned_tools)
            default_message = (
                f"You mentioned using {tools_str} but did not actually call the tool(s). "
                "Please make the actual tool call now, or provide your final answer without "
                "mentioning tools you cannot use."
            )
            self._message_adder.add_message(
                "user",
                default_message,
                metadata=_internal_prompt_metadata("force_tool_execution"),
            )

        # Increment turn in tracker (mirrors _handle_force_tool_execution_with_handler)
        if self._unified_tracker:
            self._unified_tracker.increment_turn()

        return result

    async def _handle_finish(
        self,
        action_result: ContinuationDirective,
        stream_ctx: StreamingChatContext,
        full_content: str,
    ) -> ContinuationResult:
        """Handle finish action.

        Finalizes the response with metrics and exits.
        """
        result = ContinuationResult(should_return=True)

        # Content is already yielded by the pipeline before continuation handling.

        # Finalize and display metrics
        if self._finalize_metrics:
            final_metrics = self._finalize_metrics(stream_ctx.cumulative_usage)
            elapsed_time = (
                final_metrics.total_duration
                if final_metrics
                else time.time() - stream_ctx.start_time
            )
            cost_str = None
            if self._settings.analytics.show_cost_metrics and final_metrics:
                cost_str = final_metrics.format_cost()
            metrics_line = self._chunk_generator.format_completion_metrics(
                stream_ctx, elapsed_time, cost_str
            )
            result.add_chunk(self._chunk_generator.generate_metrics_chunk(metrics_line))

        # Record outcome for Q-learning
        if self._record_outcome:
            self._record_outcome(
                success=True,
                quality_score=stream_ctx.last_quality_score,
                user_satisfied=True,
                completed=True,
            )

        result.add_chunk(self._chunk_generator.generate_final_marker_chunk())
        return result


# =============================================================================
# Factory Function
# =============================================================================


def create_continuation_handler(
    orchestrator: "AgentOrchestrator",
) -> ContinuationHandler:
    """Factory function to create a ContinuationHandler from an orchestrator.

    Args:
        orchestrator: The AgentOrchestrator instance.

    Returns:
        Configured ContinuationHandler.
    """
    return ContinuationHandler(
        message_adder=orchestrator,
        chunk_generator=orchestrator._chunk_generator,
        sanitizer=orchestrator.sanitizer,
        settings=orchestrator.settings,
        provider=orchestrator.provider,
        model=orchestrator.model,
        temperature=orchestrator.temperature,
        max_tokens=orchestrator.max_tokens,
        messages_getter=lambda: orchestrator.messages,
        unified_tracker=orchestrator.unified_tracker,
        finalize_metrics=orchestrator._finalize_stream_metrics,
        record_outcome=orchestrator._record_runtime_intelligence_outcome,
        execute_extracted_tool=getattr(orchestrator, "_execute_extracted_tool_call", None),
    )
