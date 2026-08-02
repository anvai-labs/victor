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

"""Streaming chat handler.

Main handler for streaming chat iterations, extracted from AgentOrchestrator
for better testability and separation of concerns.
"""

import logging
import time
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TYPE_CHECKING,
)

from victor.agent.conversation.history_metadata import build_internal_history_metadata
from victor.framework.temperature import escalate_temperature
from victor.agent.streaming.context import StreamingChatContext
from victor.agent.streaming.iteration import (
    IterationAction,
    IterationResult,
    ProviderResponseResult,
    ToolExecutionResult,
    create_break_result,
    create_continue_result,
    create_force_completion_result,
)
from victor.core.loop_thresholds import (
    DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD,
    DEFAULT_BLOCKED_TOTAL_THRESHOLD,
)
from victor.providers.base import StreamChunk
from victor.tools.core_tool_aliases import canonicalize_core_tool_name

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.presentation import PresentationProtocol
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


class MessageAdder(Protocol):
    """Protocol for adding messages to conversation."""

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        """Add a message to the conversation."""
        ...


class ToolExecutor(Protocol):
    """Protocol for executing tools."""

    async def execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tool calls and return results."""
        ...


class StreamingChatHandler:
    """Handler for streaming chat iterations.

    This class encapsulates the core streaming loop logic, making it
    testable independently of the full orchestrator. Dependencies are
    injected via protocols, enabling mocking in tests.
    """

    def __init__(
        self,
        settings: "Settings",
        message_adder: MessageAdder,
        tool_executor: Optional[ToolExecutor] = None,
        session_idle_timeout: float = 180.0,
        presentation: Optional[Any] = None,
    ):
        """Initialize the streaming handler.

        Args:
            settings: Application settings
            message_adder: Object that can add messages to conversation
            tool_executor: Optional tool executor for running tools
            session_idle_timeout: Maximum idle time in seconds (resets on activity)
            presentation: Optional PresentationProtocol for icon rendering
        """
        self.settings = settings
        self.message_adder = message_adder
        self.tool_executor = tool_executor
        self.session_idle_timeout = session_idle_timeout

        # Presentation adapter for icon rendering (decouples from UI)
        if presentation is None:
            from victor.agent.presentation import create_presentation_adapter

            self._presentation = create_presentation_adapter()
        else:
            self._presentation = presentation

    @staticmethod
    def _infer_preview_language(path: str, preview_kind: str) -> str:
        if preview_kind == "edit":
            return "diff"

        if "." not in path:
            return "text"
        suffix = path.rsplit(".", 1)[-1].lower()
        return suffix or "text"

    def _record_preview_message(self, kind: str, path: str, body: str) -> None:
        """Persist preview metadata for session replay without affecting stream output."""
        if not body:
            return

        header = f"{kind}: {path}" if path else kind
        preview_kind = "edit" if kind.lower().startswith("edit") else "file"
        try:
            self.message_adder.add_message(
                "system",
                header,
                preview_body=body,
                preview_kind=preview_kind,
                preview_language=self._infer_preview_language(path, preview_kind),
                preview_path=path,
            )
        except TypeError:
            # Older adapters may not accept metadata kwargs.
            self.message_adder.add_message("system", header)

    def check_time_limit(self, ctx: StreamingChatContext) -> Optional[IterationResult]:
        """Check if session idle time limit has been exceeded.

        Args:
            ctx: The streaming context

        Returns:
            IterationResult if limit exceeded, None otherwise
        """
        if ctx.is_over_time_limit(self.session_idle_timeout):
            logger.warning(
                f"Session idle timeout ({self.session_idle_timeout}s) reached - "
                f"idle={ctx.elapsed_time():.1f}s, total={ctx.total_session_time():.1f}s, forcing completion"
            )
            result = create_force_completion_result(
                f"Session idle timeout ({self.session_idle_timeout}s) reached. "
                "Providing summary of progress so far."
            )
            # Add message to force summary
            self.message_adder.add_message(
                "user",
                "IDLE TIMEOUT REACHED. Provide a summary of your findings NOW. "
                "Do NOT call any more tools.",
            )
            ctx.force_completion = True
            return result
        return None

    def check_iteration_limit(self, ctx: StreamingChatContext) -> Optional[IterationResult]:
        """Check if iteration limit has been exceeded.

        Args:
            ctx: The streaming context

        Returns:
            IterationResult if limit exceeded, None otherwise
        """
        if ctx.is_over_iteration_limit():
            logger.warning(
                f"Max iterations ({ctx.max_total_iterations}) reached - forcing completion"
            )
            return create_break_result(
                f"\n\n{self._presentation.icon('warning', with_color=False)} Maximum iterations ({ctx.max_total_iterations}) reached.\n"
            )
        return None

    def check_force_completion(self, ctx: StreamingChatContext) -> Optional[IterationResult]:
        """Check if force completion conditions are met.

        Args:
            ctx: The streaming context

        Returns:
            IterationResult if should force, None otherwise
        """
        if ctx.should_force_completion():
            logger.info("Force completion triggered")
            return create_force_completion_result("Forcing completion due to constraints.")
        return None

    def handle_blocked_attempts(self, ctx: StreamingChatContext) -> Optional[IterationResult]:
        """Handle consecutive blocked tool attempts.

        Args:
            ctx: The streaming context

        Returns:
            IterationResult if force triggered, None otherwise
        """
        force_triggered = ctx.record_blocked_attempt()
        if force_triggered:
            logger.warning(
                f"Multiple blocked attempts ({ctx.consecutive_blocked_attempts}) - "
                "forcing completion"
            )
            result = IterationResult(action=IterationAction.YIELD_AND_CONTINUE)
            result.add_chunk(
                StreamChunk(
                    content=f"\n[loop] {self._presentation.icon('warning', with_color=False)} Multiple blocked attempts - forcing completion\n"
                )
            )
            # Add strong instruction to stop tool use
            self.message_adder.add_message(
                "user",
                f"{self._presentation.icon('warning', with_color=False)} STOP: You have attempted blocked operations too many times. "
                "You MUST now provide your final response WITHOUT any tool calls. "
                "Summarize what you found and answer the user's question based on "
                "the information you have already gathered. DO NOT call any more tools.",
            )
            ctx.reset_blocked_attempts()
            return result
        return None

    def process_provider_response(
        self, response: ProviderResponseResult, ctx: StreamingChatContext
    ) -> IterationResult:
        """Process the response from the LLM provider.

        Args:
            response: The provider's response
            ctx: The streaming context

        Returns:
            IterationResult with appropriate action
        """
        # Reset activity timer on provider response to prevent idle timeout
        ctx.reset_activity_timer()

        # Update context with response content
        ctx.accumulate_content(response.content)
        ctx.update_context_message(response.content)

        # Handle garbage detection
        if response.garbage_detected and not response.has_tool_calls:
            ctx.force_completion = True
            logger.info("Setting force_completion due to garbage detection")

        # Create result based on response
        if response.has_tool_calls:
            return create_continue_result(
                content=response.content,
                tool_calls=response.tool_calls,
                tokens=response.tokens_used,
            )
        elif response.has_content:
            # Content without tool calls - may be final or need continuation
            result = IterationResult(
                action=IterationAction.YIELD_AND_CONTINUE,
                content=response.content,
                tokens_used=response.tokens_used,
            )
            result.add_chunk(StreamChunk(content=response.content))
            return result
        else:
            # Empty response
            return create_continue_result()

    def process_tool_results(
        self, execution: ToolExecutionResult, ctx: StreamingChatContext
    ) -> List[StreamChunk]:
        """Process tool execution results and generate status chunks.

        Args:
            execution: The tool execution results
            ctx: The streaming context

        Returns:
            List of StreamChunk objects for tool result status
        """
        chunks = []
        for result in execution.results:
            tool_name = result.get("name", "tool")
            elapsed = result.get("elapsed", 0.0)
            tool_args = result.get("args", {})
            success = result.get("success", False)
            error_msg = result.get("error", "failed") if not success else None
            tool_output = result.get("result")  # Get actual tool output for preview
            chunks.append(
                self.generate_tool_result_chunk(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    elapsed=elapsed,
                    success=success,
                    error=error_msg,
                    follow_up_suggestions=result.get("follow_up_suggestions"),
                    result=tool_output,  # Pass result for preview
                )
            )

        # NOTE: We no longer add a "Thinking..." status chunk here because:
        # 1. The main pipeline iteration logic already handles status updates.
        # 2. Providers like z.ai/Anthropic yield their own reasoning/thinking content.
        # 3. Adding it here often causes duplication in the UI.
        return chunks

    def generate_tool_start_chunk(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        status_msg: str,
        tool_call_id: Optional[str] = None,
    ) -> StreamChunk:
        """Generate a chunk indicating tool execution start.

        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments
            status_msg: Status message to display
            tool_call_id: Provider tool_calls[].id — lets renderers track N
                concurrent tools independently instead of clobbering one slot

        Returns:
            StreamChunk with tool start metadata
        """
        tool_start: Dict[str, Any] = {
            "name": tool_name,
            "arguments": tool_args,
            "status_msg": status_msg,
        }
        if tool_call_id:
            tool_start["tool_call_id"] = tool_call_id
        return StreamChunk(content="", metadata={"tool_start": tool_start})

    def should_continue_loop(self, result: IterationResult, ctx: StreamingChatContext) -> bool:
        """Determine if the streaming loop should continue.

        Args:
            result: The current iteration result
            ctx: The streaming context

        Returns:
            True if loop should continue, False otherwise
        """
        if result.should_break:
            return False
        if ctx.is_over_iteration_limit():
            return False
        if ctx.should_force_completion() and not result.has_tool_calls:
            return False
        return True

    def check_natural_completion(
        self, ctx: StreamingChatContext, has_tool_calls: bool, content_length: int
    ) -> Optional[IterationResult]:
        """Check if the response represents natural completion (substantial content, no tools).

        Signal-based completion detection uses explicit rare completion markers
        instead of buffer/size heuristics. The TaskCompletionDetector handles all
        completion detection based on explicit markers. This method is kept for
        backward compatibility but always returns None.

        Args:
            ctx: The streaming context
            has_tool_calls: Whether there are pending tool calls
            content_length: Length of current response content

        Returns:
            IterationResult to break if natural completion, None otherwise
        """
        # Signal-based completion is always active - TaskCompletionDetector handles
        # completion detection based on explicit rare completion markers
        return None

    def handle_empty_response(self, ctx: StreamingChatContext) -> Optional[IterationResult]:
        """Handle an empty response from the model.

        Tracks consecutive empty responses and forces summary if threshold exceeded.

        Args:
            ctx: The streaming context

        Returns:
            IterationResult if threshold exceeded and summary forced, None otherwise
        """
        threshold_exceeded = ctx.record_empty_response()
        if threshold_exceeded:
            logger.warning(
                f"Model stuck with {ctx.consecutive_empty_responses} consecutive "
                "empty responses - forcing summary"
            )
            result = IterationResult(action=IterationAction.YIELD_AND_CONTINUE)
            result.add_chunk(
                StreamChunk(content="\n[recovery] Forcing summary after repeated empty responses\n")
            )
            # Add strong instruction to summarize
            self.message_adder.add_message(
                "user",
                "You seem to be stuck. Please provide a summary of what you have found so far. "
                "DO NOT call any more tools - just summarize the information you have already gathered.",
            )
            ctx.reset_empty_responses()
            ctx.force_completion = True
            return result
        return None

    def handle_blocked_tool_call(
        self,
        ctx: StreamingChatContext,
        tool_name: str,
        tool_args: Dict[str, Any],
        block_reason: str,
    ) -> StreamChunk:
        """Handle a blocked tool call by recording it and generating feedback.

        Args:
            ctx: The streaming context
            tool_name: Name of the blocked tool
            tool_args: Arguments that were passed
            block_reason: Reason the tool was blocked

        Returns:
            StreamChunk with block notification
        """
        ctx.record_tool_blocked()

        # Log detailed information about the blocked tool
        args_summary = ", ".join(
            f"{k}={repr(v)[:30]}" for k, v in tool_args.items() if k not in {"offset", "limit"}
        )
        logger.info(
            f"[dedup-block-exec] BLOCKED tool execution: {tool_name}({args_summary}) | "
            f"reason: {block_reason}"
        )

        logger.debug(f"BLOCKED tool call: {tool_name}({tool_args}) - {block_reason}")

        # Add tool result feedback
        stop_icon = self._presentation.icon("stop", with_color=False)
        args_summary = ", ".join(f"{k}={repr(v)[:30]}" for k, v in tool_args.items())
        self.message_adder.add_message(
            "user",
            f"{stop_icon} TOOL BLOCKED: {tool_name}({args_summary})\n\n"
            f"Reason: {block_reason}\n\n"
            "This operation was permanently blocked because you already tried it multiple times. "
            "You MUST use a DIFFERENT approach - this exact operation will NEVER work again.",
        )

        return StreamChunk(content=f"\n[loop] {stop_icon} {block_reason}\n")

    def filter_blocked_tool_calls(
        self,
        ctx: StreamingChatContext,
        tool_calls: List[Dict[str, Any]],
        block_checker: Callable[[str, Dict[str, Any]], Optional[str]],
    ) -> Tuple[List[Dict[str, Any]], List[StreamChunk], int]:
        """Filter out blocked tool calls and generate notification chunks.

        This method iterates through tool calls, checks each one against the
        block_checker function, and separates them into allowed and blocked.
        For blocked calls, it generates notification chunks using
        handle_blocked_tool_call.

        Args:
            ctx: The streaming context
            tool_calls: List of tool call dicts with 'name' and 'arguments'
            block_checker: Function that takes (tool_name, tool_args) and returns
                          block_reason string if blocked, None if allowed

        Returns:
            Tuple of:
            - filtered_tool_calls: List of tool calls that are NOT blocked
            - blocked_chunks: List of StreamChunk notifications for blocked tools
            - blocked_count: Number of tools that were blocked
        """
        filtered_tool_calls = []
        blocked_chunks = []
        blocked_count = 0

        for tc in tool_calls:
            tc_name = tc.get("name", "")
            tc_args = tc.get("arguments", {})
            block_reason = block_checker(tc_name, tc_args)

            if block_reason:
                # Use existing handler method to process blocked tool
                chunk = self.handle_blocked_tool_call(ctx, tc_name, tc_args, block_reason)
                blocked_chunks.append(chunk)
                blocked_count += 1
            else:
                filtered_tool_calls.append(tc)

        return filtered_tool_calls, blocked_chunks, blocked_count

    def check_force_action(
        self,
        ctx: StreamingChatContext,
        force_checker: Callable[[], Tuple[bool, Optional[str]]],
    ) -> Tuple[bool, Optional[str]]:
        """Check if action should be forced and update context accordingly.

        This method delegates to a force_checker function that determines
        whether to force action, then updates the context if needed.

        Args:
            ctx: The streaming context
            force_checker: Function that returns (should_force, hint_string)

        Returns:
            Tuple of (was_triggered, hint):
            - was_triggered: True if force_completion was newly set
            - hint: The hint string from the force_checker if triggered
        """
        should_force, hint = force_checker()

        if should_force and not ctx.force_completion:
            ctx.force_completion = True
            logger.info(f"Force action triggered: {hint}")
            return True, hint

        return False, None

    def check_blocked_threshold(
        self,
        ctx: StreamingChatContext,
        all_blocked: bool,
        consecutive_limit: int = DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD,
        total_limit: int = DEFAULT_BLOCKED_TOTAL_THRESHOLD,
    ) -> Optional[IterationResult]:
        """Check if blocked tool attempts have exceeded thresholds.

        Args:
            ctx: The streaming context
            all_blocked: Whether all tool calls in this iteration were blocked
            consecutive_limit: Threshold for consecutive blocked attempts
            total_limit: Threshold for total blocked attempts

        Returns:
            IterationResult if force completion triggered, None otherwise
        """
        if all_blocked:
            ctx.consecutive_blocked_attempts += 1
            if ctx.consecutive_blocked_attempts >= consecutive_limit:
                logger.warning(
                    f"Model stuck after {ctx.consecutive_blocked_attempts} consecutive "
                    f"blocked attempts (limit: {consecutive_limit}) - forcing completion"
                )
                return self._create_blocked_force_result(ctx)
        else:
            ctx.reset_blocked_attempts()

        if ctx.total_blocked_attempts >= total_limit:
            logger.warning(
                f"Model stuck after {ctx.total_blocked_attempts} total blocked attempts "
                f"(limit: {total_limit}) - forcing completion"
            )
            return self._create_blocked_force_result(ctx)

        return None

    def _create_blocked_force_result(self, ctx: StreamingChatContext) -> IterationResult:
        """Create a force completion result due to blocked attempts.

        Args:
            ctx: The streaming context

        Returns:
            IterationResult with force completion
        """
        result = IterationResult(action=IterationAction.YIELD_AND_CONTINUE)
        result.add_chunk(
            StreamChunk(
                content=f"\n[loop] {self._presentation.icon('warning', with_color=False)} Multiple blocked attempts - forcing completion\n"
            )
        )
        self.message_adder.add_message(
            "user",
            f"{self._presentation.icon('warning', with_color=False)} STOP: You have attempted blocked operations too many times. "
            "You MUST now provide your final response WITHOUT any tool calls. "
            "Summarize what you found and answer the user's question based on "
            "the information you have already gathered. DO NOT call any more tools.",
        )
        ctx.reset_blocked_attempts()
        # Signal that tool_calls should be cleared
        result.clear_tool_calls = True
        return result

    def handle_force_tool_execution(
        self,
        ctx: StreamingChatContext,
        mentioned_tools: List[str],
        force_message: Optional[str] = None,
    ) -> Optional[IterationResult]:
        """Handle forcing tool execution when model mentions tools without calling them.

        Args:
            ctx: The streaming context
            mentioned_tools: Tools that were mentioned but not executed
            force_message: Optional pre-crafted message to use instead of default

        Returns:
            IterationResult with appropriate action, None if not applicable
        """
        attempt_count = ctx.record_force_tool_attempt()

        if attempt_count >= 3:
            logger.warning(
                "Giving up on forced tool execution after 3 attempts - requesting summary"
            )
            self.message_adder.add_message(
                "user",
                "You are unable to make tool calls. Please provide your response "
                "NOW based on what you know. Do not mention any tools.",
            )
            ctx.reset_force_tool_attempts()
            return create_continue_result()

        # Use provided message or fall back to default
        if force_message:
            message = force_message
        else:
            tools_str = ", ".join(mentioned_tools)
            message = (
                f"You mentioned using {tools_str} but did not actually call the tool(s). "
                "Please make the actual tool call now, or provide your final answer without "
                "mentioning tools you cannot use."
            )

        self.message_adder.add_message(
            "user",
            message,
            metadata=build_internal_history_metadata("force_tool_execution"),
        )
        return create_continue_result()

    def check_tool_budget(
        self,
        ctx: StreamingChatContext,
        warning_threshold: int = 250,
        warning_pct: float = 0.8,
        warning_remaining: int = 5,
    ) -> Optional[IterationResult]:
        """Check tool budget and generate warning if approaching limit.

        Args:
            ctx: The streaming context
            warning_threshold: Absolute used-call threshold before warning
            warning_pct: Fraction of budget used before warning
            warning_remaining: Warn when this many or fewer calls remain

        Returns:
            IterationResult with warning chunk if approaching limit, None otherwise
        """
        if ctx.budget_warning_shown:
            return None

        if ctx.is_approaching_budget_limit(
            warning_threshold=warning_threshold,
            warning_pct=warning_pct,
            warning_remaining=warning_remaining,
        ):
            ctx.budget_warning_shown = True
            result = IterationResult(action=IterationAction.YIELD_AND_CONTINUE)
            remaining = ctx.get_remaining_budget()
            result.add_chunk(
                StreamChunk(
                    content=(
                        f"[tool] {self._presentation.icon('warning', with_color=False)} "
                        f"Approaching tool budget limit: {ctx.tool_calls_used}/{ctx.tool_budget} "
                        f"calls used ({remaining} remaining)\n"
                    )
                )
            )
            return result
        return None

    def check_budget_exhausted(self, ctx: StreamingChatContext) -> bool:
        """Check if tool budget is exhausted.

        Args:
            ctx: The streaming context

        Returns:
            True if budget exhausted, False otherwise
        """
        return ctx.is_budget_exhausted()

    def check_progress_and_force(
        self, ctx: StreamingChatContext, base_max_consecutive: int = 8
    ) -> bool:
        """Check progress and set force_completion if stuck.

        Args:
            ctx: The streaming context
            base_max_consecutive: Base limit for consecutive tool calls

        Returns:
            True if force_completion was set, False otherwise
        """
        if ctx.force_completion:
            return False  # Already forcing

        if not ctx.check_progress(base_max_consecutive):
            logger.warning(
                f"Forcing completion: {ctx.tool_calls_used} tool calls but only "
                f"{len(ctx.unique_resources)} unique resources"
            )
            ctx.force_completion = True
            return True
        return False

    def get_budget_exhausted_chunks(self, ctx: StreamingChatContext) -> List[StreamChunk]:
        """Generate chunks for budget exhausted state.

        Args:
            ctx: The streaming context

        Returns:
            List of StreamChunks for budget exhausted warning
        """
        return [
            StreamChunk(
                content=f"[tool] {self._presentation.icon('warning', with_color=False)} Tool budget reached ({ctx.tool_budget}); skipping tool calls.\n"
            ),
            StreamChunk(content="Generating final summary...\n"),
        ]

    def truncate_tool_calls(
        self, tool_calls: List[Dict[str, Any]], ctx: StreamingChatContext
    ) -> List[Dict[str, Any]]:
        """Truncate tool calls to fit remaining budget.

        Args:
            tool_calls: List of tool calls to truncate
            ctx: The streaming context

        Returns:
            Truncated list of tool calls
        """
        remaining = ctx.get_remaining_budget()
        if not tool_calls:
            return []
        return tool_calls[:remaining]

    def is_research_loop(self, stop_reason_value: str, stop_hint: str) -> bool:
        """Check if the stop reason indicates a research loop.

        Args:
            stop_reason_value: The string value of the stop reason enum
            stop_hint: The hint string from the stop decision

        Returns:
            True if this is a research loop, False otherwise
        """
        return stop_reason_value == "loop_detected" and "research" in stop_hint.lower()

    def get_force_completion_chunks(
        self, ctx: StreamingChatContext, is_research_loop: bool
    ) -> tuple[StreamChunk, str]:
        """Get the warning chunk and system message for force completion.

        Args:
            ctx: The streaming context
            is_research_loop: Whether this is a research loop

        Returns:
            Tuple of (warning_chunk, system_message)
        """
        if is_research_loop:
            warning_chunk = StreamChunk(
                content=f"[tool] {self._presentation.icon('warning', with_color=False)} Research loop detected - forcing synthesis\n"
            )
            system_message = (
                "You have performed multiple consecutive research/web searches. "
                "STOP searching now. Instead, SYNTHESIZE and ANALYZE the information you've already gathered. "
                "Provide your FINAL ANSWER based on the search results you have collected. "
                "Answer all parts of the user's question comprehensively."
            )
        else:
            warning_chunk = StreamChunk(
                content=f"{self._presentation.icon('warning', with_color=False)} Reached exploration limit - summarizing findings...\n"
            )
            system_message = (
                "You have made multiple tool calls without providing substantial analysis. "
                "STOP using tools now. Instead, provide your FINAL COMPREHENSIVE ANSWER based on "
                "the information you have already gathered. Answer all parts of the user's question."
            )
        return warning_chunk, system_message

    def handle_force_completion(
        self,
        ctx: StreamingChatContext,
        stop_reason_value: str,
        stop_hint: str,
    ) -> Optional[IterationResult]:
        """Handle force completion when the model is stuck.

        Args:
            ctx: The streaming context
            stop_reason_value: The string value of the stop reason enum
            stop_hint: The hint string from the stop decision

        Returns:
            IterationResult with warning chunk if force_completion is set, None otherwise
        """
        if not ctx.force_completion:
            return None

        # Only emit warning once to avoid spamming the user
        if ctx.force_completion_warning_shown:
            return None
        ctx.force_completion_warning_shown = True

        is_research = self.is_research_loop(stop_reason_value, stop_hint)
        warning_chunk, system_message = self.get_force_completion_chunks(ctx, is_research)

        # Add system message to force summary
        self.message_adder.add_message(
            "user",
            f"[SYSTEM: {system_message}]",
            metadata=build_internal_history_metadata("force_completion"),
        )

        result = IterationResult(action=IterationAction.YIELD_AND_BREAK)
        result.add_chunk(warning_chunk)
        return result

    def get_recovery_prompts(
        self,
        ctx: StreamingChatContext,
        base_temperature: float,
        has_thinking_mode: bool,
        thinking_disable_prefix: Optional[str] = None,
    ) -> List[tuple[str, float]]:
        """Generate recovery prompts for empty response recovery.

        This method generates a list of recovery prompts with varying temperatures
        based on the task type and model capabilities.

        Args:
            ctx: The streaming context
            base_temperature: Base temperature to use for recovery
            has_thinking_mode: Whether the model supports thinking mode
            thinking_disable_prefix: Optional prefix to disable thinking

        Returns:
            List of (prompt, temperature) tuples for recovery attempts
        """
        # Check if we should continue the task vs summarize
        has_budget_remaining = ctx.tool_calls_used < ctx.tool_budget * 0.8
        should_continue_task = (ctx.is_analysis_task or ctx.is_action_task) and has_budget_remaining

        def maybe_prefix(prompt: str) -> str:
            """Add thinking disable prefix if available."""
            if thinking_disable_prefix:
                return f"{thinking_disable_prefix}\n{prompt}"
            return prompt

        if should_continue_task:
            # Task-aware recovery: first attempt to continue, then fall back to summary
            if has_thinking_mode:
                return [
                    (
                        maybe_prefix(
                            "The previous action did not complete. Use discovery tools:\n"
                            "- graph(mode='pagerank', top_k=5) to find important symbols\n"
                            "- graph(mode='callers', node='symbol', depth=2) to find reverse call chains\n"
                            "- graph(mode='callees', node='symbol', depth=2) to find forward call chains\n"
                            "- graph(mode='trace', node='entry_function', depth=3) to trace execution flow\n"
                            "- search(query='...') for semantic code search\n"
                            "- overview(path='.') for project structure\n"
                            "Pick ONE tool to continue."
                        ),
                        escalate_temperature(base_temperature, 0.1, cap=0.7),
                    ),
                    (
                        maybe_prefix(
                            "Call a tool: search(query='main') or read(path='filename'). "
                            "Pick a file to examine."
                        ),
                        escalate_temperature(base_temperature, 0.2, cap=0.8),
                    ),
                    (
                        maybe_prefix(
                            "Respond in 2-3 sentences: What files did you read and what did you find?"
                        ),
                        escalate_temperature(base_temperature, 0.3, cap=0.9),
                    ),
                ]
            else:
                return [
                    (
                        "The previous action did not complete. Use discovery tools:\n"
                        "- graph(mode='pagerank', top_k=5) - find important symbols\n"
                        "- graph(mode='callers', node='symbol', depth=2) - find who calls a function\n"
                        "- graph(mode='callees', node='symbol', depth=2) - find what a function calls\n"
                        "- graph(mode='trace', node='entry_function', depth=3) - trace execution flow\n"
                        "- search(query='...') - semantic code search\n"
                        "- overview(path='.') - project structure\n"
                        "- refs(symbol_name='...') - find usages\n"
                        "Make ONE tool call to continue exploring.",
                        escalate_temperature(base_temperature, 0.2, cap=1.0),
                    ),
                    (
                        "Call search(query='main') or read(path='filename') to examine code. "
                        "Continue your analysis.",
                        escalate_temperature(base_temperature, 0.3, cap=1.0),
                    ),
                    (
                        "Summarize your findings so far. What files did you examine? "
                        "What patterns or issues did you notice? Keep it brief.",
                        escalate_temperature(base_temperature, 0.4, cap=1.0),
                    ),
                ]
        elif has_thinking_mode:
            # Simpler prompts and lower temps for thinking models (summary mode)
            return [
                (
                    maybe_prefix(
                        "Respond in 2-3 sentences: What files did you read and what did you find?"
                    ),
                    escalate_temperature(base_temperature, 0.1, cap=0.7),
                ),
                (
                    maybe_prefix("List 3 bullet points about the code you examined."),
                    escalate_temperature(base_temperature, 0.2, cap=0.8),
                ),
                (
                    maybe_prefix("One sentence answer: What is the main thing you learned?"),
                    escalate_temperature(base_temperature, 0.3, cap=0.9),
                ),
            ]
        else:
            # Standard recovery prompts (summary mode)
            return [
                (
                    "Summarize your findings so far. What files did you examine? "
                    "What patterns or issues did you notice? Keep it brief.",
                    escalate_temperature(base_temperature, 0.2, cap=1.0),
                ),
                (
                    "Based on the code you've seen, list 3-5 observations or suggestions.",
                    escalate_temperature(base_temperature, 0.3, cap=1.0),
                ),
                (
                    "What did you learn from the files? One paragraph summary.",
                    escalate_temperature(base_temperature, 0.4, cap=1.0),
                ),
            ]

    def should_use_tools_for_recovery(self, ctx: StreamingChatContext, attempt: int) -> bool:
        """Determine if tools should be enabled for a recovery attempt.

        Args:
            ctx: The streaming context
            attempt: The recovery attempt number (1-indexed)

        Returns:
            True if tools should be enabled, False otherwise
        """
        # For task-continuation mode, enable tools on first 2 attempts
        has_budget_remaining = ctx.tool_calls_used < ctx.tool_budget * 0.8
        should_continue_task = (ctx.is_analysis_task or ctx.is_action_task) and has_budget_remaining
        return should_continue_task and attempt <= 2

    def get_recovery_fallback_message(
        self, ctx: StreamingChatContext, unique_resources: List[str]
    ) -> str:
        """Generate a fallback message when all recovery attempts fail.

        Args:
            ctx: The streaming context
            unique_resources: List of unique resources examined

        Returns:
            Fallback message string
        """
        if ctx.is_analysis_task and ctx.tool_calls_used > 0:
            files_examined = unique_resources[:10]
            return (
                f"\n\n**Analysis Summary** (auto-generated)\n\n"
                f"Examined {len(unique_resources)} files including:\n"
                + "\n".join(f"- {f}" for f in files_examined)
                + "\n\nThe model was unable to provide detailed analysis. "
                "Try with a simpler query like 'analyze victor/agent/' or use a different model."
            )
        return (
            "No tool calls were returned and the model provided no content. "
            "Please retry or simplify the request."
        )

    def format_completion_metrics(
        self,
        ctx: StreamingChatContext,
        elapsed_time: float,
        cost_str: Optional[str] = None,
    ) -> str:
        """Format performance metrics for normal completion.

        This generates the detailed metrics line with cache info when available,
        or falls back to estimated tokens.

        Args:
            ctx: The streaming context
            elapsed_time: Elapsed time in seconds
            cost_str: Optional cost string to append (e.g., "$0.0123")

        Returns:
            Formatted metrics line string
        """
        # Use actual provider-reported usage if available, else use estimate
        if ctx.cumulative_usage.get("total_tokens", 0) > 0:
            # Provider-reported tokens (accurate)
            input_tokens = ctx.cumulative_usage["prompt_tokens"]
            output_tokens = ctx.cumulative_usage["completion_tokens"]
            cache_read = ctx.cumulative_usage.get("cache_read_input_tokens", 0)
            cache_create = ctx.cumulative_usage.get("cache_creation_input_tokens", 0)

            # Build metrics line
            tokens_per_second = output_tokens / elapsed_time if elapsed_time > 0 else 0
            metrics_parts = [
                f"{self._presentation.icon('chart', with_color=False)} in={input_tokens:,}",
                f"out={output_tokens:,}",
            ]
            if cache_read > 0:
                metrics_parts.append(f"cached={cache_read:,}")
            if cache_create > 0:
                metrics_parts.append(f"cache_new={cache_create:,}")
            metrics_parts.extend(
                [
                    f"| {elapsed_time:.1f}s",
                    f"| {tokens_per_second:.1f} tok/s",
                ]
            )
            # Append cost if provided
            if cost_str:
                metrics_parts.append(f"| {cost_str}")
            return " ".join(metrics_parts)
        else:
            # Fallback to estimate
            tokens_per_second = (
                ctx.estimated_content_tokens / elapsed_time if elapsed_time > 0 else 0
            )
            base = (
                f"{self._presentation.icon('chart', with_color=False)} ~{ctx.estimated_content_tokens:.0f} tokens (est.) | "
                f"{elapsed_time:.1f}s | {tokens_per_second:.1f} tok/s"
            )
            if cost_str:
                base += f" | {cost_str}"
            return base

    def format_budget_exhausted_metrics(
        self,
        ctx: StreamingChatContext,
        elapsed_time: float,
        time_to_first_token: Optional[float] = None,
        cost_str: Optional[str] = None,
    ) -> str:
        """Format performance metrics for budget exhausted completion.

        This generates a simpler metrics line with optional TTFT info.

        Args:
            ctx: The streaming context
            elapsed_time: Elapsed time in seconds
            time_to_first_token: Optional time to first token
            cost_str: Optional cost string to append (e.g., "$0.0123")

        Returns:
            Formatted metrics line string
        """
        tokens_per_second = ctx.estimated_content_tokens / elapsed_time if elapsed_time > 0 else 0
        ttft_info = ""
        if time_to_first_token:
            ttft_info = f" | TTFT: {time_to_first_token:.2f}s"
        cost_info = ""
        if cost_str:
            cost_info = f" | {cost_str}"
        return f"{self._presentation.icon('chart', with_color=False)} {ctx.estimated_content_tokens:.0f} tokens | {elapsed_time:.1f}s | {tokens_per_second:.1f} tok/s{ttft_info}{cost_info}"

    def generate_tool_result_chunk(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        elapsed: float,
        success: bool,
        error: Optional[str] = None,
        follow_up_suggestions: Optional[List[Dict[str, Any]]] = None,
        was_pruned: bool = False,
        result: Any = None,  # Tool output for preview
        original_result: Any = None,  # Full tool output for expansion/debug
        tool_call_id: Optional[str] = None,
    ) -> StreamChunk:
        """Generate a StreamChunk for a tool result.

        Args:
            tool_name: Name of the tool
            tool_args: Arguments passed to the tool
            elapsed: Time elapsed for tool execution
            success: Whether the tool succeeded
            error: Optional error message if failed
            follow_up_suggestions: Optional follow-up suggestions
            result: Tool output (for preview display)

        Returns:
            StreamChunk with tool_result metadata
        """
        metadata: Dict[str, Any] = {
            "tool_result": {
                "name": tool_name,
                "success": success,
                "elapsed": elapsed,
                "arguments": tool_args,
            }
        }
        if tool_call_id:
            metadata["tool_result"]["tool_call_id"] = tool_call_id
        if not success and error:
            metadata["tool_result"]["error"] = error
        if follow_up_suggestions:
            metadata["tool_result"]["follow_up_suggestions"] = follow_up_suggestions
        if was_pruned:
            metadata["tool_result"]["was_pruned"] = True
        if result is not None:
            # Pass the real (already pruned) output through — how much of it
            # to show, and the /expand affordance, are presentation decisions
            # owned by the rendering layer.
            metadata["tool_result"]["result"] = str(result)
        if original_result is not None:
            metadata["tool_result"]["original_result"] = str(original_result)
        return StreamChunk(content="", metadata=metadata)

    def generate_file_preview_chunk(
        self,
        content: str,
        path: str,
        preview_lines: int = 8,
    ) -> Optional[StreamChunk]:
        """Generate a file preview chunk for write_file operations.

        Args:
            content: The file content to preview
            path: The file path
            preview_lines: Number of lines to show in preview

        Returns:
            StreamChunk with file_preview metadata, or None if no content
        """
        if not content:
            return None

        lines = content.split("\n")
        if len(lines) > preview_lines:
            preview = "\n".join(lines[:preview_lines])
            preview += f"\n... ({len(lines) - preview_lines} more lines)"
        else:
            preview = content

        return StreamChunk(
            content="",
            metadata={
                "file_preview": preview,
                "path": path,
            },
        )

    def generate_edit_preview_chunk(
        self,
        old_string: str,
        new_string: str,
        path: str,
        context_lines: int = 2,
        max_diff_lines: int = 40,
    ) -> Optional[StreamChunk]:
        """Generate an edit preview chunk as a real unified diff.

        Args:
            old_string: The old string being replaced (empty for creates)
            new_string: The new string (empty for deletes)
            path: The file path
            context_lines: Unified diff context lines (n)
            max_diff_lines: Cap on rendered diff lines to keep output compact

        Returns:
            StreamChunk with edit_preview metadata, or None if both sides empty
        """
        if not old_string and not new_string:
            return None

        import difflib

        old_lines = old_string.splitlines(keepends=False) if old_string else []
        new_lines = new_string.splitlines(keepends=False) if new_string else []

        diff_lines = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
                n=context_lines,
            )
        )

        if not diff_lines:
            # Strings differ but unified_diff produced nothing (e.g. identical
            # after newline normalization). Show a minimal hint instead of None
            # so the user still sees the file was touched.
            diff_lines = [f"--- a/{path}", f"+++ b/{path}", "(no textual changes)"]

        if len(diff_lines) > max_diff_lines:
            diff_lines = diff_lines[:max_diff_lines] + [
                f"... ({len(diff_lines) - max_diff_lines} more lines)"
            ]

        return StreamChunk(
            content="",
            metadata={
                "edit_preview": "\n".join(diff_lines),
                "path": path,
            },
        )

    def generate_tool_result_chunks(
        self,
        result: Dict[str, Any],
        max_files: int = 3,
        max_edits_per_file: int = 2,
    ) -> List[StreamChunk]:
        """Generate all chunks for a tool result including previews.

        This method generates the main tool_result chunk plus any file or
        edit preview chunks for write_file/edit_files operations.

        Args:
            result: The tool execution result dictionary
            max_files: Maximum number of files to show for edit_files
            max_edits_per_file: Maximum edits per file to preview

        Returns:
            List of StreamChunks for the tool result
        """
        chunks: List[StreamChunk] = []
        tool_name = result.get("name", "tool")
        elapsed = result.get("elapsed", 0.0)
        tool_args = result.get("args", {})
        success = result.get("success", False)
        error = result.get("error") if not success else None
        follow_up_suggestions = result.get("follow_up_suggestions")
        was_pruned = bool(result.get("was_pruned", False))

        # Main tool result chunk
        tool_output = result.get("result") or result.get("content")
        full_tool_output = result.get("full_result") or result.get("content") or tool_output
        chunks.append(
            self.generate_tool_result_chunk(
                tool_name,
                tool_args,
                elapsed,
                success,
                error,
                follow_up_suggestions=follow_up_suggestions,
                was_pruned=was_pruned,
                result=tool_output,  # Pass result for preview
                # Always carry the full output so non-CLI surfaces (web) can render
                # the real result instead of the CLI "/expand" placeholder. The Rich
                # CLI still shows the placeholder inline and uses this only for /expand.
                original_result=full_tool_output,
                tool_call_id=result.get("tool_call_id"),
            )
        )

        # Generate preview chunks for successful write/edit operations
        if success:
            canonical_tool_name = canonicalize_core_tool_name(tool_name)

            if canonical_tool_name == "write" and tool_args.get("content"):
                preview_chunk = self.generate_file_preview_chunk(
                    tool_args["content"],
                    tool_args.get("path", ""),
                )
                if preview_chunk:
                    chunks.append(preview_chunk)
                    self._record_preview_message(
                        "File preview",
                        str(preview_chunk.metadata.get("path", "")),
                        str(preview_chunk.metadata.get("file_preview", "")),
                    )

            elif canonical_tool_name == "edit":
                ops = tool_args.get("ops", [])
                if isinstance(ops, list) and ops:
                    preview_count = 0
                    for op in ops:
                        if not isinstance(op, dict):
                            continue
                        op_type = op.get("type")
                        path = op.get("path", "")
                        if op_type == "replace":
                            edit_chunk = self.generate_edit_preview_chunk(
                                op.get("old_str", ""),
                                op.get("new_str", ""),
                                path,
                            )
                            if edit_chunk:
                                chunks.append(edit_chunk)
                                self._record_preview_message(
                                    "Edit preview",
                                    str(edit_chunk.metadata.get("path", "")),
                                    str(edit_chunk.metadata.get("edit_preview", "")),
                                )
                                preview_count += 1
                        elif op_type == "create" and op.get("content"):
                            # New file — show as full-add unified diff
                            edit_chunk = self.generate_edit_preview_chunk(
                                "", op.get("content", ""), path
                            )
                            if edit_chunk:
                                chunks.append(edit_chunk)
                                self._record_preview_message(
                                    "Edit preview",
                                    str(edit_chunk.metadata.get("path", "")),
                                    str(edit_chunk.metadata.get("edit_preview", "")),
                                )
                                preview_count += 1
                        elif op_type == "modify" and op.get("content"):
                            preview_chunk = self.generate_file_preview_chunk(
                                op.get("content", ""),
                                path,
                            )
                            if preview_chunk:
                                chunks.append(preview_chunk)
                                self._record_preview_message(
                                    "File preview",
                                    str(preview_chunk.metadata.get("path", "")),
                                    str(preview_chunk.metadata.get("file_preview", "")),
                                )
                                preview_count += 1
                        elif op_type == "delete":
                            # Deletion — surface the action with a one-line hint
                            chunks.append(
                                StreamChunk(
                                    content="",
                                    metadata={
                                        "edit_preview": (
                                            f"--- a/{path}\n+++ /dev/null\n" f"(file deleted)"
                                        ),
                                        "path": path,
                                    },
                                )
                            )
                            preview_count += 1
                        elif op_type == "rename":
                            new_path = op.get("new_path", "")
                            chunks.append(
                                StreamChunk(
                                    content="",
                                    metadata={
                                        "edit_preview": (
                                            f"--- a/{path}\n+++ b/{new_path}\n" f"(renamed)"
                                        ),
                                        "path": new_path or path,
                                    },
                                )
                            )
                            preview_count += 1
                        if preview_count >= max_files * max_edits_per_file:
                            break
                elif tool_args.get("files"):
                    files = tool_args.get("files", [])
                    for file_edit in files[:max_files]:
                        path = file_edit.get("path", "")
                        edits = file_edit.get("edits", [])
                        for edit in edits[:max_edits_per_file]:
                            old_str = edit.get("old_string", "")
                            new_str = edit.get("new_string", "")
                            edit_chunk = self.generate_edit_preview_chunk(old_str, new_str, path)
                            if edit_chunk:
                                chunks.append(edit_chunk)
                                self._record_preview_message(
                                    "Edit preview",
                                    str(edit_chunk.metadata.get("path", "")),
                                    str(edit_chunk.metadata.get("edit_preview", "")),
                                )

        return chunks

    def get_loop_warning_chunks(
        self,
        warning_message: str,
    ) -> tuple[StreamChunk, str]:
        """Get the warning chunk and system message for loop detection.

        Args:
            warning_message: The warning message from unified tracker

        Returns:
            Tuple of (warning_chunk, system_message)
        """
        warning_chunk = StreamChunk(
            content=f"\n[loop] {self._presentation.icon('warning', with_color=False)} Warning: Approaching loop limit - {warning_message}\n"
        )
        system_message = (
            "WARNING: You are about to hit loop detection. You have been performing "
            "the same operation repeatedly (e.g., writing the same file, making the same call). "
            "Please do something DIFFERENT now:\n"
            "- If you're writing a file repeatedly, STOP and move to a different task\n"
            "- If you're stuck, provide your current progress and ask for clarification\n"
            "- If you've completed the task, provide a summary and finish\n\n"
            "Continuing the same operation will force the conversation to end."
        )
        return warning_chunk, system_message

    def handle_loop_warning(
        self,
        ctx: StreamingChatContext,
        warning_message: str,
    ) -> Optional[StreamChunk]:
        """Handle loop warning by generating warning chunk and adding system message.

        Args:
            ctx: The streaming context
            warning_message: The warning message from unified tracker

        Returns:
            StreamChunk with warning if warning_message is set, None otherwise
        """
        if not warning_message or ctx.force_completion:
            return None

        warning_chunk, system_message = self.get_loop_warning_chunks(warning_message)
        self.message_adder.add_message(
            "user",
            f"[SYSTEM: {system_message}]",
            metadata=build_internal_history_metadata("force_completion"),
        )
        return warning_chunk

    def generate_thinking_status_chunk(self) -> StreamChunk:
        """Generate a thinking status chunk to indicate model is processing.

        Returns:
            StreamChunk with thinking status metadata
        """
        thinking_icon = self._presentation.icon("thinking", with_color=False)
        return StreamChunk(content="", metadata={"status": f"{thinking_icon} Thinking..."})

    def generate_budget_error_chunk(self) -> StreamChunk:
        """Generate a chunk for budget limit summary error.

        Returns:
            StreamChunk with budget limit error message
        """
        return StreamChunk(content="Unable to generate summary due to budget limit.\n")

    def generate_force_response_error_chunk(self) -> StreamChunk:
        """Generate a chunk for forced response generation error.

        Returns:
            StreamChunk with force response error message
        """
        return StreamChunk(content="Unable to generate final summary. Please try a simpler query.")

    def handle_iteration_start(
        self,
        ctx: StreamingChatContext,
    ) -> Optional[IterationResult]:
        """Handle the start of an iteration.

        Performs all pre-iteration checks in priority order.

        Args:
            ctx: The streaming context.

        Returns:
            IterationResult if iteration should be skipped/stopped.
        """
        # Time limit check
        time_result = self.check_time_limit(ctx)
        if time_result is not None:
            return time_result

        # Iteration limit check
        iter_result = self.check_iteration_limit(ctx)
        if iter_result is not None:
            return iter_result

        # Force completion check
        force_result = self.check_force_completion(ctx)
        if force_result is not None:
            return force_result

        return None

    def handle_iteration_end(
        self,
        ctx: StreamingChatContext,
        result: IterationResult,
        stop_reason_value: str = "",
        stop_hint: str = "",
    ) -> Optional[IterationResult]:
        """Handle the end of an iteration.

        Performs all post-iteration checks and determines continuation.

        Args:
            ctx: The streaming context.
            result: The iteration result.
            stop_reason_value: Optional stop reason from loop detection.
            stop_hint: Optional stop hint from loop detection.

        Returns:
            IterationResult if additional action needed.
        """
        # Handle force completion with warning
        if ctx.force_completion:
            force_result = self.handle_force_completion(ctx, stop_reason_value, stop_hint)
            if force_result is not None:
                return force_result

        # Budget warning check
        budget_result = self.check_tool_budget(ctx)
        if budget_result is not None:
            return budget_result

        return None

    def handle_continuation(
        self,
        ctx: StreamingChatContext,
        has_tool_calls: bool,
        has_content: bool,
    ) -> Optional[IterationResult]:
        """Determine if loop should continue based on response.

        This method encapsulates the continuation decision logic.

        Args:
            ctx: The streaming context.
            has_tool_calls: Whether response has tool calls.
            has_content: Whether response has content.

        Returns:
            IterationResult if loop should stop, None to continue.
        """
        # With tool calls, continue to execute them
        if has_tool_calls:
            return None

        # Force completion triggers break
        if ctx.force_completion:
            if has_content:
                return create_break_result()
            # No content - will trigger recovery

        # Check for natural completion
        if has_content and ctx.has_substantial_content():
            return self.check_natural_completion(ctx, has_tool_calls, len(ctx.context_msg))

        # Handle empty response
        if not has_content:
            return self.handle_empty_response(ctx)

        return None

    def generate_final_marker_chunk(self) -> StreamChunk:
        """Generate an empty final marker chunk.

        This chunk signals the end of the streaming response.

        Returns:
            StreamChunk with is_final=True and empty content
        """
        return StreamChunk(content="", is_final=True)

    def generate_metrics_chunk(
        self, metrics_line: str, is_final: bool = False, prefix: str = "\n\n"
    ) -> StreamChunk:
        """Generate a metrics display chunk.

        Args:
            metrics_line: The formatted metrics line
            is_final: Whether this is the final chunk
            prefix: Prefix before metrics line (default: double newline)

        Returns:
            StreamChunk with formatted metrics content
        """
        return StreamChunk(
            content=f"{prefix}{metrics_line}\n",
            is_final=is_final,
        )

    def generate_content_chunk(
        self, content: str, is_final: bool = False, suffix: str = ""
    ) -> StreamChunk:
        """Generate a content chunk with optional suffix.

        Args:
            content: The sanitized content to display
            is_final: Whether this is the final chunk
            suffix: Optional suffix to append (e.g., newline)

        Returns:
            StreamChunk with content and optional suffix
        """
        return StreamChunk(
            content=f"{content}{suffix}",
            is_final=is_final,
        )


def create_streaming_handler(
    settings: "Settings",
    orchestrator: "AgentOrchestrator",
    session_idle_timeout: Optional[float] = None,
) -> StreamingChatHandler:
    """Factory function to create a StreamingChatHandler.

    Args:
        settings: Application settings
        orchestrator: The orchestrator (used as message adder)
        session_idle_timeout: Optional idle timeout override (seconds)

    Returns:
        Configured StreamingChatHandler
    """
    limit = session_idle_timeout or getattr(settings, "session_idle_timeout", 180.0)
    # Get presentation adapter from orchestrator if available
    presentation = getattr(orchestrator, "_presentation", None)
    return StreamingChatHandler(
        settings=settings,
        message_adder=orchestrator,
        tool_executor=None,  # Tool execution stays in orchestrator for now
        session_idle_timeout=limit,
        presentation=presentation,
    )
