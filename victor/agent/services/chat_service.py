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

"""Chat service implementation.

Extracts chat flow coordination from the AgentOrchestrator into
a focused, single-responsibility service following SOLID principles.

This service handles:
- Chat flow coordination
- Streaming response processing
- Response aggregation
- Integration with tool, context, provider, and recovery services
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, List, Optional

from victor.agent.services.chat_evidence import ChatEvidenceMixin

if TYPE_CHECKING:
    from victor.agent.services.protocols import (
        ToolServiceProtocol,
        ContextServiceProtocol,
        ProviderServiceProtocol,
        RecoveryServiceProtocol,
    )
    from victor.providers.base import CompletionResponse, StreamChunk

logger = logging.getLogger(__name__)


class ChatServiceConfig:
    """Configuration for ChatService.

    Attributes:
        max_iterations: Maximum agentic loop iterations
        max_continuation_prompts: Maximum continuation prompts
        stream_chunk_size: Size of stream chunks
        enable_response_caching: Enable response caching
    """

    def __init__(
        self,
        max_iterations: int = 200,
        max_continuation_prompts: int = 3,
        stream_chunk_size: int = 100,
        enable_response_caching: bool = True,
    ):
        self.max_iterations = max_iterations
        self.max_continuation_prompts = max_continuation_prompts
        self.stream_chunk_size = stream_chunk_size
        self.enable_response_caching = enable_response_caching


class ChatService(ChatEvidenceMixin):
    """[CANONICAL] Service for managing chat operations.

    The target implementation for chat operations following the
    state-passed architectural pattern.

    This service follows SOLID principles:
    - SRP: Only handles chat operations
    - OCP: Extensible through composition
    - LSP: Implements ChatServiceProtocol
    - ISP: Focused interface
    - DIP: Depends on service protocols, not concretions

    Example:
        config = ChatServiceConfig()
        service = ChatService(
            config=config,
            provider_service=provider_service,
            tool_service=tool_service,
            context_service=context_service,
            recovery_service=recovery_service,
            conversation_controller=conversation,
            streaming_coordinator=streaming,
        )

        response = await service.chat("Hello, world!")
    """

    def __init__(
        self,
        config: ChatServiceConfig,
        provider_service: "ProviderServiceProtocol",
        tool_service: "ToolServiceProtocol",
        context_service: "ContextServiceProtocol",
        recovery_service: "RecoveryServiceProtocol",
        conversation_controller: Any,
        streaming_coordinator: Any,
    ):
        """Initialize the chat service.

        Args:
            config: Service configuration
            provider_service: Provider management service
            tool_service: Tool operations service
            context_service: Context management service
            recovery_service: Error recovery service
            conversation_controller: Conversation state controller
            streaming_coordinator: Streaming response coordinator
        """
        self._config = config
        self._provider = provider_service
        self._tools = tool_service
        self._context = context_service
        self._recovery = recovery_service
        self._conversation = conversation_controller
        self._streaming = streaming_coordinator
        self._logger = logging.getLogger(f"{__name__}.{id(self)}")
        self._turn_executor: Optional[Any] = None
        self._planning_handler: Optional[Callable[[str], Any]] = None
        self._stream_chat_handler: Optional[Callable[..., AsyncIterator[Any]]] = None
        self._context_limit_handler: Optional[Callable[..., Any]] = None
        self._task_report_start_handler: Optional[Callable[..., Any]] = None
        self._task_report_finish_handler: Optional[Callable[..., Any]] = None
        self._turn_setup_handler: Optional[Callable[..., Any]] = None
        self._turn_teardown_handler: Optional[Callable[..., Any]] = None
        self._context_accepts_keyword_messages: Optional[bool] = None

        # Initialize metrics tracking
        self._metrics: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "tool_calls": 0,
            "continuations": 0,
            "response_times": [],
            "latencies": [],
            "total_streams": 0,
            "average_chunks": 0,
            "avg_chunk_size": 0,
            "chunks_per_second": 0,
            "total_chunks": 0,
        }

    def bind_runtime_components(
        self,
        *,
        turn_executor: Optional[Any] = None,
        planning_handler: Optional[Callable[[str], Any]] = None,
        stream_chat_handler: Optional[Callable[..., AsyncIterator[Any]]] = None,
        context_limit_handler: Optional[Callable[..., Any]] = None,
        task_report_start_handler: Optional[Callable[..., Any]] = None,
        task_report_finish_handler: Optional[Callable[..., Any]] = None,
        turn_setup_handler: Optional[Callable[..., Any]] = None,
        turn_teardown_handler: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Bind live runtime collaborators after bootstrap."""
        if turn_executor is not None:
            self._turn_executor = turn_executor
        if planning_handler is not None:
            self._planning_handler = planning_handler
        if stream_chat_handler is not None:
            self._stream_chat_handler = stream_chat_handler
        if context_limit_handler is not None:
            self._context_limit_handler = context_limit_handler
        if task_report_start_handler is not None:
            self._task_report_start_handler = task_report_start_handler
        if task_report_finish_handler is not None:
            self._task_report_finish_handler = task_report_finish_handler
        if turn_setup_handler is not None:
            self._turn_setup_handler = turn_setup_handler
        if turn_teardown_handler is not None:
            self._turn_teardown_handler = turn_teardown_handler

    async def chat(
        self, user_message: str, *, stream: bool = False, **kwargs
    ) -> "CompletionResponse":
        """Process a chat message through the bound canonical runtime.

        Args:
            user_message: The user's input message
            stream: If True, aggregate the bound streaming runtime
            **kwargs: Runtime options, including planning controls

        Returns:
            CompletionResponse with the generated response

        Raises:
            RuntimeError: If the canonical runtime collaborators are not bound
        """
        use_planning = kwargs.pop("use_planning", False)
        preserve_turn_state = kwargs.pop("_preserve_turn_state", False)
        constraints = kwargs.pop("constraints", None)
        vertical = kwargs.pop("vertical", None)
        runtime_context_overrides = kwargs.pop("runtime_context_overrides", None)

        if not preserve_turn_state:
            self._prepare_new_turn()

        if stream:
            chunks = []
            async for chunk in self.stream_chat(
                user_message,
                use_planning=use_planning,
                _preserve_turn_state=True,
                constraints=constraints,
                vertical=vertical,
                **kwargs,
            ):
                chunks.append(chunk)
            return self._aggregate_chunks(chunks)

        try:
            await self._enter_turn_scope(
                user_message=user_message,
                stream=False,
                constraints=constraints,
                vertical=vertical,
            )
            await self._start_task_report(
                user_message,
                stream=False,
                metadata={"use_planning": use_planning},
            )

            response: Optional["CompletionResponse"] = None
            try:
                if self._planning_handler is not None and (
                    use_planning is True or use_planning is None
                ):
                    response = await self._planning_handler(user_message)
                else:
                    executor = self._turn_executor
                    if executor is None or not hasattr(executor, "execute_agentic_loop"):
                        raise RuntimeError(
                            "ChatService runtime is not bound: "
                            "turn_executor.execute_agentic_loop is required for chat()."
                        )
                    response = await self._execute_agentic_loop(
                        executor,
                        user_message,
                        runtime_context_overrides=runtime_context_overrides,
                    )
            except Exception as exc:
                await self._finish_task_report(
                    False,
                    user_message=user_message,
                    stream=False,
                    error=exc,
                    metadata={"use_planning": use_planning},
                )
                raise

            await self._finish_task_report(
                self._response_execution_success(response),
                user_message=user_message,
                stream=False,
                response=response,
                metadata={"use_planning": use_planning},
            )
            return response
        finally:
            await self._exit_turn_scope(
                user_message=user_message,
                stream=False,
                constraints=constraints,
                vertical=vertical,
            )

    async def _execute_agentic_loop(
        self,
        executor: Any,
        user_message: str,
        *,
        runtime_context_overrides: Optional[Dict[str, Any]] = None,
    ) -> "CompletionResponse":
        """Invoke the bound turn executor with scoped runtime overrides when supported."""
        if not runtime_context_overrides:
            return await executor.execute_agentic_loop(user_message)

        execute = executor.execute_agentic_loop
        try:
            signature = inspect.signature(execute)
        except (TypeError, ValueError):
            signature = None

        if signature is not None:
            parameters = signature.parameters
            accepts_overrides = "runtime_context_overrides" in parameters or any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
            )
            if not accepts_overrides:
                return await execute(user_message)

        return await execute(
            user_message,
            runtime_context_overrides=runtime_context_overrides,
        )

    async def stream_chat(self, user_message: str, **kwargs) -> AsyncIterator["StreamChunk"]:
        """Stream a chat response through the bound canonical runtime.

        Args:
            user_message: The user's input message
            **kwargs: Additional runtime options

        Yields:
            StreamChunk objects with incremental response content

        Raises:
            RuntimeError: If the canonical streaming runtime is not bound
        """
        from victor.providers.base import StreamChunk

        use_planning = kwargs.pop("use_planning", None)
        preserve_turn_state = kwargs.pop("_preserve_turn_state", False)
        constraints = kwargs.pop("constraints", None)
        vertical = kwargs.pop("vertical", None)
        if not preserve_turn_state:
            self._prepare_new_turn()

        try:
            await self._enter_turn_scope(
                user_message=user_message,
                stream=True,
                constraints=constraints,
                vertical=vertical,
            )
            await self._start_task_report(
                user_message,
                stream=True,
                metadata={"use_planning": use_planning},
            )

            response: Optional[Any] = None
            finished = False
            failure: Optional[BaseException] = None
            try:
                if self._planning_handler is not None:
                    explicit_planning_request = self._is_explicit_planning_request(user_message)
                    should_plan = use_planning is True or (
                        use_planning is None and explicit_planning_request
                    )
                    if should_plan:
                        response = await self._planning_handler(user_message)
                        if isinstance(response, dict):
                            content = str(response.get("content", "") or "")
                        else:
                            content = str(getattr(response, "content", response) or "")
                        finished = True
                        yield StreamChunk(content=content, is_final=True)
                        return
                    if use_planning is None and self._should_use_planning(user_message):
                        self._logger.info(
                            "Streaming auto-planning suppressed to preserve live stream output; "
                            "falling back to canonical stream runtime."
                        )

                handler = self._stream_chat_handler
                if handler is None:
                    raise RuntimeError(
                        "ChatService runtime is not bound: "
                        "stream_chat_handler is required for stream_chat()."
                    )

                async for chunk in handler(user_message, **kwargs):
                    response = chunk
                    yield chunk
                finished = True
            except Exception as exc:
                failure = exc
                raise
            finally:
                await self._finish_task_report(
                    finished and failure is None and self._response_execution_success(response),
                    user_message=user_message,
                    stream=True,
                    response=response,
                    error=failure,
                    metadata={"use_planning": use_planning},
                )
        finally:
            await self._exit_turn_scope(
                user_message=user_message,
                stream=True,
                constraints=constraints,
                vertical=vertical,
            )

    async def chat_with_planning(
        self,
        user_message: str,
        use_planning: Optional[bool] = None,
    ) -> "CompletionResponse":
        """Process chat with planning enabled or auto-detected."""
        if use_planning is False:
            return await self.chat(user_message, use_planning=False)

        if self._planning_handler is not None and (use_planning is True or use_planning is None):
            return await self._planning_handler(user_message)

        return await self.chat(user_message, use_planning=use_planning)

    async def handle_context_and_iteration_limits(self, *args, **kwargs) -> Any:
        """Delegate context/iteration limit handling to the bound runtime helper."""
        if self._context_limit_handler is None:
            return False, None
        return await self._context_limit_handler(*args, **kwargs)

    def reset_conversation(self) -> None:
        """Reset the conversation history and state.

        Clears all conversation context, effectively starting
        a new conversation session.
        """
        self._logger.debug("Resetting conversation")
        self._last_task_report = None
        self._context.clear_messages(retain_system=True)
        manages_controller = getattr(self._context, "manages_conversation_controller", None)
        if callable(manages_controller) and manages_controller(self._conversation):
            return
        self._conversation.reset()

    def is_healthy(self) -> bool:
        """Check if the chat service is healthy.

        A healthy chat service requires:
        - Valid provider connection
        - Tool service available
        - Context service available

        Returns:
            True if the service is healthy, False otherwise
        """
        if not self._provider.is_healthy():
            self._logger.warning("Provider service is unhealthy")
            return False

        if not self._tools.is_healthy():
            self._logger.warning("Tool service is unhealthy")
            return False

        if not self._context.is_healthy():
            self._logger.warning("Context service is unhealthy")
            return False

        return True

    # ==========================================================================
    # Conversation Query Methods
    # ==========================================================================

    def get_message_count(self) -> int:
        """Get the number of messages in the conversation.

        Returns the total count of messages (excluding system messages)
        in the current conversation history.

        Returns:
            Number of messages in conversation

        Example:
            count = service.get_message_count()
            # Returns: 15
        """
        if hasattr(self._context, "message_count"):
            return self._context.message_count
        elif hasattr(self._context, "get_message_count"):
            return self._context.get_message_count()
        elif hasattr(self._conversation, "message_count"):
            return self._conversation.message_count
        return 0

    def is_conversation_empty(self) -> bool:
        """Check if the conversation has no messages.

        Returns True if there are no user or assistant messages
        (system messages are ignored).

        Returns:
            True if conversation is empty, False otherwise

        Example:
            if service.is_conversation_empty():
                # Start fresh conversation
        """
        return self.get_message_count() == 0

    def get_conversation_stats(self) -> Dict[str, int]:
        """Get conversation statistics.

        Returns metadata about the current conversation including
        message count, token usage, and tool call count.

        Returns:
            Dictionary with conversation statistics

        Example:
            stats = service.get_conversation_stats()
            # {"message_count": 10, "user_messages": 3, "assistant_messages": 5, ...}
        """
        stats = {
            "message_count": self.get_message_count(),
            "is_empty": self.is_conversation_empty(),
        }

        # Add token usage if available
        if hasattr(self._context, "get_token_count"):
            try:
                stats["token_count"] = self._context.get_token_count()
            except Exception:
                pass

        # Add tool call count if available
        if hasattr(self._tools, "get_tool_usage_stats"):
            try:
                usage_stats = self._tools.get_tool_usage_stats()
                stats["tool_calls"] = sum(usage_stats.values())
            except Exception:
                pass

        return stats

    def get_conversation_metadata(self) -> Dict[str, Any]:
        """Get conversation metadata.

        Returns high-level metadata about the conversation including
        model, provider, and configuration info.

        Returns:
            Dictionary with conversation metadata

        Example:
            metadata = service.get_conversation_metadata()
            # {"model": "gpt-4o", "provider": "openai", "max_iterations": 200}
        """
        metadata = {
            "max_iterations": self._config.max_iterations,
            "max_continuation_prompts": self._config.max_continuation_prompts,
            "is_healthy": self.is_healthy(),
        }

        # Add provider info if available
        if hasattr(self._provider, "get_current_provider_info"):
            try:
                provider_info = self._provider.get_current_provider_info()
                if hasattr(provider_info, "model_name"):
                    metadata["model"] = provider_info.model_name
                if hasattr(provider_info, "provider_name"):
                    metadata["provider"] = provider_info.provider_name
            except Exception:
                pass

        return metadata

    # ==========================================================================
    # Private Methods
    # ==========================================================================

    async def _run_agentic_loop(self, user_message: str, **kwargs) -> "CompletionResponse":
        """Legacy ChatService-owned loop removed in favor of bound runtimes."""
        raise RuntimeError(
            "ChatService no longer owns an internal agentic loop. "
            "Bind a turn_executor via bind_runtime_components()."
        )

    # NOTE: Orchestrator now wires the canonical service/runtime path directly.
    # ServiceStreamingRuntime owns the streaming executor + AgenticLoop
    # integration for perception, fulfillment, and progress tracking.

    async def _enter_turn_scope(
        self,
        *,
        user_message: str,
        stream: bool,
        constraints: Any = None,
        vertical: Optional[str] = None,
    ) -> None:
        """Run bound per-turn setup before task execution starts."""
        await self._run_optional_callback(
            self._turn_setup_handler,
            user_message,
            stream=stream,
            constraints=constraints,
            vertical=vertical,
        )

    async def _exit_turn_scope(
        self,
        *,
        user_message: str,
        stream: bool,
        constraints: Any = None,
        vertical: Optional[str] = None,
    ) -> None:
        """Run bound per-turn cleanup after task execution finishes."""
        await self._run_optional_callback(
            self._turn_teardown_handler,
            user_message,
            stream=stream,
            constraints=constraints,
            vertical=vertical,
        )

    def _is_response_complete(self, response: "CompletionResponse") -> bool:
        """Check if response is complete.

        A response is complete if:
        - It has a stop finish reason
        - It has no tool calls
        - It has substantial content

        Args:
            response: Response to check

        Returns:
            True if response is complete
        """
        if response.stop_reason == "stop":
            return True

        if response.content and len(response.content) > 50:
            return True

        return False

    def _has_tool_calls(self, response: "CompletionResponse") -> bool:
        """Check if response has tool calls.

        Args:
            response: Response to check

        Returns:
            True if response has tool calls
        """
        return bool(response.tool_calls)

    def _needs_continuation(self, response: "CompletionResponse") -> bool:
        """Check if response needs continuation.

        Args:
            response: Response to check

        Returns:
            True if continuation is needed
        """
        return response.stop_reason == "length"

    async def _execute_tool_calls(self, tool_calls: List[Any]) -> None:
        """Execute tool calls from response.

        Args:
            tool_calls: Tool calls to execute
        """
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments
            tool_call_id = getattr(tool_call, "id", None)

            result = await self._tools.execute_tool(tool_name, arguments)
            formatted_content = None

            if getattr(result, "success", False):
                try:
                    from victor.agent.services.tool_service import (
                        format_and_prune_tool_output,
                    )

                    _, formatted_content, _, _, _ = format_and_prune_tool_output(
                        tool_name=tool_name,
                        arguments=arguments,
                        output=result.output,
                    )
                except Exception:
                    formatted_content = None

            # Add tool result to context with proper tool_call_id
            self._add_tool_result_to_context(
                tool_name,
                result,
                tool_call_id=tool_call_id,
                formatted_content=formatted_content,
            )

    async def _create_continuation_prompt(self, response: "CompletionResponse") -> str:
        """Create continuation prompt for incomplete response.

        Args:
            response: Incomplete response

        Returns:
            Continuation prompt
        """
        return "Please continue."

    def _add_tool_result_to_context(
        self,
        tool_name: str,
        result: Any,
        tool_call_id: Optional[str] = None,
        formatted_content: Optional[str] = None,
    ) -> None:
        """Add tool result to context.

        Args:
            tool_name: Name of tool that was executed
            result: Tool result
            tool_call_id: ID of the tool call (required for OpenAI API compatibility)
            formatted_content: Optional preformatted tool content for LLM injection
        """
        # Handle output/error with proper string conversion
        # Use output if available and not None, otherwise use error
        if formatted_content is not None:
            content = formatted_content
        elif result.output is not None:
            content = str(result.output)
        elif result.error:
            content = str(result.error)
        else:
            content = ""

        if self._context:
            try:
                self._write_message_to_context(
                    role="tool",
                    content=content,
                    tool_call_id=tool_call_id
                    or tool_name,  # Use tool_call_id if provided, fallback to tool_name
                    name=tool_name,  # OpenAI spec requires 'name' for tool messages
                )
            except Exception as e:
                self._logger.warning(f"Failed to add tool result to context: {e}")

    def _aggregate_chunks(self, chunks: List["StreamChunk"]) -> "CompletionResponse":
        """Aggregate stream chunks into completion response.

        Args:
            chunks: Chunks to aggregate

        Returns:
            Aggregated CompletionResponse
        """
        from victor.providers.base import CompletionResponse

        content = "".join(chunk.content for chunk in chunks)
        # StreamChunk has optional usage field, handle safely
        total_tokens = sum(
            (chunk.usage.get("total_tokens", 0) if chunk.usage else 0) for chunk in chunks
        )

        return CompletionResponse(
            content=content,
            stop_reason="stop",
            usage={"total_tokens": total_tokens},
        )

    def _build_recovery_context(self, error: Exception) -> Any:
        """Build recovery context from an error.

        Args:
            error: Error that occurred

        Returns:
            RecoveryContextImpl with error details
        """
        from victor.agent.services.recovery_service import RecoveryContextImpl

        return RecoveryContextImpl(
            error=error,
            error_type=type(error).__name__,
            attempt_count=1,
            state={},
            metadata={},
        )

    # ==========================================================================
    # Real-Time Metrics Collection
    # ==========================================================================

    def get_chat_metrics(self) -> Dict[str, Any]:
        """Get real-time chat metrics.

        Returns comprehensive metrics about chat performance,
        including response times, token usage, error rates, etc.

        Returns:
            Dictionary with chat metrics

        Example:
            metrics = service.get_chat_metrics()
            # {
            #   "total_requests": 100,
            #   "successful_requests": 95,
            #   "failed_requests": 5,
            #   "average_response_time": 1.5,
            #   "total_tokens_used": 50000,
            #   "average_tokens_per_request": 500,
            #   "tool_calls_made": 20,
            #   "continuation_prompts": 5,
            # }
        """
        return {
            "total_requests": self._metrics.get("total_requests", 0),
            "successful_requests": self._metrics.get("successful_requests", 0),
            "failed_requests": self._metrics.get("failed_requests", 0),
            "average_response_time": self._calculate_average_response_time(),
            "total_tokens_used": self._metrics.get("total_tokens", 0),
            "average_tokens_per_request": self._calculate_average_tokens(),
            "tool_calls_made": self._metrics.get("tool_calls", 0),
            "continuation_prompts": self._metrics.get("continuations", 0),
            "last_request_time": self._metrics.get("last_request_time"),
            "success_rate": self._calculate_success_rate(),
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics.

        Returns granular performance data including latency percentiles,
        throughput metrics, and resource usage.

        Returns:
            Dictionary with performance metrics

        Example:
            perf = service.get_performance_metrics()
            # {
            #   "p50_latency": 1.2,
            #   "p95_latency": 2.5,
            #   "p99_latency": 5.0,
            #   "throughput_per_minute": 40,
            #   "average_tokens_per_second": 350,
            # }
        """
        latencies = self._metrics.get("latencies", [])

        if not latencies:
            return {
                "p50_latency": 0.0,
                "p95_latency": 0.0,
                "p99_latency": 0.0,
                "throughput_per_minute": 0.0,
                "average_tokens_per_second": 0.0,
            }

        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        return {
            "p50_latency": p50,
            "p95_latency": p95,
            "p99_latency": p99,
            "throughput_per_minute": self._calculate_throughput(),
            "average_tokens_per_second": self._calculate_tokens_per_second(),
        }

    # ==========================================================================
    # Response Quality Tracking
    # ==========================================================================

    def track_response_quality(
        self,
        response: "CompletionResponse",
        user_satisfaction: Optional[float] = None,
    ) -> None:
        """Track response quality metrics.

        Records quality indicators such as response length,
        completion rate, and user satisfaction.

        Args:
            response: The completion response
            user_satisfaction: Optional satisfaction score (0.0-1.0)

        Example:
            service.track_response_quality(response, user_satisfaction=0.8)
        """
        if "quality_metrics" not in self._metrics:
            self._metrics["quality_metrics"] = {
                "total_responses": 0,
                "average_length": 0.0,
                "completion_rate": 0.0,
                "user_satisfaction_scores": [],
            }

        quality = self._metrics["quality_metrics"]
        quality["total_responses"] += 1

        # Track response length
        content_length = len(response.content) if response.content else 0
        quality["average_length"] = (
            quality["average_length"] * (quality["total_responses"] - 1) + content_length
        ) / quality["total_responses"]

        # Track completion rate
        if response.stop_reason == "stop":
            quality["completion_rate"] = (
                quality["completion_rate"] * (quality["total_responses"] - 1) + 1.0
            ) / quality["total_responses"]
        else:
            quality["completion_rate"] = (
                quality["completion_rate"] * (quality["total_responses"] - 1) + 0.0
            ) / quality["total_responses"]

        # Track user satisfaction
        if user_satisfaction is not None:
            quality["user_satisfaction_scores"].append(user_satisfaction)

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get response quality metrics.

        Returns aggregated quality indicators.

        Returns:
            Dictionary with quality metrics

        Example:
            quality = service.get_quality_metrics()
            # {
            #   "total_responses": 100,
            #   "average_length": 500,
            #   "completion_rate": 0.95,
            #   "average_satisfaction": 0.85,
            # }
        """
        quality = self._metrics.get("quality_metrics", {})

        satisfaction_scores = quality.get("user_satisfaction_scores", [])
        avg_satisfaction = (
            sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else 0.0
        )

        return {
            "total_responses": quality.get("total_responses", 0),
            "average_length": quality.get("average_length", 0.0),
            "completion_rate": quality.get("completion_rate", 0.0),
            "average_satisfaction": avg_satisfaction,
            "satisfaction_count": len(satisfaction_scores),
        }

    # ==========================================================================
    # Conversation State Management
    # ==========================================================================

    def get_conversation_state(self) -> Dict[str, Any]:
        """Get detailed conversation state.

        Returns comprehensive state information including
        message counts, token usage, and stage information.

        Returns:
            Dictionary with conversation state

        Example:
            state = service.get_conversation_state()
            # {
            #   "message_count": 10,
            #   "total_tokens": 5000,
            #   "stage": "active",
            #   "has_tool_calls": True,
            #   "last_message_role": "assistant",
            # }
        """
        stats = self.get_conversation_stats()

        return {
            "message_count": stats.get("message_count", 0),
            "total_tokens": stats.get("total_tokens", 0),
            "user_messages": stats.get("user_messages", 0),
            "assistant_messages": stats.get("assistant_messages", 0),
            "system_messages": stats.get("system_messages", 0),
            "tool_messages": stats.get("tool_messages", 0),
            "stage": self._get_conversation_stage(),
            "has_tool_calls": stats.get("tool_calls", 0) > 0,
            "last_message_role": self._get_last_message_role(),
            "is_empty": self.is_conversation_empty(),
        }

    def _get_conversation_stage(self) -> str:
        """Get current conversation stage.

        Returns:
            Stage identifier: "initial", "active", "tool_execution", "completion"
        """
        if self.is_conversation_empty():
            return "initial"

        stats = self.get_conversation_stats()
        if stats.get("tool_calls", 0) > 0:
            return "tool_execution"

        if stats.get("assistant_messages", 0) > 0:
            return "completion"

        return "active"

    def _get_last_message_role(self) -> Optional[str]:
        """Get role of last message in conversation.

        Returns:
            Role string or None
        """
        metadata = self.get_conversation_metadata()
        last_role = metadata.get("last_message_role")
        return last_role

    # ==========================================================================
    # Performance Monitoring
    # ==========================================================================

    def _calculate_average_response_time(self) -> float:
        """Calculate average response time.

        Returns:
            Average response time in seconds
        """
        response_times = self._metrics.get("response_times", [])
        if not response_times:
            return 0.0
        return sum(response_times) / len(response_times)

    def _calculate_average_tokens(self) -> float:
        """Calculate average tokens per request.

        Returns:
            Average token count
        """
        total_tokens = self._metrics.get("total_tokens", 0)
        total_requests = self._metrics.get("total_requests", 0)
        if total_requests == 0:
            return 0.0
        return total_tokens / total_requests

    def _calculate_success_rate(self) -> float:
        """Calculate success rate.

        Returns:
            Success rate (0.0-1.0)
        """
        successful = self._metrics.get("successful_requests", 0)
        total = self._metrics.get("total_requests", 0)
        if total == 0:
            return 0.0
        return successful / total

    def _calculate_throughput(self) -> float:
        """Calculate throughput per minute.

        Returns:
            Requests per minute
        """
        # Simple calculation based on total requests and uptime
        # In production, would use actual time window
        total_requests = self._metrics.get("total_requests", 0)
        return float(total_requests)  # Placeholder

    def _calculate_tokens_per_second(self) -> float:
        """Calculate tokens per second throughput.

        Returns:
            Average tokens per second
        """
        total_tokens = self._metrics.get("total_tokens", 0)
        total_time = sum(self._metrics.get("latencies", [1.0]))
        if total_time == 0:
            return 0.0
        return total_tokens / total_time

    # ==========================================================================
    # Advanced Error Handling
    # ==========================================================================

    async def handle_chat_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> "CompletionResponse":
        """Handle chat errors with recovery strategies.

        Attempts to recover from errors using fallback strategies
        before giving up.

        Args:
            error: The error that occurred
            context: Optional context for recovery

        Returns:
            Fallback response or raises if unrecoverable

        Example:
            try:
                response = await service.chat("Hello")
            except Exception as e:
                response = await service.handle_chat_error(e)
        """
        from victor.providers.base import CompletionResponse

        # Try recovery service
        recovery_context = self._build_recovery_context(error)

        if context:
            recovery_context.state.update(context)

        # Check if recovery is possible
        should_recover = self._recovery.should_attempt_recovery(
            recovery_context.error_type,
            consecutive_failures=0,
        )

        if should_recover:
            self._logger.info(f"Attempting recovery from {recovery_context.error_type}")

            # Return fallback response
            return CompletionResponse(
                content=f"I encountered an error ({recovery_context.error_type}). "
                f"Please try rephrasing your request.",
                stop_reason="error",
                usage={},
            )

        # If no recovery possible, raise the original error
        raise error

    # ==========================================================================
    # Streaming Integration
    # ==========================================================================

    async def stream_with_callback(
        self,
        user_message: str,
        callback: Callable[[str], None],
        **kwargs,
    ) -> "CompletionResponse":
        """Stream chat with callback for each chunk.

        Streams the response and calls the callback function
        with each chunk's content.

        Args:
            user_message: User's message
            callback: Callback function called with each chunk
            **kwargs: Additional arguments for chat

        Returns:
            Aggregated completion response

        Example:
            def on_chunk(chunk: str):
                print(chunk, end="", flush=True)

            response = await service.stream_with_callback(
                "Hello, world!",
                on_chunk
            )
        """
        chunks = []
        async for chunk in self.stream_chat(user_message, **kwargs):
            chunks.append(chunk)
            if chunk.content:
                callback(chunk.content)

        return self._aggregate_chunks(chunks)

    def get_streaming_metrics(self) -> Dict[str, Any]:
        """Get streaming-specific metrics.

        Returns metrics about streaming performance including
        chunks per second, average chunk size, etc.

        Returns:
            Dictionary with streaming metrics

        Example:
            metrics = service.get_streaming_metrics()
            # {
            #   "total_streams": 50,
            #   "average_chunks_per_stream": 100,
            #   "average_chunk_size": 5,
            #   "chunks_per_second": 50,
            # }
        """
        return {
            "total_streams": self._metrics.get("total_streams", 0),
            "average_chunks_per_stream": self._metrics.get("average_chunks", 0),
            "average_chunk_size": self._metrics.get("avg_chunk_size", 0),
            "chunks_per_second": self._metrics.get("chunks_per_second", 0),
            "total_chunks_sent": self._metrics.get("total_chunks", 0),
        }

    # ==========================================================================
    # Streaming Runtime Management
    # ==========================================================================

    def set_streaming_executor(self, executor: Any) -> None:
        """Set or override the streaming executor.

        Allows injection of a pre-built streaming executor for custom
        streaming behavior or testing.

        Args:
            executor: Streaming executor instance

        Example:
            service.set_streaming_executor(custom_executor)
        """
        self._streaming = executor
        self._logger.debug("Streaming executor set to: %s", type(executor).__name__)

    # ==========================================================================
    # Planning and Task Preparation
    # ==========================================================================

    def _should_use_planning(self, user_message: str) -> bool:
        """Determine if planning should be used for this task.

        Checks for:
        1. Multi-step indicators in the message
        2. Task complexity keywords

        Args:
            user_message: User's message

        Returns:
            True if planning should be used

        Example:
            if service._should_use_planning("analyze the code"):
                # Use planning
        """
        from victor.agent.planning.constants import COMPLEXITY_KEYWORDS, STEP_INDICATORS

        message_lower = user_message.lower()
        if any(indicator in message_lower for indicator in COMPLEXITY_KEYWORDS):
            return True

        step_matches = sum(1 for indicator in STEP_INDICATORS if indicator in message_lower)
        return step_matches >= 2

    @staticmethod
    def _is_explicit_planning_request(user_message: str) -> bool:
        """Return whether the user explicitly asked for planning mode/plan output.

        Streaming auto-detection for complex tasks is useful, but routing a streamed
        turn into a non-streaming planning handler suppresses live tool and content
        updates in the CLI. Keep that special handling for explicit plan requests only.
        """
        from victor.agent.planning.intent import is_explicit_planning_request

        return is_explicit_planning_request(user_message)

    def _prepare_task(self, user_message: str, task_type: str) -> tuple[Dict[str, Any], int]:
        """Prepare task-specific guidance and budget adjustments.

        Args:
            user_message: The user's message
            task_type: The detected task type

        Returns:
            Tuple of (task_classification, complexity_tool_budget)

        Example:
            task_info, budget = service._prepare_task("fix the bug", "debug")
        """
        # Default task classification
        task_classification = {
            "type": task_type,
            "complexity": "medium",
            "requires_tools": True,
        }

        # Budget based on complexity
        complexity_budgets = {
            "simple": 10,
            "medium": 20,
            "complex": 50,
        }

        budget = complexity_budgets.get(task_classification["complexity"], 20)

        return task_classification, budget

    # ==========================================================================
    # Stream Handling
    # ==========================================================================

    def _handle_stream_chunk(
        self,
        chunk: Any,
        consecutive_garbage_chunks: int,
        max_garbage_chunks: int = 5,
        garbage_detected: bool = False,
    ) -> tuple[Any, int, bool]:
        """Handle garbage detection for a streaming chunk.

        Args:
            chunk: The stream chunk to check
            consecutive_garbage_chunks: Current count of consecutive garbage chunks
            max_garbage_chunks: Maximum consecutive garbage chunks allowed
            garbage_detected: Whether garbage has been detected

        Returns:
            Tuple of (chunk, consecutive_garbage_chunks, garbage_detected)

        Example:
            chunk, count, detected = service._handle_stream_chunk(
                chunk, consecutive_count, max_chunks, detected
            )
        """
        # Simple garbage detection: check if content is None or empty
        if chunk is None:
            consecutive_garbage_chunks += 1
            if consecutive_garbage_chunks >= max_garbage_chunks:
                if not garbage_detected:
                    self._logger.warning("Empty chunks detected - stopping stream")
                    garbage_detected = True
                return None, consecutive_garbage_chunks, garbage_detected
        else:
            consecutive_garbage_chunks = 0

        return chunk, consecutive_garbage_chunks, garbage_detected

    # ==========================================================================
    # Rate Limiting
    # ==========================================================================

    def _get_rate_limit_wait_time(self, exc: Exception, attempt: int) -> float:
        """Get wait time for rate limit retry.

        Args:
            exc: The rate limit exception
            attempt: Current retry attempt number

        Returns:
            Number of seconds to wait before retrying

        Example:
            wait_time = service._get_rate_limit_wait_time(error, attempt=1)
        """
        # Delegate to provider service if available
        if self._provider:
            try:
                base_wait = self._provider.get_rate_limit_wait_time(exc)
                backoff_multiplier = 2**attempt
                wait_time = base_wait * backoff_multiplier
                return min(wait_time, 300.0)  # Max 5 minutes
            except Exception:
                pass

        # Fallback to exponential backoff
        return min(60.0 * (2**attempt), 300.0)

    # ==========================================================================
    # Message Persistence
    # ==========================================================================

    @staticmethod
    def persist_message(
        role: str,
        content: str,
        memory_manager: Any,
        memory_session_id: Optional[str],
        usage_logger: Any,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_calls: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
        persist_synchronously: bool = False,
    ) -> None:
        """Persist a message to memory and emit usage analytics.

        This is the canonical persistence helper for orchestrator message
        writes. It preserves the legacy logging behavior expected by existing
        analytics flows while keeping ownership on ``ChatService``.
        """
        if memory_manager and memory_session_id:
            try:
                from victor.agent.conversation.types import MessageRole

                role_map = {
                    "user": MessageRole.USER,
                    "assistant": MessageRole.ASSISTANT,
                    "system": MessageRole.SYSTEM,
                    "tool": MessageRole.TOOL,
                    "tool_result": MessageRole.TOOL,
                    "tool_call": MessageRole.TOOL_CALL,
                }
                msg_role = role_map.get(role, MessageRole.USER)

                add_kwargs: Dict[str, Any] = {
                    "session_id": memory_session_id,
                    "role": msg_role,
                    "content": content,
                }
                if tool_name:
                    add_kwargs["tool_name"] = tool_name
                if tool_call_id:
                    add_kwargs["tool_call_id"] = tool_call_id
                if tool_calls:
                    add_kwargs["tool_calls"] = tool_calls
                if metadata:
                    add_kwargs["metadata"] = metadata

                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                def _persist_background_message() -> None:
                    try:
                        memory_manager.add_message(**add_kwargs)
                    except Exception as exc:
                        logger.debug("Failed to persist message in background: %s", exc)

                def _consume_background_result(future: asyncio.Future) -> None:
                    try:
                        future.exception()
                    except asyncio.CancelledError:
                        logger.debug("Background message persistence was cancelled")

                if loop is not None and loop.is_running() and not persist_synchronously:
                    future = loop.run_in_executor(None, _persist_background_message)
                    future.add_done_callback(_consume_background_result)
                else:
                    memory_manager.add_message(**add_kwargs)
            except Exception as e:
                logger.debug("Failed to persist message: %s", e)

        if not usage_logger:
            return

        try:
            if hasattr(usage_logger, "log_event"):
                if role == "user":
                    usage_logger.log_event("user_prompt", {"content": content})
                elif role == "assistant":
                    usage_logger.log_event("assistant_response", {"content": content})
                    if hasattr(usage_logger, "set_reasoning_context") and content:
                        usage_logger.set_reasoning_context(content)
                return

            if hasattr(usage_logger, "log_message"):
                usage_logger.log_message(role, content)
        except Exception as e:
            logger.debug("Failed to log message: %s", e)

    # ==========================================================================
    # Context Management Helpers
    # ==========================================================================

    def _add_user_message_to_context(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add user message to context.

        Delegates to context service to add a user message to the
        conversation context.

        Args:
            content: User message content
            metadata: Optional metadata for the message

        Example:
            service._add_user_message_to_context("Hello world")
        """
        if self._context:
            try:
                self._write_message_to_context(
                    role="user",
                    content=content,
                    metadata=metadata or {},
                )
            except Exception as e:
                self._logger.warning(f"Failed to add user message to context: {e}")

    def _add_assistant_message_to_context(
        self,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add assistant message to context.

        Delegates to context service to add an assistant message to the
        conversation context.

        Args:
            content: Assistant message content
            tool_calls: Optional list of tool calls made
            metadata: Optional metadata for the message

        Example:
            service._add_assistant_message_to_context(
                "I'll help you with that.",
                tool_calls=[{"name": "read", "arguments": {"path": "file.txt"}}]
            )
        """
        if self._context:
            try:
                self._write_message_to_context(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                    metadata=metadata or {},
                )
            except Exception as e:
                self._logger.warning(f"Failed to add assistant message to context: {e}")

    def _write_message_to_context(self, role: str, content: str, **payload: Any) -> None:
        """Write a normalized message to either keyword- or dict-based context services."""
        add_message = getattr(self._context, "add_message", None)
        if add_message is None:
            raise AttributeError("Context service does not provide add_message()")

        if self._context_uses_keyword_messages():
            add_message(role=role, content=content, **payload)
            return

        normalized_payload = {key: value for key, value in payload.items() if value is not None}
        add_message({"role": role, "content": content, **normalized_payload})

    def _context_uses_keyword_messages(self) -> bool:
        """Detect whether the bound context service expects keyword-based message writes."""
        if self._context_accepts_keyword_messages is not None:
            return self._context_accepts_keyword_messages

        add_message = getattr(self._context, "add_message", None)
        if add_message is None:
            self._context_accepts_keyword_messages = False
            return False

        try:
            signature = inspect.signature(add_message)
        except (TypeError, ValueError):
            self._context_accepts_keyword_messages = True
            return True

        parameter_names = {
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

        self._context_accepts_keyword_messages = accepts_kwargs or {
            "role",
            "content",
        }.issubset(parameter_names)
        return self._context_accepts_keyword_messages

    def handle_chat_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Handle chat error with recovery.

        Delegates to recovery service to attempt recovery from the error.

        Args:
            error: The exception that occurred
            context: Optional context for recovery

        Returns:
            Dictionary with error handling result:
            - handled: Whether error was handled
            - action: Action taken (retry, switch, abort, etc.)
            - message: Human-readable message

        Example:
            result = service.handle_chat_error(error, context)
            if result["handled"]:
                print(f"Error handled: {result['action']}")
        """
        if self._recovery:
            try:
                # Try to recover using recovery service
                recovered = self._recovery.should_attempt_recovery(
                    type(error).__name__,
                    consecutive_failures=0,
                )
                if recovered:
                    return {
                        "handled": True,
                        "action": "retry",
                        "message": "Attempting recovery",
                    }
                else:
                    return {
                        "handled": True,
                        "action": "abort",
                        "message": str(error),
                    }
            except Exception as e:
                self._logger.warning(f"Recovery check failed: {e}")

        # Default: not handled
        return {
            "handled": False,
            "action": "abort",
            "message": str(error),
        }

    def normalize_tool_arguments(
        self,
        tool_args: Dict[str, Any],
        tool_name: str,
    ) -> tuple[Dict[str, Any], str]:
        """Normalize tool arguments to handle malformed JSON.

        Simplified version for ChatService that delegates to ToolService
        if available.

        Args:
            tool_args: Raw arguments from tool call
            tool_name: Name of the tool being called

        Returns:
            Tuple of (normalized_args, strategy_used)

        Example:
            args, strategy = service.normalize_tool_arguments(
                {"query": "test"}, "code_search"
            )
        """
        # Try to delegate to tool service if available
        if self._tools:
            try:
                return self._tools.normalize_tool_arguments(tool_args, tool_name)
            except Exception:
                pass

        # Fallback: return as-is
        return tool_args, "direct"

    def _prepare_new_turn(self) -> None:
        """Reset per-prompt execution state before a new user turn."""
        start_new_turn = getattr(self._tools, "start_new_turn", None)
        if callable(start_new_turn):
            start_new_turn()
            return

        reset_tool_budget = getattr(self._tools, "reset_tool_budget", None)
        if callable(reset_tool_budget):
            reset_tool_budget()

    # ==========================================================================
    # Turn Executor Property
    # ==========================================================================

    @property
    def turn_executor(self) -> Any:
        """Get the bound turn executor for compatibility callers."""
        if self._turn_executor is None:
            raise RuntimeError(
                "ChatService runtime is not bound: "
                "turn_executor is unavailable until bind_runtime_components()."
            )
        return self._turn_executor
