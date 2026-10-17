"""Stream response handler - unified streaming response processing.

This module provides the stream_response() function which is the single
source of truth for streaming response handling across all CLI modes.

Event dispatch has been extracted to ``EventDispatcher`` in
``event_dispatcher.py`` for testability and extensibility.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from victor.agent.response_sanitizer import create_streaming_filter
from victor.core.errors import AgentError
from victor.ui.rendering.event_dispatcher import EventDispatcher
from victor.ui.rendering.protocol import StreamRenderer
from victor.ui.rendering.utils import (
    StreamDeltaNormalizer,
)

logger = logging.getLogger(__name__)


class RenderedStreamError(AgentError):
    """Terminal run failure already delivered to the renderer.

    Callers choose exit or return-to-prompt behavior without rendering the error
    twice or retrying the failed turn. Buffered renderers still need flushing.
    """


def _resolve_stream_factory(agent: Any) -> Any:
    """Resolve the streaming entrypoint while preserving legacy compatibility."""

    agent_dict = getattr(agent, "__dict__", {})

    if "stream" in agent_dict or hasattr(type(agent), "stream"):
        return agent.stream
    if "stream_chat" in agent_dict or hasattr(type(agent), "stream_chat"):
        return agent.stream_chat

    stream_factory = getattr(agent, "stream", None)
    if callable(stream_factory):
        return stream_factory

    legacy_stream_factory = getattr(agent, "stream_chat", None)
    if callable(legacy_stream_factory):
        return legacy_stream_factory

    raise AttributeError("Agent does not expose stream() or stream_chat()")


async def stream_response(
    agent: Any,  # ✅ PROPER: Use Any instead of AgentOrchestrator (duck typing)
    message: str,
    renderer: StreamRenderer,
    suppress_thinking: bool = False,
) -> str:
    """Unified streaming response handler.

    This is the single source of truth for streaming response handling.
    It processes chunks from the agent and delegates rendering to the
    provided renderer implementation.

    Event routing is delegated to ``EventDispatcher`` for clean separation
    of concerns and testability.

    Thinking Content Handling (dual-mode):
    - **API-based reasoning**: DeepSeek API sends reasoning content via
      `event.metadata["reasoning_content"]`. This is handled directly and
      rendered through `renderer.on_thinking_content()`.
    - **Inline markers**: Models like Qwen3 or local Ollama models may use
      inline markers (``<think>...</think>``, ``<|begin_of_thinking|>``).
      These are processed by ``StreamingContentFilter`` from response_sanitizer.

    State management automatically handles transitions between thinking
    and normal content, including when switching from API-based reasoning
    to regular content output.

    Args:
        agent: The agent instance to stream from (Agent facade with stream method)
        message: The user message to send
        renderer: The renderer to use for output
        suppress_thinking: If True, completely hide thinking content

    Returns:
        The accumulated response content

    Raises:
        RenderedStreamError: The run failed; partial output has been finalized.
    """
    renderer.start()

    # Register a UI-ephemeral progress sink so long-running tools can stream
    # partial output into the live display while they run. Best-effort and
    # gated by settings; cleared in the finally below so it never leaks across
    # turns. Progress never enters the conversation or reaches the model.
    _progress_sink_registered = False
    on_progress = getattr(renderer, "on_tool_progress", None)
    if callable(on_progress):
        try:
            from victor.config.tool_settings import get_tool_settings
            from victor.framework.tool_progress import set_progress_sink

            if get_tool_settings().tool_progress_streaming_enabled:
                set_progress_sink(on_progress)
                _progress_sink_registered = True
        except Exception:  # pragma: no cover - defensive wiring
            logger.debug("Failed to register tool progress sink", exc_info=True)

    stream_gen = _resolve_stream_factory(agent)(message)

    # Initialize content filter for thinking markers
    content_filter = create_streaming_filter(suppress_thinking=suppress_thinking)
    content_normalizer = StreamDeltaNormalizer()
    reasoning_normalizer = StreamDeltaNormalizer()

    # Use EventDispatcher for clean event routing
    dispatcher = EventDispatcher(
        renderer=renderer,
        content_filter=content_filter,
        content_normalizer=content_normalizer,
        reasoning_normalizer=reasoning_normalizer,
        suppress_thinking=suppress_thinking,
    )

    # ENHANCEMENT: Log chunk boundaries for debugging streaming issues
    chunk_count = 0

    try:
        async for event in stream_gen:
            chunk_count += 1
            event_content = getattr(event, "content", "")

            # Log every 100th event to diagnose duplication/buffering issues
            if chunk_count % 100 == 0:
                event_type = getattr(event, "type", None)
                content_len = len(event_content) if event_content else 0
                logger.debug(
                    f"Stream event #{chunk_count}: "
                    f"type={event_type}, "
                    f"content_len={content_len}, "
                    f"is_thinking={content_filter.is_thinking}"
                )

            # Delegate all event routing to the EventDispatcher
            dispatcher.dispatch(event)

            # Break on error events (dispatcher sets error_surfaced)
            if dispatcher.error_surfaced:
                break

            if content_filter.should_abort():
                dispatcher.surface_error(
                    content_filter.abort_reason or "Excessive thinking aborted the response."
                )
                break

        # Flush any remaining buffered content
        flush_result = content_filter.flush()
        if flush_result.content:
            if flush_result.is_thinking:
                renderer.on_thinking_content(flush_result.content)
            else:
                renderer.on_content(flush_result.content)

        # CRITICAL FIX: Ensure thinking state is properly closed
        dispatcher.flush_thinking()

        # CRITICAL FIX: Add comprehensive logging before finalize
        logger.debug(
            "stream_response completion - content_buffer_len=%d, "
            "in_thinking_mode=%s, live_active=%s",
            len(getattr(renderer, "_content_buffer", "")),
            getattr(renderer, "_in_thinking_mode", False),
            getattr(renderer, "_live", None) is not None,
        )

        # Log first 100 chars of content buffer for verification
        if logger.isEnabledFor(logging.DEBUG):
            content_preview = getattr(renderer, "_content_buffer", "")[:100]
            logger.debug("Content buffer preview: %r...", content_preview)

        # FAIL-SAFE: Verify content was displayed
        final_content = renderer.finalize()
        if dispatcher.error is not None:
            raise RenderedStreamError(dispatcher.error, recoverable=False)
        # Empty content is valid in some flows (e.g. empty stream or metadata-only events),
        # so return the renderer output directly and only log for visibility. A surfaced
        # error already explains the empty result, so don't misreport it as a bug.
        if not final_content and not renderer.had_tool_calls() and not dispatcher.error_surfaced:
            logger.warning("stream_response returned empty content - this may indicate a bug")

        return final_content

    finally:
        if _progress_sink_registered:
            try:
                from victor.framework.tool_progress import clear_progress_sink

                clear_progress_sink()
            except Exception:  # pragma: no cover - defensive
                pass
        renderer.cleanup()
        # Graceful cleanup of async generator to prevent RuntimeError on abort
        aclose = getattr(stream_gen, "aclose", None)
        if callable(aclose):
            try:
                maybe_awaitable = aclose()
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
            except RuntimeError:
                # Generator already closed or running - ignore
                pass
