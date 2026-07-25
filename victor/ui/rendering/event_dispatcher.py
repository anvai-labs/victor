"""Event dispatcher for streaming response handling.

This module extracts event routing logic from the long if-elif chain
in handler.py into a clean dispatcher pattern. Each event type is
routed to a dedicated handler method, making the dispatch logic
testable and extensible.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from victor.framework.events import EventType
from victor.ui.rendering.protocol import StreamRenderer
from victor.ui.rendering.utils import (
    extract_tool_call_arguments,
    extract_tool_result_payload,
    is_thinking_status_message,
    normalize_reasoning_content,
)

logger = logging.getLogger(__name__)


class EventDispatcher:
    """Routes streaming events to the appropriate renderer methods.

    Encapsulates the event dispatch logic that was previously a long
    if-elif chain in stream_response(). Each event type gets a dedicated
    handler method, making the logic testable and extensible.

    Usage:
        dispatcher = EventDispatcher(renderer, content_filter, ...)
        async for event in stream_gen:
            dispatcher.dispatch(event)
    """

    def __init__(
        self,
        renderer: StreamRenderer,
        content_filter: Any,
        content_normalizer: Any,
        reasoning_normalizer: Any,
        suppress_thinking: bool = False,
    ):
        """Initialize the event dispatcher.

        Args:
            renderer: The stream renderer to dispatch events to
            content_filter: StreamingContentFilter for inline thinking markers
            content_normalizer: StreamDeltaNormalizer for content dedup
            reasoning_normalizer: StreamDeltaNormalizer for reasoning dedup
            suppress_thinking: If True, hide thinking content
        """
        self.renderer = renderer
        self.content_filter = content_filter
        self.content_normalizer = content_normalizer
        self.reasoning_normalizer = reasoning_normalizer
        self.suppress_thinking = suppress_thinking
        self.was_thinking = False
        self.error_surfaced = False

    def dispatch(self, event: Any) -> None:
        """Route a single event to the appropriate handler.

        Args:
            event: The streaming event to dispatch
        """
        event_type = getattr(event, "type", None)
        event_tool_name = getattr(event, "tool_name", None)
        event_metadata = getattr(event, "metadata", None)
        event_content = getattr(event, "content", "")

        # Structured tool events (metadata-based)
        if event_metadata and "tool_start" in event_metadata:
            self._handle_tool_start_metadata(event_metadata, event_tool_name)
        elif event_metadata and "tool_result" in event_metadata:
            self._handle_tool_result_metadata(event_metadata, event_tool_name)
        # Typed tool events
        elif event_type == EventType.TOOL_CALL:
            self._handle_tool_call(event, event_tool_name)
        elif event_type == EventType.TOOL_RESULT:
            self._handle_tool_result(event, event_tool_name)
        # Error events
        elif event_type == EventType.ERROR:
            self._handle_error(event, event_content)
        # Tool progress events
        elif self._is_tool_progress_event(event_metadata, event_type):
            self._handle_tool_progress(event_metadata, event_tool_name, event_content)
        # Status events
        elif event_metadata and "status" in event_metadata:
            self._handle_status(event_metadata)
        # File preview
        elif event_metadata and "file_preview" in event_metadata:
            self._handle_file_preview(event_metadata)
        # Edit preview
        elif event_metadata and "edit_preview" in event_metadata:
            self._handle_edit_preview(event_metadata)
        # Reasoning content (DeepSeek API)
        elif event_metadata and "reasoning_content" in event_metadata:
            self._handle_reasoning_content(event_metadata, event_content)
        # Model refusal (sandhi refusal_delta -> metadata.refusal): a refusal is
        # the model's user-facing outcome for the turn — never swallow it.
        elif event_metadata and "refusal" in event_metadata:
            refusal_text = str(event_metadata.get("refusal") or "").strip()
            if refusal_text:
                self.renderer.on_status(f"\u26d4 Model refusal: {refusal_text}")
        # Plain content
        elif event_content:
            self._handle_content(event_content)

    def _is_tool_progress_event(self, event_metadata: Any, event_type: Any) -> bool:
        """Check if the event is a tool progress event."""
        return (
            event_metadata is not None and "tool_progress" in event_metadata
        ) or event_type == EventType.TOOL_PROGRESS

    def _handle_tool_start_metadata(
        self, event_metadata: dict, event_tool_name: Optional[str]
    ) -> None:
        """Handle a tool_start metadata event."""
        tool_start = event_metadata["tool_start"] or {}
        self.renderer.on_tool_start(
            name=str(tool_start.get("name", event_tool_name or "unknown")),
            arguments=tool_start.get("arguments", {}),
            tool_call_id=tool_start.get("tool_call_id"),
            batch_index=tool_start.get("batch_index"),
            batch_total=tool_start.get("batch_total"),
            execution_mode=tool_start.get("execution_mode"),
        )

    def _handle_tool_result_metadata(
        self, event_metadata: dict, event_tool_name: Optional[str]
    ) -> None:
        """Handle a tool_result metadata event."""
        result_data = event_metadata["tool_result"] or {}
        kwargs = self._build_tool_result_kwargs(result_data, event_tool_name)
        self.renderer.on_tool_result(**kwargs)

    def _handle_tool_call(self, event: Any, event_tool_name: Optional[str]) -> None:
        """Handle a TOOL_CALL typed event."""
        self.renderer.on_tool_start(
            name=event_tool_name or "unknown",
            arguments=extract_tool_call_arguments(event),
        )

    def _handle_tool_result(self, event: Any, event_tool_name: Optional[str]) -> None:
        """Handle a TOOL_RESULT typed event."""
        result_data = extract_tool_result_payload(event)
        kwargs = self._build_tool_result_kwargs(result_data, event_tool_name)
        self.renderer.on_tool_result(**kwargs)

    def _build_tool_result_kwargs(self, result_data: dict, event_tool_name: Optional[str]) -> dict:
        """Build keyword arguments for on_tool_result from result data."""
        kwargs = {
            "name": str(result_data.get("name", event_tool_name or "unknown")),
            "success": result_data.get("success", True),
            "elapsed": result_data.get("elapsed", 0),
            "arguments": result_data.get("arguments", {}),
            "error": result_data.get("error"),
            "follow_up_suggestions": result_data.get("follow_up_suggestions"),
            "result": result_data.get("result"),
            "tool_call_id": result_data.get("tool_call_id"),
        }
        if "original_result" in result_data:
            kwargs["original_result"] = result_data.get("original_result")
        if result_data.get("was_pruned"):
            kwargs["was_pruned"] = True
        return kwargs

    def _handle_error(self, event: Any, event_content: str) -> None:
        """Handle an ERROR event."""
        error_text = (
            getattr(event, "error", None) or event_content or "The provider returned an error."
        )
        self.renderer.on_status(f"\u274c {error_text}")
        self.error_surfaced = True

    def _handle_tool_progress(
        self,
        event_metadata: Optional[dict],
        event_tool_name: Optional[str],
        event_content: str,
    ) -> None:
        """Handle a tool progress event."""
        progress_data = (event_metadata or {}).get("tool_progress") or {}
        on_progress = getattr(self.renderer, "on_tool_progress", None)
        if callable(on_progress):
            on_progress(
                name=str(progress_data.get("name", event_tool_name or "unknown")),
                stdout=progress_data.get("stdout", event_content or ""),
                stderr=progress_data.get("stderr", ""),
                progress=progress_data.get("progress", 0.0),
                is_final=bool(progress_data.get("is_final", False)),
            )

    def _handle_status(self, event_metadata: dict) -> None:
        """Handle a status event."""
        status_message = str(event_metadata["status"])
        if is_thinking_status_message(status_message):
            if not self.suppress_thinking and not self.was_thinking:
                logger.debug("\u2192 Entering thinking mode (status event)")
                self.renderer.on_thinking_start()
                self.reasoning_normalizer.reset()
                self.was_thinking = True
        else:
            self.renderer.on_status(status_message)

    def _handle_file_preview(self, event_metadata: dict) -> None:
        """Handle a file preview event."""
        self.renderer.on_file_preview(
            path=event_metadata.get("path", ""),
            content=event_metadata["file_preview"],
        )

    def _handle_edit_preview(self, event_metadata: dict) -> None:
        """Handle an edit preview event."""
        self.renderer.on_edit_preview(
            path=event_metadata.get("path", ""),
            diff=event_metadata["edit_preview"],
        )

    def _handle_reasoning_content(self, event_metadata: dict, event_content: str) -> None:
        """Handle reasoning content from DeepSeek API."""
        reasoning = event_metadata["reasoning_content"]
        if reasoning and not self.suppress_thinking:
            reasoning = normalize_reasoning_content(reasoning)

            if not self.was_thinking:
                logger.debug("\u2192 Entering thinking mode (API reasoning)")
                self.renderer.on_thinking_start()
                self.reasoning_normalizer.reset()
                self.was_thinking = True

            reasoning_delta = self.reasoning_normalizer.consume(reasoning)
            if reasoning_delta:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "reasoning_normalizer: input_len=%d, delta_len=%d, "
                        "input_preview=%r..., delta_preview=%r...",
                        len(reasoning),
                        len(reasoning_delta),
                        reasoning[:60],
                        reasoning_delta[:60],
                    )
                self.renderer.on_thinking_content(reasoning_delta)
            elif logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "reasoning_normalizer: filtered duplicate, input_len=%d, "
                    "input_preview=%r...",
                    len(reasoning),
                    reasoning[:60],
                )

        if event_content:
            self._handle_content(event_content)

    def _handle_content(self, raw_content: str) -> None:
        """Handle content through the streaming content filter.

        Manages transitions between thinking and normal content modes
        for inline thinking markers.
        """
        normalized_content = self.content_normalizer.consume(raw_content)
        if not normalized_content:
            return

        # End API-based thinking state when regular content arrives
        if self.was_thinking and not self.content_filter.is_thinking:
            logger.debug("\u2190 Exiting thinking mode (API reasoning ended)")
            self.renderer.on_thinking_end()
            self.reasoning_normalizer.reset()
            self.was_thinking = False

        result = self.content_filter.process_chunk(normalized_content)

        # Log inline thinking state transitions
        if result.entering_thinking and not self.was_thinking:
            logger.debug("\u2192 Entering thinking mode (inline markers)")
            self.renderer.on_thinking_start()
            self.was_thinking = True

        if result.content:
            if result.is_thinking:
                thinking_delta = self.reasoning_normalizer.consume(result.content)
                if thinking_delta:
                    self.renderer.on_thinking_content(thinking_delta)
            else:
                self.renderer.on_content(result.content)

        if result.exiting_thinking and self.was_thinking:
            logger.debug("\u2190 Exiting thinking mode (inline markers ended)")
            self.renderer.on_thinking_end()
            self.reasoning_normalizer.reset()
            self.was_thinking = False

    def flush_thinking(self) -> None:
        """Flush any remaining thinking state.

        Should be called after all events are processed to ensure
        thinking state is properly closed.
        """
        if self.was_thinking:
            self.renderer.on_thinking_end()
            self.reasoning_normalizer.reset()
            self.was_thinking = False
