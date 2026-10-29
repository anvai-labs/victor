"""ASI (Actionable Side Information) trace enrichment for GEPA v2.

Wraps an existing UsageLogger to enrich events with detailed trace data
that serves as the "text analogue of a gradient" for GEPA prompt evolution.

When enabled, enriches:
- tool_call: adds reasoning_before_call, arguments_sanitized
- tool_result: adds result_summary, error_detail, duration_ms
- llm_response: captures thinking_content, response_summary

When disabled, passes through unchanged (zero overhead).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TraceEnricher:
    """Middleware that enriches usage events with ASI trace data.

    Duck-typed to match UsageLogger interface (log_event, session_id).
    The orchestrator can use this as a drop-in replacement.
    """

    def __init__(
        self,
        inner_logger: Any,
        enabled: bool = False,
        max_output_chars: int = 2000,
        max_reasoning_chars: int = 1000,
        capture_reasoning: bool = True,
        capture_tool_args: bool = True,
        capture_tool_output: bool = True,
    ):
        self._inner = inner_logger
        self._enabled = enabled
        self._max_output_chars = max_output_chars
        self._max_reasoning_chars = max_reasoning_chars
        self._capture_reasoning = capture_reasoning
        self._capture_tool_args = capture_tool_args
        self._capture_tool_output = capture_tool_output

        # Buffer: the last assistant reasoning text, consumed by next tool_call
        self._pending_reasoning: Optional[str] = None
        # Buffer: tool start time for duration tracking
        self._pending_duration_ms: Optional[float] = None

    @property
    def session_id(self) -> str:
        """Proxy to inner logger's session_id."""
        return getattr(self._inner, "session_id", "")

    def set_reasoning_context(self, reasoning: str) -> None:
        """Buffer the assistant's reasoning text for the next tool_call event.

        Called by the orchestrator after parsing assistant content but before
        tool execution. The reasoning is consumed by the next tool_call event.
        """
        if self._enabled and self._capture_reasoning and reasoning:
            self._pending_reasoning = reasoning[: self._max_reasoning_chars]

    def set_duration_context(self, duration_ms: float) -> None:
        """Buffer tool execution duration for the next tool_result event.

        Called by the orchestrator after tool execution completes.
        """
        if self._enabled:
            self._pending_duration_ms = duration_ms

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Enrich and forward event to inner logger."""
        if not self._enabled:
            self._inner.log_event(event_type, data)
            return

        enriched = dict(data)

        if event_type == "tool_call":
            self._enrich_tool_call(enriched)
        elif event_type == "tool_result":
            self._enrich_tool_result(enriched)
        elif event_type == "llm_response":
            self._enrich_llm_response(enriched)

        self._inner.log_event(event_type, enriched)

    def _enrich_tool_call(self, data: Dict[str, Any]) -> None:
        """Add reasoning context and sanitized args to tool_call."""
        if self._capture_reasoning and self._pending_reasoning:
            data["reasoning_before_call"] = self._pending_reasoning
            self._pending_reasoning = None

        if self._capture_tool_args:
            args = data.get("tool_args", {})
            if isinstance(args, dict):
                data["arguments_sanitized"] = self._sanitize_args(args)

    def _enrich_tool_result(self, data: Dict[str, Any]) -> None:
        """Add result summary, error detail, and duration to tool_result."""
        if self._pending_duration_ms is not None:
            data["duration_ms"] = round(self._pending_duration_ms, 1)
            self._pending_duration_ms = None

        if self._capture_tool_output:
            result = data.get("result")
            if result is not None:
                data["result_summary"] = self._truncate(str(result), self._max_output_chars)

            error = data.get("error")
            if error:
                data["error_detail"] = str(error)[: self._max_output_chars]

    def _enrich_llm_response(self, data: Dict[str, Any]) -> None:
        """Extract thinking content and summarize response."""
        content = data.get("content", "")

        # Extract <think> blocks
        think_match = re.findall(r"<think>(.*?)</think>", content, flags=re.DOTALL)
        if think_match:
            data["thinking_content"] = self._truncate(
                " ".join(think_match), self._max_reasoning_chars
            )

        # Response summary (first N chars, excluding think blocks)
        clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if clean:
            data["response_summary"] = self._truncate(clean, 500)

    def _sanitize_args(self, args: Dict[str, Any]) -> Dict[str, str]:
        """Truncate and sanitize tool arguments for trace storage."""
        sanitized = {}
        for k, v in args.items():
            s = str(v)
            if len(s) > 500:
                s = s[:500] + "..."
            sanitized[k] = s
        return sanitized

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        """Truncate text with ellipsis if over limit."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to inner logger."""
        return getattr(self._inner, name)


def create_trace_enricher(
    inner_logger: Any,
    gepa_settings: Any = None,
) -> Any:
    """Create a TraceEnricher if GEPA trace enrichment is enabled.

    Returns the enricher if enabled, otherwise the raw inner_logger
    (zero overhead when disabled). Gated by USE_GEPA master flag.
    """
    if gepa_settings is None:
        return inner_logger

    enabled = getattr(gepa_settings, "enabled", False) and getattr(
        gepa_settings, "capture_reasoning", True
    )

    if not enabled:
        return inner_logger

    return TraceEnricher(
        inner_logger=inner_logger,
        enabled=True,
        max_output_chars=getattr(gepa_settings, "max_output_chars", 2000),
        max_reasoning_chars=getattr(gepa_settings, "max_reasoning_chars", 1000),
        capture_reasoning=getattr(gepa_settings, "capture_reasoning", True),
        capture_tool_args=getattr(gepa_settings, "capture_tool_args", True),
        capture_tool_output=getattr(gepa_settings, "capture_tool_output", True),
    )
