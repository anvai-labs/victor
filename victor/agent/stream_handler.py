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

"""Stream handling for processing LLM streaming responses."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from victor.providers.base import StreamChunk

logger = logging.getLogger(__name__)


@dataclass
class StreamMetrics:
    """Metrics collected during streaming.

    Token tracking follows SOLID principles:
    - Uses actual token counts from provider API when available
    - Falls back to estimation (content_length / 4) only when API doesn't provide counts
    - has_actual_usage flag indicates whether counts are actual or estimated

    Cost tracking:
    - Optional cost calculation based on provider pricing config
    - Supports cache token costs (Anthropic)
    - cost_calculated flag indicates whether costs are available

    Observability features (consolidated from victor.observability.analytics.streaming_metrics):
    - chunk_intervals: List of time intervals between chunks for P50/P95/P99 analysis
    - metadata: Dict for arbitrary metadata (request_id, model, provider, etc.)
    - errors: List of errors encountered during streaming
    """

    start_time: float = 0.0
    first_token_time: Optional[float] = None
    end_time: float = 0.0
    total_chunks: int = 0
    total_content_length: int = 0
    tool_calls_count: int = 0

    # Actual token usage from provider API (preferred)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    has_actual_usage: bool = False  # True when counts came from API, False when estimated

    # Cache tokens (Anthropic-specific)
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    # Separately-reported reasoning tokens (Gemini thoughtsTokenCount, OpenAI
    # reasoning_tokens). May be folded into completion_tokens depending on the
    # provider — see calculate_cost for the folding-aware pricing rule.
    reasoning_tokens: int = 0

    # Cost tracking (USD)
    input_cost: float = 0.0
    output_cost: float = 0.0
    cache_cost: float = 0.0
    total_cost: float = 0.0
    cost_calculated: bool = False

    # Wire-truth latency from sandhi's typed boundary (W3b) — preferred over
    # client wall-clock when present (absent on non-sandhi providers and on
    # sandhi-gateway <= 0.1.4).
    wire_duration_ms: Optional[int] = None
    wire_ttft_ms: Optional[int] = None

    # Observability features (consolidated)
    chunk_intervals: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def record_usage(self, usage: Optional[Dict[str, Any]]) -> None:
        """Record token usage from provider API response.

        Args:
            usage: Usage dict from provider (prompt_tokens, completion_tokens, total_tokens,
                   cache_read_input_tokens, cache_creation_input_tokens)
        """
        if usage:
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)

            # Cache tokens (Anthropic-style naming)
            self.cache_read_tokens += usage.get("cache_read_input_tokens", 0)
            self.cache_write_tokens += usage.get("cache_creation_input_tokens", 0)
            self.reasoning_tokens += usage.get("reasoning_tokens", 0)

            # Mark as actual usage if we got non-zero values
            if self.prompt_tokens > 0 or self.completion_tokens > 0:
                self.has_actual_usage = True

    def record_wire_latency(self, sandhi_usage: Optional[Dict[str, Any]]) -> None:
        """Record wire-truth latency from the terminal chunk's diagnostics."""
        if not isinstance(sandhi_usage, dict):
            return
        duration = sandhi_usage.get("duration_ms")
        if isinstance(duration, (int, float)) and duration >= 0:
            self.wire_duration_ms = int(duration)
        ttft = sandhi_usage.get("time_to_first_token_ms")
        if isinstance(ttft, (int, float)) and ttft >= 0:
            self.wire_ttft_ms = int(ttft)

    def calculate_cost(self, capabilities: Any) -> None:
        """Calculate cost using provider capabilities.

        Args:
            capabilities: ProviderMetricsCapabilities with pricing info
        """
        if not capabilities or not capabilities.cost_enabled:
            return

        # Folding invariant (mirrors sandhi billable()): providers either fold
        # reasoning tokens into completion_tokens (OpenAI, Anthropic) or report
        # them separately (Gemini). Unfolded reasoning is billed output that
        # completion_tokens does not contain — price it at the output rate.
        # Detection is total: reasoning that cannot fit inside completion_tokens
        # is unfolded; otherwise assume folded and never double-count.
        unfolded_reasoning = (
            self.reasoning_tokens if self.reasoning_tokens > self.completion_tokens else 0
        )
        costs = capabilities.calculate_cost(
            self.prompt_tokens,
            self.completion_tokens + unfolded_reasoning,
            self.cache_read_tokens,
            self.cache_write_tokens,
        )
        self.input_cost = costs["input_cost"]
        self.output_cost = costs["output_cost"]
        self.cache_cost = costs["cache_cost"]
        self.total_cost = costs["total_cost"]
        self.cost_calculated = True

    @property
    def effective_total_tokens(self) -> int:
        """Get total tokens - actual from API or estimated from content length.

        Returns actual token count when provider supplied it,
        otherwise falls back to estimation.
        """
        if self.has_actual_usage and self.total_tokens > 0:
            return self.total_tokens
        # Fallback: estimate from content length (~4 chars per token)
        return self.total_content_length // 4

    @property
    def time_to_first_token(self) -> Optional[float]:
        """Time from start to first token (TTFT), in seconds.

        Prefers sandhi's wire-truth measurement (W3b) over client wall-clock —
        the latter folds host-language scheduling noise into provider latency.
        """
        if self.wire_ttft_ms is not None:
            return self.wire_ttft_ms / 1000.0
        if self.first_token_time and self.start_time:
            return self.first_token_time - self.start_time
        return None

    @property
    def total_duration(self) -> float:
        """Total streaming duration, in seconds (wire truth when available)."""
        if self.wire_duration_ms is not None:
            return self.wire_duration_ms / 1000.0
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def tokens_per_second(self) -> float:
        """Tokens per second (uses actual count when available)."""
        duration = self.total_duration
        if duration > 0:
            return self.effective_total_tokens / duration
        return 0.0

    def format_cost(self) -> str:
        """Format cost for display.

        Returns:
            Cost string like "$0.0123" or "cost n/a"
        """
        if self.cost_calculated:
            return f"${self.total_cost:.4f}"
        return "cost n/a"

    # Observability methods (consolidated from victor.observability.analytics.streaming_metrics)

    def avg_chunk_interval_ms(self) -> float:
        """Calculate average chunk interval in milliseconds.

        Returns:
            Average interval between chunks, or 0.0 if no intervals recorded
        """
        if not self.chunk_intervals:
            return 0.0
        return (sum(self.chunk_intervals) / len(self.chunk_intervals)) * 1000

    def p50_chunk_interval_ms(self) -> float:
        """Calculate median (P50) chunk interval in milliseconds.

        Returns:
            Median interval between chunks, or 0.0 if no intervals recorded
        """
        if not self.chunk_intervals:
            return 0.0
        sorted_intervals = sorted(self.chunk_intervals)
        mid = len(sorted_intervals) // 2
        if len(sorted_intervals) % 2 == 0:
            median = (sorted_intervals[mid - 1] + sorted_intervals[mid]) / 2
        else:
            median = sorted_intervals[mid]
        return median * 1000

    def p95_chunk_interval_ms(self) -> float:
        """Calculate P95 chunk interval in milliseconds.

        Returns:
            95th percentile interval between chunks, or 0.0 if no intervals recorded
        """
        if len(self.chunk_intervals) < 20:
            return 0.0
        sorted_intervals = sorted(self.chunk_intervals)
        idx = int(len(sorted_intervals) * 0.95)
        return sorted_intervals[idx] * 1000

    def p99_chunk_interval_ms(self) -> float:
        """Calculate P99 chunk interval in milliseconds.

        Returns:
            99th percentile interval between chunks, or 0.0 if no intervals recorded
        """
        if len(self.chunk_intervals) < 100:
            return 0.0
        sorted_intervals = sorted(self.chunk_intervals)
        idx = int(len(sorted_intervals) * 0.99)
        return sorted_intervals[idx] * 1000

    def record_chunk_interval(self, interval_seconds: float) -> None:
        """Record a chunk interval for percentile analysis.

        Args:
            interval_seconds: Time interval since last chunk in seconds
        """
        self.chunk_intervals.append(interval_seconds)

    def record_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Record an error encountered during streaming.

        Args:
            error: The exception that occurred
            context: Additional context about the error
        """
        self.errors.append(
            {
                "type": type(error).__name__,
                "message": str(error),
                "context": context,
            }
        )


@dataclass
class StreamResult:
    """Result from processing a stream."""

    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: Optional[str] = None
    metrics: StreamMetrics = field(default_factory=StreamMetrics)
    error: Optional[str] = None


class StreamHandler:
    """Handles streaming responses from LLM providers.

    Responsibilities:
    - Process stream chunks and accumulate content
    - Handle tool calls in streaming responses
    - Collect streaming metrics (TTFT, throughput)
    - Support callbacks for real-time updates
    - Handle stream interruption and cleanup
    """

    def __init__(
        self,
        on_content: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_complete: Optional[Callable[[StreamResult], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        timeout: float = 300.0,
    ):
        """Initialize stream handler.

        Args:
            on_content: Callback for each content chunk
            on_tool_call: Callback when a tool call is detected
            on_complete: Callback when stream completes
            on_error: Callback on stream error
            timeout: Maximum time to wait for stream (seconds)
        """
        self.on_content = on_content
        self.on_tool_call = on_tool_call
        self.on_complete = on_complete
        self.on_error = on_error
        self.timeout = timeout

        # State
        self._cancelled = False
        self._current_content = ""
        self._tool_calls: List[Dict[str, Any]] = []
        self._pending_tool_call: Dict[str, Any] = {}

    async def process_stream(
        self,
        stream: AsyncIterator[StreamChunk],
    ) -> StreamResult:
        """Process a stream of chunks and return accumulated result.

        Args:
            stream: Async iterator of StreamChunk objects

        Returns:
            StreamResult with accumulated content and metrics
        """
        metrics = StreamMetrics(start_time=time.time())
        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        stop_reason: Optional[str] = None
        error: Optional[str] = None

        try:
            # Use wait_for pattern for Python 3.9/3.10 compatibility
            async def _process_chunks() -> None:
                nonlocal stop_reason
                async for chunk in stream:
                    if self._cancelled:
                        logger.info("Stream cancelled by user")
                        break

                    metrics.total_chunks += 1

                    # Handle content
                    if chunk.content:
                        if metrics.first_token_time is None:
                            metrics.first_token_time = time.time()

                        content_parts.append(chunk.content)
                        metrics.total_content_length += len(chunk.content)

                        if self.on_content:
                            try:
                                self.on_content(chunk.content)
                            except Exception as e:
                                logger.warning(f"Content callback error: {e}")

                    # Handle tool calls
                    if chunk.tool_calls:
                        for tc in chunk.tool_calls:
                            tool_calls.append(tc)
                            metrics.tool_calls_count += 1

                            if self.on_tool_call:
                                try:
                                    self.on_tool_call(tc)
                                except Exception as e:
                                    logger.warning(f"Tool call callback error: {e}")

                    # Check for completion
                    if chunk.stop_reason:
                        stop_reason = chunk.stop_reason

                    if chunk.is_final:
                        break

            await asyncio.wait_for(_process_chunks(), timeout=self.timeout)

        except asyncio.TimeoutError:
            error = f"Stream timed out after {self.timeout}s"
            logger.error(error)
            if self.on_error:
                self.on_error(TimeoutError(error))

        except Exception as e:
            error = str(e)
            logger.error(f"Stream error: {e}")
            if self.on_error:
                self.on_error(e)

        finally:
            metrics.end_time = time.time()

        result = StreamResult(
            content="".join(content_parts),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            metrics=metrics,
            error=error,
        )

        if self.on_complete:
            try:
                self.on_complete(result)
            except Exception as e:
                logger.warning(f"Complete callback error: {e}")

        return result

    def cancel(self) -> None:
        """Cancel the current stream processing."""
        self._cancelled = True

    def reset(self) -> None:
        """Reset handler state for reuse."""
        self._cancelled = False
        self._current_content = ""
        self._tool_calls.clear()
        self._pending_tool_call.clear()


class StreamBuffer:
    """Buffer for accumulating streamed tool call arguments.

    Some providers send tool call arguments in chunks. This buffer
    accumulates them until the tool call is complete.
    """

    def __init__(self) -> None:
        self._buffers: Dict[str, Dict[str, Any]] = {}

    def add_chunk(
        self,
        tool_call_id: str,
        chunk: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Add a chunk to the buffer.

        Args:
            tool_call_id: ID of the tool call
            chunk: Partial tool call data

        Returns:
            Complete tool call if finished, None otherwise
        """
        if tool_call_id not in self._buffers:
            self._buffers[tool_call_id] = {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "",
                    "arguments": "",
                },
            }

        buffer = self._buffers[tool_call_id]

        # Update function name if provided
        if "function" in chunk:
            func_data = chunk["function"]
            if "name" in func_data and func_data["name"]:
                buffer["function"]["name"] = func_data["name"]
            if "arguments" in func_data:
                buffer["function"]["arguments"] += func_data["arguments"]

        # Check if complete (has name and arguments ends with valid JSON)
        if buffer["function"]["name"] and buffer["function"]["arguments"]:
            args = buffer["function"]["arguments"].strip()
            if args.endswith("}") or args.endswith("]"):
                # Likely complete
                complete = buffer.copy()
                del self._buffers[tool_call_id]
                return complete

        return None

    def flush(self) -> List[Dict[str, Any]]:
        """Flush all buffered tool calls (may be incomplete)."""
        results = list(self._buffers.values())
        self._buffers.clear()
        return results

    def clear(self) -> None:
        """Clear all buffers."""
        self._buffers.clear()
