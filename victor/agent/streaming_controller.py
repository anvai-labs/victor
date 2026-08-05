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

"""Streaming Controller - Manages streaming sessions and metrics.

This module extracts streaming coordination from AgentOrchestrator:
- Session lifecycle management
- Metrics collection and aggregation
- Cancellation handling
- History tracking

Design Principles:
- Single Responsibility: Manages streaming state only
- Observable: Provides metrics and history
- Composable: Works with existing StreamHandler
- Thread-safe: Supports concurrent streaming checks
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from victor.agent.stream_handler import StreamMetrics, StreamResult

if TYPE_CHECKING:
    from victor.observability.analytics.streaming_metrics import (
        StreamingMetricsCollector,
    )

logger = logging.getLogger(__name__)


@dataclass
class StreamingSession:
    """Represents an active or completed streaming session."""

    session_id: str
    model: str
    provider: str
    start_time: float
    end_time: Optional[float] = None
    metrics: Optional[StreamMetrics] = None
    cancelled: bool = False
    error: Optional[str] = None

    @property
    def is_active(self) -> bool:
        """Check if session is still active."""
        return self.end_time is None and not self.cancelled

    @property
    def duration(self) -> float:
        """Get session duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "model": self.model,
            "provider": self.provider,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "cancelled": self.cancelled,
            "error": self.error,
            "metrics": (
                {
                    "ttft": self.metrics.time_to_first_token if self.metrics else None,
                    "total_duration": (self.metrics.total_duration if self.metrics else None),
                    "tokens_per_second": (self.metrics.tokens_per_second if self.metrics else None),
                    "total_chunks": self.metrics.total_chunks if self.metrics else 0,
                }
                if self.metrics
                else None
            ),
        }


@dataclass
class StreamingControllerConfig:
    """Configuration for streaming controller."""

    max_history: int = 100
    enable_metrics_collection: bool = True
    default_timeout: float = 300.0


class StreamingController:
    """Manages streaming sessions and metrics.

    This controller handles the lifecycle of streaming sessions:
    - Starting new sessions with metrics tracking
    - Recording first token and completion times
    - Handling cancellation requests
    - Maintaining session history
    - Providing aggregated statistics

    Example:
        controller = StreamingController(config, metrics_collector)

        # Start a session
        session = controller.start_session("gpt-4", "openai")

        # Record events
        controller.record_first_token()
        controller.record_chunk(content_length=100)

        # Complete session
        controller.complete_session()

        # Get history
        history = controller.get_session_history(limit=10)
    """

    def __init__(
        self,
        config: Optional[StreamingControllerConfig] = None,
        metrics_collector: Optional["StreamingMetricsCollector"] = None,
        on_session_complete: Optional[Callable[[StreamingSession], None]] = None,
    ):
        """Initialize streaming controller.

        Args:
            config: Controller configuration
            metrics_collector: Optional analytics metrics collector
            on_session_complete: Callback when session completes
        """
        self.config = config or StreamingControllerConfig()
        self.metrics_collector = metrics_collector
        self.on_session_complete = on_session_complete

        # State
        self._current_session: Optional[StreamingSession] = None
        self._current_metrics: Optional[StreamMetrics] = None
        self._session_history: List[StreamingSession] = []
        self._cancellation_requested = False

        # Thread safety
        self._lock = threading.RLock()

    @property
    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        with self._lock:
            return self._current_session is not None and self._current_session.is_active

    @property
    def current_session(self) -> Optional[StreamingSession]:
        """Get current streaming session."""
        return self._current_session

    @property
    def current_metrics(self) -> Optional[StreamMetrics]:
        """Get current session metrics."""
        return self._current_metrics

    def start_session(self, model: str, provider: str) -> StreamingSession:
        """Start a new streaming session.

        Args:
            model: Model name
            provider: Provider name

        Returns:
            New StreamingSession
        """
        with self._lock:
            # End any existing session
            if self._current_session and self._current_session.is_active:
                self._end_current_session(error="Interrupted by new session")

            # Create new session
            session = StreamingSession(
                session_id=str(uuid.uuid4()),
                model=model,
                provider=provider,
                start_time=time.time(),
            )

            # Initialize metrics
            self._current_metrics = StreamMetrics(start_time=session.start_time)
            self._current_session = session
            self._cancellation_requested = False

            logger.debug(f"Started streaming session: {session.session_id}")
            return session

    def record_first_token(self) -> None:
        """Record the time of first token received."""
        with self._lock:
            if self._current_metrics and self._current_metrics.first_token_time is None:
                self._current_metrics.first_token_time = time.time()
                logger.debug("Recorded first token time")

    def record_chunk(self, content_length: int = 0) -> None:
        """Record a received chunk.

        Args:
            content_length: Length of content in this chunk
        """
        with self._lock:
            if self._current_metrics:
                self._current_metrics.total_chunks += 1
                self._current_metrics.total_content_length += content_length

    def record_tool_call(self) -> None:
        """Record a tool call in the stream."""
        with self._lock:
            if self._current_metrics:
                self._current_metrics.tool_calls_count += 1

    def complete_session(self, result: Optional[StreamResult] = None) -> Optional[StreamingSession]:
        """Complete the current streaming session.

        Args:
            result: Optional stream result with final metrics

        Returns:
            Completed session or None if no active session
        """
        with self._lock:
            if not self._current_session:
                return None

            return self._end_current_session(result=result)

    def cancel_session(self) -> bool:
        """Request cancellation of current session.

        Returns:
            True if cancellation was requested
        """
        with self._lock:
            if not self._current_session or not self._current_session.is_active:
                return False

            self._cancellation_requested = True
            logger.info(f"Cancellation requested for session: {self._current_session.session_id}")
            return True

    def is_cancellation_requested(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancellation_requested

    def _end_current_session(
        self,
        result: Optional[StreamResult] = None,
        error: Optional[str] = None,
    ) -> StreamingSession:
        """End the current session (internal).

        Args:
            result: Optional stream result
            error: Optional error message

        Returns:
            The ended session
        """
        session = self._current_session
        if not session:
            raise RuntimeError("No current session to end")

        # Finalize metrics
        if self._current_metrics:
            self._current_metrics.end_time = time.time()
            session.metrics = self._current_metrics

        session.end_time = time.time()
        session.cancelled = self._cancellation_requested
        session.error = error

        # Record to analytics collector if available
        if (
            self.config.enable_metrics_collection
            and self.metrics_collector
            and self._current_metrics
        ):
            self._record_to_analytics_collector(session)

        # Add to history
        self._session_history.append(session)
        if len(self._session_history) > self.config.max_history:
            self._session_history = self._session_history[-self.config.max_history :]

        # Invoke callback
        if self.on_session_complete:
            try:
                self.on_session_complete(session)
            except Exception as e:
                logger.warning(f"on_session_complete callback failed: {e}")

        # Clear current session
        self._current_session = None
        self._current_metrics = None
        self._cancellation_requested = False

        logger.debug(f"Ended streaming session: {session.session_id}")
        return session

    def _record_to_analytics_collector(self, session: StreamingSession) -> None:
        """Record session to analytics collector."""
        try:
            from victor.observability.analytics.streaming_metrics import (
                StreamMetrics as AnalyticsMetrics,
            )

            if not session.metrics:
                return

            # Use actual token count from API when available, fall back to estimate
            # StreamMetrics.effective_total_tokens handles this logic
            total_tokens = session.metrics.effective_total_tokens

            analytics_metrics = AnalyticsMetrics(
                request_id=session.session_id,
                start_time=session.start_time,
                first_token_time=session.metrics.first_token_time,
                last_token_time=session.end_time,
                total_chunks=session.metrics.total_chunks,
                total_tokens=total_tokens,
                model=session.model,
                provider=session.provider,
            )
            if self.metrics_collector:
                self.metrics_collector.record_metrics(analytics_metrics)

            # Log whether we used actual or estimated tokens
            if session.metrics.has_actual_usage:
                logger.debug(f"Recorded actual token usage: {total_tokens}")
            else:
                logger.debug(f"Recorded estimated token usage: {total_tokens} (no API data)")
        except Exception as e:
            logger.debug(f"Failed to record to metrics collector: {e}")

    def get_session_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent session history.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session dictionaries
        """
        with self._lock:
            sessions = self._session_history[-limit:]
            return [s.to_dict() for s in reversed(sessions)]

    def get_metrics_summary(self) -> Optional[Dict[str, Any]]:
        """Get aggregated metrics summary.

        Returns:
            Dictionary with aggregated metrics or None if no collector
        """
        if not self.metrics_collector:
            return None

        try:
            summary = self.metrics_collector.get_summary()
            if hasattr(summary, "__dict__"):
                return dict(vars(summary))
            return dict(summary) if summary else None
        except Exception as e:
            logger.warning(f"Failed to get metrics summary: {e}")
            return None

    def get_current_session_info(self) -> Optional[Dict[str, Any]]:
        """Get information about current session.

        Returns:
            Current session info or None
        """
        with self._lock:
            if not self._current_session:
                return None

            return {
                "session_id": self._current_session.session_id,
                "model": self._current_session.model,
                "provider": self._current_session.provider,
                "duration": self._current_session.duration,
                "is_active": self._current_session.is_active,
                "cancellation_requested": self._cancellation_requested,
                "chunks_received": (
                    self._current_metrics.total_chunks if self._current_metrics else 0
                ),
                "content_length": (
                    self._current_metrics.total_content_length if self._current_metrics else 0
                ),
            }

    def reset(self) -> None:
        """Reset controller state."""
        with self._lock:
            if self._current_session and self._current_session.is_active:
                self._end_current_session(error="Reset requested")
            self._session_history.clear()
            logger.info("Streaming controller reset")

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall streaming statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            total_sessions = len(self._session_history)
            completed = sum(1 for s in self._session_history if not s.cancelled and not s.error)
            cancelled = sum(1 for s in self._session_history if s.cancelled)
            errored = sum(1 for s in self._session_history if s.error and not s.cancelled)

            avg_duration = 0.0
            avg_ttft = 0.0
            if total_sessions > 0:
                durations = [s.duration for s in self._session_history]
                avg_duration = sum(durations) / len(durations)

                ttfts = [
                    s.metrics.time_to_first_token
                    for s in self._session_history
                    if s.metrics and s.metrics.time_to_first_token
                ]
                if ttfts:
                    avg_ttft = sum(ttfts) / len(ttfts)

            return {
                "total_sessions": total_sessions,
                "completed": completed,
                "cancelled": cancelled,
                "errored": errored,
                "average_duration_seconds": round(avg_duration, 3),
                "average_ttft_seconds": round(avg_ttft, 3),
                "is_streaming": self.is_streaming,
            }
