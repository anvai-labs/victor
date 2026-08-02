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

"""Observability Bridge - Facade for Victor observability system.

This module provides a unified facade (Facade Pattern) over all event emitters,
simplifying the API for emitting events across different categories.

SOLID Principles:
- SRP: Coordinates emitters, doesn't implement emission logic
- OCP: Extensible via adding new emitters
- LSP: All emitters are substitutable via Protocol interfaces
- ISP: Focused interface, not bloated
- DIP: Depends on Protocol interfaces, not concrete implementations

Design Patterns:
- Facade: Simplifies access to complex emitter subsystem
- Singleton: Single bridge instance for consistency
- Dependency Injection: Accepts custom emitters for testing
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import contextmanager

from victor.core.events import MessagingEvent, ObservabilityBus, get_observability_bus
from victor.observability.emitters import (
    ToolEventEmitter,
    ModelEventEmitter,
    StateEventEmitter,
    LifecycleEventEmitter,
    ErrorEventEmitter,
)
from victor.config.settings import get_project_paths

logger = logging.getLogger(__name__)


class ObservabilityBridge:
    """Facade for Victor observability system.

    Provides a simplified interface to all event emitters.
    Implements singleton pattern for global access.

    Example:
        >>> bridge = ObservabilityBridge.get_instance()
        >>> bridge.tool_start("read_file", {"path": "file.txt"})
        >>> # ... execute tool ...
        >>> bridge.tool_end("read_file", 150.0, result="content")
    """

    _instance: Optional[ObservabilityBridge] = None

    def __init__(
        self,
        tool_emitter: Optional[ToolEventEmitter] = None,
        model_emitter: Optional[ModelEventEmitter] = None,
        state_emitter: Optional[StateEventEmitter] = None,
        lifecycle_emitter: Optional[LifecycleEventEmitter] = None,
        error_emitter: Optional[ErrorEventEmitter] = None,
        event_bus: Optional[ObservabilityBus] = None,
    ):
        """Initialize the observability bridge.

        Args:
            tool_emitter: Optional tool emitter (for testing/injection)
            model_emitter: Optional model emitter (for testing/injection)
            state_emitter: Optional state emitter (for testing/injection)
            lifecycle_emitter: Optional lifecycle emitter (for testing/injection)
            error_emitter: Optional error emitter (for testing/injection)
            event_bus: Optional ObservabilityBus for all emitters
        """
        # Create emitters with shared EventBus
        bus = event_bus or get_observability_bus()

        self._tool_emitter = tool_emitter or ToolEventEmitter(bus=bus)
        self._model_emitter = model_emitter or ModelEventEmitter(bus=bus)
        self._state_emitter = state_emitter or StateEventEmitter(bus=bus)
        self._lifecycle_emitter = lifecycle_emitter or LifecycleEventEmitter(bus=bus)
        self._error_emitter = error_emitter or ErrorEventEmitter(bus=bus)

        self._event_bus = bus
        self._jsonl_exporter = None

        self._enabled = True
        self._session_start_time: Optional[float] = None
        self._session_id: Optional[str] = None

    @classmethod
    def get_instance(cls) -> ObservabilityBridge:
        """Get or create singleton bridge instance.

        Returns:
            ObservabilityBridge singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    def setup_jsonl_exporter(self, log_path: Optional[Path] = None) -> None:
        """Setup JSONL exporter for event logging.

        This enables event export to a JSONL file for dashboard visualization.
        The exporter writes events in the format expected by the dashboard's
        file watcher.

        Args:
            log_path: Optional custom log path. Defaults to ~/.victor/metrics/victor.jsonl
        """
        if self._jsonl_exporter is not None:
            # Already setup
            return

        # Determine log path
        if log_path is None:
            try:
                paths = get_project_paths()
                log_path = paths.global_victor_dir / "metrics" / "victor.jsonl"
            except Exception:
                # Fallback if paths not initialized
                from pathlib import Path

                log_path = Path.home() / ".victor" / "metrics" / "victor.jsonl"

        # Ensure metrics directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Import JsonLineExporter
        from victor.observability.exporters import JsonLineExporter

        # Create exporter
        self._jsonl_exporter = JsonLineExporter(
            path=log_path,
            buffer_size=10,  # Buffer up to 10 events
            flush_interval_seconds=60,  # Flush every 60 seconds (whichever is first)
            append=True,  # Append to existing log
        )

        # Register with ObservabilityBus
        self._event_bus.add_exporter(self._jsonl_exporter)

        logger.info(f"JSONL event logging enabled: {log_path}")

    def disable_jsonl_exporter(self) -> None:
        """Disable JSONL exporter and close log file."""
        if self._jsonl_exporter is not None:
            self._event_bus.remove_exporter(self._jsonl_exporter)
            self._jsonl_exporter.close()
            self._jsonl_exporter = None
            logger.info("JSONL event logging disabled")

    # =========================================================================
    # Convenience Methods - Tool Events
    # =========================================================================

    def tool_start(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        **metadata: Any,
    ) -> None:
        """Emit tool start event.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            **metadata: Additional metadata
        """
        if self._enabled:
            self._tool_emitter.tool_start(tool_name, arguments, **metadata)

    def tool_end(
        self,
        tool_name: str,
        duration_ms: float,
        result: Optional[Any] = None,
        **metadata: Any,
    ) -> None:
        """Emit tool end event (success).

        Args:
            tool_name: Name of the tool
            duration_ms: Execution duration in milliseconds
            result: Tool result
            **metadata: Additional metadata
        """
        if self._enabled:
            self._tool_emitter.tool_end(tool_name, duration_ms, result, **metadata)

    def tool_failure(
        self,
        tool_name: str,
        duration_ms: float,
        error: Exception,
        **metadata: Any,
    ) -> None:
        """Emit tool failure event.

        Args:
            tool_name: Name of the tool
            duration_ms: Execution duration before failure
            error: The exception
            **metadata: Additional metadata
        """
        if self._enabled:
            self._tool_emitter.tool_failure(tool_name, duration_ms, error, **metadata)

    @contextmanager
    def track_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        **metadata: Any,
    ):
        """Context manager for tracking tool execution.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            **metadata: Additional metadata

        Yields:
            None

        Example:
            >>> with bridge.track_tool("read_file", {"path": "file.txt"}):
            ...     result = await tool(**arguments)
        """
        if self._enabled:
            with self._tool_emitter.track_tool(tool_name, arguments, **metadata):
                yield
        else:
            yield

    # =========================================================================
    # Convenience Methods - Model Events
    # =========================================================================

    def model_request(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        **metadata: Any,
    ) -> None:
        """Emit model request event.

        Args:
            provider: Model provider
            model: Model name
            prompt_tokens: Prompt token count
            **metadata: Additional metadata
        """
        if self._enabled:
            self._model_emitter.model_request(provider, model, prompt_tokens, **metadata)

    def model_response(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        **metadata: Any,
    ) -> None:
        """Emit model response event.

        Args:
            provider: Model provider
            model: Model name
            prompt_tokens: Prompt token count
            completion_tokens: Completion token count
            latency_ms: Request latency
            **metadata: Additional metadata
        """
        if self._enabled:
            self._model_emitter.model_response(
                provider,
                model,
                prompt_tokens,
                completion_tokens,
                latency_ms,
                **metadata,
            )

    def model_streaming_delta(
        self,
        provider: str,
        model: str,
        delta: str,
        **metadata: Any,
    ) -> None:
        """Emit streaming delta event.

        Args:
            provider: Model provider
            model: Model name
            delta: Text delta
            **metadata: Additional metadata
        """
        if self._enabled:
            self._model_emitter.model_streaming_delta(provider, model, delta, **metadata)

    def model_error(
        self,
        provider: str,
        model: str,
        error: Exception,
        **metadata: Any,
    ) -> None:
        """Emit model error event.

        Args:
            provider: Model provider
            model: Model name
            error: The exception
            **metadata: Additional metadata
        """
        if self._enabled:
            self._model_emitter.model_error(provider, model, error, **metadata)

    # =========================================================================
    # Convenience Methods - State Events
    # =========================================================================

    def state_transition(
        self,
        old_stage: str,
        new_stage: str,
        confidence: float,
        **metadata: Any,
    ) -> None:
        """Emit state transition event.

        Args:
            old_stage: Previous state
            new_stage: New state
            confidence: Transition confidence (0.0 to 1.0)
            **metadata: Additional metadata
        """
        if self._enabled:
            self._state_emitter.state_transition(old_stage, new_stage, confidence, **metadata)

    # =========================================================================
    # Convenience Methods - Lifecycle Events
    # =========================================================================

    def session_start(
        self,
        session_id: str,
        **metadata: Any,
    ) -> None:
        """Emit session start event.

        Args:
            session_id: Unique session identifier
            **metadata: Additional metadata
        """
        if self._enabled:
            self._session_id = session_id
            self._session_start_time = time.time()
            self._lifecycle_emitter.session_start(session_id, **metadata)

    def session_end(
        self,
        session_id: Optional[str] = None,
        **metadata: Any,
    ) -> None:
        """Emit session end event.

        Args:
            session_id: Unique session identifier (uses tracked session if None)
            **metadata: Additional metadata
        """
        if not self._enabled:
            return

        sid = session_id or self._session_id
        if not sid:
            logger.warning("session_end called but no session_id tracked")
            return

        if self._session_start_time:
            duration_ms = (time.time() - self._session_start_time) * 1000
        else:
            duration_ms = 0.0

        self._lifecycle_emitter.session_end(sid, duration_ms, **metadata)
        self._session_id = None
        self._session_start_time = None

    @contextmanager
    def track_session(
        self,
        session_id: str,
        **metadata: Any,
    ):
        """Context manager for tracking session lifecycle.

        Args:
            session_id: Unique session identifier
            **metadata: Additional metadata

        Yields:
            None

        Example:
            >>> with bridge.track_session("session-123", agent_id="agent-1"):
            ...     # ... session work ...
        """
        if self._enabled:
            with self._lifecycle_emitter.track_session(session_id, **metadata):
                yield
        else:
            yield

    # =========================================================================
    # Convenience Methods - Error Events
    # =========================================================================

    def error(
        self,
        error: Exception,
        recoverable: bool,
        context: Optional[Dict[str, Any]] = None,
        **metadata: Any,
    ) -> None:
        """Emit error event.

        Args:
            error: The exception
            recoverable: Whether error is recoverable
            context: Additional error context
            **metadata: Additional metadata
        """
        if self._enabled:
            self._error_emitter.error(error, recoverable, context, **metadata)

    # =========================================================================
    # Control Methods
    # =========================================================================

    def enable(self) -> None:
        """Enable all event emission."""
        self._enabled = True
        self._tool_emitter.enable()
        self._model_emitter.enable()
        self._state_emitter.enable()
        self._lifecycle_emitter.enable()
        self._error_emitter.enable()

    def disable(self) -> None:
        """Disable all event emission."""
        self._enabled = False
        self._tool_emitter.disable()
        self._model_emitter.disable()
        self._state_emitter.disable()
        self._lifecycle_emitter.disable()
        self._error_emitter.disable()

    def is_enabled(self) -> bool:
        """Check if event emission is enabled.

        Returns:
            True if enabled, False otherwise
        """
        return self._enabled

    # =========================================================================
    # Accessor Methods (for testing/advanced use)
    # =========================================================================

    @property
    def tool(self) -> ToolEventEmitter:
        """Get tool emitter instance.

        Returns:
            ToolEventEmitter instance
        """
        return self._tool_emitter

    @property
    def model(self) -> ModelEventEmitter:
        """Get model emitter instance.

        Returns:
            ModelEventEmitter instance
        """
        return self._model_emitter

    @property
    def state(self) -> StateEventEmitter:
        """Get state emitter instance.

        Returns:
            StateEventEmitter instance
        """
        return self._state_emitter

    @property
    def lifecycle(self) -> LifecycleEventEmitter:
        """Get lifecycle emitter instance.

        Returns:
            LifecycleEventEmitter instance
        """
        return self._lifecycle_emitter

    @property
    def error_emitter(self) -> ErrorEventEmitter:
        """Get error emitter instance.

        Returns:
            ErrorEventEmitter instance
        """
        return self._error_emitter
