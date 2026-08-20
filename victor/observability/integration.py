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

"""Integration helpers to wire observability into Victor components.

This module provides integration points for connecting the EventBus
to various Victor components (orchestrator, tool pipeline, state machine).

Design Pattern: Mediator
The ObservabilityIntegration class acts as a mediator between
Victor components and the observability subsystem.

Example:
    from victor.observability.integration import ObservabilityIntegration

    # Wire up observability for an orchestrator
    integration = ObservabilityIntegration()
    integration.wire_orchestrator(orchestrator)

    # Or use the convenience function
    from victor.observability.integration import setup_observability
    setup_observability(orchestrator)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from victor.core.events import ObservabilityBus, get_observability_bus
from victor.core.events.emit_helper import emit_event_sync
from victor.observability.hooks import StateHookManager, TransitionHistory
from victor.observability.request_correlation import get_request_correlation_id

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)


class ObservabilityIntegration:
    """Integrates EventBus with Victor components.

    This class wires the observability subsystem into various
    Victor components, enabling automatic event emission.

    Features:
    - Tool execution events
    - State transition events
    - Model response events
    - Error events
    - Session lifecycle events

    Example:
        integration = ObservabilityIntegration()
        integration.wire_orchestrator(orchestrator)

        # Events are now automatically emitted
    """

    def __init__(
        self,
        event_bus: Optional["ObservabilityBus"] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Initialize the integration.

        Args:
            event_bus: ObservabilityBus to use (default: singleton).
            session_id: Optional session ID for correlation.
        """
        self._bus = event_bus or get_observability_bus()
        self._session_id = session_id
        self._wired_components: List[str] = []
        self._tool_start_times: Dict[str, float] = {}
        self._state_hook_manager: Optional[StateHookManager] = None

    @property
    def event_bus(self) -> "ObservabilityBus":
        """Get the event bus."""
        return self._bus

    @property
    def state_hook_manager(self) -> Optional[StateHookManager]:
        """Get the StateHookManager if wired to a state machine.

        Returns:
            StateHookManager instance or None.
        """
        return self._state_hook_manager

    def _schedule_emit(
        self,
        *,
        topic: str,
        data: Dict[str, Any],
        correlation_id: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Schedule an async observability emit on the active event loop."""
        event_name = description or topic
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("No event loop, skipping %s event emission", event_name)
            return

        try:
            loop.create_task(
                self._bus.emit(
                    topic=topic,
                    data=data,
                    correlation_id=correlation_id,
                )
            )
        except Exception as e:
            logger.debug("Failed to emit %s event: %s", event_name, e)

    @property
    def state_transition_history(self) -> Optional[TransitionHistory]:
        """Get the transition history from the state hook manager.

        Returns:
            TransitionHistory if available, None otherwise.
        """
        if self._state_hook_manager:
            return self._state_hook_manager.history
        return None

    def get_state_transition_metrics(self) -> Dict[str, Any]:
        """Get metrics about state transitions.

        Returns a summary of transition patterns including:
        - Total transitions
        - Unique transition paths
        - Stage visit counts
        - Cycle detection status
        - Average time per stage

        Returns:
            Dictionary with transition metrics.
        """
        if not self._state_hook_manager or not self._state_hook_manager.history:
            return {
                "total_transitions": 0,
                "unique_stages_visited": 0,
                "has_cycles": False,
                "current_stage": None,
                "stage_sequence": [],
            }

        history = self._state_hook_manager.history
        stage_sequence = history.get_stage_sequence()
        unique_stages = set(stage_sequence)

        # Calculate per-stage average durations
        stage_avg_durations = {}
        all_durations: List[float] = []
        for stage in unique_stages:
            avg = history.get_average_duration(stage)
            if avg is not None:
                stage_avg_durations[stage] = avg
                all_durations.append(avg)

        # Calculate overall average from stage averages
        overall_avg = sum(all_durations) / len(all_durations) if all_durations else None

        return {
            "total_transitions": len(history),
            "unique_stages_visited": len(unique_stages),
            "has_cycles": history.has_cycle(),
            "current_stage": history.current_stage,
            "stage_sequence": stage_sequence,
            "stage_average_durations_ms": stage_avg_durations,
            "overall_average_duration_ms": overall_avg,
            "stage_visit_counts": {
                stage: history.get_stage_visit_count(stage) for stage in unique_stages
            },
        }

    def set_session_id(self, session_id: str) -> None:
        """Set the session ID for event correlation.

        Args:
            session_id: Session identifier.
        """
        self._session_id = session_id
        # Note: The new ObservabilityBus doesn't have set_session_id()
        # Session ID is stored in self._session_id for use in event emissions

    def wire_orchestrator(self, orchestrator: "AgentOrchestrator") -> None:
        """Wire EventBus into an AgentOrchestrator.

        This connects:
        - Tool execution events
        - State machine transitions
        - Session lifecycle events

        Args:
            orchestrator: Orchestrator to wire.
        """
        # Wire state machine hooks
        if hasattr(orchestrator, "conversation_state"):
            self.wire_state_machine(orchestrator.conversation_state)

        # Store reference for event emission via public interfaces only.
        wired = False
        set_observability = getattr(orchestrator, "set_observability", None)
        if callable(set_observability):
            set_observability(self)
            wired = True
        elif hasattr(orchestrator, "observability"):
            try:
                orchestrator.observability = self
                wired = True
            except Exception as e:
                logger.debug("Failed to set observability via property: %s", e)

        if not wired:
            logger.warning(
                "Could not wire observability: orchestrator lacks public observability setter"
            )
            return

        self._wired_components.append("orchestrator")

        logger.debug("Wired observability into orchestrator")

    def wire_state_machine(self, state_machine: Any) -> None:
        """Wire EventBus into a state machine with history-aware hooks.

        Uses history-aware hooks to emit rich state events that include:
        - Transition analytics (cycle detection, visit counts)
        - Stage sequence tracking
        - Duration measurements

        Args:
            state_machine: ConversationStateMachine or compatible.
        """
        hook_manager = StateHookManager(enable_history=True, history_max_size=100)

        @hook_manager.on_transition_with_history
        def emit_transition_with_analytics(
            old_stage: str,
            new_stage: str,
            context: Dict[str, Any],
            history: TransitionHistory,
        ) -> None:
            """Emit state change with rich analytics from history."""
            # Build enhanced context with history analytics
            enhanced_context = {
                **context,
                "has_cycle": history.has_cycle(),
                "visit_count": history.get_stage_visit_count(new_stage),
                "stage_sequence": history.get_stage_sequence(),
                "transition_count": len(history),
            }

            # Get last record for duration info if available
            last_records = history.get_last(1)
            if last_records and last_records[0].duration_ms is not None:
                enhanced_context["stage_duration_ms"] = last_records[0].duration_ms

            # Emit state transition event
            self._schedule_emit(
                topic="state.stage_changed",
                data={
                    "old_stage": old_stage,
                    "new_stage": new_stage,
                    "confidence": context.get("confidence", 1.0),
                    **enhanced_context,
                },
                description="state.stage_changed",
            )

            # Emit warning event if cycle detected (potential infinite loop)
            if history.has_cycle():
                cycle_count = history.get_stage_visit_count(new_stage)
                if cycle_count >= 3:
                    # Emit cycle warning event
                    self._schedule_emit(
                        topic="error.cycle_warning",
                        data={
                            "stage": new_stage,
                            "visit_count": cycle_count,
                            "sequence": history.get_stage_sequence()[-5:],
                            "severity": "warning",
                            "category": "error",
                        },
                        description="error.cycle_warning",
                    )

        if hasattr(state_machine, "set_hooks"):
            state_machine.set_hooks(hook_manager)

        self._state_hook_manager = hook_manager
        self._wired_components.append("state_machine")
        logger.debug("Wired observability into state machine with history-aware hooks")

    # =========================================================================
    # Tool Events
    # =========================================================================

    def on_tool_start(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_id: Optional[str] = None,
    ) -> None:
        """Emit a tool start event.

        Args:
            tool_name: Name of the tool.
            arguments: Tool arguments.
            tool_id: Optional tool call ID.
        """
        self._tool_start_times[tool_id or tool_name] = time.time()
        correlation_id = get_request_correlation_id() or self._session_id
        self._schedule_emit(
            topic="tool.start",
            data={
                "tool_name": tool_name,
                "arguments": arguments,
                "tool_id": tool_id,
                "category": "tool",
            },
            correlation_id=correlation_id,
            description="tool.start",
        )

    def on_tool_end(
        self,
        tool_name: str,
        result: Any,
        success: bool = True,
        tool_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Emit a tool end event.

        Args:
            tool_name: Name of the tool.
            result: Tool result.
            success: Whether tool succeeded.
            tool_id: Optional tool call ID.
            error: Optional error message.
        """
        key = tool_id or tool_name
        start_time = self._tool_start_times.pop(key, None)
        duration_ms = (time.time() - start_time) * 1000 if start_time else None
        correlation_id = get_request_correlation_id() or self._session_id

        # Emit tool complete/end event
        self._schedule_emit(
            topic="tool.end",
            data={
                "tool_name": tool_name,
                "result": result,
                "success": success,
                "tool_id": tool_id,
                "duration_ms": duration_ms,
                "category": "tool",
            },
            correlation_id=correlation_id,
            description="tool.end",
        )

        if not success and error:
            # Emit tool error event
            self._schedule_emit(
                topic=f"error.{tool_name}",
                data={
                    "tool_name": tool_name,
                    "error": error,
                    "tool_id": tool_id,
                    "category": "error",
                },
                description=f"error.{tool_name}",
            )

    # =========================================================================
    # Model Events
    # =========================================================================

    def on_model_request(
        self,
        provider: str,
        model: str,
        message_count: int,
        tool_count: int,
    ) -> None:
        """Emit a model request event.

        Args:
            provider: LLM provider name.
            model: Model identifier.
            message_count: Number of messages in request.
            tool_count: Number of tools available.
        """
        # Emit model request event
        self._schedule_emit(
            topic="model.request",
            data={
                "provider": provider,
                "model": model,
                "message_count": message_count,
                "tool_count": tool_count,
                "category": "model",
            },
            description="model.request",
        )

    def on_model_response(
        self,
        provider: str,
        model: str,
        tokens_used: Optional[int] = None,
        tool_calls: int = 0,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Emit a model response event.

        Args:
            provider: LLM provider name.
            model: Model identifier.
            tokens_used: Optional token count.
            tool_calls: Number of tool calls in response.
            latency_ms: Optional latency in milliseconds.
        """
        # Emit model response event
        self._schedule_emit(
            topic="model.response",
            data={
                "provider": provider,
                "model": model,
                "tokens_used": tokens_used,
                "tool_calls": tool_calls,
                "latency_ms": latency_ms,
                "category": "model",
            },
            description="model.response",
        )

    # =========================================================================
    # Lifecycle Events
    # =========================================================================

    def on_session_start(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Emit a session start event.

        Args:
            metadata: Optional session metadata.
        """
        # Emit session start event
        data = metadata or {}
        data["category"] = "lifecycle"

        emit_event_sync(
            self._bus,
            topic="lifecycle.session.start",
            data=data,
        )

    def on_session_end(
        self,
        tool_calls: int = 0,
        duration_seconds: Optional[float] = None,
        success: bool = True,
    ) -> None:
        """Emit a session end event.

        Args:
            tool_calls: Total tool calls in session.
            duration_seconds: Session duration.
            success: Whether session completed successfully.
        """
        # Emit session end event
        emit_event_sync(
            self._bus,
            topic="lifecycle.session.end",
            data={
                "tool_calls": tool_calls,
                "duration_seconds": duration_seconds,
                "success": success,
                "category": "lifecycle",
            },
        )

    # =========================================================================
    # Error Events
    # =========================================================================

    def on_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
    ) -> None:
        """Emit an error event.

        Args:
            error: The exception.
            context: Optional error context.
            recoverable: Whether error is recoverable.
        """
        self._bus.emit_error(error, context, recoverable)


def setup_observability(
    orchestrator: "AgentOrchestrator",
    session_id: Optional[str] = None,
) -> ObservabilityIntegration:
    """Convenience function to set up observability for an orchestrator.

    Args:
        orchestrator: Orchestrator to wire.
        session_id: Optional session ID.

    Returns:
        Configured ObservabilityIntegration instance.

    Example:
        integration = setup_observability(orchestrator)
    """
    integration = ObservabilityIntegration(
        session_id=session_id,
    )
    integration.wire_orchestrator(orchestrator)
    return integration


class ToolEventMiddleware:
    """Middleware for automatic tool event emission.

    Can be injected into the tool pipeline to automatically
    emit tool events without modifying individual tools.

    Example:
        middleware = ToolEventMiddleware(integration)
        tool_pipeline.add_middleware(middleware)
    """

    def __init__(self, integration: ObservabilityIntegration) -> None:
        """Initialize middleware.

        Args:
            integration: ObservabilityIntegration instance.
        """
        self._integration = integration

    def before_execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_id: Optional[str] = None,
    ) -> None:
        """Called before tool execution.

        Args:
            tool_name: Name of the tool.
            arguments: Tool arguments.
            tool_id: Optional tool call ID.
        """
        self._integration.on_tool_start(tool_name, arguments, tool_id)

    def after_execute(
        self,
        tool_name: str,
        result: Any,
        success: bool = True,
        tool_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Called after tool execution.

        Args:
            tool_name: Name of the tool.
            result: Tool result.
            success: Whether tool succeeded.
            tool_id: Optional tool call ID.
            error: Optional error message.
        """
        self._integration.on_tool_end(tool_name, result, success, tool_id, error)
