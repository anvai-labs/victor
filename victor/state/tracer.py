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

"""State transition tracer for debugging and observability.

This module provides StateTracer for recording state transitions across
all scopes. Integrates with ObservabilityBus for real-time monitoring.

SOLID Principles:
- SRP: Tracing only, no business logic
- OCP: Extensible via ObservabilityBus integration
- LSP: Substitutable with other tracers
- ISP: Focused on transition recording
- DIP: Depends on ObservabilityBus abstraction

Usage:
    from victor.state.tracer import StateTracer
    from victor.core.events import ObservabilityBus, get_observability_bus

    bus = get_observability_bus()
    tracer = StateTracer(bus)

    # Record state transition
    tracer.record_transition(
        scope="workflow",
        key="my_var",
        old_value=None,
        new_value="value",
        metadata={"source": "user_input"}
    )

    # Get history
    history = tracer.get_history(scope="workflow", limit=100)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from victor.core.events import ObservabilityBus

logger = logging.getLogger(__name__)


@dataclass
class StateTransition:
    """Record of a state transition.

    Attributes:
        scope: The StateScope where transition occurred
        key: The key that changed
        old_value: Previous value
        new_value: New value
        timestamp: Unix timestamp of transition
        metadata: Optional metadata about the transition
    """

    scope: str
    key: str
    old_value: Any
    new_value: Any
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of transition
        """
        return {
            "scope": self.scope,
            "key": self.key,
            "old_value": str(self.old_value)[:100] if self.old_value else None,
            "new_value": str(self.new_value)[:100] if self.new_value else None,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class StateTracer:
    """Traces state changes across all scopes.

    Integrates with EventBus for real-time monitoring and debugging.
    Records all state transitions for later inspection.

    SOLID: SRP (tracing only), DIP (depends on ObservabilityBus abstraction)

    Attributes:
        _event_bus: ObservabilityBus instance for emitting events
        _transitions: List of recorded transitions

    Example:
        >>> from victor.core.events import ObservabilityBus, get_observability_bus
        >>> bus = get_observability_bus()
        >>> tracer = StateTracer(bus)
        >>> tracer.record_transition("workflow", "key", None, "value")
        >>> history = tracer.get_history(scope="workflow")
    """

    def __init__(self, event_bus: Optional[ObservabilityBus] = None):
        """Initialize the state tracer.

        Args:
            event_bus: Optional ObservabilityBus instance. If None, uses DI container.
        """
        self._event_bus = event_bus or self._get_default_bus()
        self._transitions: List[StateTransition] = []

        logger.info("StateTracer initialized")

    def _get_default_bus(self) -> Optional[ObservabilityBus]:
        """Get default ObservabilityBus from DI container.

        Returns:
            ObservabilityBus instance or None if unavailable
        """
        try:
            from victor.core.events import get_observability_bus

            return get_observability_bus()
        except Exception:
            return None

    def record_transition(
        self,
        scope: str,
        key: str,
        old_value: Any,
        new_value: Any,
        **metadata: Any,
    ) -> None:
        """Record a state transition.

        Args:
            scope: The scope where transition occurred
            key: The key that changed
            old_value: Previous value
            new_value: New value
            **metadata: Optional metadata about the transition
        """
        transition = StateTransition(
            scope=scope,
            key=key,
            old_value=old_value,
            new_value=new_value,
            metadata=metadata,
        )

        self._transitions.append(transition)

        # Publish event via ObservabilityBus for real-time monitoring
        if self._event_bus:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self._event_bus.emit(
                        topic="state.transition",
                        data={
                            "scope": scope,
                            "key": key,
                            "old_value": str(old_value)[:100] if old_value else None,
                            "new_value": str(new_value)[:100] if new_value else None,
                            "timestamp": transition.timestamp,
                            **metadata,
                            "category": "state",  # Preserve for observability
                        },
                    )
                )
            except RuntimeError:
                # No event loop running
                logger.debug("No event loop, skipping state transition event emission")
            except Exception as e:
                logger.warning(f"Failed to publish state transition event: {e}")

        logger.debug(f"State transition recorded: {scope}.{key} = {str(new_value)[:50]}")

    def get_history(
        self,
        scope: Optional[str] = None,
        key: Optional[str] = None,
        limit: int = 100,
    ) -> List[StateTransition]:
        """Get transition history.

        Args:
            scope: Filter by scope (optional)
            key: Filter by key (optional)
            limit: Maximum number of transitions to return

        Returns:
            List of StateTransition objects matching filters
        """
        history = self._transitions

        if scope:
            history = [t for t in history if t.scope == scope]

        if key:
            history = [t for t in history if t.key == key]

        # Return most recent transitions (limited)
        return history[-limit:]

    def clear_history(self) -> None:
        """Clear all recorded transitions.

        Useful for long-running applications to prevent memory bloat.
        """
        count = len(self._transitions)
        self._transitions.clear()
        logger.info(f"Cleared {count} state transitions from history")

    def get_transition_count(self) -> int:
        """Get total number of recorded transitions.

        Returns:
            Number of transitions in history
        """
        return len(self._transitions)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about recorded transitions.

        Returns:
            Dictionary with transition statistics
        """
        if not self._transitions:
            return {"total": 0}

        # Count transitions by scope
        by_scope: Dict[str, int] = {}
        for transition in self._transitions:
            by_scope[transition.scope] = by_scope.get(transition.scope, 0) + 1

        # Count transitions by key
        by_key: Dict[str, int] = {}
        for transition in self._transitions:
            by_key[transition.key] = by_key.get(transition.key, 0) + 1

        return {
            "total": len(self._transitions),
            "by_scope": by_scope,
            "by_key": by_key,
            "oldest_timestamp": self._transitions[0].timestamp,
            "newest_timestamp": self._transitions[-1].timestamp,
        }


__all__ = [
    "StateTransition",
    "StateTracer",
]
