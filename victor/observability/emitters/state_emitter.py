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

"""State event emitter for tracking state transitions.

SOLID Principles:
- SRP: Focused solely on state transition events
- OCP: Extensible via inheritance (can add custom state tracking)
- LSP: Substitutable with IStateEventEmitter
- ISP: Implements focused interface
- DIP: Depends on ObservabilityBus abstraction, not concrete implementation

Migration Notes:
- Migrated from legacy EventBus to canonical core/events system
- Uses topic-based routing ("state.transition") instead of category-based
- Fully async API with sync wrappers for gradual migration
"""

from __future__ import annotations

from typing import Any

from victor.observability.emitters.base import BaseEventEmitter, IStateEventEmitter


class StateEventEmitter(BaseEventEmitter, IStateEventEmitter):
    """Emits state transition events to ObservabilityBus.

    Thread-safe, performant state transition tracking.
    Handles errors gracefully to avoid impacting execution.

    Example:
        >>> emitter = StateEventEmitter()
        >>> emitter.state_transition("INITIAL", "PLANNING", 0.95, agent_id="agent-1")

    Migration:
        This emitter now uses the canonical core/events system instead of
        the legacy observability/event_bus.py. Events use topic-based routing
        ("state.transition") instead of category-based routing.
    """

    _CATEGORY = "state"

    async def state_transition_async(
        self,
        old_stage: str,
        new_stage: str,
        confidence: float,
        **metadata: Any,
    ) -> bool:
        """Emit state transition event asynchronously.

        Args:
            old_stage: Previous state/stage
            new_stage: New state/stage
            confidence: Transition confidence (0.0 to 1.0)
            **metadata: Additional metadata

        Returns:
            True if emission succeeded
        """
        return await self.emit_async(
            topic="state.transition",
            data={
                "old_stage": old_stage,
                "new_stage": new_stage,
                "confidence": confidence,
                **metadata,
            },
        )

    def state_transition(
        self,
        old_stage: str,
        new_stage: str,
        confidence: float,
        **metadata: Any,
    ) -> None:
        """Emit state transition event (sync wrapper).

        Args:
            old_stage: Previous state/stage
            new_stage: New state/stage
            confidence: Transition confidence (0.0 to 1.0)
            **metadata: Additional metadata
        """
        self.emit(
            topic="state.transition",
            data={
                "old_stage": old_stage,
                "new_stage": new_stage,
                "confidence": confidence,
                **metadata,
            },
        )
