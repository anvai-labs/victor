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

"""Breakpoint data structures and management for workflow debugging.

This module provides breakpoint types, storage, and management for
debugging StateGraph workflows. Supports node, conditional, exception,
and state breakpoints with configurable hit counts and log messages.

Key Classes:
    BreakpointType: Types of breakpoints (NODE, CONDITIONAL, EXCEPTION, STATE)
    BreakpointPosition: Position relative to node execution (BEFORE, AFTER, ON_ERROR)
    WorkflowBreakpoint: A breakpoint in workflow execution
    BreakpointStorage: Storage for active breakpoints
    BreakpointManager: Manages workflow breakpoints (SRP)

Example:
    from victor.framework.debugging.breakpoints import (
        BreakpointManager,
        BreakpointPosition,
        BreakpointType,
    )

    manager = BreakpointManager(event_bus)

    # Set node breakpoint
    bp = manager.set_breakpoint(
        node_id="analyze",
        position=BreakpointPosition.BEFORE
    )

    # Set conditional breakpoint
    bp = manager.set_breakpoint(
        node_id="process",
        position=BreakpointPosition.AFTER,
        condition=lambda state: state.get("errors", 0) > 5
    )
"""

from __future__ import annotations

from victor.core.json_utils import json_dumps, json_loads
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class BreakpointType(Enum):
    """Types of breakpoints for workflow debugging.

    Attributes:
        NODE: Pause before/after specific node
        CONDITIONAL: Pause when condition is true
        EXCEPTION: Pause on any error
        STATE: Pause when state key matches value
    """

    NODE = "node"
    CONDITIONAL = "conditional"
    EXCEPTION = "exception"
    STATE = "state"


class BreakpointPosition(Enum):
    """Position relative to node execution.

    Attributes:
        BEFORE: Pause before node executes
        AFTER: Pause after node executes
        ON_ERROR: Pause if node raises exception
    """

    BEFORE = "before"
    AFTER = "after"
    ON_ERROR = "on_error"


@dataclass
class WorkflowBreakpoint:
    """A breakpoint in workflow execution.

    Attributes:
        id: Unique breakpoint identifier (UUID)
        type: Type of breakpoint (NODE, CONDITIONAL, EXCEPTION, STATE)
        position: When to pause (BEFORE, AFTER, ON_ERROR)
        node_id: Target node ID (for NODE type)
        condition: Optional condition function (for CONDITIONAL type)
        state_key: State key to watch (for STATE type)
        state_value: Expected state value (for STATE type)
        enabled: Whether breakpoint is active
        hit_count: Number of times breakpoint was hit
        ignore_count: Skip first N hits (default: 0)
        log_message: Optional log message instead of pausing
        metadata: Additional breakpoint metadata

    Example:
        # Node breakpoint
        bp = WorkflowBreakpoint(
            id=uuid.uuid4().hex,
            type=BreakpointType.NODE,
            position=BreakpointPosition.BEFORE,
            node_id="analyze_code"
        )

        # Conditional breakpoint
        bp = WorkflowBreakpoint(
            id=uuid.uuid4().hex,
            type=BreakpointType.CONDITIONAL,
            position=BreakpointPosition.AFTER,
            node_id="process_data",
            condition=lambda state: state.get("error_count", 0) > 5
        )

        # Exception breakpoint
        bp = WorkflowBreakpoint(
            id=uuid.uuid4().hex,
            type=BreakpointType.EXCEPTION,
            position=BreakpointPosition.ON_ERROR
        )
    """

    id: str
    type: BreakpointType
    position: BreakpointPosition
    node_id: Optional[str] = None
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    state_key: Optional[str] = None
    state_value: Any = None
    enabled: bool = True
    hit_count: int = 0
    ignore_count: int = 0
    log_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def should_hit(
        self, state: Dict[str, Any], node_id: str, error: Optional[Exception] = None
    ) -> bool:
        """Check if breakpoint should be hit.

        Args:
            state: Current workflow state
            node_id: Current node ID
            error: Exception if one occurred

        Returns:
            True if breakpoint should trigger pause
        """
        if not self.enabled:
            return False

        # Check ignore count
        if self.hit_count < self.ignore_count:
            self.hit_count += 1
            return False

        # Check if breakpoint condition matches
        matches = False

        # Node breakpoint
        if self.type == BreakpointType.NODE:
            matches = self.node_id == node_id

        # Conditional breakpoint
        elif self.type == BreakpointType.CONDITIONAL:
            if self.node_id and self.node_id != node_id:
                matches = False
            elif self.condition:
                try:
                    matches = self.condition(state)
                except Exception as e:
                    logger.warning(f"Breakpoint {self.id} condition evaluation failed: {e}")
                    matches = False
            else:
                matches = True

        # Exception breakpoint
        elif self.type == BreakpointType.EXCEPTION:
            matches = error is not None

        # State breakpoint
        elif self.type == BreakpointType.STATE:
            if self.state_key:
                current_value = state.get(self.state_key)
                matches = current_value == self.state_value

        # Only increment hit_count if actually matched
        if matches:
            self.hit_count += 1
            return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation of breakpoint
        """
        return {
            "id": self.id,
            "type": self.type.value,
            "position": self.position.value,
            "node_id": self.node_id,
            "state_key": self.state_key,
            "state_value": str(self.state_value) if self.state_value else None,
            "enabled": self.enabled,
            "hit_count": self.hit_count,
            "ignore_count": self.ignore_count,
            "log_message": self.log_message,
            "metadata": self.metadata,
        }


@dataclass
class BreakpointStorage:
    """Storage for active breakpoints.

    Supports both in-memory and persistent storage backends.
    Uses node indexing for efficient breakpoint lookup.

    Attributes:
        _breakpoints: Dictionary mapping breakpoint ID to breakpoint
        _node_index: Dictionary mapping node ID to list of breakpoint IDs
        _persist_enabled: Whether to persist breakpoints to disk
        _persist_path: Path to persist breakpoints

    Example:
        storage = BreakpointStorage()

        bp = WorkflowBreakpoint(
            id=uuid.uuid4().hex,
            type=BreakpointType.NODE,
            position=BreakpointPosition.BEFORE,
            node_id="analyze"
        )
        storage.add(bp)

        bps = storage.get_for_node("analyze")
    """

    _breakpoints: Dict[str, WorkflowBreakpoint] = field(default_factory=dict)
    _node_index: Dict[str, List[str]] = field(default_factory=dict)
    _persist_enabled: bool = False
    _persist_path: Optional[str] = None

    def add(self, breakpoint: WorkflowBreakpoint) -> None:
        """Add a breakpoint.

        Args:
            breakpoint: WorkflowBreakpoint to add
        """
        self._breakpoints[breakpoint.id] = breakpoint
        if breakpoint.node_id:
            if breakpoint.node_id not in self._node_index:
                self._node_index[breakpoint.node_id] = []
            self._node_index[breakpoint.node_id].append(breakpoint.id)

    def remove(self, breakpoint_id: str) -> Optional[WorkflowBreakpoint]:
        """Remove a breakpoint by ID.

        Args:
            breakpoint_id: Breakpoint ID to remove

        Returns:
            Removed breakpoint or None if not found
        """
        bp = self._breakpoints.pop(breakpoint_id, None)
        if bp and bp.node_id:
            bp_ids = self._node_index.setdefault(bp.node_id, [])
            if breakpoint_id in bp_ids:
                bp_ids.remove(breakpoint_id)
        return bp

    def get(self, breakpoint_id: str) -> Optional[WorkflowBreakpoint]:
        """Get breakpoint by ID.

        Args:
            breakpoint_id: Breakpoint ID to get

        Returns:
            WorkflowBreakpoint or None if not found
        """
        return self._breakpoints.get(breakpoint_id)

    def list_all(self) -> List[WorkflowBreakpoint]:
        """List all breakpoints.

        Returns:
            List of all breakpoints
        """
        return list(self._breakpoints.values())

    def get_for_node(self, node_id: str) -> List[WorkflowBreakpoint]:
        """Get all breakpoints for a specific node.

        Args:
            node_id: Node ID to get breakpoints for

        Returns:
            List of breakpoints for the node
        """
        bp_ids = self._node_index.get(node_id, [])
        return [self._breakpoints[bp_id] for bp_id in bp_ids if bp_id in self._breakpoints]

    def clear(self) -> None:
        """Clear all breakpoints."""
        self._breakpoints.clear()
        self._node_index.clear()

    async def persist(self) -> None:
        """Persist breakpoints to disk if enabled.

        Note:
            Condition functions are not serialized (lost on persistence).
        """
        if not self._persist_enabled or not self._persist_path:
            return

        import json
        from pathlib import Path

        data = {
            "breakpoints": [
                {**bp.to_dict(), "condition": None}  # Can't serialize functions
                for bp in self._breakpoints.values()
            ]
        }

        Path(self._persist_path).write_text(json_dumps(data, indent=2))

    async def load(self) -> None:
        """Load persisted breakpoints from disk.

        Note:
            Condition functions are lost on persistence.
        """
        if not self._persist_enabled or not self._persist_path:
            return

        import json
        from pathlib import Path

        path = Path(self._persist_path)
        if not path.exists():
            return

        data = json_loads(path.read_text())
        # Reconstruct breakpoints (conditions are lost on persistence)
        for bp_data in data.get("breakpoints", []):
            bp = WorkflowBreakpoint(
                id=bp_data["id"],
                type=BreakpointType(bp_data["type"]),
                position=BreakpointPosition(bp_data["position"]),
                node_id=bp_data.get("node_id"),
                enabled=bp_data.get("enabled", True),
                ignore_count=bp_data.get("ignore_count", 0),
                log_message=bp_data.get("log_message"),
                metadata=bp_data.get("metadata", {}),
            )
            self.add(bp)


class BreakpointManager:
    """Manages workflow breakpoints (SRP: Single Responsibility).

    Provides CRUD operations for breakpoints and evaluation logic
    for checking if breakpoints should be hit during execution.

    Attributes:
        storage: BreakpointStorage instance
        _event_bus: EventBus for emitting breakpoint events

    Example:
        from victor.core.events import ObservabilityBus

        event_bus = ObservabilityBus()
        manager = BreakpointManager(event_bus)

        # Add node breakpoint
        bp = manager.set_breakpoint(
            node_id="analyze",
            position=BreakpointPosition.BEFORE
        )

        # Add conditional breakpoint
        bp = manager.set_breakpoint(
            node_id="process",
            position=BreakpointPosition.AFTER,
            condition=lambda state: state.get("errors", 0) > 0
        )

        # Clear breakpoint
        manager.clear_breakpoint(bp.id)
    """

    def __init__(self, event_bus: Any) -> None:
        """Initialize breakpoint manager.

        Args:
            event_bus: EventBus instance for emitting events
        """
        self.storage = BreakpointStorage()
        self._event_bus = event_bus

    def set_breakpoint(
        self,
        node_id: Optional[str] = None,
        position: BreakpointPosition = BreakpointPosition.BEFORE,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        state_key: Optional[str] = None,
        state_value: Any = None,
        bp_type: BreakpointType = BreakpointType.NODE,
        ignore_count: int = 0,
        log_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowBreakpoint:
        """Set a new breakpoint.

        Args:
            node_id: Target node ID (required for NODE type)
            position: When to pause (BEFORE, AFTER, ON_ERROR)
            condition: Optional condition function
            state_key: State key to watch (for STATE type)
            state_value: Expected state value
            bp_type: Type of breakpoint
            ignore_count: Skip first N hits
            log_message: Optional log message instead of pausing
            metadata: Additional metadata

        Returns:
            Created WorkflowBreakpoint

        Raises:
            ValueError: If parameters are invalid
        """
        # Auto-detect breakpoint type based on parameters
        actual_type = bp_type
        if condition is not None and bp_type == BreakpointType.NODE:
            actual_type = BreakpointType.CONDITIONAL
        elif state_key is not None and bp_type == BreakpointType.NODE:
            actual_type = BreakpointType.STATE

        if actual_type == BreakpointType.NODE and not node_id:
            raise ValueError("node_id required for NODE breakpoints")

        bp = WorkflowBreakpoint(
            id=uuid.uuid4().hex,
            type=actual_type,
            position=position,
            node_id=node_id,
            condition=condition,
            state_key=state_key,
            state_value=state_value,
            ignore_count=ignore_count,
            log_message=log_message,
            metadata=metadata or {},
        )

        self.storage.add(bp)

        # Emit event
        self._emit_breakpoint_set(bp)

        return bp

    def clear_breakpoint(self, breakpoint_id: str) -> bool:
        """Clear a breakpoint by ID.

        Args:
            breakpoint_id: Breakpoint ID to clear

        Returns:
            True if breakpoint was found and cleared
        """
        bp = self.storage.remove(breakpoint_id)
        if bp:
            self._emit_breakpoint_cleared(bp)
            return True
        return False

    def list_breakpoints(
        self, node_id: Optional[str] = None, enabled_only: bool = False
    ) -> List[WorkflowBreakpoint]:
        """List breakpoints.

        Args:
            node_id: Optional filter by node ID
            enabled_only: Only return enabled breakpoints

        Returns:
            List of breakpoints matching filters
        """
        if node_id:
            bps = self.storage.get_for_node(node_id)
        else:
            bps = self.storage.list_all()

        if enabled_only:
            bps = [bp for bp in bps if bp.enabled]

        return bps

    def enable_breakpoint(self, breakpoint_id: str) -> bool:
        """Enable a breakpoint.

        Args:
            breakpoint_id: Breakpoint ID to enable

        Returns:
            True if breakpoint was found and enabled
        """
        bp = self.storage.get(breakpoint_id)
        if bp:
            bp.enabled = True
            return True
        return False

    def disable_breakpoint(self, breakpoint_id: str) -> bool:
        """Disable a breakpoint.

        Args:
            breakpoint_id: Breakpoint ID to disable

        Returns:
            True if breakpoint was found and disabled
        """
        bp = self.storage.get(breakpoint_id)
        if bp:
            bp.enabled = False
            return True
        return False

    def evaluate_breakpoints(
        self,
        state: Dict[str, Any],
        node_id: str,
        position: BreakpointPosition,
        error: Optional[Exception] = None,
    ) -> List[WorkflowBreakpoint]:
        """Evaluate which breakpoints should be hit.

        Called by DebugHook during workflow execution to check if
        any breakpoints should trigger a pause.

        Args:
            state: Current workflow state
            node_id: Current node ID
            position: Current position (BEFORE/AFTER/ON_ERROR)
            error: Exception if one occurred

        Returns:
            List of breakpoints that should be hit
        """
        # Get node-specific breakpoints
        node_bps = self.storage.get_for_node(node_id)

        # Get exception breakpoints
        exception_bps = [
            bp for bp in self.storage.list_all() if bp.type == BreakpointType.EXCEPTION
        ]

        # Combine and filter
        all_bps = node_bps + exception_bps

        hit_bps = []
        for bp in all_bps:
            # Check position match
            if bp.position != position:
                continue

            # Check if breakpoint should hit (hit_count is incremented inside should_hit)
            if bp.should_hit(state, node_id, error):
                hit_bps.append(bp)

                # Emit event
                self._emit_breakpoint_hit(bp, state, node_id)

                # Log message if set
                if bp.log_message:
                    logger.info(f"Breakpoint log: {bp.log_message}")

        return hit_bps

    def _emit_breakpoint_set(self, bp: WorkflowBreakpoint) -> None:
        """Emit breakpoint_set event.

        Args:
            bp: Breakpoint that was set
        """
        try:
            import asyncio

            # Emit asynchronously if possible, otherwise sync
            if asyncio.iscoroutinefunction(self._event_bus.emit):
                # Create task but don't await
                asyncio.create_task(
                    self._event_bus.emit(
                        topic="debug.breakpoint.set",
                        data={
                            "breakpoint_id": bp.id,
                            "type": bp.type.value,
                            "node_id": bp.node_id,
                        },
                    )
                )
            else:
                # Sync emit
                self._event_bus.emit(
                    topic="debug.breakpoint.set",
                    data={
                        "breakpoint_id": bp.id,
                        "type": bp.type.value,
                        "node_id": bp.node_id,
                    },
                )
        except Exception as e:
            # Event emission failures shouldn't break debugging
            logger.debug(f"Failed to emit breakpoint_set event: {e}")

    def _emit_breakpoint_cleared(self, bp: WorkflowBreakpoint) -> None:
        """Emit breakpoint_cleared event.

        Args:
            bp: Breakpoint that was cleared
        """
        try:
            import asyncio

            if asyncio.iscoroutinefunction(self._event_bus.emit):
                asyncio.create_task(
                    self._event_bus.emit(
                        topic="debug.breakpoint.cleared",
                        data={"breakpoint_id": bp.id, "type": bp.type.value},
                    )
                )
            else:
                self._event_bus.emit(
                    topic="debug.breakpoint.cleared",
                    data={"breakpoint_id": bp.id, "type": bp.type.value},
                )
        except Exception as e:
            logger.debug(f"Failed to emit breakpoint_cleared event: {e}")

    def _emit_breakpoint_hit(
        self, bp: WorkflowBreakpoint, state: Dict[str, Any], node_id: str
    ) -> None:
        """Emit breakpoint_hit event.

        Args:
            bp: Breakpoint that was hit
            state: Current workflow state
            node_id: Current node ID
        """
        try:
            import asyncio

            if asyncio.iscoroutinefunction(self._event_bus.emit):
                asyncio.create_task(
                    self._event_bus.emit(
                        topic="debug.breakpoint.hit",
                        data={
                            "breakpoint_id": bp.id,
                            "node_id": node_id,
                            "hit_count": bp.hit_count,
                            "state_keys": list(state.keys()),
                        },
                    )
                )
            else:
                self._event_bus.emit(
                    topic="debug.breakpoint.hit",
                    data={
                        "breakpoint_id": bp.id,
                        "node_id": node_id,
                        "hit_count": bp.hit_count,
                        "state_keys": list(state.keys()),
                    },
                )
        except Exception as e:
            logger.debug(f"Failed to emit breakpoint_hit event: {e}")
