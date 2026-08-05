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

"""Centralized workflow trigger registry.

This module provides a single registry for workflow triggers, eliminating
the duplication of regex patterns across vertical workflow providers.

Design Goals:
- Single source of truth for workflow triggers
- Declarative trigger registration
- Support for regex patterns and task type matching
- Cross-vertical workflow discovery

Usage:
    from victor.workflows.trigger_registry import (
        WorkflowTriggerRegistry,
        WorkflowTrigger,
        get_trigger_registry,
    )

    # Get the global registry
    registry = get_trigger_registry()

    # Register triggers
    registry.register(WorkflowTrigger(
        pattern=r"implement\\s+.+feature",
        workflow_name="feature_implementation",
        vertical="coding",
        task_types=["feature", "implement"],
    ))

    # Find workflow for a query
    workflow = registry.find_workflow("implement new feature")
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Singleton instance
_registry_instance: Optional["WorkflowTriggerRegistry"] = None
_registry_lock = threading.Lock()


@dataclass
class WorkflowTrigger:
    """A trigger that maps patterns/task types to workflows.

    Attributes:
        pattern: Regex pattern to match against queries
        workflow_name: Name of the workflow to trigger
        vertical: Vertical that owns this workflow
        task_types: Task types that should trigger this workflow
        priority: Higher priority triggers are checked first (default 0)
        description: Human-readable description
    """

    pattern: str
    workflow_name: str
    vertical: str
    task_types: List[str] = field(default_factory=list)
    priority: int = 0
    description: str = ""

    _compiled_pattern: Optional[re.Pattern] = field(default=None, repr=False)

    def __post_init__(self):
        """Compile the regex pattern."""
        if self.pattern:
            try:
                self._compiled_pattern = re.compile(self.pattern, re.IGNORECASE)
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{self.pattern}': {e}")
                self._compiled_pattern = None

    def matches_query(self, query: str) -> bool:
        """Check if this trigger matches a query.

        Args:
            query: User query to match against

        Returns:
            True if the query matches the pattern
        """
        if self._compiled_pattern:
            return bool(self._compiled_pattern.search(query))
        return False

    def matches_task_type(self, task_type: str) -> bool:
        """Check if this trigger matches a task type.

        Args:
            task_type: Task type to match

        Returns:
            True if the task type is in the trigger's task types
        """
        return task_type.lower() in [t.lower() for t in self.task_types]


class WorkflowTriggerRegistry:
    """Central registry for workflow triggers.

    Provides pattern-based and task-type-based workflow discovery
    across all verticals.

    Thread-safe for concurrent access.

    Example:
        registry = WorkflowTriggerRegistry()

        # Register triggers from a vertical
        registry.register_from_vertical("coding", [
            WorkflowTrigger(
                pattern=r"implement\\s+.+feature",
                workflow_name="feature_implementation",
                vertical="coding",
            ),
        ])

        # Find workflow for a query
        result = registry.find_workflow("implement new feature")
        if result:
            workflow_name, vertical = result
    """

    def __init__(self):
        """Initialize the registry."""
        self._triggers: List[WorkflowTrigger] = []
        self._by_vertical: Dict[str, List[WorkflowTrigger]] = {}
        self._by_task_type: Dict[str, List[WorkflowTrigger]] = {}
        self._lock = threading.RLock()

    def register(self, trigger: WorkflowTrigger) -> None:
        """Register a workflow trigger.

        Args:
            trigger: The trigger to register
        """
        with self._lock:
            self._triggers.append(trigger)

            # Index by vertical
            if trigger.vertical not in self._by_vertical:
                self._by_vertical[trigger.vertical] = []
            self._by_vertical[trigger.vertical].append(trigger)

            # Index by task types
            for task_type in trigger.task_types:
                task_lower = task_type.lower()
                if task_lower not in self._by_task_type:
                    self._by_task_type[task_lower] = []
                self._by_task_type[task_lower].append(trigger)

            logger.debug(
                f"Registered trigger: {trigger.workflow_name} " f"(vertical={trigger.vertical})"
            )

    def register_from_vertical(
        self,
        vertical: str,
        triggers: List[Tuple[str, str]],
    ) -> int:
        """Register triggers from a vertical's get_auto_workflows() output.

        This provides backward compatibility with the existing
        get_auto_workflows() pattern.

        Args:
            vertical: Vertical name
            triggers: List of (pattern, workflow_name) tuples

        Returns:
            Number of triggers registered
        """
        count = 0
        for pattern, workflow_name in triggers:
            self.register(
                WorkflowTrigger(
                    pattern=pattern,
                    workflow_name=workflow_name,
                    vertical=vertical,
                )
            )
            count += 1

        logger.debug(f"Registered {count} triggers from vertical '{vertical}'")
        return count

    def find_workflow(
        self,
        query: str,
        preferred_vertical: Optional[str] = None,
    ) -> Optional[Tuple[str, str]]:
        """Find a workflow matching a query.

        Searches all registered triggers for a pattern match.

        Args:
            query: User query to match
            preferred_vertical: If set, prefer triggers from this vertical

        Returns:
            Tuple of (workflow_name, vertical) or None if no match
        """
        with self._lock:
            # Sort by priority (higher first), then by preferred vertical
            sorted_triggers = sorted(
                self._triggers,
                key=lambda t: (
                    -t.priority,
                    0 if t.vertical == preferred_vertical else 1,
                ),
            )

            for trigger in sorted_triggers:
                if trigger.matches_query(query):
                    return (trigger.workflow_name, trigger.vertical)

            return None

    def find_by_task_type(
        self,
        task_type: str,
        preferred_vertical: Optional[str] = None,
    ) -> Optional[Tuple[str, str]]:
        """Find a workflow by task type.

        Args:
            task_type: Task type to match
            preferred_vertical: If set, prefer triggers from this vertical

        Returns:
            Tuple of (workflow_name, vertical) or None if no match
        """
        with self._lock:
            triggers = self._by_task_type.get(task_type.lower(), [])

            if not triggers:
                return None

            # Sort by priority, prefer vertical
            sorted_triggers = sorted(
                triggers,
                key=lambda t: (
                    -t.priority,
                    0 if t.vertical == preferred_vertical else 1,
                ),
            )

            if sorted_triggers:
                trigger = sorted_triggers[0]
                return (trigger.workflow_name, trigger.vertical)

            return None

    def get_triggers_for_vertical(self, vertical: str) -> List[WorkflowTrigger]:
        """Get all triggers for a vertical.

        Args:
            vertical: Vertical name

        Returns:
            List of triggers for the vertical
        """
        with self._lock:
            return list(self._by_vertical.get(vertical, []))

    def list_verticals(self) -> List[str]:
        """List all verticals with registered triggers.

        Returns:
            List of vertical names
        """
        with self._lock:
            return list(self._by_vertical.keys())

    def list_task_types(self) -> List[str]:
        """List all registered task types.

        Returns:
            List of task type names
        """
        with self._lock:
            return list(self._by_task_type.keys())

    def clear(self) -> None:
        """Clear all registered triggers."""
        with self._lock:
            self._triggers.clear()
            self._by_vertical.clear()
            self._by_task_type.clear()
            logger.debug("Cleared workflow trigger registry")

    def to_auto_workflows(self, vertical: str) -> List[Tuple[str, str]]:
        """Get triggers in get_auto_workflows() format.

        This provides backward compatibility for verticals that
        still use the old pattern.

        Args:
            vertical: Vertical name

        Returns:
            List of (pattern, workflow_name) tuples
        """
        with self._lock:
            triggers = self._by_vertical.get(vertical, [])
            return [(t.pattern, t.workflow_name) for t in triggers if t.pattern]


def get_trigger_registry() -> WorkflowTriggerRegistry:
    """Get the global workflow trigger registry.

    Thread-safe singleton access.

    Returns:
        Global WorkflowTriggerRegistry instance
    """
    global _registry_instance

    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = WorkflowTriggerRegistry()

    return _registry_instance


def register_trigger(trigger: WorkflowTrigger) -> None:
    """Register a trigger in the global registry.

    Convenience function for quick registration.

    Args:
        trigger: Trigger to register
    """
    get_trigger_registry().register(trigger)


def find_workflow_for_query(
    query: str,
    preferred_vertical: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """Find a workflow for a query.

    Convenience function that delegates to the registry.

    Args:
        query: User query
        preferred_vertical: Preferred vertical

    Returns:
        Tuple of (workflow_name, vertical) or None
    """
    return get_trigger_registry().find_workflow(query, preferred_vertical)


__all__ = [
    "WorkflowTrigger",
    "WorkflowTriggerRegistry",
    "get_trigger_registry",
    "register_trigger",
    "find_workflow_for_query",
]
