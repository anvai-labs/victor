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

"""Observability mixin for team coordinators.

Adds EventBus integration and progress tracking to any coordinator
that uses this mixin.

Features:
    - EventBus event emission for team lifecycle
    - Progress callback for member execution tracking
    - Execution context for observability metadata

Example:
    class MyCoordinator(ObservabilityMixin):
        def __init__(self):
            ObservabilityMixin.__init__(self)

        async def execute_team(self, config):
            self._emit_team_event("started", {"config": config.name})
            try:
                result = await self._do_execution(config)
                self._emit_team_event("completed", {"success": result.success})
                return result
            except Exception as e:
                self._emit_team_event("error", {"error": str(e)})
                raise
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional
from victor.core.constants import DEFAULT_VERTICAL

logger = logging.getLogger(__name__)


class ObservabilityMixin:
    """Mixin adding observability capabilities to team coordinators.

    This mixin implements IObservableCoordinator protocol methods and
    provides helpers for emitting events and tracking progress.

    Attributes:
        _task_type: Type of task being executed
        _complexity: Task complexity level
        _vertical_name: Domain vertical name
        _trigger: How the task was triggered
        _on_progress: Progress callback function
    """

    _task_type: str = "unknown"
    _complexity: str = "medium"
    _vertical_name: str = DEFAULT_VERTICAL
    _trigger: str = "auto"
    _on_progress: Optional[Callable[[str, str, float], None]] = None
    _observability_enabled: bool = True

    def __init__(self, *, enable_observability: bool = True) -> None:
        """Initialize observability mixin.

        Args:
            enable_observability: Whether to enable EventBus integration
        """
        self._observability_enabled = enable_observability
        self._task_type = "unknown"
        self._complexity = "medium"
        self._vertical_name = DEFAULT_VERTICAL
        self._trigger = "auto"
        self._on_progress = None

    def set_execution_context(
        self,
        task_type: str = "unknown",
        complexity: str = "medium",
        vertical: str = DEFAULT_VERTICAL,
        trigger: str = "auto",
    ) -> None:
        """Set execution context for observability.

        Args:
            task_type: Type of task (e.g., "feature", "bugfix", "refactor")
            complexity: Task complexity ("low", "medium", "high", "critical")
            vertical: Domain vertical (e.g., "coding", "devops", "research")
            trigger: How triggered ("auto", "manual", "suggestion")
        """
        self._task_type = task_type
        self._complexity = complexity
        self._vertical_name = vertical
        self._trigger = trigger

    def set_progress_callback(
        self,
        callback: Callable[[str, str, float], None],
    ) -> None:
        """Set callback for progress updates.

        The callback receives:
            - member_id: ID of the member
            - status: Current status string
            - progress: Progress percentage (0.0 to 1.0)

        Args:
            callback: Progress callback function
        """
        self._on_progress = callback

    def _emit_team_event(
        self,
        event_name: str,
        data: Dict[str, Any],
    ) -> None:
        """Emit a team event to the EventBus.

        Args:
            event_name: Name of the event (e.g., "started", "completed")
            data: Event data payload
        """
        if not self._observability_enabled:
            return

        payload = {
            "task_type": self._task_type,
            "complexity": self._complexity,
            "vertical": self._vertical_name,
            "trigger": self._trigger,
            **data,
        }

        try:
            from victor.core.events import emit_event_sync, get_observability_bus

            bus = get_observability_bus()
            emit_event_sync(
                bus,
                f"team.{event_name}",
                payload,
                source="team",
                use_background_loop=True,
            )
        except ImportError:
            logger.debug("EventBus not available, skipping event emission")
        except Exception as e:
            logger.warning(f"Failed to emit team event: {e}")

    def _report_progress(
        self,
        member_id: str,
        status: str,
        progress: float,
    ) -> None:
        """Report progress for a team member.

        Args:
            member_id: ID of the member
            status: Current status string
            progress: Progress percentage (0.0 to 1.0)
        """
        if self._on_progress:
            try:
                self._on_progress(member_id, status, progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

        # Also emit as event
        self._emit_team_event(
            "member_progress",
            {
                "member_id": member_id,
                "status": status,
                "progress": progress,
            },
        )

    def _get_observability_context(self) -> Dict[str, Any]:
        """Get current observability context.

        Returns:
            Dictionary with execution context
        """
        return {
            "task_type": self._task_type,
            "complexity": self._complexity,
            "vertical": self._vertical_name,
            "trigger": self._trigger,
        }
