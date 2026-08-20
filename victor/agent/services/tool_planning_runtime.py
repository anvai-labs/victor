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

"""Service-owned runtime for tool planning.

This module hosts the canonical ToolPlanner implementation on the
service-first side of the architecture.

The legacy ``victor.agent.tool_planner`` module now re-exports this
implementation for compatibility.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.core.events import ObservabilityBus

if TYPE_CHECKING:
    from victor.agent.action_authorizer import ActionIntent
    from victor.agent.tool_registrar import ToolRegistrar
    from victor.config.settings import Settings
    from victor.tools.base import ToolDefinition

logger = logging.getLogger(__name__)


class ToolPlanner:
    """Coordinates tool planning, goal inference, and intent-based filtering.

    This component provides a semantic interface for tool planning operations,
    consolidating methods that were previously in AgentOrchestrator.
    """

    def __init__(
        self,
        tool_registrar: "ToolRegistrar",
        settings: "Settings",
        event_bus: Optional[ObservabilityBus] = None,
    ):
        """Initialize ToolPlanner."""
        self.tool_registrar = tool_registrar
        self.settings = settings
        self._event_bus = event_bus or self._get_default_bus()

    def _get_default_bus(self) -> Optional[ObservabilityBus]:
        """Get default ObservabilityBus from DI container."""
        try:
            from victor.core.events import get_observability_bus

            return get_observability_bus()
        except Exception:
            return None

    def plan_tools(
        self, goals: List[str], available_inputs: Optional[List[str]] = None
    ) -> List["ToolDefinition"]:
        """Plan a sequence of tools to satisfy goals using the dependency graph."""
        planned_tools = self.tool_registrar.plan_tools(goals, available_inputs)

        if self._event_bus:
            self._safe_emit(
                "tool.planned",
                {
                    "goals": goals,
                    "planned_count": len(planned_tools),
                    "tool_names": ([t.name for t in planned_tools] if planned_tools else []),
                    "category": "tool",
                },
            )

        return planned_tools

    def infer_goals_from_message(self, user_message: str) -> List[str]:
        """Infer planning goals from the user request."""
        return self.tool_registrar.infer_goals_from_message(user_message)

    def filter_tools_by_intent(
        self,
        tools: List[Any],
        current_intent: Optional["ActionIntent"] = None,
        user_message: Optional[str] = None,
    ) -> List[Any]:
        """Filter tools based on detected user intent."""
        if current_intent is None:
            return tools

        from victor.tools.core_tool_aliases import canonicalize_core_tool_name
        from victor.agent.action_authorizer import (
            get_intent_blocked_tools,
            is_tool_blocked_for_intent,
        )

        blocked_tools = get_intent_blocked_tools(current_intent)
        if not blocked_tools:
            return tools

        def get_tool_name(tool: Any) -> str:
            if hasattr(tool, "name"):
                return canonicalize_core_tool_name(tool.name)
            if isinstance(tool, dict):
                return canonicalize_core_tool_name(tool.get("name", ""))
            return ""

        original_count = len(tools)
        filtered = []
        blocked_names = set()
        for tool in tools:
            tool_name = get_tool_name(tool)
            if not tool_name:
                continue
            if is_tool_blocked_for_intent(tool_name, current_intent, user_message):
                blocked_names.add(tool_name)
                continue
            filtered.append(tool)
        filtered_count = original_count - len(filtered)

        if filtered_count > 0:
            logger.info(
                f"Intent {current_intent.value}: filtered {filtered_count} "
                f"write/generation tools (blocked: {blocked_names})"
            )

            if self._event_bus:
                self._safe_emit(
                    "tool.filtered_by_intent",
                    {
                        "intent": current_intent.value,
                        "original_count": original_count,
                        "filtered_count": filtered_count,
                        "blocked_tools": list(blocked_names),
                        "category": "tool",
                    },
                )

        return filtered

    def _safe_emit(self, topic: str, data: Dict[str, Any]) -> None:
        """Safely emit event without blocking or causing RuntimeWarning."""
        try:
            import asyncio

            asyncio.get_running_loop()
            asyncio.create_task(self._event_bus.emit(topic, data, source="ToolPlanner"))
        except RuntimeError:
            logger.debug(f"Skipping event emission (no event loop): {topic}")
        except Exception as e:
            logger.debug(f"Failed to emit event {topic}: {e}")


__all__ = ["ToolPlanner"]
