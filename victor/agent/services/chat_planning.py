# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# Licensed under the Apache License, Version 2.0 (the "License").

"""Planning and guidance capabilities consumed by the chat runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class TaskGuidance(Protocol):
    """Turn guidance without exposing the orchestrator facade."""

    def prepare_task(self, user_message: str, unified_task_type: Any) -> tuple[Any, int]: ...

    def apply_intent_guard(self, user_message: str) -> None: ...

    def classify_task_keywords(self, user_message: str) -> dict[str, Any]: ...

    def current_intent(self) -> Any: ...

    def apply_task_guidance(
        self,
        *,
        user_message: str,
        unified_task_type: Any,
        is_analysis_task: bool,
        is_action_task: bool,
        needs_execution: bool,
        max_exploration_iterations: int,
    ) -> None: ...


class ToolPlanner(Protocol):
    """Goal inference and tool-plan construction used by a chat turn."""

    def infer_goals_from_message(self, user_message: str) -> Any: ...

    def plan_tools(self, goals: Any, available_inputs: list[str]) -> Any: ...


class ToolSelection(Protocol):
    """Resolve the concrete tool supply for one turn."""

    async def select_tools_for_turn(
        self,
        context_msg: str,
        goals: Any,
        planned_tools: Any = None,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class ChatPlanning:
    """Narrow planning surface bound once when the chat runtime is built."""

    guidance: TaskGuidance | None = None
    planner: ToolPlanner | None = None
    selection: ToolSelection | None = None

    def _require_guidance(self) -> TaskGuidance:
        if self.guidance is None:
            raise TypeError("Chat planning requires a task-guidance service")
        return self.guidance

    def _require_planner(self) -> ToolPlanner:
        if self.planner is None:
            raise TypeError("Chat planning requires a tool planner")
        return self.planner

    def apply_intent_guard(self, user_message: str) -> None:
        self._require_guidance().apply_intent_guard(user_message)

    def prepare_task(self, user_message: str, unified_task_type: Any) -> tuple[Any, int]:
        return self._require_guidance().prepare_task(user_message, unified_task_type)

    def classify_task_keywords(self, user_message: str) -> dict[str, Any]:
        return self._require_guidance().classify_task_keywords(user_message)

    def current_intent(self) -> Any:
        return self._require_guidance().current_intent()

    def apply_task_guidance(
        self,
        *,
        user_message: str,
        unified_task_type: Any,
        is_analysis_task: bool,
        is_action_task: bool,
        needs_execution: bool,
        max_exploration_iterations: int,
    ) -> None:
        self._require_guidance().apply_task_guidance(
            user_message=user_message,
            unified_task_type=unified_task_type,
            is_analysis_task=is_analysis_task,
            is_action_task=is_action_task,
            needs_execution=needs_execution,
            max_exploration_iterations=max_exploration_iterations,
        )

    def infer_goals(self, user_message: str) -> Any:
        return self._require_planner().infer_goals_from_message(user_message)

    def plan_tools(self, goals: Any, available_inputs: list[str]) -> Any:
        return self._require_planner().plan_tools(goals, available_inputs)

    async def select_tools(
        self,
        context_msg: str,
        goals: Any,
        *,
        planned_tools: Any = None,
    ) -> Any:
        if self.selection is None:
            raise TypeError("Chat planning requires a tool-selection service")
        return await self.selection.select_tools_for_turn(
            context_msg,
            goals,
            planned_tools=planned_tools,
        )
