# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Service-owned runtime helper for TaskCoordinator orchestration bridges."""

from __future__ import annotations

from typing import Any


class TaskGuidanceRuntime:
    """Bridge orchestrator runtime state to the canonical TaskCoordinator."""

    def __init__(self, runtime_host: Any) -> None:
        self._runtime = runtime_host

    def prepare_task(self, user_message: str, unified_task_type: Any) -> tuple[Any, int]:
        """Prepare task-specific guidance and budget adjustments."""
        runtime = self._runtime
        task_coordinator = runtime.task_coordinator
        if task_coordinator._reminder_manager is None:
            task_coordinator.set_reminder_manager(runtime.reminder_manager)
        return task_coordinator.prepare_task(
            user_message,
            unified_task_type,
            runtime.conversation_controller,
        )

    def apply_intent_guard(self, user_message: str) -> None:
        """Detect intent and sync the result back to runtime state."""
        runtime = self._runtime
        task_coordinator = runtime.task_coordinator
        task_coordinator.apply_intent_guard(user_message, runtime.conversation_controller)
        runtime._current_intent = task_coordinator.current_intent
        runtime._current_user_message = user_message

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
        """Apply task guidance and sync updated coordinator state back to runtime."""
        runtime = self._runtime
        task_coordinator = runtime.task_coordinator
        task_coordinator.temperature = runtime.temperature
        task_coordinator.tool_budget = runtime.tool_budget
        task_coordinator.apply_task_guidance(
            user_message=user_message,
            unified_task_type=unified_task_type,
            is_analysis_task=is_analysis_task,
            is_action_task=is_action_task,
            needs_execution=needs_execution,
            max_exploration_iterations=max_exploration_iterations,
            conversation_controller=runtime.conversation_controller,
        )
        runtime.temperature = task_coordinator.temperature
        runtime.tool_budget = task_coordinator.tool_budget
