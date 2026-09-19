# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Service-owned runtime helper for TaskCoordinator orchestration bridges."""

from __future__ import annotations

from typing import Any

import logging

logger = logging.getLogger(__name__)


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

    def classify_task_keywords(self, user_message: str) -> dict[str, Any]:
        """Classify task shape through the configured prompt pipeline or analyzer."""
        runtime = self._runtime
        pipeline = getattr(runtime, "_prompt_pipeline", None)
        if pipeline is not None:
            return pipeline.classify_task_keywords(user_message)

        task_analyzer = getattr(runtime, "_task_analyzer", None)
        if task_analyzer is not None:
            try:
                method = getattr(
                    task_analyzer,
                    "classify_task_keywords",
                    getattr(task_analyzer, "classify_keywords", None),
                )
                if method is not None:
                    return method(user_message)
            except Exception as exc:
                logger.debug("Task keyword classification fallback failed: %s", exc)

        return {"task_type": "default", "confidence": 0.0}

    def current_intent(self) -> Any:
        """Return the coordinator-owned intent selected for the active turn."""
        return self._runtime.task_coordinator.current_intent

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
