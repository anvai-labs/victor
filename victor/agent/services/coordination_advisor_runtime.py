# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Service-owned runtime adapter for framework coordination recommendations.

This runtime keeps agent/service consumers on a narrow, stable boundary while
delegating all recommendation logic to ``victor.framework.coordination_runtime``.
It does not own routing policy itself.
"""

from __future__ import annotations

from typing import Any, Optional


class CoordinationAdvisorRuntime:
    """Thin service-owned adapter over the shared framework coordination helpers."""

    def suggest_for_task(
        self,
        *,
        task_type: str,
        complexity: str,
        mode: str = "build",
        runtime_subject: Optional[Any] = None,
        coordination_advisor: Optional[Any] = None,
        vertical_context: Optional[Any] = None,
    ) -> Any:
        """Build a coordination suggestion from the shared framework surfaces."""
        from victor.framework.coordination_runtime import (
            VerticalCoordinationAdvisor,
            build_runtime_coordination_suggestion,
        )

        if runtime_subject is not None:
            return build_runtime_coordination_suggestion(
                runtime_subject=runtime_subject,
                task_type=task_type,
                complexity=complexity,
                mode=mode,
            )

        advisor = coordination_advisor
        if advisor is None:
            advisor = VerticalCoordinationAdvisor(vertical_context=vertical_context)

        return advisor.suggest_for_task(
            task_type=task_type,
            complexity=complexity,
            mode=mode,
        )

    def serialize_suggestion(
        self,
        suggestion: Any,
        *,
        vertical: Optional[str] = None,
        available_teams: Optional[tuple[str, ...]] = None,
        available_workflows: Optional[tuple[str, ...]] = None,
        default_workflow: Optional[str] = None,
    ) -> dict[str, Any]:
        """Serialize a coordination suggestion for transport/UI surfaces."""
        from victor.framework.coordination_runtime import (
            serialize_coordination_suggestion,
        )

        return serialize_coordination_suggestion(
            suggestion,
            vertical=vertical,
            available_teams=available_teams,
            available_workflows=available_workflows,
            default_workflow=default_workflow,
        )
