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

"""Shared factory helpers for canonical coordinator/runtime surfaces."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional


def create_exploration_coordinator() -> Any:
    """Create the canonical read-only exploration runtime."""
    from victor.agent.services.exploration_runtime import ExplorationCoordinator

    return ExplorationCoordinator()


def create_exploration_state_passed_coordinator(
    *,
    settings: Any,
    project_root: Optional[Path] = None,
    max_results: int = 5,
) -> Any:
    """Create the canonical state-passed exploration wrapper."""
    from victor.agent.coordinators.exploration_state_passed import (
        ExplorationStatePassedCoordinator,
    )

    return ExplorationStatePassedCoordinator(
        project_root=project_root or resolve_project_root(settings),
        max_results=max_results,
    )


def create_system_prompt_state_passed_coordinator(
    *,
    container: Any,
    task_analyzer: Optional[Any] = None,
) -> Any:
    """Create the canonical state-passed system prompt coordinator."""
    from victor.agent.coordinators.system_prompt_state_passed import (
        SystemPromptStatePassedCoordinator,
    )

    return SystemPromptStatePassedCoordinator(
        task_analyzer=task_analyzer or resolve_task_analyzer(container),
    )


def create_safety_state_passed_coordinator() -> Any:
    """Create the canonical state-passed safety wrapper."""
    from victor.agent.coordinators.safety_state_passed import (
        SafetyStatePassedCoordinator,
    )

    return SafetyStatePassedCoordinator()


def create_coordination_advisor_runtime() -> Any:
    """Create the canonical service-owned coordination runtime surface."""
    from victor.agent.services.coordination_advisor_runtime import (
        CoordinationAdvisorRuntime,
    )

    return CoordinationAdvisorRuntime()


def create_coordination_state_passed_coordinator(
    *,
    coordination_runtime: Any,
    coordination_advisor: Any = None,
    vertical_context: Any = None,
) -> Any:
    """Create the canonical state-passed coordination wrapper."""
    from victor.agent.coordinators.coordination_state_passed import (
        CoordinationStatePassedCoordinator,
    )

    return CoordinationStatePassedCoordinator(
        coordination_runtime=coordination_runtime,
        coordination_advisor=coordination_advisor,
        vertical_context=vertical_context,
    )


def resolve_task_analyzer(container: Any) -> Any:
    """Resolve task analyzer from DI, then fall back to the singleton."""
    from victor.agent.protocols import TaskAnalyzerProtocol
    from victor.agent.task_analyzer import get_task_analyzer

    analyzer = container.get_optional(TaskAnalyzerProtocol)
    if analyzer is not None and hasattr(analyzer, "classify_task_with_context"):
        return analyzer

    analyzer = get_task_analyzer()
    if hasattr(analyzer, "classify_task_with_context"):
        return analyzer

    raise RuntimeError("TaskAnalyzer with classify_task_with_context is required")


def resolve_project_root(settings: Any) -> Optional[Path]:
    """Resolve project root from settings when available."""
    working_directory = getattr(settings, "working_directory", None)
    if not working_directory or not isinstance(working_directory, (str, os.PathLike)):
        return None
    return Path(working_directory)
