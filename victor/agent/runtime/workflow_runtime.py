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

"""Workflow runtime boundaries for AgentOrchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from victor.agent.runtime.provider_runtime import LazyRuntimeProxy


@dataclass(frozen=True)
class WorkflowRuntimeComponents:
    """Workflow runtime handles exposed to the orchestrator facade."""

    workflow_registry: LazyRuntimeProxy[Any]


def create_workflow_runtime_components(
    *,
    factory: Any,
) -> WorkflowRuntimeComponents:
    """Create workflow runtime components with lazy registry materialization."""

    def _build_workflow_registry() -> Any:
        from victor.workflows.discovery import register_builtin_workflows

        registry = factory.create_workflow_registry()
        register_builtin_workflows(registry)
        return registry

    return WorkflowRuntimeComponents(
        workflow_registry=LazyRuntimeProxy(
            factory=_build_workflow_registry,
            name="workflow_registry",
        )
    )
