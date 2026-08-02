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

"""Registry for custom workflow node executor extensions."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict


@dataclass(frozen=True)
class WorkflowNodeExecutorRegistration:
    """Registration record for a custom workflow node executor."""

    node_type: str
    executor_factory: Any
    replace: bool = False


class WorkflowNodeExecutorRegistry:
    """Thread-safe registry for custom workflow node executor registrations."""

    def __init__(self) -> None:
        self._registrations: Dict[str, WorkflowNodeExecutorRegistration] = {}
        self._lock = RLock()

    def register(self, node_type: str, executor_factory: Any, *, replace: bool = False) -> None:
        """Register a custom workflow node executor."""
        with self._lock:
            existing = self._registrations.get(node_type)
            if existing is not None:
                if existing.executor_factory is executor_factory and existing.replace == replace:
                    return
                if not replace:
                    raise ValueError(
                        f"Workflow node executor '{node_type}' is already registered. "
                        f"Use replace=True to override."
                    )

            self._registrations[node_type] = WorkflowNodeExecutorRegistration(
                node_type=node_type,
                executor_factory=executor_factory,
                replace=replace,
            )

    def get_registrations(self) -> Dict[str, WorkflowNodeExecutorRegistration]:
        """Return a snapshot of current registrations."""
        with self._lock:
            return dict(self._registrations)

    def clear(self) -> None:
        """Clear all registered custom executor types."""
        with self._lock:
            self._registrations.clear()


_GLOBAL_REGISTRY = WorkflowNodeExecutorRegistry()


def get_workflow_node_executor_registry() -> WorkflowNodeExecutorRegistry:
    """Return the global workflow node executor registry."""
    return _GLOBAL_REGISTRY


def register_workflow_node_executor(
    node_type: str,
    executor_factory: Any,
    *,
    replace: bool = False,
) -> None:
    """Register a custom workflow node executor in the global registry."""
    get_workflow_node_executor_registry().register(
        node_type,
        executor_factory,
        replace=replace,
    )


def clear_registered_workflow_node_executors() -> None:
    """Clear global custom workflow node executor registrations."""
    get_workflow_node_executor_registry().clear()


__all__ = [
    "WorkflowNodeExecutorRegistration",
    "WorkflowNodeExecutorRegistry",
    "clear_registered_workflow_node_executors",
    "get_workflow_node_executor_registry",
    "register_workflow_node_executor",
]
