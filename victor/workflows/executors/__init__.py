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

"""Node executors package.

Provides specialized executor classes for each workflow node type.
Each executor handles ONE node type (SRP compliance).

Executors:
- AgentNodeExecutor: Handles agent nodes (LLM + tools)
- ComputeNodeExecutor: Handles compute nodes (direct function calls)
- TransformNodeExecutor: Handles transform nodes (state transformations)
- ParallelNodeExecutor: Handles parallel nodes (concurrent execution)
- ConditionNodeExecutor: Handles condition nodes (branching logic)
- TeamStepExecutor: Handles declarative team steps (ad-hoc multi-agent teams)
- HITLNodeExecutor: Handles HITL nodes (resume-time metadata + result recording)
- Registry helpers: Support plugin/application registration of custom node types
"""

import warnings
from typing import Any

from victor.workflows.executors.factory import NodeExecutorFactory
from victor.workflows.executors.registry import (
    WorkflowNodeExecutorRegistration,
    WorkflowNodeExecutorRegistry,
    clear_registered_workflow_node_executors,
    get_workflow_node_executor_registry,
    register_workflow_node_executor,
)
from victor.workflows.executors.team import TeamStepExecutor

_DEPRECATED_ALIAS_MAP = {
    "TeamNodeExecutor": TeamStepExecutor,
}
_TEAM_NODE_REMOVAL_VERSION = "v0.9.0"
_TEAM_NODE_REMOVAL_DATE = "2027-03-31"


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_ALIAS_MAP:
        warnings.warn(
            f"{name} is deprecated; use "
            f"{_DEPRECATED_ALIAS_MAP[name].__name__} instead. "
            f"This compatibility alias remains supported through "
            f"{_TEAM_NODE_REMOVAL_VERSION} ({_TEAM_NODE_REMOVAL_DATE}) and "
            "will be removed after that milestone. "
            "See docs/architecture/migration.md for migration guidance.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_ALIAS_MAP[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_DEPRECATED_ALIAS_MAP))


__all__ = [
    "NodeExecutorFactory",
    "TeamStepExecutor",
    "TeamNodeExecutor",
    "WorkflowNodeExecutorRegistration",
    "WorkflowNodeExecutorRegistry",
    "clear_registered_workflow_node_executors",
    "get_workflow_node_executor_registry",
    "register_workflow_node_executor",
]
