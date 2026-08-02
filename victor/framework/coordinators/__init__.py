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

"""Workflow coordinators for SRP-compliant WorkflowEngine.

This package splits the WorkflowEngine responsibilities into focused
coordinators, each handling a single domain:

- YAMLWorkflowCoordinator: YAML workflow loading and execution
- GraphTurnExecutor: StateGraph/CompiledGraph execution
- HITLCoordinator: Human-in-the-Loop integration
- CacheCoordinator: Workflow caching management

Architecture:
    WorkflowEngine (Facade)
    ├── YAMLWorkflowCoordinator     # execute_yaml(), stream_yaml()
    ├── GraphTurnExecutor   # execute_graph(), stream_graph()
    ├── HITLCoordinator             # execute_with_hitl()
    └── CacheCoordinator            # enable_caching(), clear_cache()

This follows the Single Responsibility Principle by ensuring each
coordinator has one reason to change.
"""

from victor.framework.coordinators.protocols import (
    IWorkflowExecutor,
    IStreamingExecutor,
    IYAMLLoader,
    IGraphExecutor,
    IHITLExecutor,
    ICacheManager,
)
from victor.framework.coordinators.yaml_coordinator import YAMLWorkflowCoordinator
from victor.framework.coordinators.graph_coordinator import GraphTurnExecutor
from victor.framework.coordinators.hitl_coordinator import HITLCoordinator
from victor.framework.coordinators.cache_coordinator import CacheCoordinator

__all__ = [
    # Protocols
    "IWorkflowExecutor",
    "IStreamingExecutor",
    "IYAMLLoader",
    "IGraphExecutor",
    "IHITLExecutor",
    "ICacheManager",
    # Coordinators
    "YAMLWorkflowCoordinator",
    "GraphTurnExecutor",
    "HITLCoordinator",
    "CacheCoordinator",
]
