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

"""Factory package for creating AgentOrchestrator components.

This package decomposes the monolithic OrchestratorFactory into focused
builder modules grouped by domain:

- **tool_builders**: Tool registry, executor, pipeline, selector, cache, etc.
- **runtime_builders**: Streaming, conversation, provider, lifecycle, etc.
- **infrastructure_builders**: Observability, tracing, metrics, analytics, etc.
- **coordination_builders**: Recovery, workflow, team, safety, middleware, etc.

The OrchestratorFactory class is a thin facade that inherits from all builder
mixins and adds the DI container property and unified create_agent method.

Part of CRITICAL-001: Monolithic Orchestrator decomposition.

Usage:
    from victor.agent.factory import OrchestratorFactory, OrchestratorComponents

    factory = OrchestratorFactory(settings, provider, model)
    components = factory.create_all_components()
"""

from victor.agent.factory.tool_builders import ToolBuildersMixin
from victor.agent.factory.runtime_builders import RuntimeBuildersMixin
from victor.agent.factory.infrastructure_builders import InfrastructureBuildersMixin
from victor.agent.factory.coordination_builders import CoordinationBuildersMixin

__all__ = [
    "ToolBuildersMixin",
    "RuntimeBuildersMixin",
    "InfrastructureBuildersMixin",
    "CoordinationBuildersMixin",
]
