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

"""Sub-agent module for hierarchical task delegation.

This module provides infrastructure for spawning specialized sub-agents
that can work on focused tasks with constrained scopes.

Example:
    from victor.agent.subagents import (
        SubAgent,
        SubAgentRole,
        SubAgentConfig,
        SubAgentOrchestrator,
    )

    # Create configuration for a researcher
    config = SubAgentConfig(
        role=SubAgentRole.RESEARCHER,
        task="Find all authentication patterns in the codebase",
        allowed_tools=["read", "search", "code_search"],
        tool_budget=15,
        context_limit=50000,
    )

    # Create and execute sub-agent
    subagent = SubAgent(config, parent_orchestrator)
    result = await subagent.execute()

    # Or use the orchestrator for multiple sub-agents
    orchestrator = SubAgentOrchestrator(parent_orchestrator)
    result = await orchestrator.spawn(
        SubAgentRole.PLANNER,
        "Create implementation plan for user auth",
        tool_budget=10,
    )
"""

from victor.agent.subagents.base import (
    SubAgent,
    SubAgentConfig,
    SubAgentResult,
    SubAgentRole,
)
from victor.agent.subagents.orchestrator import SubAgentOrchestrator
from victor.agent.subagents.prompts import get_role_prompt
from victor.agent.subagents.protocols import (
    SubAgentContext,
    SubAgentContextAdapter,
)

__all__ = [
    # Core classes
    "SubAgent",
    "SubAgentConfig",
    "SubAgentResult",
    "SubAgentRole",
    # Orchestrator
    "SubAgentOrchestrator",
    # Protocols (ISP compliant)
    "SubAgentContext",
    "SubAgentContextAdapter",
    # Utilities
    "get_role_prompt",
]
