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

"""Task delegation infrastructure for agent-to-agent work handoff.

This module provides the delegation protocol that allows agents to spawn
specialized sub-agents for specific tasks. This enables:

- Dynamic task decomposition at runtime
- Specialized agents for specific sub-tasks
- Hierarchical agent structures
- Fire-and-forget or await-result patterns

Key Components:
- DelegationRequest: Request for task delegation
- DelegationResponse: Response from delegation
- DelegationHandler: Processes delegation requests
- DelegateTool: Tool interface for agents to delegate

Example Usage:
    # Agent uses the delegate tool
    result = await delegate_tool.execute(
        task="Find all authentication endpoints in the codebase",
        role="researcher",
        tool_budget=15,
        await_result=True,
    )

    if result.success:
        print(result.result)  # Findings from the researcher

The delegation system integrates with Victor's tool system, appearing as
a regular tool that agents can call during execution.
"""

from victor.agent.delegation.protocol import (
    DelegationPriority,
    DelegationRequest,
    DelegationResponse,
    DelegationStatus,
)
from victor.agent.delegation.handler import DelegationHandler
from victor.agent.delegation.tool import DelegateTool

__all__ = [
    # Protocol types
    "DelegationPriority",
    "DelegationRequest",
    "DelegationResponse",
    "DelegationStatus",
    # Handler
    "DelegationHandler",
    # Tool
    "DelegateTool",
]
