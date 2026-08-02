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

"""Protocol usage guide for Victor framework.

This module documents when to use each protocol type for ISP compliance
(Interface Segregation Principle).

Protocol Selection Guide:
=========================

1. FrameworkOrchestratorProtocol (framework/protocols.py)
   Use: Framework code, workflows, external integrations
   Provides: Complete orchestrator interface (6 sub-protocols combined)
   Size: Large (composite protocol)
   When: You need full orchestrator access in framework code

   Example:
        from victor.framework.protocols import FrameworkOrchestratorProtocol

        def process(orch: FrameworkOrchestratorProtocol):
            tools = orch.get_available_tools()
            stage = orch.get_stage()

2. OrchestratorProtocol (core/protocols.py)
   Use: Breaking circular imports, evaluation harnesses
   Provides: Minimal interface (chat, stream_chat, reset)
   Size: Small
   When: You need basic LLM interaction without full dependencies

   Example:
        from victor.core.protocols import OrchestratorProtocol

        def run_eval(orch: OrchestratorProtocol, prompt: str):
            response = orch.chat(prompt)
            return response

3. SubAgentContext (agent/subagents/protocols.py)
   Use: SubAgent implementations, ISP-compliant minimal access
   Provides: Only what SubAgent needs (settings, provider, etc.)
   Size: Small (5 properties)
   When: Implementing a SubAgent that should not depend on full orchestrator

   Example:
        from victor.agent.subagents.protocols import SubAgentContext

        def run_subagent(ctx: SubAgentContext, task: str):
            provider = ctx.provider
            settings = ctx.settings

4. VerticalContextProtocol (agent/vertical_context.py)
   Use: Vertical integration, capability access
   Provides: Vertical configuration access
   Size: Medium
   When: Accessing vertical-specific state and configuration

   Example:
        from victor.agent.vertical_context import VerticalContext

        def apply_config(context: VerticalContext, config: dict):
            context.set_capability_config("rag_config", config)

5. VerticalContextCapabilityProtocol (framework/protocols.py)
   Use: Type-safe access to orchestrator's VerticalContext
   Provides: get_vertical_context() method
   Size: Small
   When: You need to access VerticalContext from orchestrator

   Example:
        from victor.framework.protocols import VerticalContextCapabilityProtocol

        def get_configs(orch: VerticalContextCapabilityProtocol):
            ctx = orch.get_vertical_context()
            return ctx.capability_configs

Protocol Hierarchy:
===================

    OrchestratorProtocol (framework)
    ├── ConversationStateProtocol
    ├── ProviderProtocol
    ├── ToolsProtocol
    ├── SystemPromptProtocol
    ├── MessagesProtocol
    └── StreamingProtocol

    OrchestratorProtocol (core)
    └── Minimal: chat(), stream_chat(), reset()

    SubAgentContext
    └── Minimal: settings, provider, tools, etc.

    VerticalContextProtocol
    └── Vertical state: stages, middleware, configs

SOLID Compliance:
=================

- **SRP**: Each protocol has a single, well-defined responsibility
- **OCP**: New protocols can be added without modifying existing ones
- **LSP**: Sub-protocols are substitutable for the full protocol where appropriate
- **ISP**: Clients depend only on the protocols they need (minimal dependencies)
- **DIP**: All code depends on protocol abstractions, not concrete implementations

Migration Guide:
================

If you're currently using the full OrchestratorProtocol but only need
a subset of functionality, consider switching to a more specific protocol:

Before (violates ISP):
    def process(orch: OrchestratorProtocol):
        # Only uses get_stage()
        stage = orch.get_stage()

After (ISP-compliant):
    def process(orch: ConversationStateProtocol):
        # Only uses what's needed
        stage = orch.get_stage()

This makes your code more focused, testable, and less coupled to
orchestrator implementation details.
"""

from __future__ import annotations

from typing import Any, Dict

# Import all protocols for easy access
from victor.core.shared_types import SubAgentContext, VerticalContextProtocol
from victor.agent.vertical_context import VerticalContext
from victor.core.protocols import OrchestratorProtocol as CoreOrchestratorProtocol
from victor.framework.protocols import (
    CapabilityRegistryProtocol,
    OrchestratorProtocol as FrameworkOrchestratorProtocol,
)

__all__ = [
    # Protocols
    "CoreOrchestratorProtocol",
    "FrameworkOrchestratorProtocol",
    "SubAgentContext",
    "VerticalContextProtocol",
    "CapabilityRegistryProtocol",
    # Convenience
    "VerticalContext",
]


def get_protocol_recommendation(use_case: str) -> Dict[str, Any]:
    """Get recommended protocol for a given use case.

    Args:
        use_case: Description of the use case

    Returns:
        Dict with protocol recommendation and rationale
    """
    use_case = use_case.lower()

    # Framework code needing full access
    if any(kw in use_case for kw in ["workflow", "integration", "framework", "full"]):
        return {
            "protocol": "FrameworkOrchestratorProtocol",
            "import_from": "victor.framework.protocols",
            "rationale": "Framework code requires complete orchestrator interface",
        }

    # Evaluation or testing
    if any(kw in use_case for kw in ["eval", "test", "harness", "benchmark"]):
        return {
            "protocol": "CoreOrchestratorProtocol",
            "import_from": "victor.core.protocols",
            "rationale": "Minimal interface sufficient for evaluation",
        }

    # SubAgent implementation
    if any(kw in use_case for kw in ["subagent", "sub-agent", "agent delegation"]):
        return {
            "protocol": "SubAgentContext",
            "import_from": "victor.agent.subagents.protocols",
            "rationale": "ISP-compliant minimal interface for SubAgents",
        }

    # Vertical integration
    if any(kw in use_case for kw in ["vertical", "config", "capability"]):
        return {
            "protocol": "VerticalContextProtocol",
            "import_from": "victor.agent.vertical_context",
            "rationale": "Vertical-specific state and configuration",
        }

    # Default to framework protocol
    return {
        "protocol": "FrameworkOrchestratorProtocol",
        "import_from": "victor.framework.protocols",
        "rationale": "Default: use full orchestrator protocol",
    }
