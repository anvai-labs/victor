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

"""Stable compatibility API surface for framework-owned extension helpers.

New external verticals should prefer ``victor_contracts`` for definition-layer
contracts such as ``VerticalBase``, ``StageDefinition``, ``VerticalConfig``,
and ``register_vertical``. This module remains as a stable compatibility
surface for framework-owned helpers and older external packages that still
import through ``victor.framework.*``.

Importing directly from victor.agent.*, victor.core.container.*,
victor.workflows.executor.*, or victor.evaluation.* is deprecated
and may break without notice.

Usage:
    from victor.framework.extensions import SafetyCoordinator, WorkflowExecutor
    from victor_contracts import StageDefinition, VerticalBase, register_vertical
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    # Safety coordination
    "SafetyCoordinator",  # noqa: F822
    "SafetyAction",  # noqa: F822
    "SafetyCategory",  # noqa: F822
    "SafetyRule",  # noqa: F822
    # Conversation coordination
    "ConversationCoordinator",  # noqa: F822
    "ConversationStats",  # noqa: F822
    "ConversationTurn",  # noqa: F822
    "TurnType",  # noqa: F822
    # Workflow execution
    "WorkflowExecutor",  # noqa: F822
    "WorkflowContext",  # noqa: F822
    "ComputeNode",  # noqa: F822
    "NodeResult",  # noqa: F822
    "ExecutorNodeStatus",  # noqa: F822
    "register_compute_handler",  # noqa: F822
    "get_compute_handler",  # noqa: F822
    # Workflow definition
    "WorkflowBuilder",  # noqa: F822
    "WorkflowDefinition",  # noqa: F822
    "workflow",  # noqa: F822
    "AgentNode",  # noqa: F822
    "ConditionNode",  # noqa: F822
    "ParallelNode",  # noqa: F822
    # Code correction
    "CodeCorrectionMiddleware",  # noqa: F822
    "CodeCorrectionConfig",  # noqa: F822
    "CorrectionResult",  # noqa: F822
    # Code validation
    "CodeValidationResult",  # noqa: F822
    "Language",  # noqa: F822
    # Mode configuration
    "ModeConfigRegistry",  # noqa: F822
    "ModeDefinition",  # noqa: F822
    "ModeConfig",  # noqa: F822
    "RegistryBasedModeConfigProvider",  # noqa: F822
    # Service container
    "ServiceContainer",  # noqa: F822
    "ServiceLifetime",  # noqa: F822
    # Agent specs
    "AgentSpec",  # noqa: F822
    "AgentCapabilities",  # noqa: F822
    "AgentConstraints",  # noqa: F822
    "ModelPreference",  # noqa: F822
    "OutputFormat",  # noqa: F822
    "DelegationPolicy",  # noqa: F822
    # Sub-agents
    "SubAgent",  # noqa: F822
    "SubAgentConfig",  # noqa: F822
    "SubAgentResult",  # noqa: F822
    "SubAgentRole",  # noqa: F822
    "set_role_tool_provider",  # noqa: F822
    # Middleware
    "MiddlewareChain",  # noqa: F822
    "MiddlewareAbortError",  # noqa: F822
    "create_middleware_chain",  # noqa: F822
    # Vertical context
    "VerticalContext",  # noqa: F822
    "create_vertical_context",  # noqa: F822
    "VerticalContextProtocol",  # noqa: F822
    "MutableVerticalContextProtocol",  # noqa: F822
    # Handler registry
    "HandlerRegistry",  # noqa: F822
    "get_handler_registry",  # noqa: F822
    "register_global_handler",  # noqa: F822
    "register_vertical_handlers",  # noqa: F822
    # Provider access
    "ProviderRegistry",  # noqa: F822
    # Vertical registration and base types (avoids direct victor.core.* imports)
    "register_vertical",  # noqa: F822
    "VerticalBase",  # noqa: F822
    "StageDefinition",  # noqa: F822
    "VerticalConfig",  # noqa: F822
    "VerticalExtensions",  # noqa: F822
    # Tool dependency types
    "BaseToolDependencyProvider",  # noqa: F822
    "ToolDependencyConfig",  # noqa: F822
    "ToolDependency",  # noqa: F822
    "YAMLToolDependencyProvider",  # noqa: F822
    "load_tool_dependency_yaml",  # noqa: F822
    "create_vertical_tool_dependency_provider",  # noqa: F822
    "TieredToolConfig",  # noqa: F822
    "TaskTypeHint",  # noqa: F822
    "ToolDependencyLoader",  # noqa: F822
    # Safety pattern types
    "SafetyExtensionProtocol",  # noqa: F822
    "SafetyPattern",  # noqa: F822
    # Promoted protocols (from SDK, re-exported for convenience)
    "PromptContributorProtocol",  # noqa: F822
    "ModeConfigProviderProtocol",  # noqa: F822
    "ServiceProviderProtocol",  # noqa: F822
    "ToolDependencyProviderProtocol",  # noqa: F822
    "WorkflowProviderProtocol",  # noqa: F822
    "RLConfigProviderProtocol",  # noqa: F822
    "TeamSpecProviderProtocol",  # noqa: F822
    "EnrichmentStrategyProtocol",  # noqa: F822
    "MiddlewareProtocol",  # noqa: F822
]


def __getattr__(name: str):
    """Lazy imports to avoid circular dependencies and unnecessary loading."""
    _LAZY_IMPORTS = {
        # Safety
        "SafetyCoordinator": "victor_contracts.safety",
        "SafetyAction": "victor_contracts.safety",
        "SafetyCategory": "victor_contracts.safety",
        "SafetyRule": "victor_contracts.safety",
        # Conversation
        "ConversationCoordinator": "victor_contracts.conversation",
        "ConversationStats": "victor_contracts.conversation",
        "ConversationTurn": "victor_contracts.conversation",
        "TurnType": "victor_contracts.conversation",
        # Workflow execution
        "WorkflowExecutor": "victor.workflows.executor",
        "WorkflowContext": "victor.workflows.context",
        "NodeResult": "victor.workflows.executor",
        "ExecutorNodeStatus": "victor.workflows.executor",
        "register_compute_handler": "victor.workflows.executor",
        "get_compute_handler": "victor.workflows.executor",
        # Workflow definition
        "ComputeNode": "victor.workflows.definition",
        "WorkflowBuilder": "victor.workflows.definition",
        "WorkflowDefinition": "victor.workflows.definition",
        "workflow": "victor.workflows.definition",
        "AgentNode": "victor.workflows.definition",
        "ConditionNode": "victor.workflows.definition",
        "ParallelNode": "victor.workflows.definition",
        # Code correction
        "CodeCorrectionMiddleware": "victor.agent.code_correction_middleware",
        "CodeCorrectionConfig": "victor.agent.code_correction_middleware",
        "CorrectionResult": "victor.agent.code_correction_middleware",
        # Code validation
        "CodeValidationResult": "victor.evaluation.correction.types",
        "Language": "victor.evaluation.correction.types",
        # Mode configuration
        "ModeConfigRegistry": "victor.core.mode_config",
        "ModeDefinition": "victor.core.mode_config",
        "ModeConfig": "victor.core.mode_config",
        "RegistryBasedModeConfigProvider": "victor.core.mode_config",
        # Service container
        "ServiceContainer": "victor.core.container",
        "ServiceLifetime": "victor.core.container",
        # Agent specs
        "AgentSpec": "victor.agent.specs.models",
        "AgentCapabilities": "victor.agent.specs.models",
        "AgentConstraints": "victor.agent.specs.models",
        "ModelPreference": "victor.agent.specs.models",
        "OutputFormat": "victor.agent.specs.models",
        "DelegationPolicy": "victor.agent.specs.models",
        # Sub-agents
        "SubAgent": "victor.agent.subagents.base",
        "SubAgentConfig": "victor.agent.subagents.base",
        "SubAgentResult": "victor.agent.subagents.base",
        "SubAgentRole": "victor.agent.subagents.base",
        "set_role_tool_provider": "victor.agent.subagents.protocols",
        # Middleware
        "MiddlewareChain": "victor.agent.middleware_chain",
        "MiddlewareAbortError": "victor.agent.middleware_chain",
        "create_middleware_chain": "victor.agent.middleware_chain",
        # Vertical context
        "VerticalContext": "victor.agent.vertical_context",
        "create_vertical_context": "victor.agent.vertical_context",
        "VerticalContextProtocol": "victor.agent.vertical_context",
        "MutableVerticalContextProtocol": "victor.agent.vertical_context",
        # Handler registry
        "HandlerRegistry": "victor.framework.handler_registry",
        "get_handler_registry": "victor.framework.handler_registry",
        "register_global_handler": "victor.framework.handler_registry",
        "register_vertical_handlers": "victor.framework.handler_registry",
        # Provider access
        "ProviderRegistry": "victor.providers.registry",
        # Vertical registration and definition-layer types
        "register_vertical": "victor_contracts",
        "VerticalBase": "victor_contracts",
        "StageDefinition": "victor_contracts",
        "VerticalConfig": "victor_contracts",
        "VerticalExtensions": "victor_contracts",
        # Tool dependency types
        "BaseToolDependencyProvider": "victor.core.tool_dependency_base",
        "ToolDependencyConfig": "victor.core.tool_dependency_base",
        "ToolDependency": "victor.core.tool_types",
        "YAMLToolDependencyProvider": "victor.core.tool_dependency_loader",
        "load_tool_dependency_yaml": "victor.core.tool_dependency_loader",
        "create_vertical_tool_dependency_provider": "victor.core.tool_dependency_loader",
        "TieredToolConfig": "victor.core.verticals.protocols",
        "TaskTypeHint": "victor.core.vertical_types",
        "ToolDependencyLoader": "victor.core.tool_dependency_loader",
        # Safety pattern types
        "SafetyExtensionProtocol": "victor.core.verticals.protocols",
        "SafetyPattern": "victor.core.verticals.protocols",
        # Promoted protocols
        "PromptContributorProtocol": "victor.core.verticals.protocols",
        "ModeConfigProviderProtocol": "victor.core.verticals.protocols",
        "ServiceProviderProtocol": "victor.core.verticals.protocols",
        "ToolDependencyProviderProtocol": "victor.core.verticals.protocols",
        "WorkflowProviderProtocol": "victor.core.verticals.protocols",
        "RLConfigProviderProtocol": "victor.core.verticals.protocols",
        "TeamSpecProviderProtocol": "victor.core.verticals.protocols",
        "EnrichmentStrategyProtocol": "victor.core.verticals.protocols",
        "MiddlewareProtocol": "victor.core.verticals.protocols",
    }

    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)

    raise AttributeError(f"module 'victor.framework.extensions' has no attribute {name!r}")
