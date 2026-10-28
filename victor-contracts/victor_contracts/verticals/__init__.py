"""Vertical protocols and base classes.

This module provides protocol definitions and base classes for creating
Victor verticals without runtime dependencies.
"""

from victor_contracts.verticals.registration import (
    ExtensionDependency,
    register_vertical,
)
from victor_contracts.verticals.protocols.base import VerticalBase
from victor_contracts.verticals.extensions import VerticalExtensions
from victor_contracts.verticals.mode_config import (
    ModeConfig as StaticModeConfig,
    ModeDefinition,
    StaticModeConfigProvider,
    VerticalModeConfig,
)
from victor_contracts.verticals.tool_dependencies import (
    ToolDependencyConfig,
    ToolDependencyLoadError,
    ToolDependencyLoader,
    YAMLToolDependencyProvider,
    create_tool_dependency_provider,
    get_cached_provider,
    invalidate_provider_cache,
    load_tool_dependency_yaml,
)
from victor_contracts.verticals.protocols import (
    # SDK-native protocols
    CapabilityProvider,
    ToolDependency,
    ToolDependencyProviderProtocol,
    ToolProvider,
    ToolSelectionStrategy,
    SafetyProvider,
    SafetyExtension,
    PromptProvider,
    PromptContributor,
    WorkflowProvider,
    HandlerProvider,
    TeamProvider,
    MiddlewareProvider,
    ModeConfigProvider,
    RLProvider,
    EnrichmentProvider,
    ServiceProvider,
    # Promoted protocols (from victor.core.verticals.protocols)
    MiddlewareProtocol,
    SafetyExtensionProtocol,
    PromptContributorProtocol,
    ModeConfigProviderProtocol,
    ToolSelectionStrategyProtocol,
    WorkflowProviderProtocol,
    ServiceProviderProtocol,
    RLConfigProviderProtocol,
    EnrichmentStrategyProtocol,
    CapabilityProviderProtocol,
    TeamSpecProviderProtocol,
    ChainProviderProtocol,
    PersonaProviderProtocol,
)
from victor_contracts.verticals.conversation_kernel import VerticalConversationManager
from victor_contracts.verticals.safety_kernel import CoordinatorSafetyExtension
from victor_contracts.verticals.provider_kernel import (
    make_pattern_safety_provider,
    make_prompt_provider,
    make_stage_tool_strategy,
    make_tool_provider,
    make_workflow_provider,
)
from victor_contracts.verticals.protocols.promoted_types import (
    MiddlewarePriority,
    MiddlewareResult,
    ModeConfig,
    SafetyPattern,
    TaskTypeHint,
)

__all__ = [
    "register_vertical",
    "ExtensionDependency",
    "VerticalBase",
    "VerticalExtensions",
    "StaticModeConfig",
    "ModeDefinition",
    "StaticModeConfigProvider",
    "VerticalModeConfig",
    "ToolDependencyConfig",
    "ToolDependencyLoadError",
    "ToolDependencyLoader",
    "YAMLToolDependencyProvider",
    "create_tool_dependency_provider",
    "get_cached_provider",
    "invalidate_provider_cache",
    "load_tool_dependency_yaml",
    # SDK-native protocols
    "ToolDependency",
    "ToolDependencyProviderProtocol",
    "ToolProvider",
    "ToolSelectionStrategy",
    "SafetyProvider",
    "SafetyExtension",
    "PromptProvider",
    "PromptContributor",
    "WorkflowProvider",
    "HandlerProvider",
    "TeamProvider",
    "MiddlewareProvider",
    "ModeConfigProvider",
    "RLProvider",
    "EnrichmentProvider",
    "ServiceProvider",
    "CapabilityProvider",
    # Promoted data helpers
    "MiddlewarePriority",
    "MiddlewareResult",
    "ModeConfig",
    "SafetyPattern",
    "TaskTypeHint",
    # Promoted protocols
    "MiddlewareProtocol",
    "CoordinatorSafetyExtension",
    "VerticalConversationManager",
    "make_pattern_safety_provider",
    "make_prompt_provider",
    "make_stage_tool_strategy",
    "make_tool_provider",
    "make_workflow_provider",
    "SafetyExtensionProtocol",
    "PromptContributorProtocol",
    "ModeConfigProviderProtocol",
    "ToolSelectionStrategyProtocol",
    "WorkflowProviderProtocol",
    "ServiceProviderProtocol",
    "RLConfigProviderProtocol",
    "EnrichmentStrategyProtocol",
    "CapabilityProviderProtocol",
    "TeamSpecProviderProtocol",
    "ChainProviderProtocol",
    "PersonaProviderProtocol",
]
