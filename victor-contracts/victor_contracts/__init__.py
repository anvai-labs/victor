"""
Victor Contracts - Protocol definitions for vertical development.

This package provides pure protocol/ABC definitions that external verticals
can depend on without pulling in the entire Victor framework.

Version: Synchronized with victor-ai
"""

# ruff: noqa: E402

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from victor_contracts.workflow_runtime import (
        BaseYAMLWorkflowProvider as BaseYAMLWorkflowProvider,
        WorkflowBuilder as WorkflowBuilder,
        WorkflowDefinition as WorkflowDefinition,
        workflow as workflow,
    )

_MIN_SUPPORTED_PYTHON = (3, 12)


def _ensure_supported_python() -> None:
    """Fail fast with a clear error on unsupported interpreters."""
    current = sys.version_info[:3]
    if current >= _MIN_SUPPORTED_PYTHON:
        return

    required = ".".join(str(part) for part in _MIN_SUPPORTED_PYTHON)
    running = ".".join(str(part) for part in current)
    raise RuntimeError(
        f"victor-contracts requires Python {required}+; current interpreter is Python {running}. "
        f"Use Python {required} or newer."
    )


_ensure_supported_python()

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("victor-contracts")
except Exception:
    __version__ = "0.0.0"

# Core types
from victor_contracts.core.types import (
    CapabilityRequirement,
    CapabilityRequirementLike,
    CURRENT_DEFINITION_VERSION,
    MINIMUM_SUPPORTED_DEFINITION_VERSION,
    PromptMetadata,
    PromptTemplateDefinition,
    PromptTemplateLike,
    TeamDefinition,
    TeamDefinitionLike,
    TeamMemberDefinition,
    TeamMemberDefinitionLike,
    TeamMetadata,
    StageDefinitionLike,
    TaskTypeHintDefinition,
    TaskTypeHintLike,
    ToolRequirement,
    ToolRequirementLike,
    VerticalConfig,
    VerticalDefinition,
    is_supported_definition_version,
    WorkflowMetadata,
    StageDefinition,
    TieredToolConfig,
    ToolSet,
    normalize_capability_requirement,
    normalize_capability_requirements,
    normalize_prompt_metadata,
    normalize_prompt_template,
    normalize_prompt_templates,
    normalize_stage_definition,
    normalize_stage_definitions,
    normalize_task_type_hint,
    normalize_task_type_hints,
    normalize_team_definition,
    normalize_team_definitions,
    normalize_team_member_definition,
    normalize_team_member_definitions,
    normalize_team_metadata,
    normalize_tool_requirement,
    normalize_tool_requirements,
    validate_definition_version,
    normalize_workflow_metadata,
)
from victor_contracts.capabilities import (
    BaseCapabilityProvider,
    CapabilityConfigMergePolicy,
    CapabilityConfigScopePortProtocol,
    CapabilityConfigService,
    CapabilityEntry,
    CapabilityLoaderPortProtocol,
    CapabilityMetadata,
    CapabilityType,
    DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY,
    FileOperation,
    FileOperationsCapability,
    FileOperationType,
    OrchestratorCapability,
    PromptContribution,
    PromptContributionCapability,
    build_capability_loader,
    capability,
    load_capability_config,
    register_capability_entries,
    resolve_capability_config_scope_key,
    resolve_capability_config_service,
    store_capability_config,
    update_capability_config_section,
)
from victor_contracts.rl import (
    BaseRLConfig,
    DEFAULT_ACTIVE_LEARNERS,
    DEFAULT_PATIENCE_MAP,
    LearnerType,
)
from victor_contracts.conversation import (
    ConversationContext,
    ConversationCoordinator,
    ConversationStats,
    ConversationTurn,
    TurnType,
)
from victor_contracts.multi_agent import (
    CommunicationStyle,
    ExpertiseLevel,
    PersonaTemplate,
    PersonaTraits,
    TaskAssignmentStrategy,
    TeamMember,
    TeamSpec,
    TeamTemplate,
    TeamTopology,
)
from victor_contracts.team_schema import (
    TeamAgentCategory,
    TeamFormation,
    TeamMemberSpec,
)
from victor_contracts.tools import (
    AccessMode,
    CostTier,
    DangerLevel,
    ExecutionCategory,
    ToolCategory,
    ToolContract,
)
from victor_contracts.tools import CONTRACT as TOOL_CONTRACT
from victor_contracts.safety import (
    SafetyAction,
    SafetyCategory,
    SafetyCheckResult,
    SafetyCoordinator,
    SafetyRule,
    SafetyStats,
)
from victor_contracts.workflows import (
    ComputeHandlerProtocol,
    ComputeHandlerRegistrar,
    ComputeNodeProtocol,
    ExecutorNodeStatus,
    NodeResult,
    WorkflowContextProtocol,
    register_compute_handlers,
)
from victor_contracts.core.plugins import PluginContext, VictorPlugin
from victor_contracts.registries import (
    PersonaRegistryProtocol,
    TeamRegistryProtocol,
    get_default_persona_registry,
    get_default_team_registry,
    set_default_persona_registry,
    set_default_team_registry,
)

# Core exceptions
from victor_contracts.core.exceptions import (
    VerticalException,
    VerticalConfigurationError,
    VerticalProtocolError,
)

# Vertical protocols
from victor_contracts.verticals import (
    ExtensionDependency,
    MiddlewarePriority,
    MiddlewareResult,
    SafetyPattern,
    StaticModeConfigProvider,
    TaskTypeHint,
    ToolDependencyConfig,
    ToolDependency,
    ToolDependencyLoadError,
    ToolDependencyLoader,
    ToolDependencyProviderProtocol,
    YAMLToolDependencyProvider,
    VerticalModeConfig,
    create_tool_dependency_provider,
    get_cached_provider,
    invalidate_provider_cache,
    load_tool_dependency_yaml,
    register_vertical,
)
from victor_contracts.verticals.protocols.storage import CCGBuilderProtocol
from victor_contracts.verticals.mode_config import ModeDefinition
from victor_contracts.verticals.protocols.base import VerticalBase
from victor_contracts.verticals.protocols.capabilities import (
    CapabilityProvider,
    ChainProvider,
    PersonaProvider,
)

# Extension manifest
from victor_contracts.verticals.manifest import ExtensionManifest, ExtensionType
from victor_contracts.verticals.extensions import VerticalExtensions
from victor_contracts.verticals.protocols.mcp import McpProvider, McpToolProvider
from victor_contracts.verticals.protocols.sandbox import SandboxProvider
from victor_contracts.verticals.protocols.hooks import HookProvider, HookConfigProvider
from victor_contracts.verticals.protocols.permissions import PermissionProvider
from victor_contracts.verticals.protocols.compaction import CompactionProvider
from victor_contracts.verticals.protocols.plugins import ExternalPluginProvider
from victor_contracts.core.api_version import (
    CURRENT_API_VERSION,
    MIN_SUPPORTED_API_VERSION,
    is_compatible,
)

# Discovery and registration
from victor_contracts.discovery import (
    ProtocolRegistry,
    ProtocolMetadata,
    DiscoveryStats,
    collect_verticals_from_candidate,
    get_global_registry,
    reset_global_registry,
    discover_verticals,
    discover_protocols,
    get_discovery_summary,
    reload_discovery,
)
from victor_contracts.skills import SkillDefinition, SkillProvider

# Runtime-evaluation feedback contract (shared by victor.evaluation producer
# and victor.framework/victor.agent consumers; kept here so neither imports the
# other at module scope).
from victor_contracts.runtime_evaluation import RuntimeEvaluationFeedback

# Phase 4 promotions: types heavily used by external verticals
from victor_contracts.rl import RLOutcome, RLRecommendation
from victor_contracts.safety import SafetyLevel

from victor_contracts.validation import (
    ValidationIssue,
    ValidationReport,
    validate_vertical_package,
)

# Stable SDK constants
from victor_contracts.constants import (
    CapabilityIds,
    CANONICAL_TO_ALIASES,
    TOOL_ALIASES,
    ToolNameEntry,
    ToolNames,
    get_all_capability_ids,
    get_aliases,
    get_all_canonical_names,
    get_canonical_name,
    get_name_mapping,
    is_known_capability_id,
    is_valid_tool_name,
)

__all__ = [
    # Version
    "__version__",
    # Core types
    "CapabilityRequirement",
    "CapabilityRequirementLike",
    "CURRENT_DEFINITION_VERSION",
    "MINIMUM_SUPPORTED_DEFINITION_VERSION",
    "PromptMetadata",
    "PromptTemplateDefinition",
    "PromptTemplateLike",
    "TeamDefinition",
    "TeamDefinitionLike",
    "TeamMemberDefinition",
    "TeamMemberDefinitionLike",
    "TeamMetadata",
    "StageDefinitionLike",
    "TaskTypeHintDefinition",
    "TaskTypeHintLike",
    "DEFAULT_ACTIVE_LEARNERS",
    "DEFAULT_PATIENCE_MAP",
    "ToolRequirement",
    "ToolRequirementLike",
    "VerticalConfig",
    "VerticalDefinition",
    "LearnerType",
    "WorkflowContextProtocol",
    "ConversationContext",
    "ConversationCoordinator",
    "ConversationStats",
    "ConversationTurn",
    "CommunicationStyle",
    "ExpertiseLevel",
    "is_supported_definition_version",
    "PersonaTemplate",
    "PersonaTraits",
    "TaskAssignmentStrategy",
    "TeamAgentCategory",
    "TeamFormation",
    "TeamMember",
    "TeamMemberSpec",
    "TeamSpec",
    "TeamTemplate",
    "TeamTopology",
    # Tool contract (FEP-0009)
    "ToolContract",
    "ToolCategory",
    "AccessMode",
    "DangerLevel",
    "ExecutionCategory",
    "CostTier",
    "TOOL_CONTRACT",
    "WorkflowMetadata",
    "StageDefinition",
    "TurnType",
    "TieredToolConfig",
    "ToolSet",
    "normalize_capability_requirement",
    "normalize_capability_requirements",
    "normalize_prompt_metadata",
    "normalize_prompt_template",
    "normalize_prompt_templates",
    "normalize_stage_definition",
    "normalize_stage_definitions",
    "normalize_task_type_hint",
    "normalize_task_type_hints",
    "normalize_team_definition",
    "normalize_team_definitions",
    "normalize_team_member_definition",
    "normalize_team_member_definitions",
    "normalize_team_metadata",
    "normalize_tool_requirement",
    "normalize_tool_requirements",
    "validate_definition_version",
    "normalize_workflow_metadata",
    # Plugins
    "FileOperation",
    "FileOperationsCapability",
    "FileOperationType",
    "BaseCapabilityProvider",
    "BaseRLConfig",
    "ComputeHandlerProtocol",
    "ComputeHandlerRegistrar",
    "ComputeNodeProtocol",
    "CapabilityConfigMergePolicy",
    "CapabilityConfigScopePortProtocol",
    "CapabilityConfigService",
    "CapabilityEntry",
    "CapabilityLoaderPortProtocol",
    "CapabilityMetadata",
    "CapabilityType",
    "DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY",
    "ExecutorNodeStatus",
    "SafetyAction",
    "SafetyCategory",
    "SafetyCheckResult",
    "SafetyCoordinator",
    "SafetyRule",
    "SafetyStats",
    "NodeResult",
    "OrchestratorCapability",
    "PromptContribution",
    "PromptContributionCapability",
    "build_capability_loader",
    "capability",
    "load_capability_config",
    "register_compute_handlers",
    "register_capability_entries",
    "resolve_capability_config_scope_key",
    "resolve_capability_config_service",
    "store_capability_config",
    "update_capability_config_section",
    "PersonaRegistryProtocol",
    "TeamRegistryProtocol",
    "get_default_persona_registry",
    "get_default_team_registry",
    "set_default_persona_registry",
    "set_default_team_registry",
    "PluginContext",
    "VictorPlugin",
    # Exceptions
    "VerticalException",
    "VerticalConfigurationError",
    "VerticalProtocolError",
    # Base class
    "VerticalBase",
    "register_vertical",
    "CCGBuilderProtocol",
    "ExtensionDependency",
    # Capability protocols
    "CapabilityProvider",
    "ChainProvider",
    "PersonaProvider",
    "ToolDependencyConfig",
    "ToolDependency",
    "ToolDependencyLoadError",
    "ToolDependencyLoader",
    "ToolDependencyProviderProtocol",
    "YAMLToolDependencyProvider",
    "create_tool_dependency_provider",
    "get_cached_provider",
    "invalidate_provider_cache",
    "load_tool_dependency_yaml",
    "MiddlewarePriority",
    "MiddlewareResult",
    "SafetyPattern",
    "StaticModeConfigProvider",
    "TaskTypeHint",
    # Extension manifest
    "ExtensionManifest",
    "ExtensionType",
    "CURRENT_API_VERSION",
    "MIN_SUPPORTED_API_VERSION",
    "is_compatible",
    # Discovery (Phase 4)
    "ProtocolRegistry",
    "ProtocolMetadata",
    "DiscoveryStats",
    "collect_verticals_from_candidate",
    "get_global_registry",
    "reset_global_registry",
    "discover_verticals",
    "discover_protocols",
    "get_discovery_summary",
    "reload_discovery",
    # Validation
    "ValidationIssue",
    "ValidationReport",
    "validate_vertical_package",
    # Stable constants
    "CapabilityIds",
    "ToolNames",
    "ToolNameEntry",
    "TOOL_ALIASES",
    "CANONICAL_TO_ALIASES",
    "get_all_capability_ids",
    "get_canonical_name",
    "get_aliases",
    "is_valid_tool_name",
    "get_all_canonical_names",
    "get_name_mapping",
    "is_known_capability_id",
    # Lazy extension container
    "VerticalExtensions",
    # Extended vertical protocols: MCP, sandbox, hooks, permissions, compaction, plugins
    "McpProvider",
    "McpToolProvider",
    "SandboxProvider",
    "HookProvider",
    "HookConfigProvider",
    "PermissionProvider",
    "CompactionProvider",
    "ExternalPluginProvider",
    "VerticalModeConfig",
    "ModeDefinition",
    # Skills
    "SkillDefinition",
    "SkillProvider",
    # Runtime-evaluation feedback contract
    "RuntimeEvaluationFeedback",
    # Phase 4 promotions: types used by 6+ external verticals
    "TeamFormation",
    "TeamMemberSpec",
    "SafetyLevel",
    "RLOutcome",
    "RLRecommendation",
]

# Runtime-backed workflow helpers resolved from the Victor host runtime.
# Deferred via PEP 562 so `import victor_contracts` works without victor-ai
# installed; accessing these names without the host runtime raises ImportError.
# Intentionally excluded from __all__ so `from victor_contracts import *`
# (and the victor_sdk shim's re-export) stays standalone-safe.
_HOST_RUNTIME_EXPORTS = frozenset(
    {
        "BaseYAMLWorkflowProvider",
        "WorkflowBuilder",
        "WorkflowDefinition",
        "workflow",
    }
)


def __getattr__(name: str) -> Any:
    """Resolve host-runtime-backed exports lazily (PEP 562)."""
    if name in _HOST_RUNTIME_EXPORTS:
        from victor_contracts import workflow_runtime

        return getattr(workflow_runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
