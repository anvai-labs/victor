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

"""Vertical Extension Protocols Package (ISP-Compliant).

This package provides segregated protocol interfaces for vertical-framework
integration, following the Interface Segregation Principle (ISP).

Protocol definitions are now canonically defined in victor-contracts to allow
external verticals to depend on the SDK only. This module re-exports them
for backward compatibility.

Preferred import path (external verticals):
    from victor_contracts.verticals.protocols import (
        MiddlewareProtocol,
        SafetyExtensionProtocol,
        PromptContributorProtocol,
        ModeConfigProviderProtocol,
    )

Legacy import path (still works):
    from victor.core.verticals.protocols import (
        MiddlewareProtocol,
        SafetyExtensionProtocol,
        PromptContributorProtocol,
        ModeConfigProviderProtocol,
    )
"""

# =============================================================================
# Re-export promoted protocols from SDK (canonical source)
# =============================================================================

try:
    from victor_contracts.verticals.protocols.promoted import (
        # Tool Selection
        ToolSelectionStrategyProtocol,
        VerticalToolSelectionProviderProtocol,
        TieredToolConfigProviderProtocol,
        VerticalTieredToolProviderProtocol,
        # Safety
        SafetyExtensionProtocol,
        # Team
        TeamSpecProviderProtocol,
        VerticalTeamProviderProtocol,
        # Middleware
        MiddlewareProtocol,
        # Prompt
        PromptContributorProtocol,
        # Mode
        ModeConfigProviderProtocol,
        # Workflow
        WorkflowProviderProtocol,
        VerticalWorkflowProviderProtocol,
        # Service
        ServiceProviderProtocol,
        # RL
        RLConfigProviderProtocol,
        VerticalRLProviderProtocol,
        # Enrichment
        EnrichmentStrategyProtocol,
        VerticalEnrichmentProviderProtocol,
        # Capability
        CapabilityProviderProtocol,
        ChainProviderProtocol,
        PersonaProviderProtocol,
        VerticalPersonaProviderProtocol,
        # Stage Contract
        StageContract,
        StageValidator,
        StageValidationResult,
        ValidationError,
        validate_stage_contract,
        StageContractMixin,
    )

    # Re-export promoted data types from SDK
    from victor_contracts.verticals.protocols.promoted_types import (
        MiddlewarePriority,
        MiddlewareResult,
        ModeConfig,
        ToolSelectionContext,
        ToolSelectionResult,
    )

    _SDK_AVAILABLE = True
except ImportError:
    # SDK not installed yet (e.g., during victor-ai installation)
    # These will be available after victor-contracts is installed
    _SDK_AVAILABLE = False

# Re-export SafetyPattern from its original location (dataclass, not Protocol)
from victor.security.safety.types import SafetyPattern

# Re-export TaskTypeHint from its original location (dataclass)
from victor.core.vertical_types import TaskTypeHint

# Backward compat alias for ToolSelectionContext
if _SDK_AVAILABLE:
    VerticalToolSelectionContext = ToolSelectionContext

# =============================================================================
# ISP-Compliant Vertical Providers (re-exported from SDK)
# =============================================================================

if _SDK_AVAILABLE:
    from victor_contracts.verticals.protocols import (
        MiddlewareProvider,
        SafetyProvider,
        WorkflowProvider,
        TeamProvider,
        RLProvider,
        EnrichmentProvider,
        ToolProvider,
        HandlerProvider,
        CapabilityProvider,
        ModeConfigProvider,
    )

# These provider protocols are defined locally (have methods not in SDK versions)
from victor.core.verticals.protocols.providers import (
    PromptContributorProvider,
    ToolDependencyProvider,
    TieredToolConfigProvider,
    ServiceProvider,
)

# Re-export from core for backward compatibility
from victor.core.tool_types import ToolDependency, ToolDependencyProviderProtocol
from victor.core.vertical_types import StageDefinition, TieredToolConfig

# =============================================================================
# Composite Vertical Extension
# =============================================================================

import importlib
from typing import Any, Callable, Dict, List, Optional, Union


def _load_sdk_vertical_extensions() -> Optional[type[Any]]:
    """Resolve the SDK VerticalExtensions type across compatible SDK layouts."""

    try:
        return importlib.import_module("victor_contracts.verticals.extensions").VerticalExtensions
    except (AttributeError, ImportError, ModuleNotFoundError):
        pass

    try:
        return importlib.import_module("victor_contracts.verticals").VerticalExtensions
    except (AttributeError, ImportError, ModuleNotFoundError):
        return None


_SDK_VERTICAL_EXTENSIONS = _load_sdk_vertical_extensions()


class _VerticalExtensionsFallback:
    """Container for all vertical extension implementations.

    Supports both eager (concrete values) and lazy (factory callables)
    construction. When constructed with callables, each extension is
    loaded on first access, eliminating the synchronous import storm
    that occurs when all 11+ extensions are loaded eagerly at activation.

    Eager construction (backward compatible)::

        ext = VerticalExtensions(
            middleware=[my_middleware],
            safety_extensions=[my_safety],
        )

    Lazy construction (new — used by get_extensions())::

        ext = VerticalExtensions(
            middleware=lambda: load_middleware(),
            safety_extensions=lambda: [load_safety()],
        )
        # load_middleware() is NOT called until ext.middleware is accessed.
    """

    # Field definitions: (name, is_list, default)
    _FIELDS = [
        ("middleware", True),
        ("safety_extensions", True),
        ("prompt_contributors", True),
        ("mode_config_provider", False),
        ("tool_dependency_provider", False),
        ("workflow_provider", False),
        ("service_provider", False),
        ("rl_config_provider", False),
        ("team_spec_provider", False),
        ("enrichment_strategy", False),
        ("tool_selection_strategy", False),
        ("tiered_tool_config", False),
    ]

    def __init__(
        self,
        middleware: Union[List[Any], Callable, None] = None,
        safety_extensions: Union[List[Any], Callable, None] = None,
        prompt_contributors: Union[List[Any], Callable, None] = None,
        mode_config_provider: Union[Any, Callable, None] = None,
        tool_dependency_provider: Union[Any, Callable, None] = None,
        workflow_provider: Union[Any, Callable, None] = None,
        service_provider: Union[Any, Callable, None] = None,
        rl_config_provider: Union[Any, Callable, None] = None,
        team_spec_provider: Union[Any, Callable, None] = None,
        enrichment_strategy: Union[Any, Callable, None] = None,
        tool_selection_strategy: Union[Any, Callable, None] = None,
        tiered_tool_config: Union[Any, Callable, None] = None,
    ) -> None:
        # Store each value as either a resolved value or a factory.
        # _resolved stores final values; _factories stores callables.
        self._resolved: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}

        args = {
            "middleware": middleware,
            "safety_extensions": safety_extensions,
            "prompt_contributors": prompt_contributors,
            "mode_config_provider": mode_config_provider,
            "tool_dependency_provider": tool_dependency_provider,
            "workflow_provider": workflow_provider,
            "service_provider": service_provider,
            "rl_config_provider": rl_config_provider,
            "team_spec_provider": team_spec_provider,
            "enrichment_strategy": enrichment_strategy,
            "tool_selection_strategy": tool_selection_strategy,
            "tiered_tool_config": tiered_tool_config,
        }

        for name, is_list in self._FIELDS:
            val = args.get(name)
            if val is None:
                # Use default immediately
                self._resolved[name] = [] if is_list else None
            elif callable(val) and not isinstance(val, list):
                # Store factory for lazy resolution
                self._factories[name] = val
            else:
                # Concrete value — store directly
                self._resolved[name] = val

    def _resolve(self, name: str) -> Any:
        """Resolve a field, calling its factory on first access."""
        if name in self._resolved:
            return self._resolved[name]
        factory = self._factories.pop(name, None)
        if factory is not None:
            try:
                value = factory()
            except Exception:
                # On factory failure, use default
                is_list = any(n == name and il for n, il in self._FIELDS)
                value = [] if is_list else None
            self._resolved[name] = value
            return value
        # Fallback default
        is_list = any(n == name and il for n, il in self._FIELDS)
        val = [] if is_list else None
        self._resolved[name] = val
        return val

    # --- Properties for each field ---

    @property
    def middleware(self) -> List[Any]:
        return self._resolve("middleware")

    @middleware.setter
    def middleware(self, value: Any) -> None:
        self._factories.pop("middleware", None)
        self._resolved["middleware"] = value

    @property
    def safety_extensions(self) -> List[Any]:
        return self._resolve("safety_extensions")

    @safety_extensions.setter
    def safety_extensions(self, value: Any) -> None:
        self._factories.pop("safety_extensions", None)
        self._resolved["safety_extensions"] = value

    @property
    def prompt_contributors(self) -> List[Any]:
        return self._resolve("prompt_contributors")

    @prompt_contributors.setter
    def prompt_contributors(self, value: Any) -> None:
        self._factories.pop("prompt_contributors", None)
        self._resolved["prompt_contributors"] = value

    @property
    def mode_config_provider(self) -> Optional[Any]:
        return self._resolve("mode_config_provider")

    @mode_config_provider.setter
    def mode_config_provider(self, value: Any) -> None:
        self._factories.pop("mode_config_provider", None)
        self._resolved["mode_config_provider"] = value

    @property
    def tool_dependency_provider(self) -> Optional[Any]:
        return self._resolve("tool_dependency_provider")

    @tool_dependency_provider.setter
    def tool_dependency_provider(self, value: Any) -> None:
        self._factories.pop("tool_dependency_provider", None)
        self._resolved["tool_dependency_provider"] = value

    @property
    def workflow_provider(self) -> Optional[Any]:
        return self._resolve("workflow_provider")

    @workflow_provider.setter
    def workflow_provider(self, value: Any) -> None:
        self._factories.pop("workflow_provider", None)
        self._resolved["workflow_provider"] = value

    @property
    def service_provider(self) -> Optional[Any]:
        return self._resolve("service_provider")

    @service_provider.setter
    def service_provider(self, value: Any) -> None:
        self._factories.pop("service_provider", None)
        self._resolved["service_provider"] = value

    @property
    def rl_config_provider(self) -> Optional[Any]:
        return self._resolve("rl_config_provider")

    @rl_config_provider.setter
    def rl_config_provider(self, value: Any) -> None:
        self._factories.pop("rl_config_provider", None)
        self._resolved["rl_config_provider"] = value

    @property
    def team_spec_provider(self) -> Optional[Any]:
        return self._resolve("team_spec_provider")

    @team_spec_provider.setter
    def team_spec_provider(self, value: Any) -> None:
        self._factories.pop("team_spec_provider", None)
        self._resolved["team_spec_provider"] = value

    @property
    def enrichment_strategy(self) -> Optional[Any]:
        return self._resolve("enrichment_strategy")

    @enrichment_strategy.setter
    def enrichment_strategy(self, value: Any) -> None:
        self._factories.pop("enrichment_strategy", None)
        self._resolved["enrichment_strategy"] = value

    @property
    def tool_selection_strategy(self) -> Optional[Any]:
        return self._resolve("tool_selection_strategy")

    @tool_selection_strategy.setter
    def tool_selection_strategy(self, value: Any) -> None:
        self._factories.pop("tool_selection_strategy", None)
        self._resolved["tool_selection_strategy"] = value

    @property
    def tiered_tool_config(self) -> Optional[Any]:
        return self._resolve("tiered_tool_config")

    @tiered_tool_config.setter
    def tiered_tool_config(self, value: Any) -> None:
        self._factories.pop("tiered_tool_config", None)
        self._resolved["tiered_tool_config"] = value

    # --- Convenience methods (preserved from original) ---

    def get_all_task_hints(self) -> Dict[str, "TaskTypeHint"]:
        """Merge task hints from all contributors."""
        merged: Dict[str, Any] = {}
        for contributor in sorted(self.prompt_contributors, key=lambda c: c.get_priority()):
            merged.update(contributor.get_task_type_hints())
        return merged

    def get_all_safety_patterns(self) -> List["SafetyPattern"]:
        """Collect safety patterns from all extensions."""
        patterns: List[Any] = []
        for ext in self.safety_extensions:
            patterns.extend(ext.get_bash_patterns())
            patterns.extend(ext.get_file_patterns())
        return patterns

    def get_all_mode_configs(self) -> Dict[str, "ModeConfig"]:
        """Get mode configs from provider."""
        if self.mode_config_provider:
            return self.mode_config_provider.get_mode_configs()
        return {}

    @property
    def pending_factories(self) -> int:
        """Number of extensions not yet loaded (still deferred)."""
        return len(self._factories)


# VerticalExtensions canonical definition lives in the SDK.
# Re-exported here for backward compatibility, with a local fallback for
# package-install flows that resolve an older victor-contracts layout.
VerticalExtensions = _SDK_VERTICAL_EXTENSIONS or _VerticalExtensionsFallback


__all__ = [
    # Tool Selection
    "ToolSelectionContext",
    "ToolSelectionResult",
    "ToolSelectionStrategyProtocol",
    "VerticalToolSelectionProviderProtocol",
    "TieredToolConfigProviderProtocol",
    "VerticalTieredToolProviderProtocol",
    # Safety
    "SafetyExtensionProtocol",
    "SafetyPattern",
    # Team
    "TeamSpecProviderProtocol",
    "VerticalTeamProviderProtocol",
    # Middleware
    "MiddlewareProtocol",
    "MiddlewarePriority",
    "MiddlewareResult",
    # Prompt
    "PromptContributorProtocol",
    "TaskTypeHint",
    # Mode
    "ModeConfig",
    "ModeConfigProviderProtocol",
    # Workflow
    "WorkflowProviderProtocol",
    "VerticalWorkflowProviderProtocol",
    # Service
    "ServiceProviderProtocol",
    # RL
    "RLConfigProviderProtocol",
    "VerticalRLProviderProtocol",
    # Enrichment
    "EnrichmentStrategyProtocol",
    "VerticalEnrichmentProviderProtocol",
    # Capability
    "CapabilityProviderProtocol",
    "ChainProviderProtocol",
    "PersonaProviderProtocol",
    "VerticalPersonaProviderProtocol",
    # Stage Contract (Phase 2: LSP Compliance)
    "StageContract",
    "StageValidator",
    "StageValidationResult",
    "ValidationError",
    "validate_stage_contract",
    "StageContractMixin",
    # Re-exports from core
    "ToolDependency",
    "ToolDependencyProviderProtocol",
    "StageDefinition",
    "TieredToolConfig",
    # Composite
    "VerticalExtensions",
    # ISP-Compliant Vertical Providers
    "MiddlewareProvider",
    "SafetyProvider",
    "WorkflowProvider",
    "TeamProvider",
    "RLProvider",
    "EnrichmentProvider",
    "ToolProvider",
    "HandlerProvider",
    "CapabilityProvider",
    "ModeConfigProvider",
    "PromptContributorProvider",
    "ToolDependencyProvider",
    "TieredToolConfigProvider",
    "ServiceProvider",
]
