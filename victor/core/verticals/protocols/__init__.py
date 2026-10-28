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


# VerticalExtensions canonical definition lives in the SDK (victor-contracts
# >=0.10.0 ships victor_contracts/verticals/extensions.py; the root pins that
# floor, so a resolved SDK absence is an install bug, not a compatibility
# case — fail loudly instead of silently running the former 250-line fork.
if _SDK_VERTICAL_EXTENSIONS is None:
    raise ImportError(
        "victor_contracts.verticals.extensions.VerticalExtensions is not "
        "resolvable; victor-ai requires victor-contracts>=0.10.0. Reinstall "
        "with `pip install -e ./victor-contracts` (or upgrade the package)."
    )
VerticalExtensions = _SDK_VERTICAL_EXTENSIONS


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
