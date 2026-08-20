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

"""Victor contracts protocol discovery integration.

This module integrates the victor-ai framework with the contracts
protocol discovery system, enabling the framework to discover protocols
and capabilities from external verticals via entry points.

Entry Point Groups:
    - victor.extension.protocols: Protocol implementations (preferred)
    - victor.extension.capabilities: Capability providers (preferred)
    - victor.extension.validators: Validator functions (preferred)
    - victor.sdk.protocols: Legacy protocol implementations
    - victor.sdk.capabilities: Legacy capability providers
    - victor.sdk.validators: Legacy validator functions

This module provides a convenience layer over the contracts discovery
module for framework integration.

Usage:
    from victor.core.verticals.sdk_discovery import (
        discover_sdk_protocols,
        get_sdk_protocol_registry,
        get_sdk_tool_providers,
        get_sdk_capability_providers,
    )

    # Discover all SDK protocols from external verticals
    protocols = discover_sdk_protocols()

    # Get specific protocol types
    tool_providers = get_sdk_tool_providers()
    capabilities = get_sdk_capability_providers()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import victor-contracts discovery system
try:
    from victor_contracts.discovery import (
        ProtocolRegistry,
        get_global_registry,
        reset_global_registry,
        discover_verticals,
        discover_protocols,
        get_discovery_summary,
        reload_discovery,
        DiscoveryStats,
        ProtocolMetadata,
    )
except ImportError:
    # victor-contracts not installed - provide stubs for graceful degradation
    ProtocolRegistry = None

    def get_global_registry(*args, **kwargs):
        return None

    def reset_global_registry(*args, **kwargs):
        return None

    def discover_verticals(*args, **kwargs):
        return {}

    def discover_protocols(*args, **kwargs):
        return {}

    def get_discovery_summary(*args, **kwargs):
        return "victor-contracts not installed"

    def reload_discovery(*args, **kwargs):
        return DiscoveryStats()

    class DiscoveryStats:
        total_protocols: int = 0
        total_capabilities: int = 0
        failed_loads: int = 0

    class ProtocolMetadata:
        pass

    logger.warning(
        "victor-contracts not installed. SDK protocol discovery disabled. "
        "Install victor-contracts to enable protocol-based discovery from external verticals."
    )


# =============================================================================
# Framework Integration Functions
# =============================================================================


def get_sdk_protocol_registry() -> Optional[ProtocolRegistry]:
    """Get the global victor-contracts protocol registry instance.

    Returns:
        ProtocolRegistry instance or None if victor-contracts not installed
    """
    if get_global_registry is None:
        return None
    return get_global_registry()


def discover_sdk_protocols(*, reload: bool = False) -> DiscoveryStats:
    """Discover all SDK protocols from external verticals.

    This function loads protocol implementations from installed packages
    that register them via entry points.

    Args:
        reload: If True, reload even if already loaded

    Returns:
        DiscoveryStats with information about what was discovered

    Example:
        # In victor-coding's pyproject.toml:
        # [project.entry-points."victor.extension.protocols"]
        # coding-tools = "victor_coding.protocols:CodingToolProvider"
        #
        # Legacy packages may still use:
        # [project.entry-points."victor.sdk.protocols"]
        # coding-tools = "victor_coding.protocols:CodingToolProvider"

        stats = discover_sdk_protocols()
        print(f"Discovered {stats.total_protocols} protocols")
    """
    if get_global_registry is None:
        logger.debug("victor-contracts not installed, skipping protocol discovery")
        return DiscoveryStats()

    registry = get_global_registry()
    return registry.load_from_entry_points(reload=reload)


def get_sdk_tool_providers() -> List[Any]:
    """Get all discovered tool providers from SDK protocols.

    Returns:
        List of ToolProvider instances

    Example:
        tool_providers = get_sdk_tool_providers()
        for provider in tool_providers:
            tools = provider.get_tools()
            print(f"Tools: {tools}")
    """
    if get_global_registry is None:
        return []

    registry = get_global_registry()
    return registry.get_tool_providers()


def get_sdk_safety_providers() -> List[Any]:
    """Get all discovered safety providers from SDK protocols.

    Returns:
        List of SafetyProvider instances
    """
    if get_global_registry is None:
        return []

    registry = get_global_registry()
    return registry.get_safety_providers()


def get_sdk_workflow_providers() -> List[Any]:
    """Get all discovered workflow providers from SDK protocols.

    Returns:
        List of WorkflowProvider instances
    """
    if get_global_registry is None:
        return []

    registry = get_global_registry()
    return registry.get_workflow_providers()


def get_sdk_prompt_providers() -> List[Any]:
    """Get all discovered prompt providers from SDK protocols.

    Returns:
        List of PromptProvider instances
    """
    if get_global_registry is None:
        return []

    registry = get_global_registry()
    return registry.get_prompt_providers()


def get_sdk_capability_providers() -> Dict[str, Any]:
    """Get all discovered capability providers from SDK protocols.

    Returns:
        Dictionary mapping capability names to providers

    Example:
        capabilities = get_sdk_capability_providers()
        lsp_capability = capabilities.get("coding-lsp")
        if lsp_capability:
            lsp_capability.configure_capability(orchestrator)
    """
    if get_global_registry is None:
        return {}

    registry = get_global_registry()
    return registry.get_capability_providers()


def get_sdk_capability_provider(name: str) -> Optional[Any]:
    """Get a specific capability provider by name.

    Args:
        name: Capability name (e.g., "coding-lsp", "coding-git")

    Returns:
        Capability provider instance or None
    """
    if get_global_registry is None:
        return None

    registry = get_global_registry()
    return registry.get_capability_provider(name)


def get_sdk_validators() -> Dict[str, Any]:
    """Get all discovered validators from SDK protocols.

    Returns:
        Dictionary mapping validator names to validator functions
    """
    if get_global_registry is None:
        return {}

    registry = get_global_registry()
    return registry.get_validators()


def get_sdk_validator(name: str) -> Optional[Any]:
    """Get a specific validator by name.

    Args:
        name: Validator name

    Returns:
        Validator function or None
    """
    if get_global_registry is None:
        return None

    registry = get_global_registry()
    return registry.get_validator(name)


def get_sdk_discovery_stats() -> DiscoveryStats:
    """Get statistics about SDK protocol discovery.

    Returns:
        DiscoveryStats object with discovery information
    """
    if get_global_registry is None:
        return DiscoveryStats()

    registry = get_global_registry()
    return registry.get_discovery_stats()


def get_sdk_discovery_summary() -> str:
    """Get a human-readable summary of SDK protocol discovery.

    Returns:
        Formatted string with discovery information
    """
    if get_discovery_summary is None:
        return "victor-contracts not installed"

    return get_discovery_summary()


def reload_sdk_discovery() -> DiscoveryStats:
    """Reload all SDK protocols from entry points.

    This clears the existing registry and reloads everything.
    Useful for testing or when new packages are installed.

    Returns:
        DiscoveryStats with information about what was discovered
    """
    if reload_discovery is None:
        return DiscoveryStats()

    return reload_discovery()


def reset_sdk_discovery() -> None:
    """Reset the global SDK protocol registry (for testing)."""
    if reset_global_registry is not None:
        reset_global_registry()


def list_sdk_capabilities() -> List[str]:
    """List all registered SDK capability names.

    Returns:
        List of capability names
    """
    if get_global_registry is None:
        return []

    registry = get_global_registry()
    return registry.list_capability_names()


def list_sdk_validators() -> List[str]:
    """List all registered SDK validator names.

    Returns:
        List of validator names
    """
    if get_global_registry is None:
        return []

    registry = get_global_registry()
    return registry.list_validator_names()


# =============================================================================
# Vertical Enhancement via SDK Protocols
# =============================================================================


def enhance_vertical_with_sdk_protocols(
    vertical_name: str,
    vertical_extensions: Any,
) -> None:
    """Enhance a vertical with discovered SDK protocols.

    This function integrates SDK-provided protocols with a vertical's
    extension system, enabling external verticals to contribute
    capabilities without direct dependencies.

    Args:
        vertical_name: Name of the vertical to enhance
        vertical_extensions: VerticalExtensions object to enhance

    Example:
        from victor.core.verticals.sdk_discovery import enhance_vertical_with_sdk_protocols

        # In vertical activation code
        enhance_vertical_with_sdk_protocols("coding", extensions)
    """
    if get_global_registry is None:
        logger.debug("victor-contracts not installed, skipping protocol enhancement")
        return

    registry = get_global_registry()

    # Get tool providers for this vertical
    tool_providers = registry.get_tool_providers()
    if tool_providers:
        logger.debug(f"Found {len(tool_providers)} tool providers for {vertical_name}")
        # Integrate tool providers with vertical extensions
        if hasattr(vertical_extensions, "add_tool_provider"):
            for provider in tool_providers:
                vertical_extensions.add_tool_provider(provider)

    # Get safety providers for this vertical
    safety_providers = registry.get_safety_providers()
    if safety_providers:
        logger.debug(f"Found {len(safety_providers)} safety providers for {vertical_name}")
        if hasattr(vertical_extensions, "add_safety_provider"):
            for provider in safety_providers:
                vertical_extensions.add_safety_provider(provider)

    # Get workflow providers for this vertical
    workflow_providers = registry.get_workflow_providers()
    if workflow_providers:
        logger.debug(f"Found {len(workflow_providers)} workflow providers for {vertical_name}")
        if hasattr(vertical_extensions, "add_workflow_provider"):
            for provider in workflow_providers:
                vertical_extensions.add_workflow_provider(provider)


# =============================================================================
# Module Initialization
# =============================================================================


def _initialize_sdk_discovery() -> None:
    """Initialize SDK protocol discovery on module import.

    This ensures that SDK protocols are discovered early in the
    application lifecycle, making them available for vertical
    enhancement.
    """
    try:
        discover_sdk_protocols()
        stats = get_sdk_discovery_stats()
        if stats.total_protocols > 0:
            logger.info(
                f"SDK protocol discovery initialized: "
                f"{stats.total_protocols} protocols, "
                f"{stats.total_capabilities} capabilities"
            )
    except Exception as e:
        logger.debug(f"Failed to initialize SDK discovery: {e}")


# Auto-initialize on import (optional - can be made lazy)
# _initialize_sdk_discovery()


__all__ = [
    # Registry access
    "get_sdk_protocol_registry",
    "discover_sdk_protocols",
    "reload_sdk_discovery",
    "reset_sdk_discovery",
    # Protocol providers
    "get_sdk_tool_providers",
    "get_sdk_safety_providers",
    "get_sdk_workflow_providers",
    "get_sdk_prompt_providers",
    # Capability providers
    "get_sdk_capability_providers",
    "get_sdk_capability_provider",
    # Validators
    "get_sdk_validators",
    "get_sdk_validator",
    # Discovery info
    "get_sdk_discovery_stats",
    "get_sdk_discovery_summary",
    "list_sdk_capabilities",
    "list_sdk_validators",
    # Vertical enhancement
    "enhance_vertical_with_sdk_protocols",
    # Types
    "ProtocolRegistry",
    "DiscoveryStats",
    "ProtocolMetadata",
]
