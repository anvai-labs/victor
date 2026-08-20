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

"""Centralized capability registry for optional vertical capabilities.

This module provides a singleton registry that manages optional capabilities
(tree-sitter parsing, codebase indexing, LSP, etc.) with stub/enhanced semantics.

At bootstrap time:
1. All contrib stubs are registered with status STUB
2. Entry points from installed verticals (e.g., victor-coding) are discovered
3. Enhanced implementations override stubs via entry points

This eliminates direct imports from external vertical packages (e.g., victor_coding)
in victor core, making the framework fully self-contained.

Usage:
    from victor.core.capability_registry import capabilities
    from victor.framework.vertical_protocols import TreeSitterParserProtocol

    parser = capabilities.get(TreeSitterParserProtocol)
    if parser is not None:
        result = parser.get_parser("python")
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CapabilityStatus(Enum):
    """Status of a registered capability provider."""

    STUB = "stub"
    ENHANCED = "enhanced"


class OptionalFeatureRegistry:
    """Singleton registry for optional feature capabilities.

    Manages optional capabilities (tree-sitter parsing, codebase indexing, LSP, etc.)
    with stub/enhanced provider semantics.

    Capabilities are registered with a protocol type as key and a provider
    instance as value. Each registration has a status (STUB or ENHANCED).

    Enhanced registrations will not be downgraded to STUB. This ensures
    that once a vertical installs an enhanced provider, it stays active.

    Note: Previously named CapabilityRegistry - renamed for clarity.
    """

    _instance: Optional["OptionalFeatureRegistry"] = None
    _providers: Dict[Type, tuple[Any, CapabilityStatus]]
    _metadata: Dict[Type, Dict[str, Any]]

    def __init__(self) -> None:
        self._providers = {}
        self._metadata = {}
        self._bootstrapped = False

    @classmethod
    def get_instance(cls) -> "OptionalFeatureRegistry":
        """Get the singleton registry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance. For testing only."""
        cls._instance = None

    def ensure_bootstrapped(self) -> None:
        """Ensure capabilities have been bootstrapped (lazy init)."""
        if not self._bootstrapped:
            self._bootstrapped = True
            from victor.core.bootstrap import bootstrap_capabilities

            bootstrap_capabilities()

    def register(
        self,
        protocol_type: Type,
        provider: Any,
        status: CapabilityStatus = CapabilityStatus.STUB,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a capability provider.

        Enhanced providers will not be downgraded to STUB. Registering
        a STUB when an ENHANCED provider already exists is a no-op.

        Args:
            protocol_type: The protocol type to register against
            provider: The provider instance
            status: STUB or ENHANCED
            metadata: Optional metadata describing the provider registration.
        """
        existing = self._providers.get(protocol_type)
        if existing is not None:
            _, existing_status = existing
            if existing_status == CapabilityStatus.ENHANCED and status == CapabilityStatus.STUB:
                logger.debug(
                    f"Skipping STUB registration for {protocol_type.__name__} "
                    f"— ENHANCED provider already registered"
                )
                return
            if existing_status == CapabilityStatus.ENHANCED and status == CapabilityStatus.ENHANCED:
                logger.debug(
                    f"Skipping duplicate ENHANCED registration for {protocol_type.__name__} "
                    f"— provider already registered"
                )
                return

        self._providers[protocol_type] = (provider, status)
        if metadata is not None:
            self._metadata[protocol_type] = dict(metadata)
        else:
            self._metadata.pop(protocol_type, None)
        logger.debug(f"Registered {status.value} capability for {protocol_type.__name__}")

    def get(self, protocol_type: Type[T]) -> Optional[T]:
        """Get a capability provider by protocol type.

        Args:
            protocol_type: The protocol type to look up

        Returns:
            The provider instance, or None if not registered
        """
        self.ensure_bootstrapped()
        entry = self._providers.get(protocol_type)
        if entry is None:
            return None
        return entry[0]

    def is_enhanced(self, protocol_type: Type) -> bool:
        """Check if a capability has an enhanced (non-stub) provider.

        Args:
            protocol_type: The protocol type to check

        Returns:
            True if the registered provider is ENHANCED
        """
        self.ensure_bootstrapped()
        entry = self._providers.get(protocol_type)
        if entry is None:
            return False
        return entry[1] == CapabilityStatus.ENHANCED

    def get_status(self, protocol_type: Type) -> Optional[CapabilityStatus]:
        """Get the registration status for a capability.

        Returns:
            The CapabilityStatus, or None if not registered
        """
        entry = self._providers.get(protocol_type)
        if entry is None:
            return None
        return entry[1]

    def get_metadata(self, protocol_type: Type) -> Dict[str, Any]:
        """Get registration metadata for a capability provider."""
        return dict(self._metadata.get(protocol_type, {}))

    def list_capabilities(self) -> Dict[str, str]:
        """List all registered capabilities and their status.

        Returns:
            Dict mapping protocol name to status value
        """
        return {proto.__name__: status.value for proto, (_, status) in self._providers.items()}


# Module-level shortcut for convenient access
capabilities = OptionalFeatureRegistry.get_instance()


# ---------------------------------------------------------------------------
# Capability → method name mappings (consolidated from framework/capability_registry.py)
# ---------------------------------------------------------------------------

from typing import Dict  # noqa: E402  (re-import after class defs for clarity)

CAPABILITY_METHOD_MAPPINGS: Dict[str, str] = {
    # Tool capabilities
    "enabled_tools": "set_enabled_tools",
    "tool_dependencies": "set_tool_dependencies",
    "tool_sequences": "set_tool_sequences",
    "tiered_tool_config": "set_tiered_tool_config",
    # Vertical capabilities
    "vertical_middleware": "apply_vertical_middleware",
    "vertical_safety_patterns": "apply_vertical_safety_patterns",
    "vertical_context": "set_vertical_context",
    # RL capabilities
    "rl_hooks": "set_rl_hooks",
    # Team capabilities
    "team_specs": "set_team_specs",
    # Mode capabilities
    "mode_configs": "set_mode_configs",
    "default_budget": "set_default_budget",
    # Prompt capabilities
    "custom_prompt": "set_custom_prompt",
    "prompt_builder": "prompt_builder",
    "prompt_section": "add_prompt_section",
    "task_type_hints": "set_task_type_hints",
    # Safety capabilities
    "safety_patterns": "add_safety_patterns",
    # Enrichment capabilities
    "enrichment_strategy": "set_enrichment_strategy",
    "enrichment_service": "enrichment_service",
    # LSP capabilities
    "lsp": "set_lsp",
}


def get_method_for_capability(capability_name: str) -> str:
    """Get the method name for a capability.

    Args:
        capability_name: Name of the capability

    Returns:
        Method name to call for this capability
    """
    return CAPABILITY_METHOD_MAPPINGS.get(capability_name, f"set_{capability_name}")


# =============================================================================
# Backward compatibility aliases
# =============================================================================

# Alias for backward compatibility - was renamed for clarity
CapabilityRegistry = OptionalFeatureRegistry
