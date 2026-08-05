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

"""
Protocol-based team specification providers.

This module defines protocols for team spec providers, allowing verticals
to enhance the framework with team specifications without creating direct
dependencies.

Architecture:
- Framework provides protocols + safe defaults
- Verticals implement protocols (optional)
- Entry points enable auto-discovery
- No direct imports from framework to verticals
"""

from __future__ import annotations

import logging
import warnings
from typing import Protocol, Dict, Any, runtime_checkable, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

TEAM_PROVIDER_GROUPS = (
    "victor.team_spec_providers",
    "victor.framework.teams.providers",
)
WORKFLOW_PROVIDER_GROUPS = (
    "victor.workflow_providers",
    "victor.framework.workflows.providers",
)
SAFETY_PROVIDER_GROUPS = ("victor.framework.safety.providers",)
_WARNED_LEGACY_PROVIDER_GROUPS: set[str] = set()


@runtime_checkable
class TeamSpecProviderProtocol(Protocol):
    """Protocol for providing team specifications.

    Verticals can implement this protocol to provide team specs.
    The framework provides a safe default that returns empty specs.

    Example:
        class CodingTeamSpecProvider:
            def get_team_specs(self) -> Dict[str, Any]:
                return {
                    "code_review": {
                        "name": "Code Review Team",
                        "members": [...]
                    }
                }
    """

    def get_team_specs(self) -> Dict[str, Any]:
        """Return team specifications from this provider.

        Returns:
            Dictionary mapping team names to team spec dicts
        """
        ...


@runtime_checkable
class WorkflowProviderProtocol(Protocol):
    """Protocol for providing workflow specifications."""

    def get_workflows(self) -> Dict[str, Any]:
        """Return workflow specifications from this provider.

        Returns:
            Dictionary mapping workflow names to workflow spec dicts
        """
        ...


@runtime_checkable
class SafetyRulesProviderProtocol(Protocol):
    """Protocol for providing safety rules."""

    def get_safety_rules(self) -> Dict[str, Any]:
        """Return safety rules from this provider.

        Returns:
            Dictionary mapping rule names to rule configurations
        """
        ...


# =============================================================================
# Safe Default Implementations (contrib layer)
# =============================================================================


class DefaultTeamSpecProvider:
    """Safe default implementation - provides no teams.

    This ensures the framework works without any verticals installed.
    """

    def get_team_specs(self) -> Dict[str, Any]:
        return {}


class DefaultWorkflowProvider:
    """Safe default implementation - provides no workflows."""

    def get_workflows(self) -> Dict[str, Any]:
        return {}


class DefaultSafetyRulesProvider:
    """Safe default implementation - provides no safety rules."""

    def get_safety_rules(self) -> Dict[str, Any]:
        return {}


# =============================================================================
# Provider Registry with Entry Points Discovery
# =============================================================================


class ProviderRegistry:
    """
    Registry for protocol-based providers with entry points discovery.

    This registry:
    1. Starts with safe defaults (framework works standalone)
    2. Loads enhanced providers from installed packages via entry points
    3. Allows manual registration for testing
    4. Provides combined results from all providers

    Entry Points:
        Packages can register providers via entry points. Canonical groups:
        [project.entry-points."victor.team_spec_providers"]
        coding = "victor_coding.teams:CodingTeamSpecProvider"

        Legacy compatibility groups are still accepted as fallbacks:
        [project.entry-points."victor.framework.teams.providers"]
        coding = "victor_coding.teams:CodingTeamSpecProvider"

    Usage:
        registry = ProviderRegistry()
        all_team_specs = registry.get_all_team_specs()
    """

    def __init__(self):
        self._team_providers: List[TeamSpecProviderProtocol] = [DefaultTeamSpecProvider()]
        self._workflow_providers: List[WorkflowProviderProtocol] = [DefaultWorkflowProvider()]
        self._safety_providers: List[SafetyRulesProviderProtocol] = [DefaultSafetyRulesProvider()]

        # Track vertical names for namespacing
        self._team_provider_names: Dict[str, str] = {}  # provider -> vertical_name
        self._workflow_provider_names: Dict[str, str] = {}
        self._safety_provider_names: Dict[str, str] = {}

        # Load from entry points (optional verticals)
        self._load_from_entry_points()

    def _load_from_entry_points(self):
        """Load providers from installed packages via entry points."""
        try:
            import importlib.metadata

            # Load team spec providers
            try:
                for entry_point in self._iter_provider_entry_points(
                    importlib.metadata,
                    TEAM_PROVIDER_GROUPS,
                ):
                    self._load_provider_entry_point(
                        entry_point,
                        TeamSpecProviderProtocol,
                        self._team_providers,
                        self._team_provider_names,
                        "team provider",
                    )
            except Exception as e:
                logger.debug(f"No team provider entry points found: {e}")

            # Load workflow providers
            try:
                for entry_point in self._iter_provider_entry_points(
                    importlib.metadata,
                    WORKFLOW_PROVIDER_GROUPS,
                ):
                    self._load_provider_entry_point(
                        entry_point,
                        WorkflowProviderProtocol,
                        self._workflow_providers,
                        self._workflow_provider_names,
                        "workflow provider",
                    )
            except Exception as e:
                logger.debug(f"No workflow provider entry points found: {e}")

            # Load safety rules providers
            try:
                for entry_point in self._iter_provider_entry_points(
                    importlib.metadata,
                    SAFETY_PROVIDER_GROUPS,
                ):
                    self._load_provider_entry_point(
                        entry_point,
                        SafetyRulesProviderProtocol,
                        self._safety_providers,
                        self._safety_provider_names,
                        "safety provider",
                    )
            except Exception as e:
                logger.debug(f"No safety provider entry points found: {e}")

        except Exception as e:
            logger.debug(f"Entry points discovery failed: {e}")

    @staticmethod
    def _iter_provider_entry_points(metadata_module: Any, groups: tuple[str, ...]) -> list[Any]:
        """Return entry points from canonical groups with legacy fallback de-duplication."""

        discovered: list[Any] = []
        seen_names: set[str] = set()
        canonical_group = groups[0] if groups else None

        for index, group in enumerate(groups):
            group_entry_points = list(metadata_module.entry_points(group=group))
            if index > 0 and group_entry_points and canonical_group is not None:
                ProviderRegistry._warn_legacy_provider_group_usage(group, canonical_group)
            for entry_point in group_entry_points:
                if entry_point.name in seen_names:
                    continue
                seen_names.add(entry_point.name)
                discovered.append(entry_point)

        return discovered

    @staticmethod
    def _warn_legacy_provider_group_usage(group: str, canonical_group: str) -> None:
        """Emit a one-time warning when legacy provider groups are still published."""

        if group in _WARNED_LEGACY_PROVIDER_GROUPS:
            return

        message = (
            f"Legacy provider entry-point group '{group}' is deprecated. "
            f"Publish providers via '{canonical_group}' instead."
        )
        logger.warning(message)
        _WARNED_LEGACY_PROVIDER_GROUPS.add(group)

    @staticmethod
    def _instantiate_provider(entry_point: Any) -> Any:
        """Instantiate a provider entry point when it resolves to a class."""

        provider_class_or_instance = entry_point.load()
        if isinstance(provider_class_or_instance, type):
            return provider_class_or_instance()
        return provider_class_or_instance

    def _load_provider_entry_point(
        self,
        entry_point: Any,
        protocol: type[Any],
        providers: list[Any],
        provider_names: dict[str, str],
        label: str,
    ) -> None:
        """Load and register a single provider entry point with failure isolation."""

        try:
            provider = self._instantiate_provider(entry_point)
            if isinstance(provider, protocol):
                providers.append(provider)
                provider_names[id(provider)] = entry_point.name
                logger.info("Loaded %s: %s", label, entry_point.name)
        except Exception as e:
            logger.warning("Failed to load %s %s: %s", label, entry_point.name, e)

    def register_team_provider(self, provider: TeamSpecProviderProtocol):
        """Directly register a team spec provider (for testing or manual setup)."""
        self._team_providers.append(provider)

    def register_workflow_provider(self, provider: WorkflowProviderProtocol):
        """Directly register a workflow provider (for testing or manual setup)."""
        self._workflow_providers.append(provider)

    def register_safety_provider(self, provider: SafetyRulesProviderProtocol):
        """Directly register a safety rules provider (for testing or manual setup)."""
        self._safety_providers.append(provider)

    def get_all_team_specs(self) -> Dict[str, Any]:
        """Collect all team specs from registered providers.

        Returns:
            Dictionary mapping vertical names to team spec dicts.
            Example: {"coding": {"feature_team": TeamSpec(), ...}, ...}
        """
        all_specs: Dict[str, Any] = {}
        for provider in self._team_providers:
            try:
                specs = provider.get_team_specs()
                if specs:
                    # Get vertical name from entry point name
                    vertical_name = self._team_provider_names.get(id(provider))
                    if vertical_name:
                        # Namespace by vertical: {vertical: {team_name: TeamSpec}}
                        all_specs[vertical_name] = specs
                    else:
                        # Default provider (no vertical namespace)
                        all_specs.update(specs)
            except Exception as e:
                logger.warning(f"Failed to get team specs from {provider}: {e}")
        return all_specs

    def get_all_workflows(self) -> Dict[str, Any]:
        """Collect all workflows from registered providers.

        Returns:
            Dictionary mapping vertical names to workflow spec dicts.
        """
        all_specs: Dict[str, Any] = {}
        for provider in self._workflow_providers:
            try:
                specs = provider.get_workflows()
                if specs:
                    vertical_name = self._workflow_provider_names.get(id(provider))
                    if vertical_name:
                        all_specs[vertical_name] = specs
                    else:
                        all_specs.update(specs)
            except Exception as e:
                logger.warning(f"Failed to get workflows from {provider}: {e}")
        return all_specs

    def get_all_safety_rules(self) -> Dict[str, Any]:
        """Collect all safety rules from registered providers.

        Returns:
            Dictionary mapping vertical names to safety rule dicts.
        """
        all_specs: Dict[str, Any] = {}
        for provider in self._safety_providers:
            try:
                specs = provider.get_safety_rules()
                if specs:
                    vertical_name = self._safety_provider_names.get(id(provider))
                    if vertical_name:
                        all_specs[vertical_name] = specs
                    else:
                        all_specs.update(specs)
            except Exception as e:
                logger.warning(f"Failed to get safety rules from {provider}: {e}")
        return all_specs


# Global registry instance
_provider_registry: ProviderRegistry | None = None


def get_provider_registry() -> ProviderRegistry:
    """Get the global provider registry instance."""
    global _provider_registry
    if _provider_registry is None:
        _provider_registry = ProviderRegistry()
    return _provider_registry


def reset_provider_registry():
    """Reset the global provider registry (for testing)."""
    global _provider_registry
    _provider_registry = None


__all__ = [
    "TeamSpecProviderProtocol",
    "WorkflowProviderProtocol",
    "SafetyRulesProviderProtocol",
    "DefaultTeamSpecProvider",
    "DefaultWorkflowProvider",
    "DefaultSafetyRulesProvider",
    "ProviderRegistry",
    "get_provider_registry",
    "reset_provider_registry",
]
