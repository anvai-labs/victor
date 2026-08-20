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

"""Capability negotiation system for Victor framework.

This module implements a negotiation protocol where verticals and the orchestrator
can agree on compatible versions of capabilities, enabling graceful degradation
and fallback behavior when version mismatches occur.

Architecture:
    Vertical                     Negotiator                 Orchestrator
    ┌─────────────┐              ┌──────────┐              ┌─────────────┐
    │ Capabilities│─────────────>│          │<─────────────│ Capabilities│
    │ v2.0        │  Declare    │Negotiate │  Declare     │ v1.0        │
    │             │<────────────>│          │────────────>│             │
    └─────────────┘   Agree     │          │   Agree      └─────────────┘
                                └──────────┘

Benefits:
    - Version compatibility: Agree on compatible versions
    - Graceful degradation: Fallback to older versions when needed
    - Feature detection: Know which features are available
    - Future-proofing: Support new capabilities without breaking old verticals
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Version Types
# =============================================================================


class CompatibilityStrategy(str, Enum):
    """Strategy for handling version compatibility."""

    STRICT = "strict"
    """Require exact version match."""

    BACKWARD_COMPATIBLE = "backward_compatible"
    """Allow newer versions (vertical v2.0 works with orchestrator v1.0)."""

    MINIMUM_VERSION = "minimum_version"
    """Require minimum version (orchestrator v1.0 requires vertical v1.5+)."""

    BEST_EFFORT = "best_effort"
    """Use highest compatible version available."""


@dataclass(frozen=True)
class Version:
    """Semantic version.

    Attributes:
        major: Major version (breaking changes)
        minor: Minor version (backward-compatible additions)
        patch: Patch version (bug fixes)
    """

    major: int
    minor: int = 0
    patch: int = 0

    def __post_init__(self):
        """Validate version components."""
        if self.major < 0 or self.minor < 0 or self.patch < 0:
            raise ValueError("Version components must be non-negative")

    @classmethod
    def parse(cls, version_str: str) -> "Version":
        """Parse version string.

        Args:
            version_str: Version string (e.g., "1.0.0", "2.1", "3")

        Returns:
            Version instance
        """
        parts = version_str.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        return cls(major, minor, patch)

    def __str__(self) -> str:
        """Convert to string."""
        if self.patch > 0:
            return f"{self.major}.{self.minor}.{self.patch}"
        elif self.minor > 0:
            return f"{self.major}.{self.minor}"
        else:
            return f"{self.major}"

    def __lt__(self, other: "Version") -> bool:
        """Compare versions."""
        return (self.major, self.minor, self.patch) < (
            other.major,
            other.minor,
            other.patch,
        )

    def __le__(self, other: "Version") -> bool:
        """Compare versions."""
        return (self.major, self.minor, self.patch) <= (
            other.major,
            other.minor,
            other.patch,
        )

    def __gt__(self, other: "Version") -> bool:
        """Compare versions."""
        return (self.major, self.minor, self.patch) > (
            other.major,
            other.minor,
            other.patch,
        )

    def __ge__(self, other: "Version") -> bool:
        """Compare versions."""
        return (self.major, self.minor, self.patch) >= (
            other.major,
            other.minor,
            other.patch,
        )

    def __eq__(self, other: object) -> bool:
        """Compare versions."""
        if not isinstance(other, Version):
            return NotImplemented
        return (self.major, self.minor, self.patch) == (
            other.major,
            other.minor,
            other.patch,
        )

    def is_compatible_with(
        self,
        other: "Version",
        strategy: CompatibilityStrategy = CompatibilityStrategy.BACKWARD_COMPATIBLE,
    ) -> bool:
        """Check if this version is compatible with another.

        Args:
            other: Version to check compatibility with
            strategy: Compatibility strategy

        Returns:
            True if versions are compatible
        """
        if strategy == CompatibilityStrategy.STRICT:
            return self == other

        elif strategy == CompatibilityStrategy.BACKWARD_COMPATIBLE:
            # Same major version, self >= other
            return self.major == other.major and self >= other

        elif strategy == CompatibilityStrategy.MINIMUM_VERSION:
            # self must be >= other
            return self >= other

        elif strategy == CompatibilityStrategy.BEST_EFFORT:
            # Same major version
            return self.major == other.major

        else:
            return False


# =============================================================================
# Capability Types
# =============================================================================


@dataclass
class CapabilityFeature:
    """A feature within a capability.

    Attributes:
        name: Feature name
        version: Feature version (if versioned independently)
        description: Feature description
        required: Whether this feature is required
    """

    name: str
    version: Optional[Version] = None
    description: str = ""
    required: bool = False


@dataclass
class CapabilityDeclaration:
    """Declaration of a capability with version and features.

    Attributes:
        name: Capability name
        version: Capability version
        min_version: Minimum compatible version (for negotiation)
        features: List of features provided by this capability
        strategy: Compatibility strategy
        description: Capability description
    """

    name: str
    version: Version
    min_version: Optional[Version] = None
    features: List[CapabilityFeature] = field(default_factory=list)
    strategy: CompatibilityStrategy = CompatibilityStrategy.BACKWARD_COMPATIBLE
    description: str = ""

    def __post_init__(self):
        """Set default min_version."""
        if self.min_version is None:
            # Default min_version is same major version, 0.0
            object.__setattr__(self, "min_version", Version(self.version.major, 0, 0))

    def get_feature(self, name: str) -> Optional[CapabilityFeature]:
        """Get feature by name.

        Args:
            name: Feature name

        Returns:
            Feature or None if not found
        """
        for feature in self.features:
            if feature.name == name:
                return feature
        return None

    def has_feature(self, name: str) -> bool:
        """Check if capability has feature.

        Args:
            name: Feature name

        Returns:
            True if feature exists
        """
        return self.get_feature(name) is not None

    def get_required_features(self) -> List[CapabilityFeature]:
        """Get all required features.

        Returns:
            List of required features
        """
        return [f for f in self.features if f.required]

    def get_optional_features(self) -> List[CapabilityFeature]:
        """Get all optional features.

        Returns:
            List of optional features
        """
        return [f for f in self.features if not f.required]

    def is_compatible_with(
        self,
        version: Version,
        strategy: Optional[CompatibilityStrategy] = None,
    ) -> bool:
        """Check if this capability is compatible with given version.

        Args:
            version: Version to check
            strategy: Compatibility strategy (uses default if None)

        Returns:
            True if compatible
        """
        if strategy is None:
            strategy = self.strategy

        return self.version.is_compatible_with(version, strategy)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "version": str(self.version),
            "min_version": str(self.min_version) if self.min_version else None,
            "features": [
                {
                    "name": f.name,
                    "version": str(f.version) if f.version else None,
                    "description": f.description,
                    "required": f.required,
                }
                for f in self.features
            ],
            "strategy": self.strategy.value,
            "description": self.description,
        }


# =============================================================================
# Negotiation Types
# =============================================================================


class NegotiationStatus(str, Enum):
    """Status of capability negotiation."""

    SUCCESS = "success"
    """Negotiation succeeded, compatible version agreed upon."""

    FALLBACK = "fallback"
    """Negotiation succeeded using fallback version."""

    FAILURE = "failure"
    """Negotiation failed, no compatible version found."""


@dataclass
class NegotiationResult:
    """Result of capability negotiation.

    Attributes:
        capability_name: Name of the capability
        status: Negotiation status
        agreed_version: Version agreed upon (if successful)
        supported_features: Features that are supported
        unsupported_features: Features that are not supported
        missing_required_features: Required features that are missing
        fallback_version: Fallback version (if applicable)
        error: Error message (if failed)
    """

    capability_name: str
    status: NegotiationStatus
    agreed_version: Optional[Version] = None
    supported_features: List[str] = field(default_factory=list)
    unsupported_features: List[str] = field(default_factory=list)
    missing_required_features: List[str] = field(default_factory=list)
    fallback_version: Optional[Version] = None
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """Check if negotiation was successful."""
        return self.status in (NegotiationStatus.SUCCESS, NegotiationStatus.FALLBACK)

    @property
    def has_fallback(self) -> bool:
        """Check if fallback was used."""
        return self.status == NegotiationStatus.FALLBACK

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "capability_name": self.capability_name,
            "status": self.status.value,
            "agreed_version": str(self.agreed_version) if self.agreed_version else None,
            "supported_features": self.supported_features,
            "unsupported_features": self.unsupported_features,
            "missing_required_features": self.missing_required_features,
            "fallback_version": (str(self.fallback_version) if self.fallback_version else None),
            "error": self.error,
        }


# =============================================================================
# Negotiation Engine
# =============================================================================


class CapabilityNegotiator:
    """Negotiates capability versions between vertical and orchestrator.

    The negotiator takes capability declarations from both sides and
    determines the best compatible version with feature compatibility
    analysis.

    Usage:
        negotiator = CapabilityNegotiator()

        # Vertical declares capabilities
        vertical_caps = {
            "tools": CapabilityDeclaration(
                name="tools",
                version=Version(2, 0, 0),
                features=[
                    CapabilityFeature("tool_filtering", required=True),
                    CapabilityFeature("tool_dependencies"),
                ]
            )
        }

        # Orchestrator declares capabilities
        orchestrator_caps = {
            "tools": CapabilityDeclaration(
                name="tools",
                version=Version(1, 0, 0),
                min_version=Version(1, 0, 0),
            )
        }

        # Negotiate
        results = negotiator.negotiate(
            vertical_caps,
            orchestrator_caps,
        )

        for result in results.values():
            if result.is_success:
                print(f"{result.capability_name}: v{result.agreed_version}")
    """

    def __init__(
        self,
        strategy: CompatibilityStrategy = CompatibilityStrategy.BACKWARD_COMPATIBLE,
        enable_fallback: bool = True,
    ):
        """Initialize negotiator.

        Args:
            strategy: Default compatibility strategy
            enable_fallback: Enable fallback to older versions
        """
        self._strategy = strategy
        self._enable_fallback = enable_fallback

    def negotiate(
        self,
        vertical_capabilities: Dict[str, CapabilityDeclaration],
        orchestrator_capabilities: Dict[str, CapabilityDeclaration],
    ) -> Dict[str, NegotiationResult]:
        """Negotiate all capabilities.

        Args:
            vertical_capabilities: Capabilities declared by vertical
            orchestrator_capabilities: Capabilities declared by orchestrator

        Returns:
            Dictionary mapping capability names to negotiation results
        """
        results = {}

        # Get all capability names
        all_names = set(vertical_capabilities.keys()) | set(orchestrator_capabilities.keys())

        for name in all_names:
            vertical_cap = vertical_capabilities.get(name)
            orchestrator_cap = orchestrator_capabilities.get(name)

            result = self._negotiate_capability(
                name,
                vertical_cap,
                orchestrator_cap,
            )
            results[name] = result

        return results

    def _negotiate_capability(
        self,
        name: str,
        vertical_cap: Optional[CapabilityDeclaration],
        orchestrator_cap: Optional[CapabilityDeclaration],
    ) -> NegotiationResult:
        """Negotiate a single capability.

        Args:
            name: Capability name
            vertical_cap: Vertical's declaration (may be None)
            orchestrator_cap: Orchestrator's declaration (may be None)

        Returns:
            Negotiation result
        """
        # Both sides must declare the capability
        if vertical_cap is None or orchestrator_cap is None:
            return NegotiationResult(
                capability_name=name,
                status=NegotiationStatus.FAILURE,
                error=f"Capability '{name}' not declared by both sides",
            )

        # Check version compatibility
        vertical_version = vertical_cap.version
        orchestrator_version = orchestrator_cap.version

        # Try to find compatible version
        strategy = vertical_cap.strategy or self._strategy
        agreed_version, status = self._find_compatible_version(
            vertical_cap,
            orchestrator_cap,
            strategy,
        )

        if agreed_version is None:
            return NegotiationResult(
                capability_name=name,
                status=NegotiationStatus.FAILURE,
                error=f"No compatible version between v{vertical_version} and v{orchestrator_version}",
            )

        # Analyze feature compatibility
        supported_features, unsupported_features, missing_required = self._analyze_features(
            vertical_cap,
            orchestrator_cap,
        )

        # Check if required features are missing
        if missing_required and not self._enable_fallback:
            return NegotiationResult(
                capability_name=name,
                status=NegotiationStatus.FAILURE,
                error=f"Missing required features: {missing_required}",
                agreed_version=agreed_version,
                supported_features=supported_features,
                unsupported_features=unsupported_features,
                missing_required_features=missing_required,
            )

        # Determine fallback
        fallback_version = None
        if status == NegotiationStatus.FALLBACK or (missing_required and self._enable_fallback):
            fallback_version = self._find_fallback_version(vertical_cap, orchestrator_cap)

        return NegotiationResult(
            capability_name=name,
            status=status,
            agreed_version=agreed_version,
            supported_features=supported_features,
            unsupported_features=unsupported_features,
            missing_required_features=missing_required,
            fallback_version=fallback_version,
        )

    def _find_compatible_version(
        self,
        vertical_cap: CapabilityDeclaration,
        orchestrator_cap: CapabilityDeclaration,
        strategy: CompatibilityStrategy,
    ) -> Tuple[Optional[Version], NegotiationStatus]:
        """Find compatible version between two declarations.

        Args:
            vertical_cap: Vertical's declaration
            orchestrator_cap: Orchestrator's declaration
            strategy: Compatibility strategy

        Returns:
            Tuple of (agreed_version, status)
        """
        vertical_version = vertical_cap.version
        orchestrator_version = orchestrator_cap.version

        # STRICT strategy: require exact version match
        if strategy == CompatibilityStrategy.STRICT:
            if vertical_version == orchestrator_version:
                return vertical_version, NegotiationStatus.SUCCESS
            return None, NegotiationStatus.FAILURE

        # For backward compatibility, prefer the higher version
        # Check if versions are in the same major version
        if vertical_version.major == orchestrator_version.major:
            # Same major version - use higher version for backward compatibility
            higher_version = max(vertical_version, orchestrator_version)
            return higher_version, NegotiationStatus.SUCCESS

        # Different major versions - try fallback if enabled
        if self._enable_fallback:
            # Check if vertical supports the orchestrator's version range
            if vertical_cap.min_version and orchestrator_version >= vertical_cap.min_version:
                return orchestrator_version, NegotiationStatus.FALLBACK

            # Check if orchestrator supports the vertical's version range
            if orchestrator_cap.min_version and vertical_version >= orchestrator_cap.min_version:
                return vertical_version, NegotiationStatus.FALLBACK

        # No compatible version found
        return None, NegotiationStatus.FAILURE

    def _analyze_features(
        self,
        vertical_cap: CapabilityDeclaration,
        orchestrator_cap: CapabilityDeclaration,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Analyze feature compatibility.

        Args:
            vertical_cap: Vertical's declaration
            orchestrator_cap: Orchestrator's declaration

        Returns:
            Tuple of (supported, unsupported, missing_required)
        """
        vertical_features = {f.name: f for f in vertical_cap.features}
        orchestrator_features = {f.name: f for f in orchestrator_cap.features}

        all_feature_names = set(vertical_features.keys()) | set(orchestrator_features.keys())

        supported = []
        unsupported = []
        missing_required = []

        for feature_name in all_feature_names:
            vertical_feature = vertical_features.get(feature_name)
            orchestrator_feature = orchestrator_features.get(feature_name)

            # Feature exists on both sides
            if vertical_feature and orchestrator_feature:
                supported.append(feature_name)
            # Feature only on vertical
            elif vertical_feature:
                if vertical_feature.required:
                    missing_required.append(feature_name)
                else:
                    unsupported.append(feature_name)
            # Feature only on orchestrator
            elif orchestrator_feature:
                if orchestrator_feature.required:
                    missing_required.append(feature_name)
                else:
                    unsupported.append(feature_name)

        return supported, unsupported, missing_required

    def _find_fallback_version(
        self,
        vertical_cap: CapabilityDeclaration,
        orchestrator_cap: CapabilityDeclaration,
    ) -> Optional[Version]:
        """Find fallback version.

        Args:
            vertical_cap: Vertical's declaration
            orchestrator_cap: Orchestrator's declaration

        Returns:
            Fallback version or None
        """
        # Use min_version as fallback
        min_versions = []
        if vertical_cap.min_version:
            min_versions.append(vertical_cap.min_version)
        if orchestrator_cap.min_version:
            min_versions.append(orchestrator_cap.min_version)

        if min_versions:
            return max(min_versions)

        # Use lower of the two versions
        return min(vertical_cap.version, orchestrator_cap.version)


# =============================================================================
# Integration with Existing Code
# =============================================================================


class CapabilityNegotiationProtocol:
    """Protocol for capability negotiation integration.

    This protocol defines how capability negotiation integrates with
    the existing Victor framework.
    """

    @staticmethod
    def negotiate_vertical_integration(
        vertical,
        orchestrator,
        negotiator: Optional[CapabilityNegotiator] = None,
    ) -> Dict[str, NegotiationResult]:
        """Negotiate capabilities for vertical integration.

        Args:
            vertical: Vertical class or instance
            orchestrator: Orchestrator instance
            negotiator: Optional custom negotiator

        Returns:
            Dictionary of negotiation results
        """
        if negotiator is None:
            negotiator = CapabilityNegotiator()

        # Get capabilities from vertical
        vertical_caps = CapabilityNegotiationProtocol._get_vertical_capabilities(vertical)

        # Get capabilities from orchestrator
        orchestrator_caps = CapabilityNegotiationProtocol._get_orchestrator_capabilities(
            orchestrator
        )

        # Negotiate
        return negotiator.negotiate(vertical_caps, orchestrator_caps)

    @staticmethod
    def _get_vertical_capabilities(vertical) -> Dict[str, CapabilityDeclaration]:
        """Extract capabilities from vertical.

        Args:
            vertical: Vertical class or instance

        Returns:
            Dictionary of capability declarations
        """
        capabilities = {}

        # Get version from vertical
        vertical_version = Version.parse(
            getattr(vertical, "version", getattr(vertical, "get_version", lambda: "1.0.0")())
        )

        # Tool capability
        if hasattr(vertical, "get_tools"):
            _ = vertical.get_tools() if callable(vertical.get_tools) else vertical.get_tools
            capabilities["tools"] = CapabilityDeclaration(
                name="tools",
                version=vertical_version,
                features=[
                    CapabilityFeature("tool_list", required=True),
                    CapabilityFeature("tool_filtering"),
                ],
            )

        # Prompt capability
        if hasattr(vertical, "get_system_prompt"):
            capabilities["prompt"] = CapabilityDeclaration(
                name="prompt",
                version=vertical_version,
                features=[
                    CapabilityFeature("system_prompt", required=True),
                    CapabilityFeature("prompt_templates"),
                ],
            )

        # Middleware capability
        if hasattr(vertical, "get_middleware"):
            capabilities["middleware"] = CapabilityDeclaration(
                name="middleware",
                version=vertical_version,
                features=[
                    CapabilityFeature("middleware_chain"),
                ],
            )

        # Safety capability
        if hasattr(vertical, "get_safety_extension"):
            capabilities["safety"] = CapabilityDeclaration(
                name="safety",
                version=vertical_version,
                features=[
                    CapabilityFeature("safety_patterns"),
                ],
            )

        # Workflow capability
        if hasattr(vertical, "get_workflows"):
            capabilities["workflows"] = CapabilityDeclaration(
                name="workflows",
                version=vertical_version,
                features=[
                    CapabilityFeature("workflow_definitions"),
                ],
            )

        # Team capability
        if hasattr(vertical, "get_team_specifications"):
            capabilities["teams"] = CapabilityDeclaration(
                name="teams",
                version=vertical_version,
                features=[
                    CapabilityFeature("team_specifications"),
                ],
            )

        return capabilities

    @staticmethod
    def _get_orchestrator_capabilities(
        orchestrator,
    ) -> Dict[str, CapabilityDeclaration]:
        """Extract capabilities from orchestrator.

        Args:
            orchestrator: Orchestrator instance

        Returns:
            Dictionary of capability declarations
        """
        capabilities = {}

        # Get orchestrator version
        orchestrator_version = Version.parse(getattr(orchestrator, "_version", "1.0.0"))

        # Tool capability
        if hasattr(orchestrator, "get_enabled_tools"):
            capabilities["tools"] = CapabilityDeclaration(
                name="tools",
                version=orchestrator_version,
                min_version=Version(orchestrator_version.major, 0, 0),
                features=[
                    CapabilityFeature("tool_list", required=True),
                    CapabilityFeature("tool_filtering"),
                ],
            )

        # Prompt capability
        if hasattr(orchestrator, "get_system_prompt"):
            capabilities["prompt"] = CapabilityDeclaration(
                name="prompt",
                version=orchestrator_version,
                min_version=Version(orchestrator_version.major, 0, 0),
                features=[
                    CapabilityFeature("system_prompt", required=True),
                    CapabilityFeature("prompt_templates"),
                ],
            )

        return capabilities


# =============================================================================
# Public API
# =============================================================================


def negotiate_capabilities(
    vertical,
    orchestrator,
    strategy: CompatibilityStrategy = CompatibilityStrategy.BACKWARD_COMPATIBLE,
    enable_fallback: bool = True,
) -> Dict[str, NegotiationResult]:
    """Negotiate capabilities between vertical and orchestrator.

    This is the main entry point for capability negotiation in the
    Victor framework.

    Args:
        vertical: Vertical class or instance
        orchestrator: Orchestrator instance
        strategy: Compatibility strategy
        enable_fallback: Enable fallback to older versions

    Returns:
        Dictionary of negotiation results

    Example:
        from victor.framework.capability_negotiation import negotiate_capabilities

        results = negotiate_capabilities(
            vertical=CodingAssistant,
            orchestrator=orchestrator,
        )

        for name, result in results.items():
            if result.is_success:
                print(f"{name}: v{result.agreed_version} ✓")
                if result.has_fallback:
                    print(f"  (fallback from v{result.fallback_version})")
            else:
                print(f"{name}: FAILED - {result.error}")
    """
    negotiator = CapabilityNegotiator(
        strategy=strategy,
        enable_fallback=enable_fallback,
    )

    return CapabilityNegotiationProtocol.negotiate_vertical_integration(
        vertical,
        orchestrator,
        negotiator,
    )


__all__ = [
    # Version types
    "Version",
    "CompatibilityStrategy",
    # Capability types
    "CapabilityFeature",
    "CapabilityDeclaration",
    # Negotiation types
    "NegotiationStatus",
    "NegotiationResult",
    # Negotiation engine
    "CapabilityNegotiator",
    "CapabilityNegotiationProtocol",
    # Public API
    "negotiate_capabilities",
]
