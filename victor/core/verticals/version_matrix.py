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

"""Version compatibility matrix for verticals and framework.

This module provides version compatibility checking using PEP 440 version
constraints to prevent runtime conflicts between verticals and the framework.

Design Principles:
    - PEP 440 compliant version checking
    - External JSON file for rules (no code changes needed)
    - Clear error messages for incompatibilities
    - Graceful degradation for unknown versions
    - Singleton pattern for global access
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version, InvalidVersion

logger = logging.getLogger(__name__)


class CompatibilityStatus(Enum):
    """Compatibility status between vertical and framework."""

    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class CompatibilityRule:
    """Version compatibility rule.

    Attributes:
        vertical_name: Name of the vertical package
        min_framework_version: Minimum framework version (PEP 440)
        max_framework_version: Maximum framework version (PEP 440, optional)
        excluded_versions: Set of specific versions to exclude
        requires_features: Required framework features
        status: Compatibility status if rule matches
        message: Human-readable explanation
    """

    vertical_name: str
    min_framework_version: str = ">=0.1.0"
    max_framework_version: Optional[str] = None
    excluded_versions: Set[str] = field(default_factory=set)
    requires_features: Set[str] = field(default_factory=set)
    status: CompatibilityStatus = CompatibilityStatus.COMPATIBLE
    message: str = ""

    def __post_init__(self) -> None:
        """Validate version constraints and coerce types."""
        # Coerce string status to enum (e.g., from JSON/dict data)
        if isinstance(self.status, str):
            object.__setattr__(self, "status", CompatibilityStatus(self.status))
        try:
            # Validate min version
            SpecifierSet(self.min_framework_version)
            if self.max_framework_version:
                SpecifierSet(self.max_framework_version)

            # Validate excluded versions
            for version_str in self.excluded_versions:
                Version(version_str)
        except (InvalidVersion, ValueError) as e:
            raise ValueError(f"Invalid version constraint in rule for '{self.vertical_name}': {e}")


@dataclass
class CompatibilityResult:
    """Result of compatibility check.

    Attributes:
        vertical_name: Name of the vertical package
        vertical_version: Version of the vertical
        framework_version: Version of the framework
        status: Compatibility status
        message: Human-readable explanation
        required_features: Missing required features (if any)
    """

    vertical_name: str
    vertical_version: str
    framework_version: str
    status: CompatibilityStatus
    message: str = ""
    required_features: Set[str] = field(default_factory=set)

    @property
    def is_compatible(self) -> bool:
        """Check if vertical is compatible."""
        return self.status in (
            CompatibilityStatus.COMPATIBLE,
            CompatibilityStatus.DEGRADED,
        )

    @property
    def is_incompatible(self) -> bool:
        """Check if vertical is incompatible."""
        return self.status == CompatibilityStatus.INCOMPATIBLE


class VersionCompatibilityMatrix:
    """Version compatibility matrix for verticals and framework.

    This singleton class manages compatibility rules and checks whether
    specific vertical versions are compatible with the framework version.

    Rules can be loaded from:
    - Programmatic registration via register_rule()
    - Built-in defaults for known verticals
    - External JSON files (optional, for testing)

    Example:
        >>> matrix = VersionCompatibilityMatrix.get_instance()
        >>> result = matrix.check_compatibility("coding", "2.0.0", "0.6.0")
        >>> if result.is_compatible:
        ...     print("Compatible!")
    """

    _instance: Optional["VersionCompatibilityMatrix"] = None
    _lock = threading.RLock()

    # Default built-in compatibility rules
    DEFAULT_RULES: List[Dict[str, Any]] = [
        {
            "vertical_name": "coding",
            "min_framework_version": ">=0.5.0",
            "status": "compatible",
            "message": "Victor Coding vertical is fully supported",
        },
        {
            "vertical_name": "devops",
            "min_framework_version": ">=0.5.0",
            "status": "compatible",
            "message": "Victor DevOps vertical is fully supported",
        },
        {
            "vertical_name": "research",
            "min_framework_version": ">=0.5.0",
            "status": "compatible",
            "message": "Victor Research vertical is fully supported",
        },
    ]

    def __init__(self) -> None:
        """Initialize the compatibility matrix."""
        self._rules: Dict[str, CompatibilityRule] = {}
        self._lock = threading.RLock()
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "VersionCompatibilityMatrix":
        """Get singleton matrix instance.

        Returns:
            VersionCompatibilityMatrix singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_default_rules(self) -> None:
        """Load built-in default compatibility rules."""
        with self._lock:
            for rule_data in self.DEFAULT_RULES:
                try:
                    rule = CompatibilityRule(**rule_data)
                    self._rules[rule.vertical_name] = rule
                except (TypeError, ValueError) as e:
                    logger.warning(
                        f"Failed to load default rule for '{rule_data.get('vertical_name')}': {e}"
                    )

            self._loaded = True
            logger.debug(f"Loaded {len(self._rules)} default compatibility rules")

    def load_from_file(self, file_path: str | Path) -> None:
        """Load compatibility rules from external JSON file.

        Args:
            file_path: Path to JSON file containing compatibility rules

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid or rules are malformed

        Example JSON format:
            {
                "rules": [
                    {
                        "vertical_name": "coding",
                        "min_framework_version": ">=0.5.0",
                        "max_framework_version": "<1.0.0",
                        "excluded_versions": ["0.5.1"],
                        "status": "compatible",
                        "message": "Fully supported"
                    }
                ]
            }
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Compatibility matrix file not found: {file_path}")

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            rules_data = data.get("rules", [])
            with self._lock:
                for rule_data in rules_data:
                    try:
                        # Convert excluded_versions from list to set
                        if "excluded_versions" in rule_data and isinstance(
                            rule_data["excluded_versions"], list
                        ):
                            rule_data["excluded_versions"] = set(rule_data["excluded_versions"])

                        # Convert string status to enum
                        if "status" in rule_data and isinstance(rule_data["status"], str):
                            rule_data["status"] = CompatibilityStatus(rule_data["status"])

                        # Convert requires_features from list to set
                        if "requires_features" in rule_data and isinstance(
                            rule_data["requires_features"], list
                        ):
                            rule_data["requires_features"] = set(rule_data["requires_features"])

                        rule = CompatibilityRule(**rule_data)
                        self._rules[rule.vertical_name] = rule
                    except (TypeError, ValueError) as e:
                        logger.warning(
                            f"Failed to load rule for '{rule_data.get('vertical_name')}': {e}"
                        )

                self._loaded = True
                logger.info(f"Loaded {len(rules_data)} compatibility rules from {file_path}")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in compatibility matrix file: {e}")

    def register_rule(self, rule: CompatibilityRule) -> None:
        """Register a compatibility rule programmatically.

        Args:
            rule: CompatibilityRule to register

        Example:
            >>> rule = CompatibilityRule(
            ...     vertical_name="my_vertical",
            ...     min_framework_version=">=0.6.0",
            ...     status=CompatibilityStatus.COMPATIBLE
            ... )
            >>> matrix.register_rule(rule)
        """
        with self._lock:
            self._rules[rule.vertical_name] = rule
            logger.debug(f"Registered compatibility rule for '{rule.vertical_name}'")

    def unregister_rule(self, vertical_name: str) -> None:
        """Remove a compatibility rule.

        Args:
            vertical_name: Name of vertical to remove rule for
        """
        with self._lock:
            if vertical_name in self._rules:
                del self._rules[vertical_name]
                logger.debug(f"Unregistered compatibility rule for '{vertical_name}'")

    def check_compatibility(
        self,
        vertical_name: str,
        vertical_version: str,
        framework_version: str,
        available_features: Optional[Set[str]] = None,
    ) -> CompatibilityResult:
        """Check compatibility between vertical and framework versions.

        Args:
            vertical_name: Name of the vertical package
            vertical_version: Version of the vertical (PEP 440)
            framework_version: Version of the framework (PEP 440)
            available_features: Set of features available in framework

        Returns:
            CompatibilityResult with status and message

        Example:
            >>> result = matrix.check_compatibility(
            ...     "coding", "2.0.0", "0.6.0", {"async_tools"}
            ... )
            >>> if result.is_compatible:
            ...     print("Can activate this vertical")
        """
        with self._lock:
            rule = self._rules.get(vertical_name)

            if not rule:
                # No rule found - return unknown
                return CompatibilityResult(
                    vertical_name=vertical_name,
                    vertical_version=vertical_version,
                    framework_version=framework_version,
                    status=CompatibilityStatus.UNKNOWN,
                    message=f"No compatibility rule found for '{vertical_name}'",
                )

            # Check version constraints
            try:
                fw_version = Version(framework_version)

                # Check minimum version
                min_spec = SpecifierSet(rule.min_framework_version)
                if fw_version not in min_spec:
                    return CompatibilityResult(
                        vertical_name=vertical_name,
                        vertical_version=vertical_version,
                        framework_version=framework_version,
                        status=CompatibilityStatus.INCOMPATIBLE,
                        message=f"'{vertical_name}' requires framework {rule.min_framework_version}, "
                        f"but {framework_version} is installed",
                    )

                # Check maximum version (if specified)
                if rule.max_framework_version:
                    max_spec = SpecifierSet(rule.max_framework_version)
                    if fw_version not in max_spec:
                        return CompatibilityResult(
                            vertical_name=vertical_name,
                            vertical_version=vertical_version,
                            framework_version=framework_version,
                            status=CompatibilityStatus.INCOMPATIBLE,
                            message=f"'{vertical_name}' requires framework {rule.max_framework_version}, "
                            f"but {framework_version} is installed",
                        )

                # Check excluded versions
                if framework_version in rule.excluded_versions:
                    return CompatibilityResult(
                        vertical_name=vertical_name,
                        vertical_version=vertical_version,
                        framework_version=framework_version,
                        status=CompatibilityStatus.INCOMPATIBLE,
                        message=f"Framework version {framework_version} is excluded for '{vertical_name}'",
                    )

            except (InvalidVersion, ValueError) as e:
                logger.warning(f"Invalid version format during compatibility check: {e}")
                return CompatibilityResult(
                    vertical_name=vertical_name,
                    vertical_version=vertical_version,
                    framework_version=framework_version,
                    status=CompatibilityStatus.UNKNOWN,
                    message=f"Invalid version format: {e}",
                )

            # Check required features
            missing_features: Set[str] = set()
            if rule.requires_features and available_features:
                missing_features = rule.requires_features - available_features

            status = rule.status
            message = rule.message

            # Downgrade to degraded if features are missing
            if missing_features and status == CompatibilityStatus.COMPATIBLE:
                status = CompatibilityStatus.DEGRADED
                message = f"{rule.message} (degraded: missing features {missing_features})"

            return CompatibilityResult(
                vertical_name=vertical_name,
                vertical_version=vertical_version,
                framework_version=framework_version,
                status=status,
                message=message,
                required_features=missing_features,
            )

    def get_rule(self, vertical_name: str) -> Optional[CompatibilityRule]:
        """Get compatibility rule for a vertical.

        Args:
            vertical_name: Name of the vertical

        Returns:
            CompatibilityRule if found, None otherwise
        """
        with self._lock:
            return self._rules.get(vertical_name)

    def list_rules(self) -> List[str]:
        """List all vertical names with registered rules.

        Returns:
            List of vertical names
        """
        with self._lock:
            return list(self._rules.keys())

    def is_loaded(self) -> bool:
        """Check if compatibility rules have been loaded.

        Returns:
            True if rules loaded, False otherwise
        """
        with self._lock:
            return self._loaded


# Convenience functions


def get_compatibility_matrix() -> VersionCompatibilityMatrix:
    """Get the singleton compatibility matrix instance.

    Returns:
        VersionCompatibilityMatrix instance
    """
    return VersionCompatibilityMatrix.get_instance()


def check_vertical_compatibility(
    vertical_name: str,
    vertical_version: str,
    framework_version: str,
    available_features: Optional[Set[str]] = None,
) -> CompatibilityResult:
    """Check vertical compatibility using singleton matrix.

    Convenience function for VersionCompatibilityMatrix.check_compatibility().

    Args:
        vertical_name: Name of the vertical package
        vertical_version: Version of the vertical (PEP 440)
        framework_version: Version of the framework (PEP 440)
        available_features: Set of features available in framework

    Returns:
        CompatibilityResult with status and message

    Example:
        >>> result = check_vertical_compatibility("coding", "2.0.0", "0.6.0")
        >>> if result.is_compatible:
        ...     print("Compatible!")
    """
    matrix = get_compatibility_matrix()
    return matrix.check_compatibility(
        vertical_name, vertical_version, framework_version, available_features
    )


__all__ = [
    "CompatibilityStatus",
    "CompatibilityRule",
    "CompatibilityResult",
    "VersionCompatibilityMatrix",
    "get_compatibility_matrix",
    "check_vertical_compatibility",
]
