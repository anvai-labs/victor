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

"""Framework-level persona provider with versioning.

This module provides a central registry for agent personas that can be
shared across all verticals (Coding, DevOps, RAG, Research, DataAnalysis).

Design Pattern: Registry + Versioning
- Personas are registered with semantic versioning (SemVer)
- Personas can be retrieved by name and optional version
- Personas are categorized for discovery (research, planning, execution, review)

Example:
    from victor.framework.multi_agent.persona_provider import PersonaProvider

    # Register a persona
    PersonaProvider.register_persona(
        name="senior_developer",
        version="1.0.0",
        persona=CodingPersonaTraits(...),
        category="execution",
        description="Senior software engineer persona"
    )

    # Retrieve a persona
    persona = PersonaProvider.get_persona("senior_developer")
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from victor.framework.capabilities.base import (
    BaseCapabilityProvider,
    CapabilityMetadata,
)
from victor_contracts.registries import set_default_persona_registry

if TYPE_CHECKING:
    from victor.framework.multi_agent.personas import PersonaTraits

logger = logging.getLogger(__name__)


# =============================================================================
# Persona Metadata
# =============================================================================


@dataclass
class PersonaMetadata:
    """Metadata for a registered persona.

    Attributes:
        name: Unique persona name
        version: Semantic version string (e.g., "1.0.0")
        description: Human-readable description
        category: Persona category (research, planning, execution, review)
        tags: List of tags for discovery
        author: Optional author name
        vertical: Vertical this persona belongs to (coding, devops, etc.)
        deprecated: Whether this persona is deprecated
    """

    name: str
    version: str
    description: str
    category: str
    tags: List[str] = field(default_factory=list)
    author: Optional[str] = None
    vertical: Optional[str] = None
    deprecated: bool = False


# =============================================================================
# Persona Provider
# =============================================================================


class FrameworkPersonaProvider(BaseCapabilityProvider["PersonaTraits"]):
    """Framework-level persona provider with versioning.

    This provider offers a central registry for agent personas across all
    verticals, enabling cross-vertical persona discovery and reuse.

    Pattern: Singleton + Registry + CapabilityProvider
    - Thread-safe singleton implementation
    - Semantic versioning for compatibility tracking
    - Category-based discovery
    - Integrates with BaseCapabilityProvider for consistency

    Categories:
        - research: Information gathering personas
        - planning: Architecture and design personas
        - execution: Implementation and coding personas
        - review: Code review and testing personas
    """

    _instance: Optional["FrameworkPersonaProvider"] = None
    _lock: threading.RLock = threading.RLock()  # RLock allows reentrant acquisition

    def __new__(cls) -> "FrameworkPersonaProvider":
        """Create or return singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the persona provider."""
        if self._initialized:
            return

        super().__init__()
        self._personas: Dict[str, Dict[str, "PersonaTraits"]] = {}
        self._metadata: Dict[str, Dict[str, PersonaMetadata]] = {}
        self._categories: Dict[str, Set[str]] = {
            "research": set(),
            "planning": set(),
            "execution": set(),
            "review": set(),
            "other": set(),
        }
        # Initialize BaseCapabilityProvider attributes
        self._capabilities: Dict[str, "PersonaTraits"] = {}
        self._metadata_full: Dict[str, CapabilityMetadata] = {}
        self._initialized = True
        set_default_persona_registry(self)

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing).

        Thread-safe reset that properly acquires the lock before
        clearing the instance. This prevents race conditions during
        test teardown.
        """
        with cls._lock:
            cls._instance = None

    def register_persona(
        self,
        name: str,
        version: str,
        persona: "PersonaTraits",
        category: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        author: Optional[str] = None,
        vertical: Optional[str] = None,
        deprecated: bool = False,
    ) -> None:
        """Register a persona with the provider.

        Args:
            name: Unique persona name
            version: Semantic version string (e.g., "1.0.0")
            persona: PersonaTraits instance
            category: Persona category (research, planning, execution, review, other)
            description: Human-readable description
            tags: List of tags for discovery
            author: Optional author name
            vertical: Vertical this persona belongs to
            deprecated: Whether this persona is deprecated

        Raises:
            ValueError: If version is not valid SemVer or category is invalid
        """
        # Validate SemVer
        if not self._is_valid_semver(version):
            raise ValueError(f"Invalid SemVer version: {version}")

        # Validate category
        if category not in self._categories:
            raise ValueError(
                f"Invalid category: {category}. Must be one of: {list(self._categories.keys())}"
            )

        with self._lock:
            # Initialize version dict if needed
            if name not in self._personas:
                self._personas[name] = {}
                self._metadata[name] = {}

            # Register persona
            self._personas[name][version] = persona
            self._metadata[name][version] = PersonaMetadata(
                name=name,
                version=version,
                description=description,
                category=category,
                tags=tags or [],
                author=author,
                vertical=vertical,
                deprecated=deprecated,
            )

            # Add to category
            self._categories[category].add(name)

            # Add to capabilities dict
            self._capabilities[name] = persona

            # Add to metadata dict
            self._metadata_full[name] = CapabilityMetadata(
                name=name,
                description=description,
                version=version,
                tags=tags or [],
            )

            logger.debug(f"Registered persona: {name}@{version} (category={category})")

    def get_persona(self, name: str, version: Optional[str] = None) -> Optional["PersonaTraits"]:
        """Get a persona by name and optional version.

        Args:
            name: Persona name
            version: Optional version string. If None, returns latest version.

        Returns:
            PersonaTraits or None if not found
        """
        with self._lock:
            if name not in self._personas:
                return None

            if version is None:
                # Return latest version (highest by SemVer)
                versions = list(self._personas[name].keys())
                version = self._get_latest_version(versions)

            return self._personas[name].get(version)

    def get_persona_metadata(
        self, name: str, version: Optional[str] = None
    ) -> Optional[PersonaMetadata]:
        """Get metadata for a persona.

        Args:
            name: Persona name
            version: Optional version string. If None, returns latest version.

        Returns:
            PersonaMetadata or None if not found
        """
        with self._lock:
            if name not in self._metadata:
                return None

            if version is None:
                # Return latest version
                versions = list(self._metadata[name].keys())
                version = self._get_latest_version(versions)

            return self._metadata[name].get(version)

    def get_persona_version(self, name: str) -> Optional[str]:
        """Get the latest version of a persona.

        Args:
            name: Persona name

        Returns:
            Latest version string or None if not found
        """
        with self._lock:
            if name not in self._personas:
                return None

            versions = list(self._personas[name].keys())
            return self._get_latest_version(versions)

    def list_personas(self, category: Optional[str] = None) -> List[str]:
        """List all registered persona names.

        Args:
            category: Optional category filter. If None, returns all personas.

        Returns:
            List of persona names
        """
        with self._lock:
            if category is None:
                return list(self._personas.keys())

            if category not in self._categories:
                return []

            return list(self._categories[category])

    def list_persona_versions(self, name: str) -> List[str]:
        """List all versions of a persona.

        Args:
            name: Persona name

        Returns:
            List of version strings
        """
        with self._lock:
            if name not in self._personas:
                return []

            return sorted(self._personas[name].keys(), key=self._semver_key, reverse=True)

    def get_capabilities(self) -> Dict[str, "PersonaTraits"]:
        """Return all registered personas (as capabilities).

        Returns:
            Dictionary mapping persona names to PersonaTraits
        """
        with self._lock:
            # Return latest version of each persona
            result = {}
            for name in self._personas:
                result[name] = self.get_persona(name)
            return result

    def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
        """Return metadata for all personas.

        Returns:
            Dictionary mapping persona names to CapabilityMetadata
        """
        with self._lock:
            # Return latest version metadata for each persona
            result = {}
            for name in self._metadata:
                metadata = self.get_persona_metadata(name)
                if metadata:
                    result[name] = CapabilityMetadata(
                        name=metadata.name,
                        description=metadata.description,
                        version=metadata.version,
                        tags=metadata.tags,
                    )
            return result

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary with registry stats
        """
        with self._lock:
            total_personas = len(self._personas)
            total_versions = sum(len(versions) for versions in self._personas.values())

            category_counts = {
                category: len(personas) for category, personas in self._categories.items()
            }

            return {
                "total_personas": total_personas,
                "total_versions": total_versions,
                "category_counts": category_counts,
            }

    @staticmethod
    def _is_valid_semver(version: str) -> bool:
        """Check if version string is valid SemVer.

        Args:
            version: Version string

        Returns:
            True if valid SemVer
        """
        # Basic SemVer regex: MAJOR.MINOR.PATCH
        pattern = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
        return re.match(pattern, version) is not None

    @staticmethod
    def _semver_key(version: str) -> tuple:
        """Convert SemVer to tuple for sorting.

        Args:
            version: Version string

        Returns:
            Tuple (major, minor, patch) for sorting
        """
        # Strip pre-release and build metadata for sorting
        clean_version = version.split("-")[0].split("+")[0]
        parts = clean_version.split(".")
        return tuple(int(p) for p in parts)

    @staticmethod
    def _get_latest_version(versions: List[str]) -> str:
        """Get the latest version from a list of SemVer strings.

        Args:
            versions: List of version strings

        Returns:
            Latest version string
        """
        if not versions:
            return "0.0.0"

        return max(versions, key=FrameworkPersonaProvider._semver_key)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_persona_provider() -> FrameworkPersonaProvider:
    """Get the singleton FrameworkPersonaProvider instance.

    Returns:
        FrameworkPersonaProvider singleton
    """
    return FrameworkPersonaProvider()


__all__ = [
    "PersonaMetadata",
    "FrameworkPersonaProvider",
    "get_persona_provider",
]
