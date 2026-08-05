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

"""Framework-level chain registry for versioned tool chains.

This module provides a central registry for LCEL-composed tool chains
that can be shared across all verticals (Coding, DevOps, RAG, Research, DataAnalysis).

Design Pattern: Registry + Versioning
- Chains are registered with semantic versioning (SemVer)
- Chains can be retrieved by name and optional version
- Chains are categorized for discovery (exploration, editing, analysis, testing)
- Thread-safe singleton implementation

Example:
    from victor.framework.chains.registry import ChainRegistry

    # Register a chain
    ChainRegistry.register_chain(
        name="safe_edit_chain",
        version="1.0.0",
        chain=safe_edit_chain,
        category="editing",
        description="Safe edit with verification"
    )

    # Retrieve a chain
    chain = ChainRegistry.get_chain("safe_edit_chain", version="1.0.0")

    # List chains by category
    editing_chains = ChainRegistry.list_chains(category="editing")
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.tools.composition import Runnable

logger = logging.getLogger(__name__)


# =============================================================================
# Chain Metadata
# =============================================================================


@dataclass
class ChainMetadata:
    """Metadata for a registered chain.

    Attributes:
        name: Unique chain name
        version: Semantic version string (e.g., "1.0.0")
        description: Human-readable description
        category: Chain category (exploration, editing, analysis, testing)
        tags: List of tags for discovery
        author: Optional author name
        deprecated: Whether this chain is deprecated
    """

    name: str
    version: str
    description: str
    category: str
    tags: List[str] = field(default_factory=list)
    author: Optional[str] = None
    deprecated: bool = False


# =============================================================================
# Chain Registry
# =============================================================================


class ChainRegistry:
    """Central registry for versioned tool chains.

    This registry provides thread-safe storage and retrieval of LCEL-composed
    tool chains with semantic versioning support.

    Pattern: Singleton with Thread Safety
    - Single instance across the application
    - Thread-safe registration and retrieval
    - Category-based discovery

    Categories:
        - exploration: File/codebase exploration chains
        - editing: Code modification chains
        - analysis: Code analysis and review chains
        - testing: Test generation and execution chains
    """

    _instance: Optional["ChainRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ChainRegistry":
        """Create or return singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the chain registry."""
        if self._initialized:
            return

        self._chains: Dict[str, Dict[str, "Runnable"]] = {}
        self._metadata: Dict[str, Dict[str, ChainMetadata]] = {}
        self._categories: Dict[str, Set[str]] = {
            "exploration": set(),
            "editing": set(),
            "analysis": set(),
            "testing": set(),
            "other": set(),
        }
        self._initialized = True

    def register_chain(
        self,
        name: str,
        version: str,
        chain: "Runnable",
        category: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        author: Optional[str] = None,
        deprecated: bool = False,
    ) -> None:
        """Register a chain with the registry.

        Args:
            name: Unique chain name
            version: Semantic version string (e.g., "1.0.0")
            chain: LCEL Runnable chain
            category: Chain category (exploration, editing, analysis, testing, other)
            description: Human-readable description
            tags: List of tags for discovery
            author: Optional author name
            deprecated: Whether this chain is deprecated

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
            if name not in self._chains:
                self._chains[name] = {}
                self._metadata[name] = {}

            # Register chain
            self._chains[name][version] = chain
            self._metadata[name][version] = ChainMetadata(
                name=name,
                version=version,
                description=description,
                category=category,
                tags=tags or [],
                author=author,
                deprecated=deprecated,
            )

            # Add to category
            self._categories[category].add(name)

            logger.debug(f"Registered chain: {name}@{version} (category={category})")

    def get_chain(self, name: str, version: Optional[str] = None) -> Optional["Runnable"]:
        """Get a chain by name and optional version.

        Args:
            name: Chain name
            version: Optional version string. If None, returns latest version.

        Returns:
            Chain Runnable or None if not found
        """
        with self._lock:
            if name not in self._chains:
                return None

            if version is None:
                # Return latest version (highest by SemVer)
                versions = list(self._chains[name].keys())
                version = self._get_latest_version(versions)

            return self._chains[name].get(version)

    def get_chain_metadata(
        self, name: str, version: Optional[str] = None
    ) -> Optional[ChainMetadata]:
        """Get metadata for a chain.

        Args:
            name: Chain name
            version: Optional version string. If None, returns latest version.

        Returns:
            ChainMetadata or None if not found
        """
        with self._lock:
            if name not in self._metadata:
                return None

            if version is None:
                # Return latest version
                versions = list(self._metadata[name].keys())
                version = self._get_latest_version(versions)

            return self._metadata[name].get(version)

    def get_chain_version(self, name: str) -> Optional[str]:
        """Get the latest version of a chain.

        Args:
            name: Chain name

        Returns:
            Latest version string or None if not found
        """
        with self._lock:
            if name not in self._chains:
                return None

            versions = list(self._chains[name].keys())
            return self._get_latest_version(versions)

    def list_chains(self, category: Optional[str] = None) -> List[str]:
        """List all registered chain names.

        Args:
            category: Optional category filter. If None, returns all chains.

        Returns:
            List of chain names
        """
        with self._lock:
            if category is None:
                return list(self._chains.keys())

            if category not in self._categories:
                return []

            return list(self._categories[category])

    def list_chain_versions(self, name: str) -> List[str]:
        """List all versions of a chain.

        Args:
            name: Chain name

        Returns:
            List of version strings
        """
        with self._lock:
            if name not in self._chains:
                return []

            return sorted(self._chains[name].keys(), key=self._semver_key, reverse=True)

    def remove_chain(self, name: str, version: Optional[str] = None) -> bool:
        """Remove a chain from the registry.

        Args:
            name: Chain name
            version: Optional version. If None, removes all versions.

        Returns:
            True if chain was removed, False if not found
        """
        with self._lock:
            if name not in self._chains:
                return False

            if version is None:
                # Remove all versions
                for metadata in self._metadata[name].values():
                    self._categories[metadata.category].discard(name)
                del self._chains[name]
                del self._metadata[name]
                logger.debug(f"Removed all versions of chain: {name}")
                return True
            else:
                # Remove specific version
                if version in self._chains[name]:
                    metadata = self._metadata[name][version]
                    self._categories[metadata.category].discard(name)
                    del self._chains[name][version]
                    del self._metadata[name][version]

                    # Clean up if no versions left
                    if not self._chains[name]:
                        del self._chains[name]
                        del self._metadata[name]

                    logger.debug(f"Removed chain: {name}@{version}")
                    return True

            return False

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary with registry stats
        """
        with self._lock:
            total_chains = len(self._chains)
            total_versions = sum(len(versions) for versions in self._chains.values())

            category_counts = {
                category: len(chains) for category, chains in self._categories.items()
            }

            return {
                "total_chains": total_chains,
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
        # See: https://semver.org/
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

        return max(versions, key=ChainRegistry._semver_key)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_chain_registry() -> ChainRegistry:
    """Get the singleton ChainRegistry instance.

    Returns:
        ChainRegistry singleton
    """
    return ChainRegistry()


__all__ = [
    "ChainMetadata",
    "ChainRegistry",
    "get_chain_registry",
]
