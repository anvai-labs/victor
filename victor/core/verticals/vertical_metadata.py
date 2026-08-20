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

"""Type-safe vertical metadata extraction.

Replaces fragile string manipulation patterns with robust metadata access
using pattern matching and explicit attribute detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from victor.core.verticals.manifest_contract import (
    VerticalRuntimeProvenance,
    get_vertical_runtime_provenance,
)

if TYPE_CHECKING:
    from victor_contracts.verticals.manifest import ExtensionManifest

logger = logging.getLogger(__name__)


class VerticalNamingPattern(Enum):
    """Supported naming patterns for vertical classes.

    Verticals can be named using different conventions. This enum
    defines the patterns that VerticalMetadata can detect and extract.
    """

    ASSISTANT_SUFFIX = "Assistant"  # CodingAssistant -> coding
    VERTICAL_SUFFIX = "Vertical"  # CodingVertical -> coding
    EXPLICIT_NAME = "name"  # Uses class attribute 'name' directly


@dataclass(frozen=True)
class VerticalMetadata:
    """Extracted metadata from a vertical class.

    Provides type-safe access to vertical properties without relying
    on fragile string manipulation patterns.

    Attributes:
        name: The vertical name (e.g., "coding", "devops", "research")
        canonical_name: Normalized lowercase name (e.g., "coding" from "CodingAssistant")
        display_name: Human-readable name (e.g., "Coding", "Dev Ops")
        version: Vertical version string (default: "1.0.0")
        api_version: Manifest API version (default: 1)
        module_path: Full Python module path (e.g., "victor.verticals.contrib.coding")
        qualname: Qualified class name (e.g., "CodingAssistant")
        is_contrib: True if this is a built-in contrib vertical
        is_external: True if this is an external vertical package
    """

    name: str
    canonical_name: str
    display_name: str
    version: str
    api_version: int
    module_path: str
    qualname: str
    is_contrib: bool
    is_external: bool

    @property
    def class_prefix(self) -> str:
        """Return the PascalCase prefix suitable for constructing class names.

        Strips ``"Assistant"`` or ``"Vertical"`` suffixes from ``qualname``
        to recover the original casing used in class definitions.

        Examples::

            CodingAssistant   -> "Coding"
            DevOpsAssistant   -> "DevOps"
            RAGAssistant      -> "RAG"
            DataAnalysisAssistant -> "DataAnalysis"
        """
        qname = self.qualname
        if qname.endswith("Assistant"):
            return qname[: -len("Assistant")]
        if qname.endswith("Vertical"):
            return qname[: -len("Vertical")]
        return qname

    @classmethod
    def from_class(cls, vertical_class: type) -> "VerticalMetadata":
        """Extract metadata from a vertical class using type-safe patterns.

        This method replaces the fragile `.replace("Assistant", "")` pattern
        with robust pattern matching and explicit attribute detection.

        Detection priority:
        1. Explicit `_victor_manifest` (from @register_vertical decorator)
        2. Explicit `name` class attribute
        3. Pattern matching on class name (Assistant/Vertical suffixes)
        4. Fallback to class name (with deprecation warning)

        Args:
            vertical_class: The vertical class to extract metadata from

        Returns:
            VerticalMetadata instance with extracted information

        Examples:
            >>> # With explicit name
            >>> class MyVertical(VerticalBase):
            ...     name = "my_vertical"
            >>> metadata = VerticalMetadata.from_class(MyVertical)
            >>> metadata.name
            'my_vertical'

            >>> # With Assistant suffix
            >>> class CodingAssistant(VerticalBase):
            ...     pass
            >>> metadata = VerticalMetadata.from_class(CodingAssistant)
            >>> metadata.name
            'coding'
        """
        # Try explicit manifest first (from @register_vertical decorator)
        if hasattr(vertical_class, "_victor_manifest"):
            manifest = vertical_class._victor_manifest  # type: ignore
            if manifest:
                # Handle both ExtensionManifest objects and dict manifests
                if isinstance(manifest, dict):
                    # Extract from dict manifest (fallback pattern)
                    name = manifest.get("name", "")
                    if name:
                        # Use dict values but fall back to class attributes
                        return cls(
                            name=name,
                            canonical_name=cls._normalize_name(name),
                            display_name=cls._make_display_name(name),
                            version=manifest.get(
                                "version", getattr(vertical_class, "version", "1.0.0")
                            ),
                            api_version=manifest.get(
                                "api_version",
                                getattr(vertical_class, "VERTICAL_API_VERSION", 1),
                            ),
                            module_path=getattr(vertical_class, "__module__", "<unknown>"),
                            qualname=getattr(
                                vertical_class, "__qualname__", vertical_class.__name__
                            ),
                            is_contrib=(
                                get_vertical_runtime_provenance(vertical_class)
                                is VerticalRuntimeProvenance.CONTRIB
                            ),
                            is_external=(
                                get_vertical_runtime_provenance(vertical_class)
                                is VerticalRuntimeProvenance.EXTERNAL
                            ),
                        )
                else:
                    # ExtensionManifest object
                    return cls._from_manifest(manifest, vertical_class)

        # Try explicit name attribute
        if hasattr(vertical_class, "name") and vertical_class.name:
            name = vertical_class.name
        else:
            # Use pattern detection
            name = cls._extract_name_from_classname(vertical_class.__name__)

        module_path = getattr(vertical_class, "__module__", "<unknown>")
        runtime_provenance = get_vertical_runtime_provenance(vertical_class)
        is_contrib = runtime_provenance is VerticalRuntimeProvenance.CONTRIB
        is_external = runtime_provenance is VerticalRuntimeProvenance.EXTERNAL

        return cls(
            name=name,
            canonical_name=cls._normalize_name(name),
            display_name=cls._make_display_name(name),
            version=getattr(vertical_class, "version", "1.0.0"),
            api_version=getattr(vertical_class, "VERTICAL_API_VERSION", 1),
            module_path=module_path,
            qualname=getattr(vertical_class, "__qualname__", vertical_class.__name__),
            is_contrib=is_contrib,
            is_external=is_external,
        )

    @classmethod
    def _from_manifest(
        cls, manifest: "ExtensionManifest", vertical_class: type
    ) -> "VerticalMetadata":
        """Create VerticalMetadata from an ExtensionManifest.

        Args:
            manifest: The extension manifest
            vertical_class: The vertical class

        Returns:
            VerticalMetadata instance
        """
        module_path = getattr(vertical_class, "__module__", "<unknown>")
        runtime_provenance = get_vertical_runtime_provenance(vertical_class)
        is_contrib = runtime_provenance is VerticalRuntimeProvenance.CONTRIB
        is_external = runtime_provenance is VerticalRuntimeProvenance.EXTERNAL

        return cls(
            name=manifest.name,
            canonical_name=cls._normalize_name(manifest.name),
            display_name=cls._make_display_name(manifest.name),
            version=manifest.version,
            api_version=manifest.api_version,
            module_path=module_path,
            qualname=getattr(vertical_class, "__qualname__", vertical_class.__name__),
            is_contrib=is_contrib,
            is_external=is_external,
        )

    @classmethod
    def _extract_name_from_classname(cls, classname: str) -> str:
        """Extract vertical name using pattern matching instead of replace().

        This is the replacement for the fragile pattern:
            classname.replace("Assistant", "").replace("Vertical", "")

        The new implementation uses explicit suffix detection which:
        - Is more readable and explicit
        - Handles edge cases better
        - Makes the pattern clear
        - Can be easily extended with new patterns

        Args:
            classname: The class name to extract from (e.g., "CodingAssistant")

        Returns:
            Extracted vertical name (e.g., "coding")

        Examples:
            >>> VerticalMetadata._extract_name_from_classname("CodingAssistant")
            'coding'
            >>> VerticalMetadata._extract_name_from_classname("DevOpsVertical")
            'devops'
            >>> VerticalMetadata._extract_name_from_classname("Research")
            'research'
        """
        # Try each pattern in order
        # Pattern 1: Remove "Assistant" suffix
        if classname.endswith("Assistant"):
            name = classname[: -len("Assistant")]
            if name:  # Ensure we don't return empty string
                logger.debug(
                    f"Extracted vertical name '{name}' from '{classname}' "
                    f"using Assistant suffix pattern"
                )
                return name.lower()

        # Pattern 2: Remove "Vertical" suffix
        if classname.endswith("Vertical"):
            name = classname[: -len("Vertical")]
            if name:
                logger.debug(
                    f"Extracted vertical name '{name}' from '{classname}' "
                    f"using Vertical suffix pattern"
                )
                return name.lower()

        # Pattern 3: No suffix detected - use classname as-is
        # Emit deprecation warning for legacy pattern
        import warnings

        logger.warning(
            "Vertical class '%s' does not follow the recommended naming convention "
            "(Assistant or Vertical suffix). Use @register_vertical or define a 'name' attribute.",
            classname,
        )
        return classname.lower()

    @classmethod
    def _normalize_name(cls, name: str) -> str:
        """Normalize a vertical name to canonical form.

        Converts to lowercase and replaces underscores/hyphens appropriately.

        Args:
            name: The vertical name to normalize

        Returns:
            Normalized (lowercase) name
        """
        return name.lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def _make_display_name(cls, name: str) -> str:
        """Create a human-readable display name from vertical name.

        Args:
            name: The vertical name

        Returns:
            Display name (e.g., "Coding", "Dev Ops", "Research")
        """
        # Replace underscores with spaces and title-case
        return name.replace("_", " ").replace("-", " ").title()

    def __str__(self) -> str:
        """String representation showing key metadata."""
        return (
            f"VerticalMetadata(name='{self.name}', "
            f"version='{self.version}', "
            f"api_version={self.api_version}, "
            f"is_external={self.is_external})"
        )

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"VerticalMetadata("
            f"name='{self.name}', "
            f"canonical_name='{self.canonical_name}', "
            f"display_name='{self.display_name}', "
            f"version='{self.version}', "
            f"api_version={self.api_version}, "
            f"module_path='{self.module_path}', "
            f"qualname='{self.qualname}', "
            f"is_contrib={self.is_contrib}, "
            f"is_external={self.is_external})"
        )
