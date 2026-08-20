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

"""Backward compatibility for vertical name extraction.

Maintains support for legacy verticals that don't use @register_vertical
or define an explicit 'name' attribute. This module should only be used
as a temporary compatibility layer during migration.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from victor.core.verticals.vertical_metadata import VerticalMetadata

logger = logging.getLogger(__name__)


def get_vertical_name(cls: Type, emit_warning: bool = True) -> str:
    """Get vertical name with fallback to legacy pattern.

    This function provides backward compatibility for verticals that haven't
    been migrated to use @register_vertical or define an explicit 'name' attribute.

    Detection priority:
    1. Explicit `_victor_manifest` (from @register_vertical decorator)
    2. Explicit `name` class attribute
    3. Pattern matching on class name (with deprecation warning)
    4. Fallback to class name (with deprecation warning)

    Args:
        cls: The vertical class to extract name from
        emit_warning: Whether to emit deprecation warnings for legacy patterns

    Returns:
        The vertical name (e.g., "coding", "devops", "research")

    Examples:
        >>> # With explicit name (modern pattern)
        >>> class ModernVertical(VerticalBase):
        ...     name = "modern"
        >>> get_vertical_name(ModernVertical)
        'modern'

        >>> # With legacy pattern (emits warning)
        >>> class LegacyAssistant(VerticalBase):
        ...     pass
        >>> get_vertical_name(LegacyAssistant)
        'legacy'
    """
    # Try new metadata system first
    if hasattr(cls, "_victor_manifest"):
        manifest = cls._victor_manifest
        if isinstance(manifest, dict):
            return manifest.get("name", "")
        elif hasattr(manifest, "name"):
            return manifest.name

    # Try explicit name attribute
    if hasattr(cls, "name") and cls.name:
        return cls.name

    # Fallback to legacy pattern with deprecation warning
    if emit_warning:
        logger.warning(
            "Vertical %s should use @register_vertical(name='...') or define an "
            "explicit 'name' attribute. Automatic name inference will be removed in v1.0.",
            cls.__name__,
        )

    # Use legacy string replacement pattern
    # This is the old fragile pattern we're replacing
    name = cls.__name__.replace("Assistant", "").replace("Vertical", "")
    if not name:
        # If class name is just "Assistant" or "Vertical", use it as-is
        name = cls.__name__

    logger.debug(
        f"Using legacy name inference for '{cls.__name__}' -> '{name}'. "
        f"Please migrate to @register_vertical decorator."
    )

    return name.lower()


def get_vertical_metadata(cls: Type) -> "VerticalMetadata":
    """Get VerticalMetadata for a vertical class with backward compatibility.

    This function attempts to use the new VerticalMetadata system first,
    falling back to legacy patterns if needed.

    Args:
        cls: The vertical class to get metadata for

    Returns:
        VerticalMetadata instance

    Examples:
        >>> metadata = get_vertical_metadata(CodingAssistant)
        >>> metadata.name
        'coding'
        >>> metadata.canonical_name
        'coding'
    """
    # Try to use the new VerticalMetadata system
    try:
        from victor.core.verticals.vertical_metadata import VerticalMetadata

        return VerticalMetadata.from_class(cls)
    except Exception as e:
        # Fallback to legacy behavior if VerticalMetadata fails
        logger.warning(
            f"Failed to extract metadata using VerticalMetadata: {e}. "
            f"Falling back to legacy name extraction."
        )

        # Create a minimal metadata object using legacy pattern
        name = get_vertical_name(cls, emit_warning=False)

        # Import dataclass for creating fallback metadata
        from dataclasses import dataclass

        @dataclass
        class LegacyMetadata:
            """Minimal metadata for fallback compatibility."""

            name: str
            canonical_name: str
            display_name: str
            version: str
            api_version: int
            module_path: str
            qualname: str
            is_contrib: bool
            is_external: bool

        module_path = getattr(cls, "__module__", "<unknown>")
        is_contrib = "verticals.contrib" in module_path
        is_external = not is_contrib and not module_path.startswith("victor.verticals")

        return LegacyMetadata(
            name=name,
            canonical_name=name,
            display_name=name.replace("_", " ").title(),
            version=getattr(cls, "version", "1.0.0"),
            api_version=getattr(cls, "VERTICAL_API_VERSION", 1),
            module_path=module_path,
            qualname=getattr(cls, "__qualname__", cls.__name__),
            is_contrib=is_contrib,
            is_external=is_external,
        )


def is_legacy_vertical(cls: Type) -> bool:
    """Check if a vertical is using the legacy pattern (no explicit metadata).

    Args:
        cls: The vertical class to check

    Returns:
        True if the vertical doesn't have explicit metadata (legacy pattern)

    Examples:
        >>> is_legacy_vertical(ModernVertical)  # Has @register_vertical
        False
        >>> is_legacy_vertical(LegacyAssistant)  # No decorator
        True
    """
    return not hasattr(cls, "_victor_manifest") and not (hasattr(cls, "name") and cls.name)
