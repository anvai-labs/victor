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

"""Registry for personas across verticals.

Enables discovery and access to persona definitions from any vertical.

Design Philosophy:
- Singleton pattern for global persona registry
- Thread-safe operations for concurrent access
- Namespace support for vertical-specific personas
- Rich metadata for discoverability by expertise

Usage:
    from victor.framework.persona_registry import (
        PersonaRegistry,
        PersonaSpec,
        get_persona_registry,
        register_persona_spec,
        get_persona_spec,
    )

    # Get the global registry
    registry = get_persona_registry()

    # Register a persona
    spec = PersonaSpec(
        name="security_expert",
        role="Security Analyst",
        expertise=["security", "authentication", "encryption"],
        communication_style="formal",
        behavioral_traits=["thorough", "risk-aware"],
    )
    registry.register("security_expert", spec, vertical="coding")

    # Get a persona
    persona = registry.get("security_expert", vertical="coding")

    # Find personas by expertise
    security_personas = registry.find_by_expertise("security")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, TypeVar

from victor.framework.namespaced_registry import (
    NamespacedRegistry,
    make_key,
    split_qualified_name,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class PersonaSpec:
    """Specification for a persona.

    Attributes:
        name: Unique identifier for this persona
        role: The role or title of the persona (e.g., "Security Analyst")
        expertise: List of expertise areas (e.g., ["python", "security"])
        communication_style: How the persona communicates (e.g., "formal", "casual")
        behavioral_traits: List of personality quirks and behaviors
        vertical: Optional vertical namespace
        tags: Additional tags for filtering/discovery
    """

    name: str
    role: str
    expertise: List[str] = field(default_factory=list)
    communication_style: str = ""
    behavioral_traits: List[str] = field(default_factory=list)
    vertical: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        """Get the full qualified name with namespace."""
        if self.vertical:
            return f"{self.vertical}:{self.name}"
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        """Convert persona spec to dictionary for serialization.

        Returns:
            Dictionary representation of the persona spec
        """
        return {
            "name": self.name,
            "role": self.role,
            "expertise": self.expertise,
            "communication_style": self.communication_style,
            "behavioral_traits": self.behavioral_traits,
            "vertical": self.vertical,
            "tags": self.tags,
        }


class PersonaRegistry(NamespacedRegistry[PersonaSpec]):
    """Registry for personas across verticals.

    Personas are namespaced by vertical: "{vertical}:{name}"

    This class provides a centralized location for registering and
    querying persona definitions from any vertical.

    Thread-safe for concurrent access.

    Supports two registration modes:
    1. Direct registration: Register an already-created PersonaSpec object
    2. Factory registration: Register a factory function for deferred creation

    Example:
        registry = PersonaRegistry.get_instance()

        # Register a persona (direct)
        spec = PersonaSpec(
            name="senior_dev",
            role="Senior Developer",
            expertise=["python", "architecture"],
        )
        registry.register("senior_dev", spec, vertical="coding")

        # Register a factory (deferred creation)
        registry.register_factory(
            "dynamic_persona",
            lambda: PersonaSpec(name="Dynamic", role="Dynamic Role"),
            vertical="coding",
        )

        # Get direct persona
        persona = registry.get("senior_dev", vertical="coding")

        # Create from factory
        persona = registry.create("dynamic_persona", vertical="coding")

        # Query personas
        python_experts = registry.find_by_expertise("python")
    """

    _KIND = "Persona"
    _KIND_PLURAL = "personas"
    _INDEX_NAMES = ("expertise", "tag", "vertical", "role")
    _logger = logger

    def __init__(self) -> None:
        """Initialize the registry."""
        super().__init__()
        self._personas: Dict[str, PersonaSpec] = self._items
        # Inverted indexes for O(1) discovery (views onto the base indexes)
        self._expertise_index: Dict[str, Set[str]] = self._indexes["expertise"]
        self._tag_index: Dict[str, Set[str]] = self._indexes["tag"]
        self._vertical_index: Dict[str, Set[str]] = self._indexes["vertical"]
        self._roles_index: Dict[str, Set[str]] = self._indexes["role"]  # role_lower -> {keys}

    def _index_terms(self, source: Any) -> Mapping[str, Iterable[str]]:
        """Index a persona by expertise, tags, vertical and lower-cased role."""
        persona: PersonaSpec = source
        return {
            "expertise": persona.expertise,
            "tag": persona.tags,
            "vertical": [persona.vertical] if persona.vertical else [],
            "role": [persona.role.lower()],
        }

    def register(
        self,
        name: str,
        persona: PersonaSpec,
        vertical: Optional[str] = None,
        replace: bool = False,
    ) -> None:
        """Register a persona with optional vertical namespace.

        Args:
            name: Short name for the persona
            persona: PersonaSpec instance to register
            vertical: Optional vertical namespace
            replace: If True, replace existing registration
        """
        key = make_key(name, vertical)
        persona.vertical = vertical
        if self._put(key, persona, replace=replace, index_source=persona):
            logger.debug(f"Registered persona: {key}")

    def register_factory(
        self,
        name: str,
        factory: Callable[[], PersonaSpec],
        vertical: Optional[str] = None,
        replace: bool = False,
    ) -> None:
        """Register a persona factory for deferred creation.

        Factory functions are called when create() is invoked.

        Args:
            name: Short name for the persona (or "vertical:name" format)
            factory: Callable that returns a PersonaSpec when invoked
            vertical: Optional vertical namespace
            replace: If True, replace existing registration
        """
        vertical, name = split_qualified_name(name, vertical)
        key = make_key(name, vertical)
        if self._put_factory(key, factory, replace=replace, duplicate_level=logging.DEBUG):
            logger.debug(f"Registered persona factory: {key}")

    def create(self, name: str, vertical: Optional[str] = None) -> Optional[PersonaSpec]:
        """Create a persona from a registered factory.

        Invokes the factory function and returns the created persona.
        Each call creates a fresh persona instance.

        Args:
            name: Persona name (or "vertical:name" format)
            vertical: Optional vertical namespace

        Returns:
            Created PersonaSpec object, or None if factory not found

        Raises:
            RuntimeError: If factory execution fails
        """
        key, vertical, factory = self._resolve_factory(name, vertical)

        if factory is None:
            logger.debug(f"No factory registered for persona: {key}")
            return None

        try:
            persona_obj: PersonaSpec = factory()
            persona_obj.vertical = vertical
            logger.debug(f"Created persona from factory: {key}")
            return persona_obj
        except Exception as e:
            logger.error(f"Failed to create persona '{key}': {e}")
            raise RuntimeError(f"Persona factory execution failed for '{key}': {e}") from e

    def list_personas(self, vertical: Optional[str] = None) -> List[str]:
        """List all registered persona names.

        Args:
            vertical: If provided, only list personas from this vertical

        Returns:
            List of persona names (full keys)
        """
        return self._list_keys(self._personas, vertical)

    def list_specs(self, vertical: Optional[str] = None) -> List[PersonaSpec]:
        """List all persona specs.

        Args:
            vertical: If provided, only list specs from this vertical

        Returns:
            List of PersonaSpec objects
        """
        with self._lock:
            if vertical:
                prefix = f"{vertical}:"
                return [p for k, p in self._personas.items() if k.startswith(prefix)]
            return list(self._personas.values())

    def find_by_vertical(self, vertical: str) -> List[PersonaSpec]:
        """Get all personas for a specific vertical.

        Args:
            vertical: Vertical name to filter by

        Returns:
            List of PersonaSpec objects from the vertical
        """
        with self._lock:
            return self._items_list(self._index_keys("vertical", vertical))

    def find_by_expertise(self, expertise: str) -> List[PersonaSpec]:
        """Find personas with specific expertise.

        Args:
            expertise: Expertise area to filter by

        Returns:
            List of PersonaSpec objects with matching expertise
        """
        with self._lock:
            return self._items_list(self._index_keys("expertise", expertise))

    def find_by_role(self, role: str) -> List[PersonaSpec]:
        """Find personas with a specific role.

        Args:
            role: Role to filter by (case-insensitive substring match)

        Returns:
            List of PersonaSpec objects with matching role
        """
        role_lower = role.lower()
        with self._lock:
            # For exact match, use index; for substring, fall back to scan
            if role_lower in self._roles_index:
                return self._items_list(self._roles_index[role_lower])
            # Substring match fallback
            return [p for p in self._personas.values() if role_lower in p.role.lower()]

    def find_by_tag(self, tag: str) -> List[PersonaSpec]:
        """Find personas with a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of PersonaSpec objects with matching tag
        """
        with self._lock:
            return self._items_list(self._index_keys("tag", tag))

    def find_by_tags(self, tags: List[str], match_all: bool = False) -> List[PersonaSpec]:
        """Find personas matching multiple tags.

        Args:
            tags: List of tags to match
            match_all: If True, match all tags; if False, match any

        Returns:
            List of PersonaSpec objects matching the criteria
        """
        with self._lock:
            if not tags:
                return list(self._personas.values())
            return self._items_list(self._match_index("tag", tags, match_all))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the registry to a dictionary.

        Returns:
            Dict with persona names as keys and spec dicts as values
        """
        with self._lock:
            return {key: spec.to_dict() for key, spec in self._personas.items()}

    def register_from_vertical(
        self,
        vertical_name: str,
        personas: Dict[str, PersonaSpec],
        replace: bool = True,
    ) -> int:
        """Register multiple personas from a vertical.

        Convenience method for bulk registration with namespace prefixing.

        Args:
            vertical_name: Vertical name for namespace
            personas: Dict mapping persona names to PersonaSpec objects
            replace: If True, replace existing registrations

        Returns:
            Number of personas registered
        """
        return self._register_from_vertical(
            vertical_name,
            personas,
            lambda name, persona: self.register(
                name, persona, vertical=vertical_name, replace=replace
            ),
        )


def get_persona_registry() -> PersonaRegistry:
    """Get the global persona registry.

    Thread-safe singleton access; the same object as
    ``PersonaRegistry.get_instance()``.

    Returns:
        Global PersonaRegistry instance
    """
    return PersonaRegistry.get_instance()


def register_persona_spec(
    name: str,
    persona: PersonaSpec,
    *,
    vertical: Optional[str] = None,
    replace: bool = False,
) -> None:
    """Register a persona in the global registry.

    Convenience function for quick registration.

    Args:
        name: Persona name
        persona: PersonaSpec object
        vertical: Optional vertical namespace
        replace: Replace existing if present
    """
    get_persona_registry().register(
        name,
        persona,
        vertical=vertical,
        replace=replace,
    )


def get_persona_spec(name: str, vertical: Optional[str] = None) -> Optional[PersonaSpec]:
    """Get a persona from the global registry.

    Args:
        name: Persona name
        vertical: Optional vertical namespace

    Returns:
        PersonaSpec or None
    """
    return get_persona_registry().get(name, vertical=vertical)


def create_persona_spec(name: str, vertical: Optional[str] = None) -> Optional[PersonaSpec]:
    """Create a persona from a registered factory.

    Convenience function for creating personas from factories.

    Args:
        name: Persona name (or "vertical:name" format)
        vertical: Optional vertical namespace

    Returns:
        Created PersonaSpec object, or None if factory not found
    """
    return get_persona_registry().create(name, vertical=vertical)


def persona(
    name: str,
    *,
    replace: bool = False,
) -> Callable[[F], F]:
    """Decorator for registering a persona factory function.

    The decorated function becomes a factory that is registered in the
    global PersonaRegistry. When the persona is requested via create(),
    the factory function is called to create a fresh instance.

    Args:
        name: Persona name (supports "vertical:name" format)
        replace: If True, replace existing registration

    Returns:
        Decorator function

    Example:
        @persona("coding:expert_reviewer")
        def expert_reviewer():
            return PersonaSpec(
                name="Expert Reviewer",
                role="Expert Code Reviewer",
                expertise=["code review", "security"],
            )

        # Later, create the persona:
        p = create_persona_spec("coding:expert_reviewer")
    """

    def decorator(func: F) -> F:
        # Parse vertical from name if present
        vertical = None
        persona_name = name
        if ":" in name:
            vertical, persona_name = name.split(":", 1)

        # Register the factory
        get_persona_registry().register_factory(
            persona_name,
            func,
            vertical=vertical,
            replace=replace,
        )

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> PersonaSpec:
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def reset_persona_registry() -> None:
    """Reset the global persona registry (for testing).

    Creates a fresh registry on next access.
    """
    PersonaRegistry.reset_instance()


__all__ = [
    "PersonaRegistry",
    "PersonaSpec",
    "get_persona_registry",
    "register_persona_spec",
    "get_persona_spec",
    "create_persona_spec",
    "persona",
    "reset_persona_registry",
]
