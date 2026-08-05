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
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Generator, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Singleton instance
_registry_instance: Optional["PersonaRegistry"] = None
_registry_lock = threading.Lock()


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


class PersonaRegistry:
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

    _instance: Optional["PersonaRegistry"] = None
    _class_lock: threading.Lock = threading.Lock()

    def __init__(self):
        """Initialize the registry."""
        self._personas: Dict[str, PersonaSpec] = {}
        self._factories: Dict[str, Callable[[], PersonaSpec]] = {}
        self._lock = threading.RLock()
        # Batch update support for efficient bulk registration
        self._batch_depth: int = 0
        self._batch_dirty: bool = False
        # Inverted indexes for O(1) discovery
        self._expertise_index: Dict[str, Set[str]] = {}  # expertise -> {keys}
        self._tag_index: Dict[str, Set[str]] = {}  # tag -> {keys}
        self._vertical_index: Dict[str, Set[str]] = {}  # vertical -> {keys}
        self._roles_index: Dict[str, Set[str]] = {}  # role_lower -> {keys}

    @classmethod
    def get_instance(cls) -> "PersonaRegistry":
        """Get the singleton instance of PersonaRegistry.

        Thread-safe singleton access.

        Returns:
            The global PersonaRegistry instance.
        """
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing).

        Creates a fresh registry on next get_instance() call.
        """
        with cls._class_lock:
            cls._instance = None

    @contextmanager
    def batch_update(self) -> Generator[None, None, None]:
        """Batch multiple persona mutations with reduced lock overhead.

        Use when registering many personas at once (e.g., during startup
        or vertical activation) to avoid repeated lock acquisition.

        Example:
            with registry.batch_update():
                registry.register("persona1", spec1, vertical="test")
                registry.register("persona2", spec2, vertical="test")
                registry.register("persona3", spec3, vertical="test")
            # Single lock acquisition for all operations
        """
        with self._lock:
            self._batch_depth += 1
            try:
                yield
            finally:
                self._batch_depth -= 1
                if self._batch_depth == 0:
                    self._batch_dirty = False

    def _update_indexes(self, key: str, persona: PersonaSpec) -> None:
        """Update inverted indexes for a persona.

        Args:
            key: The persona key in the registry
            persona: The PersonaSpec to index
        """
        # Index by expertise
        for exp in persona.expertise:
            if exp not in self._expertise_index:
                self._expertise_index[exp] = set()
            self._expertise_index[exp].add(key)

        # Index by tags
        for tag in persona.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(key)

        # Index by vertical
        if persona.vertical:
            if persona.vertical not in self._vertical_index:
                self._vertical_index[persona.vertical] = set()
            self._vertical_index[persona.vertical].add(key)

        # Index by role (lowercase for case-insensitive search)
        role_lower = persona.role.lower()
        if role_lower not in self._roles_index:
            self._roles_index[role_lower] = set()
        self._roles_index[role_lower].add(key)

    def _remove_from_indexes(self, key: str) -> None:
        """Remove a persona key from all inverted indexes."""
        for index in (
            self._expertise_index,
            self._tag_index,
            self._vertical_index,
            self._roles_index,
        ):
            empty_entries = []
            for index_key, keys in index.items():
                keys.discard(key)
                if not keys:
                    empty_entries.append(index_key)
            for index_key in empty_entries:
                del index[index_key]

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

        Raises:
            ValueError: If name already registered and replace=False
        """
        key = f"{vertical}:{name}" if vertical else name
        persona.vertical = vertical

        # Use conditional lock: only acquire if not in batch mode
        if self._batch_depth > 0:
            self._batch_dirty = True
            if key in self._personas and not replace:
                logger.warning(f"Persona '{key}' already registered, skipping")
                return
            self._remove_from_indexes(key)
            self._personas[key] = persona
            self._update_indexes(key, persona)
        else:
            with self._lock:
                if key in self._personas and not replace:
                    logger.warning(f"Persona '{key}' already registered, skipping")
                    return

                self._remove_from_indexes(key)
                self._personas[key] = persona
                self._update_indexes(key, persona)
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
        # Support "vertical:name" format in name parameter
        if ":" in name and vertical is None:
            vertical, name = name.split(":", 1)

        key = f"{vertical}:{name}" if vertical else name

        with self._lock:
            if (key in self._factories or key in self._personas) and not replace:
                logger.debug(f"Persona/factory '{key}' already registered, skipping")
                return

            self._factories[key] = factory
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
        # Support "vertical:name" format
        if ":" in name and vertical is None:
            vertical, name = name.split(":", 1)

        key = f"{vertical}:{name}" if vertical else name

        with self._lock:
            if key in self._factories:
                factory = self._factories[key]
            elif vertical:
                factory = self._factories.get(name)
            else:
                factory = None

        if factory is None:
            logger.debug(f"No factory registered for persona: {key}")
            return None

        try:
            persona_obj = factory()
            persona_obj.vertical = vertical
            logger.debug(f"Created persona from factory: {key}")
            return persona_obj
        except Exception as e:
            logger.error(f"Failed to create persona '{key}': {e}")
            raise RuntimeError(f"Persona factory execution failed for '{key}': {e}") from e

    def has(self, name: str, vertical: Optional[str] = None) -> bool:
        """Check if a persona or factory is registered.

        Args:
            name: Persona name
            vertical: Optional vertical namespace

        Returns:
            True if registered (either as persona or factory)
        """
        key = f"{vertical}:{name}" if vertical else name
        with self._lock:
            return key in self._personas or key in self._factories

    def unregister(self, name: str, vertical: Optional[str] = None) -> bool:
        """Unregister a persona or factory.

        Args:
            name: Persona name to unregister
            vertical: Optional vertical namespace

        Returns:
            True if unregistered, False if not found
        """
        key = f"{vertical}:{name}" if vertical else name

        with self._lock:
            found = False
            if key in self._personas:
                del self._personas[key]
                found = True
                self._remove_from_indexes(key)
            if key in self._factories:
                del self._factories[key]
                found = True

            if found:
                logger.debug(f"Unregistered persona: {key}")
            return found

    def get(self, name: str, vertical: Optional[str] = None) -> Optional[PersonaSpec]:
        """Get a persona by name.

        Args:
            name: Persona name to retrieve
            vertical: Optional vertical to scope the search

        Returns:
            PersonaSpec or None if not found
        """
        with self._lock:
            if vertical:
                key = f"{vertical}:{name}"
                if key in self._personas:
                    return self._personas[key]
            return self._personas.get(name)

    def list_personas(self, vertical: Optional[str] = None) -> List[str]:
        """List all registered persona names.

        Args:
            vertical: If provided, only list personas from this vertical

        Returns:
            List of persona names (full keys)
        """
        with self._lock:
            if vertical:
                prefix = f"{vertical}:"
                return [k for k in self._personas.keys() if k.startswith(prefix)]
            return list(self._personas.keys())

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
            keys = self._vertical_index.get(vertical, set())
            return [self._personas[k] for k in keys if k in self._personas]

    def find_by_expertise(self, expertise: str) -> List[PersonaSpec]:
        """Find personas with specific expertise.

        Args:
            expertise: Expertise area to filter by

        Returns:
            List of PersonaSpec objects with matching expertise
        """
        with self._lock:
            keys = self._expertise_index.get(expertise, set())
            return [self._personas[k] for k in keys if k in self._personas]

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
                keys = self._roles_index[role_lower]
                return [self._personas[k] for k in keys if k in self._personas]
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
            keys = self._tag_index.get(tag, set())
            return [self._personas[k] for k in keys if k in self._personas]

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

            if match_all:
                # Find keys that have all tags (intersection)
                keys = None
                for tag in tags:
                    tag_keys = self._tag_index.get(tag, set())
                    if keys is None:
                        keys = tag_keys.copy()
                    else:
                        keys &= tag_keys
                        if not keys:  # Early exit if no match
                            return []
                return [self._personas[k] for k in (keys or set()) if k in self._personas]
            else:
                # Find keys that have any tag (union)
                keys = set()
                for tag in tags:
                    keys.update(self._tag_index.get(tag, set()))
                return [self._personas[k] for k in keys if k in self._personas]

    def clear(self) -> None:
        """Clear all registered personas and factories (for testing)."""
        with self._lock:
            self._personas.clear()
            self._factories.clear()
            self._expertise_index.clear()
            self._tag_index.clear()
            self._vertical_index.clear()
            self._roles_index.clear()
            logger.debug("Cleared persona registry")

    def list_factories(self, vertical: Optional[str] = None) -> List[str]:
        """List all registered factory names.

        Args:
            vertical: If provided, only list factories from this vertical

        Returns:
            List of factory names (full keys)
        """
        with self._lock:
            if vertical:
                prefix = f"{vertical}:"
                return [k for k in self._factories.keys() if k.startswith(prefix)]
            return list(self._factories.keys())

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
        count = 0
        for name, persona in personas.items():
            try:
                self.register(
                    name,
                    persona,
                    vertical=vertical_name,
                    replace=replace,
                )
                count += 1
            except Exception as e:
                logger.warning(f"Failed to register {vertical_name}:{name}: {e}")

        logger.info(f"Registered {count} personas from vertical '{vertical_name}'")
        return count


def get_persona_registry() -> PersonaRegistry:
    """Get the global persona registry.

    Thread-safe singleton access.

    Returns:
        Global PersonaRegistry instance
    """
    global _registry_instance

    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = PersonaRegistry()

    return _registry_instance


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
    global _registry_instance
    with _registry_lock:
        _registry_instance = None
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
