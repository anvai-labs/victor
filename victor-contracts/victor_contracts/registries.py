"""SDK-owned registry protocols and default host registry hooks."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

_default_team_registry: Optional["TeamRegistryProtocol"] = None
_default_persona_registry: Optional["PersonaRegistryProtocol"] = None


@runtime_checkable
class TeamRegistryProtocol(Protocol):
    """Protocol for host-owned team registries."""

    def register_from_vertical(self, vertical: str, team_specs: Dict[str, Any]) -> int:
        """Register a vertical's team specs with the host runtime."""
        ...


@runtime_checkable
class PersonaRegistryProtocol(Protocol):
    """Protocol for host-owned persona registries."""

    def register_persona(
        self,
        *,
        name: str,
        version: str,
        persona: Any,
        category: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        author: Optional[str] = None,
        vertical: Optional[str] = None,
        deprecated: bool = False,
    ) -> None:
        """Register a persona with the host runtime."""
        ...


def set_default_team_registry(registry: Optional[TeamRegistryProtocol]) -> None:
    """Set the process-wide default host team registry."""
    global _default_team_registry
    _default_team_registry = registry


def get_default_team_registry() -> Optional[TeamRegistryProtocol]:
    """Get the process-wide default host team registry."""
    return _default_team_registry


_PERSONA_REGISTRY_HOOK_DEPRECATION = (
    "victor_contracts.registries.{name} is deprecated since victor-contracts 0.14.0 "
    "and will be removed no earlier than 0.15.0: no host installs a default persona "
    "registry. Verticals expose personas through their get_personas() hook (the host "
    "registers them on activation); a host that registers personas itself should pass "
    "provider= to register_vertical_personas explicitly."
)


def set_default_persona_registry(registry: Optional[PersonaRegistryProtocol]) -> None:
    """Set the process-wide default host persona registry.

    .. deprecated:: 0.14.0
        Removable no earlier than 0.15.0. No host installs a default persona
        registry; verticals expose personas through ``get_personas()`` and hosts
        pass ``provider=`` to ``register_vertical_personas`` explicitly.
    """
    warnings.warn(
        _PERSONA_REGISTRY_HOOK_DEPRECATION.format(name="set_default_persona_registry"),
        DeprecationWarning,
        stacklevel=2,
    )
    global _default_persona_registry
    _default_persona_registry = registry


def get_default_persona_registry() -> Optional[PersonaRegistryProtocol]:
    """Get the process-wide default host persona registry.

    .. deprecated:: 0.14.0
        Removable no earlier than 0.15.0; see :func:`set_default_persona_registry`.
    """
    warnings.warn(
        _PERSONA_REGISTRY_HOOK_DEPRECATION.format(name="get_default_persona_registry"),
        DeprecationWarning,
        stacklevel=2,
    )
    return _default_persona_registry


def _install_default_persona_registry(registry: Optional[PersonaRegistryProtocol]) -> None:
    """Set the default persona registry without warning (deprecated-host internal)."""
    global _default_persona_registry
    _default_persona_registry = registry


def _current_default_persona_registry() -> Optional[PersonaRegistryProtocol]:
    """Return the default persona registry without warning (deprecated-path internal)."""
    return _default_persona_registry


__all__ = [
    "PersonaRegistryProtocol",
    "TeamRegistryProtocol",
    "get_default_persona_registry",
    "get_default_team_registry",
    "set_default_persona_registry",
    "set_default_team_registry",
]
