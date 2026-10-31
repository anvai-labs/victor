"""SDK-owned registry protocols and the default host team-registry hook."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

_default_team_registry: Optional["TeamRegistryProtocol"] = None


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


def _install_default_persona_registry(registry: Optional[PersonaRegistryProtocol]) -> None:
    """No-op kept for published victor-ai 0.22.0 (private; not SDK surface).

    victor-ai 0.22.0's deprecated ``FrameworkPersonaProvider`` imports this
    helper; the default persona registry it fed was removed in 0.15.0.
    """
    del registry


__all__ = [
    "PersonaRegistryProtocol",
    "TeamRegistryProtocol",
    "get_default_team_registry",
    "set_default_team_registry",
]
