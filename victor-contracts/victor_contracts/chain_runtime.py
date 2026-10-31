"""Inert compatibility shim for the removed versioned chain registry (deprecated).

Until 0.13.0 this module bridged to the host's versioned chain registry
(``victor.framework.chains``), which nothing reads at runtime. Since 0.14.0 it
is self-contained and never imports the host: ``ChainRegistry`` accepts and
discards registrations, and ``get_chain_registry()`` returns a process-wide
instance of it. The shim exists only so already-published verticals that
register chains at import (victor-coding 0.13.0) keep importing. Accessing
either name emits ``DeprecationWarning``; the module is removable no earlier
than 0.15.0.

Deliver chains through the vertical's ``get_chains()`` /
``get_composed_chains()`` hook; the host registers them in its flat
``victor.framework.chain_registry`` on activation.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional

__all__ = [
    "ChainRegistry",
    "get_chain_registry",
]

_DEPRECATION = (
    "victor_contracts.chain_runtime is an inert compatibility shim since "
    "victor-contracts 0.14.0 and will be removed no earlier than 0.15.0: nothing "
    "reads the versioned chain registry. Deliver chains through the vertical's "
    "get_chains()/get_composed_chains() hook."
)


class _InertChainRegistry:
    """Accepts the old versioned-registry calls and stores nothing."""

    def register_chain(self, *args: Any, **kwargs: Any) -> None:
        """Accept and discard a chain registration."""
        return None

    def get_chain(self, *args: Any, **kwargs: Any) -> None:
        """Return ``None``: no chain is ever stored."""
        return None

    def get_chain_metadata(self, *args: Any, **kwargs: Any) -> None:
        """Return ``None``: no chain is ever stored."""
        return None

    def get_chain_version(self, *args: Any, **kwargs: Any) -> None:
        """Return ``None``: no chain is ever stored."""
        return None

    def list_chains(self, *args: Any, **kwargs: Any) -> List[str]:
        """Return an empty list: no chain is ever stored."""
        return []

    def list_chain_versions(self, *args: Any, **kwargs: Any) -> List[str]:
        """Return an empty list: no chain is ever stored."""
        return []

    def remove_chain(self, *args: Any, **kwargs: Any) -> bool:
        """Return ``False``: there is nothing to remove."""
        return False

    def get_registry_stats(self) -> Dict[str, Any]:
        """Return empty statistics in the old registry's shape."""
        return {"total_chains": 0, "total_versions": 0, "category_counts": {}}


_INERT_REGISTRY: Optional[_InertChainRegistry] = None


def _get_inert_chain_registry() -> _InertChainRegistry:
    """Return the process-wide inert registry."""
    global _INERT_REGISTRY
    if _INERT_REGISTRY is None:
        _INERT_REGISTRY = _InertChainRegistry()
    return _INERT_REGISTRY


_SHIM_EXPORTS: Dict[str, Any] = {
    "ChainRegistry": _InertChainRegistry,
    "get_chain_registry": _get_inert_chain_registry,
}

if TYPE_CHECKING:
    ChainRegistry = _InertChainRegistry
    get_chain_registry = _get_inert_chain_registry


def __getattr__(name: str) -> Any:
    """Resolve the deprecated shim names, warning at the caller."""
    target = _SHIM_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=2)
    return target
