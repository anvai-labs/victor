"""SDK host adapters for capability protocols and lazy capability helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Callable

from victor_contracts._lazy import make_lazy_getattr

if TYPE_CHECKING:
    from victor.framework.vertical_protocols import (
        CodebaseIndexFactoryProtocol,
        EditorProtocol,
        TreeSitterParserProtocol,
    )
    from victor.core.search.indexer import detect_enhanced_index_factory

__all__ = [
    "TreeSitterParserProtocol",
    "TreeSitterAnalysisProtocol",
    "EditorProtocol",
    "CodebaseIndexFactoryProtocol",
    "create_lazy_capability_proxy",
    "detect_enhanced_index_factory",
    "get_capability_provider",
    "is_capability_enhanced",
]

_LAZY_IMPORTS = {
    "TreeSitterParserProtocol": "victor.framework.vertical_protocols",
    "TreeSitterAnalysisProtocol": "victor.framework.vertical_protocols",
    "EditorProtocol": "victor.framework.vertical_protocols",
    "CodebaseIndexFactoryProtocol": "victor.framework.vertical_protocols",
    "detect_enhanced_index_factory": "victor.core.search.indexer",
}


__getattr__ = make_lazy_getattr(__name__, _LAZY_IMPORTS)


def create_lazy_capability_proxy(provider: Callable[[], Any] | Any) -> Any:
    """Wrap a lazy capability provider using the Victor host proxy implementation."""
    try:
        module = importlib.import_module("victor.core.plugins.context")
        proxy_type = module._LazyCapabilityProxy
        return proxy_type(provider)
    except ImportError:
        return _SdkLazyCapabilityProxy(provider)


def get_capability_provider(protocol: Any) -> Any:
    """Look up a registered capability provider by protocol.

    Runtime bridge to the host's ``CapabilityRegistry`` — extracted verticals call
    this instead of importing ``victor.core.capability_registry`` directly. Returns
    ``None`` if the host runtime or the provider is unavailable.
    """
    try:
        registry_mod = importlib.import_module("victor.core.capability_registry")
        registry = registry_mod.CapabilityRegistry.get_instance()
        return registry.get(protocol)
    except Exception:
        return None


def is_capability_enhanced(protocol: Any) -> bool:
    """Check if a capability provider is registered as ENHANCED.

    Runtime bridge to ``CapabilityRegistry.is_enhanced()`` for extracted verticals.
    """
    try:
        registry_mod = importlib.import_module("victor.core.capability_registry")
        registry = registry_mod.CapabilityRegistry.get_instance()
        return bool(registry.is_enhanced(protocol))
    except Exception:
        return False


class _SdkLazyCapabilityProxy:
    """SDK-local lazy proxy used when the Victor host runtime is unavailable."""

    def __init__(self, factory: Callable[[], Any] | Any) -> None:
        object.__setattr__(self, "_factory", factory)
        object.__setattr__(self, "_instance", None)
        object.__setattr__(self, "_resolved", False)

    def _resolve(self) -> Any:
        if not object.__getattribute__(self, "_resolved"):
            factory = object.__getattribute__(self, "_factory")
            instance = factory() if callable(factory) else factory
            object.__setattr__(self, "_instance", instance)
            object.__setattr__(self, "_resolved", True)
        return object.__getattribute__(self, "_instance")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_factory", "_instance", "_resolved"}:
            object.__setattr__(self, name, value)
            return
        setattr(self._resolve(), name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        resolved = self._resolve()
        return resolved(*args, **kwargs)
