"""SDK host adapters for graph runtime helpers."""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

from victor_contracts._deprecation import deprecated_bridge_message

if TYPE_CHECKING:
    from victor.framework.graph import END, StateGraph

__all__ = [
    "END",
    "StateGraph",
]

_LAZY_IMPORTS = {
    "END": "victor.framework.graph",
    "StateGraph": "victor.framework.graph",
}


def __getattr__(name: str) -> Any:
    """Resolve graph helpers lazily from the Victor host runtime."""
    warnings.warn(
        deprecated_bridge_message("graph_runtime"),
        DeprecationWarning,
        stacklevel=2,
    )
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'victor_contracts.graph_runtime' has no attribute {name!r}")

    module = importlib.import_module(module_name)
    return getattr(module, name)
