"""SDK host adapters for chain registry runtime helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from victor_contracts._lazy import make_lazy_getattr

if TYPE_CHECKING:
    from victor.framework.chains import ChainRegistry, get_chain_registry

__all__ = [
    "ChainRegistry",
    "get_chain_registry",
]

_LAZY_IMPORTS = {
    "ChainRegistry": "victor.framework.chains",
    "get_chain_registry": "victor.framework.chains",
}


__getattr__ = make_lazy_getattr(__name__, _LAZY_IMPORTS)
