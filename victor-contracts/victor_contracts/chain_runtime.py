"""SDK host adapters for the versioned chain registry (deprecated).

Deprecated since 0.13.0, removable no earlier than 0.14.0: nothing in the
host reads the versioned registry. Verticals deliver chains through their
``get_chains()`` / ``get_composed_chains()`` hook and the host registers them
in its chain registry when the vertical is activated.
"""

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


_DEPRECATION = (
    "victor_contracts.chain_runtime.{name} is deprecated since victor-contracts 0.13.0 "
    "and will be removed no earlier than 0.14.0: the versioned chain registry has no "
    "runtime readers. Deliver chains through the vertical's get_chains() / "
    "get_composed_chains() hook; the host registers them on activation (the host's "
    "flat victor.framework.chain_registry is unaffected)."
)

__getattr__ = make_lazy_getattr(
    __name__,
    _LAZY_IMPORTS,
    deprecated={name: _DEPRECATION.format(name=name) for name in _LAZY_IMPORTS},
)
