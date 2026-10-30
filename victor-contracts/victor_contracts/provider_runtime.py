"""SDK host adapters for provider runtime helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from victor_contracts._lazy import make_lazy_getattr

if TYPE_CHECKING:
    from victor.providers.base import Message
    from victor.providers.registry import ProviderRegistry

__all__ = [
    "Message",
    "ProviderRegistry",
]

_LAZY_IMPORTS = {
    "Message": "victor.providers.base",
    "ProviderRegistry": "victor.providers.registry",
}


__getattr__ = make_lazy_getattr(__name__, _LAZY_IMPORTS)
