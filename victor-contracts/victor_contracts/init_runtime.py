"""SDK host adapters for init synthesis runtime helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from victor_contracts._lazy import make_lazy_getattr

if TYPE_CHECKING:
    from victor.framework.init_synthesizer import InitSynthesizer

__all__ = ["InitSynthesizer"]

_LAZY_IMPORTS = {
    "InitSynthesizer": "victor.framework.init_synthesizer",
}


__getattr__ = make_lazy_getattr(__name__, _LAZY_IMPORTS)
