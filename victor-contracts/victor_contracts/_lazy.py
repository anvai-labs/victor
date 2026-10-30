"""Shared PEP 562 lazy-import helper for the ``victor_contracts`` bridge modules.

Bridge modules (``*_runtime.py``, ``safety_patterns``, ``safety_policy``) expose
host-runtime names without importing the Victor runtime at SDK import time. Each
keeps a ``_LAZY_IMPORTS`` mapping of exported name to defining module and builds
its module-level ``__getattr__`` with :func:`make_lazy_getattr`.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable, Mapping
from typing import Any, Optional

__all__ = ["make_lazy_getattr"]


def make_lazy_getattr(
    module_name: str,
    lazy_imports: Mapping[str, str],
    deprecated: Optional[Mapping[str, str]] = None,
) -> Callable[[str], Any]:
    """Build a PEP 562 module ``__getattr__`` that resolves names lazily.

    Each access imports the defining module (``importlib`` caches it in
    ``sys.modules``) and returns the attribute; results are not stored on the
    bridge module, so the host module stays the single source of truth.

    Args:
        module_name: Fully qualified name of the bridge module, used in the
            ``AttributeError`` message (pass ``__name__``).
        lazy_imports: Mapping of exported attribute name to the fully
            qualified module that defines it. The mapping is read on every
            access, so later in-place updates are honoured.
        deprecated: Optional mapping of exported name to a deprecation
            message; accessing such a name emits ``DeprecationWarning`` with
            that message (attributed to the caller) before resolving it.

    Returns:
        A ``__getattr__(name)`` callable suitable for assignment at module
        scope.
    """

    def __getattr__(name: str) -> Any:
        target_module = lazy_imports.get(name)
        if target_module is None:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        if deprecated and name in deprecated:
            warnings.warn(deprecated[name], DeprecationWarning, stacklevel=2)
            # The caller already got the deprecation; silence the host module's
            # own import-time warning, which would be attributed to importlib.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                return getattr(importlib.import_module(target_module), name)
        return getattr(importlib.import_module(target_module), name)

    __getattr__.__qualname__ = "__getattr__"
    __getattr__.__doc__ = f"Resolve ``{module_name}`` bridge names lazily from the host runtime."
    return __getattr__
