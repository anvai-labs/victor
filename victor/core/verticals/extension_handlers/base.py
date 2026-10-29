# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Base class and context protocol for type-specific extension handlers.

Each handler encapsulates the loading logic for one extension type
(middleware, safety, RL, etc.). Handlers are stateless — they receive
the VerticalExtensionLoader class as context, which provides shared
infrastructure (caching, module resolution, entry-point loading).

Design Pattern: Strategy Pattern — each handler is a strategy for
loading one extension type. The ExtensionHandlerRegistry maps
extension type names to handler classes.
"""

from __future__ import annotations

import importlib
from typing import Any, ClassVar, Dict, List, Optional, Protocol, Type, runtime_checkable


@runtime_checkable
class ExtensionLoaderContext(Protocol):
    """Protocol for the shared infrastructure handlers need from the loader.

    The VerticalExtensionLoader class satisfies this protocol. Handlers
    receive it as the `ctx` parameter to `load()`.
    """

    name: ClassVar[str]

    @classmethod
    def _get_cached_extension(cls, key: str, factory: Any) -> Any: ...

    @classmethod
    def _load_named_entry_point_extension(cls, extension_key: str, group: str) -> Optional[Any]: ...

    @classmethod
    def _find_available_candidates(cls, suffix: str) -> List[str]: ...

    @classmethod
    def _resolve_factory_extension(
        cls, extension_key: str, suffix: str, class_name: Optional[str] = None
    ) -> Optional[Any]: ...

    @classmethod
    def _resolve_class_or_factory_extension(
        cls, extension_key: str, suffix: str, class_name: Optional[str] = None
    ) -> Optional[Any]: ...


class BaseExtensionHandler:
    """Base class for type-specific extension handlers.

    Subclasses must define:
    - `extension_type` (str): registry key for this handler
    - `load(ctx)` (classmethod): loads the extension using loader context

    Handlers are stateless. All state (caching, metrics) lives in the
    loader context.
    """

    extension_type: ClassVar[str]

    @classmethod
    def load(cls, ctx: Type[ExtensionLoaderContext]) -> Any:
        """Load the extension using the given loader context.

        Args:
            ctx: The VerticalExtensionLoader class (provides shared infrastructure)

        Returns:
            Extension instance, list, or None depending on type
        """
        raise NotImplementedError(f"{cls.__name__} must implement load()")


def _load_dict_extension(
    ctx: Type[ExtensionLoaderContext],
    *,
    suffix: str,
    constant_suffix: str,
    factory_name: str,
    cache_key: str,
) -> Dict[str, Any]:
    """Load a dict-valued extension from the vertical's runtime modules.

    Tries each candidate module for ``suffix`` in order. The first module that
    defines a callable ``factory_name`` wins; otherwise the first module that
    defines a dict constant ``<VERTICAL>_<constant_suffix>`` wins. The result
    is cached on ``ctx`` under ``cache_key``.

    Args:
        ctx: Loader context (the VerticalExtensionLoader class).
        suffix: Module suffix passed to ``ctx._find_available_candidates``.
        constant_suffix: Constant-name suffix, e.g. ``"CHAINS"``.
        factory_name: Factory function name, e.g. ``"get_composed_chains"``.
        cache_key: Extension cache key.

    Returns:
        A copy of the extension dict, or ``{}`` when no candidate provides it.

    Raises:
        ImportError: The last candidate import error, when no candidate
            provided the extension and at least one failed to import.
    """
    candidate_paths = ctx._find_available_candidates(suffix)
    if not candidate_paths:
        return {}

    vertical_name = getattr(ctx, "name", ctx.__name__).upper().replace("-", "_")
    constant_name = f"{vertical_name}_{constant_suffix}"
    last_error: Optional[Exception] = None

    for module_path in candidate_paths:
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            last_error = exc
            continue

        factory = getattr(module, factory_name, None)
        if callable(factory):
            return ctx._get_cached_extension(
                cache_key,
                lambda factory=factory: dict(factory() or {}),
            )

        value = getattr(module, constant_name, None)
        if isinstance(value, dict):
            return ctx._get_cached_extension(
                cache_key,
                lambda value=value: dict(value),
            )

    if last_error:
        raise last_error
    return {}
