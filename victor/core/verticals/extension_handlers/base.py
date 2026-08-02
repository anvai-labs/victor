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

from typing import Any, ClassVar, List, Optional, Protocol, Type, runtime_checkable


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
