# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Middleware extension handler."""

from __future__ import annotations

from typing import Any, List, Type

from victor.core.verticals.extension_handlers.base import (
    BaseExtensionHandler,
    ExtensionLoaderContext,
)
from victor.core.verticals.extension_handlers.registry import ExtensionHandlerRegistry


@ExtensionHandlerRegistry.register
class MiddlewareHandler(BaseExtensionHandler):
    """Loads middleware implementations for a vertical.

    Resolution: finds module candidates with suffix "middleware",
    calls get_middleware() factory, caches result as list.
    """

    extension_type = "middleware"

    @classmethod
    def load(cls, ctx: Type[ExtensionLoaderContext]) -> List[Any]:
        candidate_paths = ctx._find_available_candidates("middleware")
        if not candidate_paths:
            return []

        try:
            factory = ctx._module_resolver.try_load_from_candidates(
                candidate_paths, "get_middleware"
            )
        except ImportError:
            return []

        if factory is None:
            return []

        return ctx._get_cached_extension(
            "middleware",
            lambda factory=factory: list(factory() or []),
        )
