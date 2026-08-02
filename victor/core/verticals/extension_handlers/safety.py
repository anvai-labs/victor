# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Safety extension handler."""

from __future__ import annotations

from typing import Any, Optional, Type

from victor.core.verticals.extension_handlers.base import (
    BaseExtensionHandler,
    ExtensionLoaderContext,
)
from victor.core.verticals.extension_handlers.registry import ExtensionHandlerRegistry


@ExtensionHandlerRegistry.register
class SafetyHandler(BaseExtensionHandler):
    """Loads safety extension for a vertical.

    Resolution: delegates to _resolve_factory_extension with suffix "safety".
    """

    extension_type = "safety"

    @classmethod
    def load(cls, ctx: Type[ExtensionLoaderContext]) -> Optional[Any]:
        return ctx._resolve_factory_extension("safety_extension", "safety")
