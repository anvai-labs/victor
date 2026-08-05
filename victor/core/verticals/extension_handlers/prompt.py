# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Prompt contributor extension handler."""

from __future__ import annotations

from typing import Any, Optional, Type

from victor.core.verticals.extension_handlers.base import (
    BaseExtensionHandler,
    ExtensionLoaderContext,
)
from victor.core.verticals.extension_handlers.registry import ExtensionHandlerRegistry


@ExtensionHandlerRegistry.register
class PromptHandler(BaseExtensionHandler):
    """Loads prompt contributor for a vertical.

    Resolution order:
    1. Entry point: victor.prompt_contributors group
    2. Fallback: _resolve_factory_extension with suffix "prompts"
    """

    extension_type = "prompt"

    @classmethod
    def load(cls, ctx: Type[ExtensionLoaderContext]) -> Optional[Any]:
        # Try entry point first
        contributor = ctx._load_named_entry_point_extension(
            "prompt_contributor",
            "victor.prompt_contributors",
        )
        if contributor is not None:
            return contributor

        # Fallback to module discovery
        return ctx._resolve_factory_extension("prompt_contributor", "prompts")
