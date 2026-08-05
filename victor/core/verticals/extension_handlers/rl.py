# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""RL config and RL hooks handlers."""

from __future__ import annotations

from typing import Any, Optional, Type

from victor.core.verticals.extension_handlers.base import (
    BaseExtensionHandler,
    ExtensionLoaderContext,
)
from victor.core.verticals.extension_handlers.registry import ExtensionHandlerRegistry


@ExtensionHandlerRegistry.register
class RLConfigHandler(BaseExtensionHandler):
    """Loads RL configuration provider for a vertical.

    Resolution order:
    1. Entry-point: load_rl_config_provider_from_entry_points
    2. Fallback: _resolve_class_or_factory_extension with suffix "rl"
    """

    extension_type = "rl_config"

    @classmethod
    def load(cls, ctx: Type[ExtensionLoaderContext]) -> Optional[Any]:
        try:
            from victor.framework.entry_point_loader import (
                load_rl_config_provider_from_entry_points,
            )
        except ImportError:
            pass
        else:
            provider = ctx._load_cached_optional_extension(
                "rl_config_provider",
                lambda: load_rl_config_provider_from_entry_points(ctx.name),
            )
            if provider is not None:
                return provider

        from victor.core.verticals.vertical_metadata import VerticalMetadata

        metadata = VerticalMetadata.from_class(ctx)
        vertical_prefix = metadata.class_prefix
        return ctx._resolve_class_or_factory_extension(
            "rl_config_provider",
            "rl",
            class_name=f"{vertical_prefix}RLConfig",
        )


@ExtensionHandlerRegistry.register
class RLHooksHandler(BaseExtensionHandler):
    """Loads RL hooks for outcome recording."""

    extension_type = "rl_hooks"

    @classmethod
    def load(cls, ctx: Type[ExtensionLoaderContext]) -> Optional[Any]:
        from victor.core.verticals.vertical_metadata import VerticalMetadata

        metadata = VerticalMetadata.from_class(ctx)
        vertical_prefix = metadata.class_prefix
        return ctx._resolve_class_or_factory_extension(
            "rl_hooks",
            "rl",
            class_name=f"{vertical_prefix}RLHooks",
        )
