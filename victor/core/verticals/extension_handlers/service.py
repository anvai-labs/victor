# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Service provider and capability provider handlers."""

from __future__ import annotations

import importlib
from typing import Any, Optional, Type

from victor.core.verticals.extension_handlers.base import (
    BaseExtensionHandler,
    ExtensionLoaderContext,
)
from victor.core.verticals.extension_handlers.registry import ExtensionHandlerRegistry


@ExtensionHandlerRegistry.register
class ServiceHandler(BaseExtensionHandler):
    """Loads service provider for DI container registration.

    Complex fallback: entry-point → class import → factory →
    VerticalServiceProviderFactory.create().
    """

    extension_type = "service"

    @classmethod
    def load(cls, ctx: Type[ExtensionLoaderContext]) -> Optional[Any]:
        provider = ctx._load_named_entry_point_extension(
            "service_provider", "victor.service_providers"
        )
        if provider is not None:
            return provider

        from victor.core.verticals.vertical_metadata import VerticalMetadata

        metadata = VerticalMetadata.from_class(ctx)
        vertical_prefix = metadata.class_prefix
        class_name = f"{vertical_prefix}ServiceProvider"
        candidate_paths = ctx._find_available_candidates("service_provider")
        last_error: Optional[Exception] = None

        for module_path in candidate_paths:
            try:
                module = importlib.import_module(module_path)
            except ImportError as exc:
                last_error = exc
                continue

            provider_cls = getattr(module, class_name, None)
            if provider_cls is not None:
                return ctx._get_cached_extension(
                    "service_provider",
                    lambda provider_cls=provider_cls: provider_cls(),
                )

            factory = getattr(module, "get_service_provider", None)
            if callable(factory):
                return ctx._get_cached_extension("service_provider", factory)

        try:
            from victor.core.verticals.base_service_provider import (
                VerticalServiceProviderFactory,
            )

            return VerticalServiceProviderFactory.create(ctx)
        except ImportError:
            if last_error is not None:
                raise last_error
            return None


@ExtensionHandlerRegistry.register
class CapabilityHandler(BaseExtensionHandler):
    """Loads capability provider for a vertical."""

    extension_type = "capability"

    @classmethod
    def load(cls, ctx: Type[ExtensionLoaderContext]) -> Optional[Any]:
        provider = ctx._load_named_entry_point_extension(
            "capability_provider", "victor.capability_providers"
        )
        if provider is not None:
            return provider

        from victor.core.verticals.vertical_metadata import VerticalMetadata

        metadata = VerticalMetadata.from_class(ctx)
        vertical_prefix = metadata.class_prefix
        return ctx._resolve_class_or_factory_extension(
            "capability_provider",
            "capabilities",
            class_name=f"{vertical_prefix}CapabilityProvider",
        )
