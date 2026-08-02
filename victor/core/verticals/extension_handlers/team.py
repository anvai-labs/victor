# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Team spec provider and team specs handlers."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Type

from victor.core.verticals.extension_handlers.base import (
    BaseExtensionHandler,
    ExtensionLoaderContext,
)
from victor.core.verticals.extension_handlers.registry import ExtensionHandlerRegistry


@ExtensionHandlerRegistry.register
class TeamHandler(BaseExtensionHandler):
    """Loads team specification provider for a vertical.

    Complex 3-tier fallback: entry-point → class import → factory →
    definition-based provider.
    """

    extension_type = "team_spec"

    @classmethod
    def load(cls, ctx: Type[ExtensionLoaderContext]) -> Optional[Any]:
        provider = ctx._load_named_entry_point_extension(
            "team_spec_provider", "victor.team_spec_providers"
        )
        if provider is not None:
            return provider

        from victor.core.verticals.vertical_metadata import VerticalMetadata

        metadata = VerticalMetadata.from_class(ctx)
        vertical_prefix = metadata.class_prefix
        class_name = f"{vertical_prefix}TeamSpecProvider"

        candidate_paths = [
            path
            for path in ctx._extension_module_candidates("teams")
            if ctx._extension_module_available(path)
        ]

        last_error: Optional[Exception] = None
        for module_path in candidate_paths:
            try:
                module = __import__(module_path, fromlist=[class_name])
                provider_cls = getattr(module, class_name, None)
                if provider_cls is not None:
                    return provider_cls()
            except (ImportError, AttributeError) as exc:
                last_error = exc

        for module_path in candidate_paths:
            try:
                return ctx._get_extension_factory("team_spec_provider", module_path)
            except (ImportError, AttributeError) as exc:
                last_error = exc

        # Definition-based fallback
        try:
            from victor.core.verticals.extension_loader import (
                _build_team_spec_provider_from_definition,
            )

            defn = ctx.get_definition() if hasattr(ctx, "get_definition") else None
            if defn and hasattr(defn, "team_metadata") and defn.team_metadata.teams:
                provider = _build_team_spec_provider_from_definition(defn)
                if provider is not None:
                    return provider
        except Exception as exc:
            if last_error is None:
                last_error = exc

        if last_error:
            raise last_error
        return None

    @classmethod
    def load_team_specs(cls, ctx: Type[ExtensionLoaderContext]) -> Dict[str, Any]:
        """Get team specifications by delegating to team_spec_provider."""
        provider = cls.load(ctx)
        if provider is None or not hasattr(provider, "get_team_specs"):
            return {}
        return provider.get_team_specs()
