# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Workflow provider handler."""

from __future__ import annotations

from typing import Any, List, Optional, Type

from victor.core.verticals.extension_handlers.base import (
    BaseExtensionHandler,
    ExtensionLoaderContext,
)
from victor.core.verticals.extension_handlers.registry import ExtensionHandlerRegistry


@ExtensionHandlerRegistry.register
class WorkflowHandler(BaseExtensionHandler):
    """Loads workflow provider for a vertical.

    Multi-suffix search: tries "workflows" and "workflows.provider" suffixes.
    Resolution: entry-point → class import → factory.
    """

    extension_type = "workflow"

    @classmethod
    def load(cls, ctx: Type[ExtensionLoaderContext]) -> Optional[Any]:
        provider = ctx._load_named_entry_point_extension(
            "workflow_provider", "victor.workflow_providers"
        )
        if provider is not None:
            return provider

        from victor.core.verticals.vertical_metadata import VerticalMetadata

        metadata = VerticalMetadata.from_class(ctx)
        vertical_prefix = metadata.class_prefix
        class_name = f"{vertical_prefix}WorkflowProvider"

        candidate_paths: List[str] = []
        for suffix in ("workflows", "workflows.provider"):
            for module_path in ctx._extension_module_candidates(suffix):
                if ctx._extension_module_available(module_path):
                    candidate_paths.append(module_path)

        if not candidate_paths:
            return None

        last_error: Optional[Exception] = None
        for module_path in candidate_paths:
            try:
                module = __import__(module_path, fromlist=[class_name])
                provider_class = getattr(module, class_name, None)
                if provider_class is not None:
                    return provider_class()
            except (ImportError, AttributeError) as exc:
                last_error = exc

        for module_path in candidate_paths:
            try:
                return ctx._get_extension_factory("workflow_provider", module_path)
            except (ImportError, AttributeError) as exc:
                last_error = exc

        if last_error:
            raise last_error
        return None
