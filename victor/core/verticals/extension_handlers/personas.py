# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Team personas handler."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Type

from victor.core.verticals.extension_handlers.base import (
    BaseExtensionHandler,
    ExtensionLoaderContext,
)
from victor.core.verticals.extension_handlers.registry import ExtensionHandlerRegistry


@ExtensionHandlerRegistry.register
class PersonasHandler(BaseExtensionHandler):
    """Loads team personas from runtime team modules.

    Resolution: finds candidates with suffix "teams",
    tries get_personas() factory or CONSTANT_PERSONAS dict.
    """

    extension_type = "personas"

    @classmethod
    def load(cls, ctx: Type[ExtensionLoaderContext]) -> Dict[str, Any]:
        candidate_paths = ctx._find_available_candidates("teams")
        if not candidate_paths:
            return {}

        constant_name = f"{getattr(ctx, 'name', ctx.__name__).upper().replace('-', '_')}_PERSONAS"
        last_error: Optional[Exception] = None

        for module_path in candidate_paths:
            try:
                module = importlib.import_module(module_path)
            except ImportError as exc:
                last_error = exc
                continue

            factory = getattr(module, "get_personas", None)
            if callable(factory):
                return ctx._get_cached_extension(
                    "personas",
                    lambda factory=factory: dict(factory() or {}),
                )

            personas = getattr(module, constant_name, None)
            if isinstance(personas, dict):
                return ctx._get_cached_extension(
                    "personas",
                    lambda personas=personas: dict(personas),
                )

        if last_error:
            raise last_error
        return {}
