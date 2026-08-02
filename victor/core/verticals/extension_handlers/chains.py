# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Composed tool chains handler."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Type

from victor.core.verticals.extension_handlers.base import (
    BaseExtensionHandler,
    ExtensionLoaderContext,
)
from victor.core.verticals.extension_handlers.registry import ExtensionHandlerRegistry


@ExtensionHandlerRegistry.register
class ChainsHandler(BaseExtensionHandler):
    """Loads composed tool chains from runtime modules.

    Resolution: finds candidates with suffix "composed_chains",
    tries get_composed_chains() factory or CONSTANT_CHAINS dict.
    """

    extension_type = "chains"

    @classmethod
    def load(cls, ctx: Type[ExtensionLoaderContext]) -> Dict[str, Any]:
        candidate_paths = ctx._find_available_candidates("composed_chains")
        if not candidate_paths:
            return {}

        constant_name = f"{getattr(ctx, 'name', ctx.__name__).upper().replace('-', '_')}_CHAINS"
        last_error: Optional[Exception] = None

        for module_path in candidate_paths:
            try:
                module = importlib.import_module(module_path)
            except ImportError as exc:
                last_error = exc
                continue

            factory = getattr(module, "get_composed_chains", None)
            if callable(factory):
                return ctx._get_cached_extension(
                    "composed_chains",
                    lambda factory=factory: dict(factory() or {}),
                )

            chains = getattr(module, constant_name, None)
            if isinstance(chains, dict):
                return ctx._get_cached_extension(
                    "composed_chains",
                    lambda chains=chains: dict(chains),
                )

        if last_error:
            raise last_error
        return {}
