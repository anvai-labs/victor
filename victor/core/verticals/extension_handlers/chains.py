# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Composed tool chains handler."""

from __future__ import annotations

from typing import Any, Dict, Type

from victor.core.verticals.extension_handlers.base import (
    BaseExtensionHandler,
    ExtensionLoaderContext,
    _load_dict_extension,
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
        return _load_dict_extension(
            ctx,
            suffix="composed_chains",
            constant_suffix="CHAINS",
            factory_name="get_composed_chains",
            cache_key="composed_chains",
        )
