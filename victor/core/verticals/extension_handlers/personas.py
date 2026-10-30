# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Team personas handler."""

from __future__ import annotations

from typing import Any, Dict, Type

from victor.core.verticals.extension_handlers.base import (
    BaseExtensionHandler,
    ExtensionLoaderContext,
    _load_dict_extension,
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
        return _load_dict_extension(
            ctx,
            suffix="teams",
            constant_suffix="PERSONAS",
            factory_name="get_personas",
            cache_key="personas",
        )
