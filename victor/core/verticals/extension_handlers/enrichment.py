# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Enrichment strategy handler."""

from __future__ import annotations

from typing import Any, Optional, Type

from victor.core.verticals.extension_handlers.base import (
    BaseExtensionHandler,
    ExtensionLoaderContext,
)
from victor.core.verticals.extension_handlers.registry import ExtensionHandlerRegistry


@ExtensionHandlerRegistry.register
class EnrichmentHandler(BaseExtensionHandler):
    """Stub handler — returns None by default.

    Override in vertical for DSPy-like auto prompt enrichment strategies.
    """

    extension_type = "enrichment"

    @classmethod
    def load(cls, ctx: Type[ExtensionLoaderContext]) -> Optional[Any]:
        return None
