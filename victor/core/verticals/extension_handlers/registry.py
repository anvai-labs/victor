# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Registry mapping extension type names to handler classes.

Used by VerticalExtensionLoader.get_extensions() to dynamically iterate
over all extension types without a hardcoded list. New extension types
register here instead of adding methods to the loader.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Type

from victor.core.verticals.extension_handlers.base import BaseExtensionHandler

logger = logging.getLogger(__name__)


class ExtensionHandlerRegistry:
    """Maps extension type keys to handler classes.

    Thread-safe singleton. Handlers register via the `register()` decorator.
    """

    _handlers: Dict[str, Type[BaseExtensionHandler]] = {}

    @classmethod
    def register(cls, handler_cls: Type[BaseExtensionHandler]) -> Type[BaseExtensionHandler]:
        """Register a handler class. Can be used as a decorator."""
        ext_type = handler_cls.extension_type
        cls._handlers[ext_type] = handler_cls
        logger.debug("Registered extension handler: %s → %s", ext_type, handler_cls.__name__)
        return handler_cls

    @classmethod
    def get(cls, extension_type: str) -> Optional[Type[BaseExtensionHandler]]:
        """Get handler for an extension type, or None."""
        return cls._handlers.get(extension_type)

    @classmethod
    def all_handlers(cls) -> Dict[str, Type[BaseExtensionHandler]]:
        """Return a copy of all registered handlers."""
        return dict(cls._handlers)

    @classmethod
    def reset(cls) -> None:
        """Clear all registered handlers (for testing only)."""
        cls._handlers.clear()
