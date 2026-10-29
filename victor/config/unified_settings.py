"""Unified configuration settings -- DEPRECATED.

This module re-exports Settings from victor.config.settings as VictorSettings
for backward compatibility. New code should import Settings directly:

    from victor.config.settings import Settings
"""

from __future__ import annotations

import logging

from victor.config.settings import Settings

logger = logging.getLogger(__name__)

# Backward-compatible alias
VictorSettings = Settings

__all__ = ["VictorSettings"]
