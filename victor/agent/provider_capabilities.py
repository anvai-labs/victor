# Copyright 2025 Vijay Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Async-safe provider capability probes."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _sync_provider_supports_reasoning_effort(provider: Any, model: Optional[str]) -> bool:
    checker = getattr(provider, "supports_reasoning_effort", None)
    if not callable(checker):
        return False
    try:
        return bool(checker(model))
    except Exception:
        logger.debug("supports_reasoning_effort check failed", exc_info=True)
        return False


async def provider_supports_reasoning_effort(provider: Any, model: Optional[str] = None) -> bool:
    """Probe reasoning support without blocking a running event loop.

    Sandhi gateway discovery needs HTTP I/O when no matching credential-scoped
    page is cached. Async call sites therefore prefer the provider's async
    probe; protocol wrappers without one are unwrapped to their underlying
    provider before falling back to the synchronous, cache-only check.
    """
    target = provider
    async_checker = getattr(target, "supports_reasoning_effort_async", None)
    if not callable(async_checker):
        target = getattr(target, "provider", target)
        async_checker = getattr(target, "supports_reasoning_effort_async", None)
    if callable(async_checker):
        try:
            return bool(await async_checker(model))
        except Exception:
            logger.debug("async supports_reasoning_effort check failed", exc_info=True)
            return False
    return _sync_provider_supports_reasoning_effort(target, model)
