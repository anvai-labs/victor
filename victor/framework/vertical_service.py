# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared framework service for vertical integration application.

Centralizes vertical application so CLI and SDK paths use the same
pipeline instance and behavior.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Optional, Type, Union

if TYPE_CHECKING:
    from victor.core.verticals.base import VerticalBase
    from victor.framework.vertical_integration import (
        IntegrationResult,
        VerticalIntegrationPipeline,
    )

logger = logging.getLogger(__name__)

_PIPELINE_LOCK = threading.Lock()
_PIPELINE: Optional["VerticalIntegrationPipeline"] = None


def get_vertical_integration_pipeline(
    *,
    reset: bool = False,
    strict: bool = False,
    enable_cache: bool = True,
    cache_ttl: int = 3600,
    max_cache_entries: int = 256,
    cache_policy: Optional[Any] = None,
    enable_parallel: bool = True,
) -> "VerticalIntegrationPipeline":
    """Get shared VerticalIntegrationPipeline singleton.

    Args:
        reset: If True, recreate the singleton pipeline (for tests).
        strict: Strict mode for newly created pipeline.
        enable_cache: Enable caching for newly created pipeline.
        cache_ttl: Cache TTL for newly created pipeline.
        max_cache_entries: Maximum cache entries for newly created pipeline.
        cache_policy: Optional custom cache policy for newly created pipeline.
        enable_parallel: Enable async parallel step execution for newly created pipeline.

    Returns:
        Shared VerticalIntegrationPipeline instance.
    """
    global _PIPELINE

    with _PIPELINE_LOCK:
        if reset:
            _PIPELINE = None

        if _PIPELINE is None:
            from victor.framework.vertical_integration import (
                create_integration_pipeline,
            )

            _PIPELINE = create_integration_pipeline(
                strict=strict,
                enable_cache=enable_cache,
                cache_ttl=cache_ttl,
                max_cache_entries=max_cache_entries,
                cache_policy=cache_policy,
                enable_parallel=enable_parallel,
            )
            logger.debug(
                "Created shared vertical integration pipeline "
                "(strict=%s, cache=%s, parallel=%s)",
                strict,
                enable_cache,
                enable_parallel,
            )

        return _PIPELINE


def apply_vertical_configuration(
    orchestrator: Any,
    vertical: Union[Type["VerticalBase"], str],
    *,
    source: str = "framework",
) -> "IntegrationResult":
    """Apply a vertical to an orchestrator via shared pipeline.

    Args:
        orchestrator: Target orchestrator.
        vertical: Vertical class or vertical name.
        source: Source identifier for logging.

    Returns:
        IntegrationResult from pipeline application.
    """
    pipeline = get_vertical_integration_pipeline()
    result = pipeline.apply(orchestrator, vertical)

    if result.success:
        logger.info(
            "Applied vertical '%s' via %s path: tools=%d middleware=%d safety=%d hints=%d",
            result.vertical_name,
            source,
            len(result.tools_applied),
            result.middleware_count,
            result.safety_patterns_count,
            result.prompt_hints_count,
        )
    else:
        for error in result.errors:
            logger.error("Vertical integration error (%s): %s", source, error)
        for warning in result.warnings:
            logger.warning("Vertical integration warning (%s): %s", source, warning)

    return result


def clear_vertical_integration_pipeline_cache() -> None:
    """Clear cache state on shared VerticalIntegrationPipeline."""
    pipeline = get_vertical_integration_pipeline()
    clear_fn = getattr(pipeline, "clear_cache", None)
    if callable(clear_fn):
        clear_fn()
    else:
        logger.debug("Shared pipeline does not expose clear_cache(); skipping cache clear")


__all__ = [
    "get_vertical_integration_pipeline",
    "apply_vertical_configuration",
    "clear_vertical_integration_pipeline_cache",
]
