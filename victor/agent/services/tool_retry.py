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

"""Service-owned tool retry executor.

Handles tool execution with retry logic, exponential backoff,
caching, and cache invalidation.

The legacy `victor.agent.coordinators.tool_retry` module now re-exports this
implementation for compatibility.
"""

from __future__ import annotations

import asyncio
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Optional,
    Protocol,
    Tuple,
)

if TYPE_CHECKING:
    from victor.agent.tool_pipeline import ToolPipeline
    from victor.storage.cache.tool_cache import ToolCache

from victor.core.utils.log_helpers import truncate_for_log
from victor.tools.core_tool_aliases import canonicalize_core_tool_name
from victor.tools.tool_names import ToolNames, get_canonical_name

logger = logging.getLogger(__name__)


def _canonical_retry_tool_name(tool_name: str) -> str:
    """Normalize core-tool aliases before retry-time cache invalidation."""
    return get_canonical_name(canonicalize_core_tool_name(tool_name.lower()))


class ToolRetryConfig(Protocol):
    """Minimal retry config contract required by ToolRetryExecutor."""

    retry_enabled: bool
    max_retry_attempts: int
    retry_base_delay: float
    retry_max_delay: float


class ToolRetryExecutor:
    """Executes tools with retry logic and exponential backoff.

    This class encapsulates retry/backoff logic that was previously
    embedded in ToolCoordinator, including:
    - Cache lookup before execution
    - Configurable retry with exponential backoff
    - Non-retryable error detection
    - Cache invalidation for mutating tools
    - Success callbacks

    Design: Takes config, pipeline, and cache at init time.
    """

    def __init__(
        self,
        config: ToolRetryConfig,
        pipeline: "ToolPipeline",
        cache: Optional["ToolCache"] = None,
    ) -> None:
        """Initialize with execution dependencies.

        Args:
            config: ToolCoordinatorConfig for retry settings
            pipeline: ToolPipeline for actual tool execution
            cache: Optional ToolCache for result caching
        """
        self._config = config
        self._pipeline = pipeline
        self._cache = cache

    async def execute_tool_with_retry(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: Dict[str, Any],
        tool_executor: Optional[Callable[..., Awaitable[Any]]] = None,
        cache: Optional[Any] = None,
        on_success: Optional[Callable[[str, Dict[str, Any], Any], None]] = None,
        retry_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Any], bool, Optional[str]]:
        """Execute a tool with retry logic and exponential backoff.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            context: Execution context
            tool_executor: Optional custom executor callable. If provided, called as
                ``await tool_executor(tool_name, tool_args, context)``.
                Defaults to ``self._pipeline._execute_single_tool``.
            cache: Optional cache override. Defaults to ``self._cache``.
            on_success: Optional callback invoked on success as
                ``on_success(tool_name, tool_args, result)``.
            retry_config: Optional dict with keys ``retry_enabled``, ``max_attempts``,
                ``base_delay``, ``max_delay`` to override coordinator config.

        Returns:
            Tuple of (result, success, error_message or None)
        """
        effective_cache = cache if cache is not None else self._cache

        # Try cache first for allowlisted tools
        if effective_cache:
            cached = effective_cache.get(tool_name, tool_args)
            if cached is not None:
                logger.debug(f"Cache hit for tool '{tool_name}'")
                return cached, True, None

        if retry_config:
            retry_enabled = retry_config.get("retry_enabled", self._config.retry_enabled)
            max_attempts = (
                retry_config.get("max_attempts", self._config.max_retry_attempts)
                if retry_enabled
                else 1
            )
            base_delay = retry_config.get("base_delay", self._config.retry_base_delay)
            max_delay = retry_config.get("max_delay", self._config.retry_max_delay)
        else:
            retry_enabled = self._config.retry_enabled
            max_attempts = self._config.max_retry_attempts if retry_enabled else 1
            base_delay = self._config.retry_base_delay
            max_delay = self._config.retry_max_delay

        last_error = None
        for attempt in range(max_attempts):
            try:
                if tool_executor:
                    result = await tool_executor(tool_name, tool_args, context)
                else:
                    result = await self._pipeline._execute_single_tool(
                        tool_name, tool_args, context
                    )

                if result.success:
                    # Cache successful result
                    if effective_cache:
                        effective_cache.set(tool_name, tool_args, result)
                        # Invalidate related cache entries
                        canonical_tool_name = _canonical_retry_tool_name(tool_name)
                        invalidating_tools = {
                            ToolNames.WRITE,
                            ToolNames.EDIT,
                            ToolNames.SHELL,
                            "git",
                            "docker",
                        }
                        if canonical_tool_name in invalidating_tools:
                            touched_paths = []
                            if "path" in tool_args:
                                touched_paths.append(tool_args["path"])
                            if "paths" in tool_args and isinstance(tool_args["paths"], list):
                                touched_paths.extend(tool_args["paths"])
                            if touched_paths:
                                effective_cache.invalidate_paths(touched_paths)
                            else:
                                namespaces_to_clear = [ToolNames.READ, ToolNames.LS]
                                if canonical_tool_name in {
                                    "git",
                                    "docker",
                                    ToolNames.WRITE,
                                    ToolNames.EDIT,
                                }:
                                    namespaces_to_clear.extend(
                                        ["code_search", "semantic_code_search"]
                                    )
                                effective_cache.clear_namespaces(namespaces_to_clear)

                    if attempt > 0:
                        logger.info(
                            f"Tool '{tool_name}' succeeded on retry attempt "
                            f"{attempt + 1}/{max_attempts}"
                        )

                    # Invoke success callback if provided
                    if on_success:
                        on_success(tool_name, tool_args, result)

                    return result, True, None
                else:
                    # Tool returned failure - check if retryable
                    error_msg = result.error or "Unknown error"

                    # Don't retry validation errors or permanent failures
                    non_retryable_errors = [
                        "Invalid",
                        "Missing required",
                        "Not found",
                        "disabled",
                    ]
                    if any(err in error_msg for err in non_retryable_errors):
                        logger.debug(
                            f"Tool '{tool_name}' failed with non-retryable error: "
                            f"{truncate_for_log(error_msg)}"
                        )
                        return result, False, error_msg

                    last_error = error_msg
                    if attempt < max_attempts - 1:
                        # Calculate exponential backoff delay
                        delay = min(base_delay * (2**attempt), max_delay)
                        logger.warning(
                            f"Tool '{tool_name}' failed (attempt {attempt + 1}/{max_attempts}): "
                            f"{truncate_for_log(error_msg)}. Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Tool '{tool_name}' failed after {max_attempts} attempts: "
                            f"{truncate_for_log(error_msg)}"
                        )
                        return result, False, error_msg

            except Exception as e:
                # Check for non-retryable errors
                from victor.core.errors import ToolNotFoundError, ToolValidationError

                if isinstance(e, (ToolNotFoundError, ToolValidationError, PermissionError)):
                    logger.error(f"Tool '{tool_name}' permanent failure: {truncate_for_log(e)}")
                    return None, False, str(e)

                # Retryable transient errors
                last_error = str(e)
                if attempt < max_attempts - 1:
                    delay = min(base_delay * (2**attempt), max_delay)
                    logger.warning(
                        f"Tool '{tool_name}' transient error "
                        f"(attempt {attempt + 1}/{max_attempts}): {truncate_for_log(e)}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Tool '{tool_name}' failed after {max_attempts} attempts: "
                        f"{truncate_for_log(e)}"
                    )
                    return None, False, last_error

        # Should not reach here, but handle it anyway
        return None, False, last_error or "Unknown error"


__all__ = [
    "ToolRetryConfig",
    "ToolRetryExecutor",
]
