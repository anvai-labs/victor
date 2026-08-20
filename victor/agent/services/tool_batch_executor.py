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

"""Batch execution helper for ``ToolService``.

This module owns multi-call validation, budget truncation, and parallel/sequential
dispatch. Single-tool execution remains owned by ``ToolService`` so validation,
normalization, retries, and budget accounting stay on the canonical service.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Protocol


class ToolBatchExecutionConfig(Protocol):
    """Configuration fields needed by the batch executor."""

    enable_parallel_execution: bool


class ToolBatchExecutionHost(Protocol):
    """Minimal host surface required for batch tool execution."""

    _config: ToolBatchExecutionConfig
    _logger: logging.Logger

    def validate_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Validate a batch and return valid and invalid calls."""
        ...

    def get_remaining_budget(self) -> int:
        """Return remaining executable tool-call budget."""
        ...

    async def execute_tool_call(
        self,
        tool_call: Dict[str, Any],
        validate: bool = True,
        check_budget: bool = True,
    ) -> Dict[str, Any]:
        """Execute a single tool call."""
        ...


def _validation_error_results(
    invalid_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            "tool": call.get("name", "unknown"),
            "error": call.get("_validation_error", "Unknown error"),
        }
        for call in invalid_calls
    ]


def _exception_result(tool_call: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    return {
        "tool": tool_call.get("name", "unknown"),
        "success": False,
        "result": None,
        "error": str(error),
    }


async def _execute_valid_calls(
    host: ToolBatchExecutionHost,
    valid_calls: List[Dict[str, Any]],
    *,
    parallel: bool,
) -> List[Dict[str, Any]]:
    """Execute already validated and budget-trimmed tool calls."""
    if not valid_calls:
        return []

    if parallel and host._config.enable_parallel_execution:
        raw_results = await asyncio.gather(
            *[
                host.execute_tool_call(call, validate=False, check_budget=False)
                for call in valid_calls
            ],
            return_exceptions=True,
        )
        return [
            (
                _exception_result(valid_calls[index], result)
                if isinstance(result, Exception)
                else result
            )
            for index, result in enumerate(raw_results)
        ]

    results: List[Dict[str, Any]] = []
    for call in valid_calls:
        results.append(await host.execute_tool_call(call, validate=False, check_budget=False))
    return results


async def execute_tool_call_batch(
    host: ToolBatchExecutionHost,
    tool_calls: List[Dict[str, Any]],
    *,
    validate: bool = True,
    check_budget: bool = True,
    parallel: bool = True,
) -> List[Dict[str, Any]]:
    """Execute multiple tool calls with validation, budgeting, and dispatch policy."""
    if not tool_calls:
        return []

    if validate:
        valid_calls, invalid_calls = host.validate_tool_calls(tool_calls)
        validation_errors = _validation_error_results(invalid_calls)
    else:
        valid_calls = tool_calls
        validation_errors = []

    if check_budget:
        remaining = host.get_remaining_budget()
        if remaining < len(valid_calls):
            host._logger.warning(
                "Insufficient budget: %s calls, %s remaining",
                len(valid_calls),
                remaining,
            )
            valid_calls = valid_calls[:remaining]

    results = await _execute_valid_calls(host, valid_calls, parallel=parallel)
    return [*validation_errors, *results]


__all__ = ["ToolBatchExecutionHost", "execute_tool_call_batch"]
