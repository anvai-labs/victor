# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Utility functions extracted from AgentOrchestrator.

This module contains pure utility functions that don't require orchestrator
state, enabling better testability and reduced orchestrator complexity.

Part of CRITICAL-001: Monolithic Orchestrator decomposition.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING
from victor.tools.tool_names import get_canonical_name

if TYPE_CHECKING:
    from victor.config.settings import Settings
    from victor.providers.base import BaseProvider
    from victor.agent.presentation import PresentationProtocol

logger = logging.getLogger(__name__)


def calculate_max_context_chars(
    settings: "Settings",
    provider: "BaseProvider",
    model: str,
) -> int:
    """Calculate maximum context size in characters for a model.

    Uses externalized config from provider_context_limits.yaml with
    per-provider and per-model overrides.

    Args:
        settings: Application settings
        provider: LLM provider instance
        model: Model identifier

    Returns:
        Maximum context size in characters
    """
    # Check settings override first
    settings_max = getattr(settings, "max_context_chars", None)
    if settings_max and settings_max > 0:
        return settings_max

    # Use externalized config (YAML-based)
    provider_name = getattr(provider, "name", "").lower()
    try:
        from victor.config.config_loaders import get_provider_limits

        limits = get_provider_limits(provider_name, model)
        context_tokens = limits.context_window
        logger.debug(f"Using YAML config for {provider_name}/{model}: {context_tokens} tokens")
    except Exception as e:
        logger.warning(f"Could not load provider limits from config: {e}")
        context_tokens = 128000  # Default safe value

    # Convert tokens to chars: ~3.5 chars per token with 80% safety margin
    try:
        # Try to coerce numeric-like values (including strings) to float
        token_count = float(context_tokens)
        max_chars = int(token_count * 3.5 * 0.8)
    except Exception:
        # Fallback to a conservative default if parsing fails
        logger.debug(
            "Could not parse context token value %r; using failsafe 100k tokens",
            context_tokens,
        )
        max_chars = int(100000 * 3.5 * 0.8)

    # Log safely depending on whether context_tokens is numeric
    if isinstance(context_tokens, (int, float)):
        logger.info(f"Model context: {int(context_tokens):,} tokens -> {max_chars:,} chars limit")
    else:
        logger.info(
            "Model context: %r tokens -> %s chars limit",
            context_tokens,
            f"{max_chars:,}",
        )

    return max_chars


def infer_git_operation(
    original_name: str, canonical_name: str, args: Dict[str, Any]
) -> Dict[str, Any]:
    """Infer git operation from alias when not explicitly provided.

    When the model calls 'git_log' instead of 'git' with operation='log',
    we need to infer the operation from the alias.

    Args:
        original_name: Original tool name from model (e.g., 'git_log')
        canonical_name: Resolved canonical name (e.g., 'git')
        args: Tool arguments

    Returns:
        Updated args with inferred operation if applicable
    """
    # Only process git tool with operation-based aliases
    if canonical_name != "git":
        return args

    # If operation already provided, no inference needed
    if args.get("operation"):
        return args

    # Map alias suffixes to operation names
    alias_to_operation = {
        "git_status": "status",
        "git_diff": "diff",
        "git_log": "log",
        "git_commit": "commit",
        "git_branch": "branch",
        "git_stage": "stage",
    }

    inferred_op = alias_to_operation.get(original_name)
    if inferred_op:
        logger.debug(f"Inferred git operation '{inferred_op}' from alias '{original_name}'")
        args = dict(args)  # Copy to avoid mutation
        args["operation"] = inferred_op

    return args


def get_tool_status_message(
    tool_name: str,
    tool_args: Dict[str, Any],
    presentation: Optional["PresentationProtocol"] = None,
) -> str:
    """Generate a user-friendly status message for a tool execution.

    Provides context-aware status messages showing relevant details
    (command, path, query, etc.) for different tool types.

    Args:
        tool_name: Name of the tool being executed
        tool_args: Arguments passed to the tool
        presentation: Optional presentation adapter for icons (creates default if not provided)

    Returns:
        Status message string with icon prefix
    """
    # Get presentation adapter (lazy init for backward compatibility)
    if presentation is None:
        from victor.agent.presentation import create_presentation_adapter

        presentation = create_presentation_adapter()

    running_icon = presentation.icon("running")
    canonical_tool_name = get_canonical_name(tool_name)
    display_name = tool_name or canonical_tool_name

    if not isinstance(tool_args, dict):
        # Providers/models can emit non-dict arguments; a cosmetic status
        # message must never raise into the streaming loop.
        return f"{running_icon} Running {display_name}..."

    if canonical_tool_name == "shell":
        cmd = tool_args.get("cmd") or tool_args.get("command")
        if not cmd:
            return f"{running_icon} Running {display_name}..."
        cmd_display = cmd[:80] + "..." if len(cmd) > 80 else cmd
        return f"{running_icon} Running {display_name}: `{cmd_display}`"

    if canonical_tool_name == "ls":
        path = tool_args.get("path", ".")
        return f"{running_icon} Listing directory: {path}"

    if canonical_tool_name == "read":
        path = tool_args.get("path", "file")
        return f"{running_icon} Reading file: {path}"

    if canonical_tool_name == "edit":
        files = tool_args.get("ops") or tool_args.get("files") or tool_args.get("edits") or []
        if files and isinstance(files, list):
            paths = [f.get("path", "?") for f in files[:3] if isinstance(f, dict)]
            path_display = ", ".join(paths)
            if len(files) > 3:
                path_display += f" (+{len(files) - 3} more)"
            return f"{running_icon} Editing: {path_display}"
        return f"{running_icon} Running {display_name}..."

    if canonical_tool_name == "write":
        path = tool_args.get("path", "file")
        return f"{running_icon} Writing file: {path}"

    if tool_name == "code_search":
        query = tool_args.get("query", "")
        query_display = query[:50] + "..." if len(query) > 50 else query
        return f"{running_icon} Searching: {query_display}"

    return f"{running_icon} Running {display_name}..."


# Aliases for backward compatibility during migration
_calculate_max_context_chars = calculate_max_context_chars
_infer_git_operation = infer_git_operation
_get_tool_status_message = get_tool_status_message
