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

"""Argument sanitization for tool calls.

This module provides utilities to sanitize tool arguments before JSON serialization,
handling non-JSON-serializable objects that may appear in LLM-generated tool calls.
"""

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def sanitize_arguments_for_serialization(
    arguments: Dict[str, Any],
    replace_ellipsis: bool = True,
) -> Dict[str, Any]:
    """
    Sanitize tool arguments to ensure JSON serializability.

    Removes or replaces non-JSON-serializable objects that may appear
    in tool arguments before JSON serialization.

    This function handles common edge cases from LLM-generated tool calls:
    - Ellipsis (...) objects from Python placeholder code
    - Path objects from file system operations
    - Functions/lambdas passed as arguments
    - Nested dictionaries and lists containing non-serializable objects

    Args:
        arguments: Raw tool arguments from LLM
        replace_ellipsis: If True, replace Ellipsis with None; if False, remove the key

    Returns:
        Sanitized arguments dict that is JSON-serializable

    Examples:
        >>> sanitize_arguments_for_serialization({"path": ..., "limit": 10})
        {"path": None, "limit": 10}

        >>> sanitize_arguments_for_serialization({"file": Path("/tmp/test")})
        {"file": "/tmp/test"}

        >>> sanitize_arguments_for_serialization({"nested": {"key": ...}})
        {"nested": {"key": None}}
    """
    sanitized = {}

    for key, value in arguments.items():
        # Handle Ellipsis (...)
        if replace_ellipsis and value is ...:
            sanitized[key] = None
            logger.debug(f"Sanitized Ellipsis in argument '{key}' -> None")
            continue

        # Handle Path objects (convert to string)
        if hasattr(value, "__fspath__"):
            sanitized[key] = str(value)
            logger.debug(f"Sanitized Path object in argument '{key}' -> {str(value)}")
            continue

        # Handle functions/lambdas (convert to descriptive string)
        if isinstance(value, type(lambda: None)):
            sanitized[key] = f"<function: {getattr(value, '__name__', 'lambda')}>"
            logger.debug(f"Sanitized function in argument '{key}'")
            continue

        # Handle nested dicts recursively
        if isinstance(value, dict):
            sanitized[key] = sanitize_arguments_for_serialization(value, replace_ellipsis)
            continue

        # Handle lists recursively (check each element)
        if isinstance(value, list):
            sanitized_list = []
            for item in value:
                if isinstance(item, dict):
                    # Recursively sanitize dict items in list
                    sanitized_list.append(
                        sanitize_arguments_for_serialization(item, replace_ellipsis)
                    )
                elif item is ...:
                    # Handle ellipsis in lists
                    sanitized_list.append(None if replace_ellipsis else item)
                elif hasattr(item, "__fspath__"):
                    # Handle Path objects in lists
                    sanitized_list.append(str(item))
                elif isinstance(item, type(lambda: None)):
                    # Handle functions in lists
                    sanitized_list.append(f"<function: {getattr(item, '__name__', 'lambda')}>")
                else:
                    # Keep other items as-is
                    sanitized_list.append(item)
            sanitized[key] = sanitized_list
            continue

        # Keep other values as-is (already JSON-serializable)
        sanitized[key] = value

    return sanitized
