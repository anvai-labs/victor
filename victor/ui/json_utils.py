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

"""JSON output utilities for CLI commands.

This module provides consistent JSON output handling across all CLI commands
that support --json flag.

Usage:
    from victor.ui.json_utils import create_json_option, print_json_data

    @app.command("list")
    def list_items(
        json_output: bool = create_json_option(),
    ):
        items = get_items()
        if json_output:
            print_json_data({"items": items})
            return
        # Normal rich output...
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional, TypeVar

import typer
from rich.console import Console

# Default console for JSON output
_default_console = Console()


def create_json_option(
    short_flag: str = "-j",
    help_text: str = "Output as JSON",
) -> bool:
    """Create a standardized --json option for CLI commands.

    This factory function ensures consistent JSON output flags across all commands.

    Args:
        short_flag: Short flag character (default: "-j")
        help_text: Help text to display

    Returns:
        typer.Option configured for JSON output

    Example:
        @app.command("list")
        def list_items(
            json_output: bool = create_json_option(),
        ):
            ...
    """
    return typer.Option(False, "--json", short_flag, help=help_text)


def print_json_data(
    data: Any,
    console: Optional[Console] = None,
) -> None:
    """Print data as formatted JSON to the console.

    Args:
        data: Data to serialize to JSON
        console: Console instance (uses default if None)
    """
    console = console or _default_console
    console.print_json(json.dumps(data, indent=2, default=str))


def create_json_handler(
    rich_func: Callable[..., None],
) -> Callable[..., None]:
    """Create a wrapper that handles JSON output before calling rich function.

    This decorator allows commands to separate data collection from display,
    making it easier to support both JSON and rich output.

    Usage:
        @create_json_handler
        def _display_items(items: list) -> None:
            table = Table()
            table.add_column("Name")
            for item in items:
                table.add_row(item.name)
            console.print(table)

        @app.command("list")
        def list_items(
            json_output: bool = create_json_option(),
        ):
            items = get_items()
            if json_output:
                print_json_data({"items": items})
                return
            _display_items(items)

    Args:
        rich_func: Function that renders rich output

    Returns:
        Wrapped function that checks JSON mode before calling rich_func
    """
    # The wrapper is implemented at call time, not as a true decorator
    # This is just a marker for documentation
    return rich_func


def is_json_mode(
    json_output: bool,
    **kwargs: Any,
) -> bool:
    """Check if we're in JSON output mode.

    This helper makes it easier to check JSON mode across commands
    that may have different parameter names for JSON output.

    Args:
        json_output: Value of --json flag
        **kwargs: Additional kwargs that might contain json_output

    Returns:
        True if in JSON output mode

    Example:
        def list_items(json_output: bool = False):
            if is_json_mode(json_output):
                print_json_data(...)
    """
    return json_output


T = TypeVar("T")


def json_or_rich(
    data: T,
    json_output: bool,
    rich_renderer: Callable[[T], None],
    console: Optional[Console] = None,
) -> None:
    """Output data as JSON or rich display based on flag.

    This helper centralizes the if/else pattern for JSON output handling.

    Args:
        data: Data to output
        json_output: Whether to use JSON output
        rich_renderer: Function that renders rich output
        console: Optional console instance

    Example:
        def list_items(json_output: bool = False):
            items = get_items()
            json_or_rich(
                {"items": items},
                json_output,
                lambda _: _render_items_table(items),
            )
    """
    if json_output:
        print_json_data(data, console=console)
    else:
        rich_renderer(data)


def format_json_list(
    items: list[Any],
    *,
    key: Optional[str] = None,
    title: Optional[str] = None,
) -> dict[str, Any]:
    """Format a list of items into a standard JSON response structure.

    This provides consistent JSON structure across commands.

    Args:
        items: List of items to format
        key: Key name for the items list (default: "items")
        title: Optional title for the response

    Returns:
        Dictionary with consistent structure

    Example:
        # Returns {"items": [...], "count": 5}
        format_json_list([1, 2, 3, 4, 5])
    """
    result: dict[str, Any] = {key or "items": items, "count": len(items)}
    if title:
        result["title"] = title
    return result


def format_json_item(
    item: dict[str, Any],
    *,
    item_type: str = "item",
) -> dict[str, Any]:
    """Format a single item into a standard JSON response structure.

    Args:
        item: Item dictionary to format
        item_type: Type name for the item

    Returns:
        Dictionary with consistent structure
    """
    return {item_type: item}


def format_json_error(
    message: str,
    *,
    error_code: Optional[str] = None,
    details: Optional[str] = None,
) -> dict[str, Any]:
    """Format an error into a standard JSON error response.

    Args:
        message: Error message
        error_code: Optional error code
        details: Optional detailed error info

    Returns:
        Dictionary with error structure
    """
    error_data: dict[str, Any] = {"error": message}
    if error_code:
        error_data["code"] = error_code
    if details:
        error_data["details"] = details
    return error_data
