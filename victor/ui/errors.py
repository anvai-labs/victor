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

"""Error and status message utilities for CLI commands.

This module provides consistent formatting and display of error, success,
and warning messages across all CLI commands.

Usage:
    from victor.ui.errors import print_error, print_success, print_warning

    print_error("Failed to load config", details="File not found")
    print_success("Operation completed")
    print_warning("This feature is deprecated")
"""

from __future__ import annotations

import sys
from typing import Optional

from rich.console import Console
from rich.markup import escape

# Global console instance for output
_default_console = Console()


def format_error(
    message: str,
    details: Optional[str] = None,
    icon: bool = True,
) -> str:
    """Format an error message with Rich markup.

    Args:
        message: Error message to format
        details: Optional detailed error info
        icon: Whether to include error icon

    Returns:
        Formatted error string with Rich markup
    """
    escaped_msg = escape(message)
    if icon:
        formatted = f"[bold red]✗[/] [bold red]Error:[/] {escaped_msg}"
    else:
        formatted = f"[bold red]Error:[/] {escaped_msg}"

    if details:
        escaped_details = escape(details)
        formatted += f"\n[dim]{escaped_details}[/]"

    return formatted


def format_success(
    message: str,
    icon: bool = True,
) -> str:
    """Format a success message with Rich markup.

    Args:
        message: Success message to format
        icon: Whether to include success icon

    Returns:
        Formatted success string with Rich markup
    """
    escaped_msg = escape(message)
    if icon:
        return f"[green]✓[/] {escaped_msg}"
    return f"[green]{escaped_msg}[/]"


def format_warning(
    message: str,
    icon: bool = True,
) -> str:
    """Format a warning message with Rich markup.

    Args:
        message: Warning message to format
        icon: Whether to include warning icon

    Returns:
        Formatted warning string with Rich markup
    """
    escaped_msg = escape(message)
    if icon:
        return f"[yellow]⚠[/] {escaped_msg}"
    return f"[yellow]{escaped_msg}[/]"


def format_info(
    message: str,
) -> str:
    """Format an info message with Rich markup.

    Args:
        message: Info message to format

    Returns:
        Formatted info string with Rich markup
    """
    escaped_msg = escape(message)
    return f"[cyan]{escaped_msg}[/]"


def print_error(
    message: str,
    details: Optional[str] = None,
    icon: bool = True,
    console: Optional[Console] = None,
) -> None:
    """Print an error message to the console.

    Args:
        message: Error message to display
        details: Optional detailed error info
        icon: Whether to include error icon
        console: Console instance (uses default if None)
    """
    console = console or _default_console
    console.print(format_error(message, details=details, icon=icon))


def print_success(
    message: str,
    icon: bool = True,
    console: Optional[Console] = None,
) -> None:
    """Print a success message to the console.

    Args:
        message: Success message to display
        icon: Whether to include success icon
        console: Console instance (uses default if None)
    """
    console = console or _default_console
    console.print(format_success(message, icon=icon))


def print_warning(
    message: str,
    icon: bool = True,
    console: Optional[Console] = None,
) -> None:
    """Print a warning message to the console.

    Args:
        message: Warning message to display
        icon: Whether to include warning icon
        console: Console instance (uses default if None)
    """
    console = console or _default_console
    console.print(format_warning(message, icon=icon))


def print_info(
    message: str,
    console: Optional[Console] = None,
) -> None:
    """Print an info message to the console.

    Args:
        message: Info message to display
        console: Console instance (uses default if None)
    """
    console = console or _default_console
    console.print(format_info(message))


def exit_with_error(
    message: str,
    details: Optional[str] = None,
    code: int = 1,
) -> None:
    """Print an error and exit with the given code.

    This is a convenience function for commands that need to print
    an error and exit immediately.

    Args:
        message: Error message to display
        details: Optional detailed error info
        code: Exit code (default: 1)
    """
    print_error(message, details=details)
    sys.exit(code)


# Internal default console
_default_console = Console()
