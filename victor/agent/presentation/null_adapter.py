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

"""Null presentation adapter for headless/testing scenarios.

This adapter implements PresentationProtocol with minimal output,
returning plain text without any icons or Rich markup. Useful for:
- Unit tests that don't need presentation formatting
- Headless/batch processing
- CI environments
- Scenarios where clean text output is preferred

Usage:
    from victor.agent.presentation import NullPresentationAdapter

    adapter = NullPresentationAdapter()
    icon = adapter.icon("success")  # Returns ""
    status = adapter.format_status("Done", "success")  # Returns "Done"
"""

from __future__ import annotations

from victor.core.symbols import get_text_symbol


class NullPresentationAdapter:
    """Null presentation adapter that returns plain text.

    This adapter provides minimal presentation formatting, returning
    text alternatives without any Rich markup or emojis.

    Use this adapter when:
    - Writing unit tests that don't need presentation formatting
    - Running in headless/batch mode
    - Processing output for machine consumption
    - CI environments without terminal support

    Example:
        adapter = NullPresentationAdapter()

        # Returns text alternative without color
        icon = adapter.icon("success")  # "+"

        # Format status (plain text)
        msg = adapter.format_status("Done", "success")  # "+ Done"

        # Tool names without styling
        name = adapter.format_tool_name("read_file")  # "read_file"
    """

    @property
    def emojis_enabled(self) -> bool:
        """Whether emoji display is enabled.

        Always returns False for NullPresentationAdapter.

        Returns:
            False - null adapter never uses emojis.
        """
        return False

    def icon(
        self,
        name: str,
        *,
        force_emoji: bool = False,
        force_text: bool = False,
        with_color: bool = True,
    ) -> str:
        """Get an icon by name.

        Returns text alternative without any color markup.
        The force_emoji, force_text, and with_color parameters are ignored.

        Args:
            name: Icon name (e.g., "success", "error", "warning", "info")
            force_emoji: Ignored by null adapter
            force_text: Ignored by null adapter
            with_color: Ignored by null adapter

        Returns:
            Plain text alternative for the icon.

        Raises:
            KeyError: If icon name is not found.
        """
        return get_text_symbol(name)

    def format_status(self, message: str, icon_name: str) -> str:
        """Format a status message with a text icon prefix.

        Args:
            message: The status message text
            icon_name: Name of the icon to prepend

        Returns:
            Plain text formatted message like "[icon] message"
        """
        icon = self.icon(icon_name)
        return f"{icon} {message}"

    def format_tool_name(self, name: str) -> str:
        """Format a tool name for display.

        Returns the tool name without any styling.

        Args:
            name: The tool name (e.g., "read_file", "shell_exec")

        Returns:
            Plain tool name without any markup
        """
        return name


__all__ = ["NullPresentationAdapter"]
