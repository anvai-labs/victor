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

"""Presentation abstraction protocols for agent layer.

This module defines the protocol for presentation/formatting concerns,
decoupling the agent layer from direct UI dependencies.

Usage:
    from victor.agent.presentation import PresentationProtocol

    class MyComponent:
        def __init__(self, presentation: PresentationProtocol):
            self._presentation = presentation

        def show_status(self):
            icon = self._presentation.icon("success")
            print(f"{icon} Operation complete")
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PresentationProtocol(Protocol):
    """Protocol for presentation/formatting concerns in the agent layer.

    This protocol abstracts away UI-specific formatting such as emoji rendering,
    allowing the agent layer to remain independent of UI implementation details.

    Implementations:
        - EmojiPresentationAdapter: Uses the UI-free core symbol registry
        - NullPresentationAdapter: Returns plain text (for testing/headless)

    Attributes:
        emojis_enabled: Whether emoji display is currently enabled
    """

    @property
    def emojis_enabled(self) -> bool:
        """Whether emoji display is enabled.

        Returns:
            True if emojis should be displayed, False for text alternatives.
        """
        ...

    def icon(
        self,
        name: str,
        *,
        force_emoji: bool = False,
        force_text: bool = False,
        with_color: bool = True,
    ) -> str:
        """Get an icon by name.

        Args:
            name: Icon name (e.g., "success", "error", "warning", "info",
                  "running", "pending", "done", "arrow_right", "bullet", etc.)
            force_emoji: Force emoji regardless of settings
            force_text: Force text regardless of settings
            with_color: Include Rich color markup

        Returns:
            The icon string, optionally with Rich color markup.

        Raises:
            KeyError: If icon name is not found.
        """
        ...

    def format_status(self, message: str, icon_name: str) -> str:
        """Format a status message with an icon prefix.

        Args:
            message: The status message text
            icon_name: Name of the icon to prepend

        Returns:
            Formatted message like "[icon] message"
        """
        ...

    def format_tool_name(self, name: str) -> str:
        """Format a tool name for display.

        Args:
            name: The tool name (e.g., "read_file", "shell_exec")

        Returns:
            Formatted tool name with appropriate styling
        """
        ...


__all__ = ["PresentationProtocol"]
