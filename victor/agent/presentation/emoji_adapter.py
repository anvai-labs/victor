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

"""Emoji presentation adapter for agent layer.

This adapter implements PresentationProtocol with the UI-free symbol registry,
providing settings-aware emoji/icon rendering for the agent layer.

Usage:
    from victor.agent.presentation import EmojiPresentationAdapter

    adapter = EmojiPresentationAdapter()
    icon = adapter.icon("success")  # Respects use_emojis setting
    status = adapter.format_status("Done", "success")  # "[green]✓[/] Done"
"""

from __future__ import annotations

from typing import Optional

from victor.core.symbols import get_symbol

_settings_cache: Optional[object] = None


def _get_settings():
    """Get settings lazily to avoid importing configuration at module import time."""
    global _settings_cache
    if _settings_cache is None:
        try:
            from victor.config.settings import get_settings

            _settings_cache = get_settings()
        except ImportError:
            return None
    return _settings_cache


def is_emoji_enabled() -> bool:
    """Return whether emoji output is enabled for presentation rendering."""
    settings = _get_settings()
    if settings is None:
        return True
    return settings.ui.use_emojis


def reset_settings_cache() -> None:
    """Reset the settings cache for tests."""
    global _settings_cache
    _settings_cache = None


class EmojiPresentationAdapter:
    """Presentation adapter backed by the UI-free symbol registry.

    This adapter provides a clean abstraction for the agent layer to use
    icons/emojis without directly depending on the UI module.

    The adapter respects the `use_emojis` setting from configuration,
    automatically switching between emoji and text alternatives.

    Example:
        adapter = EmojiPresentationAdapter()

        # Get icon (respects settings)
        icon = adapter.icon("success")

        # Force emoji regardless of settings
        icon = adapter.icon("error", force_emoji=True)

        # Format status message
        msg = adapter.format_status("Operation complete", "success")
        # Returns: "[green]✓[/] Operation complete"
    """

    @property
    def emojis_enabled(self) -> bool:
        """Whether emoji display is enabled.

        Returns:
            True if emojis should be displayed, False for text alternatives.
        """
        return is_emoji_enabled()

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
        if force_emoji:
            use_emoji = True
        elif force_text:
            use_emoji = False
        else:
            use_emoji = is_emoji_enabled()

        return get_symbol(name, use_emoji=use_emoji, with_color=with_color)

    def format_status(self, message: str, icon_name: str) -> str:
        """Format a status message with an icon prefix.

        Args:
            message: The status message text
            icon_name: Name of the icon to prepend

        Returns:
            Formatted message like "[icon] message"
        """
        icon = self.icon(icon_name)
        return f"{icon} {message}"

    def format_tool_name(self, name: str) -> str:
        """Format a tool name for display.

        Applies consistent styling to tool names for display in the agent output.

        Args:
            name: The tool name (e.g., "read_file", "shell_exec")

        Returns:
            Formatted tool name with Rich markup styling
        """
        # Use cyan color for tool names, consistent with existing UI
        return f"[cyan]{name}[/]"


__all__ = ["EmojiPresentationAdapter", "is_emoji_enabled", "reset_settings_cache"]
