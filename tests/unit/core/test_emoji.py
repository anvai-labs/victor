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

"""Tests for the emoji utility module."""

from unittest.mock import MagicMock, patch

import pytest

from victor.ui.emoji import (
    ICONS,
    IconSet,
    Icons,
    get_icon,
    is_emoji_enabled,
    reset_settings_cache,
)

# =============================================================================
# IconSet Tests
# =============================================================================


class TestIconSet:
    """Tests for the IconSet dataclass."""

    def test_iconset_creation(self):
        """Test basic IconSet creation."""
        icon = IconSet(emoji="✓", text="+", rich_color="green")
        assert icon.emoji == "✓"
        assert icon.text == "+"
        assert icon.rich_color == "green"

    def test_iconset_get_emoji(self):
        """Test getting emoji version."""
        icon = IconSet(emoji="✓", text="+", rich_color="green")
        result = icon.get(use_emoji=True, with_color=False)
        assert result == "✓"

    def test_iconset_get_text(self):
        """Test getting text version."""
        icon = IconSet(emoji="✓", text="+", rich_color="green")
        result = icon.get(use_emoji=False, with_color=False)
        assert result == "+"

    def test_iconset_get_with_color(self):
        """Test getting icon with Rich color markup."""
        icon = IconSet(emoji="✓", text="+", rich_color="green")
        result = icon.get(use_emoji=True, with_color=True)
        assert result == "[green]✓[/]"

    def test_iconset_get_text_with_color(self):
        """Test getting text with Rich color markup."""
        icon = IconSet(emoji="✓", text="+", rich_color="green")
        result = icon.get(use_emoji=False, with_color=True)
        assert result == "[green]+[/]"

    def test_iconset_no_color(self):
        """Test IconSet without color."""
        icon = IconSet(emoji="→", text="->", rich_color=None)
        result = icon.get(use_emoji=True, with_color=True)
        assert result == "→"  # No color wrapping


# =============================================================================
# ICONS Dictionary Tests
# =============================================================================


class TestIconsDictionary:
    """Tests for the predefined ICONS dictionary."""

    def test_icons_contains_essential_icons(self):
        """Test that essential icons are defined."""
        essential = ["success", "error", "warning", "info", "running", "done"]
        for name in essential:
            assert name in ICONS, f"Missing essential icon: {name}"

    def test_all_icons_are_iconsets(self):
        """Test that all icons are IconSet instances."""
        for name, icon in ICONS.items():
            assert isinstance(icon, IconSet), f"{name} is not an IconSet"

    def test_all_icons_have_emoji_and_text(self):
        """Test that all icons have both emoji and text alternatives."""
        for name, icon in ICONS.items():
            assert icon.emoji, f"{name} missing emoji"
            assert icon.text, f"{name} missing text alternative"


# =============================================================================
# get_icon Function Tests
# =============================================================================


class TestGetIcon:
    """Tests for the get_icon function."""

    def setup_method(self):
        """Reset settings cache before each test."""
        reset_settings_cache()

    def test_get_icon_success(self):
        """Test getting success icon."""
        icon = get_icon("success", force_emoji=True)
        assert "✓" in icon

    def test_get_icon_error(self):
        """Test getting error icon."""
        icon = get_icon("error", force_emoji=True)
        assert "✗" in icon

    def test_get_icon_force_emoji(self):
        """Test forcing emoji regardless of settings."""
        icon = get_icon("success", force_emoji=True, with_color=False)
        assert icon == "✓"

    def test_get_icon_force_text(self):
        """Test forcing text regardless of settings."""
        icon = get_icon("success", force_text=True, with_color=False)
        assert icon == "+"

    def test_get_icon_with_color(self):
        """Test getting icon with Rich color markup."""
        icon = get_icon("success", force_emoji=True, with_color=True)
        assert icon == "[green]✓[/]"

    def test_get_icon_without_color(self):
        """Test getting icon without Rich color markup."""
        icon = get_icon("success", force_emoji=True, with_color=False)
        assert icon == "✓"

    def test_get_icon_unknown_raises(self):
        """Test that unknown icon name raises KeyError."""
        with pytest.raises(KeyError) as excinfo:
            get_icon("nonexistent_icon")
        assert "Unknown icon" in str(excinfo.value)

    def test_get_icon_respects_settings_enabled(self):
        """Test that get_icon respects use_emojis=True setting."""
        mock_settings = MagicMock()
        mock_settings.ui.use_emojis = True

        with patch("victor.ui.emoji._get_settings", return_value=mock_settings):
            reset_settings_cache()
            icon = get_icon("success", with_color=False)
            assert icon == "✓"

    def test_get_icon_respects_settings_disabled(self):
        """Test that get_icon respects use_emojis=False setting."""
        mock_settings = MagicMock()
        mock_settings.ui.use_emojis = False

        with patch("victor.ui.emoji._get_settings", return_value=mock_settings):
            reset_settings_cache()
            # Need to clear the cache and re-mock
            import victor.ui.emoji

            victor.ui.emoji._settings_cache = mock_settings

            icon = get_icon("success", with_color=False)
            assert icon == "+"


# =============================================================================
# Icons Class Tests
# =============================================================================


class TestIconsClass:
    """Tests for the Icons class interface."""

    def setup_method(self):
        """Reset settings cache before each test."""
        reset_settings_cache()

    def test_icons_success(self):
        """Test Icons.success()."""
        icon = Icons.success(force_emoji=True)
        assert "[green]✓[/]" == icon

    def test_icons_error(self):
        """Test Icons.error()."""
        icon = Icons.error(force_emoji=True)
        assert "[red]✗[/]" == icon

    def test_icons_warning(self):
        """Test Icons.warning()."""
        icon = Icons.warning(force_emoji=True)
        assert "[yellow]⚠[/]" == icon

    def test_icons_info(self):
        """Test Icons.info()."""
        icon = Icons.info(force_emoji=True)
        assert "[blue]ℹ[/]" == icon

    def test_icons_running(self):
        """Test Icons.running()."""
        icon = Icons.running(force_emoji=True)
        assert "[cyan]🔧[/]" == icon

    def test_icons_done(self):
        """Test Icons.done()."""
        icon = Icons.done(force_emoji=True)
        assert "[green]✔[/]" == icon

    def test_icons_force_text(self):
        """Test forcing text alternative."""
        icon = Icons.success(force_text=True)
        assert "[green]+[/]" == icon

    def test_icons_error_force_text(self):
        """Test error icon force text."""
        icon = Icons.error(force_text=True)
        assert "[red]x[/]" == icon


# =============================================================================
# is_emoji_enabled Function Tests
# =============================================================================


class TestIsEmojiEnabled:
    """Tests for the is_emoji_enabled function."""

    def setup_method(self):
        """Reset settings cache before each test."""
        reset_settings_cache()

    def test_is_emoji_enabled_default(self):
        """Test default emoji enabled state."""
        # When settings can't be loaded, should default to True
        with patch("victor.ui.emoji._get_settings", return_value=None):
            reset_settings_cache()
            assert is_emoji_enabled() is True

    def test_is_emoji_enabled_from_settings_true(self):
        """Test emoji enabled from settings."""
        mock_settings = MagicMock()
        mock_settings.ui.use_emojis = True

        import victor.ui.emoji

        victor.ui.emoji._settings_cache = mock_settings

        assert is_emoji_enabled() is True

    def test_is_emoji_enabled_from_settings_false(self):
        """Test emoji disabled from settings."""
        mock_settings = MagicMock()
        mock_settings.ui.use_emojis = False

        import victor.ui.emoji

        victor.ui.emoji._settings_cache = mock_settings

        assert is_emoji_enabled() is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestEmojiIntegration:
    """Integration tests for emoji module."""

    def setup_method(self):
        """Reset settings cache before each test."""
        reset_settings_cache()

    def test_all_icons_have_distinct_text_alternatives(self):
        """Test that text alternatives are usable (not empty or whitespace)."""
        for name, icon in ICONS.items():
            text = icon.text.strip()
            assert text, f"{name} has empty/whitespace text alternative"

    def test_icons_can_be_printed(self):
        """Test that all icons can be safely converted to strings."""
        for name in ICONS:
            # Should not raise any encoding errors
            emoji_str = get_icon(name, force_emoji=True, with_color=False)
            text_str = get_icon(name, force_text=True, with_color=False)
            assert isinstance(emoji_str, str)
            assert isinstance(text_str, str)


# =============================================================================
# Marker Tests
# =============================================================================


@pytest.mark.unit
class TestEmojiMarker:
    """Unit tests marked for CI."""

    def test_emoji_module_import(self):
        """Test that emoji module can be imported."""
        from victor.ui.emoji import Icons, get_icon

        assert Icons is not None
        assert get_icon is not None

    def test_basic_functionality(self):
        """Test basic emoji functionality."""
        icon = get_icon("success", force_emoji=True, with_color=False)
        assert icon == "✓"
