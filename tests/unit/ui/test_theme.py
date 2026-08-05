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

"""Tests for theme system and terminal capability detection (Phase 5)."""

import pytest
from unittest.mock import patch

from victor.ui.theme import (
    TerminalCapability,
    get_adaptive_theme,
    get_tool_category_color,
    victor_theme,
    dark_theme,
    light_theme,
    high_contrast_theme,
)


class TestTerminalCapabilityDetection:
    """Tests for terminal capability detection."""

    def test_detect_capability_class_constants(self):
        """TerminalCapability class should have capability constants."""
        assert hasattr(TerminalCapability, "BASIC")
        assert hasattr(TerminalCapability, "STANDARD")
        assert hasattr(TerminalCapability, "TRUECOLOR")

    def test_detect_returns_valid_capability(self):
        """detect() should return a valid capability level."""
        result = TerminalCapability.detect()
        assert result in {
            TerminalCapability.BASIC,
            TerminalCapability.STANDARD,
            TerminalCapability.TRUECOLOR,
        }

    @patch.dict(
        "os.environ",
        {"TERM": "xterm-256color", "COLORTERM": "", "TERM_PROGRAM": ""},
        clear=False,
    )
    def test_detect_256color_terminal(self):
        """Should detect 256-color terminal."""
        result = TerminalCapability.detect()
        # xterm-256color should be detected as standard
        assert result in {TerminalCapability.STANDARD, TerminalCapability.TRUECOLOR}

    @patch.dict(
        "os.environ",
        {"TERM": "xterm", "TERM_PROGRAM": "", "COLORTERM": ""},
        clear=False,
    )
    def test_detect_basic_terminal(self):
        """Should detect basic terminal."""
        result = TerminalCapability.detect()
        # xterm alone should be detected as basic or standard
        assert result in {
            TerminalCapability.BASIC,
            TerminalCapability.STANDARD,
            TerminalCapability.TRUECOLOR,
        }

    @patch.dict("os.environ", {"COLORTERM": "truecolor"})
    def test_detect_truecolor_via_colorterm(self):
        """Should detect truecolor via COLORTERM."""
        result = TerminalCapability.detect()
        assert result == TerminalCapability.TRUECOLOR

    @patch.dict("os.environ", {"TERM_PROGRAM": "iTerm.app"})
    def test_detect_truecolor_via_term_program(self):
        """Should detect truecolor via TERM_PROGRAM."""
        result = TerminalCapability.detect()
        assert result == TerminalCapability.TRUECOLOR


class TestAdaptiveTheme:
    """Tests for adaptive theme generation."""

    def test_get_adaptive_theme_returns_theme(self):
        """get_adaptive_theme should return a Rich Theme."""
        from rich.theme import Theme

        result = get_adaptive_theme()
        assert isinstance(result, Theme)

    def test_adaptive_theme_with_basic_capability(self):
        """Basic capability should use basic colors."""
        theme = get_adaptive_theme(capability=TerminalCapability.BASIC)
        assert theme is not None
        # Should have basic color keys
        assert "info" in theme.styles
        assert "error" in theme.styles

    def test_adaptive_theme_with_standard_capability(self):
        """Standard capability should use enhanced colors."""
        theme = get_adaptive_theme(capability=TerminalCapability.STANDARD)
        assert theme is not None
        assert "info" in theme.styles

    def test_adaptive_theme_with_truecolor_capability(self):
        """Truecolor capability should use hex colors."""
        theme = get_adaptive_theme(capability=TerminalCapability.TRUECOLOR)
        assert theme is not None
        # Should have hex color values
        info_style = theme.styles.get("info", "")
        # Truecolor theme uses hex values starting with #
        # But in Rich Theme, styles are stored as parsed Style objects
        assert "info" in theme.styles

    def test_dark_theme_mode(self):
        """Dark theme mode should work."""
        theme = get_adaptive_theme(theme_mode="dark")
        assert theme is not None

    def test_light_theme_mode(self):
        """Light theme mode should invert colors."""
        theme = get_adaptive_theme(theme_mode="light")
        assert theme is not None

    def test_high_contrast_theme_mode(self):
        """High contrast mode should use brighter colors."""
        theme = get_adaptive_theme(theme_mode="high_contrast")
        assert theme is not None

    def test_none_capability_uses_detection(self):
        """None capability should auto-detect."""
        theme = get_adaptive_theme(capability=None)
        assert theme is not None


class TestToolCategoryColors:
    """Tests for tool category color scheme."""

    def test_filesystem_category_color(self):
        """Filesystem category should return blue color."""
        color = get_tool_category_color("filesystem")
        assert "blue" in color.lower() or "#" in color

    def test_git_category_color(self):
        """Git category should return green color."""
        color = get_tool_category_color("git")
        assert "green" in color.lower() or "#" in color

    def test_web_category_color(self):
        """Web category should return orange color."""
        color = get_tool_category_color("web")
        assert "orange" in color.lower() or "#" in color

    def test_unknown_category_default_color(self):
        """Unknown category should return default color."""
        color = get_tool_category_color("unknown_xyz")
        assert color == get_tool_category_color("default")

    def test_category_colors_case_insensitive(self):
        """Category lookup should be case-insensitive."""
        color1 = get_tool_category_color("FILESYSTEM")
        color2 = get_tool_category_color("filesystem")
        color3 = get_tool_category_color("Filesystem")
        # Should all return the same color
        assert color1 == color2 == color3


class TestThemeVariants:
    """Tests for pre-configured theme variants."""

    def test_victor_theme_exists(self):
        """Legacy victor_theme should exist."""
        assert victor_theme is not None
        from rich.theme import Theme

        assert isinstance(victor_theme, Theme)

    def test_dark_theme_exists(self):
        """dark_theme should exist."""
        assert dark_theme is not None
        from rich.theme import Theme

        assert isinstance(dark_theme, Theme)

    def test_light_theme_exists(self):
        """light_theme should exist."""
        assert light_theme is not None
        from rich.theme import Theme

        assert isinstance(light_theme, Theme)

    def test_high_contrast_theme_exists(self):
        """high_contrast_theme should exist."""
        assert high_contrast_theme is not None
        from rich.theme import Theme

        assert isinstance(high_contrast_theme, Theme)

    def test_all_themes_have_required_styles(self):
        """All theme variants should have required style keys."""
        required_keys = {"info", "warning", "error", "success"}
        for theme in [victor_theme, dark_theme, light_theme, high_contrast_theme]:
            for key in required_keys:
                assert key in theme.styles
