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

"""Tests for presentation adapters.

Tests the PresentationProtocol implementations:
- EmojiPresentationAdapter: Settings-aware icon rendering
- NullPresentationAdapter: Plain text for testing/headless
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from victor.agent.presentation import (
    PresentationProtocol,
    EmojiPresentationAdapter,
    NullPresentationAdapter,
    create_presentation_adapter,
)


class TestPresentationProtocol:
    """Test protocol compliance and factory function."""

    def test_emoji_adapter_implements_protocol(self):
        """EmojiPresentationAdapter should implement PresentationProtocol."""
        adapter = EmojiPresentationAdapter()
        assert isinstance(adapter, PresentationProtocol)

    def test_null_adapter_implements_protocol(self):
        """NullPresentationAdapter should implement PresentationProtocol."""
        adapter = NullPresentationAdapter()
        assert isinstance(adapter, PresentationProtocol)

    def test_create_presentation_adapter_default(self):
        """create_presentation_adapter() should return EmojiPresentationAdapter by default."""
        adapter = create_presentation_adapter()
        assert isinstance(adapter, EmojiPresentationAdapter)

    def test_create_presentation_adapter_null(self):
        """create_presentation_adapter(use_null=True) should return NullPresentationAdapter."""
        adapter = create_presentation_adapter(use_null=True)
        assert isinstance(adapter, NullPresentationAdapter)


class TestEmojiPresentationAdapter:
    """Tests for EmojiPresentationAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create an EmojiPresentationAdapter instance."""
        return EmojiPresentationAdapter()

    def test_icon_success(self, adapter):
        """icon('success') should return a success icon."""
        icon = adapter.icon("success")
        # Should contain either checkmark emoji or + text
        assert icon  # Not empty
        # Should have color markup
        assert "[green]" in icon or "+" in icon or "✓" in icon

    def test_icon_error(self, adapter):
        """icon('error') should return an error icon."""
        icon = adapter.icon("error")
        assert icon
        assert "[red]" in icon or "x" in icon or "✗" in icon

    def test_icon_warning(self, adapter):
        """icon('warning') should return a warning icon."""
        icon = adapter.icon("warning")
        assert icon
        assert "[yellow]" in icon or "!" in icon or "⚠" in icon

    def test_icon_info(self, adapter):
        """icon('info') should return an info icon."""
        icon = adapter.icon("info")
        assert icon
        assert "[blue]" in icon or "i" in icon or "ℹ" in icon

    def test_icon_running(self, adapter):
        """icon('running') should return a running icon."""
        icon = adapter.icon("running")
        assert icon
        assert "[cyan]" in icon or "*" in icon or "🔧" in icon

    def test_icon_unknown_raises(self, adapter):
        """icon() with unknown name should raise KeyError."""
        with pytest.raises(KeyError) as exc_info:
            adapter.icon("nonexistent_icon")
        assert "Unknown icon" in str(exc_info.value)

    def test_icon_force_emoji(self, adapter):
        """icon() with force_emoji=True should return emoji."""
        icon = adapter.icon("success", force_emoji=True)
        # Should contain the emoji checkmark
        assert "✓" in icon

    def test_icon_force_text(self, adapter):
        """icon() with force_text=True should return text alternative."""
        icon = adapter.icon("success", force_text=True)
        # Should contain the text alternative
        assert "+" in icon

    def test_icon_without_color(self, adapter):
        """icon() with with_color=False should not have color markup."""
        icon = adapter.icon("success", with_color=False)
        # Should not have Rich color markup
        assert "[green]" not in icon
        assert "[/]" not in icon

    def test_format_status(self, adapter):
        """format_status() should combine icon and message."""
        status = adapter.format_status("Done", "success")
        # Should have icon followed by space and message
        assert "Done" in status
        # Should have some icon (emoji or text)
        assert len(status) > len("Done")

    def test_format_tool_name(self, adapter):
        """format_tool_name() should add cyan styling."""
        formatted = adapter.format_tool_name("read_file")
        assert "[cyan]read_file[/]" == formatted

    @patch("victor.agent.presentation.emoji_adapter.is_emoji_enabled")
    def test_emojis_enabled_true(self, mock_enabled, adapter):
        """emojis_enabled should return True when setting is True."""
        mock_enabled.return_value = True
        assert adapter.emojis_enabled is True

    @patch("victor.agent.presentation.emoji_adapter.is_emoji_enabled")
    def test_emojis_enabled_false(self, mock_enabled, adapter):
        """emojis_enabled should return False when setting is False."""
        mock_enabled.return_value = False
        assert adapter.emojis_enabled is False


class TestNullPresentationAdapter:
    """Tests for NullPresentationAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create a NullPresentationAdapter instance."""
        return NullPresentationAdapter()

    def test_emojis_enabled_always_false(self, adapter):
        """emojis_enabled should always return False."""
        assert adapter.emojis_enabled is False

    def test_icon_success(self, adapter):
        """icon('success') should return text alternative."""
        icon = adapter.icon("success")
        assert icon == "+"

    def test_icon_error(self, adapter):
        """icon('error') should return text alternative."""
        icon = adapter.icon("error")
        assert icon == "x"

    def test_icon_warning(self, adapter):
        """icon('warning') should return text alternative."""
        icon = adapter.icon("warning")
        assert icon == "!"

    def test_icon_info(self, adapter):
        """icon('info') should return text alternative."""
        icon = adapter.icon("info")
        assert icon == "i"

    def test_icon_running(self, adapter):
        """icon('running') should return text alternative."""
        icon = adapter.icon("running")
        assert icon == "*"

    def test_icon_pending(self, adapter):
        """icon('pending') should return text alternative."""
        icon = adapter.icon("pending")
        assert icon == "..."

    def test_icon_done(self, adapter):
        """icon('done') should return text alternative."""
        icon = adapter.icon("done")
        assert icon == "OK"

    def test_icon_arrow_right(self, adapter):
        """icon('arrow_right') should return text alternative."""
        icon = adapter.icon("arrow_right")
        assert icon == "->"

    def test_icon_unknown_raises(self, adapter):
        """icon() with unknown name should raise KeyError."""
        with pytest.raises(KeyError) as exc_info:
            adapter.icon("nonexistent_icon")
        assert "Unknown icon" in str(exc_info.value)

    def test_icon_ignores_force_emoji(self, adapter):
        """icon() should ignore force_emoji parameter."""
        icon = adapter.icon("success", force_emoji=True)
        # Should still return text alternative
        assert icon == "+"

    def test_icon_ignores_with_color(self, adapter):
        """icon() should ignore with_color parameter."""
        icon = adapter.icon("success", with_color=True)
        # Should return plain text without color markup
        assert icon == "+"
        assert "[" not in icon

    def test_format_status(self, adapter):
        """format_status() should combine icon and message."""
        status = adapter.format_status("Done", "success")
        assert status == "+ Done"

    def test_format_tool_name_no_styling(self, adapter):
        """format_tool_name() should return plain text."""
        formatted = adapter.format_tool_name("read_file")
        assert formatted == "read_file"
        assert "[" not in formatted


class TestAllIconNames:
    """Test that both adapters support the same icon names."""

    ICON_NAMES = [
        "success",
        "error",
        "warning",
        "info",
        "running",
        "pending",
        "done",
        "arrow_right",
        "arrow_left",
        "bullet",
        "file",
        "folder",
        "sparkle",
        "search",
        "chart",
        "target",
        "rocket",
        "bolt",
        "bulb",
        "note",
        "refresh",
        # Safety/risk level icons
        "risk_high",
        "risk_critical",
        "unknown",
        # Step status icons
        "skipped",
        "blocked",
        # Additional UI icons
        "thinking",
        "gear",
        "clipboard",
        "stop",
        "clock",
        "stop_sign",
        # Severity/complexity indicators
        "level_low",
        "level_medium",
        "level_high",
        "level_critical",
        "level_info",
        "level_unknown",
        "person",
    ]

    @pytest.fixture
    def emoji_adapter(self):
        """Create an EmojiPresentationAdapter instance."""
        return EmojiPresentationAdapter()

    @pytest.fixture
    def null_adapter(self):
        """Create a NullPresentationAdapter instance."""
        return NullPresentationAdapter()

    @pytest.mark.parametrize("icon_name", ICON_NAMES)
    def test_emoji_adapter_supports_icon(self, emoji_adapter, icon_name):
        """EmojiPresentationAdapter should support all standard icons."""
        icon = emoji_adapter.icon(icon_name)
        assert icon  # Not empty

    @pytest.mark.parametrize("icon_name", ICON_NAMES)
    def test_null_adapter_supports_icon(self, null_adapter, icon_name):
        """NullPresentationAdapter should support all standard icons."""
        icon = null_adapter.icon(icon_name)
        assert icon  # Not empty


class TestEmojiAdapterLazyImport:
    """Test that EmojiPresentationAdapter uses lazy imports and is resilient."""

    def test_adapter_can_be_instantiated(self):
        """EmojiPresentationAdapter can be instantiated without triggering imports."""
        from victor.agent.presentation.emoji_adapter import EmojiPresentationAdapter

        adapter = EmojiPresentationAdapter()
        assert adapter is not None

    def test_adapter_methods_work_when_symbols_available(self):
        """Methods work correctly when the core symbol registry is available."""
        from victor.agent.presentation.emoji_adapter import EmojiPresentationAdapter

        adapter = EmojiPresentationAdapter()
        icon = adapter.icon("success")
        assert icon
        enabled = adapter.emojis_enabled
        assert isinstance(enabled, bool)

    def test_module_getattr_provides_class(self):
        """Module __getattr__ provides EmojiPresentationAdapter lazily."""
        import sys

        # Use sys.modules to avoid re-import confusion from mirrored test directory
        # structure (tests/unit/agent/ shadows 'agent' in sys.path when imported fresh)
        mod = sys.modules["victor.agent.presentation"]
        cls = mod.EmojiPresentationAdapter
        assert cls.__name__ == "EmojiPresentationAdapter"

    def test_module_getattr_raises_for_unknown(self):
        """Module __getattr__ raises AttributeError for unknown names."""
        import sys

        mod = sys.modules["victor.agent.presentation"]
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = mod.NonExistentClass
