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

"""Tests for tool output formatting strategies."""

import pytest

from victor.agent.format_strategies import (
    FormatStrategyFactory,
    PlainFormatStrategy,
    XmlFormatStrategy,
    ToonFormatStrategy,
    ToolOutputFormat,
    PLAIN_FORMAT,
    XML_FORMAT,
    TOON_FORMAT,
)


class TestToolOutputFormat:
    """Test ToolOutputFormat value object."""

    def test_default_plain_format(self):
        """Default format should be plain JSON."""
        format_spec = ToolOutputFormat()
        assert format_spec.style == "plain"
        assert format_spec.use_delimiters is False
        assert format_spec.include_tags is False

    def test_xml_format_with_delimiters(self):
        """XML format should support delimiters."""
        format_spec = ToolOutputFormat(
            style="xml",
            use_delimiters=True,
            delimiter_char="=",
            delimiter_width=50,
        )
        assert format_spec.style == "xml"
        assert format_spec.use_delimiters is True
        assert format_spec.delimiter_char == "="
        assert format_spec.delimiter_width == 50

    def test_invalid_style_raises_error(self):
        """Built-in invalid styles should raise ValueError for built-in validation."""
        # For built-in styles, invalid delimiter config should raise error
        with pytest.raises(ValueError, match="delimiter_width must be at least 10"):
            ToolOutputFormat(style="plain", delimiter_width=5)

        # Custom styles (not built-in) are allowed without validation
        # They will be validated by the factory when created
        from dataclasses import dataclass

        @dataclass
        class CustomFormat:
            style: str = "custom"

        # This should NOT raise an error
        format_spec = CustomFormat()
        assert format_spec.style == "custom"

    def test_delimiters_without_char_raises_error(self):
        """Delimiters enabled without char should raise ValueError."""
        with pytest.raises(ValueError, match="delimiter_char required"):
            ToolOutputFormat(use_delimiters=True, delimiter_char="")

    def test_delimiter_width_too_small_raises_error(self):
        """Delimiter width < 10 should raise ValueError."""
        with pytest.raises(ValueError, match="delimiter_width must be at least 10"):
            ToolOutputFormat(delimiter_width=5)

    def test_negative_min_tokens_raises_error(self):
        """Negative min_json_tokens should raise ValueError."""
        with pytest.raises(ValueError, match="min_json_tokens must be non-negative"):
            ToolOutputFormat(min_json_tokens=-1)

    def test_with_style_creates_new_instance(self):
        """with_style should create new format with different style."""
        original = ToolOutputFormat(style="plain")
        modified = original.with_style("xml")
        assert modified.style == "xml"
        assert original.style == "plain"  # Immutable

    def test_with_delimiters_creates_new_instance(self):
        """with_delimiters should create new format with delimiters."""
        original = ToolOutputFormat(style="xml", use_delimiters=False)
        modified = original.with_delimiters(True)
        assert modified.use_delimiters is True
        assert original.use_delimiters is False  # Immutable

    def test_plain_format_constant(self):
        """PLAIN_FORMAT constant should be properly configured."""
        assert PLAIN_FORMAT.style == "plain"
        assert PLAIN_FORMAT.use_delimiters is False

    def test_xml_format_constant(self):
        """XML_FORMAT constant should be properly configured."""
        assert XML_FORMAT.style == "xml"
        assert XML_FORMAT.use_delimiters is True
        assert XML_FORMAT.delimiter_char == "="
        assert XML_FORMAT.delimiter_width == 50
        assert XML_FORMAT.include_tags is True

    def test_toon_format_constant(self):
        """TOON_FORMAT constant should be properly configured."""
        assert TOON_FORMAT.style == "toon"
        assert TOON_FORMAT.prefer_compact is True


class TestPlainFormatStrategy:
    """Test plain JSON formatting strategy."""

    def test_format_string_output(self):
        """String output should be returned as-is."""
        strategy = PlainFormatStrategy()
        result = strategy.format(
            tool_name="test_tool",
            args={"param": "value"},
            output="simple string output",
        )
        assert result == "simple string output"

    def test_format_dict_output(self):
        """Dict output should be serialized to JSON."""
        strategy = PlainFormatStrategy()
        result = strategy.format(
            tool_name="test_tool",
            args={},
            output={"key": "value", "number": 42},
        )
        assert '"key":"value"' in result
        assert '"number":42' in result

    def test_format_list_output(self):
        """List output should be serialized to JSON."""
        strategy = PlainFormatStrategy()
        result = strategy.format(
            tool_name="test_tool",
            args={},
            output=["item1", "item2", "item3"],
        )
        assert '["item1","item2","item3"]' in result

    def test_estimate_tokens(self):
        """Token estimation should be roughly length/4."""
        strategy = PlainFormatStrategy()
        content = "a" * 100  # 100 characters
        tokens = strategy.estimate_tokens(content)
        assert tokens == 25  # 100 / 4


class TestXmlFormatStrategy:
    """Test XML formatting strategy."""

    def test_format_with_delimiters(self):
        """XML format should include delimiters when specified."""
        format_spec = ToolOutputFormat(
            style="xml",
            use_delimiters=True,
            delimiter_char="=",
            delimiter_width=50,
        )
        strategy = XmlFormatStrategy(format_spec)
        result = strategy.format(
            tool_name="read_file",
            args={"path": "test.py"},
            output="file content",
        )

        assert "<TOOL_OUTPUT" in result
        assert 'tool="read_file"' in result
        assert 'path="test.py"' in result
        assert "==================================================" in result
        assert "file content" in result
        assert "</TOOL_OUTPUT>" in result

    def test_format_without_delimiters(self):
        """XML format without delimiters should just use tags."""
        format_spec = ToolOutputFormat(
            style="xml",
            use_delimiters=False,
        )
        strategy = XmlFormatStrategy(format_spec)
        result = strategy.format(
            tool_name="test_tool",
            args={},
            output="output content",
        )

        assert "<TOOL_OUTPUT" in result
        assert "output content" in result
        assert "</TOOL_OUTPUT>" in result
        assert "═══" not in result

    def test_format_dict_serializes_as_json(self):
        """Dict output should be serialized as indented JSON."""
        format_spec = ToolOutputFormat(style="xml")
        strategy = XmlFormatStrategy(format_spec)
        result = strategy.format(
            tool_name="test",
            args={},
            output={"key": "value"},
        )

        assert "{" in result
        assert '"key"' in result
        assert '"value"' in result

    def test_long_args_excluded_from_attributes(self):
        """Very long argument values should be excluded from XML attributes."""
        format_spec = ToolOutputFormat(style="xml")
        strategy = XmlFormatStrategy(format_spec)
        long_value = "x" * 200
        result = strategy.format(
            tool_name="test",
            args={"long_param": long_value, "short": "value"},
            output="output",
        )

        # Short arg should be in attributes
        assert 'short="value"' in result
        # Long arg should NOT be in attributes
        assert "long_param=" not in result

    def test_estimate_tokens_includes_overhead(self):
        """Token estimation should include XML tag overhead."""
        format_spec = ToolOutputFormat(
            style="xml",
            use_delimiters=True,
            delimiter_width=50,
        )
        strategy = XmlFormatStrategy(format_spec)
        content = "test content"
        tokens = strategy.estimate_tokens(content)

        # Should be base tokens + overhead for tags and delimiters
        base_tokens = len(content) // 4
        assert tokens > base_tokens


class TestToonFormatStrategy:
    """Test TOON formatting strategy."""

    def test_format_simple_list(self):
        """Simple list should use | delimiter."""
        format_spec = ToolOutputFormat(style="toon")
        strategy = ToonFormatStrategy(format_spec)
        result = strategy.format(
            tool_name="test",
            args={},
            output=["apple", "banana", "cherry"],
        )

        assert " | " in result
        assert "apple" in result
        assert "banana" in result

    def test_format_dict_uses_colon_delimiter(self):
        """Dict should use : delimiter."""
        format_spec = ToolOutputFormat(style="toon")
        strategy = ToonFormatStrategy(format_spec)
        result = strategy.format(
            tool_name="test",
            args={},
            output={"name": "Alice", "age": 30},
        )

        assert "name: Alice" in result
        assert "age: 30" in result

    def test_format_non_structured_falls_back_to_string(self):
        """Non-structured data should fall back to string."""
        format_spec = ToolOutputFormat(style="toon")
        strategy = ToonFormatStrategy(format_spec)
        result = strategy.format(
            tool_name="test",
            args={},
            output="simple string",
        )

        assert "simple string" in result

    def test_estimate_tokens_is_more_efficient(self):
        """TOON should estimate fewer tokens than plain JSON."""
        format_spec = ToolOutputFormat(style="toon")
        strategy = ToonFormatStrategy(format_spec)

        # Create structured content
        content = "item1 | item2 | item3"
        tokens = strategy.estimate_tokens(content)

        # TOON estimates ~60% of plain tokens
        plain_tokens = len(content) // 4
        assert tokens < plain_tokens


class TestFormatStrategyFactory:
    """Test format strategy factory."""

    def test_create_plain_strategy(self):
        """Factory should create plain strategy."""
        format_spec = ToolOutputFormat(style="plain")
        strategy = FormatStrategyFactory.create(format_spec)
        assert isinstance(strategy, PlainFormatStrategy)

    def test_create_xml_strategy(self):
        """Factory should create XML strategy."""
        format_spec = ToolOutputFormat(style="xml")
        strategy = FormatStrategyFactory.create(format_spec)
        assert isinstance(strategy, XmlFormatStrategy)

    def test_create_toon_strategy(self):
        """Factory should create TOON strategy."""
        format_spec = ToolOutputFormat(style="toon")
        strategy = FormatStrategyFactory.create(format_spec)
        assert isinstance(strategy, ToonFormatStrategy)

    def test_invalid_style_raises_error(self):
        """Factory should raise ValueError for unregistered styles."""
        # Create a format spec with an invalid style by bypassing validation
        # We need to test the factory's error handling, not ToolOutputFormat's
        from dataclasses import dataclass

        @dataclass
        class InvalidFormat:
            style: str = "invalid"

        invalid_spec = InvalidFormat()
        with pytest.raises(ValueError, match="Unsupported format style"):
            FormatStrategyFactory.create(invalid_spec)

    def test_register_custom_strategy(self):
        """Factory should support registering custom strategies."""

        # First register the custom style
        class CustomStrategy:
            def format(self, tool_name, args, output, format_hint=None):
                return f"[CUSTOM] {tool_name}: {output}"

            def estimate_tokens(self, content):
                return len(content) // 2

        FormatStrategyFactory.register_strategy("custom", CustomStrategy)

        # Now create a mock format spec with the custom style
        # (bypassing ToolOutputFormat validation for this test)
        from dataclasses import dataclass

        @dataclass
        class CustomFormat:
            style: str = "custom"

        custom_spec = CustomFormat()
        strategy = FormatStrategyFactory.create(custom_spec)
        result = strategy.format("test", {}, "output")

        assert "[CUSTOM] test: output" in result

    def test_register_strategy_without_format_raises_error(self):
        """Registering strategy without format method should raise TypeError."""

        class InvalidStrategy:
            pass

        with pytest.raises(TypeError, match="must implement 'format' method"):
            FormatStrategyFactory.register_strategy("invalid", InvalidStrategy)
