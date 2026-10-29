"""Unit tests for formatter registry and centralized formatting logic."""

from victor.tools.formatters.base import ToolFormatter, FormattedOutput
from victor.tools.formatters.registry import (
    FormatterRegistry,
    get_formatter_registry,
    format_tool_output,
)
from victor.tools.formatters.generic import GenericFormatter


class MockFormatter(ToolFormatter):
    """Mock formatter for testing."""

    def format(self, data, **kwargs):
        content = data.get("content", "mock output")
        return FormattedOutput(
            content=content,
            format_type="mock",
            summary="Mock output",
            contains_markup=False,
        )


class FailingFormatter(ToolFormatter):
    """Formatter that always fails for testing error handling."""

    def format(self, data, **kwargs):
        raise ValueError("Intentional failure")


class TestFormatterRegistry:
    """Test FormatterRegistry singleton."""

    def test_singleton(self):
        """Test that FormatterRegistry is a singleton."""
        registry1 = FormatterRegistry()
        registry2 = FormatterRegistry()

        assert registry1 is registry2

    def test_get_formatter_registry_singleton(self):
        """Test that get_formatter_registry() returns singleton."""
        registry1 = get_formatter_registry()
        registry2 = get_formatter_registry()

        assert registry1 is registry2

    def test_register_formatter(self):
        """Test registering a formatter."""
        registry = get_formatter_registry()
        formatter = MockFormatter()

        registry.register("mock", formatter)

        assert registry.get_formatter("mock") is formatter

    def test_get_formatter_returns_registered(self):
        """Test that get_formatter() returns registered formatter."""
        registry = get_formatter_registry()
        formatter = MockFormatter()

        registry.register("test_tool", formatter)
        result = registry.get_formatter("test_tool")

        assert result is formatter

    def test_get_formatter_fallback_to_generic(self):
        """Test that get_formatter() falls back to GenericFormatter."""
        registry = get_formatter_registry()

        result = registry.get_formatter("nonexistent_tool")

        assert isinstance(result, GenericFormatter)

    def test_list_formatters(self):
        """Test listing all registered formatters."""
        registry = get_formatter_registry()
        formatter1 = MockFormatter()
        formatter2 = MockFormatter()

        registry.register("tool1", formatter1)
        registry.register("tool2", formatter2)

        formatters = registry.list_formatters()

        assert "tool1" in formatters
        assert "tool2" in formatters


class TestFormatToolOutput:
    """Test format_tool_output() convenience function."""

    def test_format_with_registered_formatter(self):
        """Test formatting with a registered formatter."""
        registry = get_formatter_registry()
        formatter = MockFormatter()
        registry.register("test", formatter)  # Use "test" which is in allowed list

        result = format_tool_output("test", {"content": "test"})

        assert result.content == "test"
        assert result.format_type == "mock"
        assert result.summary == "Mock output"

    def test_format_with_generic_fallback(self):
        """Test formatting falls back to GenericFormatter."""
        result = format_tool_output("unknown_tool", {"key": "value"})

        assert result.format_type == "plain"
        # Generic formatter creates key: value pairs
        assert "key" in result.content or "value" in result.content

    def test_format_with_invalid_input(self):
        """Test formatting with invalid input."""
        registry = get_formatter_registry()

        class ValidatingFormatter(ToolFormatter):
            def validate_input(self, data):
                return isinstance(data, dict) and "valid" in data

            def format(self, data, **kwargs):
                return FormattedOutput(content="valid")

        registry.register("test", ValidatingFormatter())  # Use exact allowed name

        result = format_tool_output("test", {"invalid": "data"})

        # With production guards, invalid input returns plain text
        assert result.summary is not None  # Should have a summary
        assert "invalid" in result.summary.lower() or "invalid" in str(result.content).lower()

    def test_format_with_formatter_error(self):
        """Test that formatter errors are handled gracefully."""
        registry = get_formatter_registry()
        formatter = FailingFormatter()
        registry.register("pytest", formatter)  # Use exact allowed name

        result = format_tool_output("pytest", {})

        # Should fall back to plain text on error
        assert result.format_type == "plain"
        assert result.summary is not None  # Should have a summary
        assert "failed" in result.summary.lower() or "pytest" in result.summary.lower()

    def test_format_with_kwargs(self):
        """Test that kwargs are passed to formatter."""

        class KwargsFormatter(ToolFormatter):
            def format(self, data, **kwargs):
                max_items = kwargs.get("max_items", 10)
                return FormattedOutput(
                    content=f"max_items={max_items}",
                    summary=f"Limit: {max_items}",
                )

        registry = get_formatter_registry()
        registry.register("code_search", KwargsFormatter())  # Use exact allowed name

        result = format_tool_output("code_search", {}, max_items=5)

        # Should pass through kwargs to formatter
        assert "5" in result.content or "5" in str(result.summary)

    def test_format_with_fallback_formatter(self):
        """Test that fallback formatter is used on error."""

        class FormatterWithFallback(ToolFormatter):
            def format(self, data, **kwargs):
                raise ValueError("Primary formatter failed")

            def get_fallback(self):
                return MockFormatter()

        registry = get_formatter_registry()
        registry.register("test_fallback", FormatterWithFallback())

        result = format_tool_output("test_fallback", {"content": "fallback test"})

        # Should use fallback formatter
        assert "fallback test" in result.content or "content" in result.content
