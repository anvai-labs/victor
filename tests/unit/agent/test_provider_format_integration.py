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

"""Tests for provider-specific tool output format integration."""

import pytest

from victor.agent.tool_output_formatter import (
    ToolOutputFormatter,
    FormattingContext,
    ToolOutputFormatterConfig,
)
from victor.agent.format_strategies import (
    ToolOutputFormat,
    PLAIN_FORMAT,
    XML_FORMAT,
)
from victor.providers.openai_provider import OpenAIProvider
from victor.providers.ollama_provider import OllamaProvider
from victor.providers.vllm_provider import VLLMProvider
from victor.providers.llamacpp_provider import LlamaCppProvider


class TestProviderFormatOverrides:
    """Test provider get_tool_output_format() overrides."""

    def test_openai_uses_plain_format_by_default(self):
        """OpenAI should use plain JSON format by default."""
        provider = OpenAIProvider(api_key="test-key")
        format_spec = provider.get_tool_output_format()

        assert isinstance(format_spec, ToolOutputFormat)
        assert format_spec.style == "plain"
        assert format_spec.use_delimiters is False
        assert format_spec.include_tags is False

    def test_ollama_uses_xml_format(self):
        """Ollama should use XML format for model cognition."""
        provider = OllamaProvider()
        format_spec = provider.get_tool_output_format()

        assert isinstance(format_spec, ToolOutputFormat)
        assert format_spec.style == "xml"
        assert format_spec.use_delimiters is True
        assert format_spec.delimiter_char == "="
        assert format_spec.delimiter_width == 50
        assert format_spec.include_tags is True
        assert format_spec.tag_name == "TOOL_OUTPUT"

    def test_vllm_uses_xml_format(self):
        """vLLM should use XML format for model cognition."""
        provider = VLLMProvider(base_url="http://localhost:8000")
        format_spec = provider.get_tool_output_format()

        assert isinstance(format_spec, ToolOutputFormat)
        assert format_spec.style == "xml"
        assert format_spec.use_delimiters is True

    def test_llamacpp_uses_xml_format(self):
        """llama.cpp should use XML format for model cognition."""
        provider = LlamaCppProvider(base_url="http://localhost:8080")
        format_spec = provider.get_tool_output_format()

        assert isinstance(format_spec, ToolOutputFormat)
        assert format_spec.style == "xml"
        assert format_spec.use_delimiters is True


class TestToolOutputFormatterIntegration:
    """Test ToolOutputFormatter with provider-specific strategies."""

    def test_formatter_uses_plain_format_for_openai(self):
        """Formatter should use plain JSON for OpenAI provider."""
        formatter = ToolOutputFormatter()
        provider = OpenAIProvider(api_key="test-key")

        context = FormattingContext(
            provider=provider,
            provider_name="openai",
        )

        result = formatter.format_tool_output(
            tool_name="test_tool",
            args={"param": "value"},
            output={"result": "data"},
            context=context,
        )

        # Should NOT have XML tags
        assert "<TOOL_OUTPUT>" not in result
        assert "==================================================" not in result
        # Should have plain JSON (either double or single quotes depending on implementation)
        assert "result" in result
        assert "data" in result

    def test_formatter_uses_xml_format_for_ollama(self):
        """Formatter should use XML format for Ollama provider."""
        formatter = ToolOutputFormatter()
        provider = OllamaProvider()

        context = FormattingContext(
            provider=provider,
            provider_name="ollama",
        )

        result = formatter.format_tool_output(
            tool_name="read_file",
            args={"path": "test.py"},
            output="file content here",
            context=context,
        )

        # Should have XML tags and delimiters
        assert "<TOOL_OUTPUT" in result
        assert 'tool="read_file"' in result
        assert "===" in result  # XML_FORMAT uses '=' as delimiter_char
        assert "file content here" in result
        assert "</TOOL_OUTPUT>" in result

    def test_formatter_uses_xml_format_for_vllm(self):
        """Formatter should use XML format for vLLM provider."""
        formatter = ToolOutputFormatter()
        provider = VLLMProvider(base_url="http://localhost:8000")

        context = FormattingContext(
            provider=provider,
            provider_name="vllm",
        )

        result = formatter.format_tool_output(
            tool_name="test_tool",
            args={},
            output="test output",
            context=context,
        )

        assert "<TOOL_OUTPUT" in result
        assert "==================================================" in result
        assert "</TOOL_OUTPUT>" in result

    def test_formatter_uses_xml_format_for_llamacpp(self):
        """Formatter should use XML format for llama.cpp provider."""
        formatter = ToolOutputFormatter()
        provider = LlamaCppProvider(base_url="http://localhost:8080")

        context = FormattingContext(
            provider=provider,
            provider_name="llamacpp",
        )

        result = formatter.format_tool_output(
            tool_name="test_tool",
            args={},
            output="test output",
            context=context,
        )

        assert "<TOOL_OUTPUT" in result
        assert "==================================================" in result
        assert "</TOOL_OUTPUT>" in result

    def test_formatter_falls_back_to_plain_without_provider(self):
        """Formatter should fall back to plain format when no provider."""
        formatter = ToolOutputFormatter()

        context = FormattingContext(
            provider=None,
            provider_name="unknown",
        )

        result = formatter.format_tool_output(
            tool_name="test_tool",
            args={},
            output="test output",
            context=context,
        )

        # Should use plain format (default)
        assert "<TOOL_OUTPUT>" not in result
        assert "test output" in result

    def test_formatter_handles_provider_without_format_method(self):
        """Formatter should handle providers without get_tool_output_format."""
        formatter = ToolOutputFormatter()

        # Mock provider without get_tool_output_format
        class MockProvider:
            pass

        context = FormattingContext(
            provider=MockProvider(),
            provider_name="mock",
        )

        result = formatter.format_tool_output(
            tool_name="test_tool",
            args={},
            output="test output",
            context=context,
        )

        # Should fall back to plain format
        assert "test output" in result


class TestTokenEfficiency:
    """Test token efficiency improvements."""

    def test_plain_format_more_efficient_than_xml(self):
        """Plain format should be more token-efficient than XML."""
        provider_openai = OpenAIProvider(api_key="test-key")
        provider_ollama = OllamaProvider()

        context_openai = FormattingContext(provider=provider_openai)
        context_ollama = FormattingContext(provider=provider_ollama)

        formatter = ToolOutputFormatter()

        result_openai = formatter.format_tool_output(
            tool_name="test",
            args={},
            output={"data": "value"},
            context=context_openai,
        )

        result_ollama = formatter.format_tool_output(
            tool_name="test",
            args={},
            output={"data": "value"},
            context=context_ollama,
        )

        # Plain format should be shorter
        assert len(result_openai) < len(result_ollama)

        # Count token overhead difference
        overhead_openai = len(result_openai) - len('{"data": "value"}')
        overhead_ollama = len(result_ollama) - len('{"data": "value"}')

        # XML should have more overhead
        assert overhead_ollama > overhead_openai

    def test_xml_overhead_approximately_50_tokens(self):
        """XML format overhead should be ~50 tokens per tool call."""
        provider = OllamaProvider()
        context = FormattingContext(provider=provider)
        formatter = ToolOutputFormatter()

        result = formatter.format_tool_output(
            tool_name="test",
            args={"param": "value"},
            output="output content",
            context=context,
        )

        # Rough token estimate
        tokens = len(result) // 4

        # Output content is ~14 chars (~4 tokens)
        # Total should be ~50-60 tokens due to XML overhead
        assert tokens > 40
        assert tokens < 80


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_formatting_context_without_provider(self):
        """FormattingContext should work without provider field."""
        # Old code might not pass provider
        context = FormattingContext(
            provider_name="openai",
            model="gpt-4",
        )

        assert context.provider is None
        assert context.provider_name == "openai"
        assert context.model == "gpt-4"

    def test_formatter_works_with_legacy_context(self):
        """Formatter should work with legacy FormattingContext."""
        formatter = ToolOutputFormatter()

        # Legacy context without provider
        context = FormattingContext(
            provider_name="openai",
            model="gpt-4",
            remaining_tokens=50000,
            max_tokens=100000,
        )

        result = formatter.format_tool_output(
            tool_name="test",
            args={},
            output="output",
            context=context,
        )

        # Should work and use default plain format
        assert "output" in result

    def test_specialized_tools_still_work(self):
        """Tools should use provider-specific formatting (Plain for cloud, XML for local)."""
        formatter = ToolOutputFormatter()

        # Test cloud provider (OpenAI) - uses Plain format
        openai_provider = OpenAIProvider(api_key="test-key")
        openai_context = FormattingContext(provider=openai_provider)

        openai_result = formatter.format_tool_output(
            tool_name="read_file",
            args={"path": "test.py"},
            output="file content",
            context=openai_context,
        )

        # Cloud providers use Plain format (no metadata, just output)
        assert openai_result == "file content"
        assert "<TOOL_OUTPUT" not in openai_result

        # Test local provider (Ollama) - uses XML format
        ollama_provider = OllamaProvider()
        ollama_context = FormattingContext(provider=ollama_provider)

        ollama_result = formatter.format_tool_output(
            tool_name="read_file",
            args={"path": "test.py"},
            output="file content",
            context=ollama_context,
        )

        # Local providers use XML format (with metadata and delimiters)
        assert "<TOOL_OUTPUT" in ollama_result
        assert 'tool="read_file"' in ollama_result
        assert "file content" in ollama_result
        assert "</TOOL_OUTPUT>" in ollama_result

    def test_truncation_still_works(self):
        """Truncation logic should still work."""
        config = ToolOutputFormatterConfig(max_output_chars=50)
        formatter = ToolOutputFormatter(config=config)
        provider = OpenAIProvider(api_key="test-key")
        context = FormattingContext(provider=provider)

        long_output = "x" * 1000

        result = formatter.format_tool_output(
            tool_name="test",
            args={},
            output=long_output,
            context=context,
        )

        # Should be truncated
        assert len(result) < 1000
        # Should contain truncation marker
        assert "[OUTPUT TRUNCATED]" in result or len(result) <= 150
