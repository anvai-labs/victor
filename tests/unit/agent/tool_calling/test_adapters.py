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

"""Unit tests for tool calling adapters.

Tests for Anthropic, OpenAI, and Ollama tool calling adapters.
"""

import pytest
from unittest.mock import MagicMock, patch

from victor.agent.tool_calling.adapters import (
    AnthropicToolCallingAdapter,
    GoogleToolCallingAdapter,
    OpenAIToolCallingAdapter,
    OllamaToolCallingAdapter,
    _CapabilityLoaderHolder,
    _get_capability_loader,
)
from victor.agent.tool_calling.base import (
    ToolCall,
    ToolCallFormat,
    ToolCallParseResult,
)
from victor.providers.base import ToolDefinition

# =============================================================================
# _CapabilityLoaderHolder Tests
# =============================================================================


class TestCapabilityLoaderHolder:
    """Tests for _CapabilityLoaderHolder singleton."""

    def test_get_returns_singleton(self):
        """Get returns same instance on multiple calls."""
        holder1 = _CapabilityLoaderHolder.get()
        holder2 = _CapabilityLoaderHolder.get()

        assert holder1 is holder2

    def test_get_capability_loader_singleton(self):
        """_get_capability_loader returns singleton."""
        loader1 = _get_capability_loader()
        loader2 = _get_capability_loader()

        assert loader1 is loader2


# =============================================================================
# AnthropicToolCallingAdapter Tests
# =============================================================================


class TestAnthropicAdapterInit:
    """Tests for Anthropic adapter initialization."""

    def test_init_with_model(self):
        """Initialize adapter with model."""
        adapter = AnthropicToolCallingAdapter(
            model="claude-3-sonnet",
            config=None,
        )

        assert adapter.model == "claude-3-sonnet"
        # Config defaults to empty dict
        assert adapter.config == {} or adapter.config is None

    def test_provider_name(self):
        """Provider name is 'anthropic'."""
        adapter = AnthropicToolCallingAdapter(
            model="claude-3-opus",
            config=None,
        )

        assert adapter.provider_name == "anthropic"


class TestAnthropicAdapterConvertTools:
    """Tests for Anthropic adapter tool conversion."""

    @pytest.fixture
    def sample_tools(self):
        """Create sample tool definitions."""
        return [
            ToolDefinition(
                name="read_file",
                description="Read a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                },
            ),
            ToolDefinition(
                name="write_file",
                description="Write a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            ),
        ]

    def test_convert_tools_to_anthropic_format(self, sample_tools):
        """Convert tools to Anthropic format."""
        adapter = AnthropicToolCallingAdapter(
            model="claude-3-sonnet",
            config=None,
        )

        result = adapter.convert_tools(sample_tools)

        assert len(result) == 2
        assert result[0]["name"] == "read_file"
        assert result[0]["description"] == "Read a file"
        assert "input_schema" in result[0]
        assert result[0]["input_schema"] == sample_tools[0].parameters

    def test_convert_empty_tools_list(self):
        """Convert empty tools list."""
        adapter = AnthropicToolCallingAdapter(
            model="claude-3-sonnet",
            config=None,
        )

        result = adapter.convert_tools([])

        assert result == []


class TestAnthropicAdapterParseToolCalls:
    """Tests for Anthropic adapter tool call parsing."""

    def test_parse_tool_calls_with_raw_calls(self):
        """Parse tool calls from raw_tool_calls list."""
        adapter = AnthropicToolCallingAdapter(
            model="claude-3-sonnet",
            config=None,
        )
        # Mock valid tool names
        adapter._valid_tool_names = {"read_file", "write_file"}

        raw_calls = [
            {
                "id": "call-1",
                "name": "read_file",
                "arguments": {"path": "/test/file.txt"},
            },
            {
                "id": "call-2",
                "name": "write_file",
                "arguments": {"path": "/test/out.txt", "content": "test"},
            },
        ]

        result = adapter.parse_tool_calls(
            content="Some content",
            raw_tool_calls=raw_calls,
        )

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[0].arguments == {"path": "/test/file.txt"}
        assert result.tool_calls[0].id == "call-1"
        assert result.parse_method == "native"
        assert result.confidence == 1.0

    def test_parse_tool_calls_rejects_hallucinated_names(self):
        """Reject tool calls with hallucinated/malformed names."""
        adapter = AnthropicToolCallingAdapter(
            model="claude-3-sonnet",
            config=None,
        )

        raw_calls = [
            {
                "id": "call-1",
                "name": "read_file",  # Valid
                "arguments": {"path": "/test/file.txt"},
            },
            {
                "id": "call-2",
                "name": "example_tool",  # Invalid - starts with example_
                "arguments": {},
            },
            {
                "id": "call-3",
                "name": "func_test",  # Invalid - starts with func_
                "arguments": {},
            },
            {
                "id": "call-4",
                "name": "",  # Invalid - empty
                "arguments": {},
            },
        ]

        result = adapter.parse_tool_calls(
            content="Some content",
            raw_tool_calls=raw_calls,
        )

        # read_file passes validation; invalid names kept with ids for error responses
        # (OpenAI spec: every tool_calls[].id needs a role=tool response)
        assert len(result.tool_calls) == 4  # 1 valid + 3 invalid (all have ids)
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[0].arguments == {"path": "/test/file.txt"}
        # Invalid tools have empty args
        assert result.tool_calls[1].name == "example_tool"
        assert result.tool_calls[1].arguments == {}
        assert result.tool_calls[1].id == "call-2"

    def test_parse_tool_calls_empty_raw_calls(self):
        """Empty raw_tool_calls returns result with no tool calls."""
        adapter = AnthropicToolCallingAdapter(
            model="claude-3-sonnet",
            config=None,
        )

        result = adapter.parse_tool_calls(
            content="Some content",
            raw_tool_calls=None,
        )

        assert len(result.tool_calls) == 0
        assert result.remaining_content == "Some content"

    def test_parse_tool_calls_with_empty_list(self):
        """Empty list returns result with no tool calls."""
        adapter = AnthropicToolCallingAdapter(
            model="claude-3-sonnet",
            config=None,
        )

        result = adapter.parse_tool_calls(
            content="Some content",
            raw_tool_calls=[],
        )

        assert len(result.tool_calls) == 0


class TestAnthropicAdapterCapabilities:
    """Tests for Anthropic adapter capabilities."""

    def test_get_capabilities(self):
        """Get capabilities returns Anthropic-specific capabilities."""
        adapter = AnthropicToolCallingAdapter(
            model="claude-3-sonnet",
            config=None,
        )

        caps = adapter.get_capabilities()

        assert caps.native_tool_calls is True
        assert caps.streaming_tool_calls is True
        assert caps.parallel_tool_calls is True
        assert caps.tool_choice_param is True
        assert caps.tool_call_format == ToolCallFormat.ANTHROPIC


# =============================================================================
# OpenAIToolCallingAdapter Tests
# =============================================================================


class TestOpenAIAdapterInit:
    """Tests for OpenAI adapter initialization."""

    def test_init_with_model(self):
        """Initialize adapter with model."""
        adapter = OpenAIToolCallingAdapter(
            model="gpt-4",
            config=None,
        )

        assert adapter.model == "gpt-4"
        # Config defaults to empty dict
        assert adapter.config == {} or adapter.config is None

    def test_provider_name(self):
        """Provider name is 'openai'."""
        adapter = OpenAIToolCallingAdapter(
            model="gpt-4-turbo",
            config=None,
        )

        assert adapter.provider_name == "openai"


class TestOpenAIAdapterConvertTools:
    """Tests for OpenAI adapter tool conversion."""

    @pytest.fixture
    def sample_tools(self):
        """Create sample tool definitions."""
        return [
            ToolDefinition(
                name="search",
                description="Search code",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                },
            ),
        ]

    def test_convert_tools_to_openai_format(self, sample_tools):
        """Convert tools to OpenAI function format."""
        adapter = OpenAIToolCallingAdapter(
            model="gpt-4",
            config=None,
        )

        result = adapter.convert_tools(sample_tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert "function" in result[0]
        assert result[0]["function"]["name"] == "search"
        assert result[0]["function"]["description"] == "Search code"


class TestOpenAIAdapterParseToolCalls:
    """Tests for OpenAI adapter tool call parsing."""

    def test_parse_native_tool_calls(self):
        """Parse native OpenAI tool calls."""
        adapter = OpenAIToolCallingAdapter(
            model="gpt-4",
            config=None,
        )
        adapter._valid_tool_names = {"search"}

        raw_calls = [
            {
                "id": "call-1",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "test"}',
                },
            },
        ]

        # Convert to OpenAI format (flattened)
        raw_calls_flat = [
            {
                "id": "call-1",
                "name": "search",
                "arguments": '{"query": "test"}',
            },
        ]

        result = adapter.parse_tool_calls(
            content="Some content",
            raw_tool_calls=raw_calls_flat,
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"query": "test"}
        assert result.parse_method == "native"

    def test_parse_native_tool_calls_with_dict_args(self):
        """Parse tool calls with dict arguments (already parsed)."""
        adapter = OpenAIToolCallingAdapter(
            model="gpt-4",
            config=None,
        )
        adapter._valid_tool_names = {"search"}

        raw_calls = [
            {
                "id": "call-1",
                "name": "search",
                "arguments": {"query": "test"},  # Already a dict
            },
        ]

        result = adapter.parse_tool_calls(
            content="Some content",
            raw_tool_calls=raw_calls,
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {"query": "test"}

    def test_parse_tool_calls_invalid_json_args(self):
        """Handle tool calls with invalid JSON in arguments."""
        adapter = OpenAIToolCallingAdapter(
            model="gpt-4",
            config=None,
        )
        adapter._valid_tool_names = {"search"}

        raw_calls = [
            {
                "id": "call-1",
                "name": "search",
                "arguments": "{invalid json}",
            },
        ]

        result = adapter.parse_tool_calls(
            content="Some content",
            raw_tool_calls=raw_calls,
        )

        # OpenAI adapter includes the tool call but with empty args and a warning
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {}
        assert len(result.warnings) > 0
        assert "Failed to parse arguments" in result.warnings[0]

    def test_parse_tool_calls_no_raw_calls_returns_empty(self):
        """No raw_tool_calls returns empty result."""
        adapter = OpenAIToolCallingAdapter(
            model="gpt-4",
            config=None,
        )

        result = adapter.parse_tool_calls(
            content="Some content",
            raw_tool_calls=None,
        )

        assert len(result.tool_calls) == 0


class TestOpenAIAdapterSystemPromptHints:
    """Tests for OpenAI adapter system prompt hints."""

    def test_get_system_prompt_hints_for_openai_model(self):
        """Standard OpenAI models get parallel hints."""
        adapter = OpenAIToolCallingAdapter(
            model="gpt-4-turbo",
            config=None,
        )

        hints = adapter.get_system_prompt_hints()

        assert "parallel" in hints.lower()

    def test_get_system_prompt_hints_for_open_weight_model(self):
        """Open-weight models get stronger tool usage hints."""
        adapter = OpenAIToolCallingAdapter(
            model="llama-3.1-70b",
            config=None,
        )

        hints = adapter.get_system_prompt_hints()

        assert "TOOL USAGE REQUIRED" in hints
        assert "MUST use the provided tools" in hints
        assert "DO NOT speculate" in hints

    def test_get_system_prompt_hints_for_qwen_model(self):
        """Qwen models get stronger hints."""
        adapter = OpenAIToolCallingAdapter(
            model="qwen2.5-coder",
            config=None,
        )

        hints = adapter.get_system_prompt_hints()

        assert "TOOL USAGE REQUIRED" in hints

    def test_get_system_prompt_hints_for_deepseek_model(self):
        """DeepSeek models get stronger hints."""
        adapter = OpenAIToolCallingAdapter(
            model="deepseek-r1",
            config=None,
        )

        hints = adapter.get_system_prompt_hints()

        assert "TOOL USAGE REQUIRED" in hints


# =============================================================================
# OllamaToolCallingAdapter Tests
# =============================================================================


class TestOllamaAdapterInit:
    """Tests for Ollama adapter initialization."""

    def test_init_with_model(self):
        """Initialize adapter with model."""
        adapter = OllamaToolCallingAdapter(
            model="llama3.1",
            config=None,
        )

        assert adapter.model == "llama3.1"
        # Config defaults to empty dict
        assert adapter.config == {} or adapter.config is None

    def test_provider_name(self):
        """Provider name is 'ollama'."""
        adapter = OllamaToolCallingAdapter(
            model="mistral",
            config=None,
        )

        assert adapter.provider_name == "ollama"


class TestOllamaAdapterNativeToolModels:
    """Tests for native tool model detection."""

    def test_native_tool_models_set(self):
        """NATIVE_TOOL_MODELS is a frozenset."""
        assert isinstance(OllamaToolCallingAdapter.NATIVE_TOOL_MODELS, frozenset)
        assert len(OllamaToolCallingAdapter.NATIVE_TOOL_MODELS) > 0

    def test_native_tool_models_includes_llama(self):
        """Llama models are in native tool models list."""
        assert "llama3.1" in OllamaToolCallingAdapter.NATIVE_TOOL_MODELS
        assert "llama-3.1" in OllamaToolCallingAdapter.NATIVE_TOOL_MODELS
        assert "llama3.2" in OllamaToolCallingAdapter.NATIVE_TOOL_MODELS

    def test_native_tool_models_includes_qwen(self):
        """Qwen models are in native tool models list."""
        assert "qwen2.5" in OllamaToolCallingAdapter.NATIVE_TOOL_MODELS
        assert "qwen-2.5" in OllamaToolCallingAdapter.NATIVE_TOOL_MODELS
        assert "qwen3" in OllamaToolCallingAdapter.NATIVE_TOOL_MODELS

    def test_native_tool_models_includes_mistral(self):
        """Mistral models are in native tool models list."""
        assert "mistral" in OllamaToolCallingAdapter.NATIVE_TOOL_MODELS
        assert "mixtral" in OllamaToolCallingAdapter.NATIVE_TOOL_MODELS


class TestOllamaAdapterParameterAliases:
    """Tests for parameter alias mappings."""

    def test_parameter_aliases_exist(self):
        """PARAMETER_ALIASES is a dict."""
        assert isinstance(OllamaToolCallingAdapter.PARAMETER_ALIASES, dict)

    def test_parameter_aliases_for_read(self):
        """Read tool has parameter aliases."""
        assert "read" in OllamaToolCallingAdapter.PARAMETER_ALIASES
        read_aliases = OllamaToolCallingAdapter.PARAMETER_ALIASES["read"]
        # Values are the standard parameter names
        assert "offset" in read_aliases.values()
        assert "limit" in read_aliases.values()
        # Key is the model-specific param, value is standard
        assert read_aliases.get("line_start") == "offset"
        assert read_aliases.get("num_lines") == "limit"

    def test_parameter_aliases_for_write(self):
        """Write tool has parameter aliases."""
        assert "write" in OllamaToolCallingAdapter.PARAMETER_ALIASES
        write_aliases = OllamaToolCallingAdapter.PARAMETER_ALIASES["write"]
        # Values are the standard parameter names
        assert "path" in write_aliases.values()
        assert "content" in write_aliases.values()
        # Key is the model-specific param, value is standard
        assert write_aliases.get("file") == "path"
        assert write_aliases.get("text") == "content"

    def test_parameter_aliases_for_shell(self):
        """Shell tool has parameter aliases."""
        assert "shell" in OllamaToolCallingAdapter.PARAMETER_ALIASES
        shell_aliases = OllamaToolCallingAdapter.PARAMETER_ALIASES["shell"]
        # Key is the model-specific param, value is standard
        assert shell_aliases.get("cmd") == "command"
        assert "command" in shell_aliases.values()


class TestOllamaAdapterThinkingMode:
    """Tests for thinking mode detection."""

    def test_has_thinking_mode_for_qwen3(self):
        """Qwen3 models have thinking mode."""
        adapter = OllamaToolCallingAdapter(
            model="qwen3:32b",
            config=None,
        )

        assert adapter._has_thinking_mode() is True

    def test_has_thinking_mode_for_qwen_with_dash(self):
        """Qwen-3 models have thinking mode."""
        adapter = OllamaToolCallingAdapter(
            model="qwen-3:32b",
            config=None,
        )

        assert adapter._has_thinking_mode() is True

    def test_no_thinking_mode_for_other_models(self):
        """Non-Qwen3 models don't have thinking mode."""
        adapter = OllamaToolCallingAdapter(
            model="llama3.1",
            config=None,
        )

        assert adapter._has_thinking_mode() is False


class TestGoogleAdapterFallbackCanonicalization:
    """Tests for canonical tool names emitted by Gemini fallback parsing."""

    def test_execute_bash_text_fallback_emits_shell(self):
        """Gemini execute_bash tags should normalize to canonical shell tool."""
        adapter = GoogleToolCallingAdapter(
            model="gemini-2.5-pro",
            config=None,
        )
        adapter._valid_tool_names = {"shell"}

        result = adapter.parse_tool_calls("<execute_bash>ls -la</execute_bash>", None)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "shell"
        assert result.tool_calls[0].arguments == {"cmd": "ls -la"}

    def test_tool_code_read_file_emits_canonical_read(self):
        """Gemini tool_code read_file snippets should normalize to canonical read."""
        adapter = GoogleToolCallingAdapter(
            model="gemini-2.5-pro",
            config=None,
        )
        adapter._valid_tool_names = {"read"}

        result = adapter.parse_tool_calls('<tool_code>print(read_file("foo.py"))</tool_code>', None)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "read"
        assert result.tool_calls[0].arguments == {"path": "foo.py"}


# =============================================================================
# Cross-Adapter Tests
# =============================================================================


class TestAdapterConsistency:
    """Tests for adapter interface consistency."""

    def test_all_adapters_have_provider_name(self):
        """All adapters have provider_name property."""
        anthropic = AnthropicToolCallingAdapter("claude-3", None)
        openai = OpenAIToolCallingAdapter("gpt-4", None)
        ollama = OllamaToolCallingAdapter("llama3.1", None)

        assert anthropic.provider_name == "anthropic"
        assert openai.provider_name == "openai"
        assert ollama.provider_name == "ollama"

    def test_all_adapters_have_get_capabilities(self):
        """All adapters have get_capabilities method."""
        anthropic = AnthropicToolCallingAdapter("claude-3", None)
        openai = OpenAIToolCallingAdapter("gpt-4", None)
        ollama = OllamaToolCallingAdapter("llama3.1", None)

        anthropic_caps = anthropic.get_capabilities()
        openai_caps = openai.get_capabilities()
        ollama_caps = ollama.get_capabilities()

        assert anthropic_caps.native_tool_calls is True
        assert openai_caps.native_tool_calls is True
        # Ollama depends on model, but should return some capabilities
        assert ollama_caps is not None

    def test_all_adapters_have_convert_tools(self):
        """All adapters have convert_tools method."""
        tool = ToolDefinition(
            name="test",
            description="Test tool",
            parameters={"type": "object"},
        )

        anthropic = AnthropicToolCallingAdapter("claude-3", None)
        openai = OpenAIToolCallingAdapter("gpt-4", None)
        ollama = OllamaToolCallingAdapter("llama3.1", None)

        anthropic_result = anthropic.convert_tools([tool])
        openai_result = openai.convert_tools([tool])
        ollama_result = ollama.convert_tools([tool])

        assert len(anthropic_result) == 1
        assert len(openai_result) == 1
        assert len(ollama_result) == 1

    def test_all_adapters_have_parse_tool_calls(self):
        """All adapters have parse_tool_calls method."""
        anthropic = AnthropicToolCallingAdapter("claude-3", None)
        openai = OpenAIToolCallingAdapter("gpt-4", None)
        ollama = OllamaToolCallingAdapter("llama3.1", None)

        anthropic_result = anthropic.parse_tool_calls("content", None)
        openai_result = openai.parse_tool_calls("content", None)
        ollama_result = ollama.parse_tool_calls("content", None)

        assert isinstance(anthropic_result, ToolCallParseResult)
        assert isinstance(openai_result, ToolCallParseResult)
        assert isinstance(ollama_result, ToolCallParseResult)
