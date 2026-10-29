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

from victor.agent.tool_calling.adapters import (
    AnthropicToolCallingAdapter,
    AzureOpenAIToolCallingAdapter,
    DeepSeekToolCallingAdapter,
    GoogleToolCallingAdapter,
    OpenAICompatToolCallingAdapter,
    OpenAIToolCallingAdapter,
    OllamaToolCallingAdapter,
    _CapabilityLoaderHolder,
    _get_capability_loader,
)
from victor.agent.tool_calling.base import (
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
        # Shared base contract: "{name}: Failed to parse JSON arguments"
        assert result.warnings[0] == "search: Failed to parse JSON arguments"

    @pytest.mark.parametrize(
        "content",
        [
            "Some content",
            "Use get_account(name='missing') to demonstrate the failure.",
            "I'll read the file: read(path='test.py')",
            "Example: shell(cmd='echo example')",
            '```json\n{"name":"read","arguments":{"path":"test.py"}}\n```',
            '<tool_call name="read">{"path":"test.py"}</tool_call>',
        ],
    )
    @pytest.mark.parametrize("raw_calls", [None, []])
    def test_parse_tool_calls_no_raw_calls_returns_empty(self, content, raw_calls):
        """Native tool contracts never grant execution authority to prose/examples."""
        adapter = OpenAIToolCallingAdapter(
            model="gpt-4",
            config=None,
        )

        result = adapter.parse_tool_calls(
            content=content,
            raw_tool_calls=raw_calls,
        )

        assert len(result.tool_calls) == 0
        assert result.remaining_content == content
        assert result.parse_method == "none"


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


class TestGeminiCtrl42AndSimpleCallParsing:
    """Direct coverage for the unified ctrl42/simple dual loop (jscpd +
    supervised extraction): both patterns parse through one body, with the
    label and warning-snippet knobs preserved per pattern."""

    @staticmethod
    def _adapter(valid):
        adapter = GoogleToolCallingAdapter(model="gemini-2.5-pro", config=None)
        adapter._valid_tool_names = valid
        return adapter

    def test_ctrl42_tagged_call_parses(self):
        adapter = self._adapter({"shell"})
        result = adapter.parse_tool_calls(
            '<ctrl42>call: `shell` ```json {"cmd": "ls"}```</ctrl42>', None
        )
        # FIXED (was a pinned quirk): the simple pass now skips spans the
        # ctrl42 pass already consumed — the call parses once.
        assert [t.name for t in result.tool_calls] == ["shell"]
        assert result.tool_calls[0].arguments == {"cmd": "ls"}

    def test_simple_call_parses(self):
        adapter = self._adapter({"shell"})
        result = adapter.parse_tool_calls('call: `shell` ```json {"cmd": "pwd"}```', None)
        assert [t.name for t in result.tool_calls] == ["shell"]
        assert result.tool_calls[0].arguments == {"cmd": "pwd"}

    def test_ctrl42_bad_json_warning_carries_snippet(self):
        # One good + one bad call: warnings survive only alongside parsed
        # calls (the empty-calls return path drops them — pre-existing quirk
        # pinned below).
        adapter = self._adapter({"shell", "read"})
        result = adapter.parse_tool_calls(
            'call: `read` ```json {"path": "f"}``` '
            'and <ctrl42>call: `shell` ```json {"a": b}```</ctrl42>',
            None,
        )
        assert any(
            w.startswith("Failed to parse JSON for shell: {") for w in result.warnings
        ), "ctrl42 warning carries the json snippet"

    def test_simple_bad_json_warning_has_no_snippet(self):
        adapter = self._adapter({"shell", "read"})
        result = adapter.parse_tool_calls(
            'call: `read` ```json {"path": "f"}``` ' 'and call: `shell` ```json {"a": b}```',
            None,
        )
        assert any(
            w == "Failed to parse JSON for shell" for w in result.warnings
        ), "simple-pattern warning keeps its snippet-free form"

    def test_warnings_survive_when_no_calls_parse(self):
        # FIXED (was a pinned quirk): the empty-calls return path dropped
        # warnings entirely; they now reach the caller.
        adapter = self._adapter({"shell"})
        result = adapter.parse_tool_calls('call: `shell` ```json {"a": b}```', None)
        assert result.tool_calls == []
        assert result.warnings == ["Failed to parse JSON for shell"]

    def test_both_patterns_parse_in_one_response(self):
        adapter = self._adapter({"shell", "read"})
        result = adapter.parse_tool_calls(
            '<ctrl42>call: `shell` ```json {"cmd": "ls"}```</ctrl42>'
            ' and call: `read` ```json {"path": "f"}```',
            None,
        )
        # ctrl42-consumed shell parses once; read parses once.
        assert sorted(t.name for t in result.tool_calls) == ["read", "shell"]


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


class TestCapabilityFieldPreservation:
    """The thinking-mode override must PRESERVE YAML-configured capability
    fields it does not touch (regression: the manual 12-field rebuild silently
    reset thinking_disable_prefix, exploration_multiplier,
    continuation_patience, and timeout_multiplier to defaults — discovered by
    the supervised dedup review)."""

    def test_ollama_thinking_override_preserves_configured_fields(self, monkeypatch):
        from types import SimpleNamespace

        import victor.agent.tool_calling.adapters as adapters_mod
        from victor.agent.tool_calling.adapters import OllamaToolCallingAdapter
        from victor.agent.tool_calling.base import ToolCallingCapabilities

        configured = ToolCallingCapabilities(
            thinking_disable_prefix="/no_think",
            exploration_multiplier=2.5,
            continuation_patience=3,
            timeout_multiplier=1.8,
        )
        monkeypatch.setattr(
            adapters_mod._CapabilityLoaderHolder,
            "_instance",
            SimpleNamespace(get_capabilities=lambda *a, **k: configured),
        )

        adapter = OllamaToolCallingAdapter(model="qwen3-coder", config=None)
        adapter._has_thinking_mode = lambda: True
        caps = adapter.get_capabilities()

        assert caps.thinking_mode is True
        # The four fields the old manual rebuild silently reset:
        assert caps.thinking_disable_prefix == "/no_think"
        assert caps.exploration_multiplier == 2.5
        assert caps.continuation_patience == 3
        assert caps.timeout_multiplier == 1.8

    def test_replace_semantics_keep_untouched_fields(self):
        from dataclasses import replace

        from victor.agent.tool_calling.base import ToolCallingCapabilities

        base = ToolCallingCapabilities(
            thinking_disable_prefix="/no_think",
            exploration_multiplier=2.5,
            continuation_patience=3,
            timeout_multiplier=1.8,
        )
        patched = replace(base, thinking_mode=True)
        assert patched.thinking_disable_prefix == "/no_think"
        assert patched.exploration_multiplier == 2.5
        assert patched.continuation_patience == 3
        assert patched.timeout_multiplier == 1.8
        assert patched.thinking_mode is True

    def test_lmstudio_thinking_override_preserves_configured_fields(self, monkeypatch):
        from types import SimpleNamespace

        import victor.agent.tool_calling.adapters as adapters_mod
        from victor.agent.tool_calling.adapters import LMStudioToolCallingAdapter
        from victor.agent.tool_calling.base import ToolCallingCapabilities

        configured = ToolCallingCapabilities(
            thinking_disable_prefix="/no_think",
            exploration_multiplier=2.5,
            continuation_patience=3,
            timeout_multiplier=1.8,
        )
        monkeypatch.setattr(
            adapters_mod._CapabilityLoaderHolder,
            "_instance",
            SimpleNamespace(get_capabilities=lambda *a, **k: configured),
        )
        adapter = LMStudioToolCallingAdapter(model="qwen3-coder", config=None)
        adapter._has_thinking_mode = lambda: True
        caps = adapter.get_capabilities()
        assert caps.thinking_mode is True
        assert caps.thinking_disable_prefix == "/no_think"
        assert caps.exploration_multiplier == 2.5
        assert caps.continuation_patience == 3
        assert caps.timeout_multiplier == 1.8

    def test_openai_compat_lmstudio_branch_preserves_configured_fields(self, monkeypatch):
        from types import SimpleNamespace

        import victor.agent.tool_calling.adapters as adapters_mod
        from victor.agent.tool_calling.adapters import OpenAICompatToolCallingAdapter
        from victor.agent.tool_calling.base import ToolCallingCapabilities

        configured = ToolCallingCapabilities(
            thinking_disable_prefix="/no_think",
            exploration_multiplier=2.5,
            continuation_patience=3,
            timeout_multiplier=1.8,
        )
        monkeypatch.setattr(
            adapters_mod._CapabilityLoaderHolder,
            "_instance",
            SimpleNamespace(get_capabilities=lambda *a, **k: configured),
        )
        adapter = OpenAICompatToolCallingAdapter(
            model="qwen3-coder", config=None, provider_variant="lmstudio"
        )
        caps = adapter.get_capabilities()
        assert caps.thinking_mode is True
        assert caps.thinking_disable_prefix == "/no_think"
        assert caps.exploration_multiplier == 2.5
        assert caps.continuation_patience == 3
        assert caps.timeout_multiplier == 1.8


class TestNativeParseAdoptionContract:
    """The four adopting adapters (OpenAI, OpenAICompat, DeepSeek, Azure)
    route native parsing through the shared base helper (jscpd 4-copy
    consolidation). One contract, driven through each adapter's public
    method: valid calls parse, invalid names are KEPT for the spec's error
    response, bad JSON degrades to empty args with a warning."""

    @staticmethod
    def _raw():
        return [
            {"id": "call_1", "name": "read", "arguments": {"path": "f.py"}},
            {"id": "call_2", "name": "example_bad", "arguments": {}},
            {"id": "call_3", "name": "shell", "arguments": "{broken json"},
        ]

    def _assert_contract(self, adapter, parse_method="native"):
        result = adapter.parse_tool_calls(content="", raw_tool_calls=self._raw())
        names = [t.name for t in result.tool_calls]
        # Invalid name KEPT (spec error response), bad-JSON call kept with {}.
        assert names == ["read", "example_bad", "shell"]
        assert result.tool_calls[0].arguments == {"path": "f.py"}
        assert result.tool_calls[2].arguments == {}
        assert result.parse_method == parse_method
        assert any("example_bad" in w and "Skipped" in w for w in result.warnings)
        assert any("shell" in w and "Failed to parse JSON arguments" in w for w in result.warnings)
        assert isinstance(result.warnings, list)

    def test_openai_adopts_base_helper(self):
        adapter = OpenAIToolCallingAdapter(model="gpt-4", config=None)
        adapter._valid_tool_names = {"read", "shell"}
        self._assert_contract(adapter)

    def test_openai_compat_adopts_base_helper(self):
        adapter = OpenAICompatToolCallingAdapter(model="qwen", config=None, provider_variant="vllm")
        adapter._valid_tool_names = {"read", "shell"}
        self._assert_contract(adapter)

    def test_deepseek_adopts_base_helper(self):
        adapter = DeepSeekToolCallingAdapter(model="deepseek-chat", config=None)
        adapter._valid_tool_names = {"read", "shell"}
        self._assert_contract(adapter)

    def test_azure_adopts_base_helper_and_always_list(self):
        adapter = AzureOpenAIToolCallingAdapter(model="gpt-4", config=None)
        adapter._valid_tool_names = {"read", "shell"}
        self._assert_contract(adapter)

    def test_azure_clean_parse_returns_list_not_none(self):
        # The convention fix: no warnings -> empty LIST, never None.
        adapter = AzureOpenAIToolCallingAdapter(model="gpt-4", config=None)
        adapter._valid_tool_names = {"read"}
        result = adapter.parse_tool_calls(
            content="", raw_tool_calls=[{"id": "c", "name": "read", "arguments": {}}]
        )
        assert result.warnings == []

    def test_openai_no_raw_calls_returns_empty(self):
        adapter = OpenAIToolCallingAdapter(model="gpt-4", config=None)
        result = adapter.parse_tool_calls(content="answer text", raw_tool_calls=None)
        assert result.tool_calls == []
        assert result.remaining_content == "answer text"


class TestAdoptionEdgeCoverage:
    """Edge paths of the shared-helper adoption and the moved base methods
    (diff-coverage: these lines execute only through specific inputs)."""

    def test_openai_empty_name_kept_with_invalid_warning(self):
        # Base contract discovered by probe: the validator rejects '' FIRST
        # (invalid-name keep branch), so an empty-named call is KEPT for the
        # spec error response with an 'Skipped invalid tool name' warning —
        # the empty-name drop branch only fires without a validator.
        adapter = OpenAIToolCallingAdapter(model="gpt-4", config=None)
        result = adapter.parse_tool_calls(
            content="", raw_tool_calls=[{"id": "c", "name": "", "arguments": {}}]
        )
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "c"
        assert any("Skipped invalid tool name" in w for w in result.warnings)

    def test_openai_no_raw_calls_terminal_path(self):
        # raw_tool_calls=None: the terminal passthrough returns empty with
        # the original content.
        adapter = OpenAIToolCallingAdapter(model="gpt-4", config=None)
        result = adapter.parse_tool_calls(content="plain answer", raw_tool_calls=None)
        assert result.tool_calls == []
        assert result.remaining_content == "plain answer"

    def test_azure_o1_model_returns_early_without_tools(self):
        from victor.agent.tool_calling.adapters import AzureOpenAIToolCallingAdapter

        adapter = AzureOpenAIToolCallingAdapter(model="o1-preview", config=None)
        result = adapter.parse_tool_calls(
            content="hello", raw_tool_calls=[{"id": "c", "name": "read", "arguments": {}}]
        )
        assert result.tool_calls == []
        assert result.remaining_content == "hello"

    def test_is_valid_tool_name_rejects_hallucination_patterns(self):
        from victor.agent.tool_calling.adapters import OpenAIToolCallingAdapter

        adapter = OpenAIToolCallingAdapter(model="gpt-4", config=None)
        for bad in [
            "example_foo",
            "function_bar",
            "tool_name",
            "my_thing",
            "9tool",
            "has space",
            "<tag>",
            "trail/",
        ]:
            assert adapter.is_valid_tool_name(bad) is False, bad
        assert adapter.is_valid_tool_name("read") is True
        assert adapter.is_valid_tool_name("") is False
        assert adapter.is_valid_tool_name(None) is False  # type: ignore[arg-type]

    def test_sanitize_content_strips_patterns_and_whitespace(self):
        from victor.agent.tool_calling.adapters import OpenAIToolCallingAdapter

        adapter = OpenAIToolCallingAdapter(model="gpt-4", config=None)
        dirty = "answer\n\n\n\n\nmore   text"
        cleaned = adapter.sanitize_content(dirty)
        assert "\n\n\n\n" not in cleaned
        assert cleaned == cleaned.strip()


class TestBaseHelperFullBranchCoverage:
    """The moved base methods' remaining branches (diff-coverage): the
    helper's empty-input early return, the nested-function vs direct-format
    parse, the empty-name drop (no-validator path), parse_json_arguments'
    non-dict / bad-JSON / unexpected-type branches, and sanitize_content."""

    def test_helper_empty_input_returns_empty_result(self):
        from victor.agent.tool_calling.adapters import OpenAIToolCallingAdapter

        adapter = OpenAIToolCallingAdapter(model="gpt-4", config=None)
        result = adapter.parse_native_tool_calls([], adapter.is_valid_tool_name)
        assert result.tool_calls == []
        assert result.warnings == []

    def test_helper_direct_and_nested_formats_without_validator(self):
        from victor.agent.tool_calling.adapters import OpenAIToolCallingAdapter

        adapter = OpenAIToolCallingAdapter(model="gpt-4", config=None)
        # No validator: nested-function AND direct formats both parse; an
        # empty-name entry takes the drop branch (unreachable with a
        # validator, which is why this test passes validator=None).
        result = adapter.parse_native_tool_calls(
            [
                {"id": "n", "function": {"name": "read", "arguments": {"p": 1}}},
                {"id": "d", "name": "shell", "arguments": {}},
                {"id": "e", "name": "", "arguments": {}},
            ],
            validate_name_fn=None,
        )
        assert [t.name for t in result.tool_calls] == ["read", "shell"]
        assert any("empty name" in w for w in result.warnings)

    def test_parse_json_arguments_branches(self):
        from victor.agent.tool_calling.adapters import OpenAIToolCallingAdapter

        adapter = OpenAIToolCallingAdapter(model="gpt-4", config=None)
        # dict passthrough
        assert adapter.parse_json_arguments({"a": 1}) == ({"a": 1}, None)
        # valid JSON string
        assert adapter.parse_json_arguments('{"b": 2}') == ({"b": 2}, None)
        # valid JSON but not a dict
        args, warn = adapter.parse_json_arguments("[1, 2]")
        assert args == {} and "not a dict" in warn
        # invalid JSON string
        args, warn = adapter.parse_json_arguments("{broken")
        assert args == {} and warn == "Failed to parse JSON arguments"
        # unexpected type
        args, warn = adapter.parse_json_arguments(42)
        assert args == {} and "Unexpected argument type" in warn

    def test_azure_clean_terminal_return(self):
        # Azure's terminal return (line ~1865): no raw calls, no content.
        from victor.agent.tool_calling.adapters import AzureOpenAIToolCallingAdapter

        adapter = AzureOpenAIToolCallingAdapter(model="gpt-4", config=None)
        result = adapter.parse_tool_calls(content="", raw_tool_calls=None)
        assert result.tool_calls == []
        assert result.warnings == []
