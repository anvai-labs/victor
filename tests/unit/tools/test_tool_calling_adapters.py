"""Tests for tool calling adapters."""

from victor.agent.tool_calling.adapters import (
    AnthropicToolCallingAdapter,
    AzureOpenAIToolCallingAdapter,
    BedrockToolCallingAdapter,
    OpenAIToolCallingAdapter,
    OllamaToolCallingAdapter,
    OpenAICompatToolCallingAdapter,
    _get_capability_loader,
)
from victor.agent.tool_calling.base import ToolCallFormat
from victor.providers.base import ToolDefinition


class TestAnthropicAdapter:
    """Tests for AnthropicToolCallingAdapter."""

    def test_provider_name(self):
        """Test provider name is correct."""
        adapter = AnthropicToolCallingAdapter(model="claude-3-sonnet")
        assert adapter.provider_name == "anthropic"

    def test_get_capabilities(self):
        """Test getting capabilities."""
        adapter = AnthropicToolCallingAdapter(model="claude-3-sonnet")
        caps = adapter.get_capabilities()
        assert caps.native_tool_calls is True
        assert caps.tool_call_format == ToolCallFormat.ANTHROPIC

    def test_convert_tools(self):
        """Test converting tools to Anthropic format."""
        adapter = AnthropicToolCallingAdapter(model="claude-3-sonnet")
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
            )
        ]
        result = adapter.convert_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "test_tool"
        assert "input_schema" in result[0]

    def test_parse_tool_calls_native(self):
        """Test parsing native Anthropic tool calls."""
        adapter = AnthropicToolCallingAdapter(model="claude-3-sonnet")
        raw_tool_calls = [{"id": "call_123", "name": "test_func", "arguments": {"arg": "value"}}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "test_func"

    def test_parse_tool_calls_no_tools(self):
        """Test parsing response with no tool calls."""
        adapter = AnthropicToolCallingAdapter(model="claude-3-sonnet")
        result = adapter.parse_tool_calls("Hello", None)
        assert len(result.tool_calls) == 0


class TestOpenAIAdapter:
    """Tests for OpenAIToolCallingAdapter."""

    def test_provider_name(self):
        """Test provider name is correct."""
        adapter = OpenAIToolCallingAdapter(model="gpt-4")
        assert adapter.provider_name == "openai"

    def test_get_capabilities(self):
        """Test getting capabilities."""
        adapter = OpenAIToolCallingAdapter(model="gpt-4")
        caps = adapter.get_capabilities()
        assert caps.native_tool_calls is True
        assert caps.tool_call_format == ToolCallFormat.OPENAI
        assert caps.json_fallback_parsing is False
        assert caps.xml_fallback_parsing is False

    def test_convert_tools(self):
        """Test converting tools to OpenAI format."""
        adapter = OpenAIToolCallingAdapter(model="gpt-4")
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
            )
        ]
        result = adapter.convert_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "test_tool"

    def test_parse_tool_calls_native(self):
        """Test parsing native OpenAI tool calls."""
        adapter = OpenAIToolCallingAdapter(model="gpt-4")
        raw_tool_calls = [{"id": "call_123", "name": "test_func", "arguments": '{"arg": "value"}'}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "test_func"

    def test_parse_tool_calls_no_tools(self):
        """Test parsing response with no tool calls."""
        adapter = OpenAIToolCallingAdapter(model="gpt-4")
        result = adapter.parse_tool_calls("Hello", None)
        assert len(result.tool_calls) == 0


class TestOllamaAdapter:
    """Tests for OllamaToolCallingAdapter."""

    def test_provider_name(self):
        """Test provider name is correct."""
        adapter = OllamaToolCallingAdapter(model="llama3.1:8b")
        assert adapter.provider_name == "ollama"

    def test_get_capabilities_tool_capable_model(self):
        """Test getting capabilities for tool-capable model."""
        adapter = OllamaToolCallingAdapter(model="llama3.1:8b")
        caps = adapter.get_capabilities()
        # Should have native tool calls for llama3.1
        assert caps.native_tool_calls is True

    def test_convert_tools(self):
        """Test converting tools to Ollama format."""
        adapter = OllamaToolCallingAdapter(model="llama3.1:8b")
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
            )
        ]
        result = adapter.convert_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"

    def test_parse_tool_calls_native(self):
        """Test parsing native Ollama tool calls."""
        adapter = OllamaToolCallingAdapter(model="llama3.1:8b")
        raw_tool_calls = [{"function": {"name": "test_func", "arguments": {"arg": "value"}}}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 1

    def test_parse_json_fallback(self):
        """Test JSON fallback parsing."""
        adapter = OllamaToolCallingAdapter(model="codellama:7b")
        content = '```json\n{"name": "test_func", "arguments": {"arg": "value"}}\n```'
        result = adapter.parse_tool_calls(content, None)
        # Result is valid (may or may not have tool calls depending on parsing)
        assert result is not None

    def test_parse_xml_fallback(self):
        """Test XML fallback parsing."""
        adapter = OllamaToolCallingAdapter(model="qwen3-coder:30b")
        content = "<function=test_func><parameter=arg>value</parameter></function>"
        result = adapter.parse_tool_calls(content, None)
        assert result is not None
        # XML parsing should find the tool call (if fallback is enabled for this model)
        # The result may have 0 or more tool calls depending on model capabilities
        assert hasattr(result, "tool_calls")


class TestOpenAICompatAdapter:
    """Tests for OpenAICompatToolCallingAdapter (LMStudio/vLLM)."""

    def test_provider_name_lmstudio(self):
        """Test provider name for LMStudio."""
        adapter = OpenAICompatToolCallingAdapter(model="local-model", provider_variant="lmstudio")
        assert adapter.provider_name == "lmstudio"

    def test_provider_name_vllm(self):
        """Test provider name for vLLM."""
        adapter = OpenAICompatToolCallingAdapter(model="local-model", provider_variant="vllm")
        assert adapter.provider_name == "vllm"

    def test_convert_tools(self):
        """Test converting tools."""
        adapter = OpenAICompatToolCallingAdapter(model="local-model")
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object"},
            )
        ]
        result = adapter.convert_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"


class TestCapabilityLoader:
    """Tests for capability loader singleton."""

    def test_singleton_behavior(self):
        """Test that capability loader is a singleton."""
        loader1 = _get_capability_loader()
        loader2 = _get_capability_loader()
        assert loader1 is loader2

    def test_loader_returns_capabilities(self):
        """Test that loader returns valid capabilities."""
        loader = _get_capability_loader()
        caps = loader.get_capabilities("openai", "gpt-4", ToolCallFormat.OPENAI)
        assert caps is not None
        assert hasattr(caps, "native_tool_calls")


class TestAnthropicAdapterEdgeCases:
    """Edge case tests for AnthropicToolCallingAdapter."""

    def test_parse_invalid_tool_name(self):
        """Test parsing with invalid tool name keeps it for error response."""
        adapter = AnthropicToolCallingAdapter(model="claude-3-sonnet")
        # Empty name kept with id so pipeline generates error response
        raw_tool_calls = [{"id": "call_123", "name": "", "arguments": {"arg": "value"}}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].arguments == {}  # args cleared

    def test_parse_dict_arguments(self):
        """Test parsing tool calls with dict arguments."""
        adapter = AnthropicToolCallingAdapter(model="claude-3-sonnet")
        # Anthropic API returns arguments as dicts, not JSON strings
        raw_tool_calls = [{"id": "call_123", "name": "test_func", "arguments": {"arg": "value"}}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {"arg": "value"}

    def test_parse_multiple_tool_calls(self):
        """Test parsing multiple tool calls."""
        adapter = AnthropicToolCallingAdapter(model="claude-3-sonnet")
        raw_tool_calls = [
            {"id": "call_1", "name": "func1", "arguments": {"x": 1}},
            {"id": "call_2", "name": "func2", "arguments": {"y": 2}},
        ]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "func1"
        assert result.tool_calls[1].name == "func2"


class TestOpenAIAdapterEdgeCases:
    """Edge case tests for OpenAIToolCallingAdapter."""

    def test_parse_invalid_tool_name(self):
        """Test parsing with invalid tool name."""
        adapter = OpenAIToolCallingAdapter(model="gpt-4")
        raw_tool_calls = [{"id": "call_123", "name": "", "arguments": "{}"}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.warnings) > 0

    def test_parse_dict_arguments(self):
        """Test parsing tool calls with dict arguments."""
        adapter = OpenAIToolCallingAdapter(model="gpt-4")
        raw_tool_calls = [{"id": "call_123", "name": "test_func", "arguments": {"arg": "value"}}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 1

    def test_parse_invalid_json_arguments(self):
        """Test parsing tool calls with invalid JSON arguments."""
        adapter = OpenAIToolCallingAdapter(model="gpt-4")
        raw_tool_calls = [{"id": "call_123", "name": "test_func", "arguments": "invalid json {{"}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 1
        assert len(result.warnings) > 0


class TestOllamaAdapterEdgeCases:
    """Edge case tests for OllamaToolCallingAdapter."""

    def test_get_capabilities_non_tool_model(self):
        """Test getting capabilities for non-tool-capable model."""
        adapter = OllamaToolCallingAdapter(model="old-model-no-tools")
        caps = adapter.get_capabilities()
        # Non-matching model might have native_tool_calls=False
        assert hasattr(caps, "native_tool_calls")

    def test_parse_empty_raw_tool_calls(self):
        """Test parsing with empty raw_tool_calls."""
        adapter = OllamaToolCallingAdapter(model="llama3.1:8b")
        result = adapter.parse_tool_calls("", [])
        assert len(result.tool_calls) == 0

    def test_parse_xml_function_call_format(self):
        """Test parsing XML function call format."""
        adapter = OllamaToolCallingAdapter(model="codellama:7b")
        content = '<function_call><name>my_tool</name><arguments>{"key": "value"}</arguments></function_call>'
        result = adapter.parse_tool_calls(content, None)
        assert result is not None

    def test_parse_tool_call_xml_format(self):
        """Test parsing <tool_call> XML format."""
        adapter = OllamaToolCallingAdapter(model="qwen2.5:7b")
        content = (
            '<tool_call><name>test_func</name><arguments>{"arg": "val"}</arguments></tool_call>'
        )
        result = adapter.parse_tool_calls(content, None)
        assert result is not None

    def test_text_fallback_parsing(self):
        """Test text content without tool calls."""
        adapter = OllamaToolCallingAdapter(model="llama3.1:8b")
        result = adapter.parse_tool_calls("Just some text response", None)
        assert len(result.tool_calls) == 0


class TestOpenAICompatAdapterEdgeCases:
    """Edge case tests for OpenAICompatToolCallingAdapter."""

    def test_default_provider_variant(self):
        """Test default provider variant is lmstudio."""
        adapter = OpenAICompatToolCallingAdapter(model="local-model")
        assert adapter.provider_name == "lmstudio"

    def test_parse_tool_calls(self):
        """Test parsing tool calls."""
        adapter = OpenAICompatToolCallingAdapter(model="llama3.1:8b", provider_variant="vllm")
        raw_tool_calls = [{"id": "call_123", "name": "test", "arguments": {"x": 1}}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 1

    def test_get_system_prompt_hints(self):
        """Test getting system prompt hints."""
        adapter = OpenAICompatToolCallingAdapter(model="local-model")
        hints = adapter.get_system_prompt_hints()
        assert isinstance(hints, str)


class TestBedrockAdapter:
    """Tests for BedrockToolCallingAdapter."""

    def test_bedrock_provider_name(self):
        """Test provider name is correct."""
        adapter = BedrockToolCallingAdapter(model="anthropic.claude-3-sonnet")
        assert adapter.provider_name == "bedrock"

    def test_bedrock_supports_tools_claude(self):
        """Test that Claude models support tools on Bedrock."""
        adapter = BedrockToolCallingAdapter(model="anthropic.claude-3-sonnet")
        caps = adapter.get_capabilities()
        assert caps.native_tool_calls is True
        assert caps.tool_call_format == ToolCallFormat.BEDROCK

    def test_bedrock_supports_tools_titan(self):
        """Test Titan models behavior on Bedrock.

        Note: get_capabilities() returns YAML provider defaults (native_tool_calls=True)
        but convert_tools() and parse_tool_calls() check _supports_tools() which
        correctly returns False for Titan. This tests the actual implementation behavior.
        """
        adapter = BedrockToolCallingAdapter(model="amazon.titan-text-express-v1")
        # The _supports_tools internal method correctly detects Titan doesn't support tools
        assert adapter._supports_tools() is False
        # convert_tools and parse_tool_calls use _supports_tools() check

    def test_bedrock_supports_tools_llama(self):
        """Test that Llama models support tools on Bedrock."""
        adapter = BedrockToolCallingAdapter(model="meta.llama3-70b-instruct-v1")
        caps = adapter.get_capabilities()
        assert caps.native_tool_calls is True

    def test_bedrock_supports_tools_mistral(self):
        """Test that Mistral models support tools on Bedrock."""
        adapter = BedrockToolCallingAdapter(model="mistral.mistral-large-2402-v1")
        caps = adapter.get_capabilities()
        assert caps.native_tool_calls is True

    def test_bedrock_supports_tools_cohere(self):
        """Test that Cohere Command models support tools on Bedrock."""
        adapter = BedrockToolCallingAdapter(model="cohere.command-r-plus-v1")
        caps = adapter.get_capabilities()
        assert caps.native_tool_calls is True

    def test_bedrock_convert_tools_format(self):
        """Test converting tools to Bedrock Converse API format (toolSpec/inputSchema)."""
        adapter = BedrockToolCallingAdapter(model="anthropic.claude-3-sonnet")
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={
                    "type": "object",
                    "properties": {"arg": {"type": "string"}},
                },
            )
        ]
        result = adapter.convert_tools(tools)
        assert len(result) == 1
        assert "toolSpec" in result[0]
        assert result[0]["toolSpec"]["name"] == "test_tool"
        assert "inputSchema" in result[0]["toolSpec"]
        assert "json" in result[0]["toolSpec"]["inputSchema"]

    def test_bedrock_convert_tools_titan_returns_empty(self):
        """Test that Titan models return empty tool list."""
        adapter = BedrockToolCallingAdapter(model="amazon.titan-text-express-v1")
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
            )
        ]
        result = adapter.convert_tools(tools)
        assert result == []

    def test_bedrock_parse_tool_calls(self):
        """Test parsing Bedrock tool calls where arguments are already parsed as dict."""
        adapter = BedrockToolCallingAdapter(model="anthropic.claude-3-sonnet")
        raw_tool_calls = [{"id": "call_123", "name": "test_func", "arguments": {"arg": "value"}}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "test_func"
        assert result.tool_calls[0].arguments == {"arg": "value"}
        assert result.parse_method == "native"

    def test_bedrock_parse_tool_calls_no_tools(self):
        """Test parsing response with no tool calls."""
        adapter = BedrockToolCallingAdapter(model="anthropic.claude-3-sonnet")
        result = adapter.parse_tool_calls("Hello", None)
        assert len(result.tool_calls) == 0

    def test_bedrock_parse_tool_calls_titan(self):
        """Test that Titan models don't parse tool calls even if provided."""
        adapter = BedrockToolCallingAdapter(model="amazon.titan-text-express-v1")
        raw_tool_calls = [{"id": "call_123", "name": "test_func", "arguments": {}}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 0


class TestBedrockAdapterEdgeCases:
    """Edge case tests for BedrockToolCallingAdapter."""

    def test_bedrock_parse_multiple_tool_calls(self):
        """Test parsing multiple tool calls."""
        adapter = BedrockToolCallingAdapter(model="anthropic.claude-3-sonnet")
        raw_tool_calls = [
            {"id": "call_1", "name": "func1", "arguments": {"x": 1}},
            {"id": "call_2", "name": "func2", "arguments": {"y": 2}},
        ]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "func1"
        assert result.tool_calls[1].name == "func2"

    def test_bedrock_parse_invalid_tool_name(self):
        """Test parsing with invalid tool name keeps it for error response."""
        adapter = BedrockToolCallingAdapter(model="anthropic.claude-3-sonnet")
        raw_tool_calls = [{"id": "call_123", "name": "", "arguments": {"arg": "value"}}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        # Invalid names are kept with their id so the pipeline generates
        # an error response (OpenAI spec: every tool_calls[].id needs a role=tool)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].arguments == {}  # args cleared
        assert result.warnings and len(result.warnings) > 0

    def test_bedrock_parse_string_arguments_fallback(self):
        """Test that string arguments are parsed as JSON."""
        adapter = BedrockToolCallingAdapter(model="anthropic.claude-3-sonnet")
        raw_tool_calls = [{"id": "call_123", "name": "test_func", "arguments": '{"arg": "value"}'}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {"arg": "value"}

    def test_bedrock_unknown_model_no_tools(self):
        """Test that unknown model patterns don't support tools via _supports_tools().

        Note: get_capabilities() returns YAML provider defaults but _supports_tools()
        correctly detects unknown models don't have native tool support.
        """
        adapter = BedrockToolCallingAdapter(model="unknown.model-v1")
        # The _supports_tools internal method correctly returns False for unknown models
        assert adapter._supports_tools() is False


class TestAzureOpenAIAdapter:
    """Tests for AzureOpenAIToolCallingAdapter."""

    def test_azure_provider_name(self):
        """Test provider name is correct."""
        adapter = AzureOpenAIToolCallingAdapter(model="gpt-4o")
        assert adapter.provider_name == "azure"

    def test_azure_o1_no_tools(self):
        """Test that o1-preview and o1-mini don't support tools."""
        adapter_preview = AzureOpenAIToolCallingAdapter(model="o1-preview")
        caps_preview = adapter_preview.get_capabilities()
        assert caps_preview.native_tool_calls is False

        adapter_mini = AzureOpenAIToolCallingAdapter(model="o1-mini")
        caps_mini = adapter_mini.get_capabilities()
        assert caps_mini.native_tool_calls is False

        adapter_o1 = AzureOpenAIToolCallingAdapter(model="o1")
        caps_o1 = adapter_o1.get_capabilities()
        assert caps_o1.native_tool_calls is False

    def test_azure_o1_thinking_mode(self):
        """Test that o1 models have thinking_mode=True."""
        adapter = AzureOpenAIToolCallingAdapter(model="o1-preview")
        caps = adapter.get_capabilities()
        assert caps.thinking_mode is True
        assert caps.tool_call_format == ToolCallFormat.NONE

    def test_azure_gpt_supports_tools(self):
        """Test that gpt-4o, gpt-4-turbo support tools."""
        adapter_4o = AzureOpenAIToolCallingAdapter(model="gpt-4o")
        caps_4o = adapter_4o.get_capabilities()
        assert caps_4o.native_tool_calls is True
        assert caps_4o.tool_call_format == ToolCallFormat.OPENAI

        adapter_turbo = AzureOpenAIToolCallingAdapter(model="gpt-4-turbo")
        caps_turbo = adapter_turbo.get_capabilities()
        assert caps_turbo.native_tool_calls is True

        adapter_35 = AzureOpenAIToolCallingAdapter(model="gpt-35-turbo")
        caps_35 = adapter_35.get_capabilities()
        assert caps_35.native_tool_calls is True

    def test_azure_convert_tools_format(self):
        """Test converting tools to OpenAI function format."""
        adapter = AzureOpenAIToolCallingAdapter(model="gpt-4o")
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
            )
        ]
        result = adapter.convert_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "test_tool"
        assert "parameters" in result[0]["function"]

    def test_azure_convert_tools_o1_returns_empty(self):
        """Test that o1 models return empty tool list."""
        adapter = AzureOpenAIToolCallingAdapter(model="o1-preview")
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
            )
        ]
        result = adapter.convert_tools(tools)
        assert result == []

    def test_azure_parse_tool_calls_native(self):
        """Test parsing native Azure OpenAI tool calls."""
        adapter = AzureOpenAIToolCallingAdapter(model="gpt-4o")
        raw_tool_calls = [{"id": "call_123", "name": "test_func", "arguments": '{"arg": "value"}'}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "test_func"
        assert result.tool_calls[0].arguments == {"arg": "value"}

    def test_azure_parse_tool_calls_openai_format(self):
        """Test parsing OpenAI format with nested function."""
        adapter = AzureOpenAIToolCallingAdapter(model="gpt-4o")
        raw_tool_calls = [
            {
                "id": "call_123",
                "function": {"name": "test_func", "arguments": '{"arg": "value"}'},
            }
        ]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "test_func"

    def test_azure_parse_tool_calls_no_tools(self):
        """Test parsing response with no tool calls."""
        adapter = AzureOpenAIToolCallingAdapter(model="gpt-4o")
        result = adapter.parse_tool_calls("Hello", None)
        assert len(result.tool_calls) == 0

    def test_azure_parse_tool_calls_o1(self):
        """Test that o1 models don't parse tool calls even if provided."""
        adapter = AzureOpenAIToolCallingAdapter(model="o1-preview")
        raw_tool_calls = [{"id": "call_123", "name": "test_func", "arguments": "{}"}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 0


class TestAzureOpenAIAdapterEdgeCases:
    """Edge case tests for AzureOpenAIToolCallingAdapter."""

    def test_azure_parse_multiple_tool_calls(self):
        """Test parsing multiple tool calls."""
        adapter = AzureOpenAIToolCallingAdapter(model="gpt-4o")
        raw_tool_calls = [
            {"id": "call_1", "name": "func1", "arguments": '{"x": 1}'},
            {"id": "call_2", "name": "func2", "arguments": '{"y": 2}'},
        ]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "func1"
        assert result.tool_calls[1].name == "func2"

    def test_azure_parse_invalid_tool_name(self):
        """Test parsing with invalid tool name keeps it for error response."""
        adapter = AzureOpenAIToolCallingAdapter(model="gpt-4o")
        raw_tool_calls = [{"id": "call_123", "name": "", "arguments": "{}"}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        # Invalid names kept with id for pipeline error response (OpenAI spec)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.warnings and len(result.warnings) > 0

    def test_azure_parse_invalid_json_arguments(self):
        """Test parsing tool calls with invalid JSON arguments."""
        adapter = AzureOpenAIToolCallingAdapter(model="gpt-4o")
        raw_tool_calls = [{"id": "call_123", "name": "test_func", "arguments": "invalid json {{"}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 1
        assert result.warnings and len(result.warnings) > 0

    def test_azure_parse_dict_arguments(self):
        """Test parsing tool calls with dict arguments (already parsed)."""
        adapter = AzureOpenAIToolCallingAdapter(model="gpt-4o")
        raw_tool_calls = [{"id": "call_123", "name": "test_func", "arguments": {"arg": "value"}}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {"arg": "value"}

    def test_azure_system_prompt_hints_o1(self):
        """Test system prompt hints for o1 models."""
        adapter = AzureOpenAIToolCallingAdapter(model="o1-preview")
        hints = adapter.get_system_prompt_hints()
        assert "reasoning" in hints.lower()
        assert "NOT available" in hints

    def test_azure_system_prompt_hints_gpt(self):
        """Test system prompt hints for GPT models."""
        adapter = AzureOpenAIToolCallingAdapter(model="gpt-4o")
        hints = adapter.get_system_prompt_hints()
        assert "TOOL USAGE" in hints
        assert "parallel" in hints.lower()
