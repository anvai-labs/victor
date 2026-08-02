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

"""Tests for tool_calling base module."""

from typing import Any, Dict, List, Optional

from victor.agent.tool_calling.base import (
    ToolCallFormat,
    ToolCallingCapabilities,
    ToolCall,
    ToolCallParseResult,
    BaseToolCallingAdapter,
)
from victor.providers.base import ToolDefinition


class ConcreteToolCallingAdapter(BaseToolCallingAdapter):
    """Concrete implementation for testing abstract base class."""

    def __init__(
        self,
        model: str = "",
        config: Optional[Dict[str, Any]] = None,
        capabilities: Optional[ToolCallingCapabilities] = None,
    ):
        super().__init__(model, config)
        self._capabilities = capabilities or ToolCallingCapabilities()

    @property
    def provider_name(self) -> str:
        return "test_provider"

    def get_capabilities(self) -> ToolCallingCapabilities:
        return self._capabilities

    def convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        return [{"name": t.name, "description": t.description} for t in tools]

    def parse_tool_calls(
        self,
        content: str,
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolCallParseResult:
        return ToolCallParseResult()


class TestToolCallFormat:
    """Tests for ToolCallFormat enum."""

    def test_format_values(self):
        """Test all format values exist."""
        assert ToolCallFormat.OPENAI.value == "openai"
        assert ToolCallFormat.ANTHROPIC.value == "anthropic"
        assert ToolCallFormat.OLLAMA_NATIVE.value == "ollama_native"
        assert ToolCallFormat.OLLAMA_JSON.value == "ollama_json"
        assert ToolCallFormat.LMSTUDIO_NATIVE.value == "lmstudio_native"
        assert ToolCallFormat.LMSTUDIO_DEFAULT.value == "lmstudio_default"
        assert ToolCallFormat.VLLM.value == "vllm"
        assert ToolCallFormat.XML.value == "xml"
        assert ToolCallFormat.UNKNOWN.value == "unknown"


class TestToolCallingCapabilities:
    """Tests for ToolCallingCapabilities dataclass."""

    def test_default_values(self):
        """Test default capability values."""
        caps = ToolCallingCapabilities()
        assert caps.native_tool_calls is False
        assert caps.streaming_tool_calls is False
        assert caps.parallel_tool_calls is False
        assert caps.tool_choice_param is False
        assert caps.json_fallback_parsing is False
        assert caps.xml_fallback_parsing is False
        assert caps.thinking_mode is False
        assert caps.requires_strict_prompting is False
        assert caps.tool_call_format == ToolCallFormat.UNKNOWN
        assert caps.argument_format == "json"
        assert caps.recommended_max_tools == 20
        assert caps.recommended_tool_budget == 12

    def test_custom_values(self):
        """Test custom capability values."""
        caps = ToolCallingCapabilities(
            native_tool_calls=True,
            streaming_tool_calls=True,
            parallel_tool_calls=True,
            tool_call_format=ToolCallFormat.OPENAI,
            recommended_max_tools=10,
        )
        assert caps.native_tool_calls is True
        assert caps.streaming_tool_calls is True
        assert caps.parallel_tool_calls is True
        assert caps.tool_call_format == ToolCallFormat.OPENAI
        assert caps.recommended_max_tools == 10


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_basic_tool_call(self):
        """Test basic ToolCall creation."""
        tc = ToolCall(name="test_tool", arguments={"x": 1})
        assert tc.name == "test_tool"
        assert tc.arguments == {"x": 1}
        assert tc.id is None

    def test_tool_call_with_id(self):
        """Test ToolCall with ID."""
        tc = ToolCall(name="test_tool", arguments={}, id="call_123")
        assert tc.id == "call_123"

    def test_to_dict_without_id(self):
        """Test to_dict without ID (covers lines 89-92)."""
        tc = ToolCall(name="my_tool", arguments={"arg": "value"})
        result = tc.to_dict()
        assert result == {"name": "my_tool", "arguments": {"arg": "value"}}
        assert "id" not in result

    def test_to_dict_with_id(self):
        """Test to_dict with ID (covers lines 89-92)."""
        tc = ToolCall(name="my_tool", arguments={"arg": "value"}, id="call_456")
        result = tc.to_dict()
        assert result == {
            "name": "my_tool",
            "arguments": {"arg": "value"},
            "id": "call_456",
        }

    def test_from_dict_basic(self):
        """Test from_dict basic (covers line 97)."""
        data = {"name": "test_tool", "arguments": {"x": 1}}
        tc = ToolCall.from_dict(data)
        assert tc.name == "test_tool"
        assert tc.arguments == {"x": 1}
        assert tc.id is None

    def test_from_dict_with_id(self):
        """Test from_dict with ID."""
        data = {"name": "test_tool", "arguments": {"y": 2}, "id": "call_789"}
        tc = ToolCall.from_dict(data)
        assert tc.name == "test_tool"
        assert tc.arguments == {"y": 2}
        assert tc.id == "call_789"

    def test_from_dict_missing_fields(self):
        """Test from_dict with missing fields uses defaults."""
        data = {}
        tc = ToolCall.from_dict(data)
        assert tc.name == ""
        assert tc.arguments == {}
        assert tc.id is None


class TestToolCallParseResult:
    """Tests for ToolCallParseResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = ToolCallParseResult()
        assert result.tool_calls == []
        assert result.remaining_content == ""
        assert result.parse_method == "none"
        assert result.confidence == 1.0
        assert result.warnings == []

    def test_custom_values(self):
        """Test custom values."""
        tool_calls = [ToolCall(name="test", arguments={})]
        result = ToolCallParseResult(
            tool_calls=tool_calls,
            remaining_content="text",
            parse_method="native",
            confidence=0.95,
            warnings=["warning1"],
        )
        assert len(result.tool_calls) == 1
        assert result.remaining_content == "text"
        assert result.parse_method == "native"
        assert result.confidence == 0.95
        assert result.warnings == ["warning1"]


class TestBaseToolCallingAdapterInit:
    """Tests for BaseToolCallingAdapter initialization."""

    def test_init_with_model(self):
        """Test adapter initialization with model."""
        adapter = ConcreteToolCallingAdapter(model="test-model")
        assert adapter.model == "test-model"
        assert adapter.model_lower == "test-model"
        assert adapter.config == {}

    def test_init_with_uppercase_model(self):
        """Test model_lower is lowercase."""
        adapter = ConcreteToolCallingAdapter(model="Test-Model-V1")
        assert adapter.model == "Test-Model-V1"
        assert adapter.model_lower == "test-model-v1"

    def test_init_with_config(self):
        """Test adapter initialization with config."""
        config = {"timeout": 30, "max_retries": 3}
        adapter = ConcreteToolCallingAdapter(model="model", config=config)
        assert adapter.config == config

    def test_init_empty_model(self):
        """Test adapter with empty model."""
        adapter = ConcreteToolCallingAdapter(model="")
        assert adapter.model == ""
        assert adapter.model_lower == ""


class TestBaseToolCallingAdapterGetSystemPromptHints:
    """Tests for get_system_prompt_hints method."""

    def test_native_no_hints(self):
        """Test native tool calls with no strict prompting returns empty."""
        caps = ToolCallingCapabilities(native_tool_calls=True, requires_strict_prompting=False)
        adapter = ConcreteToolCallingAdapter(capabilities=caps)
        hints = adapter.get_system_prompt_hints()
        assert hints == ""

    def test_strict_prompting_hints(self):
        """Test strict prompting adds hints (covers lines 205-208)."""
        caps = ToolCallingCapabilities(native_tool_calls=False, requires_strict_prompting=True)
        adapter = ConcreteToolCallingAdapter(capabilities=caps)
        hints = adapter.get_system_prompt_hints()
        assert "ONE AT A TIME" in hints
        assert "2-3 tool calls" in hints
        assert "JSON" in hints

    def test_thinking_mode_hints(self):
        """Test thinking mode adds hints (covers lines 210-211)."""
        caps = ToolCallingCapabilities(native_tool_calls=False, thinking_mode=True)
        adapter = ConcreteToolCallingAdapter(capabilities=caps)
        hints = adapter.get_system_prompt_hints()
        assert "/no_think" in hints

    def test_combined_hints(self):
        """Test combined strict prompting and thinking mode."""
        caps = ToolCallingCapabilities(
            native_tool_calls=False,
            requires_strict_prompting=True,
            thinking_mode=True,
        )
        adapter = ConcreteToolCallingAdapter(capabilities=caps)
        hints = adapter.get_system_prompt_hints()
        assert "ONE AT A TIME" in hints
        assert "/no_think" in hints


class TestBaseToolCallingAdapterNormalizeArguments:
    """Tests for normalize_arguments method."""

    def test_default_returns_unchanged(self):
        """Test default implementation returns arguments unchanged (covers line 228)."""
        adapter = ConcreteToolCallingAdapter()
        args = {"path": "/test", "content": "hello"}
        result = adapter.normalize_arguments(args, "write_file")
        # Result should contain same key/values (but may be a new dict due to filtering)
        assert result == args


class TestBaseToolCallingAdapterIsValidToolName:
    """Tests for is_valid_tool_name method (covers lines 241-270)."""

    def test_valid_tool_names(self):
        """Test valid tool names."""
        adapter = ConcreteToolCallingAdapter()
        assert adapter.is_valid_tool_name("read_file") is True
        assert adapter.is_valid_tool_name("write_file") is True
        assert adapter.is_valid_tool_name("listDir") is True
        assert adapter.is_valid_tool_name("a") is True
        assert adapter.is_valid_tool_name("tool123") is True

    def test_invalid_empty_name(self):
        """Test empty name is invalid."""
        adapter = ConcreteToolCallingAdapter()
        assert adapter.is_valid_tool_name("") is False
        assert adapter.is_valid_tool_name(None) is False

    def test_invalid_non_string(self):
        """Test non-string is invalid."""
        adapter = ConcreteToolCallingAdapter()
        assert adapter.is_valid_tool_name(123) is False
        assert adapter.is_valid_tool_name(["test"]) is False

    def test_invalid_hallucinated_patterns(self):
        """Test hallucinated patterns are rejected."""
        adapter = ConcreteToolCallingAdapter()
        assert adapter.is_valid_tool_name("example_tool") is False
        assert adapter.is_valid_tool_name("func_name") is False
        assert adapter.is_valid_tool_name("function_call") is False
        assert adapter.is_valid_tool_name("tool_name") is False
        assert adapter.is_valid_tool_name("my_function") is False
        assert adapter.is_valid_tool_name("test_tool") is False
        assert adapter.is_valid_tool_name("sample_data") is False

    def test_invalid_xml_patterns(self):
        """Test XML-like patterns are rejected."""
        adapter = ConcreteToolCallingAdapter()
        assert adapter.is_valid_tool_name("<function>") is False
        assert adapter.is_valid_tool_name("tag>") is False
        assert adapter.is_valid_tool_name("tool/") is False

    def test_invalid_whitespace(self):
        """Test whitespace is rejected."""
        adapter = ConcreteToolCallingAdapter()
        assert adapter.is_valid_tool_name("tool name") is False
        assert adapter.is_valid_tool_name("tool\tname") is False
        assert adapter.is_valid_tool_name("tool\nname") is False

    def test_invalid_starts_with_number(self):
        """Test names starting with number are rejected."""
        adapter = ConcreteToolCallingAdapter()
        assert adapter.is_valid_tool_name("123tool") is False
        assert adapter.is_valid_tool_name("1_tool") is False

    def test_invalid_special_characters(self):
        """Test special characters are rejected."""
        adapter = ConcreteToolCallingAdapter()
        assert adapter.is_valid_tool_name("tool-name") is False
        assert adapter.is_valid_tool_name("tool.name") is False
        assert adapter.is_valid_tool_name("tool@name") is False


class TestBaseToolCallingAdapterSanitizeContent:
    """Tests for sanitize_content method (covers lines 281-311)."""

    def test_empty_content(self):
        """Test empty content returns as-is."""
        adapter = ConcreteToolCallingAdapter()
        assert adapter.sanitize_content("") == ""
        assert adapter.sanitize_content(None) is None

    def test_normal_content_unchanged(self):
        """Test normal content is unchanged."""
        adapter = ConcreteToolCallingAdapter()
        content = "This is a normal response."
        assert adapter.sanitize_content(content) == content

    def test_removes_repeated_closing_tags(self):
        """Test removes repeated closing tags."""
        adapter = ConcreteToolCallingAdapter()
        content = "Text</div></div></div></div>"
        result = adapter.sanitize_content(content)
        assert result.count("</div>") < 3

    def test_removes_orphaned_function_tags(self):
        """Test removes orphaned XML function tags."""
        adapter = ConcreteToolCallingAdapter()
        content = "Result: <function>test</function> data"
        result = adapter.sanitize_content(content)
        assert "<function>" not in result
        assert "</function>" not in result

    def test_removes_orphaned_parameter_tags(self):
        """Test removes orphaned parameter tags."""
        adapter = ConcreteToolCallingAdapter()
        content = "Data <parameter>value</parameter> here"
        result = adapter.sanitize_content(content)
        assert "<parameter>" not in result
        assert "</parameter>" not in result

    def test_removes_orphaned_tool_tags(self):
        """Test removes orphaned tool tags."""
        adapter = ConcreteToolCallingAdapter()
        content = "Output <tool>name</tool> done"
        result = adapter.sanitize_content(content)
        assert "<tool>" not in result
        assert "</tool>" not in result

    def test_removes_important_tags(self):
        """Test removes IMPORTANT tags."""
        adapter = ConcreteToolCallingAdapter()
        content = "Text <IMPORTANT>note</IMPORTANT> more"
        result = adapter.sanitize_content(content)
        assert "<IMPORTANT>" not in result
        assert "</IMPORTANT>" not in result

    def test_removes_instruction_leakage(self):
        """Test removes instruction leakage patterns."""
        adapter = ConcreteToolCallingAdapter()
        content = "Result\nDo NOT include this\nActual output"
        result = adapter.sanitize_content(content)
        assert "Do NOT" not in result

        content2 = "Data\nNEVER do this\nMore data"
        result2 = adapter.sanitize_content(content2)
        assert "NEVER" not in result2

        content3 = "Output\nALWAYS include the header\nResult"
        result3 = adapter.sanitize_content(content3)
        assert "ALWAYS include" not in result3

    def test_reduces_excessive_newlines(self):
        """Test reduces excessive newlines."""
        adapter = ConcreteToolCallingAdapter()
        content = "Line 1\n\n\n\n\n\n\n\nLine 2"
        result = adapter.sanitize_content(content)
        # Should have at most 3 consecutive newlines
        assert "\n\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result

    def test_strips_whitespace(self):
        """Test strips leading/trailing whitespace."""
        adapter = ConcreteToolCallingAdapter()
        content = "   Result   "
        result = adapter.sanitize_content(content)
        assert result == "Result"
