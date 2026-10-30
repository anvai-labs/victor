"""Tests for OpenAI-compatible utilities."""

import json

import pytest

from victor.providers.openai_compat import (
    convert_tools_to_openai_format,
    convert_tools_to_anthropic_format,
    consume_last_tool_message_cleanup_stats,
    convert_messages_to_openai_format,
    parse_openai_tool_calls,
    parse_openai_stream_chunk,
    build_openai_messages,
    fix_orphaned_tool_messages,
)
from victor.providers.base import Message, ToolDefinition


class TestConvertToolsToOpenAIFormat:
    """Tests for convert_tools_to_openai_format."""

    def test_basic_tool_conversion(self):
        """Test converting a basic tool."""
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
            )
        ]
        result = convert_tools_to_openai_format(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "test_tool"
        assert result[0]["function"]["description"] == "A test tool"

    def test_multiple_tools_conversion(self):
        """Test converting multiple tools."""
        tools = [
            ToolDefinition(name="tool1", description="Tool 1", parameters={}),
            ToolDefinition(name="tool2", description="Tool 2", parameters={}),
        ]
        result = convert_tools_to_openai_format(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool1"
        assert result[1]["function"]["name"] == "tool2"

    def test_empty_tools_list(self):
        """Test converting empty tools list."""
        result = convert_tools_to_openai_format([])
        assert result == []


class TestConvertToolsToAnthropicFormat:
    """Tests for convert_tools_to_anthropic_format."""

    def test_basic_tool_conversion(self):
        """Test converting a basic tool to Anthropic format."""
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
            )
        ]
        result = convert_tools_to_anthropic_format(tools)
        assert len(result) == 1
        assert result[0]["name"] == "test_tool"
        assert result[0]["description"] == "A test tool"
        assert "input_schema" in result[0]


class TestConvertMessagesToOpenAIFormat:
    """Tests for convert_messages_to_openai_format."""

    def test_basic_message(self):
        """Test converting a basic message."""
        messages = [Message(role="user", content="Hello")]
        result = convert_messages_to_openai_format(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_message_with_tool_calls(self):
        """Test converting message with tool calls."""
        messages = [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "call_123",
                        "name": "test_func",
                        "arguments": {"arg1": "val1"},
                    }
                ],
            )
        ]
        result = convert_messages_to_openai_format(messages)
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["id"] == "call_123"
        assert result[0]["tool_calls"][0]["function"]["name"] == "test_func"

    def test_tool_response_message(self):
        """Test converting tool response message."""
        msg = Message(role="tool", content="Tool result")
        msg.tool_call_id = "call_123"
        msg.name = "test_func"
        messages = [msg]
        result = convert_messages_to_openai_format(messages)
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_123"
        assert result[0]["name"] == "test_func"

    def test_multiple_messages(self):
        """Test converting multiple messages."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        result = convert_messages_to_openai_format(messages)
        assert len(result) == 2

    def test_empty_content(self):
        """Test message with empty content."""
        messages = [Message(role="assistant", content="")]
        result = convert_messages_to_openai_format(messages)
        assert result[0]["content"] == ""

    def test_supports_all_chat_completion_roles(self):
        messages = [
            Message(role="developer", content="policy", name="app"),
            Message(role="system", content="compatibility policy"),
            Message(role="user", content="question"),
            Message(
                role="assistant",
                content="",
                tool_calls=[{"id": "call_1", "name": "lookup", "arguments": {}}],
            ),
            Message(role="tool", content="result", tool_call_id="call_1"),
            Message(role="function", content="legacy result", name="lookup"),
        ]

        result = convert_messages_to_openai_format(messages)

        assert [message["role"] for message in result] == [
            "developer",
            "system",
            "user",
            "assistant",
            "tool",
            "function",
        ]
        assert result[0]["name"] == "app"
        assert result[4]["tool_call_id"] == "call_1"
        assert result[5]["name"] == "lookup"

    @pytest.mark.parametrize(
        "message,match",
        [
            (Message(role="model", content="x"), "must be one of"),
            (Message(role="tool", content="x"), "requires tool_call_id"),
            (Message(role="function", content="x"), "requires name"),
        ],
    )
    def test_rejects_unknown_or_unlinked_roles(self, message, match):
        with pytest.raises(ValueError, match=match):
            convert_messages_to_openai_format([message])


class TestParseOpenAIToolCalls:
    """Tests for parse_openai_tool_calls."""

    def test_parse_valid_tool_calls(self):
        """Test parsing valid tool calls."""
        data = [
            {
                "id": "call_123",
                "function": {"name": "test_func", "arguments": '{"arg": "value"}'},
            }
        ]
        result = parse_openai_tool_calls(data)
        assert result is not None
        assert len(result) == 1
        assert result[0]["id"] == "call_123"
        assert result[0]["name"] == "test_func"
        assert result[0]["arguments"] == {"arg": "value"}

    def test_parse_invalid_json_arguments(self):
        """Test parsing tool calls with invalid JSON arguments."""
        data = [
            {
                "id": "call_123",
                "function": {"name": "test", "arguments": "invalid json"},
            }
        ]
        result = parse_openai_tool_calls(data)
        assert result is not None
        assert result[0]["arguments"] == {"raw": "invalid json"}

    def test_parse_none_input(self):
        """Test parsing None input."""
        result = parse_openai_tool_calls(None)
        assert result is None

    def test_parse_empty_list(self):
        """Test parsing empty list."""
        result = parse_openai_tool_calls([])
        assert result is None

    def test_parse_dict_arguments(self):
        """Test parsing tool calls with dict arguments (not string)."""
        data = [
            {
                "id": "call_123",
                "function": {"name": "test", "arguments": {"already": "dict"}},
            }
        ]
        result = parse_openai_tool_calls(data)
        assert result[0]["arguments"] == {"already": "dict"}


class TestParseOpenAIStreamChunk:
    """Tests for parse_openai_stream_chunk."""

    def test_parse_content_chunk(self):
        """Test parsing chunk with content."""
        chunk = {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]}
        result = parse_openai_stream_chunk(chunk)
        assert result["content"] == "Hello"
        assert result["finish_reason"] is None

    def test_parse_tool_calls_chunk(self):
        """Test parsing chunk with tool calls."""
        chunk = {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"name": "test"}}]},
                    "finish_reason": None,
                }
            ]
        }
        result = parse_openai_stream_chunk(chunk)
        assert result["tool_calls"] is not None

    def test_parse_finish_chunk(self):
        """Test parsing chunk with finish reason."""
        chunk = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        result = parse_openai_stream_chunk(chunk)
        assert result["finish_reason"] == "stop"

    def test_parse_empty_choices(self):
        """Test parsing chunk with no choices."""
        chunk = {"choices": []}
        result = parse_openai_stream_chunk(chunk)
        assert result["content"] is None
        assert result["tool_calls"] is None
        assert result["finish_reason"] is None

    def test_parse_missing_choices(self):
        """Test parsing chunk without choices key."""
        chunk = {}
        result = parse_openai_stream_chunk(chunk)
        assert result["content"] is None


class TestBuildOpenAIMessages:
    """Tests for build_openai_messages."""

    def test_tool_message_with_name_and_tool_call_id(self):
        """Test that tool messages preserve both tool_call_id and name fields when paired."""
        messages = [
            Message(role="user", content="Use the tool"),
        ]
        # Create assistant message with tool_calls
        assistant_msg = Message(role="assistant", content="")
        assistant_msg.tool_calls = [
            {"id": "call_123", "name": "test_tool", "arguments": {"arg": "value"}}
        ]
        messages.append(assistant_msg)
        # Create a tool message with both tool_call_id and name
        tool_msg = Message(role="tool", content="Result")
        tool_msg.tool_call_id = "call_123"
        tool_msg.name = "test_tool"
        messages.append(tool_msg)

        result = build_openai_messages(messages)

        # Find the tool message in result
        tool_result = [m for m in result if m["role"] == "tool"][0]
        assert tool_result["tool_call_id"] == "call_123"
        assert tool_result["name"] == "test_tool"

    def test_orphaned_tool_message_is_removed(self):
        """Test that orphaned tool messages (no matching tool_call) are removed."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]
        # Create an orphaned tool message (no assistant message with tool_calls)
        tool_msg = Message(role="tool", content="Orphaned result")
        tool_msg.tool_call_id = "orphaned_123"
        tool_msg.name = "test_tool"
        messages.append(tool_msg)

        result = build_openai_messages(messages)
        stats = consume_last_tool_message_cleanup_stats()

        # The orphaned tool message should be removed by fix_orphaned_tool_messages
        assert not any(m["role"] == "tool" for m in result)
        assert stats["history_repaired"] is True
        assert stats["removed_orphaned_tool_responses"] == 1

    def test_paired_tool_call_and_response(self):
        """Test that properly paired tool_calls and tool responses are preserved."""
        messages = [
            Message(role="user", content="Use the tool"),
        ]
        # Create assistant message with tool_calls
        assistant_msg = Message(role="assistant", content="")
        assistant_msg.tool_calls = [
            {"id": "call_abc", "name": "test_tool", "arguments": {"arg": "value"}}
        ]
        messages.append(assistant_msg)

        # Create tool response
        tool_msg = Message(role="tool", content="Success")
        tool_msg.tool_call_id = "call_abc"
        tool_msg.name = "test_tool"
        messages.append(tool_msg)

        result = build_openai_messages(messages)

        # Both should be present
        assert len(result) == 3
        assistant_result = [m for m in result if m["role"] == "assistant"][0]
        tool_result = [m for m in result if m["role"] == "tool"][0]

        assert "tool_calls" in assistant_result
        assert assistant_result["tool_calls"][0]["id"] == "call_abc"
        assert tool_result["tool_call_id"] == "call_abc"
        assert tool_result["name"] == "test_tool"

    def test_no_fallback_tool_call_id_generated(self):
        """Test that no fallback tool_call_id is generated for missing IDs."""
        messages = [
            Message(role="user", content="Hello"),
        ]
        # Create a tool message without tool_call_id
        tool_msg = Message(role="tool", content="Result")
        # tool_call_id is None
        messages.append(tool_msg)

        result = build_openai_messages(messages)
        stats = consume_last_tool_message_cleanup_stats()

        # Tool message should be removed (orphaned) since no tool_call_id
        assert not any(m["role"] == "tool" for m in result)
        assert stats["history_repaired"] is True
        assert stats["skipped_tool_messages_without_id"] == 1

    def test_empty_tool_content_gets_placeholder(self):
        """Test that tool messages with empty content get a placeholder value.

        DeepSeek and some other providers reject tool messages with empty content.
        This test verifies that empty content is replaced with a placeholder.
        """
        messages = [
            Message(role="user", content="Use the tool"),
        ]
        # Create assistant message with tool_calls
        assistant_msg = Message(role="assistant", content="")
        assistant_msg.tool_calls = [{"id": "call_123", "name": "test_tool", "arguments": {}}]
        messages.append(assistant_msg)

        # Create tool message with empty content
        tool_msg = Message(role="tool", content="")
        tool_msg.tool_call_id = "call_123"
        tool_msg.name = "test_tool"
        messages.append(tool_msg)

        result = build_openai_messages(messages)

        # Find the tool message in result
        tool_result = [m for m in result if m["role"] == "tool"][0]
        # Empty content should be replaced with placeholder
        assert tool_result["content"] == "(no output)"
        assert tool_result["tool_call_id"] == "call_123"

    def test_raw_dict_messages_are_accepted(self):
        """OpenAI-compatible callers may pass raw chat dicts."""
        result = build_openai_messages([{"role": "user", "content": "Summarize this"}])

        assert result == [{"role": "user", "content": "Summarize this"}]

    def test_preserves_every_supported_role_and_role_linkage(self):
        messages = [
            {"role": "developer", "content": "policy", "name": "app"},
            {"role": "system", "content": "compatibility policy"},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "result", "tool_call_id": "call_1"},
            {"role": "function", "content": "legacy result", "name": "lookup"},
        ]

        result = build_openai_messages(messages)

        assert [message["role"] for message in result] == [
            "developer",
            "system",
            "user",
            "assistant",
            "tool",
            "function",
        ]
        assert result[0]["name"] == "app"
        assert result[4]["tool_call_id"] == "call_1"
        assert result[5]["name"] == "lookup"

    @pytest.mark.parametrize(
        "message,match",
        [
            ({"role": "model", "content": "x"}, "must be one of"),
            ({"role": "function", "content": "x"}, "requires name"),
        ],
    )
    def test_rejects_unknown_or_invalid_function_role(self, message, match):
        with pytest.raises(ValueError, match=match):
            build_openai_messages([message])

    def test_raw_dict_tool_calls_are_normalized(self):
        """Raw OpenAI-style dict tool calls should survive provider serialization."""
        messages = [
            {"role": "user", "content": "Use the tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_dict",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path": "a.py"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "contents",
                "tool_call_id": "call_dict",
                "name": "read",
            },
        ]

        result = build_openai_messages(messages)

        assert result[1]["tool_calls"][0]["id"] == "call_dict"
        assert result[1]["content"] is None
        assert result[2]["tool_call_id"] == "call_dict"
        assert result[2]["name"] == "read"


class TestFixOrphanedToolMessages:
    """Tests for fix_orphaned_tool_messages."""

    def test_removes_orphaned_tool_response(self):
        """Test that tool responses without matching tool_calls are removed."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "tool_call_id": "orphan_123", "content": "Result"},
        ]

        result = fix_orphaned_tool_messages(messages)

        # Orphaned tool message should be removed
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_removes_tool_calls_without_responses(self):
        """Test that tool_calls without matching responses are stripped."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "test", "arguments": "{}"},
                    }
                ],
            },
        ]

        result = fix_orphaned_tool_messages(messages)

        # tool_calls should be stripped from assistant message
        assert len(result) == 2
        assert result[1]["role"] == "assistant"
        assert "tool_calls" not in result[1]
        assert result[1]["content"] == ""

    def test_preserves_valid_tool_call_pairs(self):
        """Test that valid tool_call/response pairs are preserved."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "test", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "Result"},
        ]

        result = fix_orphaned_tool_messages(messages)

        # Both should be preserved
        assert len(result) == 3
        assert "tool_calls" in result[1]
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_123"


# Consolidated from per-provider duplicates (vLLM/llama.cpp) — the extraction
# helpers now live in victor.providers.openai_compat.
from victor.providers.openai_compat import (  # noqa: E402
    extract_thinking_content,
    extract_tool_calls_from_content,
)


class TestExtractThinkingContent:
    """Tests for extract_thinking_content function."""

    def test_extract_thinking(self):
        """Test extracting thinking content."""
        response = "<think>Let me think about this...</think>Here is my answer."
        thinking, content = extract_thinking_content(response)
        assert thinking == "Let me think about this..."
        assert content == "Here is my answer."

    def test_extract_multiple_thinking_blocks(self):
        """Test extracting multiple thinking blocks."""
        response = "<think>First thought</think>Text<think>Second thought</think>More text"
        thinking, content = extract_thinking_content(response)
        assert "First thought" in thinking
        assert "Second thought" in thinking
        assert "Text" in content
        assert "More text" in content

    def test_no_thinking_tags(self):
        """Test content without thinking tags."""
        response = "Just a normal response"
        thinking, content = extract_thinking_content(response)
        assert thinking == ""
        assert content == "Just a normal response"

    def test_case_insensitive(self):
        """Test case-insensitive tag matching."""
        response = "<THINK>Thinking</THINK>Answer"
        thinking, content = extract_thinking_content(response)
        assert thinking == "Thinking"
        assert content == "Answer"


class TestExtractToolCallsFromContent:
    """Tests for extract_tool_calls_from_content function."""

    def test_extract_json_block(self):
        """Test extracting tool call from JSON code block."""
        content = '```json\n{"name": "read_file", "arguments": {"path": "test.py"}}\n```'
        tool_calls, remaining = extract_tool_calls_from_content(content)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "read_file"
        assert tool_calls[0]["arguments"]["path"] == "test.py"

    def test_extract_tool_output_tags(self):
        """Test extracting tool call from TOOL_OUTPUT tags."""
        content = '<TOOL_OUTPUT>{"name": "search", "arguments": {"query": "test"}}</TOOL_OUTPUT>'
        tool_calls, remaining = extract_tool_calls_from_content(content)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "search"

    def test_extract_inline_json(self):
        """Test extracting inline JSON tool call."""
        content = '{"name": "list_files", "arguments": {"directory": "."}}'
        tool_calls, remaining = extract_tool_calls_from_content(content)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "list_files"
        assert remaining == ""

    def test_no_tool_calls(self):
        """Test content without tool calls."""
        content = "This is just regular text without any tool calls."
        tool_calls, remaining = extract_tool_calls_from_content(content)
        assert len(tool_calls) == 0
        assert remaining == content

    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        content = "```json\n{invalid json here}\n```"
        tool_calls, remaining = extract_tool_calls_from_content(content)
        assert len(tool_calls) == 0

    def test_json_without_name(self):
        """Test JSON that doesn't have name field."""
        content = '{"type": "object", "value": 42}'
        tool_calls, remaining = extract_tool_calls_from_content(content)
        assert len(tool_calls) == 0


class TestProviderPairHelpers:
    """Shared Azure/Hugging Face/Vertex payload, response and SSE helpers."""

    def _messages(self):
        return [
            Message(role="user", content="hi"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {"id": "c1", "name": "read", "arguments": {"path": "a"}},
                    {"id": "c2", "name": "grep", "arguments": '{"q": 1}'},
                ],
            ),
            Message(role="tool", content="r", tool_call_id="c1"),
        ]

    def test_payload_omits_model_unless_given(self):
        from victor.providers.openai_compat import build_provider_request_payload

        without = build_provider_request_payload(self._messages(), 0.2, 50, None, False)
        with_model = build_provider_request_payload(
            self._messages(), 0.2, 50, None, True, model="m"
        )
        assert "model" not in without
        assert with_model["model"] == "m" and with_model["stream"] is True
        assert "tools" not in without and "tool_choice" not in without

    def test_payload_serializes_tool_calls_and_tools(self):
        from victor.providers.openai_compat import build_provider_request_payload

        tools = [ToolDefinition(name="read", description="d", parameters={"type": "object"})]
        payload = build_provider_request_payload(self._messages(), 0.2, 50, tools, False)
        calls = payload["messages"][1]["tool_calls"]
        assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a"}
        assert calls[1]["function"]["arguments"] == '{"q": 1}'
        assert payload["messages"][2]["tool_call_id"] == "c1"
        assert payload["tools"][0]["function"]["name"] == "read"
        assert payload["tool_choice"] == "auto"

    def test_parse_response_recovers_cached_tokens(self):
        from victor.providers.base import CompletionResponse
        from victor.providers.openai_compat import parse_provider_response

        result = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": "{bad"}}],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
        }
        response = parse_provider_response(result, "m", CompletionResponse)
        assert response.content == ""
        assert response.tool_calls == [{"id": "1", "name": "f", "arguments": {}}]
        assert response.usage["cache_read_input_tokens"] == 3
        assert response.stop_reason == "tool_calls"

    def test_parse_response_without_choices_or_usage(self):
        from victor.providers.base import CompletionResponse
        from victor.providers.openai_compat import parse_provider_response

        empty = parse_provider_response({"choices": []}, "m", CompletionResponse)
        assert empty.content == "" and empty.usage is None and empty.model == "m"

    async def test_sse_iterator_skips_noise_and_finishes_on_done(self):
        from victor.providers.base import StreamChunk
        from victor.providers.openai_compat import (
            iter_provider_sse_chunks,
            parse_provider_stream_chunk,
        )

        class _Response:
            async def aiter_lines(self):
                for line in [
                    "",
                    ": keepalive",
                    'data: {"choices":[{"delta":{"content":"he"}}]}',
                    "data: {broken",
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t",'
                    '"function":{"name":"f","arguments":"{}"}}]}}]}',
                    "data: [DONE]",
                    'data: {"choices":[{"delta":{"content":"after"}}]}',
                ]:
                    yield line

        decode_errors = []
        chunks = [
            chunk
            async for chunk in iter_provider_sse_chunks(
                _Response(),
                lambda data, acc: parse_provider_stream_chunk(
                    data, acc, lambda kw: StreamChunk(**kw)
                ),
                StreamChunk,
                on_decode_error=decode_errors.append,
            )
        ]
        assert [c.content for c in chunks] == ["he", "", ""]
        assert decode_errors == ["data: {broken"]
        final = chunks[-1]
        assert final.is_final and final.stop_reason == "stop"
        assert final.tool_calls and final.tool_calls[0]["name"] == "f"
