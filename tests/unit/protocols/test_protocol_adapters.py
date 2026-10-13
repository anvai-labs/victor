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

"""Tests for protocol adapters."""

import pytest
import httpx
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import json
import sys

from victor.integrations.protocol.adapters import (
    DirectProtocolAdapter,
    HTTPProtocolAdapter,
)
from victor.integrations.protocol.interface import (
    ChatMessage,
    ChatResponse,
    ClientStreamChunk,
    ToolCall,
    UndoRedoResult,
    AgentMode,
    AgentStatus,
)
from victor.integrations.search_types import CodeSearchResult

# =============================================================================
# DIRECT PROTOCOL ADAPTER TESTS
# =============================================================================


class TestDirectProtocolAdapterInit:
    """Tests for DirectProtocolAdapter initialization."""

    def test_init_with_orchestrator(self):
        """Test initialization with orchestrator."""
        mock_orch = MagicMock()
        adapter = DirectProtocolAdapter(mock_orch)
        assert adapter._orchestrator is mock_orch


class TestDirectProtocolAdapterChat:
    """Tests for DirectProtocolAdapter chat methods."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked orchestrator."""
        mock_orch = MagicMock()
        return DirectProtocolAdapter(mock_orch)

    @pytest.mark.asyncio
    async def test_chat_basic(self, adapter):
        """Test basic chat functionality."""
        mock_response = MagicMock()
        mock_response.content = "Hello there!"
        mock_response.tool_calls = None
        adapter._orchestrator.chat = AsyncMock(return_value=mock_response)

        messages = [ChatMessage(role="user", content="Hello")]
        response = await adapter.chat(messages)

        assert isinstance(response, ChatResponse)
        assert response.content == "Hello there!"
        adapter._orchestrator.chat.assert_called_once_with("Hello")

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self, adapter):
        """Test chat with tool calls in response."""
        mock_tc = MagicMock()
        mock_tc.id = "tc_123"
        mock_tc.name = "read_file"
        mock_tc.arguments = {"path": "test.py"}

        mock_response = MagicMock()
        mock_response.content = "Let me read that file"
        mock_response.tool_calls = [mock_tc]
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20}
        adapter._orchestrator.chat = AsyncMock(return_value=mock_response)

        messages = [ChatMessage(role="user", content="Read test.py")]
        response = await adapter.chat(messages)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "read_file"
        assert response.tool_calls[0].id == "tc_123"

    @pytest.mark.asyncio
    async def test_chat_empty_messages(self, adapter):
        """Test chat with empty messages."""
        mock_response = MagicMock()
        mock_response.content = "Default response"
        mock_response.tool_calls = None
        adapter._orchestrator.chat = AsyncMock(return_value=mock_response)

        response = await adapter.chat([])

        adapter._orchestrator.chat.assert_called_once_with("")

    @pytest.mark.asyncio
    async def test_stream_chat(self, adapter):
        """Test streaming chat."""

        async def mock_stream(msg):
            chunks = [
                MagicMock(content="Hello", finish_reason=None),
                MagicMock(content=" world", finish_reason=None),
                MagicMock(content="!", finish_reason="stop"),
            ]
            for chunk in chunks:
                yield chunk

        adapter._orchestrator.stream_chat = mock_stream

        messages = [ChatMessage(role="user", content="Hi")]
        chunks = []
        async for chunk in adapter.stream_chat(messages):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[2].finish_reason == "stop"


class TestDirectProtocolAdapterConversation:
    """Tests for DirectProtocolAdapter conversation management."""

    @pytest.fixture
    def adapter(self):
        mock_orch = MagicMock()
        return DirectProtocolAdapter(mock_orch)

    @pytest.mark.asyncio
    async def test_reset_conversation(self, adapter):
        """Test reset conversation."""
        await adapter.reset_conversation()
        adapter._orchestrator.reset_conversation.assert_called_once()


class TestDirectProtocolAdapterSearch:
    """Tests for DirectProtocolAdapter search methods."""

    @pytest.fixture
    def adapter(self):
        mock_orch = MagicMock()
        return DirectProtocolAdapter(mock_orch)

    @pytest.mark.asyncio
    async def test_semantic_search_success(self, adapter):
        """Test successful semantic search."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {
            "matches": [{"file": "test.py", "line": 10, "content": "def test()", "score": 0.9}]
        }

        # Create mock module and class
        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value=mock_result)
        mock_class = MagicMock(return_value=mock_tool)
        mock_module = MagicMock(SemanticCodeSearchTool=mock_class)

        with patch.dict(sys.modules, {"victor.tools.semantic_search": mock_module}):
            results = await adapter.semantic_search("test function", max_results=5)

            assert len(results) == 1
            assert results[0].file == "test.py"
            assert results[0].score == 0.9

    @pytest.mark.asyncio
    async def test_semantic_search_failure(self, adapter):
        """Test failed semantic search."""
        mock_result = MagicMock()
        mock_result.success = False

        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value=mock_result)
        mock_class = MagicMock(return_value=mock_tool)
        mock_module = MagicMock(SemanticCodeSearchTool=mock_class)

        with patch.dict(sys.modules, {"victor.tools.semantic_search": mock_module}):
            results = await adapter.semantic_search("test")

            assert results == []

    @pytest.mark.asyncio
    async def test_code_search_success(self, adapter):
        """Test successful code search."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {"matches": [{"file": "main.py", "line": 5, "content": "import os"}]}

        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value=mock_result)
        mock_class = MagicMock(return_value=mock_tool)
        mock_module = MagicMock(CodeSearchTool=mock_class)

        with patch.dict(sys.modules, {"victor.tools.code_search": mock_module}):
            results = await adapter.code_search("import os", regex=True)

            assert len(results) == 1
            assert results[0].file == "main.py"
            assert results[0].score == 1.0  # Exact matches


class TestDirectProtocolAdapterModel:
    """Tests for DirectProtocolAdapter model switching."""

    @pytest.fixture
    def adapter(self):
        mock_orch = MagicMock()
        return DirectProtocolAdapter(mock_orch)

    @pytest.mark.asyncio
    async def test_switch_model(self, adapter):
        """Test model switching."""
        with patch("victor.agent.model_switcher.get_model_switcher") as mock_get:
            mock_switcher = MagicMock()
            mock_get.return_value = mock_switcher

            await adapter.switch_model("anthropic", "claude-3-opus")

            mock_switcher.switch.assert_called_once_with("anthropic", "claude-3-opus")
            assert adapter._pending_provider == "anthropic"
            assert adapter._pending_model == "claude-3-opus"

    @pytest.mark.asyncio
    async def test_switch_mode(self, adapter):
        """Test mode switching."""
        adapter._orchestrator.set_mode = MagicMock()

        await adapter.switch_mode(AgentMode.EXPLORE)

        adapter._orchestrator.set_mode.assert_called_once_with("explore")


class TestDirectProtocolAdapterStatus:
    """Tests for DirectProtocolAdapter status methods."""

    @pytest.fixture
    def adapter(self):
        mock_orch = MagicMock()
        mock_orch.provider.name = "openai"
        mock_orch.provider.model = "gpt-4"
        mock_orch.tools = ["read", "write"]
        mock_orch.messages = [{"role": "user", "content": "hi"}]
        # ModeAwareMixin property used by get_status()
        mock_orch.current_mode_name = "build"
        return DirectProtocolAdapter(mock_orch)

    @pytest.mark.asyncio
    async def test_get_status(self, adapter):
        """Test getting status.

        Uses orchestrator's current_mode_name property (via ModeAwareMixin).
        """
        status = await adapter.get_status()

        assert isinstance(status, AgentStatus)
        assert status.provider == "openai"
        assert status.mode == AgentMode.BUILD
        assert status.connected is True
        assert status.tools_available == 2


class TestDirectProtocolAdapterUndoRedo:
    """Tests for DirectProtocolAdapter undo/redo."""

    @pytest.fixture
    def adapter(self):
        mock_orch = MagicMock()
        return DirectProtocolAdapter(mock_orch)

    @pytest.mark.asyncio
    async def test_undo_success(self, adapter):
        """Test successful undo."""
        adapter._orchestrator.change_tracker.undo.return_value = {
            "success": True,
            "message": "Undone",
            "files": ["test.py"],
        }

        result = await adapter.undo()

        assert result.success is True
        assert result.message == "Undone"
        assert "test.py" in result.files_modified

    @pytest.mark.asyncio
    async def test_undo_no_tracker(self, adapter):
        """Test undo without change tracker."""
        del adapter._orchestrator.change_tracker

        result = await adapter.undo()

        assert result.success is False
        assert "not available" in result.message

    @pytest.mark.asyncio
    async def test_redo_success(self, adapter):
        """Test successful redo."""
        adapter._orchestrator.change_tracker.redo.return_value = {
            "success": True,
            "message": "Redone",
            "files": ["main.py"],
        }

        result = await adapter.redo()

        assert result.success is True
        assert result.message == "Redone"

    @pytest.mark.asyncio
    async def test_get_history(self, adapter):
        """Test getting history."""
        adapter._orchestrator.change_tracker.get_history.return_value = [
            {"id": 1, "action": "edit"}
        ]

        history = await adapter.get_history(limit=5)

        assert len(history) == 1
        adapter._orchestrator.change_tracker.get_history.assert_called_once_with(5)


class TestDirectProtocolAdapterPatch:
    """Tests for DirectProtocolAdapter patch operations."""

    @pytest.fixture
    def adapter(self):
        mock_orch = MagicMock()
        return DirectProtocolAdapter(mock_orch)

    @pytest.mark.asyncio
    async def test_apply_patch(self, adapter):
        """Test applying patch."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {"files_modified": ["test.py"]}

        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value=mock_result)
        mock_class = MagicMock(return_value=mock_tool)
        mock_module = MagicMock(PatchTool=mock_class)

        with patch.dict(sys.modules, {"victor.tools.patch_tool": mock_module}):
            result = await adapter.apply_patch("--- a\n+++ b", dry_run=False)

            assert result["success"] is True
            assert "test.py" in result["files_modified"]


class TestDirectProtocolAdapterClose:
    """Tests for DirectProtocolAdapter close."""

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing adapter."""
        mock_orch = MagicMock()
        mock_orch.provider.close = AsyncMock()
        adapter = DirectProtocolAdapter(mock_orch)

        await adapter.close()

        mock_orch.provider.close.assert_called_once()


# =============================================================================
# HTTP PROTOCOL ADAPTER TESTS
# =============================================================================


class TestHTTPProtocolAdapterInit:
    """Tests for HTTPProtocolAdapter initialization."""

    def test_init_default(self):
        """Test default initialization."""
        adapter = HTTPProtocolAdapter()
        assert adapter._base_url == "http://localhost:8765"
        assert adapter._timeout == 60.0

    def test_init_custom(self):
        """Test custom initialization."""
        adapter = HTTPProtocolAdapter(base_url="http://localhost:9000/", timeout=30.0)
        assert adapter._base_url == "http://localhost:9000"
        assert adapter._timeout == 30.0


class _WireStream(httpx.AsyncByteStream):
    def __init__(self, parts, wait=False, forbid_next_read=False):
        self.parts = parts
        self.closed = False
        self.wait = wait
        self.forbid_next_read = forbid_next_read
        self.waiting = asyncio.Event()

    async def __aiter__(self):
        for part in self.parts:
            yield part
        if self.forbid_next_read:
            raise AssertionError("Read past complete terminal frame")
        if self.wait:
            self.waiting.set()
            await asyncio.Future()

    async def aclose(self):
        self.closed = True


@pytest.fixture
async def http_adapter():
    adapters = []

    async def create(
        *,
        payload=None,
        status=200,
        parts=None,
        content_type="text/event-stream",
        wait=False,
        forbid_next_read=False,
    ):
        requests = []
        stream = _WireStream(parts or [], wait, forbid_next_read)

        def handler(request):
            requests.append(request)
            if parts is None:
                return httpx.Response(status, json=payload)
            return httpx.Response(status, headers={"content-type": content_type}, stream=stream)

        adapter = HTTPProtocolAdapter()
        await adapter._client.aclose()
        adapter._client = httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        )
        adapters.append(adapter)
        return adapter, requests, stream

    yield create
    for adapter in adapters:
        await adapter.close()


class TestHTTPProtocolAdapterChat:
    """Tests for HTTPProtocolAdapter chat methods."""

    @pytest.fixture
    def adapter(self):
        return HTTPProtocolAdapter()

    @pytest.mark.parametrize(
        "status,metadata",
        [
            (200, {}),
            (202, {"status": "awaiting_approval", "run_id": "run-1", "approval_request": {}}),
        ],
    )
    async def test_chat(self, http_adapter, status, metadata):
        adapter, requests, _ = await http_adapter(
            payload={"content": "Hello!", "tool_calls": None, **metadata},
            status=status,
        )
        response = await adapter.chat([ChatMessage(role="user", content="Hi")])
        assert response.content == "Hello!"
        assert response.finish_reason == ("awaiting_approval" if status == 202 else "stop")
        assert len(requests) == 1
        assert json.loads(requests[0].content)["messages"][0]["content"] == "Hi"

    @pytest.mark.parametrize(
        "status,payload",
        [
            (202, {"content": "partial"}),
            (202, {"content": "partial", "status": "ok"}),
            (200, []),
            (200, {"content": 9}),
            (200, {"content": "partial", "status": "failed"}),
        ],
    )
    async def test_invalid_chat_outcome_never_replays(self, http_adapter, status, payload):
        adapter, requests, _ = await http_adapter(payload=payload, status=status)
        with pytest.raises(ValueError):
            await adapter.chat([])
        assert len(requests) == 1

    @pytest.mark.parametrize("ending", [b"\n", b"\r\n", b"\r"])
    async def test_stream_framing_and_tool_list(self, http_adapter, ending):
        frames = [
            b"\xef\xbb\xbf: heartbeat",
            b'data:{"type":"request","request_id":"r"}',
            b"",
            'data: {"type":"content",'.encode(),
            'data:"content":"hello 🧪"}'.encode(),
            b"",
            b'data: {"type":"tool_call","tool_call":[{"name":"graph"},{"name":"read","arguments":{"path":"a"}}]}',
            b"",
            b'data: {"v":1,"event":"content","content":"v1"}',
            b"",
            b'data: {"v":1,"event":"tool_call","tool":"list","call_id":"call-1"}',
            b"",
            b"data: [DONE]",
            b"",
            b"",
        ]
        raw = ending.join(frames)
        adapter, requests, stream = await http_adapter(parts=[bytes([b]) for b in raw])
        chunks = [chunk async for chunk in adapter.stream_chat([])]
        assert [chunk.content for chunk in chunks] == ["hello 🧪", "", "", "v1", ""]
        assert [chunk.tool_call.name for chunk in chunks if chunk.tool_call] == [
            "graph",
            "read",
            "list",
        ]
        assert chunks[1].tool_call.arguments == {}
        assert chunks[-1].tool_call.id == "call-1"
        assert stream.closed and len(requests) == 1

    @pytest.mark.parametrize(
        "raw",
        [
            b"",
            b'data: {"content":"partial"}\n\n',
            b"data: [DONE]\n",
            b'data: {"content":"partial","finish_reason":"stop"}\n\n',
            b"data: not-json\n\ndata: [DONE]\n\n",
            b"data: []\n\n",
            b'data: {"content":9}\n\n',
            b'data: {"type":"error","message":"private"}\n\ndata: [DONE]\n\n',
            b'data: {"v":1,"event":"error","message":"private"}\n\n',
            b'data: {"v":2,"event":"stream_end"}\n\n',
            b'data: {"type":"invented"}\n\n',
            b'data: {"event":null,"content":"partial"}\n\ndata: [DONE]\n\n',
            b'data: {"type":null,"content":"partial"}\n\ndata: [DONE]\n\n',
            b'data: {"v":1,"content":"partial"}\n\ndata: [DONE]\n\n',
            b'data: {"v":1,"type":"content","content":"partial"}\n\ndata: [DONE]\n\n',
            b'data: {"content":"\xff"}\n\ndata: [DONE]\n\n',
            b'data: {"type":"tool_call","tool_call":42}\n\n',
        ],
    )
    async def test_bad_stream_closes_without_replay(self, http_adapter, raw):
        adapter, requests, stream = await http_adapter(parts=[raw])
        with pytest.raises(ValueError) as error:
            _ = [chunk async for chunk in adapter.stream_chat([])]
        assert "private" not in str(error.value)
        assert stream.closed and len(requests) == 1

    @pytest.mark.parametrize("ending", [b"\n", b"\r\n", b"\r"])
    async def test_terminal_never_waits_for_another_read(self, http_adapter, ending):
        adapter, requests, stream = await http_adapter(
            parts=[b"data: [DONE]" + ending + ending],
            forbid_next_read=True,
        )
        assert [chunk async for chunk in adapter.stream_chat([])] == []
        assert stream.closed and len(requests) == 1

    @pytest.mark.parametrize("terminal", [b"[DONE]", b'{"v":1,"event":"stream_end"}'])
    async def test_terminal_stops_before_decoding_trailing_bytes(self, http_adapter, terminal):
        adapter, requests, stream = await http_adapter(parts=[b"data: " + terminal + b"\n\n\xff"])
        assert [chunk async for chunk in adapter.stream_chat([])] == []
        assert stream.closed and len(requests) == 1

    @pytest.mark.parametrize("prefix", [b"data: ", b":", b"ignored: "])
    async def test_bounds_unterminated_frames_including_ignored_lines(
        self, http_adapter, monkeypatch, prefix
    ):
        monkeypatch.setattr(
            "victor.integrations.protocol.adapters._MAX_SSE_FRAME_BYTES", 64, raising=False
        )
        adapter, requests, stream = await http_adapter(parts=[prefix, b"x" * 65])
        with pytest.raises(ValueError, match="limit"):
            _ = [chunk async for chunk in adapter.stream_chat([])]
        assert stream.closed and len(requests) == 1

    async def test_exact_frame_bound_and_reset(self, http_adapter, monkeypatch):
        monkeypatch.setattr(
            "victor.integrations.protocol.adapters._MAX_SSE_FRAME_BYTES", 64, raising=False
        )
        frame = b":" + b"x" * 61 + b"\n\n"  # 64 bytes including ignored comment and delimiter
        adapter, _, stream = await http_adapter(parts=[frame, b"data: [DONE]\n\n"])
        assert [chunk async for chunk in adapter.stream_chat([])] == []
        assert stream.closed

    @pytest.mark.parametrize(
        "status,content_type", [(403, "text/event-stream"), (200, "application/json")]
    )
    async def test_rejects_http_failure_and_wrong_media_type(
        self, http_adapter, status, content_type
    ):
        adapter, requests, stream = await http_adapter(
            parts=[b"data: [DONE]\n\n"], status=status, content_type=content_type
        )
        with pytest.raises((ValueError, httpx.HTTPStatusError)):
            _ = [chunk async for chunk in adapter.stream_chat([])]
        assert stream.closed and len(requests) == 1

    @pytest.mark.parametrize("cancel", [False, True])
    async def test_consumer_close_or_cancel_releases_response(self, http_adapter, cancel):
        adapter, requests, stream = await http_adapter(
            parts=[b'data: {"content":"partial"}\n\n'], wait=True
        )
        result = adapter.stream_chat([])
        assert (await anext(result)).content == "partial"
        if cancel:
            task = asyncio.create_task(anext(result))
            await stream.waiting.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await result.aclose()
        assert stream.closed and len(requests) == 1

    @pytest.mark.asyncio
    async def test_reset_conversation(self, adapter):
        """Test reset conversation via HTTP."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        adapter._client.post = AsyncMock(return_value=mock_response)

        await adapter.reset_conversation()

        adapter._client.post.assert_called_once_with("/conversation/reset")


class TestHTTPProtocolAdapterSearch:
    """Tests for HTTPProtocolAdapter search methods."""

    @pytest.fixture
    def adapter(self):
        return HTTPProtocolAdapter()

    @pytest.mark.asyncio
    async def test_semantic_search(self, adapter):
        """Test semantic search via HTTP."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"file": "test.py", "line": 1, "content": "code", "score": 0.9}]
        }
        mock_response.raise_for_status = MagicMock()
        adapter._client.post = AsyncMock(return_value=mock_response)

        results = await adapter.semantic_search("test", max_results=10)

        assert len(results) == 1
        adapter._client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_code_search(self, adapter):
        """Test code search via HTTP."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()
        adapter._client.post = AsyncMock(return_value=mock_response)

        results = await adapter.code_search("pattern", regex=True)

        assert results == []


class TestHTTPProtocolAdapterModel:
    """Tests for HTTPProtocolAdapter model/mode switching."""

    @pytest.fixture
    def adapter(self):
        return HTTPProtocolAdapter()

    @pytest.mark.asyncio
    async def test_switch_model(self, adapter):
        """Test model switching via HTTP."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        adapter._client.post = AsyncMock(return_value=mock_response)

        await adapter.switch_model("anthropic", "claude-3")

        adapter._client.post.assert_called_once_with(
            "/model/switch", json={"provider": "anthropic", "model": "claude-3"}
        )

    @pytest.mark.asyncio
    async def test_get_effective_config(self, adapter):
        """Test effective runtime config discovery via HTTP."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"profile": "default", "mode": "build"}
        adapter._client.get = AsyncMock(return_value=mock_response)

        result = await adapter.get_effective_config()

        assert result == {"profile": "default", "mode": "build"}
        adapter._client.get.assert_called_once_with("/config/effective")

    @pytest.mark.asyncio
    async def test_get_profiles(self, adapter):
        """Test profile discovery via HTTP."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "profiles": [{"name": "default", "provider": "ollama", "model": "qwen"}]
        }
        adapter._client.get = AsyncMock(return_value=mock_response)

        result = await adapter.get_profiles()

        assert result == [{"name": "default", "provider": "ollama", "model": "qwen"}]
        adapter._client.get.assert_called_once_with("/profiles")

    @pytest.mark.asyncio
    async def test_get_modes(self, adapter):
        """Test mode discovery via HTTP."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"modes": [{"name": "review", "description": "Review"}]}
        adapter._client.get = AsyncMock(return_value=mock_response)

        result = await adapter.get_modes()

        assert result == [{"name": "review", "description": "Review"}]
        adapter._client.get.assert_called_once_with("/modes")

    @pytest.mark.asyncio
    async def test_switch_profile(self, adapter):
        """Test profile switching via HTTP."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        adapter._client.post = AsyncMock(return_value=mock_response)

        await adapter.switch_profile("cloud")

        adapter._client.post.assert_called_once_with("/profile/switch", json={"profile": "cloud"})

    @pytest.mark.asyncio
    async def test_switch_mode(self, adapter):
        """Test mode switching via HTTP."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        adapter._client.post = AsyncMock(return_value=mock_response)

        await adapter.switch_mode(AgentMode.EXPLORE)

        adapter._client.post.assert_called_once_with("/mode/switch", json={"mode": "explore"})


class TestHTTPProtocolAdapterStatus:
    """Tests for HTTPProtocolAdapter status methods."""

    @pytest.fixture
    def adapter(self):
        return HTTPProtocolAdapter()

    @pytest.mark.asyncio
    async def test_get_status(self, adapter):
        """Test getting status via HTTP."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "provider": "openai",
            "model": "gpt-4",
            "mode": "build",  # Valid AgentMode value
            "connected": True,
            "tools_available": 10,
            "conversation_length": 5,
        }
        mock_response.raise_for_status = MagicMock()
        adapter._client.get = AsyncMock(return_value=mock_response)

        status = await adapter.get_status()

        assert status.provider == "openai"
        assert status.model == "gpt-4"
        assert status.connected is True


class TestHTTPProtocolAdapterUndoRedo:
    """Tests for HTTPProtocolAdapter undo/redo."""

    @pytest.fixture
    def adapter(self):
        return HTTPProtocolAdapter()

    @pytest.mark.asyncio
    async def test_undo(self, adapter):
        """Test undo via HTTP."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "message": "Undone",
            "files_modified": ["test.py"],
        }
        mock_response.raise_for_status = MagicMock()
        adapter._client.post = AsyncMock(return_value=mock_response)

        result = await adapter.undo()

        assert result.success is True
        adapter._client.post.assert_called_once_with("/undo")

    @pytest.mark.asyncio
    async def test_redo(self, adapter):
        """Test redo via HTTP."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "message": "Redone",
            "files_modified": [],
        }
        mock_response.raise_for_status = MagicMock()
        adapter._client.post = AsyncMock(return_value=mock_response)

        result = await adapter.redo()

        assert result.success is True

    @pytest.mark.asyncio
    async def test_get_history(self, adapter):
        """Test get history via HTTP."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"history": [{"id": 1}]}
        mock_response.raise_for_status = MagicMock()
        adapter._client.get = AsyncMock(return_value=mock_response)

        history = await adapter.get_history(limit=10)

        assert len(history) == 1


class TestHTTPProtocolAdapterPatch:
    """Tests for HTTPProtocolAdapter patch operations."""

    @pytest.fixture
    def adapter(self):
        return HTTPProtocolAdapter()

    @pytest.mark.asyncio
    async def test_apply_patch(self, adapter):
        """Test applying patch via HTTP."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True, "files_modified": []}
        mock_response.raise_for_status = MagicMock()
        adapter._client.post = AsyncMock(return_value=mock_response)

        result = await adapter.apply_patch("--- a\n+++ b", dry_run=True)

        assert result["success"] is True
        adapter._client.post.assert_called_once_with(
            "/patch/apply", json={"patch": "--- a\n+++ b", "dry_run": True}
        )


class TestHTTPProtocolAdapterLSP:
    """Tests for HTTPProtocolAdapter LSP methods."""

    @pytest.fixture
    def adapter(self):
        return HTTPProtocolAdapter()

    @pytest.mark.asyncio
    async def test_get_definition(self, adapter):
        """Test get definition via HTTP."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"locations": [{"file": "test.py", "line": 1}]}
        mock_response.raise_for_status = MagicMock()
        adapter._client.post = AsyncMock(return_value=mock_response)

        locations = await adapter.get_definition("test.py", 10, 5)

        assert len(locations) == 1
        assert locations[0]["file"] == "test.py"

    @pytest.mark.asyncio
    async def test_get_references(self, adapter):
        """Test get references via HTTP."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"locations": []}
        mock_response.raise_for_status = MagicMock()
        adapter._client.post = AsyncMock(return_value=mock_response)

        locations = await adapter.get_references("main.py", 5, 3)

        assert locations == []

    @pytest.mark.asyncio
    async def test_get_hover_success(self, adapter):
        """Test get hover success via HTTP."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"contents": "Function documentation"}
        mock_response.raise_for_status = MagicMock()
        adapter._client.post = AsyncMock(return_value=mock_response)

        result = await adapter.get_hover("test.py", 1, 1)

        assert result == "Function documentation"

    @pytest.mark.asyncio
    async def test_get_hover_failure(self, adapter):
        """Test get hover failure returns None."""
        adapter._client.post = AsyncMock(side_effect=Exception("Network error"))

        result = await adapter.get_hover("test.py", 1, 1)

        assert result is None


class TestHTTPProtocolAdapterClose:
    """Tests for HTTPProtocolAdapter close and health."""

    @pytest.fixture
    def adapter(self):
        return HTTPProtocolAdapter()

    @pytest.mark.asyncio
    async def test_close(self, adapter):
        """Test closing adapter."""
        adapter._client.aclose = AsyncMock()

        await adapter.close()

        adapter._client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_health_success(self, adapter):
        """Test health check success."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        adapter._client.get = AsyncMock(return_value=mock_response)

        result = await adapter.check_health()

        assert result is True

    @pytest.mark.asyncio
    async def test_check_health_failure(self, adapter):
        """Test health check failure."""
        adapter._client.get = AsyncMock(side_effect=Exception("Connection refused"))

        result = await adapter.check_health()

        assert result is False
