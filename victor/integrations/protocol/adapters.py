"""Protocol Adapters - Implementations of VictorProtocol.

This module provides adapters that implement the VictorProtocol interface
for different communication methods:

- DirectProtocolAdapter: Opt-in legacy access to a supplied orchestrator
- HTTPProtocolAdapter: Opt-in legacy Python client of the core HTTP API
"""

import json
import re
from contextlib import aclosing
from typing import Any, AsyncIterator

import httpx

from victor.integrations.protocol.interface import (
    _decode_tool_call,
    VictorProtocol,
    ChatMessage,
    ChatResponse,
    ClientStreamChunk,
    ToolCall,
    UndoRedoResult,
    AgentMode,
    AgentStatus,
)
from victor.integrations.search_types import CodeSearchResult

_MAX_SSE_FRAME_BYTES = 1024 * 1024


async def _iter_sse_payloads(response: httpx.Response) -> AsyncIterator[str]:
    """Bound wire frames before allocation; decode complete lines strictly.

    This is a POST consumer, not reconnecting EventSource. EOF never dispatches
    an unfinished frame, and only the caller recognizes a protocol terminator.
    """
    line = bytearray()
    data: list[str] = []
    frame_bytes = 0
    skip_lf = False
    first_line = True

    def consume(segment: bytes, delimiter_bytes: int = 0) -> None:
        nonlocal frame_bytes
        frame_bytes += len(segment) + delimiter_bytes
        if frame_bytes > _MAX_SSE_FRAME_BYTES:
            raise ValueError("SSE frame exceeds byte limit")
        line.extend(segment)

    def dispatch_line() -> str | None:
        nonlocal frame_bytes, first_line
        try:
            text = line.decode("utf-8-sig" if first_line else "utf-8", errors="strict")
        except UnicodeDecodeError:
            raise ValueError("Invalid UTF-8 in SSE frame") from None
        first_line = False
        line.clear()
        if not text:
            payload = "\n".join(data) if data else None
            data.clear()
            frame_bytes = 0
            return payload
        if not text.startswith(":"):
            name, separator, value = text.partition(":")
            if name == "data":
                data.append(value.removeprefix(" ") if separator else "")
        return None

    async for chunk in response.aiter_bytes():
        offset = 0
        for match in re.finditer(rb"[\r\n]", chunk):
            end = match.start()
            segment = chunk[offset:end]
            offset = end + 1
            if not segment and chunk[end] == 10 and skip_lf:
                # A CR already dispatched the line. Count the LF inside an
                # unfinished frame, but never delay a complete CR terminator.
                if frame_bytes:
                    consume(b"", 1)
                skip_lf = False
                continue
            consume(segment, 1)
            skip_lf = chunk[end] == 13
            payload = dispatch_line()
            if payload is not None:
                yield payload
        if offset < len(chunk):
            skip_lf = False
            consume(chunk[offset:])
    raise ValueError("Chat stream ended before an explicit terminator; outcome is unknown")


def _decode_stream_payload(payload: str) -> tuple[list[ClientStreamChunk], bool]:
    if payload == "[DONE]":
        return [], True
    try:
        data = json.loads(payload)
    except (ValueError, RecursionError):
        raise ValueError("Invalid JSON in chat stream") from None
    if not isinstance(data, dict):
        raise ValueError("Invalid chat stream event")
    if "v" in data and (type(data["v"]) is not int or data["v"] != 1):
        raise ValueError("Unsupported chat stream version")
    for discriminator in ("event", "type"):
        if discriminator in data and (
            not isinstance(data[discriminator], str) or not data[discriminator]
        ):
            raise ValueError("Invalid chat stream event type")
    if "v" in data and "event" not in data:
        raise ValueError("Versioned chat stream event requires an event type")
    kind = data.get("event", data.get("type"))
    if "event" in data and "type" in data and data["event"] != data["type"]:
        raise ValueError("Conflicting chat stream event types")
    if kind == "error" or data.get("error") is not None:
        raise ValueError("Server reported a chat stream error; outcome is unknown")
    if data.get("status") not in (None, "ok") or data.get("approval_request") is not None:
        raise ValueError("Unsupported streaming outcome; inspect recorded run state")
    if kind == "stream_end":
        if data.get("v") != 1:
            raise ValueError("Unsupported chat stream terminator")
        return [], True
    if kind in ("request", "thinking", "tool_result"):
        # This legacy content/tool-call iterator has no telemetry surface.
        return [], False
    if kind not in (None, "content", "tool_call"):
        raise ValueError("Unsupported chat stream event")
    content = data.get("content", "")
    finish = data.get("finish_reason")
    if not isinstance(content, str) or (finish is not None and not isinstance(finish, str)):
        raise ValueError("Invalid chat stream content or finish reason")
    if kind == "tool_call" and "v" in data:
        calls = [
            {
                "name": data.get("tool"),
                "arguments": data.get("arguments", {}),
                "id": data.get("call_id"),
            }
        ]
    else:
        raw_calls = data.get("tool_call")
        calls = (
            raw_calls
            if isinstance(raw_calls, list)
            else [raw_calls] if raw_calls is not None else []
        )
    if (
        not calls
        and kind not in ("content",)
        and "content" not in data
        and "finish_reason" not in data
    ):
        raise ValueError("Chat stream event has no supported payload")
    if calls:
        tools = [_decode_tool_call(call) for call in calls]
        return [
            ClientStreamChunk(
                content=content if i == 0 else "", tool_call=call, finish_reason=finish
            )
            for i, call in enumerate(tools)
        ], False
    return [ClientStreamChunk(content=content, finish_reason=finish)], False


class DirectProtocolAdapter(VictorProtocol):
    """Legacy adapter that calls a supplied orchestrator directly.

    This is not the CLI's current framework client or a durability boundary.

    Usage:
        adapter = await DirectProtocolAdapter.create()
        response = await adapter.chat([ChatMessage(role="user", content="Hello")])
    """

    def __init__(self, orchestrator: Any) -> None:
        """Initialize with an orchestrator instance.

        Args:
            orchestrator: Agent runtime instance (Agent-created)
        """
        self._orchestrator = orchestrator

    @classmethod
    async def create(
        cls,
        profile: str = "default",
        thinking: bool = False,
    ) -> "DirectProtocolAdapter":
        """Create a DirectProtocolAdapter with a new orchestrator.

        Args:
            profile: Profile name to load
            thinking: Enable thinking mode

        Returns:
            Configured adapter
        """
        from victor.config.settings import load_settings
        from victor.framework.agent_factory import AgentFactory

        settings = load_settings()
        factory = AgentFactory(settings, profile=profile, thinking=thinking)
        orchestrator = await factory.create()
        return cls(orchestrator)

    async def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        """Send messages and get a response."""
        # Convert to orchestrator message format
        message = messages[-1].content if messages else ""
        response = await self._orchestrator.chat(message)

        # Extract tool calls if present
        tool_calls = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=getattr(tc, "id", ""),
                        name=getattr(tc, "name", ""),
                        arguments=getattr(tc, "arguments", {}),
                    )
                )

        return ChatResponse(
            content=response.content or "",
            tool_calls=tool_calls,
            finish_reason="stop",
            usage=response.usage if hasattr(response, "usage") else {},
        )

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[ClientStreamChunk]:
        """Stream a chat response."""
        message = messages[-1].content if messages else ""

        async for chunk in self._orchestrator.stream_chat(message):
            yield ClientStreamChunk(
                content=chunk.content or "",
                tool_call=None,  # Tool calls handled separately
                finish_reason=chunk.finish_reason,
            )

    async def reset_conversation(self) -> None:
        """Clear conversation history."""
        self._orchestrator.reset_conversation()

    async def semantic_search(self, query: str, max_results: int = 10) -> list[CodeSearchResult]:
        """Search code by semantic meaning."""
        from victor.tools.semantic_search import SemanticCodeSearchTool

        tool = SemanticCodeSearchTool()
        result = await tool.execute(query=query, max_results=max_results)

        if not result.success:
            return []

        # Parse results
        search_results = []
        for match in result.data.get("matches", []):
            search_results.append(
                CodeSearchResult(
                    file=match.get("file", ""),
                    line=match.get("line", 0),
                    content=match.get("content", ""),
                    score=match.get("score", 0.0),
                    context=match.get("context", ""),
                )
            )
        return search_results

    async def code_search(
        self,
        query: str,
        regex: bool = False,
        case_sensitive: bool = False,
        file_pattern: str | None = None,
    ) -> list[CodeSearchResult]:
        """Search code by pattern."""
        from victor.tools.code_search import CodeSearchTool

        tool = CodeSearchTool()
        result = await tool.execute(
            query=query,
            regex=regex,
            case_sensitive=case_sensitive,
            file_pattern=file_pattern or "*",
        )

        if not result.success:
            return []

        search_results = []
        for match in result.data.get("matches", []):
            search_results.append(
                CodeSearchResult(
                    file=match.get("file", ""),
                    line=match.get("line", 0),
                    content=match.get("content", ""),
                    score=1.0,  # Exact matches
                )
            )
        return search_results

    async def switch_model(self, provider: str, model: str) -> None:
        """Switch to a different model."""
        from victor.agent.model_switcher import get_model_switcher

        switcher = get_model_switcher()
        switcher.switch(provider, model)

        # Store for potential orchestrator reinitialization
        self._pending_provider = provider
        self._pending_model = model

    async def get_effective_config(self) -> dict[str, Any]:
        """Return effective runtime configuration for direct clients."""
        from victor.config.settings import load_settings
        from victor.framework.runtime_discovery import effective_runtime_config

        return effective_runtime_config(load_settings())

    async def get_profiles(self) -> list[dict[str, Any]]:
        """List configured runtime profiles."""
        from victor.config.settings import load_settings
        from victor.framework.runtime_discovery import list_runtime_profiles

        return [profile.to_dict() for profile in list_runtime_profiles(load_settings())]

    async def get_modes(self) -> list[dict[str, str]]:
        """List supported runtime modes."""
        from victor.framework.runtime_discovery import list_runtime_modes

        return list_runtime_modes()

    async def switch_profile(self, profile: str) -> None:
        """Record a pending runtime profile switch for direct clients."""
        self._pending_profile = profile

    async def switch_mode(self, mode: AgentMode) -> None:
        """Switch agent mode."""
        if hasattr(self._orchestrator, "set_mode"):
            self._orchestrator.set_mode(mode.value)

    async def get_status(self) -> AgentStatus:
        """Get current agent status.

        Uses orchestrator's ModeAwareMixin properties for mode access.
        """
        # Get mode from orchestrator's ModeAwareMixin (consistent access)
        current_mode = self._orchestrator.current_mode_name.lower()

        return AgentStatus(
            provider=self._orchestrator.provider.name,
            model=getattr(self._orchestrator.provider, "model", "unknown"),
            mode=AgentMode(current_mode),
            connected=True,
            tools_available=(
                len(self._orchestrator.tools) if hasattr(self._orchestrator, "tools") else 0
            ),
            conversation_length=(
                len(self._orchestrator.messages) if hasattr(self._orchestrator, "messages") else 0
            ),
        )

    async def undo(self) -> UndoRedoResult:
        """Undo the last change."""
        if hasattr(self._orchestrator, "change_tracker"):
            result = self._orchestrator.change_tracker.undo()
            return UndoRedoResult(
                success=result.get("success", False),
                message=result.get("message", ""),
                files_modified=result.get("files", []),
            )
        return UndoRedoResult(
            success=False,
            message="Undo not available",
        )

    async def redo(self) -> UndoRedoResult:
        """Redo the last undone change."""
        if hasattr(self._orchestrator, "change_tracker"):
            result = self._orchestrator.change_tracker.redo()
            return UndoRedoResult(
                success=result.get("success", False),
                message=result.get("message", ""),
                files_modified=result.get("files", []),
            )
        return UndoRedoResult(
            success=False,
            message="Redo not available",
        )

    async def get_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get change history."""
        if hasattr(self._orchestrator, "change_tracker"):
            return self._orchestrator.change_tracker.get_history(limit)
        return []

    async def apply_patch(self, patch: str, dry_run: bool = False) -> dict[str, Any]:
        """Apply a unified diff patch."""
        from victor.tools.patch_tool import PatchTool

        tool = PatchTool()
        result = await tool.execute(patch=patch, dry_run=dry_run)

        return {
            "success": result.success,
            "files_modified": result.data.get("files_modified", []),
            "preview": result.data.get("preview") if dry_run else None,
        }

    async def close(self) -> None:
        """Close the connection and clean up resources."""
        if hasattr(self._orchestrator, "provider"):
            await self._orchestrator.provider.close()


class HTTPProtocolAdapter(VictorProtocol):
    """Legacy Python client of the core Victor HTTP API.

    The VS Code extension has its own TypeScript client. Streams require a
    framed [DONE] or v1 stream_end, reject malformed/oversized frames, and never
    replay POST on failure. A terminator is transport completion, not a verified
    business outcome. Plain response serialization retains its legacy shape.

    Usage:
        adapter = HTTPProtocolAdapter("http://localhost:8765")
        response = await adapter.chat([ChatMessage(role="user", content="Hello")])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        timeout: float = 60.0,
    ) -> None:
        """Initialize HTTP adapter.

        Args:
            base_url: Base URL of Victor server
            timeout: Request timeout in seconds
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
        )

    async def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        """Send messages and get a response."""
        response = await self._client.post(
            "/chat",
            json={"messages": [m.to_dict() for m in messages]},
        )
        response.raise_for_status()
        result = ChatResponse.from_dict(response.json())
        if response.status_code == 202 and result.status != "awaiting_approval":
            raise ValueError("HTTP 202 lacks valid paused outcome; outcome is unknown")
        return result

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[ClientStreamChunk]:
        """Stream a chat response."""
        async with self._client.stream(
            "POST",
            "/chat/stream",
            json={"messages": [m.to_dict() for m in messages]},
        ) as response:
            response.raise_for_status()
            media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if media_type != "text/event-stream":
                raise ValueError("Expected text/event-stream response")
            async with aclosing(_iter_sse_payloads(response)) as frames:
                async for payload in frames:
                    chunks, terminal = _decode_stream_payload(payload)
                    if terminal:
                        return
                    for chunk in chunks:
                        yield chunk

    async def reset_conversation(self) -> None:
        """Clear conversation history."""
        response = await self._client.post("/conversation/reset")
        response.raise_for_status()

    async def semantic_search(self, query: str, max_results: int = 10) -> list[CodeSearchResult]:
        """Search code by semantic meaning."""
        response = await self._client.post(
            "/search/semantic",
            json={"query": query, "max_results": max_results},
        )
        response.raise_for_status()
        data = response.json()
        return [CodeSearchResult.from_dict(r) for r in data.get("results", [])]

    async def code_search(
        self,
        query: str,
        regex: bool = False,
        case_sensitive: bool = False,
        file_pattern: str | None = None,
    ) -> list[CodeSearchResult]:
        """Search code by pattern."""
        response = await self._client.post(
            "/search/code",
            json={
                "query": query,
                "regex": regex,
                "case_sensitive": case_sensitive,
                "file_pattern": file_pattern,
            },
        )
        response.raise_for_status()
        data = response.json()
        return [CodeSearchResult.from_dict(r) for r in data.get("results", [])]

    async def switch_model(self, provider: str, model: str) -> None:
        """Switch to a different model."""
        response = await self._client.post(
            "/model/switch",
            json={"provider": provider, "model": model},
        )
        response.raise_for_status()

    async def get_effective_config(self) -> dict[str, Any]:
        """Return effective runtime configuration."""
        response = await self._client.get("/config/effective")
        response.raise_for_status()
        return dict(response.json())

    async def get_profiles(self) -> list[dict[str, Any]]:
        """List configured runtime profiles."""
        response = await self._client.get("/profiles")
        response.raise_for_status()
        data = response.json()
        return list(data.get("profiles", []))

    async def get_modes(self) -> list[dict[str, str]]:
        """List supported runtime modes."""
        response = await self._client.get("/modes")
        response.raise_for_status()
        data = response.json()
        return list(data.get("modes", []))

    async def switch_profile(self, profile: str) -> None:
        """Switch active runtime profile."""
        response = await self._client.post(
            "/profile/switch",
            json={"profile": profile},
        )
        response.raise_for_status()

    async def switch_mode(self, mode: AgentMode) -> None:
        """Switch agent mode."""
        response = await self._client.post(
            "/mode/switch",
            json={"mode": mode.value},
        )
        response.raise_for_status()

    async def get_status(self) -> AgentStatus:
        """Get current agent status."""
        response = await self._client.get("/status")
        response.raise_for_status()
        data = response.json()
        return AgentStatus(
            provider=data["provider"],
            model=data["model"],
            mode=AgentMode(data["mode"]),
            connected=data["connected"],
            tools_available=data.get("tools_available", 0),
            conversation_length=data.get("conversation_length", 0),
        )

    async def undo(self) -> UndoRedoResult:
        """Undo the last change."""
        response = await self._client.post("/undo")
        response.raise_for_status()
        data = response.json()
        return UndoRedoResult(
            success=data["success"],
            message=data["message"],
            files_modified=data.get("files_modified", []),
        )

    async def redo(self) -> UndoRedoResult:
        """Redo the last undone change."""
        response = await self._client.post("/redo")
        response.raise_for_status()
        data = response.json()
        return UndoRedoResult(
            success=data["success"],
            message=data["message"],
            files_modified=data.get("files_modified", []),
        )

    async def get_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get change history."""
        response = await self._client.get("/history", params={"limit": limit})
        response.raise_for_status()
        return response.json().get("history", [])

    async def apply_patch(self, patch: str, dry_run: bool = False) -> dict[str, Any]:
        """Apply a unified diff patch."""
        response = await self._client.post(
            "/patch/apply",
            json={"patch": patch, "dry_run": dry_run},
        )
        response.raise_for_status()
        return response.json()

    async def get_definition(self, file: str, line: int, character: int) -> list[dict[str, Any]]:
        """Get definition locations for symbol at position."""
        response = await self._client.post(
            "/lsp/definition",
            json={"file": file, "line": line, "character": character},
        )
        response.raise_for_status()
        return response.json().get("locations", [])

    async def get_references(self, file: str, line: int, character: int) -> list[dict[str, Any]]:
        """Get reference locations for symbol at position."""
        response = await self._client.post(
            "/lsp/references",
            json={"file": file, "line": line, "character": character},
        )
        response.raise_for_status()
        return response.json().get("locations", [])

    async def get_hover(self, file: str, line: int, character: int) -> str | None:
        """Get hover information for symbol at position."""
        try:
            response = await self._client.post(
                "/lsp/hover",
                json={"file": file, "line": line, "character": character},
            )
            response.raise_for_status()
            return response.json().get("contents")
        except Exception:
            return None

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def check_health(self) -> bool:
        """Check if the server is healthy."""
        try:
            response = await self._client.get("/health", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False
