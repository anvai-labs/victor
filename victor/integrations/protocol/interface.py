"""Legacy Python protocol types for opt-in direct and HTTP adapters.

Current CLI/IDE/framework consumers do not all implement this interface. Keep
public imports compatible while the shared agent-service contract is reviewed.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Any
from datetime import datetime

# Import canonical AgentMode from mode_controller
from victor.agent.mode_controller import AgentMode
from victor.integrations.search_types import CodeSearchResult


@dataclass
class ChatMessage:
    """A message in a conversation."""

    role: str  # "user", "assistant", or "system"
    content: str
    tool_calls: list["ToolCall"] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatMessage":
        return cls(
            role=data["role"],
            content=data["content"],
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])],
            timestamp=(
                datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now()
            ),
        )


# Import canonical ToolCall for basic tool call representation
from victor.agent.tool_calling.base import ToolCall


@dataclass
class ToolInvocation:
    """A tool invocation with its result.

    Different from ToolCall - this pairs a tool call with its execution result.
    Use ToolCall for the request, ToolInvocation for request+response pair.
    """

    id: str
    name: str
    arguments: dict[str, Any]
    result: "ToolResult | None" = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result.to_dict() if self.result else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolInvocation":
        return cls(
            id=data["id"],
            name=data["name"],
            arguments=data["arguments"],
            result=ToolResult.from_dict(data["result"]) if data.get("result") else None,
        )


@dataclass
class ToolResult:
    """Result from a tool execution."""

    success: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResult":
        return cls(
            success=data["success"],
            output=data["output"],
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ChatResponse:
    """Response from a chat request."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    status: str | None = None
    run_id: str | None = None
    approval_request: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        paused = self.status == "awaiting_approval"
        if (
            self.status not in (None, "ok", "awaiting_approval")
            or (self.run_id is not None and not isinstance(self.run_id, str))
            or (self.approval_request is not None and not isinstance(self.approval_request, dict))
            or (not paused and self.approval_request is not None)
            or (
                paused
                and (not self.run_id or not self.run_id.strip() or self.approval_request is None)
            )
        ):
            raise ValueError("Invalid chat outcome metadata; outcome is unknown")
        if paused:
            self.finish_reason = "awaiting_approval"

    def to_dict(self) -> dict[str, Any]:
        result = {
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "finish_reason": self.finish_reason,
            "usage": self.usage,
        }
        # Keep the legacy four-key shape byte-identical when metadata is absent.
        for key in ("status", "run_id", "approval_request"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatResponse":
        if not isinstance(data, dict) or not isinstance(data.get("content"), str):
            raise ValueError("Invalid chat response; outcome is unknown")
        if "status" in data and data["status"] is None:
            raise ValueError("Invalid chat outcome metadata; outcome is unknown")
        calls = data.get("tool_calls")
        if calls is None:
            calls = []
        if not isinstance(calls, list):
            raise ValueError("Invalid chat tool calls; outcome is unknown")
        finish = data.get("finish_reason", "stop")
        usage = data.get("usage", {})
        if not isinstance(finish, str) or not isinstance(usage, dict):
            raise ValueError("Invalid chat response; outcome is unknown")
        return cls(
            content=data["content"],
            tool_calls=[_decode_tool_call(tc) for tc in calls],
            finish_reason=finish,
            usage=usage,
            status=data.get("status"),
            run_id=data.get("run_id"),
            approval_request=data.get("approval_request"),
        )


def _decode_tool_call(data: Any) -> ToolCall:
    """Check the wire shape, retaining canonical optional argument/id defaults."""
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("name", ""), str)
        or not isinstance(data.get("arguments", {}), dict)
        or (data.get("id") is not None and not isinstance(data["id"], str))
    ):
        raise ValueError("Invalid tool call in chat response")
    return ToolCall.from_dict(data)


@dataclass
class ClientStreamChunk:
    """A chunk from a streaming response for client protocols.

    Used by opt-in legacy Python protocol adapters for streaming responses.

    Renamed from StreamChunk to be semantically distinct from other streaming types:
    - StreamChunk (victor.providers.base): Provider-level raw streaming
    - OrchestratorStreamChunk: Orchestrator protocol with typed ChunkType
    - TypedStreamChunk: Safe typed accessor with nested StreamDelta
    - ClientStreamChunk: Opt-in legacy Python protocol adapters
    """

    content: str
    tool_call: ToolCall | None = None
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "finish_reason": self.finish_reason,
        }


@dataclass
class UndoRedoResult:
    """Result from an undo/redo operation."""

    success: bool
    message: str
    files_modified: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "files_modified": self.files_modified,
        }


@dataclass
class AgentStatus:
    """Current agent status."""

    provider: str
    model: str
    mode: AgentMode
    connected: bool
    tools_available: int
    conversation_length: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "mode": self.mode.value,
            "connected": self.connected,
            "tools_available": self.tools_available,
            "conversation_length": self.conversation_length,
        }


class VictorProtocol(ABC):
    """Abstract protocol interface for Victor clients.

    Retained for callers of the legacy Python adapters. This interface alone
    does not establish parity with current CLI, IDE or MCP application services.
    """

    # =========================================================================
    # Chat Operations
    # =========================================================================

    @abstractmethod
    async def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        """Send messages and get a response.

        Args:
            messages: Conversation history

        Returns:
            Chat response with content and tool calls
        """
        ...

    @abstractmethod
    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[ClientStreamChunk]:
        """Stream a chat response.

        Args:
            messages: Conversation history

        Yields:
            Stream chunks with content and tool calls
        """
        ...

    @abstractmethod
    async def reset_conversation(self) -> None:
        """Clear conversation history."""
        ...

    # =========================================================================
    # Search Operations
    # =========================================================================

    @abstractmethod
    async def semantic_search(self, query: str, max_results: int = 10) -> list[CodeSearchResult]:
        """Search code by semantic meaning.

        Args:
            query: Natural language query
            max_results: Maximum results to return

        Returns:
            List of search results ranked by relevance
        """
        ...

    @abstractmethod
    async def code_search(
        self,
        query: str,
        regex: bool = False,
        case_sensitive: bool = False,
        file_pattern: str | None = None,
    ) -> list[CodeSearchResult]:
        """Search code by pattern.

        Args:
            query: Search pattern (literal or regex)
            regex: Treat query as regex
            case_sensitive: Match case
            file_pattern: Glob pattern to filter files

        Returns:
            List of search results
        """
        ...

    # =========================================================================
    # Runtime Configuration
    # =========================================================================

    @abstractmethod
    async def get_effective_config(self) -> dict[str, Any]:
        """Return the effective runtime configuration.

        This includes the active profile, provider/model, mode, and discovered
        profile/mode options for workflow-driven clients.
        """
        ...

    @abstractmethod
    async def get_profiles(self) -> list[dict[str, Any]]:
        """List available runtime profiles."""
        ...

    @abstractmethod
    async def get_modes(self) -> list[dict[str, str]]:
        """List supported runtime modes."""
        ...

    @abstractmethod
    async def switch_profile(self, profile: str) -> None:
        """Switch to a different runtime profile.

        Args:
            profile: Profile name from profiles.yaml
        """
        ...

    @abstractmethod
    async def switch_model(self, provider: str, model: str) -> None:
        """Switch to a different model.

        Args:
            provider: Provider name (anthropic, openai, ollama, etc.)
            model: Model identifier
        """
        ...

    @abstractmethod
    async def switch_mode(self, mode: AgentMode) -> None:
        """Switch agent mode.

        Args:
            mode: New agent mode
        """
        ...

    @abstractmethod
    async def get_status(self) -> AgentStatus:
        """Get current agent status.

        Returns:
            Current agent status including provider, model, mode
        """
        ...

    # =========================================================================
    # Undo/Redo Operations
    # =========================================================================

    @abstractmethod
    async def undo(self) -> UndoRedoResult:
        """Undo the last change.

        Returns:
            Result with success status and affected files
        """
        ...

    @abstractmethod
    async def redo(self) -> UndoRedoResult:
        """Redo the last undone change.

        Returns:
            Result with success status and affected files
        """
        ...

    @abstractmethod
    async def get_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get change history.

        Args:
            limit: Maximum entries to return

        Returns:
            List of history entries
        """
        ...

    # =========================================================================
    # Patch Operations
    # =========================================================================

    @abstractmethod
    async def apply_patch(self, patch: str, dry_run: bool = False) -> dict[str, Any]:
        """Apply a unified diff patch.

        Args:
            patch: Unified diff patch content
            dry_run: If True, preview without applying

        Returns:
            Result with success status and affected files
        """
        ...

    # =========================================================================
    # LSP Operations (optional, for IDE integrations)
    # =========================================================================

    async def get_definition(self, file: str, line: int, character: int) -> list[dict[str, Any]]:
        """Get definition locations for symbol at position.

        Args:
            file: File path
            line: Line number (0-indexed)
            character: Character position

        Returns:
            List of definition locations
        """
        return []

    async def get_references(self, file: str, line: int, character: int) -> list[dict[str, Any]]:
        """Get reference locations for symbol at position.

        Args:
            file: File path
            line: Line number (0-indexed)
            character: Character position

        Returns:
            List of reference locations
        """
        return []

    async def get_hover(self, file: str, line: int, character: int) -> str | None:
        """Get hover information for symbol at position.

        Args:
            file: File path
            line: Line number (0-indexed)
            character: Character position

        Returns:
            Hover content or None
        """
        return None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    @abstractmethod
    async def close(self) -> None:
        """Close the connection and clean up resources."""
        ...

    async def check_health(self) -> bool:
        """Check if the connection is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            await self.get_status()
            return True
        except Exception:
            return False
