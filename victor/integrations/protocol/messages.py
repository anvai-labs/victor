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

"""Unified Message Format.

Provides a consistent message structure across all Victor clients:
- CLI (Python)
- VS Code Extension (TypeScript)
- MCP Clients (JSON-RPC)
- API Clients (REST/WebSocket)

This ensures feature parity and consistent behavior across all interfaces.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union


class MessageRole(str, Enum):
    """Role of the message sender."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ContentType(str, Enum):
    """Type of content block."""

    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    FILE = "file"
    THINKING = "thinking"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    ERROR = "error"


class ToolCallStatus(str, Enum):
    """Status of a tool call."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class TextContent:
    """Plain text content."""

    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class CodeContent:
    """Code block content with language and optional file path."""

    type: Literal["code"] = "code"
    code: str = ""
    language: str = "text"
    file_path: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None


@dataclass
class ImageContent:
    """Image content (base64 or URL)."""

    type: Literal["image"] = "image"
    source_type: Literal["base64", "url"] = "base64"
    data: str = ""
    media_type: str = "image/png"
    alt_text: Optional[str] = None


@dataclass
class FileContent:
    """File reference content."""

    type: Literal["file"] = "file"
    path: str = ""
    name: Optional[str] = None
    size: Optional[int] = None
    mime_type: Optional[str] = None


@dataclass
class ThinkingContent:
    """Thinking/reasoning content (for models that support it)."""

    type: Literal["thinking"] = "thinking"
    thinking: str = ""
    summary: Optional[str] = None


@dataclass
class ToolUseContent:
    """Tool invocation request."""

    type: Literal["tool_use"] = "tool_use"
    id: str = ""
    name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultContent:
    """Result of a tool invocation."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str = ""
    output: str = ""
    is_error: bool = False
    duration_ms: Optional[int] = None


@dataclass
class ErrorContent:
    """Error content."""

    type: Literal["error"] = "error"
    message: str = ""
    code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# Union type for all content types
MessageContent = Union[
    TextContent,
    CodeContent,
    ImageContent,
    FileContent,
    ThinkingContent,
    ToolUseContent,
    ToolResultContent,
    ErrorContent,
]


# Import canonical ToolCall for basic representation
from victor.agent.tool_calling.base import ToolCall


@dataclass
class ToolCallExecution:
    """Tool call execution with full lifecycle tracking.

    Used for displaying tool execution in the UI with status, timing, and metadata.
    Different from ToolCall which is just the request representation.
    """

    id: str = ""
    name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    status: ToolCallStatus = ToolCallStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    category: Optional[str] = None
    cost_tier: Optional[str] = None
    is_dangerous: bool = False

    @property
    def duration_ms(self) -> Optional[int]:
        """Get duration in milliseconds."""
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time) * 1000)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "category": self.category,
            "cost_tier": self.cost_tier,
            "is_dangerous": self.is_dangerous,
        }


@dataclass
class UnifiedMessage:
    """Unified message format for all Victor clients.

    This is the canonical representation that all clients should use.
    Clients can extend with additional fields as needed.
    """

    id: str = ""
    role: MessageRole = MessageRole.USER
    content: List[MessageContent] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    tool_calls: List[ToolCall] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = f"msg-{uuid.uuid4().hex[:12]}"

    @property
    def text_content(self) -> str:
        """Get concatenated text content."""
        texts = []
        for block in self.content:
            if isinstance(block, TextContent):
                texts.append(block.text)
            elif isinstance(block, CodeContent):
                texts.append(f"```{block.language}\n{block.code}\n```")
            elif isinstance(block, ThinkingContent):
                texts.append(f"<thinking>{block.thinking}</thinking>")
        return "\n".join(texts)

    @classmethod
    def user(cls, text: str, **kwargs) -> "UnifiedMessage":
        """Create a user message."""
        return cls(
            role=MessageRole.USER,
            content=[TextContent(text=text)],
            **kwargs,
        )

    @classmethod
    def assistant(cls, text: str, **kwargs) -> "UnifiedMessage":
        """Create an assistant message."""
        return cls(
            role=MessageRole.ASSISTANT,
            content=[TextContent(text=text)],
            **kwargs,
        )

    @classmethod
    def system(cls, text: str, **kwargs) -> "UnifiedMessage":
        """Create a system message."""
        return cls(
            role=MessageRole.SYSTEM,
            content=[TextContent(text=text)],
            **kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedMessage":
        """Create from dictionary."""
        content = []
        for block in data.get("content", []):
            block_type = block.get("type", "text")
            if block_type == "text":
                content.append(TextContent(**block))
            elif block_type == "code":
                content.append(CodeContent(**block))
            elif block_type == "image":
                content.append(ImageContent(**block))
            elif block_type == "file":
                content.append(FileContent(**block))
            elif block_type == "thinking":
                content.append(ThinkingContent(**block))
            elif block_type == "tool_use":
                content.append(ToolUseContent(**block))
            elif block_type == "tool_result":
                content.append(ToolResultContent(**block))
            elif block_type == "error":
                content.append(ErrorContent(**block))

        tool_calls = [
            ToolCall(
                id=tc.get("id", ""),
                name=tc.get("name", ""),
                arguments=tc.get("arguments", {}),
                status=ToolCallStatus(tc.get("status", "pending")),
                result=tc.get("result"),
                error=tc.get("error"),
                start_time=tc.get("start_time"),
                end_time=tc.get("end_time"),
                category=tc.get("category"),
                cost_tier=tc.get("cost_tier"),
                is_dangerous=tc.get("is_dangerous", False),
            )
            for tc in data.get("tool_calls", [])
        ]

        return cls(
            id=data.get("id", ""),
            role=MessageRole(data.get("role", "user")),
            content=content,
            timestamp=data.get("timestamp", time.time()),
            tool_calls=tool_calls,
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        content_dicts = []
        for block in self.content:
            if hasattr(block, "__dict__"):
                content_dicts.append({k: v for k, v in block.__dict__.items() if v is not None})
            else:
                content_dicts.append({"type": "text", "text": str(block)})

        return {
            "id": self.id,
            "role": self.role.value,
            "content": content_dicts,
            "timestamp": self.timestamp,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "metadata": self.metadata,
        }

    def to_api_format(self) -> Dict[str, Any]:
        """Convert to API message format (for LLM providers).

        Returns a simplified format compatible with most LLM APIs.
        """
        # Combine text content
        text_parts = []
        for block in self.content:
            if isinstance(block, TextContent):
                text_parts.append(block.text)
            elif isinstance(block, CodeContent):
                text_parts.append(f"```{block.language}\n{block.code}\n```")

        return {
            "role": self.role.value,
            "content": "\n".join(text_parts),
        }


@dataclass
class StreamChunk:
    """Streaming chunk for real-time updates.

    Used for streaming responses from the server to clients.
    """

    type: str  # "content", "tool_call", "thinking", "done", "error"
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def content(cls, text: str, **kwargs) -> "StreamChunk":
        """Create a content chunk."""
        return cls(type="content", data={"text": text, **kwargs})

    @classmethod
    def tool_call(cls, tool_call: ToolCall, **kwargs) -> "StreamChunk":
        """Create a tool call chunk."""
        return cls(type="tool_call", data={**tool_call.to_dict(), **kwargs})

    @classmethod
    def thinking(cls, text: str, **kwargs) -> "StreamChunk":
        """Create a thinking chunk."""
        return cls(type="thinking", data={"text": text, **kwargs})

    @classmethod
    def done(cls, **kwargs) -> "StreamChunk":
        """Create a done chunk."""
        return cls(type="done", data=kwargs)

    @classmethod
    def error(cls, message: str, code: Optional[str] = None, **kwargs) -> "StreamChunk":
        """Create an error chunk."""
        return cls(type="error", data={"message": message, "code": code, **kwargs})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp,
        }


# Type aliases for convenience
Messages = List[UnifiedMessage]
ContentBlocks = List[MessageContent]
