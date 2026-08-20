# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Canonical conversation types.

Single source of truth for message representation. ConversationMessage
is a superset of provider Message — it adds priority, token tracking,
timestamps, and metadata needed for context management and persistence.

All conversation modules import types from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

if TYPE_CHECKING:
    from victor.providers.base import Message


class MessageRole(Enum):
    """Message roles in conversation.

    Values match the OpenAI Chat Completions API spec.
    Non-OpenAI providers should adapt at the provider layer.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"  # OpenAI spec: role=tool with tool_call_id
    TOOL_CALL = "tool_call"  # Internal: assistant requesting tool execution


class MessagePriority(Enum):
    """Priority levels for context management.

    Higher values indicate higher priority (kept longer during compaction).
    """

    CRITICAL = 100  # System prompts, current task
    HIGH = 75  # Recent tool results, code context
    MEDIUM = 50  # Previous exchanges
    LOW = 25  # Old context, summaries
    EPHEMERAL = 0  # Can be dropped immediately


MESSAGE_SOURCE_METADATA_KEY = "source"


class MessageSource(Enum):
    """Origin of a ConversationMessage within the conversation system.

    Stored in ConversationMessage.metadata["source"] — no DB migration needed.
    Used by scoring and compaction to distinguish permanent ground-truth messages
    from transient synthetic scaffolding.

    Wire format is unaffected: providers still receive role=user/assistant/etc.
    This enum lives only in the metadata dict and the source property.
    """

    USER_TYPED = "u"
    AGENT_NUDGE = "an"
    AGENT_CONTINUATION = "ac"
    AGENT_GUIDANCE = "ag"
    AGENT_GROUNDING = "agr"
    AGENT_RESPONSE = "ar"
    COMPACTION_SUMMARY = "cs"
    LEDGER_RENDER = "lr"
    SYSTEM_INJECTED = "si"
    UNKNOWN = "?"


@dataclass
class ConversationMessage:
    """Canonical message type for the conversation system.

    Superset of provider Message — adds priority, token tracking,
    timestamps, and metadata for context management and persistence.

    Usage:
        # Create from scratch
        msg = ConversationMessage(role="user", content="Fix the bug")

        # Create from provider Message
        msg = ConversationMessage.from_provider_message(provider_msg)

        # Convert back to provider Message for LLM calls
        provider_msg = msg.to_provider_message()
    """

    role: str
    content: str
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    token_count: int = 0
    priority: MessagePriority = MessagePriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    name: Optional[str] = None

    @property
    def role_enum(self) -> MessageRole:
        """Get role as MessageRole enum."""
        try:
            return MessageRole(self.role)
        except ValueError:
            return MessageRole.ASSISTANT

    @property
    def source(self) -> MessageSource:
        """Origin of this message. UNKNOWN for legacy messages without source metadata."""
        raw = self.metadata.get(MESSAGE_SOURCE_METADATA_KEY)
        if raw is None:
            return MessageSource.UNKNOWN
        try:
            return MessageSource(raw)
        except ValueError:
            return MessageSource.UNKNOWN

    @source.setter
    def source(self, value: MessageSource) -> None:
        self.metadata[MESSAGE_SOURCE_METADATA_KEY] = value.value

    def to_provider_message(self) -> "Message":
        """Convert to provider Message for LLM API calls."""
        from victor.providers.base import Message

        return Message(
            role=self._normalize_role(),
            content=self.content,
            name=self.name,
            tool_calls=self.tool_calls,
            tool_call_id=self.tool_call_id,
        )

    @classmethod
    def from_provider_message(
        cls,
        msg: "Message",
        *,
        priority: MessagePriority = MessagePriority.MEDIUM,
        token_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationMessage:
        """Create from provider Message, carrying its metadata across.

        The provider ``Message``'s own metadata is merged in, with any explicitly
        passed ``metadata`` taking precedence. Without this the bridge dropped
        ``MessageSource`` on every message, so ``_SOURCE_ROLE_OVERRIDES`` in
        ``conversation/scoring.py`` never fired: everything scored as UNKNOWN and
        fell through to ``_ROLE_SCORES``. Agent guidance is injected with
        ``role="user"``, so it scored 0.8 — identical to something the user typed,
        rather than the 0.2 the override intends — and could evict real user
        intent during compaction.
        """
        merged: Dict[str, Any] = dict(getattr(msg, "metadata", None) or {})
        if metadata:
            merged.update(metadata)
        return cls(
            role=msg.role,
            content=msg.content,
            name=msg.name,
            tool_calls=msg.tool_calls,
            tool_call_id=msg.tool_call_id,
            priority=priority,
            token_count=token_count,
            metadata=merged,
        )

    def to_provider_format(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible message dict for provider API calls."""
        role_str = self.role.value if isinstance(self.role, MessageRole) else self.role
        if role_str not in ("system", "user", "assistant", "tool"):
            role_str = "assistant"
        base: Dict[str, Any] = {"role": role_str, "content": self.content}
        if role_str == "tool" and self.tool_call_id:
            base["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            base["tool_calls"] = self.tool_calls
        return base

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        role_str = self.role.value if isinstance(self.role, MessageRole) else self.role
        d: Dict[str, Any] = {
            "id": self.id,
            "role": role_str,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count,
            "priority": self.priority.value,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "metadata": self.metadata,
        }
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConversationMessage:
        """Deserialize from dictionary."""
        priority_val = data.get("priority", 50)
        try:
            priority = MessagePriority(priority_val)
        except ValueError:
            priority = MessagePriority.MEDIUM

        return cls(
            id=data.get("id", str(uuid4())),
            role=data["role"],
            content=data["content"],
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if "timestamp" in data
                else datetime.now(tz=timezone.utc)
            ),
            token_count=data.get("token_count", 0),
            priority=priority,
            tool_name=data.get("tool_name"),
            tool_call_id=data.get("tool_call_id"),
            tool_calls=data.get("tool_calls"),
            name=data.get("name"),
            metadata=data.get("metadata", {}),
        )

    def _normalize_role(self) -> str:
        """Normalize role for provider API (only system/user/assistant/tool)."""
        if self.role in ("system", "user", "assistant", "tool"):
            return self.role
        if self.role == "tool_result":
            return "tool"
        return "assistant"


__all__ = [
    "ConversationMessage",
    "MESSAGE_SOURCE_METADATA_KEY",
    "MessagePriority",
    "MessageRole",
    "MessageSource",
]
