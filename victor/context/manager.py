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

"""Context management for maintaining relevant conversation context.

This module handles:
1. Token counting for messages and files
2. Context window budgeting
3. Automatic pruning when approaching limits
4. Smart file selection based on relevance
5. Message prioritization
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class PruningStrategy(str, Enum):
    """Strategy for pruning context when approaching token limit."""

    FIFO = "fifo"  # Remove oldest messages first
    PRIORITY = "priority"  # Remove lowest priority messages
    SMART = "smart"  # Keep system messages, remove middle conversation
    SUMMARIZE = "summarize"  # Summarize old messages (future)


class Message(BaseModel):
    """Represents a message in the conversation."""

    role: str = Field(description="Message role (user/assistant/system)")
    content: str = Field(description="Message content")
    tokens: int = Field(default=0, description="Token count for this message")
    priority: int = Field(default=5, description="Priority (1-10, higher = more important)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class FileContext(BaseModel):
    """Represents a file included in context."""

    path: str = Field(description="File path")
    content: str = Field(description="File content")
    tokens: int = Field(default=0, description="Token count")
    relevance_score: float = Field(default=0.0, description="Relevance score (0-1)")
    line_range: Optional[Tuple[int, int]] = Field(default=None, description="Specific line range")


class ContextWindow(BaseModel):
    """Represents the current context window state."""

    messages: List[Message] = Field(default_factory=list, description="Conversation messages")
    files: List[FileContext] = Field(default_factory=list, description="Included files")
    total_tokens: int = Field(default=0, description="Total tokens in context")
    max_tokens: int = Field(default=128000, description="Maximum context window size")
    reserved_tokens: int = Field(default=4096, description="Reserved for response")

    @property
    def available_tokens(self) -> int:
        """Get available tokens for new content."""
        return self.max_tokens - self.total_tokens - self.reserved_tokens

    @property
    def usage_percentage(self) -> float:
        """Get context usage as percentage."""
        return (self.total_tokens / (self.max_tokens - self.reserved_tokens)) * 100


class ProjectContextLoader:
    """Manages conversation context and token budgeting.

    Features:
    - Native-accelerated token counting with Python fallback
    - Automatic pruning when approaching limits
    - Smart file selection based on relevance
    - Message prioritization
    - Context window optimization
    """

    def __init__(
        self,
        model: str = "gpt-4",
        max_tokens: int = 128000,
        reserved_tokens: int = 4096,
        pruning_strategy: PruningStrategy = PruningStrategy.SMART,
        prune_threshold: float = 0.85,
    ):
        """Initialize context manager.

        Args:
            model: Model name for token encoding
            max_tokens: Maximum context window size
            reserved_tokens: Tokens to reserve for response
            pruning_strategy: Strategy for pruning context
            prune_threshold: Prune when usage exceeds this percentage
        """
        self.model = model
        self.max_tokens = max_tokens
        self.reserved_tokens = reserved_tokens
        self.pruning_strategy = pruning_strategy
        self.prune_threshold = prune_threshold

        # Kept for backwards compatibility with callers that access ``encoder``.
        # Actual token counting is delegated to the shared native tokenizer shim.
        self.encoder = None

        # Context window
        self.context = ContextWindow(max_tokens=max_tokens, reserved_tokens=reserved_tokens)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count

        Uses the shared native tokenizer wrapper, which prefers Rust,
        then falls back to tiktoken, then word-based estimation.

        Returns:
            Number of tokens
        """
        from victor.processing.native.tokenizer import count_tokens

        return count_tokens(text)

    def add_message(
        self,
        role: str,
        content: str,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a message to context.

        Args:
            role: Message role (user/assistant/system)
            content: Message content
            priority: Priority level (1-10)
            metadata: Additional metadata
        """
        tokens = self.count_tokens(content)

        message = Message(
            role=role,
            content=content,
            tokens=tokens,
            priority=priority,
            metadata=metadata or {},
        )

        self.context.messages.append(message)
        self.context.total_tokens += tokens

        # Auto-prune if needed
        if self.context.usage_percentage > (self.prune_threshold * 100):
            self._auto_prune()

    def add_file(
        self,
        path: str,
        content: str,
        relevance_score: float = 1.0,
        line_range: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Add a file to context.

        Args:
            path: File path
            content: File content
            relevance_score: Relevance score (0-1)
            line_range: Optional specific line range
        """
        # Extract line range if specified
        if line_range:
            lines = content.split("\n")
            start, end = line_range
            content = "\n".join(lines[start:end])

        tokens = self.count_tokens(content)

        file_ctx = FileContext(
            path=path,
            content=content,
            tokens=tokens,
            relevance_score=relevance_score,
            line_range=line_range,
        )

        self.context.files.append(file_ctx)
        self.context.total_tokens += tokens

        # Auto-prune if needed
        if self.context.usage_percentage > (self.prune_threshold * 100):
            self._auto_prune()

    def get_context_for_prompt(self) -> List[Dict[str, str]]:
        """Get context formatted for LLM prompt.

        Returns:
            List of message dicts with role and content
        """
        messages = []

        # Add file context as system messages
        if self.context.files:
            file_content = self._format_file_context()
            messages.append({"role": "system", "content": file_content})

        # Add conversation messages
        for msg in self.context.messages:
            messages.append({"role": msg.role, "content": msg.content})

        return messages

    def _format_file_context(self) -> str:
        """Format file context for prompt.

        Returns:
            Formatted file context string
        """
        parts = ["Here are the relevant files for context:\n"]

        for file in sorted(self.context.files, key=lambda f: f.relevance_score, reverse=True):
            parts.append(f"\n--- {file.path} ---")
            if file.line_range:
                parts.append(f"(Lines {file.line_range[0]}-{file.line_range[1]})")
            parts.append(f"\n{file.content}\n")

        return "\n".join(parts)

    def _auto_prune(self) -> None:
        """Automatically prune context based on strategy."""
        print(f"⚠️  Context at {self.context.usage_percentage:.1f}% - pruning...")

        if self.pruning_strategy == PruningStrategy.FIFO:
            self._prune_fifo()
        elif self.pruning_strategy == PruningStrategy.PRIORITY:
            self._prune_by_priority()
        elif self.pruning_strategy == PruningStrategy.SMART:
            self._prune_smart()
        else:
            self._prune_fifo()  # Default

        print(f"✓ Pruned to {self.context.usage_percentage:.1f}%")

    def _prune_fifo(self) -> None:
        """Prune oldest messages first (keep system messages)."""
        # Keep system messages and most recent messages
        target_tokens = int(self.max_tokens * (self.prune_threshold - 0.1))

        system_messages = [m for m in self.context.messages if m.role == "system"]
        other_messages = [m for m in self.context.messages if m.role != "system"]

        # Calculate tokens to remove
        tokens_to_remove = self.context.total_tokens - target_tokens

        # Remove oldest non-system messages
        # Iterate in reverse, removing oldest messages until we hit the target
        removed_tokens = 0
        messages_to_keep: List[Message] = []

        for msg in reversed(other_messages):
            if removed_tokens < tokens_to_remove:
                removed_tokens += msg.tokens
            else:
                messages_to_keep.append(msg)  # Append (will reverse once at end)

        # Reverse to maintain original order
        messages_to_keep = list(reversed(messages_to_keep))

        # Update context
        self.context.messages = system_messages + messages_to_keep
        self.context.total_tokens -= removed_tokens

    def _prune_by_priority(self) -> None:
        """Prune lowest priority messages."""
        target_tokens = int(self.max_tokens * (self.prune_threshold - 0.1))
        tokens_to_remove = self.context.total_tokens - target_tokens

        # Sort by priority (lowest first)
        sorted_messages = sorted(
            self.context.messages, key=lambda m: (m.role == "system", m.priority)
        )

        removed_tokens = 0
        messages_to_keep = []

        for msg in sorted_messages:
            if msg.role == "system" or removed_tokens >= tokens_to_remove:
                messages_to_keep.append(msg)
            else:
                removed_tokens += msg.tokens

        self.context.messages = messages_to_keep
        self.context.total_tokens -= removed_tokens

    def _prune_smart(self) -> None:
        """Smart pruning: Keep system, first user message, and recent messages."""
        target_tokens = int(self.max_tokens * (self.prune_threshold - 0.1))

        # Always keep system messages
        system_messages = [m for m in self.context.messages if m.role == "system"]

        # Keep first user message (context setting)
        first_user = next((m for m in self.context.messages if m.role == "user"), None)

        # Keep most recent messages
        recent_messages: List[Message] = []
        recent_tokens = 0
        for msg in reversed(self.context.messages):
            if msg.role == "system" or msg == first_user:
                continue
            if recent_tokens + msg.tokens <= (
                target_tokens
                - sum(m.tokens for m in system_messages)
                - (first_user.tokens if first_user else 0)
            ):
                recent_messages.insert(0, msg)
                recent_tokens += msg.tokens
            else:
                break

        # Reconstruct messages
        new_messages = system_messages.copy()
        if first_user and first_user not in system_messages:
            new_messages.append(first_user)

        # Add gap indicator if we skipped messages
        if len(self.context.messages) > len(system_messages) + len(recent_messages) + (
            1 if first_user else 0
        ):
            new_messages.append(
                Message(
                    role="system",
                    content="[... earlier conversation pruned for context length ...]",
                    tokens=self.count_tokens(
                        "[... earlier conversation pruned for context length ...]"
                    ),
                    priority=10,
                )
            )

        new_messages.extend(recent_messages)

        removed_tokens = self.context.total_tokens - sum(m.tokens for m in new_messages)
        self.context.messages = new_messages
        self.context.total_tokens -= removed_tokens

    def clear_files(self) -> None:
        """Clear all file context."""
        tokens_removed = sum(f.tokens for f in self.context.files)
        self.context.files.clear()
        self.context.total_tokens -= tokens_removed

    def clear_messages(self, keep_system: bool = True) -> None:
        """Clear conversation messages.

        Args:
            keep_system: Whether to keep system messages
        """
        if keep_system:
            system_messages = [m for m in self.context.messages if m.role == "system"]
            removed_tokens = sum(m.tokens for m in self.context.messages if m.role != "system")
            self.context.messages = system_messages
        else:
            removed_tokens = sum(m.tokens for m in self.context.messages)
            self.context.messages.clear()

        self.context.total_tokens -= removed_tokens

    def get_stats(self) -> Dict[str, Any]:
        """Get context statistics.

        Returns:
            Dictionary with context stats
        """
        return {
            "total_messages": len(self.context.messages),
            "total_files": len(self.context.files),
            "total_tokens": self.context.total_tokens,
            "max_tokens": self.max_tokens,
            "available_tokens": self.context.available_tokens,
            "usage_percentage": self.context.usage_percentage,
            "pruning_strategy": self.pruning_strategy.value,
            "messages_by_role": {
                "system": len([m for m in self.context.messages if m.role == "system"]),
                "user": len([m for m in self.context.messages if m.role == "user"]),
                "assistant": len([m for m in self.context.messages if m.role == "assistant"]),
            },
        }
