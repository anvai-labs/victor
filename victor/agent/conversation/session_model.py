from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from victor.agent.conversation.types import ConversationMessage
from victor.agent.ml_metadata import ContextSize, ModelFamily, ModelSize


@dataclass(frozen=True)
class ConversationSessionSummary:
    """Persisted picker metadata, without transcript or execution-state hydration."""

    session_id: str
    title: str
    created_at: datetime
    last_activity: datetime
    provider: Optional[str]
    model: Optional[str]
    profile: Optional[str]
    message_count: int
    preview_count: int
    tags: tuple[str, ...]


@dataclass
class ConversationSession:
    """A conversation session with context management."""

    session_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    # Context management
    max_tokens: int = 100000  # Claude's context window
    reserved_tokens: int = 4096  # Reserved for response
    current_tokens: int = 0

    # Session metadata
    project_path: Optional[str] = None
    active_files: List[str] = field(default_factory=list)
    tool_usage_count: int = 0

    # Provider info (original fields)
    provider: Optional[str] = None
    model: Optional[str] = None
    profile: Optional[str] = None  # User-facing profile name (e.g., "groq-fast")

    # ML/RL-friendly derived fields for multi-dimensional learning
    model_family: Optional[ModelFamily] = None  # Architecture family
    model_size: Optional[ModelSize] = None  # Size category
    model_params_b: Optional[float] = None  # Parameters in billions (numeric)
    context_size: Optional[ContextSize] = None  # Context window category
    context_tokens: Optional[int] = None  # Actual context window tokens
    tool_capable: bool = False  # Whether model supports tool calling
    is_moe: bool = False  # Mixture of Experts architecture
    is_reasoning: bool = False  # Explicit reasoning model (R1, o1)

    # Rich session metadata (for slash commands, CLI, resume)
    title: Optional[str] = None  # Session title (generated or user-provided)
    tags: List[str] = field(default_factory=list)  # User-assigned tags

    # State persistence (for session resume)
    conversation_state: Optional[Dict[str, Any]] = None  # ConversationStateMachine.to_dict()
    execution_state: Optional[Dict[str, Any]] = None  # ExecutionState.to_dict()
    session_ledger: Optional[Dict[str, Any]] = None  # SessionLedger.to_dict()
    compaction_hierarchy: Optional[Dict[str, Any]] = None  # Message compaction hierarchy

    # Preview messages (separated from regular messages for display)
    preview_messages: List[ConversationMessage] = field(default_factory=list)

    @property
    def available_tokens(self) -> int:
        """Get available tokens for new messages."""
        return self.max_tokens - self.reserved_tokens - self.current_tokens

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "max_tokens": self.max_tokens,
            "reserved_tokens": self.reserved_tokens,
            "current_tokens": self.current_tokens,
            "project_path": self.project_path,
            "active_files": self.active_files,
            "tool_usage_count": self.tool_usage_count,
            "provider": self.provider,
            "model": self.model,
            "profile": self.profile,
            # ML-friendly fields
            "model_family": self.model_family.value if self.model_family else None,
            "model_size": self.model_size.value if self.model_size else None,
            "model_params_b": self.model_params_b,
            "context_size": self.context_size.value if self.context_size else None,
            "context_tokens": self.context_tokens,
            "tool_capable": self.tool_capable,
            "is_moe": self.is_moe,
            "is_reasoning": self.is_reasoning,
            # Rich session metadata
            "title": self.title,
            "tags": self.tags,
            # State persistence
            "conversation_state": self.conversation_state,
            "execution_state": self.execution_state,
            "session_ledger": self.session_ledger,
            "compaction_hierarchy": self.compaction_hierarchy,
        }
