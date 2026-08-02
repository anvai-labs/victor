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

"""State-passed architecture for coordinator decoupling.

This module implements the state-passed architecture pattern, which decouples
coordinators from the orchestrator by using immutable context snapshots and
state transitions.

Key Components:
- ContextSnapshot: Immutable snapshot of orchestrator state
- StateTransition: Encapsulates state changes and side effects
- CoordinatorResult: Combines transitions with metadata

Benefits:
- Coordinators become pure functions (no direct orchestrator mutation)
- Easier testing (no need to mock full orchestrator)
- Better thread safety (immutable snapshots)
- Clearer data flow (transitions explicitly declared)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from enum import Enum
from datetime import datetime, timezone

if TYPE_CHECKING:
    from victor.providers.base import Message, ToolCall
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


class TransitionType(Enum):
    """Types of state transitions."""

    # Message transitions
    ADD_MESSAGE = "add_message"
    ADD_MESSAGES = "add_messages"

    # State transitions
    UPDATE_STATE = "update_state"
    DELETE_STATE = "delete_state"

    # Session transitions
    CREATE_SESSION = "create_session"
    CLOSE_SESSION = "close_session"

    # Tool transitions
    EXECUTE_TOOL = "execute_tool"
    TOOL_RESULT = "tool_result"

    # Metadata transitions
    UPDATE_CAPABILITY = "update_capability"
    UPDATE_STAGE = "update_stage"

    # Composite
    BATCH = "batch"


@dataclass(frozen=True)
class ContextSnapshot:
    """Immutable snapshot of orchestrator state at a point in time.

    This snapshot provides coordinators with read-only access to orchestrator
    state without holding a reference to the orchestrator itself. This enables
    the state-passed architecture pattern.

    The snapshot is intentionally frozen (immutable) to prevent coordinators
    from modifying state directly. All state changes must be expressed through
    StateTransition objects.
    """

    # Core conversation state
    messages: tuple["Message", ...]  # Conversation history
    session_id: str
    conversation_stage: str

    # Configuration
    settings: "Settings"
    model: str
    provider: str
    max_tokens: int
    temperature: float

    # State (key-value store)
    conversation_state: Dict[str, Any]
    session_state: Dict[str, Any]

    # Metadata
    observed_files: tuple[str, ...]  # Files seen during session
    capabilities: Dict[str, Any]  # Capability flags and values

    # Timestamp
    snapshot_timestamp: float = field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp()
    )

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state value, checking conversation state first.

        Args:
            key: State key
            default: Default value if not found

        Returns:
            State value or default
        """
        if key in self.conversation_state:
            return self.conversation_state[key]
        return self.session_state.get(key, default)

    def has_capability(self, capability: str) -> bool:
        """Check if a capability is enabled.

        Args:
            capability: Capability name

        Returns:
            True if capability exists and is truthy
        """
        return bool(self.capabilities.get(capability))

    def get_capability_value(self, capability: str) -> Any:
        """Get a capability value.

        Args:
            capability: Capability name

        Returns:
            Capability value or None
        """
        return self.capabilities.get(capability)

    @property
    def is_complete(self) -> bool:
        """Check if conversation is complete."""
        return self.conversation_stage == "complete"

    @property
    def message_count(self) -> int:
        """Get number of messages in history."""
        return len(self.messages)


@dataclass
class StateTransition:
    """Encapsulates a state change or side effect.

    State transitions are returned by coordinators to express what should
    happen next. The orchestrator is responsible for applying these transitions.

    Transitions are intentionally serializable and can be batched for
    atomic application.
    """

    transition_type: TransitionType
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate transition data."""
        if self.transition_type == TransitionType.ADD_MESSAGE:
            if "message" not in self.data:
                raise ValueError("ADD_MESSAGE transition requires 'message' data")
        elif self.transition_type == TransitionType.UPDATE_STATE:
            if "key" not in self.data or "value" not in self.data:
                raise ValueError("UPDATE_STATE transition requires 'key' and 'value' data")
        elif self.transition_type == TransitionType.EXECUTE_TOOL:
            if "tool_name" not in self.data or "arguments" not in self.data:
                raise ValueError(
                    "EXECUTE_TOOL transition requires 'tool_name' and 'arguments' data"
                )


@dataclass
class TransitionBatch:
    """A batch of transitions to be applied atomically."""

    transitions: List[StateTransition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, transition: StateTransition) -> None:
        """Add a transition to the batch.

        Args:
            transition: Transition to add
        """
        self.transitions.append(transition)

    def add_message(self, message: "Message") -> "TransitionBatch":
        """Add an ADD_MESSAGE transition.

        Args:
            message: Message to add

        Returns:
            Self for chaining
        """
        self.add(
            StateTransition(transition_type=TransitionType.ADD_MESSAGE, data={"message": message})
        )
        return self

    def update_state(self, key: str, value: Any, scope: str = "conversation") -> "TransitionBatch":
        """Add an UPDATE_STATE transition.

        Args:
            key: State key
            value: New value
            scope: "conversation" or "session"

        Returns:
            Self for chaining
        """
        self.add(
            StateTransition(
                transition_type=TransitionType.UPDATE_STATE,
                data={"key": key, "value": value, "scope": scope},
            )
        )
        return self

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> "TransitionBatch":
        """Add an EXECUTE_TOOL transition.

        Args:
            tool_name: Tool to execute
            arguments: Tool arguments

        Returns:
            Self for chaining
        """
        self.add(
            StateTransition(
                transition_type=TransitionType.EXECUTE_TOOL,
                data={"tool_name": tool_name, "arguments": arguments},
            )
        )
        return self

    def extend(self, other: "TransitionBatch") -> "TransitionBatch":
        """Extend this batch with another batch.

        Args:
            other: Other batch to merge in

        Returns:
            Self for chaining
        """
        self.transitions.extend(other.transitions)
        self.metadata.update(other.metadata)
        return self

    def is_empty(self) -> bool:
        """Check if batch has no transitions.

        Returns:
            True if no transitions
        """
        return len(self.transitions) == 0


@dataclass
class CoordinatorResult:
    """Result of a coordinator operation.

    Combines state transitions with metadata about the coordinator's
    analysis and decisions.
    """

    # State transitions to apply
    transitions: TransitionBatch

    # Coordinator analysis (optional)
    reasoning: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Control flow
    should_continue: bool = True
    handoff_to: Optional[str] = None  # Next coordinator or phase

    @staticmethod
    def no_op(reasoning: Optional[str] = None) -> "CoordinatorResult":
        """Create a no-op result with no transitions.

        Args:
            reasoning: Optional explanation for no-op

        Returns:
            CoordinatorResult with empty transitions
        """
        return CoordinatorResult(
            transitions=TransitionBatch(),
            reasoning=reasoning,
        )

    @staticmethod
    def transitions_only(
        *transitions: StateTransition, reasoning: Optional[str] = None
    ) -> "CoordinatorResult":
        """Create a result with only transitions.

        Args:
            *transitions: Transitions to include
            reasoning: Optional explanation

        Returns:
            CoordinatorResult with specified transitions
        """
        batch = TransitionBatch()
        batch.transitions.extend(transitions)
        return CoordinatorResult(
            transitions=batch,
            reasoning=reasoning,
        )

    def add_message(self, message: "Message") -> "CoordinatorResult":
        """Add a message transition.

        Args:
            message: Message to add

        Returns:
            Self for chaining
        """
        self.transitions.add_message(message)
        return self

    def update_state(
        self, key: str, value: Any, scope: str = "conversation"
    ) -> "CoordinatorResult":
        """Add a state update transition.

        Args:
            key: State key
            value: New value
            scope: State scope

        Returns:
            Self for chaining
        """
        self.transitions.update_state(key, value, scope)
        return self


class TransitionApplier:
    """Applies state transitions to an orchestrator.

    This class is responsible for taking StateTransition objects and
    applying them to the actual orchestrator state. It's used by the
    orchestrator to process coordinator results.
    """

    def __init__(self, orchestrator: Any) -> None:
        """Initialize the applier.

        Args:
            orchestrator: The orchestrator to apply transitions to
        """
        self._orchestrator = orchestrator

    async def apply(self, transition: StateTransition) -> None:
        """Apply a single transition.

        Args:
            transition: Transition to apply
        """
        ttype = transition.transition_type

        if ttype == TransitionType.ADD_MESSAGE:
            message = transition.data["message"]
            self._orchestrator.add_message(message)

        elif ttype == TransitionType.ADD_MESSAGES:
            messages = transition.data["messages"]
            for msg in messages:
                self._orchestrator.add_message(msg)

        elif ttype == TransitionType.UPDATE_STATE:
            key = transition.data["key"]
            value = transition.data["value"]
            scope = transition.data.get("scope", "conversation")

            if scope == "conversation":
                self._orchestrator.conversation_state[key] = value
            else:
                self._orchestrator.session_state[key] = value

        elif ttype == TransitionType.DELETE_STATE:
            key = transition.data["key"]
            scope = transition.data.get("scope", "conversation")

            if scope == "conversation" and key in self._orchestrator.conversation_state:
                del self._orchestrator.conversation_state[key]
            elif key in self._orchestrator.session_state:
                del self._orchestrator.session_state[key]

        elif ttype == TransitionType.CREATE_SESSION:
            session_id = transition.data.get("session_id")
            kwargs = transition.data.get("kwargs", {})
            if session_id:
                self._orchestrator.create_session(session_id=session_id, **kwargs)
            else:
                self._orchestrator.create_session(**kwargs)

        elif ttype == TransitionType.CLOSE_SESSION:
            session_id = transition.data.get("session_id")
            if session_id:
                self._orchestrator.close_session(session_id=session_id)

        elif ttype == TransitionType.EXECUTE_TOOL:
            tool_name = transition.data["tool_name"]
            arguments = transition.data["arguments"]
            await self._orchestrator.execute_tool(tool_name, arguments)

        elif ttype == TransitionType.UPDATE_CAPABILITY:
            capability = transition.data["capability"]
            value = transition.data.get("value", True)
            self._orchestrator._capabilities[capability] = value

        elif ttype == TransitionType.UPDATE_STAGE:
            stage = transition.data["stage"]
            self._orchestrator.conversation_stage = stage

        else:
            logger.warning(f"Unknown transition type: {ttype}")

    async def apply_batch(self, batch: TransitionBatch) -> None:
        """Apply a batch of transitions atomically.

        Args:
            batch: Batch of transitions to apply
        """
        for transition in batch.transitions:
            await self.apply(transition)


def create_snapshot(orchestrator: Any) -> ContextSnapshot:
    """Create a ContextSnapshot from an orchestrator.

    This is a utility function to easily create snapshots from
    the current orchestrator state.

    Args:
        orchestrator: AgentOrchestrator instance

    Returns:
        Immutable ContextSnapshot
    """
    # Access orchestrator properties
    messages = tuple(orchestrator.messages) if hasattr(orchestrator, "messages") else ()

    # Copy state dictionaries to prevent mutation
    conv_state = dict(getattr(orchestrator, "conversation_state", {}))
    sess_state = dict(getattr(orchestrator, "session_state", {}))

    # Copy capabilities
    caps = dict(getattr(orchestrator, "_capabilities", {}))

    # Copy observed files
    observed = tuple(getattr(orchestrator, "observed_files", []))

    return ContextSnapshot(
        messages=messages,
        session_id=getattr(orchestrator, "session_id", ""),
        conversation_stage=getattr(orchestrator, "conversation_stage", "initial"),
        settings=getattr(orchestrator, "settings", None),
        model=getattr(orchestrator, "model", ""),
        provider=getattr(orchestrator, "provider_name", "")
        or getattr(orchestrator, "provider", ""),
        max_tokens=getattr(orchestrator, "max_tokens", 4096),
        temperature=getattr(orchestrator, "temperature", 0.7),
        conversation_state=conv_state,
        session_state=sess_state,
        observed_files=observed,
        capabilities=caps,
    )
