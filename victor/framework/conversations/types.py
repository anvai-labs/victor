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

"""Type definitions for conversational agent coordination.

This module provides the core data types used throughout the conversation
framework. Types are designed to be compatible with the canonical AgentMessage
from victor/teams/types.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class MessageType(Enum):
    """Types of conversational messages."""

    CONVERSATION = "conversation"  # Regular conversation message
    QUESTION = "question"  # A question to be answered
    ANSWER = "answer"  # Answer to a question
    PROPOSAL = "proposal"  # A proposal for consideration
    VOTE = "vote"  # A vote on a proposal
    ARGUMENT_FOR = "argument_for"  # Argument supporting a position
    ARGUMENT_AGAINST = "argument_against"  # Argument opposing a position
    SUMMARY = "summary"  # Summary of conversation
    CONSENSUS = "consensus"  # Consensus reached
    SYSTEM = "system"  # System message (e.g., turn change)


class ConversationStatus(Enum):
    """Status of a conversation."""

    PENDING = "pending"  # Not yet started
    ACTIVE = "active"  # Currently in progress
    COMPLETED = "completed"  # Successfully completed
    TERMINATED = "terminated"  # Terminated early
    TIMED_OUT = "timed_out"  # Exceeded time limit
    FAILED = "failed"  # Failed due to error


@dataclass
class ConversationParticipant:
    """A participant in a multi-agent conversation.

    Attributes:
        id: Unique identifier for the participant
        role: Participant's role (e.g., moderator, expert, critic)
        name: Human-readable name
        capabilities: List of capabilities/skills
        persona: Optional persona description for the agent
        metadata: Optional additional metadata
    """

    id: str
    role: str
    name: str = ""
    capabilities: List[str] = field(default_factory=list)
    persona: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = self.id


@dataclass
class ConversationalMessage:
    """A message in a multi-agent conversation.

    Designed for compatibility with victor/teams/types.py AgentMessage.

    Attributes:
        id: Unique message identifier
        conversation_id: Conversation this message belongs to
        sender: Sender agent ID
        content: Message content
        message_type: Type of message
        recipients: Optional list of recipient IDs (None = broadcast)
        turn_number: Turn number in conversation
        reply_to: Optional message ID being replied to
        timestamp: When message was created
        metadata: Additional message metadata
        requires_response: Whether a response is expected
    """

    sender: str
    content: str
    message_type: MessageType = MessageType.CONVERSATION
    recipients: Optional[List[str]] = None
    conversation_id: str = ""
    turn_number: int = 0
    reply_to: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    requires_response: bool = True
    id: str = field(default_factory=lambda: str(uuid4())[:8])

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_agent_message(self) -> Dict[str, Any]:
        """Convert to canonical AgentMessage format.

        Returns:
            Dictionary compatible with AgentMessage structure
        """
        return {
            "sender_id": self.sender,
            "content": self.content,
            "message_type": self.message_type.value,
            "recipient_id": self.recipients[0] if self.recipients else None,
            "data": {
                "conversation_id": self.conversation_id,
                "turn_number": self.turn_number,
                "reply_to": self.reply_to,
                **self.metadata,
            },
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def from_agent_message(
        cls, msg: Dict[str, Any], conversation_id: str = ""
    ) -> "ConversationalMessage":
        """Create from canonical AgentMessage format.

        Args:
            msg: AgentMessage dictionary
            conversation_id: Conversation ID to assign

        Returns:
            ConversationalMessage instance
        """
        data = msg.get("data", {})
        return cls(
            sender=msg.get("sender_id", "unknown"),
            content=msg.get("content", ""),
            message_type=MessageType(msg.get("message_type", "conversation")),
            recipients=[msg["recipient_id"]] if msg.get("recipient_id") else None,
            conversation_id=data.get("conversation_id", conversation_id),
            turn_number=data.get("turn_number", 0),
            reply_to=data.get("reply_to"),
            metadata={
                k: v
                for k, v in data.items()
                if k not in ["conversation_id", "turn_number", "reply_to"]
            },
        )


@dataclass
class ConversationContext:
    """Context shared across a conversation.

    Holds all state needed for conversation execution.

    Attributes:
        conversation_id: Unique conversation identifier
        topic: Conversation topic/goal
        protocol: Conversation protocol type
        participants: Dictionary of participant ID -> ConversationParticipant
        current_turn: Current turn number
        current_speaker: Current speaker ID
        started_at: When conversation started
        shared_state: Shared state accessible to all participants
        is_terminated: Whether conversation has ended
        termination_reason: Why conversation ended
        metadata: Additional metadata
    """

    conversation_id: str
    topic: str
    protocol: str
    participants: Dict[str, ConversationParticipant] = field(default_factory=dict)
    current_turn: int = 0
    current_speaker: Optional[str] = None
    started_at: Optional[datetime] = None
    shared_state: Dict[str, Any] = field(default_factory=dict)
    is_terminated: bool = False
    termination_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.now()

    def add_participant(self, participant: ConversationParticipant) -> None:
        """Add a participant to the conversation."""
        self.participants[participant.id] = participant

    def get_participant(self, participant_id: str) -> Optional[ConversationParticipant]:
        """Get participant by ID."""
        return self.participants.get(participant_id)

    def list_participants(self) -> List[ConversationParticipant]:
        """List all participants."""
        return list(self.participants.values())

    def terminate(self, reason: str) -> None:
        """Terminate the conversation."""
        self.is_terminated = True
        self.termination_reason = reason


@dataclass
class ConversationResult:
    """Result of a completed conversation.

    Attributes:
        conversation_id: Conversation identifier
        status: Final conversation status
        outcome: Primary outcome/decision
        summary: Summary of discussion
        votes: Optional vote tallies (for consensus protocols)
        final_speaker: Who had the final word
        total_turns: Total number of turns
        duration_seconds: Conversation duration
        metadata: Additional result metadata
    """

    conversation_id: str
    status: ConversationStatus
    outcome: Optional[str] = None
    summary: Optional[str] = None
    votes: Optional[Dict[str, int]] = None
    final_speaker: Optional[str] = None
    total_turns: int = 0
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "status": self.status.value,
            "outcome": self.outcome,
            "summary": self.summary,
            "votes": self.votes,
            "final_speaker": self.final_speaker,
            "total_turns": self.total_turns,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


@dataclass
class ConversationalTurn:
    """A single turn in a conversation.

    Attributes:
        turn_number: Turn number in sequence
        speaker: Speaker agent ID
        message: The speaker's message
        responses: Responses from other participants
        timestamp: When turn occurred
        duration_seconds: Turn duration
    """

    turn_number: int
    speaker: str
    message: ConversationalMessage
    responses: List[ConversationalMessage] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    duration_seconds: float = 0.0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


__all__ = [
    "MessageType",
    "ConversationStatus",
    "ConversationParticipant",
    "ConversationalMessage",
    "ConversationContext",
    "ConversationResult",
    "ConversationalTurn",
]
