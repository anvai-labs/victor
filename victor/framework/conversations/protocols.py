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

"""Focused protocols for conversational agent coordination.

This module defines ISP-compliant protocols for multi-turn conversations
between agents. Each protocol has a single, well-defined responsibility.

Protocol Separation:
- ConversationProtocol: Core conversation lifecycle (turn management)
- MessageFormatterProtocol: Message formatting and parsing
- ConversationResultProtocol: Result extraction and aggregation
- ConversationHistoryProtocol: History tracking and retrieval
- ConversationRoutingProtocol: Dynamic routing between agents
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from victor.framework.protocols import (
    FrameworkOrchestratorProtocol as OrchestratorProtocol,
)

# =============================================================================
# Data Classes
# =============================================================================


class ConversationParticipant:
    """A participant in a multi-agent conversation."""

    def __init__(
        self,
        agent_id: str,
        role: str,
        capabilities: List[str],
        persona: Optional[str] = None,
    ) -> None:
        """Initialize conversation participant.

        Args:
            agent_id: Unique agent identifier
            role: Participant role (e.g., "moderator", "expert", "critic")
            capabilities: List of agent capabilities/skills
            persona: Optional persona description
        """
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.persona = persona


class ConversationContext:
    """Context for conversation execution."""

    def __init__(
        self,
        conversation_id: str,
        topic: str,
        protocol: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize conversation context.

        Args:
            conversation_id: Unique conversation identifier
            topic: Conversation topic/goal
            protocol: Conversation protocol (e.g., "request_response", "debate")
            metadata: Optional additional context
        """
        self.conversation_id = conversation_id
        self.topic = topic
        self.protocol = protocol
        self.metadata = metadata or {}
        self.turn_count = 0
        self.current_speaker: Optional[str] = None


class ConversationalMessage:
    """A message in a multi-agent conversation.

    Extends canonical AgentMessage from victor/teams/types.py for compatibility.
    """

    def __init__(
        self,
        sender: str,
        content: str,
        message_type: str = "conversation",
        recipients: Optional[List[str]] = None,
        conversation_id: str = "",
        turn_number: int = 0,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize conversational message.

        Args:
            sender: Sender agent ID
            content: Message content
            message_type: Type of message
            recipients: Optional list of recipient IDs
            conversation_id: Conversation identifier
            turn_number: Turn number in conversation
            reply_to: Optional message ID being replied to
            metadata: Optional message metadata
        """
        self.sender = sender
        self.content = content
        self.message_type = message_type
        self.recipients = recipients
        self.conversation_id = conversation_id
        self.turn_number = turn_number
        self.reply_to = reply_to
        self.metadata = metadata or {}
        self.id = f"{conversation_id}_t{turn_number}_{sender}"

    def to_agent_message(self) -> Dict[str, Any]:
        """Convert to canonical AgentMessage format.

        Returns:
            Dictionary compatible with AgentMessage structure
        """
        return {
            "sender_id": self.sender,
            "content": self.content,
            "message_type": self.message_type,
            "recipient_id": self.recipients[0] if self.recipients else None,
            "data": self.metadata,
            "timestamp": None,  # Added by EventBus
        }


class ConversationalTurn:
    """A single turn in a conversation."""

    def __init__(
        self,
        turn_number: int,
        speaker: str,
        message: ConversationalMessage,
        responses: List[ConversationalMessage],
    ) -> None:
        """Initialize conversational turn.

        Args:
            turn_number: Turn number
            speaker: Speaking agent ID
            message: Message from speaker
            responses: Responses from other agents
        """
        self.turn_number = turn_number
        self.speaker = speaker
        self.message = message
        self.responses = responses


# =============================================================================
# Core Protocol: Conversation Lifecycle
# =============================================================================


@runtime_checkable
class ConversationProtocol(Protocol):
    """Core conversation lifecycle management.

    Single Responsibility: Manage conversation flow and turn-taking.

    Key Methods:
    - initialize: Set up conversation with participants
    - get_next_speaker: Determine who speaks next
    - should_continue: Check if conversation should continue
    - execute_turn: Execute a single conversational turn

    This protocol focuses ONLY on turn management and conversation flow.
    Message formatting, result extraction, and history are separate protocols.
    """

    async def initialize(
        self,
        participants: List[ConversationParticipant],
        context: ConversationContext,
    ) -> None:
        """Initialize conversation with participants.

        Args:
            participants: List of conversation participants
            context: Conversation context

        Raises:
            ValueError: If participants or context invalid
        """
        ...

    async def get_next_speaker(
        self,
        context: ConversationContext,
    ) -> Optional[str]:
        """Get next speaker based on conversation protocol.

        Args:
            context: Current conversation context

        Returns:
            Next speaker agent_id, or None if conversation complete
        """
        ...

    async def should_continue(
        self,
        context: ConversationContext,
    ) -> tuple[bool, Optional[str]]:
        """Check if conversation should continue.

        Args:
            context: Current conversation context

        Returns:
            Tuple of (should_continue, reason)
        """
        ...

    async def execute_turn(
        self,
        speaker: str,
        context: ConversationContext,
        orchestrator: OrchestratorProtocol,
    ) -> ConversationalTurn:
        """Execute a single conversational turn.

        Args:
            speaker: Speaking agent ID
            context: Conversation context
            orchestrator: Orchestrator for executing agent logic

        Returns:
            Completed conversational turn
        """
        ...


# =============================================================================
# Message Formatting Protocol
# =============================================================================


@runtime_checkable
class MessageFormatterProtocol(Protocol):
    """Message formatting and parsing.

    Single Responsibility: Format messages for agents and parse responses.

    Key Methods:
    - format_message: Format message for specific recipient
    - parse_response: Parse agent response into ConversationalMessage

    This protocol is SEPARATE from ConversationProtocol because:
    - Different conversation types may need different formatting
    - Formatting can be swapped independently from turn management
    - Testing is easier with separate protocol
    """

    async def format_message(
        self,
        message: ConversationalMessage,
        recipient: str,
        context: ConversationContext,
    ) -> str:
        """Format message for specific recipient.

        Args:
            message: Message to format
            recipient: Recipient agent ID
            context: Conversation context

        Returns:
            Formatted message string
        """
        ...

    async def parse_response(
        self,
        response: str,
        sender: str,
        context: ConversationContext,
    ) -> ConversationalMessage:
        """Parse agent response into ConversationalMessage.

        Args:
            response: Raw response string from agent
            sender: Sender agent ID
            context: Conversation context

        Returns:
            Parsed ConversationalMessage
        """
        ...


# =============================================================================
# Result Extraction Protocol
# =============================================================================


@runtime_checkable
class ConversationResultProtocol(Protocol):
    """Result extraction and aggregation.

    Single Responsibility: Extract final results from conversation.

    Key Methods:
    - get_result: Extract aggregated result from conversation context
    - format_result: Format result for presentation

    This protocol is SEPARATE from ConversationProtocol because:
    - Different result formats may be needed
    - Aggregation logic is independent of turn management
    - Allows different result strategies for different protocols
    """

    async def get_result(
        self,
        context: ConversationContext,
    ) -> Dict[str, Any]:
        """Extract result from conversation context.

        Args:
            context: Conversation context with history

        Returns:
            Aggregated result dictionary
        """
        ...

    async def format_result(
        self,
        result: Dict[str, Any],
        format: str = "summary",
    ) -> str:
        """Format result for presentation.

        Args:
            result: Result dictionary
            format: Format type (e.g., "summary", "detailed", "json")

        Returns:
            Formatted result string
        """
        ...


# =============================================================================
# History Management Protocol
# =============================================================================


@runtime_checkable
class ConversationHistoryProtocol(Protocol):
    """Conversation history tracking and retrieval.

    Single Responsibility: Track and retrieve conversation history.

    Key Methods:
    - add_turn: Add turn to history
    - get_history: Retrieve conversation history
    - get_last_n_turns: Get last N turns
    - clear_history: Clear conversation history

    This protocol is SEPARATE from ConversationProtocol because:
    - History storage can be implemented independently (in-memory, DB, etc.)
    - History queries don't affect conversation flow
    - Different storage strategies can be swapped
    """

    async def add_turn(
        self,
        context: ConversationContext,
        turn: ConversationalTurn,
    ) -> None:
        """Add turn to conversation history.

        Args:
            context: Conversation context
            turn: Turn to add
        """
        ...

    async def get_history(
        self,
        context: ConversationContext,
    ) -> List[ConversationalTurn]:
        """Get full conversation history.

        Args:
            context: Conversation context

        Returns:
            List of all turns
        """
        ...

    async def get_last_n_turns(
        self,
        context: ConversationContext,
        n: int,
    ) -> List[ConversationalTurn]:
        """Get last N turns from conversation.

        Args:
            context: Conversation context
            n: Number of turns to retrieve

        Returns:
            List of last N turns
        """
        ...

    async def clear_history(
        self,
        context: ConversationContext,
    ) -> None:
        """Clear conversation history.

        Args:
            context: Conversation context
        """
        ...


# =============================================================================
# Routing Protocol
# =============================================================================


@runtime_checkable
class ConversationRoutingProtocol(Protocol):
    """Dynamic routing between conversation participants.

    Single Responsibility: Route messages between participants.

    Key Methods:
    - route_message: Route message to appropriate recipients
    - get_recipients: Determine recipients for message
    - should_reply: Check if agent should reply to message

    This protocol is SEPARATE from ConversationProtocol because:
    - Routing logic can be complex and independent
    - Different routing strategies for different conversation types
    - Testing is easier with isolated routing logic
    """

    async def route_message(
        self,
        message: ConversationalMessage,
        participants: List[ConversationParticipant],
        context: ConversationContext,
    ) -> List[str]:
        """Route message to appropriate recipients.

        Args:
            message: Message to route
            participants: All conversation participants
            context: Conversation context

        Returns:
            List of recipient agent IDs
        """
        ...

    async def get_recipients(
        self,
        speaker: str,
        participants: List[ConversationParticipant],
        context: ConversationContext,
    ) -> List[str]:
        """Get recipients for speaker's message.

        Args:
            speaker: Speaking agent ID
            participants: All conversation participants
            context: Conversation context

        Returns:
            List of recipient agent IDs
        """
        ...

    async def should_reply(
        self,
        agent: str,
        message: ConversationalMessage,
        context: ConversationContext,
    ) -> bool:
        """Check if agent should reply to message.

        Args:
            agent: Agent ID
            message: Message to check
            context: Conversation context

        Returns:
            True if agent should reply
        """
        ...


# =============================================================================
# Concrete Protocol Implementations
# =============================================================================


class BaseConversationProtocol:
    """Base implementation for conversation protocols.

    Provides default implementations that can be overridden by specific
    protocol types (RequestResponse, Debate, Consensus).
    """

    def __init__(
        self,
        max_turns: int = 10,
        require_all_participants: bool = False,
    ):
        """Initialize base protocol.

        Args:
            max_turns: Maximum conversation turns
            require_all_participants: Whether all participants must speak
        """
        self.max_turns = max_turns
        self.require_all_participants = require_all_participants
        self._participants: List[ConversationParticipant] = []
        self._turn_order: List[str] = []
        self._current_index = 0

    async def initialize(
        self,
        participants: List[ConversationParticipant],
        context: ConversationContext,
    ) -> None:
        """Initialize conversation with participants."""
        self._participants = participants
        self._turn_order = [p.agent_id for p in participants]
        self._current_index = 0
        context.metadata["protocol_initialized"] = True

    async def get_next_speaker(
        self,
        context: ConversationContext,
    ) -> Optional[str]:
        """Get next speaker (round-robin by default)."""
        if not self._turn_order:
            return None

        speaker = self._turn_order[self._current_index % len(self._turn_order)]
        self._current_index += 1
        return speaker

    async def should_continue(
        self,
        context: ConversationContext,
    ) -> tuple[bool, Optional[str]]:
        """Check if conversation should continue."""
        if context.turn_count >= self.max_turns:
            return False, "Maximum turns reached"
        if context.metadata.get("consensus_reached"):
            return False, "Consensus reached"
        if context.metadata.get("terminate"):
            return False, context.metadata.get("termination_reason", "Terminated")
        return True, None


class RequestResponseProtocol(BaseConversationProtocol):
    """Simple request-response conversation protocol.

    One agent asks questions, others respond in turn.
    """

    def __init__(
        self,
        questioner: str,
        max_questions: int = 5,
    ):
        """Initialize request-response protocol.

        Args:
            questioner: Agent ID who asks questions
            max_questions: Maximum number of questions
        """
        super().__init__(max_turns=max_questions * 2)
        self.questioner = questioner
        self.max_questions = max_questions
        self._question_count = 0
        self._awaiting_response = False

    async def get_next_speaker(
        self,
        context: ConversationContext,
    ) -> Optional[str]:
        """Alternate between questioner and responders."""
        if not self._awaiting_response:
            self._awaiting_response = True
            self._question_count += 1
            return self.questioner
        else:
            self._awaiting_response = False
            # Next responder
            responders = [p.agent_id for p in self._participants if p.agent_id != self.questioner]
            if responders:
                idx = (context.turn_count // 2) % len(responders)
                return responders[idx]
            return None

    async def should_continue(
        self,
        context: ConversationContext,
    ) -> tuple[bool, Optional[str]]:
        """Continue until max questions reached."""
        if self._question_count >= self.max_questions and not self._awaiting_response:
            return False, "All questions answered"
        return await super().should_continue(context)


class DebateProtocol(BaseConversationProtocol):
    """Debate conversation protocol.

    Two sides present arguments for and against a position.
    """

    def __init__(
        self,
        proposition: str,
        rounds: int = 3,
        moderator: Optional[str] = None,
    ):
        """Initialize debate protocol.

        Args:
            proposition: The proposition being debated
            rounds: Number of debate rounds
            moderator: Optional moderator agent ID
        """
        super().__init__(max_turns=rounds * 2 + (2 if moderator else 0))
        self.proposition = proposition
        self.rounds = rounds
        self.moderator = moderator
        self._current_round = 0
        self._for_speaker: Optional[str] = None
        self._against_speaker: Optional[str] = None

    async def initialize(
        self,
        participants: List[ConversationParticipant],
        context: ConversationContext,
    ) -> None:
        """Initialize debate with FOR and AGAINST sides."""
        await super().initialize(participants, context)

        # Assign sides based on role or first two participants
        for p in participants:
            if p.role in ["proponent", "for", "defender"]:
                self._for_speaker = p.agent_id
            elif p.role in ["opponent", "against", "critic"]:
                self._against_speaker = p.agent_id

        # Fallback to first two non-moderator participants
        non_moderators = [p for p in participants if p.agent_id != self.moderator]
        if not self._for_speaker and len(non_moderators) > 0:
            self._for_speaker = non_moderators[0].agent_id
        if not self._against_speaker and len(non_moderators) > 1:
            self._against_speaker = non_moderators[1].agent_id

        context.metadata["proposition"] = self.proposition
        context.metadata["for_speaker"] = self._for_speaker
        context.metadata["against_speaker"] = self._against_speaker

    async def get_next_speaker(
        self,
        context: ConversationContext,
    ) -> Optional[str]:
        """Alternate between FOR and AGAINST speakers."""
        turn = context.turn_count

        # Opening statement by moderator
        if turn == 0 and self.moderator:
            return self.moderator

        # Alternate FOR and AGAINST
        effective_turn = turn - (1 if self.moderator else 0)
        if effective_turn % 2 == 0:
            return self._for_speaker
        else:
            self._current_round += 1
            return self._against_speaker

    async def should_continue(
        self,
        context: ConversationContext,
    ) -> tuple[bool, Optional[str]]:
        """Continue for specified rounds."""
        if self._current_round >= self.rounds:
            return False, f"Debate completed after {self.rounds} rounds"
        return await super().should_continue(context)


class ConsensusProtocol(BaseConversationProtocol):
    """Consensus-building conversation protocol.

    Participants discuss and vote to reach agreement.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        max_voting_rounds: int = 3,
    ):
        """Initialize consensus protocol.

        Args:
            threshold: Required agreement threshold (0.0 - 1.0)
            max_voting_rounds: Maximum voting rounds
        """
        super().__init__(max_turns=30)
        self.threshold = threshold
        self.max_voting_rounds = max_voting_rounds
        self._voting_round = 0
        self._votes: Dict[str, bool] = {}
        self._in_voting_phase = False

    async def initialize(
        self,
        participants: List[ConversationParticipant],
        context: ConversationContext,
    ) -> None:
        """Initialize consensus protocol."""
        await super().initialize(participants, context)
        context.metadata["consensus_threshold"] = self.threshold
        context.metadata["votes"] = {}

    async def get_next_speaker(
        self,
        context: ConversationContext,
    ) -> Optional[str]:
        """Get next speaker, entering voting phase periodically."""
        # Check if we should enter voting phase
        if context.turn_count > 0 and context.turn_count % (len(self._participants) * 2) == 0:
            self._in_voting_phase = True
            self._votes.clear()

        if self._in_voting_phase:
            # Get next voter
            voters_needed = [
                p.agent_id for p in self._participants if p.agent_id not in self._votes
            ]
            if voters_needed:
                return voters_needed[0]
            else:
                # Check if consensus reached
                await self._check_consensus(context)
                self._in_voting_phase = False
                self._voting_round += 1

        return await super().get_next_speaker(context)

    async def _check_consensus(self, context: ConversationContext) -> None:
        """Check if consensus has been reached."""
        if not self._votes:
            return

        agree_count = sum(1 for v in self._votes.values() if v)
        total = len(self._votes)
        agreement_ratio = agree_count / total if total > 0 else 0

        context.metadata["votes"] = dict(self._votes)
        context.metadata["agreement_ratio"] = agreement_ratio

        if agreement_ratio >= self.threshold:
            context.metadata["consensus_reached"] = True
            context.metadata["consensus_type"] = "agreement"
        elif self._voting_round >= self.max_voting_rounds:
            context.metadata["consensus_reached"] = True
            context.metadata["consensus_type"] = "no_agreement"

    async def should_continue(
        self,
        context: ConversationContext,
    ) -> tuple[bool, Optional[str]]:
        """Continue until consensus or max voting rounds."""
        if context.metadata.get("consensus_reached"):
            consensus_type = context.metadata.get("consensus_type", "unknown")
            ratio = context.metadata.get("agreement_ratio", 0)
            return False, f"Consensus {consensus_type} (ratio: {ratio:.1%})"

        if self._voting_round >= self.max_voting_rounds:
            return False, "Max voting rounds reached without consensus"

        return await super().should_continue(context)
