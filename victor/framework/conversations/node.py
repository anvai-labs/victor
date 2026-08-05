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

"""ConversationalNode for StateGraph workflows.

This module provides the ConversationalNode, which enables multi-turn
conversations between agents within StateGraph workflows.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import uuid4

from victor.framework.conversations.types import (
    ConversationalMessage,
    ConversationalTurn,
    ConversationContext,
    ConversationParticipant,
    ConversationResult,
    ConversationStatus,
    MessageType,
)
from victor.framework.conversations.history import ConversationHistory
from victor.framework.conversations.router import MessageRouter, RoutingStrategy

if TYPE_CHECKING:
    from victor.framework.conversations.protocols import ConversationProtocol

logger = logging.getLogger(__name__)


@dataclass
class ConversationalNodeConfig:
    """Configuration for a ConversationalNode.

    Attributes:
        max_turns: Maximum number of conversation turns
        timeout_seconds: Maximum time for conversation
        require_consensus: Whether consensus is required for completion
        consensus_threshold: Threshold for consensus (0.0 - 1.0)
        allow_early_termination: Whether agents can terminate early
        routing_strategy: Message routing strategy
        emit_events: Whether to emit EventBus events
    """

    max_turns: int = 10
    timeout_seconds: float = 300.0
    require_consensus: bool = False
    consensus_threshold: float = 0.7
    allow_early_termination: bool = True
    routing_strategy: RoutingStrategy = RoutingStrategy.BROADCAST
    emit_events: bool = True


class ConversationalNode:
    """A workflow node that hosts multi-turn conversations.

    Enables agents to engage in dynamic conversations within StateGraph
    workflows. Supports multiple conversation protocols (request-response,
    debate, consensus).

    Example:
        # Create participants
        participants = [
            ConversationParticipant(
                id="architect",
                role="planner",
                name="Solution Architect",
            ),
            ConversationParticipant(
                id="developer",
                role="executor",
                name="Senior Developer",
            ),
        ]

        # Create node
        node = ConversationalNode(
            id="design_discussion",
            name="Architecture Discussion",
            participants=participants,
            topic="Discuss best approach for implementing caching",
            config=ConversationalNodeConfig(max_turns=15),
        )

        # Add to StateGraph
        graph.add_node("discuss", node.execute)

    Attributes:
        id: Unique node identifier
        name: Human-readable name
        participants: List of conversation participants
        topic: Conversation topic/goal
        config: Node configuration
        output_key: Key for storing result in state
    """

    def __init__(
        self,
        id: str,
        participants: List[ConversationParticipant],
        topic: str,
        name: str = "",
        config: Optional[ConversationalNodeConfig] = None,
        output_key: str = "conversation_result",
        protocol: Optional["ConversationProtocol"] = None,
    ):
        """Initialize ConversationalNode.

        Args:
            id: Unique node identifier
            participants: List of conversation participants
            topic: Conversation topic/goal
            name: Human-readable name (defaults to id)
            config: Node configuration
            output_key: Key for storing result in state
            protocol: Optional conversation protocol
        """
        self.id = id
        self.name = name or id
        self.participants = participants
        self.topic = topic
        self.config = config or ConversationalNodeConfig()
        self.output_key = output_key
        self.protocol = protocol

        # Runtime state
        self._context: Optional[ConversationContext] = None
        self._history: Optional[ConversationHistory] = None
        self._router: Optional[MessageRouter] = None

    async def execute(
        self,
        state: Dict[str, Any],
        orchestrator: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute the conversational node.

        Args:
            state: Current workflow state
            orchestrator: Optional orchestrator for agent execution

        Returns:
            Updated state with conversation result
        """
        start_time = time.time()
        conversation_id = f"conv_{self.id}_{str(uuid4())[:8]}"

        logger.info(
            f"Starting conversation '{self.name}' "
            f"(id={conversation_id}, participants={len(self.participants)})"
        )

        # Initialize context
        self._context = ConversationContext(
            conversation_id=conversation_id,
            topic=self.topic,
            protocol=self.protocol.__class__.__name__ if self.protocol else "default",
            participants={p.id: p for p in self.participants},
            shared_state=dict(state),
        )

        # Initialize history and router
        self._history = ConversationHistory(conversation_id=conversation_id)
        self._router = MessageRouter(strategy=self.config.routing_strategy)

        # Initialize protocol if present
        if self.protocol:
            await self.protocol.initialize(self.participants, self._context)

        # Run conversation loop
        try:
            result = await self._conversation_loop(orchestrator)
        except asyncio.TimeoutError:
            result = ConversationResult(
                conversation_id=conversation_id,
                status=ConversationStatus.TIMED_OUT,
                outcome="Conversation timed out",
                total_turns=self._context.current_turn,
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            logger.exception(f"Conversation failed: {e}")
            result = ConversationResult(
                conversation_id=conversation_id,
                status=ConversationStatus.FAILED,
                outcome=str(e),
                total_turns=self._context.current_turn,
                duration_seconds=time.time() - start_time,
            )

        # Update state
        state[self.output_key] = result.to_dict()
        state[f"{self.output_key}_history"] = (
            await self._history.get_history(self._context) if self._history else []
        )

        logger.info(
            f"Conversation '{self.name}' completed: "
            f"status={result.status.value}, turns={result.total_turns}"
        )

        return state

    async def _conversation_loop(
        self,
        orchestrator: Optional[Any],
    ) -> ConversationResult:
        """Run the main conversation loop.

        Args:
            orchestrator: Orchestrator for agent execution

        Returns:
            Conversation result
        """
        start_time = time.time()

        while self._context.current_turn < self.config.max_turns:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.config.timeout_seconds:
                raise asyncio.TimeoutError("Conversation timeout")

            # Get next speaker
            speaker = await self._get_next_speaker()
            if not speaker:
                logger.debug("No more speakers, ending conversation")
                break

            self._context.current_speaker = speaker
            self._context.current_turn += 1

            logger.debug(f"Turn {self._context.current_turn}: speaker={speaker}")

            # Execute turn
            turn = await self._execute_turn(speaker, orchestrator)

            # Record in history
            if self._history:
                await self._history.add_turn(self._context, turn)

            # Update shared state for next speaker
            self._context.shared_state["last_speaker"] = speaker
            self._context.shared_state["last_message"] = turn.message.content

            # Check if conversation should continue
            should_continue, reason = await self._should_continue()
            if not should_continue:
                self._context.terminate(reason or "Conversation complete")
                break

        # Build result
        return ConversationResult(
            conversation_id=self._context.conversation_id,
            status=(
                ConversationStatus.COMPLETED
                if not self._context.is_terminated
                or "complete" in (self._context.termination_reason or "").lower()
                else ConversationStatus.TERMINATED
            ),
            outcome=self._context.termination_reason or "Conversation completed",
            summary=await self._generate_summary(),
            final_speaker=self._context.current_speaker,
            total_turns=self._context.current_turn,
            duration_seconds=time.time() - start_time,
            metadata={"topic": self.topic},
        )

    async def _get_next_speaker(self) -> Optional[str]:
        """Determine the next speaker.

        Returns:
            Next speaker ID or None if no more speakers
        """
        if self.protocol:
            return await self.protocol.get_next_speaker(self._context)

        # Default: round-robin through participants
        participant_ids = list(self._context.participants.keys())
        if not participant_ids:
            return None

        # Start with first participant, then rotate
        idx = self._context.current_turn % len(participant_ids)
        return participant_ids[idx]

    async def _execute_turn(
        self,
        speaker: str,
        orchestrator: Optional[Any],
    ) -> ConversationalTurn:
        """Execute a single conversation turn.

        Args:
            speaker: Speaker agent ID
            orchestrator: Orchestrator for agent execution

        Returns:
            Completed turn
        """
        participant = self._context.participants.get(speaker)
        if not participant:
            raise ValueError(f"Unknown speaker: {speaker}")

        # Build prompt for speaker
        prompt = self._build_speaker_prompt(participant)

        # Get speaker's response
        if orchestrator and hasattr(orchestrator, "chat"):
            response_text = await orchestrator.chat(prompt)
        else:
            # Fallback for testing
            response_text = f"[{speaker}] Response to turn {self._context.current_turn}"

        # Create message
        message = ConversationalMessage(
            sender=speaker,
            content=response_text,
            message_type=MessageType.CONVERSATION,
            conversation_id=self._context.conversation_id,
            turn_number=self._context.current_turn,
        )

        # Route message and get responses
        responses = []
        if self._router:
            recipients = await self._router.route_message(message, self.participants, self._context)

            # Collect responses (optional, depending on protocol)
            for recipient_id in recipients:
                should_reply = await self._router.should_reply(recipient_id, message, self._context)
                if should_reply and orchestrator:
                    # Get response from recipient
                    recipient = self._context.participants.get(recipient_id)
                    if recipient:
                        reply_prompt = self._build_reply_prompt(recipient, message)
                        if hasattr(orchestrator, "chat"):
                            reply_text = await orchestrator.chat(reply_prompt)
                        else:
                            reply_text = f"[{recipient_id}] Reply to {speaker}"

                        responses.append(
                            ConversationalMessage(
                                sender=recipient_id,
                                content=reply_text,
                                message_type=MessageType.ANSWER,
                                recipients=[speaker],
                                conversation_id=self._context.conversation_id,
                                turn_number=self._context.current_turn,
                                reply_to=message.id,
                            )
                        )

        return ConversationalTurn(
            turn_number=self._context.current_turn,
            speaker=speaker,
            message=message,
            responses=responses,
        )

    async def _should_continue(self) -> tuple[bool, Optional[str]]:
        """Check if conversation should continue.

        Returns:
            Tuple of (should_continue, reason)
        """
        if self.protocol:
            return await self.protocol.should_continue(self._context)

        # Default: continue until max turns
        if self._context.current_turn >= self.config.max_turns:
            return False, "Maximum turns reached"

        # Check for termination signals in shared state
        if self._context.shared_state.get("terminate"):
            return False, self._context.shared_state.get(
                "termination_reason", "Terminated by participant"
            )

        return True, None

    def _build_speaker_prompt(self, participant: ConversationParticipant) -> str:
        """Build prompt for speaker's turn.

        Args:
            participant: Speaking participant

        Returns:
            Prompt string
        """
        # Get recent history context
        history_context = ""
        if self._history and self._history.turns:
            recent = self._history.turns[-3:]  # Last 3 turns
            history_parts = []
            for turn in recent:
                history_parts.append(f"[{turn.speaker}]: {turn.message.content[:200]}...")
                for resp in turn.responses[:2]:
                    history_parts.append(f"  -> [{resp.sender}]: {resp.content[:100]}...")
            history_context = "\n".join(history_parts)

        prompt = f"""You are participating in a multi-agent conversation.

**Your Role:** {participant.role}
**Your Name:** {participant.name}
{f'**Your Persona:** {participant.persona}' if participant.persona else ''}

**Conversation Topic:** {self.topic}
**Current Turn:** {self._context.current_turn}/{self.config.max_turns}

{('**Recent History:**' + chr(10) + history_context) if history_context else ''}

Please provide your contribution to this conversation. Be concise and relevant to the topic.
If you believe the conversation has reached a conclusion, include [CONCLUSION] in your response."""

        return prompt

    def _build_reply_prompt(
        self,
        recipient: ConversationParticipant,
        message: ConversationalMessage,
    ) -> str:
        """Build prompt for reply to a message.

        Args:
            recipient: Recipient participant
            message: Message to reply to

        Returns:
            Reply prompt string
        """
        return f"""You are participating in a multi-agent conversation.

**Your Role:** {recipient.role}
**Your Name:** {recipient.name}

**Message from {message.sender}:**
{message.content}

Please provide a brief response to this message. Focus on:
- Addressing the key points raised
- Contributing your perspective based on your role
- Being concise (1-2 paragraphs max)"""

    async def _generate_summary(self) -> Optional[str]:
        """Generate a summary of the conversation.

        Returns:
            Summary string or None
        """
        if not self._history or not self._history.turns:
            return None

        # Simple summary from last turn's content
        last_turn = self._history.turns[-1]
        return (
            f"Conversation on '{self.topic}' completed after "
            f"{len(self._history.turns)} turns. "
            f"Final speaker: {last_turn.speaker}."
        )


__all__ = [
    "ConversationalNode",
    "ConversationalNodeConfig",
]
