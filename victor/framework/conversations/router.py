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

"""Message routing for conversational agents.

This module provides intelligent message routing between conversation
participants based on roles, capabilities, and context.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import List, Optional

from victor.framework.conversations.types import (
    ConversationalMessage,
    ConversationContext,
    ConversationParticipant,
)

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategy for message delivery."""

    BROADCAST = "broadcast"  # Send to all participants
    ROLE_BASED = "role_based"  # Route based on recipient roles
    CAPABILITY_BASED = "capability_based"  # Route based on capabilities
    ROUND_ROBIN = "round_robin"  # Round-robin across participants
    DIRECTED = "directed"  # Explicitly directed (use message.recipients)
    REPLY_TO_SENDER = "reply_to_sender"  # Only reply to original sender


class MessageRouter:
    """Intelligent message routing between conversation participants.

    Implements ConversationRoutingProtocol with multiple routing strategies.

    Attributes:
        strategy: Default routing strategy
        exclude_sender: Whether to exclude sender from recipients
    """

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.BROADCAST,
        exclude_sender: bool = True,
    ):
        """Initialize message router.

        Args:
            strategy: Default routing strategy
            exclude_sender: Whether to exclude sender from recipients
        """
        self.strategy = strategy
        self.exclude_sender = exclude_sender
        self._round_robin_index = 0

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
        # Use explicit recipients if specified
        if message.recipients:
            return message.recipients

        # Use routing strategy
        return await self.get_recipients(message.sender, participants, context)

    async def get_recipients(
        self,
        speaker: str,
        participants: List[ConversationParticipant],
        context: ConversationContext,
        target_role: Optional[str] = None,
        target_capability: Optional[str] = None,
    ) -> List[str]:
        """Get recipients for speaker's message.

        Args:
            speaker: Speaking agent ID
            participants: All conversation participants
            context: Conversation context
            target_role: Optional target role for ROLE_BASED routing
            target_capability: Optional target capability for CAPABILITY_BASED routing

        Returns:
            List of recipient agent IDs
        """
        if self.strategy == RoutingStrategy.BROADCAST:
            return self._broadcast_route(speaker, participants)

        elif self.strategy == RoutingStrategy.ROLE_BASED:
            return self._role_based_route(speaker, participants, target_role)

        elif self.strategy == RoutingStrategy.CAPABILITY_BASED:
            return self._capability_based_route(speaker, participants, target_capability)

        elif self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_route(speaker, participants)

        elif self.strategy == RoutingStrategy.DIRECTED:
            # For DIRECTED, return empty - caller must specify recipients
            return []

        elif self.strategy == RoutingStrategy.REPLY_TO_SENDER:
            # Reply to whoever last spoke (from shared state)
            last_speaker = context.shared_state.get("last_speaker")
            if last_speaker and last_speaker != speaker:
                return [last_speaker]
            return []

        return self._broadcast_route(speaker, participants)

    def _broadcast_route(
        self, speaker: str, participants: List[ConversationParticipant]
    ) -> List[str]:
        """Broadcast routing - send to all except sender.

        Args:
            speaker: Speaker ID
            participants: All participants

        Returns:
            All participant IDs except sender (if exclude_sender=True)
        """
        recipients = [p.id for p in participants]
        if self.exclude_sender and speaker in recipients:
            recipients.remove(speaker)
        return recipients

    def _role_based_route(
        self,
        speaker: str,
        participants: List[ConversationParticipant],
        target_role: Optional[str],
    ) -> List[str]:
        """Role-based routing - send to specific role(s).

        Args:
            speaker: Speaker ID
            participants: All participants
            target_role: Target role (or None for complementary roles)

        Returns:
            Participant IDs matching target role
        """
        # Get speaker's role
        speaker_participant = next((p for p in participants if p.id == speaker), None)
        if not speaker_participant:
            return self._broadcast_route(speaker, participants)

        if target_role:
            # Route to specific role
            recipients = [p.id for p in participants if p.role == target_role]
        else:
            # Route to complementary roles
            # e.g., expert -> critic, planner -> executor
            role_pairs = {
                "expert": ["critic", "reviewer"],
                "critic": ["expert", "defender"],
                "planner": ["executor"],
                "executor": ["reviewer", "planner"],
                "moderator": [],  # Moderator speaks to all
                "researcher": ["analyst", "synthesizer"],
            }
            target_roles = role_pairs.get(speaker_participant.role, [])
            if not target_roles:
                return self._broadcast_route(speaker, participants)
            recipients = [p.id for p in participants if p.role in target_roles]

        # Exclude sender
        if self.exclude_sender and speaker in recipients:
            recipients.remove(speaker)

        return recipients if recipients else self._broadcast_route(speaker, participants)

    def _capability_based_route(
        self,
        speaker: str,
        participants: List[ConversationParticipant],
        target_capability: Optional[str],
    ) -> List[str]:
        """Capability-based routing - send to agents with specific capability.

        Args:
            speaker: Speaker ID
            participants: All participants
            target_capability: Target capability to match

        Returns:
            Participant IDs with matching capability
        """
        if not target_capability:
            return self._broadcast_route(speaker, participants)

        recipients = [p.id for p in participants if target_capability in p.capabilities]

        if self.exclude_sender and speaker in recipients:
            recipients.remove(speaker)

        return recipients if recipients else self._broadcast_route(speaker, participants)

    def _round_robin_route(
        self, speaker: str, participants: List[ConversationParticipant]
    ) -> List[str]:
        """Round-robin routing - rotate through participants.

        Args:
            speaker: Speaker ID
            participants: All participants

        Returns:
            Single recipient ID (next in rotation)
        """
        # Build list excluding sender
        candidates = [p.id for p in participants if p.id != speaker]
        if not candidates:
            return []

        # Get next recipient
        recipient = candidates[self._round_robin_index % len(candidates)]
        self._round_robin_index += 1

        return [recipient]

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
        # Don't reply to own messages
        if message.sender == agent:
            return False

        # If message has explicit recipients, check if agent is included
        if message.recipients:
            return agent in message.recipients

        # If message doesn't require response, don't reply
        if not message.requires_response:
            return False

        # Otherwise, agent should reply (broadcast)
        return True


__all__ = [
    "MessageRouter",
    "RoutingStrategy",
]
