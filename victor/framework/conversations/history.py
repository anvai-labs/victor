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

"""Conversation history tracking and export.

This module provides history management for conversations,
implementing the ConversationHistoryProtocol.
"""

from __future__ import annotations

from victor.core.json_utils import json_dumps
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from victor.framework.conversations.types import (
    ConversationalMessage,
    ConversationalTurn,
    ConversationContext,
    ConversationParticipant,
)

logger = logging.getLogger(__name__)


@dataclass
class ConversationHistory:
    """In-memory conversation history storage.

    Implements ConversationHistoryProtocol for tracking conversation turns.

    Attributes:
        conversation_id: The conversation being tracked
        turns: List of all conversation turns
        messages_by_sender: Messages indexed by sender ID
        created_at: When history was created
    """

    conversation_id: str
    turns: List[ConversationalTurn] = field(default_factory=list)
    messages_by_sender: Dict[str, List[ConversationalMessage]] = field(
        default_factory=lambda: defaultdict(list)
    )
    created_at: datetime = field(default_factory=datetime.now)

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
        self.turns.append(turn)

        # Index by sender
        self.messages_by_sender[turn.speaker].append(turn.message)
        for response in turn.responses:
            self.messages_by_sender[response.sender].append(response)

        logger.debug(
            f"Added turn {turn.turn_number} to history " f"(total turns: {len(self.turns)})"
        )

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
        return list(self.turns)

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
        return self.turns[-n:] if n > 0 else []

    async def clear_history(
        self,
        context: ConversationContext,
    ) -> None:
        """Clear conversation history.

        Args:
            context: Conversation context
        """
        self.turns.clear()
        self.messages_by_sender.clear()
        logger.debug(f"Cleared history for conversation {self.conversation_id}")

    def get_messages_by_sender(self, sender_id: str) -> List[ConversationalMessage]:
        """Get all messages from a specific sender.

        Args:
            sender_id: The sender to filter by

        Returns:
            List of messages from sender
        """
        return list(self.messages_by_sender.get(sender_id, []))

    def get_turn_count(self) -> int:
        """Get total number of turns."""
        return len(self.turns)

    def get_message_count(self) -> int:
        """Get total number of messages (including responses)."""
        total = 0
        for turn in self.turns:
            total += 1  # Speaker's message
            total += len(turn.responses)
        return total

    def get_participation_stats(self) -> Dict[str, int]:
        """Get message count per participant.

        Returns:
            Dictionary of participant_id -> message_count
        """
        return {sender: len(msgs) for sender, msgs in self.messages_by_sender.items()}


class ConversationExporter:
    """Export conversation history to various formats."""

    @staticmethod
    def to_json(history: ConversationHistory) -> str:
        """Export history to JSON.

        Args:
            history: Conversation history to export

        Returns:
            JSON string
        """
        data = {
            "conversation_id": history.conversation_id,
            "created_at": history.created_at.isoformat(),
            "turn_count": history.get_turn_count(),
            "message_count": history.get_message_count(),
            "participation_stats": history.get_participation_stats(),
            "turns": [
                {
                    "turn_number": turn.turn_number,
                    "speaker": turn.speaker,
                    "message": {
                        "sender": turn.message.sender,
                        "content": turn.message.content,
                        "type": turn.message.message_type.value,
                        "timestamp": (
                            turn.message.timestamp.isoformat() if turn.message.timestamp else None
                        ),
                    },
                    "responses": [
                        {
                            "sender": resp.sender,
                            "content": resp.content,
                            "type": resp.message_type.value,
                        }
                        for resp in turn.responses
                    ],
                }
                for turn in history.turns
            ],
        }
        return json_dumps(data, indent=2)

    @staticmethod
    def to_markdown(
        history: ConversationHistory,
        participants: Optional[Dict[str, ConversationParticipant]] = None,
    ) -> str:
        """Export history to Markdown.

        Args:
            history: Conversation history to export
            participants: Optional participant info for names

        Returns:
            Markdown string
        """
        lines = [
            f"# Conversation: {history.conversation_id}",
            "",
            f"**Created:** {history.created_at.isoformat()}",
            f"**Turns:** {history.get_turn_count()}",
            f"**Messages:** {history.get_message_count()}",
            "",
            "## Participation",
            "",
        ]

        for sender, count in history.get_participation_stats().items():
            name = sender
            if participants and sender in participants:
                name = participants[sender].name or sender
            lines.append(f"- **{name}**: {count} messages")

        lines.extend(["", "## Conversation", ""])

        for turn in history.turns:
            name = turn.speaker
            if participants and turn.speaker in participants:
                name = participants[turn.speaker].name or turn.speaker

            lines.append(f"### Turn {turn.turn_number}: {name}")
            lines.append("")
            lines.append(f"> {turn.message.content}")
            lines.append("")

            if turn.responses:
                lines.append("**Responses:**")
                for resp in turn.responses:
                    resp_name = resp.sender
                    if participants and resp.sender in participants:
                        resp_name = participants[resp.sender].name or resp.sender
                    lines.append(f"- **{resp_name}**: {resp.content[:100]}...")
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def to_mermaid(history: ConversationHistory) -> str:
        """Export conversation flow to Mermaid sequence diagram.

        Args:
            history: Conversation history to export

        Returns:
            Mermaid diagram string
        """
        lines = ["sequenceDiagram"]

        # Collect all participants
        participants = set()
        for turn in history.turns:
            participants.add(turn.speaker)
            for resp in turn.responses:
                participants.add(resp.sender)

        # Declare participants
        for p in sorted(participants):
            lines.append(f"    participant {p}")

        lines.append("")

        # Add messages
        for turn in history.turns:
            # Speaker's message (broadcast if no specific recipients)
            for resp in turn.responses:
                arrow = "->>" if turn.message.requires_response else "->"
                content = turn.message.content[:30].replace('"', "'")
                lines.append(f"    {turn.speaker}{arrow}{resp.sender}: {content}...")

                # Response
                resp_content = resp.content[:30].replace('"', "'")
                lines.append(f"    {resp.sender}-->{turn.speaker}: {resp_content}...")

        return "\n".join(lines)


__all__ = [
    "ConversationHistory",
    "ConversationExporter",
]
