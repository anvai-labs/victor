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

"""Conversational Agent Coordination Framework.

This module provides dynamic conversational capabilities for multi-agent workflows,
enabling agents to engage in multi-turn conversations, debate, and reach consensus.

Key Components:
    - ConversationalNode: Workflow node for hosting conversations
    - ConversationProtocol: Base protocol for conversation patterns
    - MessageRouter: Intelligent message routing
    - ConversationHistory: Full tracking of interactions

Conversation Protocols:
    - RequestResponseProtocol: Simple Q&A pattern
    - DebateProtocol: Arguments for/against positions
    - ConsensusProtocol: Building agreement through voting

Example:
    from victor.framework.conversations import (
        ConversationalNode,
        ConsensusProtocol,
        ConversationParticipant,
    )

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

    # Create consensus node
    node = ConversationalNode(
        id="design_discussion",
        participants=participants,
        topic="Discuss best approach for caching",
        config=ConversationalNodeConfig(max_turns=10),
    )

    # Execute in workflow
    result = await node.execute(state)
"""

# Types
from victor.framework.conversations.types import (
    MessageType,
    ConversationStatus,
    ConversationalMessage,
    ConversationParticipant,
    ConversationContext,
    ConversationResult,
    ConversationalTurn,
)

# Protocols
from victor.framework.conversations.protocols import (
    ConversationProtocol,
    MessageFormatterProtocol,
    ConversationResultProtocol,
    ConversationHistoryProtocol,
    ConversationRoutingProtocol,
    BaseConversationProtocol,
    RequestResponseProtocol,
    DebateProtocol,
    ConsensusProtocol,
)

# Router
from victor.framework.conversations.router import (
    MessageRouter,
    RoutingStrategy,
)

# History
from victor.framework.conversations.history import (
    ConversationHistory,
    ConversationExporter,
)

# Node
from victor.framework.conversations.node import (
    ConversationalNode,
    ConversationalNodeConfig,
)

__all__ = [
    # Types
    "MessageType",
    "ConversationStatus",
    "ConversationalMessage",
    "ConversationParticipant",
    "ConversationContext",
    "ConversationResult",
    "ConversationalTurn",
    # Protocols
    "ConversationProtocol",
    "MessageFormatterProtocol",
    "ConversationResultProtocol",
    "ConversationHistoryProtocol",
    "ConversationRoutingProtocol",
    "BaseConversationProtocol",
    "RequestResponseProtocol",
    "DebateProtocol",
    "ConsensusProtocol",
    # Router
    "MessageRouter",
    "RoutingStrategy",
    # History
    "ConversationHistory",
    "ConversationExporter",
    # Node
    "ConversationalNode",
    "ConversationalNodeConfig",
]
