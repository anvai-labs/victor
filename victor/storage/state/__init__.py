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

"""Victor State - Standalone state machine package.

This package exposes the conversation state machine as a standalone,
reusable component that can be used independently of the full agent.

Key Components:
- StateMachine: Generic state machine with configurable stages
- ConversationStateMachine: Pre-configured for conversation flows
- Stage: Stage definitions with tools and keywords
- StateProtocol: Protocol for state machine implementations

Design Patterns:
- State Pattern: Encapsulates state-specific behavior
- Observer Pattern: Hooks for state transitions
- Strategy Pattern: Pluggable stage detection

Example:
    from victor.storage.state import ConversationStateMachine, ConversationStage

    # Create state machine
    machine = ConversationStateMachine()

    # Record tool usage
    machine.record_tool_execution("read", {"path": "/tmp/file.txt"})

    # Check current stage
    print(f"Current stage: {machine.get_stage()}")

    # Get recommended tools for current stage
    tools = machine.get_stage_tools()

Standalone Usage (without full agent):
    from victor.storage.state import StateMachine, StateConfig

    # Define custom stages
    config = StateConfig(
        stages=["START", "PROCESSING", "COMPLETE"],
        initial_stage="START",
        transitions={
            "START": ["PROCESSING"],
            "PROCESSING": ["COMPLETE", "START"],
            "COMPLETE": [],
        },
    )

    machine = StateMachine(config)
    machine.transition_to("PROCESSING")
"""

from __future__ import annotations

# Re-export from agent.conversation_state for backward compatibility
from victor.agent.conversation.state_machine import (
    ConversationStage,
    ConversationState,
    ConversationStateMachine,
    STAGE_KEYWORDS,
)

# New standalone components
from victor.storage.state.machine import (
    StateMachine,
    StateConfig,
    StateTransition,
)
from victor.storage.state.protocols import (
    StateProtocol,
    StageProtocol,
    TransitionValidatorProtocol,
)

__all__ = [
    # Conversation-specific (re-exports)
    "ConversationStage",
    "ConversationState",
    "ConversationStateMachine",
    "STAGE_KEYWORDS",
    # Generic state machine
    "StateMachine",
    "StateConfig",
    "StateTransition",
    # Protocols
    "StateProtocol",
    "StageProtocol",
    "TransitionValidatorProtocol",
]

__version__ = "0.5.1"
