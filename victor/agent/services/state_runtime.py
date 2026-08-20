# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Canonical live conversation-state runtime.

This module provides the internal runtime adapter that fulfills
``StateRuntimeProtocol`` without routing through the deprecated
``StateCoordinator`` compatibility shim.

Live conversation stage state belongs to ``ConversationController`` and
``ConversationStateMachine``. This adapter exposes that state through the
runtime protocol expected by DI and service wiring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from victor.agent.conversation.state_machine import ConversationStage

if TYPE_CHECKING:
    from victor.agent.conversation.controller import ConversationController
    from victor.agent.conversation.state_machine import ConversationStateMachine
    from victor.providers.base import Message


class StateRuntimeAdapter:
    """Canonical adapter for live conversation-stage state."""

    def __init__(
        self,
        conversation_controller: "ConversationController",
        state_machine: Optional["ConversationStateMachine"] = None,
    ) -> None:
        self._controller = conversation_controller
        self._state_machine = state_machine or getattr(
            conversation_controller, "_state_machine", None
        )

    def get_current_stage(self) -> ConversationStage:
        """Get the current conversation stage."""
        if self._state_machine is not None and hasattr(self._state_machine, "get_stage"):
            return self._state_machine.get_stage()
        return self._controller.stage

    def transition_to(
        self,
        stage: ConversationStage,
        reason: str = "",
        tool_name: Optional[str] = None,
    ) -> bool:
        """Transition to a new conversation stage.

        ``reason`` and ``tool_name`` are accepted for protocol compatibility.
        The canonical runtime delegates to the real state machine when present.
        """
        del reason, tool_name

        current = self.get_current_stage()
        if current == stage:
            return True

        machine = self._state_machine or getattr(self._controller, "_state_machine", None)
        if machine is None:
            return False

        if hasattr(machine, "_transition_to"):
            machine._transition_to(stage, confidence=1.0)
            return self.get_current_stage() == stage

        if hasattr(machine, "set_stage"):
            machine.set_stage(stage)
            return self.get_current_stage() == stage

        state = getattr(machine, "state", None)
        if state is not None and hasattr(state, "stage"):
            state.stage = stage
            return self.get_current_stage() == stage

        return False

    def get_message_history(self) -> List["Message"]:
        """Get the full conversation message history."""
        return self._controller.messages

    def get_recent_messages(
        self,
        limit: int = 10,
        include_system: bool = False,
    ) -> List["Message"]:
        """Get recent messages from history."""
        messages = self._controller.messages
        if not include_system:
            messages = [m for m in messages if getattr(m, "role", None) != "system"]
        return messages[-limit:] if limit < len(messages) else messages

    def is_in_exploration_phase(self) -> bool:
        """Check if the current stage is exploratory."""
        stage = self.get_current_stage()
        return stage in {
            ConversationStage.INITIAL,
            ConversationStage.PLANNING,
            ConversationStage.READING,
            ConversationStage.ANALYSIS,
        }

    def is_in_execution_phase(self) -> bool:
        """Check if the current stage is execution."""
        return self.get_current_stage() == ConversationStage.EXECUTION
