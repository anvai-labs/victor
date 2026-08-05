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

"""Base conversation manager for Victor verticals.

This module provides BaseConversationManager, a reusable base class that
implements enhanced conversation management using the framework's
ConversationCoordinator. Verticals can inherit from this base class to
get common conversation functionality while adding vertical-specific features.

Design Pattern: Template Method
- Base class provides common conversation management infrastructure
- Verticals override get_vertical_name() and get_system_prompt()
- Verticals can optionally override message processing and context building

Usage:
    from victor.contrib.conversation import BaseConversationManager
    from victor_contracts.conversation import TurnType

    class MyVerticalConversationManager(BaseConversationManager):
        def get_vertical_name(self) -> str:
            return \"myvertical\"

        def get_system_prompt(self) -> str:
            return \"You are a specialized assistant for myvertical.\"

        def get_vertical_rules(self) -> List[SafetyRule]:
            return [
                SafetyRule(
                    rule_id=\"myvertical_rule\",
                    pattern=r\"dangerous-operation\",
                    description=\"Dangerous operation\",
                    action=SafetyAction.WARN,
                    severity=5,
                )
            ]
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from victor_contracts.conversation import (
    ConversationCoordinator,
    TurnType,
)
from victor.contrib.conversation.vertical_context import VerticalConversationContext
from victor.contrib.safety.base_extension import BaseSafetyExtension
from victor.contrib.safety.vertical_mixin import VerticalSafetyMixin

logger = logging.getLogger(__name__)


class BaseConversationManager:
    """Base conversation manager for Victor verticals.

    Provides common conversation functionality using ConversationCoordinator:
    - Message history management with deduplication
    - Turn tracking and statistics
    - Context window management and summarization
    - Vertical-specific context tracking
    - Integration with safety extensions

    Verticals should:
    1. Inherit from BaseConversationManager
    2. Implement get_vertical_name() to return vertical identifier
    3. Implement get_system_prompt() to return system prompt
    4. Optionally override get_vertical_rules() for safety rules
    5. Optionally override message processing methods
    """

    def __init__(
        self,
        max_history_turns: int = 50,
        summarization_threshold: int = 40,
        context_window_size: int = 128000,
        enable_safety: bool = True,
        safety_strict_mode: bool = False,
    ):
        """Initialize the conversation manager.

        Args:
            max_history_turns: Maximum turns to keep in history
            summarization_threshold: Turns before triggering summarization
            context_window_size: Size of context window in tokens
            enable_safety: Whether to enable safety checking
            safety_strict_mode: If True, safety warnings become blocks
        """
        # Initialize conversation coordinator
        self._coordinator = ConversationCoordinator(
            max_history_turns=max_history_turns,
            summarization_threshold=summarization_threshold,
            context_window_size=context_window_size,
            enable_deduplication=True,
            enable_statistics=True,
        )

        # Initialize vertical context
        self._context = VerticalConversationContext(
            vertical_name=self.get_vertical_name(),
            domain=self._infer_domain(),
        )

        # Initialize safety extension if enabled
        self._safety_extension: Optional[BaseSafetyExtension] = None
        if enable_safety:
            if self.get_vertical_rules():
                # Create a safety extension for this vertical
                safety_class = type(
                    f"{self.get_vertical_name().title()}SafetyExtension",
                    (BaseSafetyExtension, VerticalSafetyMixin),
                    {
                        "get_vertical_name": lambda self: self.get_vertical_name(),
                        "get_vertical_rules": lambda self: self.get_vertical_rules(),
                    },
                )
                self._safety_extension = safety_class(
                    strict_mode=safety_strict_mode,
                    enable_custom_rules=True,
                )

        logger.info(
            f"{self.__class__.__name__} initialized for '{self.get_vertical_name()}' "
            f"with {max_history_turns} max turns, safety={enable_safety}"
        )

    # ==========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # ==========================================================================

    @abstractmethod
    def get_vertical_name(self) -> str:
        """Get the vertical name for context tracking.

        Returns:
            Vertical name (e.g., "devops", "rag", "research")
        """
        ...

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this vertical.

        Returns:
            System prompt string
        """
        ...

    # ==========================================================================
    # Template Methods - Can be overridden by subclasses
    # ==========================================================================

    def get_vertical_rules(self) -> List:
        """Get vertical-specific safety rules.

        Returns:
            List of SafetyRule instances for this vertical
        """
        return []

    def pre_process_message(
        self,
        role: str,
        content: str,
        turn_type: TurnType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, str, TurnType, Optional[Dict[str, Any]]]:
        """Pre-process a message before adding to conversation.

        Args:
            role: Message role
            content: Message content
            turn_type: Type of turn
            metadata: Optional metadata

        Returns:
            Tuple of (role, content, turn_type, metadata) potentially modified
        """
        return role, content, turn_type, metadata

    def post_process_message(
        self,
        turn_id: str,
        role: str,
        content: str,
    ) -> None:
        """Post-process a message after adding to conversation.

        Args:
            turn_id: ID of the turn that was added
            role: Message role
            content: Message content
        """
        pass

    def build_context_for_prompt(self) -> str:
        """Build vertical-specific context for inclusion in prompts.

        Returns:
            Context string to include in prompts
        """
        parts = [f"# {self.get_vertical_name().title()} Context"]

        # Add active tasks
        active_tasks = self._context.get_active_tasks()
        if active_tasks:
            parts.append("\n## Active Tasks")
            for task in active_tasks[:5]:
                parts.append(f"- {task.task_id}: {task.status}")

        # Add domain knowledge
        if self._context.knowledge:
            parts.append("\n## Relevant Knowledge")
            for k in self._context.knowledge[-5:]:
                parts.append(f"- {k.topic}")
                for fact in k.facts[:3]:
                    parts.append(f"  - {fact}")

        return "\n".join(parts) if len(parts) > 1 else ""

    # ==========================================================================
    # Message Management - Common for all verticals
    # ==========================================================================

    def add_message(
        self,
        role: str,
        content: str,
        turn_type: TurnType,
        metadata: Optional[Dict[str, Any]] = None,
        check_safety: bool = True,
    ) -> str:
        """Add a message to the conversation.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            turn_type: Type of turn
            metadata: Optional metadata
            check_safety: Whether to check safety rules

        Returns:
            Turn ID for the added message
        """
        # Pre-process message
        role, content, turn_type, metadata = self.pre_process_message(
            role, content, turn_type, metadata
        )

        # Check safety if enabled
        if check_safety and self._safety_extension and role == "user":
            is_safe = self._safety_extension.is_operation_safe(
                tool_name="conversation",
                args=[content[:100]],  # Check first 100 chars
                context={"role": role},
            )
            if not is_safe:
                logger.warning(f"Message blocked by safety rules for '{self.get_vertical_name()}'")
                # Still add but with blocked metadata
                if metadata is None:
                    metadata = {}
                metadata["safety_blocked"] = True

        # Add to coordinator
        turn_id = self._coordinator.add_message(
            role=role,
            content=content,
            turn_type=turn_type,
            metadata=metadata,
        )

        # Post-process message
        self.post_process_message(turn_id, role, content)

        return turn_id

    def get_history(
        self,
        max_turns: Optional[int] = None,
        include_system: bool = True,
        include_tool: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get conversation history.

        Args:
            max_turns: Maximum number of turns to return
            include_system: Whether to include system messages
            include_tool: Whether to include tool messages

        Returns:
            List of message dictionaries
        """
        return self._coordinator.get_history(
            max_turns=max_turns,
            include_system=include_system,
            include_tool=include_tool,
        )

    def get_conversation_for_prompt(
        self,
        max_turns: Optional[int] = None,
        include_vertical_context: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get conversation formatted for LLM prompt.

        Args:
            max_turns: Maximum number of turns to include
            include_vertical_context: Whether to include vertical-specific context

        Returns:
            List of message dictionaries ready for prompting
        """
        messages = []

        # Add system prompt
        if include_vertical_context:
            context = self.build_context_for_prompt()
            if context:
                system_content = f"{self.get_system_prompt()}\n\n{context}"
            else:
                system_content = self.get_system_prompt()
        else:
            system_content = self.get_system_prompt()

        messages.append({"role": "system", "content": system_content})

        # Add conversation history
        messages.extend(self.get_history(max_turns=max_turns))

        return messages

    # ==========================================================================
    # Context and Statistics
    # ==========================================================================

    def get_vertical_context(self) -> VerticalConversationContext:
        """Get the vertical conversation context.

        Returns:
            VerticalConversationContext instance
        """
        return self._context

    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics.

        Returns:
            Dictionary with conversation and vertical statistics
        """
        stats = self._coordinator.get_stats_dict()
        stats["vertical"] = self._context.to_dict()
        return stats

    def get_coordinator(self) -> ConversationCoordinator:
        """Get the underlying ConversationCoordinator instance.

        Returns:
            ConversationCoordinator instance
        """
        return self._coordinator

    def get_safety_extension(self) -> Optional[BaseSafetyExtension]:
        """Get the safety extension if enabled.

        Returns:
            SafetyExtension instance or None
        """
        return self._safety_extension

    # ==========================================================================
    # Conversation Management
    # ==========================================================================

    def clear_history(self, keep_summaries: bool = True) -> None:
        """Clear conversation history.

        Args:
            keep_summaries: Whether to keep conversation summaries
        """
        self._coordinator.clear_history(keep_summaries=keep_summaries)
        logger.info(f"Cleared conversation history for '{self.get_vertical_name()}'")

    def needs_summarization(self) -> bool:
        """Check if conversation needs summarization.

        Returns:
            True if summarization is recommended
        """
        return self._coordinator.needs_summarization()

    def add_summary(self, summary: str) -> None:
        """Add a conversation summary.

        Args:
            summary: Summary text
        """
        self._coordinator.add_summary(summary)

    def get_summaries(self) -> List[str]:
        """Get all conversation summaries.

        Returns:
            List of summary strings
        """
        return self._coordinator.get_summaries()

    # ==========================================================================
    # Private Helper Methods
    # ==========================================================================

    def _infer_domain(self) -> str:
        """Infer domain from vertical name.

        Returns:
            Inferred domain name
        """
        vertical_name = self.get_vertical_name().lower()

        # Map vertical names to domains
        domain_map = {
            "coding": "software_development",
            "devops": "devops_and_infrastructure",
            "rag": "retrieval_augmented_generation",
            "research": "research_and_analysis",
            "dataanalysis": "data_science",
            "security": "cybersecurity",
            "iac": "infrastructure_as_code",
            "classification": "machine_learning",
            "benchmark": "benchmarking_and_testing",
        }

        return domain_map.get(vertical_name, vertical_name)


__all__ = [
    "BaseConversationManager",
]
