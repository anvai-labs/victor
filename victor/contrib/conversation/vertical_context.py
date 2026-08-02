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

"""Vertical conversation context for tracking domain-specific information.

This module provides VerticalConversationContext, a dataclass for tracking
vertical-specific conversation context beyond the framework's generic
conversation tracking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TaskContext:
    """Context about a task being discussed.

    Attributes:
        task_id: Unique identifier for the task
        task_type: Type of task (coding, devops, research, etc.)
        status: Current task status
        metadata: Additional task metadata
        created_at: When the task was created
        updated_at: When the task was last updated
    """

    task_id: str
    task_type: str
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "metadata": self.metadata.copy(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class DomainKnowledge:
    """Domain-specific knowledge extracted from conversation.

    Attributes:
        topic: Knowledge topic
        facts: List of facts about the topic
        confidence: Confidence level (0.0 to 1.0)
        source_messages: Message IDs where this knowledge came from
        extracted_at: When this knowledge was extracted
    """

    topic: str
    facts: List[str]
    confidence: float = 0.8
    source_messages: List[str] = field(default_factory=list)
    extracted_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "topic": self.topic,
            "facts": self.facts.copy(),
            "confidence": self.confidence,
            "source_messages": self.source_messages.copy(),
            "extracted_at": self.extracted_at.isoformat(),
        }


@dataclass
class VerticalConversationContext:
    """Context for vertical-specific conversation information.

    Tracks domain-specific information for a vertical, including:
    - Vertical name and domain
    - Active tasks
    - Domain knowledge extracted
    - Conversation patterns

    Attributes:
        vertical_name: Name of the vertical
        domain: Domain this vertical operates in (coding, devops, research, etc.)
        tasks: Active tasks being discussed
        knowledge: Domain knowledge extracted from conversation
        metadata: Additional metadata
    """

    vertical_name: str
    domain: str
    tasks: Dict[str, TaskContext] = field(default_factory=dict)
    knowledge: List[DomainKnowledge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_task(self, task: TaskContext) -> None:
        """Add a task to the context.

        Args:
            task: Task to add
        """
        self.tasks[task.task_id] = task
        logger.debug(f"Added task '{task.task_id}' to '{self.vertical_name}' context")

    def get_task(self, task_id: str) -> Optional[TaskContext]:
        """Get a task by ID.

        Args:
            task_id: Task ID

        Returns:
            Task context or None if not found
        """
        return self.tasks.get(task_id)

    def update_task_status(self, task_id: str, status: str) -> bool:
        """Update task status.

        Args:
            task_id: Task ID
            status: New status

        Returns:
            True if task was updated, False if not found
        """
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            self.tasks[task_id].updated_at = datetime.now()
            logger.debug(
                f"Updated task '{task_id}' status to '{status}' " f"for '{self.vertical_name}'"
            )
            return True
        return False

    def add_knowledge(self, knowledge: DomainKnowledge) -> None:
        """Add domain knowledge to the context.

        Args:
            knowledge: Knowledge to add
        """
        self.knowledge.append(knowledge)
        logger.debug(
            f"Added knowledge about '{knowledge.topic}' to " f"'{self.vertical_name}' context"
        )

    def get_knowledge_by_topic(self, topic: str) -> List[DomainKnowledge]:
        """Get knowledge about a specific topic.

        Args:
            topic: Topic to search for

        Returns:
            List of knowledge entries about the topic
        """
        return [k for k in self.knowledge if topic.lower() in k.topic.lower()]

    def get_active_tasks(self) -> List[TaskContext]:
        """Get all active (non-completed) tasks.

        Returns:
            List of active tasks
        """
        return [
            task
            for task in self.tasks.values()
            if task.status not in ["completed", "cancelled", "failed"]
        ]

    def clear_completed_tasks(self) -> int:
        """Remove completed tasks from context.

        Returns:
            Number of tasks removed
        """
        completed_ids = [
            task_id
            for task_id, task in self.tasks.items()
            if task.status in ["completed", "cancelled", "failed"]
        ]

        for task_id in completed_ids:
            del self.tasks[task_id]

        logger.debug(
            f"Cleared {len(completed_ids)} completed tasks from " f"'{self.vertical_name}' context"
        )
        return len(completed_ids)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary.

        Returns:
            Dictionary representation of the context
        """
        return {
            "vertical_name": self.vertical_name,
            "domain": self.domain,
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
            "knowledge": [k.to_dict() for k in self.knowledge],
            "active_task_count": len(self.get_active_tasks()),
            "metadata": self.metadata.copy(),
        }

    def get_summary(self) -> str:
        """Get a human-readable summary of the context.

        Returns:
            Summary string
        """
        parts = [
            f"Vertical: {self.vertical_name} ({self.domain})",
            f"Active tasks: {len(self.get_active_tasks())}",
            f"Knowledge entries: {len(self.knowledge)}",
        ]

        if self.get_active_tasks():
            parts.append("\nActive tasks:")
            for task in self.get_active_tasks()[:5]:
                parts.append(f"  - {task.task_id}: {task.status}")

        if self.knowledge:
            parts.append("\nRecent knowledge topics:")
            for k in self.knowledge[-5:]:
                parts.append(f"  - {k.topic} ({len(k.facts)} facts)")

        return "\n".join(parts)


__all__ = [
    "VerticalConversationContext",
    "TaskContext",
    "DomainKnowledge",
]
