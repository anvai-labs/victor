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

"""Session-scoped task management for agent workflows.

Provides structured task tracking with status progression
(pending -> in_progress -> completed) for multi-step agent work.
"""

from __future__ import annotations

from victor.core.json_utils import json_dumps, json_loads
from json import JSONDecodeError
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Lifecycle status of a task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class Task:
    """A single tracked task.

    Args:
        id: Unique task identifier (incrementing integer as string).
        subject: Short summary of the task.
        description: Detailed description of the task.
        status: Current lifecycle status.
        created_at: Unix timestamp when the task was created.
        updated_at: Unix timestamp of the last update.
        metadata: Arbitrary key-value metadata attached to the task.
    """

    id: str
    subject: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the task to a JSON-compatible dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Task:
        """Deserialize a task from a dictionary.

        Args:
            data: Dictionary previously produced by :meth:`to_dict`.

        Returns:
            A reconstructed :class:`Task` instance.
        """
        data = dict(data)  # shallow copy to avoid mutating input
        data["status"] = TaskStatus(data["status"])
        return cls(**data)


class TaskStore:
    """In-memory task store with optional file persistence.

    Tasks are identified by incrementing integer IDs (stored as strings).
    When *persist_path* is provided, the store auto-saves to disk on every
    mutation and loads existing data on construction.

    Args:
        persist_path: Optional filesystem path for JSON persistence.
    """

    def __init__(self, persist_path: Optional[Path] = None) -> None:
        self._tasks: Dict[str, Task] = {}
        self._next_id: int = 1
        self._persist_path: Optional[Path] = persist_path

        if self._persist_path is not None:
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        subject: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """Create a new task with PENDING status.

        Args:
            subject: Short summary of the task.
            description: Detailed description.
            metadata: Optional key-value metadata.

        Returns:
            The newly created :class:`Task`.
        """
        now = time.time()
        task_id = str(self._next_id)
        self._next_id += 1

        task = Task(
            id=task_id,
            subject=subject,
            description=description,
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        self._tasks[task_id] = task
        self._save()
        logger.info("Created task %s: %s", task_id, subject)
        return task

    def get(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by its ID.

        Args:
            task_id: The task identifier.

        Returns:
            The :class:`Task` if found, otherwise ``None``.
        """
        return self._tasks.get(task_id)

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        """List tasks, optionally filtered by status.

        Args:
            status: If provided, only tasks with this status are returned.

        Returns:
            A list of matching :class:`Task` instances ordered by creation
            time (oldest first).
        """
        tasks = list(self._tasks.values())
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        tasks.sort(key=lambda t: t.created_at)
        return tasks

    def update(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        subject: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """Update fields of an existing task.

        Only the fields that are not ``None`` will be modified.

        Args:
            task_id: The task identifier.
            status: New status value.
            subject: New subject text.
            description: New description text.
            metadata: Metadata dict to merge into existing metadata.

        Returns:
            The updated :class:`Task`.

        Raises:
            KeyError: If no task with *task_id* exists.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task '{task_id}' not found")

        if status is not None:
            task.status = status
        if subject is not None:
            task.subject = subject
        if description is not None:
            task.description = description
        if metadata is not None:
            task.metadata.update(metadata)

        task.updated_at = time.time()
        self._save()
        logger.info("Updated task %s", task_id)
        return task

    def delete(self, task_id: str) -> bool:
        """Delete a task by its ID.

        Args:
            task_id: The task identifier.

        Returns:
            ``True`` if the task was deleted, ``False`` if it did not exist.
        """
        if task_id not in self._tasks:
            return False
        del self._tasks[task_id]
        self._save()
        logger.info("Deleted task %s", task_id)
        return True

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Persist the current store to disk (if a path is configured)."""
        if self._persist_path is None:
            return

        data = {
            "next_id": self._next_id,
            "tasks": [t.to_dict() for t in self._tasks.values()],
        }
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(
                json_dumps(data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("Failed to save task store to %s: %s", self._persist_path, exc)

    def _load(self) -> None:
        """Load the store from disk (if the file exists)."""
        if self._persist_path is None or not self._persist_path.exists():
            return

        try:
            text = self._persist_path.read_text(encoding="utf-8")
            data = json_loads(text)
        except (OSError, JSONDecodeError) as exc:
            logger.error("Failed to load task store from %s: %s", self._persist_path, exc)
            return

        self._next_id = data.get("next_id", 1)
        for task_data in data.get("tasks", []):
            try:
                task = Task.from_dict(task_data)
                self._tasks[task.id] = task
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed task entry: %s", exc)

        logger.info("Loaded %d tasks from %s", len(self._tasks), self._persist_path)
