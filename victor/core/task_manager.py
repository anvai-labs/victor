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

"""Task Manager - Mixin for managing background tasks with proper cleanup.

This module provides a reusable mixin for tracking asyncio background tasks
and ensuring proper cleanup to prevent resource leaks.

Design Pattern: Mixin
- Provides task tracking functionality to any class
- Manages task lifecycle (creation, tracking, cleanup)
- Handles exceptions in background tasks

Usage:
    class MyComponent(TaskManager):
        def __init__(self):
            super().__init__()
            # Component initialization

        async def start(self):
            # Create tracked background task
            self.create_tracked_task(
                self._background_worker(),
                name="background_worker"
            )

        async def shutdown(self):
            # Clean up all tasks
            await self.cleanup_tasks()

Example:
    component = MyComponent()
    await component.start()
    # ... do work ...
    await component.shutdown()  # Ensures all tasks are cancelled and awaited
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Optional, Set

logger = logging.getLogger(__name__)


class TaskManager:
    """Mixin for managing background tasks with proper cleanup.

    Provides methods to create and track asyncio tasks, ensuring they are
    properly cleaned up during shutdown to prevent resource leaks and
    unhandled exceptions.

    Attributes:
        _background_tasks: Set of currently running background tasks.
    """

    def __init__(self) -> None:
        """Initialize the TaskManager."""
        # Use a set to track tasks efficiently
        self._background_tasks: Set[asyncio.Task] = set()

    def create_tracked_task(
        self,
        coro: Awaitable,
        name: Optional[str] = None,
    ) -> asyncio.Task:
        """Create a task and track it for cleanup.

        The task will be automatically tracked and removed from the tracking
        set when it completes. Any exceptions raised by the task will be
        logged (but not raised) to prevent silent failures.

        Args:
            coro: Coroutine to run as a background task.
            name: Optional task name for debugging and logging.

        Returns:
            The created asyncio.Task.

        Raises:
            RuntimeError: If no event loop is running.

        Example:
            task = component.create_tracked_task(
                my_async_function(),
                name="my_worker"
            )
        """
        # Create the task with optional name
        task = asyncio.create_task(coro, name=name)

        # Add to tracking set
        self._background_tasks.add(task)

        # Register cleanup callback
        task.add_done_callback(self._task_done_callback)

        logger.debug(f"Created tracked task: {name or task.get_name()}")
        return task

    def _task_done_callback(self, task: asyncio.Task) -> None:
        """Callback invoked when a tracked task completes.

        Removes the task from the tracking set and logs any exceptions
        that occurred during task execution.

        Args:
            task: The completed task.
        """
        # Remove from tracking set
        self._background_tasks.discard(task)

        # Log any exceptions (prevents silent failures)
        try:
            if not task.cancelled():
                exc = task.exception()
                if exc:
                    task_name = task.get_name()
                    logger.error(
                        f"Background task failed: {task_name}",
                        exc_info=exc,
                    )
        except asyncio.CancelledError:
            # Task was cancelled - this is expected during shutdown
            pass
        except Exception as e:
            # Unexpected error accessing task state
            logger.warning(f"Error in task done callback: {e}")

    async def cleanup_tasks(self, timeout: Optional[float] = 5.0) -> None:
        """Cancel all background tasks and wait for completion.

        This method should be called during shutdown to ensure all
        background tasks are properly cleaned up.

        Args:
            timeout: Maximum time to wait for tasks to complete (seconds).
                    None means wait indefinitely. Default: 5.0 seconds.

        Example:
            async def shutdown(self):
                logger.info("Shutting down component...")
                await self.cleanup_tasks()
                # Other cleanup...
        """
        if not self._background_tasks:
            logger.debug("No background tasks to clean up")
            return

        task_count = len(self._background_tasks)
        logger.info(f"Cleaning up {task_count} background task(s)...")

        # Cancel all tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        # Wait for all tasks to complete (with timeout if specified)
        if timeout is not None:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._background_tasks, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Task cleanup timed out after {timeout}s, "
                    f"{len([t for t in self._background_tasks if not t.done()])} task(s) still running"
                )
        else:
            # Wait indefinitely
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Clear the tracking set
        self._background_tasks.clear()
        logger.debug("Background task cleanup complete")

    @property
    def active_task_count(self) -> int:
        """Get count of active background tasks.

        Returns:
            Number of tasks currently tracked and running.
        """
        return len(self._background_tasks)

    @property
    def has_active_tasks(self) -> bool:
        """Check if there are any active background tasks.

        Returns:
            True if there are active tasks, False otherwise.
        """
        return bool(self._background_tasks)
