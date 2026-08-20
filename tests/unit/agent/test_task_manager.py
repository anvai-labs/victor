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

"""Tests for TaskManager mixin.

Tests verify:
- Task tracking and lifecycle
- Automatic cleanup on completion
- Exception logging
- Cancellation and timeout handling
- No resource leaks
"""

import asyncio
import logging

import pytest

from victor.core.task_manager import TaskManager


class TestComponent(TaskManager):
    """Test component using TaskManager."""

    def __init__(self):
        super().__init__()
        self.results = []
        self.errors = []

    async def do_work(self, value, delay=0.1):
        """Simulate work with optional delay."""
        await asyncio.sleep(delay)
        self.results.append(value)
        return value

    async def failing_work(self, error_msg="Test error"):
        """Simulate failing work."""
        await asyncio.sleep(0.05)
        raise ValueError(error_msg)


@pytest.mark.asyncio
async def test_task_tracking():
    """Verify tasks are tracked."""
    component = TestComponent()

    # Create some tasks
    task1 = component.create_tracked_task(component.do_work(1), name="task1")
    task2 = component.create_tracked_task(component.do_work(2), name="task2")

    # Verify tasks are tracked
    assert component.active_task_count == 2
    assert component.has_active_tasks

    # Wait for completion
    await asyncio.gather(task1, task2)

    # Verify tasks auto-removed
    assert component.active_task_count == 0
    assert not component.has_active_tasks
    assert component.results == [1, 2]


@pytest.mark.asyncio
async def test_cleanup():
    """Verify cleanup cancels all tasks."""
    component = TestComponent()

    # Create long-running task
    async def slow_work():
        await asyncio.sleep(10)  # Will be cancelled

    component.create_tracked_task(slow_work(), name="slow")
    assert component.active_task_count == 1

    # Cleanup should cancel
    await component.cleanup_tasks()

    assert component.active_task_count == 0


@pytest.mark.asyncio
async def test_exception_logging(caplog):
    """Verify exceptions are logged."""
    component = TestComponent()

    caplog.set_level(logging.ERROR)

    task = component.create_tracked_task(component.failing_work("Test error"), name="failing")

    # Wait for task to fail
    with pytest.raises(ValueError):
        await task

    # Give callback time to run
    await asyncio.sleep(0.1)

    # Verify logged
    assert "Background task failed: failing" in caplog.text
    assert "Test error" in caplog.text


@pytest.mark.asyncio
async def test_multiple_tasks():
    """Test tracking multiple tasks simultaneously."""
    component = TestComponent()

    # Create 10 tasks
    tasks = [
        component.create_tracked_task(component.do_work(i, delay=0.05), name=f"task{i}")
        for i in range(10)
    ]

    assert component.active_task_count == 10

    # Wait for all
    await asyncio.gather(*tasks)

    assert component.active_task_count == 0
    assert len(component.results) == 10
    assert set(component.results) == set(range(10))


@pytest.mark.asyncio
async def test_cleanup_timeout():
    """Test cleanup respects timeout."""
    component = TestComponent()

    async def very_slow_work():
        await asyncio.sleep(100)  # Very slow

    # Create slow task
    component.create_tracked_task(very_slow_work(), name="very_slow")

    # Cleanup with short timeout
    await component.cleanup_tasks(timeout=0.1)

    # Should complete without hanging
    assert component.active_task_count == 0


@pytest.mark.asyncio
async def test_no_cleanup_needed():
    """Test cleanup with no active tasks."""
    component = TestComponent()

    # No tasks created
    assert component.active_task_count == 0

    # Cleanup should handle gracefully
    await component.cleanup_tasks()

    assert component.active_task_count == 0


@pytest.mark.asyncio
async def test_task_name_assignment():
    """Verify task names are assigned correctly."""
    component = TestComponent()

    task = component.create_tracked_task(component.do_work(1), name="my_task")

    assert task.get_name() == "my_task"
    assert component.active_task_count == 1

    await task
    assert component.active_task_count == 0


@pytest.mark.asyncio
async def test_mixed_success_and_failure():
    """Test cleanup handles mix of successful and failed tasks."""
    component = TestComponent()

    # Create mix of tasks
    task1 = component.create_tracked_task(component.do_work(1, delay=0.05), name="success1")
    task2 = component.create_tracked_task(component.failing_work(), name="fail1")
    task3 = component.create_tracked_task(component.do_work(2, delay=0.05), name="success2")

    assert component.active_task_count == 3

    # Wait for completion (some will fail)
    results = await asyncio.gather(task1, task2, task3, return_exceptions=True)

    # Verify cleanup
    assert component.active_task_count == 0
    assert len([r for r in results if isinstance(r, ValueError)]) == 1
    assert len(component.results) == 2


@pytest.mark.asyncio
async def test_cancellation_handling():
    """Test proper handling of cancelled tasks."""
    component = TestComponent()

    task = component.create_tracked_task(
        asyncio.sleep(10),  # Long sleep that will be cancelled
        name="cancellable",
    )
    assert component.active_task_count == 1

    # Cancel the task
    task.cancel()

    # Wait for cancellation to complete
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Give callback time to run
    await asyncio.sleep(0.1)

    # Verify task was removed from tracking
    assert component.active_task_count == 0


@pytest.mark.asyncio
async def test_cleanup_indefinite_wait():
    """Test cleanup with no timeout (cancels but waits for cancellation)."""
    component = TestComponent()

    # Create task that completes eventually (but will be cancelled)
    component.create_tracked_task(component.do_work(1, delay=0.2), name="medium")

    # Cleanup with no timeout cancels tasks and waits for cancellation
    await component.cleanup_tasks(timeout=None)

    # Tasks are cancelled, so they don't complete their work
    assert component.active_task_count == 0
    # Note: result won't be added because task was cancelled
    assert 1 not in component.results


@pytest.mark.asyncio
async def test_task_done_callback_exception_handling():
    """Test that exceptions in done callback don't propagate."""
    component = TestComponent()

    # This should not raise even though getting result of cancelled task
    # can cause issues - the callback handles it
    task = component.create_tracked_task(component.do_work(1), name="test")
    task.cancel()

    # Wait a bit for callback
    await asyncio.sleep(0.1)

    # Should have been removed from tracking
    assert component.active_task_count == 0


@pytest.mark.asyncio
async def test_integration_with_multiple_components():
    """Test multiple components each with their own TaskManager."""
    component1 = TestComponent()
    component2 = TestComponent()

    # Create tasks in each
    task1 = component1.create_tracked_task(component1.do_work(1, delay=0.2), name="c1_task1")
    task2 = component1.create_tracked_task(component1.do_work(2, delay=0.2), name="c1_task2")

    task3 = component2.create_tracked_task(component2.do_work(10, delay=0.2), name="c2_task1")
    task4 = component2.create_tracked_task(component2.do_work(20, delay=0.2), name="c2_task2")

    # Verify isolation
    assert component1.active_task_count == 2
    assert component2.active_task_count == 2

    # Let component1 tasks complete
    await asyncio.gather(task1, task2)
    # Small delay for callbacks to run
    await asyncio.sleep(0.05)
    assert component1.active_task_count == 0
    # Component2 tasks may have also completed by now, which is fine
    assert component2.active_task_count in (0, 2)

    # Let component2 tasks complete if not already done
    await asyncio.gather(task3, task4)
    await asyncio.sleep(0.05)
    assert component2.active_task_count == 0

    # Verify results
    assert set(component1.results) == {1, 2}
    assert set(component2.results) == {10, 20}
