"""Tests for SessionService.create_background_task (extracted from orchestrator)."""

import asyncio
import threading

import pytest

from victor.agent.services.session_service import SessionService


class TestCreateBackgroundTask:
    @pytest.fixture
    def task_state(self):
        return {
            "tasks": set(),
            "lock": threading.Lock(),
        }

    @pytest.mark.asyncio
    async def test_creates_and_tracks_task(self, task_state):
        async def noop():
            pass

        task = SessionService.create_background_task(
            noop(), "test-task", task_state["tasks"], task_state["lock"]
        )

        assert task is not None
        assert task.get_name() == "test-task"
        assert task in task_state["tasks"]

        await task
        # After completion, done callback should discard the task
        # Give the callback a moment to fire
        await asyncio.sleep(0)
        assert task not in task_state["tasks"]

    @pytest.mark.asyncio
    async def test_removes_task_on_completion(self, task_state):
        async def quick():
            return 42

        task = SessionService.create_background_task(
            quick(), "quick-task", task_state["tasks"], task_state["lock"]
        )

        assert task in task_state["tasks"]
        await task
        await asyncio.sleep(0)
        assert task not in task_state["tasks"]

    @pytest.mark.asyncio
    async def test_removes_task_on_exception(self, task_state):
        async def failing():
            raise ValueError("boom")

        task = SessionService.create_background_task(
            failing(), "fail-task", task_state["tasks"], task_state["lock"]
        )

        assert task in task_state["tasks"]
        with pytest.raises(ValueError, match="boom"):
            await task
        await asyncio.sleep(0)
        assert task not in task_state["tasks"]

    def test_returns_none_without_event_loop(self, task_state):
        async def noop():
            pass

        coro = noop()
        task = SessionService.create_background_task(
            coro, "no-loop", task_state["tasks"], task_state["lock"]
        )

        assert task is None
        assert len(task_state["tasks"]) == 0

    def test_closes_coroutine_when_no_loop(self, task_state):
        async def noop():
            pass

        coro = noop()
        SessionService.create_background_task(
            coro, "close-test", task_state["tasks"], task_state["lock"]
        )

        # Coroutine should have been closed (no ResourceWarning)
        # If it wasn't closed, Python would warn about never-awaited coroutine

    @pytest.mark.asyncio
    async def test_multiple_concurrent_tasks(self, task_state):
        results = []

        async def worker(n):
            results.append(n)

        tasks = []
        for i in range(5):
            t = SessionService.create_background_task(
                worker(i), f"worker-{i}", task_state["tasks"], task_state["lock"]
            )
            tasks.append(t)

        assert len(task_state["tasks"]) == 5
        await asyncio.gather(*tasks)
        await asyncio.sleep(0)
        assert len(task_state["tasks"]) == 0
        assert sorted(results) == [0, 1, 2, 3, 4]
