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

"""Async/Sync Bridging Utilities.

This module provides utilities for bridging async and sync code following
the async-first pattern:

1. Implement core logic as async
2. Provide sync wrappers at public API boundaries only
3. Avoid nested event loops

Design Philosophy:
- Async is the primary implementation
- Sync wrappers are convenience methods at API boundaries
- Never nest asyncio.run() calls
- Use run_sync() for safe sync-to-async bridging

Example usage:
    from victor.core.async_utils import run_sync, SyncAsyncBridge

    # Simple sync wrapper for an async function
    async def fetch_data():
        return await some_async_operation()

    # Call from sync context
    result = run_sync(fetch_data())

    # Or use the bridge for class methods
    class MyService(SyncAsyncBridge):
        async def afetch(self):
            return await some_operation()

        def fetch(self):
            return self.run_sync(self.afetch())
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import threading
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_sync(coro: Awaitable[T]) -> T:
    """Run an async coroutine from sync context safely.

    Handles the case where we might already be in an async context
    by checking for a running event loop.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine

    Raises:
        RuntimeError: If called from within an async context (nested loop)

    Example:
        async def async_fetch():
            return await fetch_data()

        # Safe to call from sync code
        result = run_sync(async_fetch())
    """
    try:
        # Check if an event loop is already running
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except (RuntimeError, AttributeError):
            # No running loop - safe to use asyncio.run()
            pass

        if loop is not None and loop.is_running():
            # We're in an async context - close bare coroutines to avoid leak warnings.
            if inspect.iscoroutine(coro):
                coro.close()

            raise RuntimeError(
                "Cannot use run_sync() from within an async context. Use 'await' instead."
            )

        # Run on a private loop WITHOUT touching the thread's current-loop
        # binding. asyncio.run() would call set_event_loop(None) on exit,
        # silently unbinding whatever loop the caller had set — e.g. the
        # pytest-asyncio test loop, killing every async test that runs after
        # a sync path emits a security-audit event. Passing an explicit
        # loop_factory makes Runner skip current-loop management while
        # keeping asyncio.run()'s cleanup semantics (cancel pending tasks,
        # shutdown asyncgens and the default executor).
        with asyncio.Runner(loop_factory=asyncio.new_event_loop) as runner:
            return runner.run(coro)

    except Exception as e:
        # Re-raise runtime errors as they are likely configuration/logic issues
        if isinstance(e, RuntimeError):
            raise
        # For others, log and re-raise
        logger.debug(f"run_sync failed: {e}")
        raise


def run_sync_in_thread(coro: Awaitable[T], timeout: float | None = None) -> T:
    """Run an async coroutine from sync context using a thread.

    This is useful when called from within a sync callback that's
    running inside an async context. It runs the coroutine in a
    new thread with its own event loop.

    Warning: This has higher overhead than run_sync(). Only use when
    necessary to avoid nested loop issues.

    Args:
        coro: The coroutine to run
        timeout: Optional timeout in seconds for the worker thread

    Returns:
        The result of the coroutine

    Raises:
        TimeoutError: If timeout is reached before the coroutine completes
    """
    result: list[T | None] = [None]
    error: list[Exception | None] = [None]

    def run_in_loop() -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result[0] = loop.run_until_complete(coro)
        except Exception as exc:  # pragma: no cover - exercised via wrapper tests
            error[0] = exc
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    worker = threading.Thread(target=run_in_loop, daemon=True)
    worker.start()
    worker.join(timeout=timeout)

    if worker.is_alive():
        raise TimeoutError("Async operation timed out")
    if error[0] is not None:
        raise error[0]

    return result[0]  # type: ignore[return-value]


def async_to_sync(func: Callable[..., Awaitable[T]]) -> Callable[..., T]:
    """Decorator to create a sync version of an async function.

    Use this sparingly, only at public API boundaries.

    Args:
        func: The async function to wrap

    Returns:
        A sync wrapper that calls run_sync()

    Example:
        @async_to_sync
        async def fetch_data():
            return await some_operation()

        # Now callable synchronously
        result = fetch_data()
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return run_sync(func(*args, **kwargs))

    return wrapper


class SyncAsyncBridge:
    """Mixin for classes that provide both sync and async interfaces.

    Provides utilities for implementing the async-first pattern
    in service classes.

    Example:
        class DataService(SyncAsyncBridge):
            async def afetch(self, key: str) -> Data:
                '''Async primary implementation.'''
                return await self._fetch_impl(key)

            def fetch(self, key: str) -> Data:
                '''Sync convenience wrapper.'''
                return self._run_sync(self.afetch(key))

            async def _fetch_impl(self, key: str) -> Data:
                '''Core async implementation.'''
                ...
    """

    def _run_sync(self, coro: Awaitable[T]) -> T:
        """Run a coroutine synchronously.

        Subclasses can override for custom behavior.
        """
        return run_sync(coro)

    @classmethod
    def sync_method(cls, async_method: Callable[..., Awaitable[T]]) -> Callable[..., T]:
        """Create a sync method wrapper for an async method.

        Args:
            async_method: The async method to wrap

        Returns:
            A sync method that delegates to the async method

        Example:
            class MyService(SyncAsyncBridge):
                async def afetch(self): ...
                fetch = SyncAsyncBridge.sync_method(afetch)
        """

        @functools.wraps(async_method)
        def wrapper(self, *args: Any, **kwargs: Any) -> T:
            return self._run_sync(async_method(self, *args, **kwargs))

        return wrapper


async def run_blocking(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a blocking sync function from async context without blocking the event loop.

    Uses asyncio.to_thread() to offload blocking I/O (file reads, subprocess,
    database queries, etc.) to a thread pool, keeping the event loop responsive.

    This is the inverse of run_sync(): while run_sync() lets sync code call async,
    run_blocking() lets async code call sync blocking functions safely.

    Args:
        func: The blocking synchronous function to run
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func

    Returns:
        The result of the blocking function

    Example:
        async def async_handler():
            # Don't block the event loop with file I/O
            content = await run_blocking(Path("large.txt").read_text)

            # Or with any blocking library call
            result = await run_blocking(blocking_api_call, param1, param2)
    """
    if kwargs:
        func_with_kwargs = functools.partial(func, **kwargs)
        return await asyncio.to_thread(func_with_kwargs, *args)
    return await asyncio.to_thread(func, *args)


def ensure_async(func: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
    """Ensure a function is async-compatible.

    If the function is already async, returns it unchanged.
    If sync, wraps it to be async.

    Args:
        func: Function to ensure is async

    Returns:
        An async function

    Example:
        async_fn = ensure_async(some_function)
        result = await async_fn(args)
    """
    if asyncio.iscoroutinefunction(func):
        return func

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper


__all__ = [
    "run_sync",
    "run_sync_in_thread",
    "run_blocking",
    "async_to_sync",
    "SyncAsyncBridge",
    "ensure_async",
]
