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

"""Runtime helpers for compiled StateGraph execution."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class GraphRuntimeOutcome:
    """Internal outcome for compiled graph execution helpers."""

    state: Any
    success: bool
    error: Optional[str] = None
    iterations: int = 0
    duration: float = 0.0
    node_history: List[str] = field(default_factory=list)
    state_history: List[Tuple[str, Any]] = field(default_factory=list)


async def run_graph_execution(
    *,
    state: Any,
    current_node: str,
    end_node_token: str,
    entry_point: str,
    node_count: int,
    thread_id: str,
    iteration_controller: Any,
    timeout_manager: Any,
    interrupt_handler: Any,
    node_executor: Any,
    checkpoint_manager: Any,
    event_emitter: Any,
    hook: Optional[Any],
    validate_state: Optional[Callable[[str, Any], Optional[str]]],
    snapshot_state: Callable[[Any], Any],
    get_next_node: Callable[[str, Any], Union[str, List[Any]]],
    execute_parallel: Callable[[List[Any], Any, Any, Any], Awaitable[Any]],
) -> GraphRuntimeOutcome:
    """Run compiled graph execution using focused collaborators."""
    return await _run_graph_execution_loop(
        state=state,
        current_node=current_node,
        end_node_token=end_node_token,
        entry_point=entry_point,
        node_count=node_count,
        thread_id=thread_id,
        iteration_controller=iteration_controller,
        timeout_manager=timeout_manager,
        interrupt_handler=interrupt_handler,
        node_executor=node_executor,
        checkpoint_manager=checkpoint_manager,
        event_emitter=event_emitter,
        hook=hook,
        validate_state=validate_state,
        snapshot_state=snapshot_state,
        get_next_node=get_next_node,
        execute_parallel=execute_parallel,
        on_node_complete=None,
    )


async def stream_graph_execution(
    *,
    state: Any,
    current_node: str,
    end_node_token: str,
    entry_point: str,
    node_count: int,
    thread_id: str,
    iteration_controller: Any,
    timeout_manager: Any,
    interrupt_handler: Any,
    node_executor: Any,
    checkpoint_manager: Any,
    event_emitter: Any,
    hook: Optional[Any],
    validate_state: Optional[Callable[[str, Any], Optional[str]]],
    snapshot_state: Callable[[Any], Any],
    get_next_node: Callable[[str, Any], Union[str, List[Any]]],
    execute_parallel: Callable[[List[Any], Any, Any, Any], Awaitable[Any]],
):
    """Stream compiled graph execution using the same runtime path as invoke()."""
    queue: asyncio.Queue[Any] = asyncio.Queue()
    sentinel = object()

    async def _on_node_complete(node_id: str, node_state: Any) -> None:
        await queue.put((node_id, snapshot_state(node_state)))

    async def _runner() -> GraphRuntimeOutcome:
        try:
            return await _run_graph_execution_loop(
                state=state,
                current_node=current_node,
                end_node_token=end_node_token,
                entry_point=entry_point,
                node_count=node_count,
                thread_id=thread_id,
                iteration_controller=iteration_controller,
                timeout_manager=timeout_manager,
                interrupt_handler=interrupt_handler,
                node_executor=node_executor,
                checkpoint_manager=checkpoint_manager,
                event_emitter=event_emitter,
                hook=hook,
                validate_state=validate_state,
                snapshot_state=snapshot_state,
                get_next_node=get_next_node,
                execute_parallel=execute_parallel,
                on_node_complete=_on_node_complete,
            )
        finally:
            await queue.put(sentinel)

    runner = asyncio.create_task(_runner())
    try:
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield item
    finally:
        await runner


async def _run_graph_execution_loop(
    *,
    state: Any,
    current_node: str,
    end_node_token: str,
    entry_point: str,
    node_count: int,
    thread_id: str,
    iteration_controller: Any,
    timeout_manager: Any,
    interrupt_handler: Any,
    node_executor: Any,
    checkpoint_manager: Any,
    event_emitter: Any,
    hook: Optional[Any],
    validate_state: Optional[Callable[[str, Any], Optional[str]]],
    snapshot_state: Callable[[Any], Any],
    get_next_node: Callable[[str, Any], Union[str, List[Any]]],
    execute_parallel: Callable[[List[Any], Any, Any, Any], Awaitable[Any]],
    on_node_complete: Optional[Callable[[str, Any], Awaitable[None]]],
) -> GraphRuntimeOutcome:
    """Shared compiled graph execution loop for invoke() and stream()."""
    timeout_manager.start()
    event_emitter.emit_graph_started(
        entry_point=entry_point,
        node_count=node_count,
        thread_id=thread_id,
    )

    node_history: List[str] = []
    state_history: List[Tuple[str, Any]] = []

    try:
        while current_node != end_node_token:
            should_continue, error = iteration_controller.should_continue(current_node)
            if not should_continue:
                logger.warning("Iteration limit reached: %s", error)
                return GraphRuntimeOutcome(
                    state=state,
                    success=False,
                    error=error,
                    iterations=iteration_controller.iterations,
                    duration=timeout_manager.get_elapsed(),
                    node_history=node_history,
                    state_history=state_history,
                )

            if interrupt_handler.should_interrupt_before(current_node):
                logger.info("Interrupt before node: %s", current_node)
                await checkpoint_manager.save_checkpoint(thread_id, current_node, state)
                return GraphRuntimeOutcome(
                    state=state,
                    success=True,
                    iterations=iteration_controller.iterations,
                    duration=timeout_manager.get_elapsed(),
                    node_history=node_history,
                    state_history=state_history,
                )

            if hook:
                await hook.before_node(current_node, state)

            node_start_time = time.time()
            event_emitter.emit_node_start(
                node_id=current_node,
                iteration=iteration_controller.iterations,
            )

            success, error, state = await node_executor.execute(
                node_id=current_node,
                state=state,
                timeout_manager=timeout_manager,
            )

            if hook:
                await hook.after_node(current_node, state, error if not success else None)

            if not success:
                return GraphRuntimeOutcome(
                    state=state,
                    success=False,
                    error=error,
                    iterations=iteration_controller.iterations,
                    duration=timeout_manager.get_elapsed(),
                    node_history=node_history,
                    state_history=state_history,
                )

            if validate_state is not None:
                validation_error = validate_state(current_node, state)
                if validation_error is not None:
                    return GraphRuntimeOutcome(
                        state=state,
                        success=False,
                        error=validation_error,
                        iterations=iteration_controller.iterations,
                        duration=timeout_manager.get_elapsed(),
                        node_history=node_history,
                        state_history=state_history,
                    )

            logger.debug("Executed node: %s", current_node)
            node_history.append(current_node)
            await checkpoint_manager.save_checkpoint(thread_id, current_node, state)
            state_history.append((current_node, snapshot_state(state)))
            if on_node_complete is not None:
                await on_node_complete(current_node, state)

            event_emitter.emit_node_complete(
                node_id=current_node,
                iteration=iteration_controller.iterations,
                duration=time.time() - node_start_time,
            )

            if interrupt_handler.should_interrupt_after(current_node):
                logger.info("Interrupt after node: %s", current_node)
                return GraphRuntimeOutcome(
                    state=state,
                    success=True,
                    iterations=iteration_controller.iterations,
                    duration=timeout_manager.get_elapsed(),
                    node_history=node_history,
                    state_history=state_history,
                )

            next_target = get_next_node(current_node, state)
            if isinstance(next_target, list):
                state = await execute_parallel(
                    next_target,
                    node_executor,
                    timeout_manager,
                    state,
                )
                for send in next_target:
                    node_history.append(f"send:{send.node}")

                join_node = next((send.join_at for send in next_target if send.join_at), None)
                if join_node is not None:
                    logger.debug("Fan-out complete, continuing at join node '%s'", join_node)
                    current_node = join_node
                else:
                    current_node = end_node_token
            else:
                current_node = next_target

        event_emitter.emit_graph_completed(
            success=True,
            iterations=iteration_controller.iterations,
            duration=timeout_manager.get_elapsed(),
            node_count=len(node_history),
        )
        return GraphRuntimeOutcome(
            state=state,
            success=True,
            iterations=iteration_controller.iterations,
            duration=timeout_manager.get_elapsed(),
            node_history=node_history,
            state_history=state_history,
        )
    except asyncio.TimeoutError:
        event_emitter.emit_graph_error(
            error="Execution timeout",
            iterations=iteration_controller.iterations,
            duration=timeout_manager.get_elapsed(),
        )
        return GraphRuntimeOutcome(
            state=state,
            success=False,
            error="Execution timeout",
            iterations=iteration_controller.iterations,
            duration=timeout_manager.get_elapsed(),
            node_history=node_history,
            state_history=state_history,
        )
    except Exception as error:
        logger.error("Graph execution failed: %s", error, exc_info=True)
        event_emitter.emit_graph_error(
            error=str(error),
            iterations=iteration_controller.iterations,
            duration=timeout_manager.get_elapsed(),
        )
        return GraphRuntimeOutcome(
            state=state,
            success=False,
            error=str(error),
            iterations=iteration_controller.iterations,
            duration=timeout_manager.get_elapsed(),
            node_history=node_history,
            state_history=state_history,
        )
