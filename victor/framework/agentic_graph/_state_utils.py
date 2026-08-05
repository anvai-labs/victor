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

"""Shared state helpers for agentic graph nodes.

Consolidation Note:
    This module is an internal compatibility layer for agentic-graph callables.
    With the removal of the ``AgenticLoopState`` TypedDict, the only legacy
    runtime inputs that still need normalization are:

    1. ``AgenticLoopStateModel`` — the common case (direct Pydantic model)
    2. ``CopyOnWriteState`` — when the graph engine wraps state in COW
    3. ``dict`` — when a raw dict is passed (e.g. from deserialized checkpoints)

    ``unwrap_state()`` normalises all three to ``AgenticLoopStateModel`` and
    ``normalize_state_argument()`` applies that normalization at the callable
    boundary.
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable, Dict, TypeVar, cast

from victor.framework.agentic_graph.state import AgenticLoopStateModel
from victor.framework.graph import CopyOnWriteState

CallableT = TypeVar("CallableT", bound=Callable[..., Any])


def unwrap_state(state: Any) -> AgenticLoopStateModel:
    """Normalize node state to AgenticLoopStateModel.

    Handles three input shapes:
    - ``AgenticLoopStateModel`` → returned as-is (zero cost for the common path)
    - ``CopyOnWriteState`` → unwrapped, then converted if needed
    - ``dict`` → passed to ``AgenticLoopStateModel(**state)``

    Args:
        state: Node state in any supported form.

    Returns:
        A validated ``AgenticLoopStateModel``.

    Raises:
        TypeError: If the state is not a supported agentic-graph input shape.
    """
    if isinstance(state, AgenticLoopStateModel):
        return state
    if isinstance(state, CopyOnWriteState):
        unwrapped = state.get_state()
        if isinstance(unwrapped, AgenticLoopStateModel):
            return unwrapped
        if isinstance(unwrapped, dict):
            return AgenticLoopStateModel(**unwrapped)
        raise TypeError(
            "Unsupported CopyOnWriteState payload for agentic graph state: "
            f"{type(unwrapped).__name__}"
        )
    if isinstance(state, dict):
        return AgenticLoopStateModel(**state)
    raise TypeError(
        "Unsupported agentic graph state type: "
        f"{type(state).__name__}. Expected AgenticLoopStateModel, dict, "
        "or CopyOnWriteState wrapping one of those."
    )


def normalize_state_argument(
    *,
    position: int = 0,
    keyword: str = "state",
) -> Callable[[CallableT], CallableT]:
    """Normalize a callable's state argument to AgenticLoopStateModel.

    This keeps node and edge call sites backward compatible with dict and
    ``CopyOnWriteState`` inputs while removing per-function normalization
    boilerplate.

    Args:
        position: Positional index of the state argument.
        keyword: Keyword name of the state argument.

    Returns:
        Decorator that normalizes the targeted state argument before calling
        the wrapped function.
    """

    def decorator(func: CallableT) -> CallableT:
        def _normalize_call(
            args: tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> tuple[tuple[Any, ...], Dict[str, Any]]:
            updated_kwargs = kwargs
            if len(args) > position:
                updated_args = list(args)
                updated_args[position] = unwrap_state(updated_args[position])
                return tuple(updated_args), updated_kwargs
            if keyword in kwargs:
                updated_kwargs = dict(kwargs)
                updated_kwargs[keyword] = unwrap_state(updated_kwargs[keyword])
            return args, updated_kwargs

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                normalized_args, normalized_kwargs = _normalize_call(args, kwargs)
                return await func(*normalized_args, **normalized_kwargs)

            return cast(CallableT, async_wrapper)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            normalized_args, normalized_kwargs = _normalize_call(args, kwargs)
            return func(*normalized_args, **normalized_kwargs)

        return cast(CallableT, sync_wrapper)

    return decorator
