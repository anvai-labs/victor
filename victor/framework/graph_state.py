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

"""State wrappers and helpers for StateGraph execution."""

from __future__ import annotations

import copy
from typing import Any, Dict, Generic, Optional, TypeVar

StateType = TypeVar("StateType", bound=Dict[str, Any])


class CopyOnWriteState(Generic[StateType]):
    """Copy-on-write wrapper for workflow state.

    MIGRATION NOTICE: For persistent state storage across workflow executions,
    use the canonical state management system:
        - victor.state.WorkflowStateManager - Workflow scope state
        - victor.state.get_global_manager() - Unified access to all scopes

    CopyOnWriteState is kept as a performance optimization for workflow graphs,
    providing copy-on-write semantics for state within a single execution.

    ⚠️ THREAD SAFETY WARNING ⚠️:
        This class is NOT thread-safe and must NOT be shared across threads.

        Each thread MUST have its own CopyOnWriteState wrapper instance.
        Sharing the same wrapper instance between threads will lead to
        race conditions, data corruption, and undefined behavior.
    """

    __slots__ = ("_source", "_copy", "_modified", "_owner_thread", "_owner_task")

    def __init__(self, source: StateType):
        """Initialize with source state."""
        import asyncio
        import threading

        self._source: StateType = source
        self._copy: Optional[StateType] = None
        self._modified: bool = False
        self._owner_thread: int = threading.current_thread().ident or 0
        try:
            self._owner_task: Optional[int] = id(asyncio.current_task())
        except RuntimeError:
            self._owner_task = None

    def _ensure_copy(self) -> StateType:
        """Ensure a mutable copy exists before mutation."""
        import threading

        current = threading.current_thread().ident or 0
        if current != self._owner_thread:
            raise RuntimeError(
                f"CopyOnWriteState thread violation: created on thread "
                f"{self._owner_thread}, mutated from thread {current}. "
                f"Each thread must use its own CopyOnWriteState wrapper."
            )
        if self._owner_task is not None and __debug__:
            import asyncio

            try:
                current_task_id = id(asyncio.current_task())
            except RuntimeError:
                current_task_id = None
            if current_task_id is not None and current_task_id != self._owner_task:
                raise RuntimeError(
                    "CopyOnWriteState task violation: created in a different "
                    "asyncio task. Each task must use its own CopyOnWriteState "
                    "wrapper."
                )
        if not self._modified:
            self._copy = copy.deepcopy(self._source)
            self._modified = True
        return self._copy  # type: ignore[return-value]

    def __getitem__(self, key: str) -> Any:
        if self._modified:
            return self._copy[key]  # type: ignore[index]
        return self._source[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._ensure_copy()[key] = value

    def __delitem__(self, key: str) -> None:
        del self._ensure_copy()[key]

    def __contains__(self, key: object) -> bool:
        if self._modified:
            return key in self._copy  # type: ignore[operator]
        return key in self._source

    def __len__(self) -> int:
        if self._modified:
            return len(self._copy)  # type: ignore[arg-type]
        return len(self._source)

    def __iter__(self):
        if self._modified:
            return iter(self._copy)  # type: ignore[arg-type]
        return iter(self._source)

    def get(self, key: str, default: Any = None) -> Any:
        if self._modified:
            return self._copy.get(key, default)  # type: ignore[union-attr]
        return self._source.get(key, default)

    def keys(self):
        if self._modified:
            return self._copy.keys()  # type: ignore[union-attr]
        return self._source.keys()

    def values(self):
        if self._modified:
            return self._copy.values()  # type: ignore[union-attr]
        return self._source.values()

    def items(self):
        if self._modified:
            return self._copy.items()  # type: ignore[union-attr]
        return self._source.items()

    def update(self, other: Dict[str, Any]) -> None:
        self._ensure_copy().update(other)

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self._ensure_copy()[key] = default
            return default
        return self[key]

    def pop(self, key: str, *args: Any) -> Any:
        return self._ensure_copy().pop(key, *args)

    def copy(self) -> Dict[str, Any]:
        if self._modified:
            return self._copy.copy()  # type: ignore[union-attr]
        return self._source.copy()  # type: ignore[return-value]

    def get_state(self) -> StateType:
        if self._modified:
            return self._copy  # type: ignore[return-value]
        return self._source

    @property
    def was_modified(self) -> bool:
        return self._modified

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.get_state())

    def __repr__(self) -> str:
        status = "modified" if self._modified else "unmodified"
        return f"CopyOnWriteState({status}, keys={list(self.keys())})"
