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

"""Checkpointing primitives for StateGraph execution."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Protocol

_MAX_TOLERANT_DEEPCOPY_DEPTH = 24


import logging

logger = logging.getLogger(__name__)

_RUNTIME_HANDLE_KEYS = frozenset({"_execution_context"})
_MAX_TOLERANT_DEEPCOPY_DEPTH = 24


def deepcopy_preserving_runtime_handles(value: Any) -> Any:
    """Deep-copy *value* while keeping runtime handles live (FOU-D5f-S6c).

    ``load_initial_state`` deep-copies the caller's input state so the graph
    works on a private copy — but that state legitimately carries the LIVE
    RuntimeExecutionContext under ``context["_execution_context"]`` (nodes
    reach services through it), and deep-copying the handle fails on its
    futures/tasks. Iterative + memoized: deep and cyclic graphs are safe.
    The handles are lifted out, the remainder deep-copied, and the live
    references restored into the copy.
    """
    memo: Dict[int, Any] = {}
    handles: List[Any] = []  # (path_tuple, live_object)
    stack: List[Any] = [("enter", value, None, None, ())]
    while stack:
        kind, obj, parent, key, path = stack.pop()
        oid = id(obj)
        if oid in memo:
            if parent is not None:
                if isinstance(parent, dict):
                    parent[key] = memo[oid]
                elif isinstance(parent, list):
                    parent[key] = memo[oid]
                else:
                    setattr(parent, key, memo[oid])
            continue
        if isinstance(obj, dict):
            placeholder: Dict[Any, Any] = {}
            memo[oid] = placeholder
            if parent is not None:
                if isinstance(parent, dict):
                    parent[key] = placeholder
                elif isinstance(parent, list):
                    parent[key] = placeholder
                else:
                    setattr(parent, key, placeholder)
            for child_key, child in obj.items():
                if child_key in _RUNTIME_HANDLE_KEYS:
                    handles.append((path + (child_key,), child))
                    continue
                stack.append(("enter", child, placeholder, child_key, path + (child_key,)))
            continue
        if isinstance(obj, list):
            placeholder_list: List[Any] = []
            memo[oid] = placeholder_list
            if parent is not None:
                if isinstance(parent, dict):
                    parent[key] = placeholder_list
                elif isinstance(parent, list):
                    parent[key] = placeholder_list
                else:
                    setattr(parent, key, placeholder_list)
            for index, child in enumerate(obj):
                stack.append(("enter", child, placeholder_list, index, path + (index,)))
            continue
        value_dict = getattr(obj, "__dict__", None)
        if isinstance(value_dict, dict):
            try:
                copied = copy.copy(obj)
            except Exception:
                # Un-copiable leaf (e.g. TextIOWrapper): substitute its repr.
                logger.debug("tolerant_deepcopy: substituted repr for %r", type(obj))
                memo[oid] = repr(obj)
                if parent is not None:
                    if isinstance(parent, dict):
                        parent[key] = memo[oid]
                    elif isinstance(parent, list):
                        parent[key] = memo[oid]
                    else:
                        setattr(parent, key, memo[oid])
                continue
            memo[oid] = copied
            if parent is not None:
                if isinstance(parent, dict):
                    parent[key] = copied
                elif isinstance(parent, list):
                    parent[key] = copied
                else:
                    setattr(parent, key, copied)
            priv = getattr(obj, "__pydantic_private__", None)
            copied_priv = dict(priv) if isinstance(priv, dict) else None
            for child_key, child in value_dict.items():
                if child_key in _RUNTIME_HANDLE_KEYS:
                    handles.append((path + (child_key,), child))
                    continue
                stack.append(("enter", child, copied, child_key, path + (child_key,)))
            if copied_priv is not None:
                for priv_key, priv_val in list(copied_priv.items()):
                    if priv_key in _RUNTIME_HANDLE_KEYS:
                        handles.append((path + (priv_key,), priv_val))
                        copied_priv.pop(priv_key, None)
                    else:
                        stack.append(("enter", priv_val, copied_priv, priv_key, path + (priv_key,)))
            continue
        handles.append((path, obj))
        memo[oid] = obj
        if parent is not None:
            if isinstance(parent, dict):
                parent[key] = obj
            elif isinstance(parent, list):
                parent[key] = obj
            else:
                setattr(parent, key, obj)
    copied_root = memo.get(id(value), value)
    for path, original in handles:
        try:
            target = copied_root
            for key in path[:-1]:
                candidate = (
                    target.get(key) if isinstance(target, dict) else getattr(target, key, None)
                )
                if candidate is None:
                    candidate = (getattr(target, "__pydantic_private__", None) or {}).get(key)
                if candidate is None:
                    raise KeyError(key)
                target = candidate
            last = path[-1]
            if isinstance(target, dict):
                target[last] = original
            else:
                object.__setattr__(target, last, original)
        except (KeyError, IndexError, TypeError, AttributeError):
            logger.debug(
                "deepcopy_preserving_runtime_handles: could not restore handle "
                "at %r (copy shape changed)",
                path,
            )
    return copied_root


def strip_runtime_handles(value: Any, _seen: "Optional[Dict[int, Any]]" = None) -> Any:
    """Recursively drop runtime execution handles from copyable structures.

    ``inject_execution_context`` stores the live RuntimeExecutionContext under
    ``context["_execution_context"]`` so nodes can reach services. That handle
    carries futures and service graphs: deep-copying it (result snapshots,
    MemoryCheckpointer.save) fails with "cannot pickle _asyncio.Future" and
    kills the run (FOU-D5f-S6c). Runtime handles are never snapshot data.
    Cycle-safe: already-visited containers are shared by reference (the
    downstream deepcopy resolves shared references through its own memo).
    """
    if _seen is None:
        _seen = {}
    seen_marker = _seen.get(id(value))
    if seen_marker is not None:
        return seen_marker
    if isinstance(value, dict):
        copied = {}
        _seen[id(value)] = copied
        for key, item in value.items():
            if key not in _RUNTIME_HANDLE_KEYS:
                copied[key] = strip_runtime_handles(item, _seen)
        return copied
    if isinstance(value, list):
        copied = []
        _seen[id(value)] = copied
        for item in value:
            copied.append(strip_runtime_handles(item, _seen))
        return copied
    if isinstance(value, tuple):
        return tuple(strip_runtime_handles(item, _seen) for item in value)
    # Descend into objects (checkpoints, state models, namespaces) so
    # shielded handles never survive a strip.
    value_dict = getattr(value, "__dict__", None)
    if isinstance(value_dict, dict):
        copied = copy.copy(value)
        _seen[id(value)] = copied
        stripped = {
            key: strip_runtime_handles(item, _seen)
            for key, item in value_dict.items()
            if key not in _RUNTIME_HANDLE_KEYS
        }
        copied.__dict__.update(stripped)
        return copied
    return value


def tolerant_deepcopy(value: Any, _depth: int = 0, _seen: "Optional[Dict[int, Any]]" = None) -> Any:
    """Deep-copy that never dies on unpicklable leaves or cycles (FOU-D5f-S6c).

    Live graph states carry runtime objects (futures, tasks, execution
    contexts, cyclic service graphs). Plain ``copy.deepcopy`` raises
    "cannot pickle ..." (or hits recursion limits on cycles) and kills the
    run at the first checkpoint save or result snapshot. Containers are
    walked with a memo; uncopiable leaves are substituted with ``repr()``.
    """
    if _seen is None:
        _seen = {}
    seen_marker = _seen.get(id(value))
    if seen_marker is not None:
        return seen_marker
    if _depth > _MAX_TOLERANT_DEEPCOPY_DEPTH:
        return repr(value)
    if isinstance(value, dict):
        copied = {}
        _seen[id(value)] = copied
        for key, item in value.items():
            copied[key] = tolerant_deepcopy(item, _depth + 1, _seen)
        return copied
    if isinstance(value, list):
        copied = []
        _seen[id(value)] = copied
        for item in value:
            copied.append(tolerant_deepcopy(item, _depth + 1, _seen))
        return copied
    if isinstance(value, tuple):
        return tuple(tolerant_deepcopy(item, _depth + 1, _seen) for item in value)
    if isinstance(value, set):
        copied = set()
        _seen[id(value)] = copied
        for item in value:
            copied.add(tolerant_deepcopy(item, _depth + 1, _seen))
        return copied
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    value_dict = getattr(value, "__dict__", None)
    if isinstance(value_dict, dict):
        copied = copy.copy(value)
        _seen[id(value)] = copied
        copied.__dict__.update(
            {key: tolerant_deepcopy(item, _depth + 1, _seen) for key, item in value_dict.items()}
        )
        return copied
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                key: tolerant_deepcopy(item, _depth + 1, _seen)
                for key, item in value.__dict__.items()
            }
        )
    try:
        return copy.deepcopy(value)
    except Exception:
        logger.debug("tolerant_deepcopy: substituted repr for %r", type(value))
        return repr(value)


@dataclass
class WorkflowCheckpoint:
    """Workflow checkpoint for StateGraph state persistence."""

    checkpoint_id: str
    thread_id: str
    node_id: str
    state: Dict[str, Any]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def next_node(self) -> Optional[str]:
        """Node a resume continues at, when the checkpoint recorded one (FEP-0032).

        Stored in ``metadata`` so every checkpointer backend persists it unchanged.
        ``node_id`` keeps naming the node the checkpoint was taken at.
        """
        value = self.metadata.get("next_node")
        return value if isinstance(value, str) else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "thread_id": self.thread_id,
            "node_id": self.node_id,
            "state": self.state,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowCheckpoint":
        return cls(
            checkpoint_id=data["checkpoint_id"],
            thread_id=data["thread_id"],
            node_id=data["node_id"],
            state=data["state"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
        )


class CheckpointerProtocol(Protocol):
    """Protocol for checkpoint persistence."""

    async def save(self, checkpoint: WorkflowCheckpoint) -> None: ...

    async def load(self, thread_id: str) -> Optional[WorkflowCheckpoint]: ...

    async def list(self, thread_id: str) -> List[WorkflowCheckpoint]: ...


class MemoryCheckpointer:
    """In-memory checkpoint storage for development and tests."""

    def __init__(self):
        self._checkpoints: Dict[str, List[WorkflowCheckpoint]] = {}

    async def save(self, checkpoint: WorkflowCheckpoint) -> None:
        if checkpoint.thread_id not in self._checkpoints:
            self._checkpoints[checkpoint.thread_id] = []
        # Snapshot at save: the caller's live state keeps mutating after this
        # point (e.g. _bind_graph_checkpoint_id rewrites state["context"] on
        # every subsequent node), so by-reference storage would retroactively
        # rewrite earlier checkpoints through shared nested dicts.
        try:
            snapshot = copy.deepcopy(checkpoint)
        except Exception:
            # Live states carry runtime objects (futures, tasks, execution
            # contexts). Strip the documented handles first (they are never
            # snapshot data), then tolerate anything else that refuses to
            # copy — keep the checkpoint instead of killing the run
            # (FOU-D5f-S6c).
            snapshot = tolerant_deepcopy(strip_runtime_handles(checkpoint))
        self._checkpoints[checkpoint.thread_id].append(snapshot)

    async def load(self, thread_id: str) -> Optional[WorkflowCheckpoint]:
        checkpoints = self._checkpoints.get(thread_id, [])
        # Copy on load too: callers mutate the resumed state in place.
        return copy.deepcopy(checkpoints[-1]) if checkpoints else None

    async def list(self, thread_id: str) -> List[WorkflowCheckpoint]:
        # Copies: callers (get_checkpoints, replay_from) must not be able to
        # mutate the stored snapshots through the returned list
        # (adversarial-review finding — save/load were isolated, list was not).
        return [copy.deepcopy(c) for c in self._checkpoints.get(thread_id, [])]


class RLCheckpointerAdapter:
    """Adapter to bridge graph checkpointing to the RL checkpoint store."""

    def __init__(self, learner_name: str = "state_graph"):
        self.learner_name = learner_name
        self._store = None

    def _get_store(self):
        if self._store is None:
            from victor.framework.rl.checkpoint_store import get_checkpoint_store

            self._store = get_checkpoint_store()
        return self._store

    async def save(self, checkpoint: WorkflowCheckpoint) -> None:
        store = self._get_store()
        store.create_checkpoint(
            learner_name=f"{self.learner_name}_{checkpoint.thread_id}",
            version=checkpoint.checkpoint_id,
            state={
                "node_id": checkpoint.node_id,
                "state": checkpoint.state,
                "timestamp": checkpoint.timestamp,
            },
            metadata=checkpoint.metadata,
        )

    async def load(self, thread_id: str) -> Optional[WorkflowCheckpoint]:
        store = self._get_store()
        policy_cp = store.get_latest_checkpoint(f"{self.learner_name}_{thread_id}")
        if not policy_cp:
            return None

        return WorkflowCheckpoint(
            checkpoint_id=policy_cp.version,
            thread_id=thread_id,
            node_id=policy_cp.state.get("node_id", ""),
            state=policy_cp.state.get("state", {}),
            timestamp=policy_cp.state.get("timestamp", 0.0),
            metadata=policy_cp.metadata,
        )

    async def list(self, thread_id: str) -> List[WorkflowCheckpoint]:
        store = self._get_store()
        policy_cps = store.list_checkpoints(f"{self.learner_name}_{thread_id}")
        return [
            WorkflowCheckpoint(
                checkpoint_id=cp.version,
                thread_id=thread_id,
                node_id=cp.state.get("node_id", ""),
                state=cp.state.get("state", {}),
                timestamp=cp.state.get("timestamp", 0.0),
                metadata=cp.metadata,
            )
            for cp in policy_cps
        ]
