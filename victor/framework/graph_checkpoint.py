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
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# These names exist only in legacy dict-based graph states. Canonical agentic
# state is an immutable Pydantic value and receives collaborators explicitly.
_LEGACY_RUNTIME_HANDLE_KEYS = frozenset(
    {"_execution_context", "_execution_context_private", "_turn_record_channel"}
)
_MAX_TOLERANT_DEEPCOPY_DEPTH = 24


def deepcopy_preserving_runtime_handles(value: Any) -> Any:
    """Deep-copy a legacy dict graph state while keeping named handles live.

    This is compatibility support for graphs that still carry runtime handles in
    state. Canonical agentic state uses a shallow immutable-value copy and
    explicit collaborators instead. Handle names are explicit; recursion never
    guesses that an arbitrary lock, module, or socket is a runtime handle.
    """
    memo: Dict[int, Any] = {}

    def clone(obj: Any) -> Any:
        marker = id(obj)
        if marker in memo:
            return memo[marker]
        if obj is None or isinstance(obj, (str, bytes, int, float, bool, type)):
            return obj

        if isinstance(obj, dict):
            copied: Dict[Any, Any] = {}
            memo[marker] = copied
            for key, item in obj.items():
                copied[key] = item if key in _LEGACY_RUNTIME_HANDLE_KEYS else clone(item)
            return copied

        if isinstance(obj, list):
            copied_list: List[Any] = []
            memo[marker] = copied_list
            copied_list.extend(clone(item) for item in obj)
            return copied_list

        if isinstance(obj, tuple):
            return tuple(clone(item) for item in obj)
        if isinstance(obj, (set, frozenset)):
            try:
                return type(obj)(clone(item) for item in obj)
            except TypeError:
                # Frozen-set subclasses may enforce a hashable member contract.
                return obj

        if is_dataclass(obj) and not isinstance(obj, type):
            dataclass_values = {
                item.name: clone(getattr(obj, item.name))
                for item in fields(obj)
                if getattr(obj, item.name, None) is not obj
            }
            try:
                return type(obj)(**dataclass_values)
            except TypeError:
                # Dataclasses without a synthesized constructor are safe to
                # shallow-copy and mutate only when their fields are mutable.
                copied_dataclass = copy.copy(obj)
                memo[marker] = copied_dataclass
                for item in fields(obj):
                    object.__setattr__(copied_dataclass, item.name, dataclass_values[item.name])
                return copied_dataclass

        if isinstance(obj, BaseModel):
            copied_obj = copy.copy(obj)
            memo[marker] = copied_obj
            for key, item in vars(obj).items():
                value = item if key in _LEGACY_RUNTIME_HANDLE_KEYS else clone(item)
                try:
                    object.__setattr__(copied_obj, key, value)
                except TypeError:
                    setattr(copied_obj, key, value)
            private = getattr(obj, "__pydantic_private__", None)
            if isinstance(private, dict):
                copied_private = {
                    key: (value if key in _LEGACY_RUNTIME_HANDLE_KEYS else clone(value))
                    for key, value in private.items()
                }
                object.__setattr__(copied_obj, "__pydantic_private__", copied_private)
            return copied_obj

        attributes = getattr(obj, "__dict__", None)
        if isinstance(attributes, dict):
            copied_object = copy.copy(obj)
            memo[marker] = copied_object
            for key, item in attributes.items():
                value = item if key in _LEGACY_RUNTIME_HANDLE_KEYS else clone(item)
                try:
                    setattr(copied_object, key, value)
                except (AttributeError, TypeError) as exc:
                    raise ValueError(
                        f"Cannot safely clone immutable object {type(obj).__name__!r}"
                    ) from exc
            return copied_object

        # Modules, functions, locks, sockets, and similar leaves are not state
        # containers. Return them by reference; named runtime handles remain
        # live while accidental leaves cannot be silently replaced by repr().
        return obj

    return clone(value)


def strip_runtime_handles(value: Any, _seen: "Optional[Dict[int, Any]]" = None) -> Any:
    """Recursively drop legacy runtime handles before durable persistence.

    Older dict states stored a live RuntimeExecutionContext under
    context["_execution_context"]. Such handles may contain futures and
    service graphs and are never checkpoint data. Canonical agentic state has
    no such fields; this cleanup remains only so legacy generic states fail
    less mysteriously while strict JSON still rejects unknown live values.
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
            if key not in _LEGACY_RUNTIME_HANDLE_KEYS:
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
        try:
            copied = copy.copy(value)
        except Exception:
            # Un-copiable leaf (runtime context with futures/streams):
            # substitute its repr — the handle is not snapshot data anyway.
            return repr(value)
        _seen[id(value)] = copied
        stripped = {
            key: strip_runtime_handles(item, _seen)
            for key, item in value_dict.items()
            if key not in _LEGACY_RUNTIME_HANDLE_KEYS
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
        try:
            copied = copy.copy(value)
        except Exception:
            logger.debug("tolerant_deepcopy: substituted repr for %r", type(value))
            return repr(value)
        _seen[id(value)] = copied
        copied.__dict__.update(
            {key: tolerant_deepcopy(item, _depth + 1, _seen) for key, item in value_dict.items()}
        )
        return copied
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return type(value)(
                **{
                    key: tolerant_deepcopy(item, _depth + 1, _seen)
                    for key, item in value.__dict__.items()
                }
            )
        except Exception:
            logger.debug("tolerant_deepcopy: substituted repr for %r", type(value))
            return repr(value)
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
