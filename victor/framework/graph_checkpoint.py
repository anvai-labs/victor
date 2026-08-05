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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class WorkflowCheckpoint:
    """Workflow checkpoint for StateGraph state persistence."""

    checkpoint_id: str
    thread_id: str
    node_id: str
    state: Dict[str, Any]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

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
        self._checkpoints[checkpoint.thread_id].append(checkpoint)

    async def load(self, thread_id: str) -> Optional[WorkflowCheckpoint]:
        checkpoints = self._checkpoints.get(thread_id, [])
        return checkpoints[-1] if checkpoints else None

    async def list(self, thread_id: str) -> List[WorkflowCheckpoint]:
        return self._checkpoints.get(thread_id, [])


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
