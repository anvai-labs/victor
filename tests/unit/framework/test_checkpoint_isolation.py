# Copyright 2025 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""MemoryCheckpointer isolation guarantees.

Checkpoints must be snapshots, not aliases: the caller's live state keeps
mutating after save (graph_execution._bind_graph_checkpoint_id rewrites
state["context"] on every subsequent node), and callers mutate the resumed
state in place while the graph runs. Previously save stored the checkpoint
by reference and load returned the stored object, so earlier checkpoints
were retroactively rewritten through shared nested dicts (co-design review
U2-F5 / U6-F9).
"""

from __future__ import annotations

from victor.framework.graph_checkpoint import MemoryCheckpointer, WorkflowCheckpoint


def _checkpoint(node_id: str, counter: int) -> WorkflowCheckpoint:
    return WorkflowCheckpoint(
        checkpoint_id=f"t_{node_id}",
        thread_id="thread-1",
        node_id=node_id,
        state={"counter": counter, "context": {"nested": {"value": counter}}},
        timestamp=1_000.0 + counter,
    )


class TestCheckpointIsolation:
    async def test_saved_checkpoint_isolated_from_live_mutation(self):
        store = MemoryCheckpointer()
        cp = _checkpoint("node_a", 1)
        await store.save(cp)

        # Caller keeps mutating the same state dict after save.
        cp.state["counter"] = 999
        cp.state["context"]["nested"]["value"] = 999

        loaded = await store.load("thread-1")
        assert loaded is not None
        assert loaded.state["counter"] == 1
        assert loaded.state["context"]["nested"]["value"] == 1

    async def test_load_returns_copy(self):
        store = MemoryCheckpointer()
        await store.save(_checkpoint("node_a", 1))

        loaded = await store.load("thread-1")
        assert loaded is not None
        loaded.state["counter"] = 42
        loaded.state["context"]["nested"]["value"] = 42

        again = await store.load("thread-1")
        assert again is not None
        assert again.state["counter"] == 1
        assert again.state["context"]["nested"]["value"] == 1

    async def test_earlier_checkpoint_not_rewritten_by_later_save(self):
        store = MemoryCheckpointer()
        state = {"counter": 1, "context": {"nested": {"value": 1}}}
        await store.save(
            WorkflowCheckpoint(
                checkpoint_id="t_a",
                thread_id="thread-1",
                node_id="a",
                state=state,
                timestamp=1.0,
            )
        )
        # Same live dict saved again under a later node (aliasing scenario).
        state["context"]["nested"]["value"] = 2
        await store.save(
            WorkflowCheckpoint(
                checkpoint_id="t_b",
                thread_id="thread-1",
                node_id="b",
                state=state,
                timestamp=2.0,
            )
        )

        history = await store.list("thread-1")
        assert len(history) == 2
        assert history[0].state["context"]["nested"]["value"] == 1
        assert history[1].state["context"]["nested"]["value"] == 2


class TestListIsolation:
    """Negative from adversarial review: list() handed out the stored
    snapshots by reference — get_checkpoints/replay_from could corrupt the
    store through them."""

    async def test_list_returns_copies(self):
        store = MemoryCheckpointer()
        await store.save(_checkpoint("node_a", 1))

        listed = await store.list("thread-1")
        listed[0].state["counter"] = 777

        again = await store.list("thread-1")
        assert again[0].state["counter"] == 1
