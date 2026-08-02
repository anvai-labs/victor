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

"""Tests for parallel formation context isolation.

Verifies that agents in parallel formation receive isolated state
copies and cannot interfere with each other's execution.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from victor.coordination.formations.base import TeamContext
from victor.coordination.formations.parallel import ParallelFormation
from victor.teams.types import AgentMessage, MemberResult, MessageType


def _make_agent(agent_id: str, side_effect=None):
    """Create a mock agent with the given ID."""
    agent = MagicMock()
    agent.id = agent_id
    if side_effect:
        agent.execute = AsyncMock(side_effect=side_effect)
    else:
        agent.execute = AsyncMock(
            return_value=MemberResult(
                member_id=agent_id,
                success=True,
                output=f"result-{agent_id}",
            )
        )
    return agent


def _make_task():
    return AgentMessage(
        sender_id="coordinator",
        content="test task",
        message_type=MessageType.TASK,
    )


class TestParallelContextIsolation:
    """Test that parallel agents get isolated contexts."""

    async def test_agents_receive_independent_copies(self):
        """Each agent should get its own copy of shared_state."""
        captured_contexts = []

        async def capture_context(task, context):
            # Record the context's shared_state id
            captured_contexts.append(id(context.shared_state))
            return MemberResult(member_id="test", success=True, output="ok")

        agents = [_make_agent(f"agent-{i}") for i in range(3)]
        for agent in agents:
            agent.execute = AsyncMock(side_effect=capture_context)

        context = TeamContext(
            "team-1",
            "parallel",
            shared_state={"data": [1, 2, 3], "nested": {"key": "value"}},
        )

        formation = ParallelFormation()
        await formation.execute(agents, context, _make_task())

        # All contexts should have different shared_state objects
        assert len(set(captured_contexts)) == 3, "Agents should get independent copies"

    async def test_agent_mutation_does_not_affect_others(self):
        """Mutation by one agent should not be visible to others during execution."""
        seen_values = []

        async def mutating_agent(task, context):
            # Record current state before mutation
            seen_values.append(context.shared_state.get("counter", 0))
            # Mutate context — should only affect this copy
            context.set("counter", 999)
            await asyncio.sleep(0.01)  # Yield control
            return MemberResult(member_id="test", success=True, output="ok")

        agents = [_make_agent(f"agent-{i}") for i in range(5)]
        for agent in agents:
            agent.execute = AsyncMock(side_effect=mutating_agent)

        context = TeamContext("team-1", "parallel", shared_state={"counter": 0})

        formation = ParallelFormation()
        await formation.execute(agents, context, _make_task())

        # All agents should have seen counter=0 (not 999 from another agent)
        assert all(v == 0 for v in seen_values), f"Agents saw mutated state: {seen_values}"

    async def test_nested_dict_isolation(self):
        """Deep nested structures should be isolated between agents."""
        seen_nested = []

        async def nested_mutator(task, context):
            seen_nested.append(context.shared_state["nested"]["key"])
            context.shared_state["nested"]["key"] = "mutated"
            await asyncio.sleep(0.01)
            return MemberResult(member_id="test", success=True, output="ok")

        agents = [_make_agent(f"agent-{i}") for i in range(3)]
        for agent in agents:
            agent.execute = AsyncMock(side_effect=nested_mutator)

        context = TeamContext(
            "team-1",
            "parallel",
            shared_state={"nested": {"key": "original"}},
        )

        formation = ParallelFormation()
        await formation.execute(agents, context, _make_task())

        # All agents should have seen "original" (deep copy isolation)
        assert all(v == "original" for v in seen_nested)

    async def test_context_merge_after_execution(self):
        """Agent state changes should be merged back after completion."""

        async def make_writer(agent_id):
            async def writer_agent(task, context):
                context.set(f"result_{agent_id}", f"value_{agent_id}")
                return MemberResult(member_id=agent_id, success=True, output="ok")

            return writer_agent

        agents = [_make_agent(f"agent-{i}") for i in range(3)]
        for i, agent in enumerate(agents):
            aid = f"agent-{i}"

            async def _side_effect(t, c, _aid=aid):
                c.set(f"result_{_aid}", f"value_{_aid}")
                return MemberResult(member_id=_aid, success=True, output="ok")

            agent.execute = AsyncMock(side_effect=_side_effect)

        context = TeamContext("team-1", "parallel", shared_state={})

        formation = ParallelFormation()
        await formation.execute(agents, context, _make_task())

        # All agent results should be merged back
        for i in range(3):
            assert context.shared_state.get(f"result_agent-{i}") == f"value_agent-{i}"


class TestTeamContextThreadSafety:
    """Test that TeamContext set/update are thread-safe."""

    async def test_concurrent_set(self):
        """Concurrent set operations should not corrupt state."""
        context = TeamContext("team-1", "parallel", shared_state={})

        def setter(i: int):
            context.set(f"key_{i}", i)

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(setter, i) for i in range(50)]
            for f in futures:
                f.result()

        # All 50 keys should be present
        assert len(context.shared_state) == 50

    async def test_concurrent_update(self):
        """Concurrent update operations should not corrupt state."""
        context = TeamContext("team-1", "parallel", shared_state={})

        def updater(i: int):
            context.update({f"batch_{i}_a": i, f"batch_{i}_b": i + 1})

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(updater, i) for i in range(25)]
            for f in futures:
                f.result()

        assert len(context.shared_state) == 50
