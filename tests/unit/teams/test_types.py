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

"""Tests for canonical team types."""

import pytest

from victor.teams.types import (
    AgentMessage,
    MemberResult,
    MessagePriority,
    MessageType,
    TeamFormation,
    TeamResult,
)


class TestTeamFormation:
    """Tests for TeamFormation enum."""

    def test_all_formations_defined(self):
        """All 5 formations should be available."""
        assert TeamFormation.SEQUENTIAL
        assert TeamFormation.PARALLEL
        assert TeamFormation.HIERARCHICAL
        assert TeamFormation.PIPELINE
        assert TeamFormation.CONSENSUS

    def test_formation_values(self):
        """Formations should have lowercase string values."""
        assert TeamFormation.SEQUENTIAL.value == "sequential"
        assert TeamFormation.PARALLEL.value == "parallel"
        assert TeamFormation.HIERARCHICAL.value == "hierarchical"
        assert TeamFormation.PIPELINE.value == "pipeline"
        assert TeamFormation.CONSENSUS.value == "consensus"

    def test_formation_is_string_enum(self):
        """TeamFormation should be a string enum."""
        assert isinstance(TeamFormation.SEQUENTIAL, str)
        assert TeamFormation.SEQUENTIAL == "sequential"


class TestMessageType:
    """Tests for MessageType enum."""

    def test_framework_types_exist(self):
        """Types from framework/agent_protocols should exist."""
        assert MessageType.TASK
        assert MessageType.RESULT
        assert MessageType.QUERY
        assert MessageType.FEEDBACK
        assert MessageType.DELEGATION

    def test_agent_teams_types_exist(self):
        """Types from agent/teams/communication should exist."""
        assert MessageType.DISCOVERY
        assert MessageType.REQUEST
        assert MessageType.RESPONSE
        assert MessageType.STATUS
        assert MessageType.ALERT
        assert MessageType.HANDOFF

    def test_message_type_values(self):
        """Message types should have lowercase string values."""
        assert MessageType.TASK.value == "task"
        assert MessageType.HANDOFF.value == "handoff"


class TestAgentMessage:
    """Tests for AgentMessage dataclass."""

    def test_message_creation(self):
        """AgentMessage should be creatable with required fields."""
        msg = AgentMessage(
            sender_id="agent1",
            recipient_id="agent2",
            content="Hello",
            message_type=MessageType.TASK,
        )
        assert msg.sender_id == "agent1"
        assert msg.recipient_id == "agent2"
        assert msg.content == "Hello"
        assert msg.message_type == MessageType.TASK

    def test_message_auto_generated_fields(self):
        """AgentMessage should auto-generate id and timestamp."""
        msg = AgentMessage(
            sender_id="agent1",
            recipient_id=None,
            content="Test",
            message_type=MessageType.STATUS,
        )
        assert msg.id  # Auto-generated
        assert len(msg.id) == 12
        assert msg.timestamp > 0

    def test_message_unique_ids(self):
        """Each message should have a unique ID."""
        msg1 = AgentMessage(
            sender_id="a",
            recipient_id="b",
            content="1",
            message_type=MessageType.TASK,
        )
        msg2 = AgentMessage(
            sender_id="a",
            recipient_id="b",
            content="2",
            message_type=MessageType.TASK,
        )
        assert msg1.id != msg2.id

    def test_broadcast_message(self):
        """Broadcast messages have recipient_id=None."""
        msg = AgentMessage(
            sender_id="agent1",
            recipient_id=None,
            content="Broadcast",
            message_type=MessageType.STATUS,
        )
        assert msg.is_broadcast()
        assert msg.recipient_id is None

    def test_reply_message(self):
        """Reply messages have reply_to set."""
        original = AgentMessage(
            sender_id="a",
            recipient_id="b",
            content="Question",
            message_type=MessageType.QUERY,
        )
        reply = AgentMessage(
            sender_id="b",
            recipient_id="a",
            content="Answer",
            message_type=MessageType.RESPONSE,
            reply_to=original.id,
        )
        assert reply.is_reply()
        assert reply.reply_to == original.id

    def test_compatibility_aliases(self):
        """Compatibility aliases should work."""
        msg = AgentMessage(
            sender_id="agent1",
            recipient_id="agent2",
            content="Test",
            message_type=MessageType.TASK,
            data={"key": "value"},
        )
        # Framework aliases
        assert msg.metadata == {"key": "value"}
        # Agent/teams aliases
        assert msg.from_agent == "agent1"
        assert msg.to_agent == "agent2"
        assert msg.type == MessageType.TASK


class TestMemberResult:
    """Tests for MemberResult dataclass."""

    def test_simple_result(self):
        """MemberResult should work with minimal fields."""
        result = MemberResult(
            member_id="member1",
            success=True,
            output="Done",
        )
        assert result.member_id == "member1"
        assert result.success is True
        assert result.output == "Done"
        assert result.error is None

    def test_failed_result(self):
        """MemberResult should capture errors."""
        result = MemberResult(
            member_id="member1",
            success=False,
            output="",
            error="Task failed",
        )
        assert result.success is False
        assert result.error == "Task failed"

    def test_rich_result(self):
        """MemberResult should capture metrics."""
        result = MemberResult(
            member_id="member1",
            success=True,
            output="Completed",
            tool_calls_used=5,
            duration_seconds=10.5,
            discoveries=["Found bug", "Fixed issue"],
        )
        assert result.tool_calls_used == 5
        assert result.duration_seconds == 10.5
        assert len(result.discoveries) == 2


class TestTeamResult:
    """Tests for TeamResult dataclass."""

    def test_team_result_creation(self):
        """TeamResult should be creatable with member results."""
        member_result = MemberResult(
            member_id="m1",
            success=True,
            output="Done",
        )
        result = TeamResult(
            success=True,
            final_output="Team completed",
            member_results={"m1": member_result},
            formation=TeamFormation.SEQUENTIAL,
        )
        assert result.success is True
        assert result.final_output == "Team completed"
        assert "m1" in result.member_results
        assert result.formation == TeamFormation.SEQUENTIAL

    def test_consensus_result(self):
        """TeamResult should track consensus metadata."""
        result = TeamResult(
            success=True,
            final_output="Agreed",
            member_results={},
            formation=TeamFormation.CONSENSUS,
            consensus_achieved=True,
            consensus_rounds=2,
        )
        assert result.consensus_achieved is True
        assert result.consensus_rounds == 2


class TestMessagePriority:
    """Tests for MessagePriority enum."""

    def test_priority_ordering(self):
        """Priority should be orderable."""
        assert MessagePriority.LOW < MessagePriority.NORMAL
        assert MessagePriority.NORMAL < MessagePriority.HIGH
        assert MessagePriority.HIGH < MessagePriority.URGENT

    def test_priority_values(self):
        """Priority should have integer values."""
        assert MessagePriority.LOW.value == 0
        assert MessagePriority.NORMAL.value == 1
        assert MessagePriority.HIGH.value == 2
        assert MessagePriority.URGENT.value == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
