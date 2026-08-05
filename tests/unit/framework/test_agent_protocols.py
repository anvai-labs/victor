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

"""Tests for agent protocols for multi-agent team coordination.

These tests follow TDD - written before implementation.
They verify the agent protocols that enable CrewAI-style multi-agent orchestration.
"""

import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock

# =============================================================================
# AgentCapability Enum Tests
# =============================================================================


class TestAgentCapability:
    """Tests for AgentCapability enum."""

    def test_capability_read_exists(self):
        """AgentCapability should have READ capability."""
        from victor.framework.agent_protocols import AgentCapability

        assert AgentCapability.READ is not None
        assert AgentCapability.READ.value == "read"

    def test_capability_write_exists(self):
        """AgentCapability should have WRITE capability."""
        from victor.framework.agent_protocols import AgentCapability

        assert AgentCapability.WRITE is not None
        assert AgentCapability.WRITE.value == "write"

    def test_capability_execute_exists(self):
        """AgentCapability should have EXECUTE capability."""
        from victor.framework.agent_protocols import AgentCapability

        assert AgentCapability.EXECUTE is not None
        assert AgentCapability.EXECUTE.value == "execute"

    def test_capability_search_exists(self):
        """AgentCapability should have SEARCH capability."""
        from victor.framework.agent_protocols import AgentCapability

        assert AgentCapability.SEARCH is not None
        assert AgentCapability.SEARCH.value == "search"

    def test_capability_communicate_exists(self):
        """AgentCapability should have COMMUNICATE capability."""
        from victor.framework.agent_protocols import AgentCapability

        assert AgentCapability.COMMUNICATE is not None
        assert AgentCapability.COMMUNICATE.value == "communicate"

    def test_capability_delegate_exists(self):
        """AgentCapability should have DELEGATE capability."""
        from victor.framework.agent_protocols import AgentCapability

        assert AgentCapability.DELEGATE is not None
        assert AgentCapability.DELEGATE.value == "delegate"

    def test_capability_approve_exists(self):
        """AgentCapability should have APPROVE capability."""
        from victor.framework.agent_protocols import AgentCapability

        assert AgentCapability.APPROVE is not None
        assert AgentCapability.APPROVE.value == "approve"

    def test_all_capabilities_count(self):
        """AgentCapability should have exactly 7 capabilities."""
        from victor.framework.agent_protocols import AgentCapability

        # Verify we have all expected capabilities
        expected = {
            "read",
            "write",
            "execute",
            "search",
            "communicate",
            "delegate",
            "approve",
        }
        actual = {cap.value for cap in AgentCapability}
        assert actual == expected


# =============================================================================
# AgentMessage Tests
# =============================================================================


class TestAgentMessage:
    """Tests for AgentMessage dataclass."""

    def test_message_creation(self):
        """AgentMessage should be created with required fields."""
        from victor.framework.agent_protocols import AgentMessage, MessageType

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

    def test_message_with_metadata(self):
        """AgentMessage should support metadata."""
        from victor.framework.agent_protocols import AgentMessage, MessageType

        msg = AgentMessage(
            sender_id="agent1",
            recipient_id="agent2",
            content="Hello",
            message_type=MessageType.RESULT,
            data={"key": "value"},
        )

        assert msg.metadata == {"key": "value"}

    def test_message_defaults(self):
        """AgentMessage should have sensible defaults."""
        from victor.framework.agent_protocols import AgentMessage, MessageType

        msg = AgentMessage(
            sender_id="agent1",
            recipient_id="agent2",
            content="Hello",
            message_type=MessageType.TASK,
        )

        assert msg.metadata == {}

    def test_message_type_task(self):
        """MessageType should have TASK type."""
        from victor.framework.agent_protocols import MessageType

        assert MessageType.TASK is not None
        assert MessageType.TASK.value == "task"

    def test_message_type_result(self):
        """MessageType should have RESULT type."""
        from victor.framework.agent_protocols import MessageType

        assert MessageType.RESULT is not None
        assert MessageType.RESULT.value == "result"

    def test_message_type_query(self):
        """MessageType should have QUERY type."""
        from victor.framework.agent_protocols import MessageType

        assert MessageType.QUERY is not None
        assert MessageType.QUERY.value == "query"

    def test_message_type_feedback(self):
        """MessageType should have FEEDBACK type."""
        from victor.framework.agent_protocols import MessageType

        assert MessageType.FEEDBACK is not None
        assert MessageType.FEEDBACK.value == "feedback"

    def test_message_type_delegation(self):
        """MessageType should have DELEGATION type."""
        from victor.framework.agent_protocols import MessageType

        assert MessageType.DELEGATION is not None
        assert MessageType.DELEGATION.value == "delegation"


# =============================================================================
# IAgentRole Protocol Tests
# =============================================================================


class TestIAgentRoleProtocol:
    """Tests for IAgentRole protocol compliance."""

    def test_role_has_name(self):
        """IAgentRole should have name property."""
        from victor.framework.agent_protocols import IAgentRole

        # Check protocol has 'name' attribute (works across Python versions)
        assert "name" in dir(IAgentRole) or hasattr(IAgentRole, "name")

    def test_role_has_capabilities(self):
        """IAgentRole should have capabilities property."""
        from victor.framework.agent_protocols import IAgentRole

        # Protocol should define capabilities
        assert "capabilities" in dir(IAgentRole) or hasattr(IAgentRole, "capabilities")

    def test_role_has_allowed_tools(self):
        """IAgentRole should have allowed_tools property."""
        from victor.framework.agent_protocols import IAgentRole

        # Protocol should define allowed_tools
        assert "allowed_tools" in dir(IAgentRole) or hasattr(IAgentRole, "allowed_tools")

    def test_role_has_tool_budget(self):
        """IAgentRole should have tool_budget property."""
        from victor.framework.agent_protocols import IAgentRole

        # Protocol should define tool_budget
        assert "tool_budget" in dir(IAgentRole) or hasattr(IAgentRole, "tool_budget")

    def test_role_has_get_system_prompt_section(self):
        """IAgentRole should have get_system_prompt_section method."""
        from victor.framework.agent_protocols import IAgentRole

        # Protocol should define get_system_prompt_section
        assert "get_system_prompt_section" in dir(IAgentRole) or hasattr(
            IAgentRole, "get_system_prompt_section"
        )

    def test_role_protocol_is_runtime_checkable(self):
        """IAgentRole should be runtime checkable."""
        from typing import runtime_checkable

        from victor.framework.agent_protocols import IAgentRole

        # Should be decorated with @runtime_checkable
        assert hasattr(IAgentRole, "_is_runtime_protocol")


# =============================================================================
# IAgentPersona Protocol Tests
# =============================================================================


class TestIAgentPersonaProtocol:
    """Tests for IAgentPersona protocol compliance."""

    def test_persona_has_name(self):
        """IAgentPersona should have name property."""
        from victor.framework.agent_protocols import IAgentPersona

        assert "name" in dir(IAgentPersona) or hasattr(IAgentPersona, "name")

    def test_persona_has_background(self):
        """IAgentPersona should have background property."""
        from victor.framework.agent_protocols import IAgentPersona

        assert "background" in dir(IAgentPersona) or hasattr(IAgentPersona, "background")

    def test_persona_has_communication_style(self):
        """IAgentPersona should have communication_style property."""
        from victor.framework.agent_protocols import IAgentPersona

        assert "communication_style" in dir(IAgentPersona) or hasattr(
            IAgentPersona, "communication_style"
        )

    def test_persona_has_format_message(self):
        """IAgentPersona should have format_message method."""
        from victor.framework.agent_protocols import IAgentPersona

        assert "format_message" in dir(IAgentPersona) or hasattr(IAgentPersona, "format_message")

    def test_persona_protocol_is_runtime_checkable(self):
        """IAgentPersona should be runtime checkable."""
        from victor.framework.agent_protocols import IAgentPersona

        assert hasattr(IAgentPersona, "_is_runtime_protocol")


# =============================================================================
# ITeamMember Protocol Tests
# =============================================================================


class TestITeamMemberProtocol:
    """Tests for ITeamMember protocol compliance."""

    def test_member_has_id(self):
        """ITeamMember should have id property."""
        from victor.framework.agent_protocols import ITeamMember

        assert "id" in dir(ITeamMember) or hasattr(ITeamMember, "id")

    def test_member_has_role(self):
        """ITeamMember should have role property."""
        from victor.framework.agent_protocols import ITeamMember

        assert "role" in dir(ITeamMember) or hasattr(ITeamMember, "role")

    def test_member_has_persona(self):
        """ITeamMember should have persona property."""
        from victor.framework.agent_protocols import ITeamMember

        assert "persona" in dir(ITeamMember) or hasattr(ITeamMember, "persona")

    def test_member_has_execute_task(self):
        """ITeamMember should have execute_task method."""
        from victor.framework.agent_protocols import ITeamMember

        assert "execute_task" in dir(ITeamMember) or hasattr(ITeamMember, "execute_task")

    def test_member_has_receive_message(self):
        """ITeamMember should have receive_message method."""
        from victor.framework.agent_protocols import ITeamMember

        assert "receive_message" in dir(ITeamMember) or hasattr(ITeamMember, "receive_message")

    def test_member_protocol_is_runtime_checkable(self):
        """ITeamMember should be runtime checkable."""
        from victor.framework.agent_protocols import ITeamMember

        assert hasattr(ITeamMember, "_is_runtime_protocol")


# =============================================================================
# ITeamCoordinator Protocol Tests
# =============================================================================


class TestITeamCoordinatorProtocol:
    """Tests for ITeamCoordinator protocol compliance."""

    def test_coordinator_has_add_member(self):
        """ITeamCoordinator should have add_member method."""
        from victor.framework.agent_protocols import ITeamCoordinator

        assert "add_member" in dir(ITeamCoordinator) or hasattr(ITeamCoordinator, "add_member")

    def test_coordinator_has_set_formation(self):
        """ITeamCoordinator should have set_formation method."""
        from victor.framework.agent_protocols import ITeamCoordinator

        assert "set_formation" in dir(ITeamCoordinator) or hasattr(
            ITeamCoordinator, "set_formation"
        )

    def test_coordinator_has_execute_task(self):
        """ITeamCoordinator should have execute_task method."""
        from victor.framework.agent_protocols import ITeamCoordinator

        assert "execute_task" in dir(ITeamCoordinator) or hasattr(ITeamCoordinator, "execute_task")

    def test_coordinator_has_broadcast(self):
        """ITeamCoordinator should have broadcast method."""
        from victor.framework.agent_protocols import ITeamCoordinator

        assert "broadcast" in dir(ITeamCoordinator) or hasattr(ITeamCoordinator, "broadcast")

    def test_coordinator_protocol_is_runtime_checkable(self):
        """ITeamCoordinator should be runtime checkable."""
        from victor.framework.agent_protocols import ITeamCoordinator

        assert hasattr(ITeamCoordinator, "_is_runtime_protocol")


class TestIDelegateFollowUpCoordinatorProtocol:
    """Tests for delegate follow-up coordinator capability protocol."""

    def test_follow_up_protocol_has_request_execution(self):
        """IDelegateFollowUpCoordinator should expose request execution."""
        from victor.framework.agent_protocols import IDelegateFollowUpCoordinator

        assert "execute_follow_up_request" in dir(IDelegateFollowUpCoordinator) or hasattr(
            IDelegateFollowUpCoordinator, "execute_follow_up_request"
        )

    def test_follow_up_protocol_has_contract_execution(self):
        """IDelegateFollowUpCoordinator should expose contract execution."""
        from victor.framework.agent_protocols import IDelegateFollowUpCoordinator

        assert "execute_follow_up_contract" in dir(IDelegateFollowUpCoordinator) or hasattr(
            IDelegateFollowUpCoordinator, "execute_follow_up_contract"
        )

    def test_follow_up_protocol_is_runtime_checkable(self):
        """IDelegateFollowUpCoordinator should be runtime checkable."""
        from victor.framework.agent_protocols import IDelegateFollowUpCoordinator

        assert hasattr(IDelegateFollowUpCoordinator, "_is_runtime_protocol")


# =============================================================================
# TeamFormation Enum Tests
# =============================================================================


class TestTeamFormationEnum:
    """Tests for TeamFormation enum."""

    def test_formation_sequential_exists(self):
        """TeamFormation should have SEQUENTIAL value."""
        from victor.framework.agent_protocols import TeamFormation

        assert TeamFormation.SEQUENTIAL is not None
        assert TeamFormation.SEQUENTIAL.value == "sequential"

    def test_formation_parallel_exists(self):
        """TeamFormation should have PARALLEL value."""
        from victor.framework.agent_protocols import TeamFormation

        assert TeamFormation.PARALLEL is not None
        assert TeamFormation.PARALLEL.value == "parallel"

    def test_formation_hierarchical_exists(self):
        """TeamFormation should have HIERARCHICAL value."""
        from victor.framework.agent_protocols import TeamFormation

        assert TeamFormation.HIERARCHICAL is not None
        assert TeamFormation.HIERARCHICAL.value == "hierarchical"

    def test_formation_pipeline_exists(self):
        """TeamFormation should have PIPELINE value."""
        from victor.framework.agent_protocols import TeamFormation

        assert TeamFormation.PIPELINE is not None
        assert TeamFormation.PIPELINE.value == "pipeline"

    def test_formation_consensus_exists(self):
        """TeamFormation should have CONSENSUS value."""
        from victor.framework.agent_protocols import TeamFormation

        assert TeamFormation.CONSENSUS is not None
        assert TeamFormation.CONSENSUS.value == "consensus"

    def test_formation_reflection_exists(self):
        """TeamFormation should have REFLECTION value."""
        from victor.framework.agent_protocols import TeamFormation

        assert TeamFormation.REFLECTION is not None
        assert TeamFormation.REFLECTION.value == "reflection"

    def test_all_formations_count(self):
        """TeamFormation should have exactly 6 formations."""
        from victor.framework.agent_protocols import TeamFormation

        expected = {
            "sequential",
            "parallel",
            "hierarchical",
            "pipeline",
            "consensus",
            "reflection",
        }
        actual = {f.value for f in TeamFormation}
        assert actual == expected


# =============================================================================
# Protocol Implementation Compliance Tests
# =============================================================================


class TestProtocolImplementationCompliance:
    """Tests that verify mock implementations can satisfy protocols."""

    def test_role_implementation_satisfies_protocol(self):
        """A dataclass with correct attributes should satisfy IAgentRole."""
        from victor.framework.agent_protocols import AgentCapability, IAgentRole

        @dataclass
        class MockRole:
            name: str = "mock_role"
            capabilities: Set[AgentCapability] = None
            allowed_tools: Set[str] = None
            tool_budget: int = 10

            def __post_init__(self):
                if self.capabilities is None:
                    self.capabilities = {AgentCapability.READ}
                if self.allowed_tools is None:
                    self.allowed_tools = {"read_file"}

            def get_system_prompt_section(self) -> str:
                return f"You are a {self.name}."

        role = MockRole()
        assert isinstance(role, IAgentRole)

    def test_persona_implementation_satisfies_protocol(self):
        """A dataclass with correct attributes should satisfy IAgentPersona."""
        from victor.framework.agent_protocols import IAgentPersona

        @dataclass
        class MockPersona:
            name: str = "Expert"
            background: str = "10 years experience"
            communication_style: str = "Professional and concise"

            def format_message(self, content: str) -> str:
                return f"[{self.name}] {content}"

        persona = MockPersona()
        assert isinstance(persona, IAgentPersona)

    def test_team_member_implementation_satisfies_protocol(self):
        """A class with correct methods should satisfy ITeamMember."""
        from victor.framework.agent_protocols import (
            AgentCapability,
            AgentMessage,
            IAgentPersona,
            IAgentRole,
            ITeamMember,
            MessageType,
        )

        @dataclass
        class MockRole:
            name: str = "mock_role"
            capabilities: Set[AgentCapability] = None
            allowed_tools: Set[str] = None
            tool_budget: int = 10

            def __post_init__(self):
                if self.capabilities is None:
                    self.capabilities = set()
                if self.allowed_tools is None:
                    self.allowed_tools = set()

            def get_system_prompt_section(self) -> str:
                return ""

        @dataclass
        class MockPersona:
            name: str = "Expert"
            background: str = ""
            communication_style: str = ""

            def format_message(self, content: str) -> str:
                return content

        class MockTeamMember:
            def __init__(self):
                self.id = "member_1"
                self.role = MockRole()
                self.persona = MockPersona()

            async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
                return f"Executed: {task}"

            async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
                return AgentMessage(
                    sender_id=self.id,
                    recipient_id=message.sender_id,
                    content="Received",
                    message_type=MessageType.RESULT,
                )

        member = MockTeamMember()
        assert isinstance(member, ITeamMember)


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module-level exports."""

    def test_agent_protocols_exports_capability(self):
        """agent_protocols should export AgentCapability."""
        from victor.framework.agent_protocols import AgentCapability

        assert AgentCapability is not None

    def test_agent_protocols_exports_message(self):
        """agent_protocols should export AgentMessage."""
        from victor.framework.agent_protocols import AgentMessage

        assert AgentMessage is not None

    def test_agent_protocols_exports_message_type(self):
        """agent_protocols should export MessageType."""
        from victor.framework.agent_protocols import MessageType

        assert MessageType is not None

    def test_agent_protocols_exports_role_protocol(self):
        """agent_protocols should export IAgentRole."""
        from victor.framework.agent_protocols import IAgentRole

        assert IAgentRole is not None

    def test_agent_protocols_exports_persona_protocol(self):
        """agent_protocols should export IAgentPersona."""
        from victor.framework.agent_protocols import IAgentPersona

        assert IAgentPersona is not None

    def test_agent_protocols_exports_member_protocol(self):
        """agent_protocols should export ITeamMember."""
        from victor.framework.agent_protocols import ITeamMember

        assert ITeamMember is not None

    def test_agent_protocols_exports_coordinator_protocol(self):
        """agent_protocols should export ITeamCoordinator."""
        from victor.framework.agent_protocols import ITeamCoordinator

        assert ITeamCoordinator is not None

    def test_agent_protocols_exports_formation(self):
        """agent_protocols should export TeamFormation."""
        from victor.framework.agent_protocols import TeamFormation

        assert TeamFormation is not None
