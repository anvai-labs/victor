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

"""Integration tests for agent roles and capability enforcement.

These tests verify that agent roles properly enforce:
- Capability restrictions based on role
- Tool budget limits per role
- Persona formatting of messages
- Role-based system prompt generation

Tests exercise the real implementations in:
- victor/framework/agent_roles.py
- victor/framework/personas.py
- victor/framework/agent_protocols.py
- victor/framework/team_coordinator.py
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import pytest

from victor.framework.agent_protocols import (
    AgentCapability,
    AgentMessage,
    IAgentRole,
    MessageType,
    TeamFormation,
)
from victor.framework.agent_roles import (
    ExecutorRole,
    ManagerRole,
    ResearcherRole,
    ReviewerRole,
    ROLE_REGISTRY,
    get_role,
)
from victor.framework.personas import (
    Persona,
    PERSONA_REGISTRY,
    get_persona,
    list_personas,
    register_persona,
)
from victor.teams import UnifiedTeamCoordinator

# =============================================================================
# Test Agent Implementation with Capability Enforcement
# =============================================================================


class CapabilityAwareAgent:
    """Agent that enforces role-based capabilities.

    This agent tracks capability checks and enforces tool budget limits.
    """

    def __init__(
        self,
        agent_id: str,
        role: IAgentRole,
        persona: Optional[Persona] = None,
    ):
        self._id = agent_id
        self._role = role
        self._persona = persona

        # Tracking
        self.tool_calls_made: int = 0
        self.capability_checks: List[tuple] = []
        self.received_messages: List[AgentMessage] = []
        self.executed_tasks: List[str] = []

    @property
    def id(self) -> str:
        return self._id

    @property
    def role(self) -> IAgentRole:
        return self._role

    @property
    def persona(self) -> Optional[Persona]:
        return self._persona

    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if this agent has a specific capability."""
        result = capability in self._role.capabilities
        self.capability_checks.append((capability, result))
        return result

    def can_use_tool(self, tool_name: str) -> bool:
        """Check if this agent can use a specific tool."""
        return tool_name in self._role.allowed_tools

    def is_within_budget(self) -> bool:
        """Check if agent is within tool budget."""
        return self.tool_calls_made < self._role.tool_budget

    def use_tool(self, tool_name: str) -> bool:
        """Attempt to use a tool, respecting budget and permissions."""
        if not self.can_use_tool(tool_name):
            return False
        if not self.is_within_budget():
            return False
        self.tool_calls_made += 1
        return True

    async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
        """Execute a task using allowed capabilities."""
        self.executed_tasks.append(task)

        # Simulate using tools based on role capabilities
        output_parts = []

        if self.has_capability(AgentCapability.READ):
            if self.use_tool("read"):
                output_parts.append("Read files")

        if self.has_capability(AgentCapability.SEARCH):
            if self.can_use_tool("semantic_search") and self.use_tool("semantic_search"):
                output_parts.append("Searched codebase")

        if self.has_capability(AgentCapability.WRITE):
            if self.can_use_tool("write") and self.use_tool("write"):
                output_parts.append("Wrote files")

        return (
            f"{self._id}: {', '.join(output_parts)}" if output_parts else f"{self._id}: No actions"
        )

    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Receive and respond to a message."""
        self.received_messages.append(message)

        content = f"Acknowledged from {self.id}"
        if self._persona:
            content = self._persona.format_message(content)

        return AgentMessage(
            sender_id=self.id,
            recipient_id=message.sender_id,
            content=content,
            message_type=MessageType.RESULT,
        )


# =============================================================================
# Role Definition Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestRoleDefinitions:
    """Tests for role definitions and properties."""

    def test_manager_role_has_delegate_capability(self):
        """ManagerRole has DELEGATE capability."""
        role = ManagerRole()
        assert AgentCapability.DELEGATE in role.capabilities

    def test_manager_role_has_approve_capability(self):
        """ManagerRole has APPROVE capability."""
        role = ManagerRole()
        assert AgentCapability.APPROVE in role.capabilities

    def test_researcher_role_has_search_capability(self):
        """ResearcherRole has SEARCH capability."""
        role = ResearcherRole()
        assert AgentCapability.SEARCH in role.capabilities

    def test_researcher_role_lacks_write_capability(self):
        """ResearcherRole does not have WRITE capability."""
        role = ResearcherRole()
        assert AgentCapability.WRITE not in role.capabilities

    def test_executor_role_has_write_capability(self):
        """ExecutorRole has WRITE capability."""
        role = ExecutorRole()
        assert AgentCapability.WRITE in role.capabilities

    def test_executor_role_has_execute_capability(self):
        """ExecutorRole has EXECUTE capability."""
        role = ExecutorRole()
        assert AgentCapability.EXECUTE in role.capabilities

    def test_reviewer_role_has_approve_capability(self):
        """ReviewerRole has APPROVE capability."""
        role = ReviewerRole()
        assert AgentCapability.APPROVE in role.capabilities

    def test_all_roles_have_communicate_capability(self):
        """All roles have COMMUNICATE capability."""
        for role_class in [ManagerRole, ResearcherRole, ExecutorRole, ReviewerRole]:
            role = role_class()
            assert AgentCapability.COMMUNICATE in role.capabilities


# =============================================================================
# Capability Restriction Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestCapabilityRestrictions:
    """Tests for role-based capability restrictions."""

    @pytest.mark.asyncio
    async def test_researcher_cannot_write_files(self):
        """Researcher agent cannot write files."""
        agent = CapabilityAwareAgent("researcher", ResearcherRole())

        # Attempt to use write capability
        can_write = agent.has_capability(AgentCapability.WRITE)
        assert can_write is False

    @pytest.mark.asyncio
    async def test_researcher_can_search(self):
        """Researcher agent can search."""
        agent = CapabilityAwareAgent("researcher", ResearcherRole())

        can_search = agent.has_capability(AgentCapability.SEARCH)
        assert can_search is True

    @pytest.mark.asyncio
    async def test_executor_can_write_and_execute(self):
        """Executor agent can write and execute."""
        agent = CapabilityAwareAgent("executor", ExecutorRole())

        assert agent.has_capability(AgentCapability.WRITE) is True
        assert agent.has_capability(AgentCapability.EXECUTE) is True

    @pytest.mark.asyncio
    async def test_executor_cannot_delegate(self):
        """Executor agent cannot delegate."""
        agent = CapabilityAwareAgent("executor", ExecutorRole())

        assert agent.has_capability(AgentCapability.DELEGATE) is False

    @pytest.mark.asyncio
    async def test_manager_cannot_execute_commands(self):
        """Manager agent cannot execute commands."""
        agent = CapabilityAwareAgent("manager", ManagerRole())

        assert agent.has_capability(AgentCapability.EXECUTE) is False

    @pytest.mark.asyncio
    async def test_reviewer_cannot_write_directly(self):
        """Reviewer agent cannot write files directly."""
        agent = CapabilityAwareAgent("reviewer", ReviewerRole())

        assert agent.has_capability(AgentCapability.WRITE) is False

    @pytest.mark.asyncio
    async def test_capability_checks_are_tracked(self):
        """Capability checks are tracked for auditing."""
        agent = CapabilityAwareAgent("researcher", ResearcherRole())

        agent.has_capability(AgentCapability.READ)
        agent.has_capability(AgentCapability.WRITE)
        agent.has_capability(AgentCapability.SEARCH)

        assert len(agent.capability_checks) == 3
        assert agent.capability_checks[0] == (AgentCapability.READ, True)
        assert agent.capability_checks[1] == (AgentCapability.WRITE, False)
        assert agent.capability_checks[2] == (AgentCapability.SEARCH, True)


# =============================================================================
# Tool Budget Enforcement Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestToolBudgetEnforcement:
    """Tests for tool budget enforcement per role."""

    def test_manager_tool_budget(self):
        """Manager has specific tool budget."""
        role = ManagerRole()
        assert role.tool_budget == 20

    def test_researcher_tool_budget(self):
        """Researcher has higher tool budget."""
        role = ResearcherRole()
        assert role.tool_budget == 25

    def test_executor_tool_budget(self):
        """Executor has highest tool budget."""
        role = ExecutorRole()
        assert role.tool_budget == 30

    def test_reviewer_tool_budget(self):
        """Reviewer has moderate tool budget."""
        role = ReviewerRole()
        assert role.tool_budget == 20

    def test_agent_respects_tool_budget(self):
        """Agent respects tool budget limit."""

        # Create agent with small budget role
        @dataclass
        class SmallBudgetRole:
            name: str = "small_budget"
            capabilities: Set[AgentCapability] = field(
                default_factory=lambda: {AgentCapability.READ}
            )
            allowed_tools: Set[str] = field(default_factory=lambda: {"read_file"})
            tool_budget: int = 3

            def get_system_prompt_section(self) -> str:
                return "Small budget role"

        agent = CapabilityAwareAgent("limited", SmallBudgetRole())

        # Make tool calls up to budget
        for _ in range(3):
            result = agent.use_tool("read_file")
            assert result is True

        # Next call should fail
        result = agent.use_tool("read_file")
        assert result is False

    def test_agent_tracks_tool_usage(self):
        """Agent tracks number of tool calls made."""
        agent = CapabilityAwareAgent("researcher", ResearcherRole())

        agent.use_tool("read")
        agent.use_tool("grep")
        agent.use_tool("semantic_search")

        assert agent.tool_calls_made == 3


# =============================================================================
# Tool Access Restriction Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestToolAccessRestrictions:
    """Tests for tool access restrictions by role."""

    def test_manager_allowed_tools(self):
        """Manager has limited tool access."""
        role = ManagerRole()

        # Manager can use these
        assert "read" in role.allowed_tools
        assert "ls" in role.allowed_tools
        assert "task_complete" in role.allowed_tools

        # Manager cannot use these
        assert "write" not in role.allowed_tools
        assert "shell" not in role.allowed_tools

    def test_researcher_allowed_tools(self):
        """Researcher has search-focused tools."""
        role = ResearcherRole()

        # Researcher can use search tools
        assert "semantic_search" in role.allowed_tools
        assert "code_search" in role.allowed_tools
        assert "web_search" in role.allowed_tools
        assert "grep" in role.allowed_tools

        # Researcher cannot modify files
        assert "write" not in role.allowed_tools
        assert "edit" not in role.allowed_tools

    def test_executor_allowed_tools(self):
        """Executor has full modification tools."""
        role = ExecutorRole()

        # Executor can modify
        assert "write" in role.allowed_tools
        assert "edit" in role.allowed_tools
        assert "delete_file" in role.allowed_tools
        assert "shell" in role.allowed_tools

    def test_reviewer_allowed_tools(self):
        """Reviewer has read and review tools."""
        role = ReviewerRole()

        # Reviewer can read and search
        assert "read" in role.allowed_tools
        assert "git" in role.allowed_tools
        assert "code_search" in role.allowed_tools

        # Reviewer cannot modify
        assert "write" not in role.allowed_tools

    def test_agent_cannot_use_disallowed_tool(self):
        """Agent cannot use tools not in allowed list."""
        agent = CapabilityAwareAgent("researcher", ResearcherRole())

        # Researcher cannot use write
        result = agent.use_tool("write")
        assert result is False
        assert agent.tool_calls_made == 0  # Didn't count


# =============================================================================
# Persona Formatting Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestPersonaFormatting:
    """Tests for persona message formatting."""

    def test_formal_persona_formatting(self):
        """Formal persona capitalizes and punctuates."""
        persona = Persona(
            name="Formal Agent",
            background="Test",
            communication_style="formal",
        )

        result = persona.format_message("hello there")
        assert result == "Hello there."

    def test_casual_persona_formatting(self):
        """Casual persona lowercases."""
        persona = Persona(
            name="Casual Agent",
            background="Test",
            communication_style="casual",
        )

        result = persona.format_message("HELLO THERE")
        assert result == "hello there"

    def test_professional_persona_preserves_content(self):
        """Professional persona preserves original."""
        persona = Persona(
            name="Professional Agent",
            background="Test",
            communication_style="professional",
        )

        result = persona.format_message("Hello There")
        assert result == "Hello There"

    @pytest.mark.asyncio
    async def test_persona_formats_agent_responses(self):
        """Persona formats agent message responses."""
        formal_persona = Persona(
            name="Formal",
            background="Test",
            communication_style="formal",
        )

        agent = CapabilityAwareAgent(
            "formal_agent",
            ResearcherRole(),
            persona=formal_persona,
        )

        message = AgentMessage(
            sender_id="coordinator",
            recipient_id="formal_agent",
            content="hello",
            message_type=MessageType.QUERY,
        )

        response = await agent.receive_message(message)

        # Response should be formatted formally
        assert response.content[0].isupper()  # Capitalized
        assert response.content.endswith(".")  # Punctuated

    def test_builtin_personas_exist(self):
        """Built-in personas are available."""
        personas = list_personas()

        assert "friendly_assistant" in personas
        assert "senior_developer" in personas
        assert "code_reviewer" in personas
        assert "mentor" in personas

    def test_get_builtin_persona(self):
        """Can retrieve built-in personas."""
        senior_dev = get_persona("senior_developer")

        assert senior_dev is not None
        assert senior_dev.name == "Senior Developer"
        assert "professional" in senior_dev.communication_style

    def test_register_custom_persona(self):
        """Can register custom personas."""
        custom = Persona(
            name="Custom Expert",
            background="Custom background",
            communication_style="technical",
            expertise_areas=("custom", "testing"),
        )

        register_persona("custom_expert", custom)

        retrieved = get_persona("custom_expert")
        assert retrieved is not None
        assert retrieved.name == "Custom Expert"


# =============================================================================
# System Prompt Generation Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestSystemPromptGeneration:
    """Tests for role and persona system prompt generation."""

    def test_manager_system_prompt(self):
        """Manager role generates appropriate prompt."""
        role = ManagerRole()
        prompt = role.get_system_prompt_section()

        # ManagerRole is now a compatibility alias for SupervisorRole; assert the
        # role's supervision/coordination vocabulary rather than a literal name.
        assert "supervisor" in prompt.lower()
        assert any(term in prompt.lower() for term in ("coordinat", "delegate", "supervis"))

    def test_researcher_system_prompt(self):
        """Researcher role generates search-focused prompt."""
        role = ResearcherRole()
        prompt = role.get_system_prompt_section()

        assert "Researcher" in prompt
        assert "search" in prompt.lower() or "analyze" in prompt.lower()

    def test_executor_system_prompt(self):
        """Executor role generates implementation-focused prompt."""
        role = ExecutorRole()
        prompt = role.get_system_prompt_section()

        assert "Executor" in prompt
        assert "implement" in prompt.lower() or "write" in prompt.lower()

    def test_reviewer_system_prompt(self):
        """Reviewer role generates review-focused prompt."""
        role = ReviewerRole()
        prompt = role.get_system_prompt_section()

        assert "Reviewer" in prompt
        assert "review" in prompt.lower() or "quality" in prompt.lower()

    def test_persona_system_prompt(self):
        """Persona generates system prompt section."""
        persona = Persona(
            name="Test Expert",
            background="Expert in testing",
            communication_style="professional",
            expertise_areas=("testing", "qa", "automation"),
            quirks=("thorough", "detail-oriented"),
        )

        prompt = persona.get_system_prompt_section()

        assert "Test Expert" in prompt
        assert "Expert in testing" in prompt
        assert "professional" in prompt
        assert "testing" in prompt
        assert "thorough" in prompt


# =============================================================================
# Role Registry Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestRoleRegistry:
    """Tests for role registry functionality."""

    def test_get_role_by_name(self):
        """Can retrieve roles by name."""
        manager = get_role("manager")
        researcher = get_role("researcher")
        executor = get_role("executor")
        reviewer = get_role("reviewer")

        assert manager is not None
        assert researcher is not None
        assert executor is not None
        assert reviewer is not None

    def test_get_role_case_insensitive(self):
        """Role lookup is case insensitive."""
        manager1 = get_role("manager")
        manager2 = get_role("MANAGER")
        manager3 = get_role("Manager")

        assert manager1 is not None
        assert manager2 is not None
        assert manager3 is not None

    def test_get_nonexistent_role(self):
        """Getting nonexistent role returns None."""
        role = get_role("nonexistent_role")
        assert role is None

    def test_role_registry_contains_all_roles(self):
        """Role registry contains all expected roles."""
        assert "manager" in ROLE_REGISTRY
        assert "researcher" in ROLE_REGISTRY
        assert "executor" in ROLE_REGISTRY
        assert "reviewer" in ROLE_REGISTRY


# =============================================================================
# Integration with Team Coordinator Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestRolesWithTeamCoordinator:
    """Tests for roles integrated with team coordinator."""

    @pytest.mark.asyncio
    async def test_team_with_different_roles(self):
        """Team with different roles executes correctly."""
        coordinator = UnifiedTeamCoordinator()

        manager = CapabilityAwareAgent("manager", ManagerRole())
        researcher = CapabilityAwareAgent("researcher", ResearcherRole())
        executor = CapabilityAwareAgent("executor", ExecutorRole())
        reviewer = CapabilityAwareAgent("reviewer", ReviewerRole())

        coordinator.add_member(manager)
        coordinator.add_member(researcher)
        coordinator.add_member(executor)
        coordinator.add_member(reviewer)

        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        result = await coordinator.execute_task("Complete feature", {})

        assert result["success"] is True
        assert len(result["member_results"]) == 4

    @pytest.mark.asyncio
    async def test_hierarchical_team_role_awareness(self):
        """Hierarchical team respects role-based manager selection."""
        coordinator = UnifiedTeamCoordinator()

        # Worker added first
        worker = CapabilityAwareAgent("worker", ExecutorRole())
        # Manager added second
        manager = CapabilityAwareAgent("manager", ManagerRole())

        coordinator.add_member(worker)
        coordinator.add_member(manager)
        coordinator.set_formation(TeamFormation.HIERARCHICAL)

        result = await coordinator.execute_task("Test task", {})

        assert result["success"] is True
        # Both should have executed
        assert len(worker.executed_tasks) == 1
        assert len(manager.executed_tasks) == 1

    @pytest.mark.asyncio
    async def test_pipeline_with_specialized_roles(self):
        """Pipeline with specialized roles passes context."""
        coordinator = UnifiedTeamCoordinator()

        researcher = CapabilityAwareAgent("researcher", ResearcherRole())
        executor = CapabilityAwareAgent("executor", ExecutorRole())
        reviewer = CapabilityAwareAgent("reviewer", ReviewerRole())

        coordinator.add_member(researcher)
        coordinator.add_member(executor)
        coordinator.add_member(reviewer)

        coordinator.set_formation(TeamFormation.PIPELINE)

        result = await coordinator.execute_task("Implement feature", {})

        assert result["success"] is True
        assert "final_output" in result

    @pytest.mark.asyncio
    async def test_role_capabilities_preserved_in_team(self):
        """Role capabilities are preserved when used in team."""
        coordinator = UnifiedTeamCoordinator()

        researcher = CapabilityAwareAgent("researcher", ResearcherRole())
        executor = CapabilityAwareAgent("executor", ExecutorRole())

        coordinator.add_member(researcher)
        coordinator.add_member(executor)

        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        await coordinator.execute_task("Test capabilities", {})

        # Verify capabilities were checked during execution
        # Researcher should have checked READ and SEARCH
        researcher_read_check = any(
            cap == AgentCapability.READ for cap, _ in researcher.capability_checks
        )
        researcher_search_check = any(
            cap == AgentCapability.SEARCH for cap, _ in researcher.capability_checks
        )
        assert researcher_read_check or researcher_search_check

        # Executor should have checked WRITE
        executor_write_check = any(
            cap == AgentCapability.WRITE for cap, _ in executor.capability_checks
        )
        assert executor_write_check


# =============================================================================
# Combined Role and Persona Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestRoleAndPersonaCombination:
    """Tests combining roles with personas."""

    @pytest.mark.asyncio
    async def test_agent_with_role_and_persona(self):
        """Agent can have both role and persona."""
        persona = get_persona("senior_developer")
        agent = CapabilityAwareAgent(
            "senior_researcher",
            ResearcherRole(),
            persona=persona,
        )

        # Role defines capabilities
        assert agent.has_capability(AgentCapability.SEARCH)
        assert not agent.has_capability(AgentCapability.WRITE)

        # Persona is available
        assert agent.persona is not None
        assert agent.persona.name == "Senior Developer"

    @pytest.mark.asyncio
    async def test_persona_affects_message_formatting(self):
        """Persona affects how agent formats messages."""
        formal_persona = Persona(
            name="Formal",
            background="Test",
            communication_style="formal",
        )

        agent = CapabilityAwareAgent(
            "formal_researcher",
            ResearcherRole(),
            persona=formal_persona,
        )

        message = AgentMessage(
            sender_id="coordinator",
            recipient_id="formal_researcher",
            content="test",
            message_type=MessageType.QUERY,
        )

        response = await agent.receive_message(message)

        # Response formatted by persona
        assert response.content[0].isupper()

    @pytest.mark.asyncio
    async def test_team_with_roles_and_personas(self):
        """Team members can have both roles and personas."""
        coordinator = UnifiedTeamCoordinator()

        agent1 = CapabilityAwareAgent(
            "senior_researcher",
            ResearcherRole(),
            persona=get_persona("senior_developer"),
        )
        agent2 = CapabilityAwareAgent(
            "code_reviewer",
            ReviewerRole(),
            persona=get_persona("code_reviewer"),
        )

        coordinator.add_member(agent1)
        coordinator.add_member(agent2)

        coordinator.set_formation(TeamFormation.PIPELINE)

        result = await coordinator.execute_task("Review code", {})

        assert result["success"] is True


# =============================================================================
# Role Immutability Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestRoleImmutability:
    """Tests verifying role properties are consistent."""

    def test_role_instances_are_independent(self):
        """Multiple role instances are independent."""
        role1 = ResearcherRole()
        role2 = ResearcherRole()

        # They should have same values but be different objects
        assert role1.name == role2.name
        assert role1.capabilities == role2.capabilities
        assert role1.tool_budget == role2.tool_budget

    def test_role_capabilities_are_sets(self):
        """Role capabilities are proper sets."""
        for role_class in [ManagerRole, ResearcherRole, ExecutorRole, ReviewerRole]:
            role = role_class()
            assert isinstance(role.capabilities, set)

    def test_role_allowed_tools_are_sets(self):
        """Role allowed tools are proper sets."""
        for role_class in [ManagerRole, ResearcherRole, ExecutorRole, ReviewerRole]:
            role = role_class()
            assert isinstance(role.allowed_tools, set)
