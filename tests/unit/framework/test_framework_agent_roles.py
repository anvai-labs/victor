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

"""Tests for built-in agent roles.

These tests follow TDD - written before implementation.
They verify the pre-defined roles for multi-agent team coordination.
"""

import pytest
from typing import Set

# =============================================================================
# ManagerRole Tests
# =============================================================================


class TestSupervisorRole:
    """Tests for the canonical SupervisorRole."""

    def test_supervisor_role_exists(self):
        """SupervisorRole should be importable."""
        from victor.framework.agent_roles import SupervisorRole

        assert SupervisorRole is not None

    def test_supervisor_role_name(self):
        """SupervisorRole should have name 'supervisor'."""
        from victor.framework.agent_roles import SupervisorRole

        role = SupervisorRole()
        assert role.name == "supervisor"

    def test_supervisor_role_has_delegate_capability(self):
        """SupervisorRole should have DELEGATE capability."""
        from victor.framework.agent_protocols import AgentCapability
        from victor.framework.agent_roles import SupervisorRole

        role = SupervisorRole()
        assert AgentCapability.DELEGATE in role.capabilities


class TestManagerRole:
    """Tests for ManagerRole."""

    def test_manager_role_exists(self):
        """ManagerRole should be importable."""
        from victor.framework.agent_roles import ManagerRole

        assert ManagerRole is not None

    def test_manager_role_name(self):
        """ManagerRole should have name 'manager'."""
        from victor.framework.agent_roles import ManagerRole

        role = ManagerRole()
        assert role.name == "manager"

    def test_manager_role_has_delegate_capability(self):
        """ManagerRole should have DELEGATE capability."""
        from victor.framework.agent_protocols import AgentCapability
        from victor.framework.agent_roles import ManagerRole

        role = ManagerRole()
        assert AgentCapability.DELEGATE in role.capabilities

    def test_manager_role_has_communicate_capability(self):
        """ManagerRole should have COMMUNICATE capability."""
        from victor.framework.agent_protocols import AgentCapability
        from victor.framework.agent_roles import ManagerRole

        role = ManagerRole()
        assert AgentCapability.COMMUNICATE in role.capabilities

    def test_manager_role_has_approve_capability(self):
        """ManagerRole should have APPROVE capability."""
        from victor.framework.agent_protocols import AgentCapability
        from victor.framework.agent_roles import ManagerRole

        role = ManagerRole()
        assert AgentCapability.APPROVE in role.capabilities

    def test_manager_role_tool_budget(self):
        """ManagerRole should have appropriate tool budget."""
        from victor.framework.agent_roles import ManagerRole

        role = ManagerRole()
        assert role.tool_budget >= 15

    def test_manager_role_allowed_tools(self):
        """ManagerRole should have appropriate allowed tools."""
        from victor.framework.agent_roles import ManagerRole

        role = ManagerRole()
        assert isinstance(role.allowed_tools, set)
        # Manager should have limited direct file access tools

    def test_manager_role_system_prompt_section(self):
        """ManagerRole should provide system prompt section."""
        from victor.framework.agent_roles import ManagerRole

        role = ManagerRole()
        prompt_section = role.get_system_prompt_section()
        assert isinstance(prompt_section, str)
        assert len(prompt_section) > 0
        assert "manager" in prompt_section.lower() or "coordinate" in prompt_section.lower()

    def test_manager_role_satisfies_protocol(self):
        """ManagerRole should satisfy IAgentRole protocol."""
        from victor.framework.agent_protocols import IAgentRole
        from victor.framework.agent_roles import ManagerRole

        role = ManagerRole()
        assert isinstance(role, IAgentRole)


# =============================================================================
# ResearcherRole Tests
# =============================================================================


class TestResearcherRole:
    """Tests for ResearcherRole."""

    def test_researcher_role_exists(self):
        """ResearcherRole should be importable."""
        from victor.framework.agent_roles import ResearcherRole

        assert ResearcherRole is not None

    def test_researcher_role_name(self):
        """ResearcherRole should have name 'researcher'."""
        from victor.framework.agent_roles import ResearcherRole

        role = ResearcherRole()
        assert role.name == "researcher"

    def test_researcher_role_has_read_capability(self):
        """ResearcherRole should have READ capability."""
        from victor.framework.agent_protocols import AgentCapability
        from victor.framework.agent_roles import ResearcherRole

        role = ResearcherRole()
        assert AgentCapability.READ in role.capabilities

    def test_researcher_role_has_search_capability(self):
        """ResearcherRole should have SEARCH capability."""
        from victor.framework.agent_protocols import AgentCapability
        from victor.framework.agent_roles import ResearcherRole

        role = ResearcherRole()
        assert AgentCapability.SEARCH in role.capabilities

    def test_researcher_role_does_not_have_write_capability(self):
        """ResearcherRole should NOT have WRITE capability by default."""
        from victor.framework.agent_protocols import AgentCapability
        from victor.framework.agent_roles import ResearcherRole

        role = ResearcherRole()
        assert AgentCapability.WRITE not in role.capabilities

    def test_researcher_role_tool_budget(self):
        """ResearcherRole should have appropriate tool budget."""
        from victor.framework.agent_roles import ResearcherRole

        role = ResearcherRole()
        assert role.tool_budget >= 20  # Researchers need to explore

    def test_researcher_role_allowed_tools_includes_read(self):
        """ResearcherRole allowed_tools should include read-related tools."""
        from victor.framework.agent_roles import ResearcherRole

        role = ResearcherRole()
        # Should include at least some search/read tools
        assert isinstance(role.allowed_tools, set)

    def test_researcher_role_system_prompt_section(self):
        """ResearcherRole should provide system prompt section."""
        from victor.framework.agent_roles import ResearcherRole

        role = ResearcherRole()
        prompt_section = role.get_system_prompt_section()
        assert isinstance(prompt_section, str)
        assert len(prompt_section) > 0
        assert "research" in prompt_section.lower() or "analyze" in prompt_section.lower()

    def test_researcher_role_satisfies_protocol(self):
        """ResearcherRole should satisfy IAgentRole protocol."""
        from victor.framework.agent_protocols import IAgentRole
        from victor.framework.agent_roles import ResearcherRole

        role = ResearcherRole()
        assert isinstance(role, IAgentRole)


# =============================================================================
# ExecutorRole Tests
# =============================================================================


class TestExecutorRole:
    """Tests for ExecutorRole."""

    def test_executor_role_exists(self):
        """ExecutorRole should be importable."""
        from victor.framework.agent_roles import ExecutorRole

        assert ExecutorRole is not None

    def test_executor_role_name(self):
        """ExecutorRole should have name 'executor'."""
        from victor.framework.agent_roles import ExecutorRole

        role = ExecutorRole()
        assert role.name == "executor"

    def test_executor_role_has_write_capability(self):
        """ExecutorRole should have WRITE capability."""
        from victor.framework.agent_protocols import AgentCapability
        from victor.framework.agent_roles import ExecutorRole

        role = ExecutorRole()
        assert AgentCapability.WRITE in role.capabilities

    def test_executor_role_has_execute_capability(self):
        """ExecutorRole should have EXECUTE capability."""
        from victor.framework.agent_protocols import AgentCapability
        from victor.framework.agent_roles import ExecutorRole

        role = ExecutorRole()
        assert AgentCapability.EXECUTE in role.capabilities

    def test_executor_role_has_read_capability(self):
        """ExecutorRole should have READ capability."""
        from victor.framework.agent_protocols import AgentCapability
        from victor.framework.agent_roles import ExecutorRole

        role = ExecutorRole()
        assert AgentCapability.READ in role.capabilities

    def test_executor_role_tool_budget(self):
        """ExecutorRole should have appropriate tool budget."""
        from victor.framework.agent_roles import ExecutorRole

        role = ExecutorRole()
        assert role.tool_budget >= 25  # Executors need many tools

    def test_executor_role_allowed_tools(self):
        """ExecutorRole allowed_tools should include write-related tools."""
        from victor.framework.agent_roles import ExecutorRole

        role = ExecutorRole()
        assert isinstance(role.allowed_tools, set)

    def test_executor_role_system_prompt_section(self):
        """ExecutorRole should provide system prompt section."""
        from victor.framework.agent_roles import ExecutorRole

        role = ExecutorRole()
        prompt_section = role.get_system_prompt_section()
        assert isinstance(prompt_section, str)
        assert len(prompt_section) > 0
        assert "implement" in prompt_section.lower() or "execute" in prompt_section.lower()

    def test_executor_role_satisfies_protocol(self):
        """ExecutorRole should satisfy IAgentRole protocol."""
        from victor.framework.agent_protocols import IAgentRole
        from victor.framework.agent_roles import ExecutorRole

        role = ExecutorRole()
        assert isinstance(role, IAgentRole)


# =============================================================================
# ReviewerRole Tests
# =============================================================================


class TestReviewerRole:
    """Tests for ReviewerRole."""

    def test_reviewer_role_exists(self):
        """ReviewerRole should be importable."""
        from victor.framework.agent_roles import ReviewerRole

        assert ReviewerRole is not None

    def test_reviewer_role_name(self):
        """ReviewerRole should have name 'reviewer'."""
        from victor.framework.agent_roles import ReviewerRole

        role = ReviewerRole()
        assert role.name == "reviewer"

    def test_reviewer_role_has_approve_capability(self):
        """ReviewerRole should have APPROVE capability."""
        from victor.framework.agent_protocols import AgentCapability
        from victor.framework.agent_roles import ReviewerRole

        role = ReviewerRole()
        assert AgentCapability.APPROVE in role.capabilities

    def test_reviewer_role_has_read_capability(self):
        """ReviewerRole should have READ capability."""
        from victor.framework.agent_protocols import AgentCapability
        from victor.framework.agent_roles import ReviewerRole

        role = ReviewerRole()
        assert AgentCapability.READ in role.capabilities

    def test_reviewer_role_has_communicate_capability(self):
        """ReviewerRole should have COMMUNICATE capability."""
        from victor.framework.agent_protocols import AgentCapability
        from victor.framework.agent_roles import ReviewerRole

        role = ReviewerRole()
        assert AgentCapability.COMMUNICATE in role.capabilities

    def test_reviewer_role_tool_budget(self):
        """ReviewerRole should have appropriate tool budget."""
        from victor.framework.agent_roles import ReviewerRole

        role = ReviewerRole()
        assert role.tool_budget >= 15

    def test_reviewer_role_allowed_tools(self):
        """ReviewerRole allowed_tools should include review-related tools."""
        from victor.framework.agent_roles import ReviewerRole

        role = ReviewerRole()
        assert isinstance(role.allowed_tools, set)

    def test_reviewer_role_system_prompt_section(self):
        """ReviewerRole should provide system prompt section."""
        from victor.framework.agent_roles import ReviewerRole

        role = ReviewerRole()
        prompt_section = role.get_system_prompt_section()
        assert isinstance(prompt_section, str)
        assert len(prompt_section) > 0
        assert "review" in prompt_section.lower() or "quality" in prompt_section.lower()

    def test_reviewer_role_satisfies_protocol(self):
        """ReviewerRole should satisfy IAgentRole protocol."""
        from victor.framework.agent_protocols import IAgentRole
        from victor.framework.agent_roles import ReviewerRole

        role = ReviewerRole()
        assert isinstance(role, IAgentRole)


# =============================================================================
# Role Registry Tests
# =============================================================================


class TestRoleRegistry:
    """Tests for role registry and get_role function."""

    def test_role_registry_exists(self):
        """ROLE_REGISTRY should be importable."""
        from victor.framework.agent_roles import ROLE_REGISTRY

        assert ROLE_REGISTRY is not None
        assert isinstance(ROLE_REGISTRY, dict)

    def test_registry_contains_manager(self):
        """ROLE_REGISTRY should contain 'manager'."""
        from victor.framework.agent_roles import ROLE_REGISTRY

        assert "manager" in ROLE_REGISTRY

    def test_registry_contains_supervisor(self):
        """ROLE_REGISTRY should contain 'supervisor'."""
        from victor.framework.agent_roles import ROLE_REGISTRY

        assert "supervisor" in ROLE_REGISTRY

    def test_registry_contains_researcher(self):
        """ROLE_REGISTRY should contain 'researcher'."""
        from victor.framework.agent_roles import ROLE_REGISTRY

        assert "researcher" in ROLE_REGISTRY

    def test_registry_contains_executor(self):
        """ROLE_REGISTRY should contain 'executor'."""
        from victor.framework.agent_roles import ROLE_REGISTRY

        assert "executor" in ROLE_REGISTRY

    def test_registry_contains_reviewer(self):
        """ROLE_REGISTRY should contain 'reviewer'."""
        from victor.framework.agent_roles import ROLE_REGISTRY

        assert "reviewer" in ROLE_REGISTRY

    def test_get_role_function_exists(self):
        """get_role function should be importable."""
        from victor.framework.agent_roles import get_role

        assert callable(get_role)

    def test_get_role_returns_manager(self):
        """get_role('manager') should return ManagerRole instance."""
        from victor.framework.agent_protocols import IAgentRole
        from victor.framework.agent_roles import ManagerRole, get_role

        role = get_role("manager")
        assert role is not None
        assert isinstance(role, ManagerRole)
        assert isinstance(role, IAgentRole)

    def test_get_role_returns_supervisor(self):
        """get_role('supervisor') should return SupervisorRole instance."""
        from victor.framework.agent_protocols import IAgentRole
        from victor.framework.agent_roles import SupervisorRole, get_role

        role = get_role("supervisor")
        assert role is not None
        assert isinstance(role, SupervisorRole)
        assert isinstance(role, IAgentRole)

    def test_get_role_returns_researcher(self):
        """get_role('researcher') should return ResearcherRole instance."""
        from victor.framework.agent_protocols import IAgentRole
        from victor.framework.agent_roles import ResearcherRole, get_role

        role = get_role("researcher")
        assert role is not None
        assert isinstance(role, ResearcherRole)
        assert isinstance(role, IAgentRole)

    def test_get_role_returns_executor(self):
        """get_role('executor') should return ExecutorRole instance."""
        from victor.framework.agent_protocols import IAgentRole
        from victor.framework.agent_roles import ExecutorRole, get_role

        role = get_role("executor")
        assert role is not None
        assert isinstance(role, ExecutorRole)
        assert isinstance(role, IAgentRole)

    def test_get_role_returns_reviewer(self):
        """get_role('reviewer') should return ReviewerRole instance."""
        from victor.framework.agent_protocols import IAgentRole
        from victor.framework.agent_roles import ReviewerRole, get_role

        role = get_role("reviewer")
        assert role is not None
        assert isinstance(role, ReviewerRole)
        assert isinstance(role, IAgentRole)

    def test_get_role_returns_none_for_unknown(self):
        """get_role with unknown name should return None."""
        from victor.framework.agent_roles import get_role

        role = get_role("unknown_role")
        assert role is None

    def test_get_role_case_insensitive(self):
        """get_role should be case-insensitive."""
        from victor.framework.agent_roles import ManagerRole, get_role

        role_lower = get_role("manager")
        role_upper = get_role("MANAGER")
        role_mixed = get_role("Manager")

        assert isinstance(role_lower, ManagerRole)
        assert isinstance(role_upper, ManagerRole)
        assert isinstance(role_mixed, ManagerRole)


# =============================================================================
# Role Capabilities Combinations Tests
# =============================================================================


class TestRoleCapabilityCombinations:
    """Tests for role capability combinations and constraints."""

    def test_only_manager_has_delegate(self):
        """Only manager role should have DELEGATE capability by default."""
        from victor.framework.agent_protocols import AgentCapability
        from victor.framework.agent_roles import (
            ExecutorRole,
            ManagerRole,
            ResearcherRole,
            ReviewerRole,
        )

        manager = ManagerRole()
        researcher = ResearcherRole()
        executor = ExecutorRole()
        reviewer = ReviewerRole()

        assert AgentCapability.DELEGATE in manager.capabilities
        assert AgentCapability.DELEGATE not in researcher.capabilities
        assert AgentCapability.DELEGATE not in executor.capabilities
        assert AgentCapability.DELEGATE not in reviewer.capabilities

    def test_only_executor_has_full_write(self):
        """Only executor should have both WRITE and EXECUTE capabilities."""
        from victor.framework.agent_protocols import AgentCapability
        from victor.framework.agent_roles import (
            ExecutorRole,
            ManagerRole,
            ResearcherRole,
            ReviewerRole,
        )

        manager = ManagerRole()
        researcher = ResearcherRole()
        executor = ExecutorRole()
        reviewer = ReviewerRole()

        # Only executor has both
        def has_both_write_execute(r):
            return (
                AgentCapability.WRITE in r.capabilities
                and AgentCapability.EXECUTE in r.capabilities
            )

        assert not has_both_write_execute(manager)
        assert not has_both_write_execute(researcher)
        assert has_both_write_execute(executor)
        assert not has_both_write_execute(reviewer)

    def test_all_roles_have_communicate(self):
        """All roles should have COMMUNICATE capability."""
        from victor.framework.agent_protocols import AgentCapability
        from victor.framework.agent_roles import (
            ExecutorRole,
            ManagerRole,
            ResearcherRole,
            ReviewerRole,
        )

        for role_class in [ManagerRole, ResearcherRole, ExecutorRole, ReviewerRole]:
            role = role_class()
            assert AgentCapability.COMMUNICATE in role.capabilities


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module-level exports."""

    def test_exports_manager_role(self):
        """agent_roles should export ManagerRole."""
        from victor.framework.agent_roles import ManagerRole

        assert ManagerRole is not None

    def test_exports_supervisor_role(self):
        """agent_roles should export SupervisorRole."""
        from victor.framework.agent_roles import SupervisorRole

        assert SupervisorRole is not None

    def test_framework_exports_supervisor_role(self):
        """victor.framework should export SupervisorRole."""
        from victor.framework import SupervisorRole

        assert SupervisorRole is not None

    def test_exports_researcher_role(self):
        """agent_roles should export ResearcherRole."""
        from victor.framework.agent_roles import ResearcherRole

        assert ResearcherRole is not None

    def test_exports_executor_role(self):
        """agent_roles should export ExecutorRole."""
        from victor.framework.agent_roles import ExecutorRole

        assert ExecutorRole is not None

    def test_exports_reviewer_role(self):
        """agent_roles should export ReviewerRole."""
        from victor.framework.agent_roles import ReviewerRole

        assert ReviewerRole is not None

    def test_exports_role_registry(self):
        """agent_roles should export ROLE_REGISTRY."""
        from victor.framework.agent_roles import ROLE_REGISTRY

        assert ROLE_REGISTRY is not None

    def test_exports_get_role(self):
        """agent_roles should export get_role."""
        from victor.framework.agent_roles import get_role

        assert get_role is not None
