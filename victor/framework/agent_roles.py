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

"""Built-in agent roles for multi-agent team coordination.

This module provides pre-defined roles for common team configurations.
Each role defines capabilities, allowed tools, and system prompt sections.

Roles:
- SupervisorRole: Coordinates team, delegates tasks, approves work
- ResearcherRole: Analyzes codebase, searches for information
- ExecutorRole: Implements changes, writes code, executes commands
- ReviewerRole: Reviews work, provides feedback, approves changes

Example:
    from victor.framework.agent_roles import get_role, SupervisorRole

    # Get a role by name
    researcher = get_role("researcher")

    # Or instantiate directly
    supervisor = SupervisorRole()

    # Use in team coordination
    if AgentCapability.DELEGATE in supervisor.capabilities:
        # Supervisor can delegate tasks
        pass
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Type

from victor.framework.agent_protocols import AgentCapability, IAgentRole

# =============================================================================
# Supervisor Role
# =============================================================================


@dataclass
class SupervisorRole:
    """Role for team supervisors who coordinate and delegate work.

    Supervisors have the ability to delegate tasks to other team members,
    communicate with all agents, and approve completed work. They typically
    have limited direct file manipulation tools but strong coordination
    capabilities.

    Attributes:
        name: Role identifier ('supervisor')
        capabilities: Set of capabilities (DELEGATE, COMMUNICATE, APPROVE)
        allowed_tools: Tools this role can use
        tool_budget: Maximum tool calls (default 20)

    Example:
        supervisor = SupervisorRole()
        assert AgentCapability.DELEGATE in supervisor.capabilities
    """

    name: str = "supervisor"
    capabilities: Set[AgentCapability] = field(
        default_factory=lambda: {
            AgentCapability.DELEGATE,
            AgentCapability.COMMUNICATE,
            AgentCapability.APPROVE,
            AgentCapability.READ,
        }
    )
    allowed_tools: Set[str] = field(
        default_factory=lambda: {
            "read",
            "ls",
            "grep",
            "semantic_search",
            "task_complete",
        }
    )
    tool_budget: int = 20

    def get_system_prompt_section(self) -> str:
        """Get the system prompt section for this role.

        Returns:
            System prompt text describing the supervisor role.
        """
        return """## Role: Supervisor

You are the team supervisor responsible for coordinating team activities.

Your responsibilities:
- Analyze tasks and break them down into subtasks
- Delegate work to appropriate team members
- Coordinate communication between team members
- Review and approve completed work
- Ensure overall task completion and quality

You should NOT directly write code or make changes. Instead, delegate
implementation tasks to executor agents and research tasks to researcher agents.
Focus on coordination, planning, and quality assurance."""


@dataclass
class ManagerRole(SupervisorRole):
    """Compatibility alias for SupervisorRole."""

    name: str = "manager"


# =============================================================================
# Researcher Role
# =============================================================================


@dataclass
class ResearcherRole:
    """Role for research agents who analyze and gather information.

    Researchers have strong read and search capabilities but cannot write
    files directly. They analyze codebases, search for patterns, and
    provide insights to other team members.

    Attributes:
        name: Role identifier ('researcher')
        capabilities: Set of capabilities (READ, SEARCH, COMMUNICATE)
        allowed_tools: Tools this role can use
        tool_budget: Maximum tool calls (default 25)

    Example:
        researcher = ResearcherRole()
        assert AgentCapability.SEARCH in researcher.capabilities
    """

    name: str = "researcher"
    capabilities: Set[AgentCapability] = field(
        default_factory=lambda: {
            AgentCapability.READ,
            AgentCapability.SEARCH,
            AgentCapability.COMMUNICATE,
        }
    )
    allowed_tools: Set[str] = field(
        default_factory=lambda: {
            "read",
            "ls",
            "grep",
            "glob",
            "semantic_search",
            "code_search",
            "web_search",
            "web_fetch",
        }
    )
    tool_budget: int = 25

    def get_system_prompt_section(self) -> str:
        """Get the system prompt section for this role.

        Returns:
            System prompt text describing the researcher role.
        """
        return """## Role: Researcher

You are a research specialist focused on analyzing code and gathering information.

Your responsibilities:
- Search and analyze the codebase to find relevant code
- Identify patterns, dependencies, and potential issues
- Research external documentation and best practices
- Provide detailed findings to other team members
- Document your discoveries clearly

You should NOT modify files or execute commands. Your role is to gather
information and provide insights that inform implementation decisions."""


# =============================================================================
# Executor Role
# =============================================================================


@dataclass
class ExecutorRole:
    """Role for executor agents who implement changes.

    Executors have full read, write, and execute capabilities. They are
    responsible for implementing code changes, creating files, and
    running commands.

    Attributes:
        name: Role identifier ('executor')
        capabilities: Set of capabilities (READ, WRITE, EXECUTE, COMMUNICATE)
        allowed_tools: Tools this role can use
        tool_budget: Maximum tool calls (default 30)

    Example:
        executor = ExecutorRole()
        assert AgentCapability.WRITE in executor.capabilities
    """

    name: str = "executor"
    capabilities: Set[AgentCapability] = field(
        default_factory=lambda: {
            AgentCapability.READ,
            AgentCapability.WRITE,
            AgentCapability.EXECUTE,
            AgentCapability.COMMUNICATE,
        }
    )
    allowed_tools: Set[str] = field(
        default_factory=lambda: {
            "read",
            "write",
            "edit",
            "delete_file",
            "ls",
            "grep",
            "glob",
            "shell",
            "git",
            "semantic_search",
            "code_search",
        }
    )
    tool_budget: int = 30

    def get_system_prompt_section(self) -> str:
        """Get the system prompt section for this role.

        Returns:
            System prompt text describing the executor role.
        """
        return """## Role: Executor

You are an implementation specialist responsible for writing and modifying code.

Your responsibilities:
- Implement code changes based on specifications
- Create new files and modify existing ones
- Execute commands and run tests
- Follow coding standards and best practices
- Ensure implementations are complete and functional

Focus on clean, maintainable code. When implementing changes, read the
existing code first to understand patterns and conventions."""


# =============================================================================
# Reviewer Role
# =============================================================================


@dataclass
class ReviewerRole:
    """Role for reviewer agents who review and approve work.

    Reviewers have read and approve capabilities. They review code changes,
    provide feedback, and ensure quality standards are met.

    Attributes:
        name: Role identifier ('reviewer')
        capabilities: Set of capabilities (READ, APPROVE, COMMUNICATE)
        allowed_tools: Tools this role can use
        tool_budget: Maximum tool calls (default 20)

    Example:
        reviewer = ReviewerRole()
        assert AgentCapability.APPROVE in reviewer.capabilities
    """

    name: str = "reviewer"
    capabilities: Set[AgentCapability] = field(
        default_factory=lambda: {
            AgentCapability.READ,
            AgentCapability.APPROVE,
            AgentCapability.COMMUNICATE,
        }
    )
    allowed_tools: Set[str] = field(
        default_factory=lambda: {
            "read",
            "ls",
            "grep",
            "glob",
            "git",
            "semantic_search",
            "code_search",
        }
    )
    tool_budget: int = 20

    def get_system_prompt_section(self) -> str:
        """Get the system prompt section for this role.

        Returns:
            System prompt text describing the reviewer role.
        """
        return """## Role: Reviewer

You are a quality assurance specialist responsible for reviewing work.

Your responsibilities:
- Review code changes for correctness and quality
- Check for bugs, security issues, and performance problems
- Ensure code follows project conventions and standards
- Provide constructive feedback to executors
- Approve or request changes to submitted work

Focus on quality, maintainability, and adherence to best practices.
When you find issues, provide specific, actionable feedback."""


# =============================================================================
# Role Registry
# =============================================================================


# Map of role names to role classes
ROLE_REGISTRY: Dict[str, Type[IAgentRole]] = {
    "supervisor": SupervisorRole,
    "manager": ManagerRole,
    "researcher": ResearcherRole,
    "executor": ExecutorRole,
    "reviewer": ReviewerRole,
}


def get_role(name: str) -> Optional[IAgentRole]:
    """Get a role instance by name.

    Args:
        name: Role name (case-insensitive)

    Returns:
        Role instance if found, None otherwise

    Example:
        supervisor = get_role("supervisor")
        researcher = get_role("RESEARCHER")  # Case insensitive
    """
    name_lower = name.lower()
    role_class = ROLE_REGISTRY.get(name_lower)
    if role_class:
        return role_class()
    return None


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Role classes
    "SupervisorRole",
    "ManagerRole",
    "ResearcherRole",
    "ExecutorRole",
    "ReviewerRole",
    # Registry
    "ROLE_REGISTRY",
    "get_role",
]
