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

"""Team specifications for coding tasks.

Provides pre-defined team configurations for common software development
workflows including feature implementation, bug fixing, refactoring,
and code review.

This module uses the framework's multi-agent types (TeamTemplate, TeamSpec,
TeamMember) while providing domain-specific team configurations for coding tasks.

DEPRECATION NOTICE:
    CodingTeamSpec is deprecated and will be removed in a future version.
    Use TeamSpec from victor_contracts.team_schema instead:

        from victor_contracts.team_schema import TeamSpec

    CodingTeamSpec is maintained for backwards compatibility.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

from victor_contracts.team_schema import TeamFormation, TeamMemberSpec, TeamSpec

# Import framework multi-agent types for composition
from victor_contracts import (
    TeamTemplate,
    TeamSpec as FrameworkTeamSpec,
    TeamMember as FrameworkTeamMember,
    TeamTopology,
    TaskAssignmentStrategy,
    PersonaTraits,
    CommunicationStyle,
    ExpertiseLevel,
)


@dataclass
class CodingRoleConfig:
    """Configuration for a coding-specific role.

    This is a domain-specific wrapper that can be converted to the framework's
    PersonaTraits for use with multi-agent teams.

    Attributes:
        base_role: Base agent role (researcher, planner, executor, reviewer)
        tools: Tools available to this role
        tool_budget: Default tool budget
        description: Role description
    """

    base_role: str
    tools: List[str]
    tool_budget: int
    description: str = ""

    def to_persona_traits(self, name: Optional[str] = None) -> PersonaTraits:
        """Convert to framework PersonaTraits.

        Args:
            name: Optional persona name (defaults to base_role title)

        Returns:
            PersonaTraits instance for use with framework multi-agent teams
        """
        return PersonaTraits(
            name=name or self.base_role.replace("_", " ").title(),
            role=self.base_role,
            description=self.description,
            communication_style=CommunicationStyle.TECHNICAL,
            expertise_level=ExpertiseLevel.EXPERT,
            preferred_tools=self.tools,
        )


# Coding-specific roles with tool allocations
# Note: Uses canonical tool names from victor/tools/tool_names.py
CODING_ROLES: Dict[str, CodingRoleConfig] = {
    "code_researcher": CodingRoleConfig(
        base_role="researcher",
        tools=[
            "read",
            "grep",
            "code_search",
            "overview",
            "symbol",
            "refs",
            "ls",
        ],
        tool_budget=25,
        description="Researches and analyzes codebase patterns",
    ),
    "code_planner": CodingRoleConfig(
        base_role="planner",
        tools=[
            "read",
            "overview",
            "plan",
            "grep",
        ],
        tool_budget=15,
        description="Plans implementation approach",
    ),
    "code_executor": CodingRoleConfig(
        base_role="executor",
        tools=[
            "read",
            "write",
            "edit",
            "shell",
            "git",
        ],
        tool_budget=40,
        description="Implements code changes",
    ),
    "code_reviewer": CodingRoleConfig(
        base_role="reviewer",
        tools=[
            "read",
            "git",
            "test",
            "shell",
            "grep",
        ],
        tool_budget=20,
        description="Reviews code and runs tests",
    ),
    "test_writer": CodingRoleConfig(
        base_role="executor",
        tools=[
            "read",
            "write",
            "test",
            "shell",
        ],
        tool_budget=30,
        description="Writes and runs tests",
    ),
    "doc_writer": CodingRoleConfig(
        base_role="executor",
        tools=[
            "read",
            "write",
            "edit",
            "grep",
        ],
        tool_budget=20,
        description="Writes documentation",
    ),
    "security_reviewer": CodingRoleConfig(
        base_role="researcher",
        tools=[
            "read",
            "grep",
            "code_search",
            "shell",
        ],
        tool_budget=20,
        description="Reviews code for security issues",
    ),
}


# Mapping from TeamFormation to TeamTopology
_FORMATION_TO_TOPOLOGY: Dict[TeamFormation, TeamTopology] = {
    TeamFormation.SEQUENTIAL: TeamTopology.PIPELINE,
    TeamFormation.PIPELINE: TeamTopology.PIPELINE,
    TeamFormation.PARALLEL: TeamTopology.MESH,
    TeamFormation.HIERARCHICAL: TeamTopology.HIERARCHY,
}


@dataclass
class CodingTeamSpec:
    """Specification for a coding team.

    .. deprecated::
        CodingTeamSpec is deprecated. Use TeamSpec from
        victor_contracts.team_schema instead. CodingTeamSpec is maintained
        for backwards compatibility but will be removed in a future version.

    This class provides backward-compatible team specifications while
    supporting conversion to the framework's TeamTemplate and TeamSpec types.

    Attributes:
        name: Team name
        description: Team description
        formation: How agents are organized
        members: Team member specifications
        total_tool_budget: Total tool budget for the team
        max_iterations: Maximum iterations
    """

    name: str
    description: str
    formation: TeamFormation
    members: List[TeamMemberSpec]
    total_tool_budget: int = 100
    max_iterations: int = 50

    def __post_init__(self):
        """Emit deprecation warning on instantiation."""
        warnings.warn(
            "CodingTeamSpec is deprecated. Use TeamSpec from "
            "victor_contracts.team_schema instead.",
            DeprecationWarning,
            stacklevel=3,
        )

    def to_canonical_team_spec(self) -> TeamSpec:
        """Convert to canonical TeamSpec from victor_contracts.team_schema.

        Returns:
            TeamSpec instance with vertical set to "coding"
        """
        return TeamSpec(
            name=self.name,
            description=self.description,
            vertical="coding",
            formation=self.formation,
            members=self.members,
            total_tool_budget=self.total_tool_budget,
            max_iterations=self.max_iterations,
        )

    def to_team_template(self) -> TeamTemplate:
        """Convert to framework TeamTemplate.

        Returns:
            TeamTemplate instance for use with framework multi-agent system
        """
        # Count roles for member_slots
        role_counts: Dict[str, int] = {}
        for member in self.members:
            role_counts[member.role] = role_counts.get(member.role, 0) + 1

        # Determine topology from formation
        topology = _FORMATION_TO_TOPOLOGY.get(self.formation, TeamTopology.PIPELINE)

        # Determine assignment strategy based on formation
        if self.formation == TeamFormation.HIERARCHICAL:
            assignment_strategy = TaskAssignmentStrategy.SKILL_MATCH
        elif self.formation == TeamFormation.PARALLEL:
            assignment_strategy = TaskAssignmentStrategy.LOAD_BALANCED
        else:
            assignment_strategy = TaskAssignmentStrategy.ROUND_ROBIN

        return TeamTemplate(
            name=self.name,
            description=self.description,
            topology=topology,
            assignment_strategy=assignment_strategy,
            member_slots=role_counts,
            max_iterations=self.max_iterations,
            config={
                "total_tool_budget": self.total_tool_budget,
                "formation": self.formation.value,
            },
        )

    def to_framework_team_spec(self) -> FrameworkTeamSpec:
        """Convert to framework TeamSpec with members.

        Returns:
            FrameworkTeamSpec instance for use with framework multi-agent system
        """
        template = self.to_team_template()

        # Convert members to FrameworkTeamMember instances
        framework_members: List[FrameworkTeamMember] = []
        for member in self.members:
            # Create PersonaTraits from TeamMemberSpec
            persona = PersonaTraits(
                name=member.name or f"{member.role.title()} Agent",
                role=member.role,
                description=member.backstory or member.goal,
                communication_style=CommunicationStyle.TECHNICAL,
                expertise_level=ExpertiseLevel.EXPERT,
                strengths=member.expertise if member.expertise else [],
            )

            framework_member = FrameworkTeamMember(
                persona=persona,
                role_in_team=member.role,
                is_leader=member.is_manager,
                tool_access=[],  # Tools managed at team level
            )
            framework_members.append(framework_member)

        return FrameworkTeamSpec(
            template=template,
            members=framework_members,
        )


# Pre-defined team specifications with rich personas
CODING_TEAM_SPECS: Dict[str, TeamSpec] = {
    "feature_team": TeamSpec(
        name="Feature Implementation Team",
        description=(
            "End-to-end feature implementation with research, planning, "
            "implementation, and review"
        ),
        vertical="coding",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Analyze codebase for relevant patterns and dependencies",
                name="Code Researcher",
                tool_budget=25,
                backstory=(
                    "You are a meticulous code archaeologist with expertise in understanding "
                    "large codebases. You've analyzed hundreds of projects and have an eye for "
                    "patterns, dependencies, and architectural decisions. You notice subtle "
                    "connections between modules that others miss. Your research is thorough "
                    "and you never make assumptions without evidence from the code."
                ),
                memory=True,  # Persist discoveries for team
            ),
            TeamMemberSpec(
                role="planner",
                goal="Design implementation approach based on research",
                name="Implementation Planner",
                tool_budget=15,
                backstory=(
                    "You are a pragmatic software architect who values clean, maintainable "
                    "solutions. You've designed systems ranging from startups to enterprise. "
                    "You consider edge cases, error handling, and future extensibility while "
                    "keeping solutions simple. You create plans that are detailed enough to "
                    "guide implementation but flexible enough for adaptation."
                ),
                memory=True,  # Plan builds on research
            ),
            TeamMemberSpec(
                role="executor",
                goal="Implement the feature according to plan",
                name="Feature Implementer",
                tool_budget=40,
                backstory=(
                    "You are a skilled craftsman who takes pride in writing clean, efficient "
                    "code. You follow established patterns in the codebase and write code that "
                    "looks like it belongs. You handle errors gracefully, write meaningful "
                    "variable names, and add comments only where the logic isn't self-evident. "
                    "You test as you go and never leave debugging artifacts behind."
                ),
                cache=True,  # Cache file reads for efficiency
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Review code and run tests",
                name="Code Reviewer",
                tool_budget=20,
                backstory=(
                    "You are a detail-oriented reviewer who has caught countless bugs before "
                    "they reached production. You check for logic errors, edge cases, security "
                    "issues, and performance concerns. You verify that tests pass and that new "
                    "code doesn't break existing functionality. Your feedback is constructive "
                    "and specific, always including how to fix issues you find."
                ),
            ),
        ],
        total_tool_budget=100,
    ),
    "bug_fix_team": TeamSpec(
        name="Bug Fix Team",
        description="Systematic bug investigation and fix with verification",
        vertical="coding",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Investigate bug root cause through code analysis",
                name="Bug Investigator",
                tool_budget=25,
                backstory=(
                    "You are a debugging expert who loves solving puzzles. You've tracked down "
                    "the most elusive bugs across race conditions, memory leaks, and edge cases. "
                    "You approach bugs systematically: reproduce, isolate, trace, identify. "
                    "You look at stack traces, logs, and code flow. You never assume the obvious "
                    "cause is the real cause until you've verified it."
                ),
                memory=True,  # Share root cause with fixer
                verbose=True,  # Detailed investigation logs
            ),
            TeamMemberSpec(
                role="executor",
                goal="Apply the fix based on investigation",
                name="Bug Fixer",
                tool_budget=25,
                backstory=(
                    "You are a surgeon who makes precise, minimal fixes. You understand that "
                    "the best bug fix changes as little as possible while fully addressing the "
                    "root cause. You consider whether the bug exists elsewhere and fix all "
                    "occurrences. You add regression tests to prevent the bug from returning. "
                    "Your fixes are clean and well-documented."
                ),
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Verify fix with tests",
                name="Fix Verifier",
                tool_budget=20,
                backstory=(
                    "You are a quality gatekeeper who ensures fixes actually work. You run "
                    "existing tests, verify the new regression test catches the bug, and check "
                    "for unintended side effects. You test edge cases related to the fix. "
                    "You don't approve fixes that might introduce new problems."
                ),
            ),
        ],
        total_tool_budget=70,
    ),
    "refactoring_team": TeamSpec(
        name="Refactoring Team",
        description="Safe refactoring with analysis and testing",
        vertical="coding",
        formation=TeamFormation.HIERARCHICAL,
        members=[
            TeamMemberSpec(
                role="planner",
                goal="Plan refactoring approach and identify affected areas",
                name="Refactoring Planner",
                tool_budget=20,
                is_manager=True,
                backstory=(
                    "You are a refactoring strategist who has safely restructured legacy "
                    "codebases without breaking functionality. You identify code smells, plan "
                    "incremental improvements, and map out dependencies. You know which changes "
                    "are safe to make in parallel and which require careful sequencing. You "
                    "delegate effectively and keep the team focused on the goal."
                ),
                memory=True,  # Track overall refactoring state
            ),
            TeamMemberSpec(
                role="executor",
                goal="Execute refactoring changes",
                name="Refactoring Executor",
                tool_budget=35,
                backstory=(
                    "You are a refactoring specialist who transforms code while preserving "
                    "behavior. You extract methods, rename for clarity, reduce duplication, "
                    "and improve structure. You make small, testable changes rather than big "
                    "bang rewrites. You run tests frequently to catch regressions early. "
                    "You know the difference between cleanup and feature changes."
                ),
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Ensure tests pass and code quality maintained",
                name="Quality Verifier",
                tool_budget=20,
                backstory=(
                    "You are the guardian of code quality who ensures refactoring improves "
                    "rather than degrades the codebase. You verify that behavior is preserved, "
                    "tests still pass, and the code is actually cleaner. You check that "
                    "refactoring hasn't introduced subtle bugs or performance issues. You "
                    "give the final approval only when quality is maintained or improved."
                ),
            ),
        ],
        total_tool_budget=75,
    ),
    "review_team": TeamSpec(
        name="Code Review Team",
        description="Comprehensive code review with parallel analysis",
        vertical="coding",
        formation=TeamFormation.PARALLEL,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Check for security vulnerabilities",
                name="Security Reviewer",
                tool_budget=15,
                backstory=(
                    "You are a security-focused reviewer who has prevented countless breaches. "
                    "You look for injection vulnerabilities, authentication flaws, data "
                    "exposure, and insecure configurations. You know the OWASP Top 10 by heart "
                    "and apply it to every review. You think like an attacker to find "
                    "vulnerabilities before they can be exploited."
                ),
            ),
            TeamMemberSpec(
                role="researcher",
                goal="Check code style and conventions",
                name="Style Reviewer",
                tool_budget=15,
                backstory=(
                    "You are a consistency advocate who ensures code is readable and "
                    "maintainable. You check naming conventions, formatting, documentation, "
                    "and adherence to project standards. You look for code that would confuse "
                    "future maintainers. You balance perfectionism with pragmatism, focusing "
                    "on issues that actually matter for maintenance."
                ),
            ),
            TeamMemberSpec(
                role="researcher",
                goal="Check logic correctness and edge cases",
                name="Logic Reviewer",
                tool_budget=15,
                backstory=(
                    "You are a logic specialist who thinks in terms of invariants and edge "
                    "cases. You trace through code paths to find potential bugs. You check "
                    "error handling, null safety, bounds checking, and race conditions. You "
                    "ask 'what if?' for every assumption the code makes. You find bugs that "
                    "only appear under specific conditions."
                ),
            ),
            TeamMemberSpec(
                role="planner",
                goal="Synthesize findings into actionable feedback",
                name="Review Synthesizer",
                tool_budget=10,
                backstory=(
                    "You are a diplomatic communicator who turns review findings into "
                    "constructive feedback. You prioritize issues by severity and impact. "
                    "You consolidate duplicate concerns and present a clear action plan. "
                    "Your reviews are respectful, specific, and actionable. You distinguish "
                    "between blocking issues and suggestions."
                ),
            ),
        ],
        total_tool_budget=55,
    ),
    "testing_team": TeamSpec(
        name="Testing Team",
        description="Comprehensive test coverage improvement",
        vertical="coding",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Analyze existing test coverage and identify gaps",
                name="Coverage Analyzer",
                tool_budget=20,
                backstory=(
                    "You are a test coverage expert who knows that lines covered doesn't mean "
                    "well tested. You identify untested edge cases, error paths, and complex "
                    "logic branches. You prioritize gaps by risk: critical paths, error "
                    "handling, security checks. You understand which code most needs testing "
                    "based on complexity and importance."
                ),
                memory=True,  # Share gaps with test writer
            ),
            TeamMemberSpec(
                role="executor",
                goal="Write tests for uncovered areas",
                name="Test Writer",
                tool_budget=35,
                backstory=(
                    "You are a testing craftsman who writes tests that actually catch bugs. "
                    "You test behavior, not implementation. You write clear test names that "
                    "describe what they verify. You use appropriate assertions and helpful "
                    "failure messages. You know when to use mocks and when to use real "
                    "dependencies. Your tests are maintainable, not brittle."
                ),
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Run tests and verify coverage improvement",
                name="Test Verifier",
                tool_budget=15,
                backstory=(
                    "You are a test quality reviewer who ensures tests actually work and add "
                    "value. You run the full test suite and check for flaky tests. You verify "
                    "that coverage improved in the areas identified. You check that tests "
                    "would actually fail if the code was broken. You don't accept tests that "
                    "pass for the wrong reasons."
                ),
            ),
        ],
        total_tool_budget=70,
    ),
    "documentation_team": TeamSpec(
        name="Documentation Team",
        description="Generate and update documentation",
        vertical="coding",
        formation=TeamFormation.SEQUENTIAL,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Analyze code to understand functionality",
                name="Code Analyzer",
                tool_budget=20,
                backstory=(
                    "You are a code comprehension expert who can understand any codebase. "
                    "You read code for intent, not just mechanics. You identify key concepts, "
                    "public interfaces, usage patterns, and important behaviors. You note "
                    "what would surprise or confuse a new developer. Your analysis provides "
                    "everything needed to write clear documentation."
                ),
                memory=True,  # Share understanding with doc writer
            ),
            TeamMemberSpec(
                role="executor",
                goal="Write documentation",
                name="Doc Writer",
                tool_budget=25,
                backstory=(
                    "You are a technical writer who makes complex things simple. You write "
                    "for your audience: API docs for developers, guides for users, READMEs "
                    "for contributors. You include examples that actually work. You explain "
                    "the 'why' not just the 'what'. Your documentation is accurate, complete, "
                    "and easy to navigate. You avoid jargon unless necessary."
                ),
            ),
        ],
        total_tool_budget=45,
    ),
}


def get_team_for_task(task_type: str) -> Optional[TeamSpec]:
    """Get appropriate team specification for task type.

    Args:
        task_type: Type of task (feature, bug, review, etc.)

    Returns:
        TeamSpec or None if no matching team
    """
    mapping = {
        # Feature tasks
        "feature": "feature_team",
        "implement": "feature_team",
        "add": "feature_team",
        # Bug fix tasks
        "bug": "bug_fix_team",
        "fix": "bug_fix_team",
        "bugfix": "bug_fix_team",
        "debug": "bug_fix_team",
        # Refactoring tasks
        "refactor": "refactoring_team",
        "refactoring": "refactoring_team",
        "restructure": "refactoring_team",
        # Review tasks
        "review": "review_team",
        "code_review": "review_team",
        "audit": "review_team",
        # Testing tasks
        "test": "testing_team",
        "testing": "testing_team",
        "coverage": "testing_team",
        # Documentation tasks
        "doc": "documentation_team",
        "documentation": "documentation_team",
        "docs": "documentation_team",
    }
    spec_name = mapping.get(task_type.lower())
    if spec_name:
        return CODING_TEAM_SPECS.get(spec_name)
    return None


def get_role_config(role_name: str) -> Optional[CodingRoleConfig]:
    """Get configuration for a coding role.

    Args:
        role_name: Role name

    Returns:
        CodingRoleConfig or None
    """
    return CODING_ROLES.get(role_name.lower())


def list_team_types() -> List[str]:
    """List all available team types.

    Returns:
        List of team type names
    """
    return list(CODING_TEAM_SPECS.keys())


def list_roles() -> List[str]:
    """List all available coding roles.

    Returns:
        List of role names
    """
    return list(CODING_ROLES.keys())


__all__ = [
    # Types
    "CodingRoleConfig",
    "TeamSpec",  # Canonical from framework.team_schema (use this)
    # Role configurations
    "CODING_ROLES",
    # Team specifications
    "CODING_TEAM_SPECS",
    # Helper functions
    "get_team_for_task",
    "get_role_config",
    "list_team_types",
    "list_roles",
    # Re-exported framework types for convenience
    "TeamTemplate",
    "FrameworkTeamSpec",
    "FrameworkTeamMember",
    "TeamTopology",
    "TaskAssignmentStrategy",
]
