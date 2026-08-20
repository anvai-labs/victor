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

"""Preset agent definitions.

Ready-to-use agent configurations for common use cases.
These can be used directly or as templates for customization.

Example:
    from victor.agent.specs.presets import researcher_agent, coder_agent

    # Use directly
    pipeline = Pipeline([researcher_agent, coder_agent])

    # Or customize
    my_coder = coder_agent.with_capabilities(
        tools={"deploy_script"},
        can_execute_code=True,
    )
"""

from victor.agent.specs.models import (
    AgentSpec,
    AgentCapabilities,
    AgentConstraints,
    ModelPreference,
    OutputFormat,
    DelegationPolicy,
)

# =============================================================================
# RESEARCHER AGENT
# =============================================================================

researcher_agent = AgentSpec(
    name="researcher",
    description="Investigates codebases, gathers information, and analyzes requirements",
    capabilities=AgentCapabilities(
        tools={
            "glob",
            "grep",
            "read",
            "ls",
            "code_search",
            "semantic_code_search",
            "web_search",
            "web_fetch",
        },
        skills={"research", "analysis", "documentation"},
        can_browse_web=True,
        can_ask_user=True,
    ),
    constraints=AgentConstraints(
        max_iterations=30,
        max_tool_calls=50,
    ),
    model_preference=ModelPreference.REASONING,
    system_prompt="""You are a research specialist. Your role is to:
1. Thoroughly investigate the codebase and documentation
2. Gather relevant context for the task at hand
3. Identify patterns, dependencies, and potential issues
4. Summarize findings clearly for the next agent

Be thorough but focused. Don't modify any files - just gather information.""",
    output_format=OutputFormat.MARKDOWN,
    tags={"research", "analysis", "readonly"},
)


# =============================================================================
# CODER AGENT
# =============================================================================

coder_agent = AgentSpec(
    name="coder",
    description="Writes, modifies, and refactors code following best practices",
    capabilities=AgentCapabilities(
        tools={
            "read",
            "edit",
            "write",
            "ls",
            "glob",
            "grep",
            "shell",
            "code_search",
        },
        skills={"coding", "refactoring", "debugging"},
        can_execute_code=True,
        can_modify_files=True,
    ),
    constraints=AgentConstraints(
        max_iterations=40,
        max_tool_calls=80,
    ),
    model_preference=ModelPreference.CODING,
    system_prompt="""You are an expert software engineer. Your role is to:
1. Write clean, maintainable code
2. Follow the project's coding conventions
3. Implement features based on requirements
4. Fix bugs and improve code quality

Always:
- Read existing code before modifying
- Use consistent style with the codebase
- Add appropriate error handling
- Consider edge cases""",
    output_format=OutputFormat.CODE,
    tags={"coding", "development"},
)


# =============================================================================
# REVIEWER AGENT
# =============================================================================

reviewer_agent = AgentSpec(
    name="reviewer",
    description="Reviews code changes for quality, security, and best practices",
    capabilities=AgentCapabilities(
        tools={
            "read",
            "glob",
            "grep",
            "git_diff",
            "git_log",
            "code_search",
        },
        skills={"review", "security", "quality"},
        can_ask_user=True,
    ),
    constraints=AgentConstraints(
        max_iterations=20,
        max_tool_calls=30,
    ),
    model_preference=ModelPreference.REASONING,
    system_prompt="""You are a senior code reviewer. Your role is to:
1. Review code changes for correctness and clarity
2. Check for security vulnerabilities (OWASP Top 10)
3. Verify adherence to coding standards
4. Suggest improvements and optimizations

Provide constructive feedback with specific suggestions.
Focus on significant issues rather than nitpicking.""",
    output_format=OutputFormat.MARKDOWN,
    tags={"review", "security", "readonly"},
)


# =============================================================================
# DEVOPS AGENT
# =============================================================================

devops_agent = AgentSpec(
    name="devops",
    description="Handles infrastructure, deployment, and operational tasks",
    capabilities=AgentCapabilities(
        tools={
            "read",
            "edit",
            "write",
            "shell",
            "ls",
            "glob",
        },
        tool_patterns=["docker_*", "k8s_*", "terraform_*"],
        skills={"infrastructure", "deployment", "automation"},
        can_execute_code=True,
        can_modify_files=True,
    ),
    constraints=AgentConstraints(
        max_iterations=25,
        max_tool_calls=40,
        timeout_seconds=300.0,
    ),
    model_preference=ModelPreference.TOOL_USE,
    system_prompt="""You are a DevOps engineer. Your role is to:
1. Manage infrastructure configuration
2. Set up CI/CD pipelines
3. Handle deployments and automation
4. Monitor and troubleshoot systems

Always:
- Use infrastructure as code principles
- Consider security implications
- Document configuration changes
- Test changes in isolation when possible""",
    output_format=OutputFormat.STRUCTURED,
    tags={"devops", "infrastructure", "deployment"},
)


# =============================================================================
# ANALYST AGENT
# =============================================================================

analyst_agent = AgentSpec(
    name="analyst",
    description="Analyzes data, generates reports, and provides insights",
    capabilities=AgentCapabilities(
        tools={
            "read",
            "execute_python",
            "write",
            "web_search",
            "ls",
        },
        skills={"analysis", "visualization", "reporting"},
        can_execute_code=True,
    ),
    constraints=AgentConstraints(
        max_iterations=30,
        max_tool_calls=50,
    ),
    model_preference=ModelPreference.REASONING,
    system_prompt="""You are a data analyst. Your role is to:
1. Analyze data and identify patterns
2. Generate visualizations and reports
3. Provide actionable insights
4. Explain findings clearly

Use Python for data processing and visualization.
Always validate data quality before analysis.""",
    output_format=OutputFormat.MARKDOWN,
    tags={"analysis", "data", "reporting"},
)


# =============================================================================
# TESTER AGENT
# =============================================================================

tester_agent = AgentSpec(
    name="tester",
    description="Writes and runs tests, ensures code quality",
    capabilities=AgentCapabilities(
        tools={
            "read",
            "write",
            "edit",
            "shell",
            "glob",
            "grep",
        },
        skills={"testing", "quality", "automation"},
        can_execute_code=True,
        can_modify_files=True,
    ),
    constraints=AgentConstraints(
        max_iterations=35,
        max_tool_calls=60,
    ),
    model_preference=ModelPreference.CODING,
    system_prompt="""You are a QA engineer specializing in testing. Your role is to:
1. Write comprehensive unit and integration tests
2. Run existing tests and analyze results
3. Identify edge cases and failure modes
4. Ensure test coverage for critical paths

Follow the project's testing conventions.
Use descriptive test names and clear assertions.""",
    output_format=OutputFormat.CODE,
    tags={"testing", "quality", "automation"},
)


# =============================================================================
# PRESET REGISTRY
# =============================================================================

_PRESET_AGENTS = {
    "researcher": researcher_agent,
    "coder": coder_agent,
    "reviewer": reviewer_agent,
    "devops": devops_agent,
    "analyst": analyst_agent,
    "tester": tester_agent,
}


def get_preset_agent(name: str) -> AgentSpec:
    """Get a preset agent by name.

    Args:
        name: Preset agent name

    Returns:
        AgentSpec for the preset

    Raises:
        ValueError: If preset not found
    """
    if name not in _PRESET_AGENTS:
        available = ", ".join(_PRESET_AGENTS.keys())
        raise ValueError(f"Unknown preset agent '{name}'. Available: {available}")
    return _PRESET_AGENTS[name]


def list_preset_agents() -> list[str]:
    """List available preset agent names.

    Returns:
        List of preset agent names
    """
    return list(_PRESET_AGENTS.keys())


def register_preset_agent(name: str, agent: AgentSpec) -> None:
    """Register a custom preset agent.

    Args:
        name: Preset name
        agent: AgentSpec to register
    """
    _PRESET_AGENTS[name] = agent
