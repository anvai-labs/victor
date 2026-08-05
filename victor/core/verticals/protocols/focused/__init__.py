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

"""Focused Protocols Package (ISP: Interface Segregation Principle).

This package provides segregated, focused protocol interfaces extracted from
the larger VerticalExtensions interface. Each protocol in this package follows
the Interface Segregation Principle (ISP), containing fields and methods for
a single responsibility.

What are Focused Protocols?
============================

Focused protocols are lean, single-purpose interfaces extracted from the
monolithic VerticalExtensions interface. They embody the ISP principle:
"clients should not be forced to depend on interfaces they don't use."

Each protocol focuses on one aspect of vertical extensions:
- ConfigExtensionsProtocol: Mode configurations for domain-specific operations
- FrameworkExtensionsProtocol: Workflows, RL configs, and team specifications
- PromptExtensionsProtocol: Prompt contributors and enrichment strategies
- ToolExtensionsProtocol: Tool middleware and dependency management
- SafetyExtensionsProtocol: Domain-specific safety patterns for dangerous operations

When to Use Each Protocol
===========================

ConfigExtensionsProtocol
------------------------
Use when your vertical needs to define domain-specific operational modes.

Example: Coding vertical defines "fast", "thorough", "test-only" modes.

    from victor.core.verticals.protocols.focused import ConfigExtensionsProtocol

    class CodingConfigExtensions(ConfigExtensionsProtocol):
        @property
        def mode_config_provider(self) -> ModeConfigProviderProtocol | None:
            return CodingModeProvider()

FrameworkExtensionsProtocol
---------------------------
Use when your vertical provides workflows, RL configurations, or team specs.

    from victor.core.verticals.protocols.focused import FrameworkExtensionsProtocol

    class DevOpsFrameworkExtensions(FrameworkExtensionsProtocol):
        workflow_provider: Optional[WorkflowProviderProtocol] = DevOpsWorkflowProvider()
        rl_config_provider: Optional[RLConfigProviderProtocol] = None
        team_spec_provider: Optional[TeamSpecProviderProtocol] = DevOpsTeamProvider()

PromptExtensionsProtocol
------------------------
Use when your vertical needs to contribute domain-specific prompts or enrichment.

    from victor.core.verticals.protocols.focused import PromptExtensionsProtocol

    class ResearchPromptExtensions(PromptExtensionsProtocol):
        @property
        def prompt_contributors(self) -> List[PromptContributorProtocol]:
            return [CitationContributor(), SourceQualityContributor()]

        @property
        def enrichment_strategy(self) -> EnrichmentStrategyProtocol | None:
            return ResearchEnrichmentStrategy()

ToolExtensionsProtocol
---------------------
Use when your vertical needs tool middleware or dependency management.

    from victor.core.verticals.protocols.focused import ToolExtensionsProtocol

    class CodingToolExtensions(ToolExtensionsProtocol):
        @property
        def middleware(self) -> List[MiddlewareProtocol]:
            return [SyntaxValidationMiddleware(), ImportAnalysisMiddleware()]

        @property
        def tool_dependency_provider(self) -> Optional[ToolDependencyProviderProtocol]:
            return CodingToolDependencyProvider()

SafetyExtensionsProtocol
------------------------
Use when your vertical needs domain-specific safety patterns for dangerous operations.

    from victor.core.verticals.protocols.focused import SafetyExtensionsProtocol

    class DevOpsSafetyExtensions(SafetyExtensionsProtocol):
        def get_safety_extensions(self) -> List[SafetyExtensionProtocol]:
            return [KubernetesSafetyExtension(), TerraformSafetyExtension()]

How to Combine Multiple Protocols
==================================

Verticals can implement multiple focused protocols using composition or
multiple inheritance. The recommended pattern is composition:

    class CodingVertical(VerticalBase):
        def __init__(self):
            self._config = CodingConfigExtensions()
            self._prompt = CodingPromptExtensions()
            self._tools = CodingToolExtensions()

        # Expose focused protocols via properties
        @property
        def config_extensions(self) -> ConfigExtensionsProtocol:
            return self._config

        @property
        def prompt_extensions(self) -> PromptExtensionsProtocol:
            return self._prompt

        @property
        def tool_extensions(self) -> ToolExtensionsProtocol:
            return self._tools

Alternative: Protocol aggregation using a focused interface:

    from typing import Protocol, runtime_checkable

    @runtime_checkable
    class CodingVerticalExtensions(
        ConfigExtensionsProtocol,
        PromptExtensionsProtocol,
        ToolExtensionsProtocol,
        Protocol):
        \"\"\"Combined protocol for coding vertical extensions.\"\"\"
        pass

Migration Guide from Fat VerticalExtensions Interface
======================================================

Before (fat interface):
-----------------------

    from victor.core.verticals.protocols import VerticalExtensions

    class MyVertical:
        mode_config_provider: ModeConfigProviderProtocol | None = None
        workflow_provider: Optional[WorkflowProviderProtocol] = None
        prompt_contributors: List[PromptContributorProtocol] = []
        enrichment_strategy: EnrichmentStrategyProtocol | None = None
        middleware: List[MiddlewareProtocol] = []
        # ... many more fields, most unused

After (focused protocols):
--------------------------

    from victor.core.verticals.protocols.focused import (
        ConfigExtensionsProtocol,
        PromptExtensionsProtocol,
    )

    class MyVertical(ConfigExtensionsProtocol, PromptExtensionsProtocol):
        # Only implement what you actually need
        @property
        def mode_config_provider(self) -> ModeConfigProviderProtocol | None:
            return self._config_provider

        @property
        def prompt_contributors(self) -> List[PromptContributorProtocol]:
            return self._contributors

        @property
        def enrichment_strategy(self) -> EnrichmentStrategyProtocol | None:
            return self._enrichment

Benefits:
- No unused dependencies (ISP compliance)
- Clearer intent (single responsibility)
- Easier testing (mock only what's needed)
- Better type hints (IDE understands actual capabilities)

Protocol Reference
==================

ConfigExtensionsProtocol
    Properties:
        mode_config_provider: ModeConfigProviderProtocol | None
    Methods:
        get_all_mode_configs() -> Dict[str, ModeConfig]

FrameworkExtensionsProtocol
    Properties:
        workflow_provider: Optional[WorkflowProviderProtocol]
        rl_config_provider: Optional[RLConfigProviderProtocol]
        team_spec_provider: Optional[TeamSpecProviderProtocol]
    Methods:
        get_workflows() -> Dict[str, Any]
        get_auto_workflows() -> List[Any]
        get_rl_config() -> Dict[str, Any]
        get_rl_hooks() -> Optional[Any]
        get_team_specs() -> Dict[str, Any]
        get_default_team() -> Optional[str]

PromptExtensionsProtocol
    Properties:
        prompt_contributors: List[PromptContributorProtocol]
        enrichment_strategy: EnrichmentStrategyProtocol | None
    Methods:
        get_all_task_hints() -> Dict[str, TaskTypeHint]
        get_all_system_prompt_sections() -> List[str]

ToolExtensionsProtocol
    Properties:
        middleware: List[MiddlewareProtocol]
        tool_dependency_provider: Optional[ToolDependencyProviderProtocol]

SafetyExtensionsProtocol
    Methods:
        get_safety_extensions() -> List[SafetyExtensionProtocol]
        get_all_safety_patterns() -> List[SafetyPattern]

MiddlewareProtocol (Re-exported)
    Methods:
        before_tool_call(tool_name, arguments) -> MiddlewareResult
        after_tool_call(tool_name, arguments, result, success) -> Optional[Any]
        get_priority() -> MiddlewarePriority
        get_applicable_tools() -> Optional[Set[str]]
"""

# Config Extensions
from victor.core.verticals.protocols.focused.config_extensions import (
    ConfigExtensionsProtocol,
)

# Framework Extensions
from victor.core.verticals.protocols.focused.framework_extensions import (
    FrameworkExtensionsProtocol,
)

# Prompt Extensions
from victor.core.verticals.protocols.focused.prompt_extensions import (
    PromptExtensionsProtocol,
)

# Tool Extensions
from victor.core.verticals.protocols.focused.tool_extensions import (
    MiddlewareProtocol,
    ToolExtensionsProtocol,
)

# Safety Extensions
from victor.core.verticals.protocols.focused.safety_extensions import (
    SafetyExtensionsProtocol,
)

__all__ = [
    "ConfigExtensionsProtocol",
    "FrameworkExtensionsProtocol",
    "PromptExtensionsProtocol",
    "ToolExtensionsProtocol",
    "MiddlewareProtocol",
    "SafetyExtensionsProtocol",
]
