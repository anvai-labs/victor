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

"""Tool Dependency Provider Protocol.

This protocol defines the interface that vertical packages must implement
to register their tool dependency providers with the Victor framework.

Verticals register tool dependency providers via the `victor.tool_dependencies`
entry point group. Each registration function should conform to this protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from victor_contracts.verticals.tool_dependencies import BaseToolDependencyProvider


@runtime_checkable
class ToolDependencyProvider(Protocol):
    """Protocol for tool dependency provider factory functions.

    Tool dependency provider factory functions are called by the framework to
    obtain vertical-specific tool dependency providers. These functions are
    registered via the `victor.tool_dependencies` entry point group.

    The function signature must be:
        def get_provider() -> BaseToolDependencyProvider

    Example:
        # In victor_coding/tool_dependencies.py:
        def get_provider() -> BaseToolDependencyProvider:
            \"\"\"Return tool dependency provider for coding vertical.\"\"\"
            return YAMLToolDependencyProvider(
                yaml_path=Path(__file__).parent / "tool_dependencies.yaml",
                canonicalize=True,
            )

        # In victor-coding/pyproject.toml:
        [project.entry-points."victor.tool_dependencies"]
        coding = "victor_coding.tool_dependencies:get_provider"

        # Framework usage:
        from importlib.metadata import entry_points
        eps = entry_points(group="victor.tool_dependencies")
        for ep in eps:
            if ep.name == "coding":
                provider_factory = ep.load()
                provider = provider_factory()
                deps = provider.get_dependencies()
    """

    def __call__(self) -> BaseToolDependencyProvider:
        """Return a configured tool dependency provider.

        Returns:
            A tool dependency provider configured for the vertical.
        """
        ...
