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

"""Escape Hatch Provider Protocol.

This protocol defines the interface that vertical packages must implement
to register their escape hatches with the Victor framework.

Verticals register escape hatches via the `victor.escape_hatches` entry point
group. Each registration function should conform to this protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from victor.framework.escape_hatch_registry import EscapeHatchRegistry


@runtime_checkable
class EscapeHatchProvider(Protocol):
    """Protocol for escape hatch registration functions.

    Escape hatch registration functions are called by the framework to
    register vertical-specific escape hatches. These functions are registered
    via the `victor.escape_hatches` entry point group.

    The function signature must be:
        def register_escape_hatches(registry: EscapeHatchRegistry) -> None

    Example:
        # In victor_coding/escape_hatches.py:
        def register_escape_hatches(registry: EscapeHatchRegistry) -> None:
            \"\"\"Register coding-specific escape hatches.\"\"\"
            registry.register(
                "code_execution",
                description="Allow direct code execution in sandboxed environment",
                safety_check=lambda: check_sandbox_available(),
            )

        # In victor-coding/pyproject.toml:
        [project.entry-points."victor.escape_hatches"]
        coding = "victor_coding.escape_hatches:register_escape_hatches"

        # Framework usage:
        from importlib.metadata import entry_points
        registry = EscapeHatchRegistry()
        for ep in entry_points(group="victor.escape_hatches"):
            register_func = ep.load()
            register_func(registry)
    """

    def __call__(self, registry: EscapeHatchRegistry) -> None:
        """Register escape hatches with the given registry.

        Args:
            registry: The escape hatch registry to register hatches with.
        """
        ...
