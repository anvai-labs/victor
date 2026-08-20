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

"""Safety Rule Provider Protocol.

This protocol defines the interface that vertical packages must implement
to register their safety rules with the Victor framework.

Verticals register safety rules via the `victor.safety_rules` entry point group.
Each registration function should conform to this protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from victor.framework.config import SafetyEnforcer


@runtime_checkable
class SafetyRuleProvider(Protocol):
    """Protocol for safety rule registration functions.

    Safety rule registration functions are called by the framework to
    register vertical-specific safety rules. These functions are registered
    via the `victor.safety_rules` entry point group.

    The function signature must be:
        def register_<vertical>_safety_rules(enforcer: SafetyEnforcer) -> None

    Example:
        # In victor_rag/safety.py:
        def register_rag_safety_rules(enforcer: SafetyEnforcer) -> None:
            \"\"\"Register RAG-specific safety rules with the enforcer.\"\"\"
            enforcer.register_rule(RAGDeletionRule())
            enforcer.register_rule(RAGIngestionRule())

        # In victor-rag/pyproject.toml:
        [project.entry-points."victor.safety_rules"]
        rag = "victor_rag.safety:register_rag_safety_rules"

        # Framework usage:
        from importlib.metadata import entry_points
        enforcer = SafetyEnforcer()
        for ep in entry_points(group="victor.safety_rules"):
            register_func = ep.load()
            register_func(enforcer)
    """

    def __call__(self, enforcer: SafetyEnforcer) -> None:
        """Register safety rules with the given enforcer.

        Args:
            enforcer: The safety enforcer to register rules with
        """
        ...
