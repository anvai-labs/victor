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

"""Prompt Provider Protocols (ISP: Interface Segregation Principle).

This module contains protocols specifically for prompt contribution.
Following ISP, these protocols are focused on a single responsibility:
contributing to system prompts and task hints.

Usage:
    from victor.core.verticals.protocols.prompt_provider import (
        PromptContributorProtocol,
    )

    class CodingPromptContributor(PromptContributorProtocol):
        def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
            return {
                "edit": TaskTypeHint(
                    task_type="edit",
                    hint="[EDIT] Read target file first.",
                    tool_budget=5,
                ),
            }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from abc import abstractmethod
from typing import Any, Dict, List, Protocol, runtime_checkable

from victor.core.vertical_types import TaskTypeHint

# =============================================================================
# Prompt Contributor Protocol
# =============================================================================


@dataclass(frozen=True)
class PromptSectionContribution:
    """Named system-prompt section contributed by a vertical.

    This is the canonical contributor-owned metadata shape for prompt sections
    that should participate in section registries and, optionally, prompt
    optimization.
    """

    name: str
    text: str
    aliases: set[str] = field(default_factory=set)
    category: str = "context"
    evolvable: bool = False
    required: bool = False
    priority: int = 50
    default_strategies: tuple[str, ...] = ()


@runtime_checkable
class PromptContributorProtocol(Protocol):
    """Protocol for contributing to system prompts.

    Verticals can contribute domain-specific task hints and system
    prompt sections without modifying framework code.

    Example:
        class CodingPromptContributor(PromptContributorProtocol):
            def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
                return {
                    "edit": TaskTypeHint(
                        task_type="edit",
                        hint="[EDIT] Read target file first, then modify.",
                        tool_budget=5,
                        priority_tools=["read_file", "edit_files"],
                    ),
                }

            def get_system_prompt_section(self) -> str:
                return "When modifying code, always run tests afterward."
    """

    @abstractmethod
    def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
        """Get task-type-specific prompt hints.

        Returns:
            Dict mapping task types to their hints
        """
        ...

    def get_system_prompt_section(self) -> str:
        """Get a section to append to the system prompt.

        Returns:
            Additional system prompt text (or empty string)
        """
        return ""

    def get_grounding_rules(self) -> str:
        """Get vertical-specific grounding rules.

        Returns:
            Grounding rules text (or empty string for default)
        """
        return ""

    def get_priority(self) -> int:
        """Get priority for prompt section ordering.

        Lower values appear first.

        Returns:
            Priority value (default 50)
        """
        return 50


def collect_prompt_section_contributions(
    contributor: Any,
) -> List[PromptSectionContribution]:
    """Normalize contributor-owned prompt sections into a named metadata shape."""
    named_sections = getattr(contributor, "get_prompt_section_contributions", None)
    if callable(named_sections):
        try:
            contributions = named_sections()
        except Exception:
            contributions = []
        normalized = [c for c in (contributions or []) if getattr(c, "text", "").strip()]
        if normalized:
            return normalized

    section_getter = getattr(contributor, "get_system_prompt_section", None)
    if not callable(section_getter):
        return []

    text = str(section_getter() or "").strip()
    if not text:
        return []

    contributor_name = type(contributor).__name__.strip("_") or type(contributor).__name__
    canonical_name = f"VERTICAL_{contributor_name.upper()}"
    return [
        PromptSectionContribution(
            name=canonical_name,
            text=text,
            aliases={f"vertical_{contributor_name.lower()}"},
            category="context",
            evolvable=False,
            required=False,
            priority=getattr(contributor, "get_priority", lambda: 50)(),
        )
    ]


__all__ = [
    "PromptSectionContribution",
    "collect_prompt_section_contributions",
    "PromptContributorProtocol",
    "TaskTypeHint",
]
