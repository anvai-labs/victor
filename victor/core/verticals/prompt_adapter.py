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

"""Prompt contributor adapter for legacy format compatibility.

This module provides adapters for bridging different prompt contribution
formats, enabling backward compatibility with existing implementations
while supporting the new protocol-based system.

Design Philosophy:
- Support multiple input formats (dict, list, object)
- Normalize to PromptContributorProtocol interface
- Maintain backward compatibility
- Type-safe conversions

Usage:
    from victor.core.verticals.prompt_adapter import PromptContributorAdapter

    # From dict format (legacy)
    adapter = PromptContributorAdapter.from_dict(
        task_hints={"edit": {"hint": "...", "tool_budget": 5}},
        system_prompt_section="When editing code...",
    )

    # From existing contributor
    adapter = PromptContributorAdapter.wrap(existing_contributor)

    # Use as protocol
    hints = adapter.get_task_type_hints()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from victor.core.verticals.protocols import (
    PromptContributorProtocol,
    TaskTypeHint,
)
from victor.core.verticals.protocols.prompt_provider import (
    collect_prompt_section_contributions,
)

# =============================================================================
# Adapter Types
# =============================================================================


@dataclass
class LegacyTaskHint:
    """Legacy task hint format (dict-based).

    This represents the older format where task hints were
    simple dictionaries rather than TaskTypeHint dataclasses.

    Attributes:
        hint: The prompt hint text
        tool_budget: Recommended tool budget
        priority_tools: List of priority tools
    """

    hint: str
    tool_budget: Optional[int] = None
    priority_tools: List[str] = field(default_factory=list)

    def to_task_type_hint(self, task_type: str) -> TaskTypeHint:
        """Convert to TaskTypeHint.

        Args:
            task_type: The task type identifier

        Returns:
            TaskTypeHint instance
        """
        return TaskTypeHint(
            task_type=task_type,
            hint=self.hint,
            tool_budget=self.tool_budget,
            priority_tools=self.priority_tools,
        )


# =============================================================================
# Prompt Contributor Adapter
# =============================================================================


class PromptContributorAdapter(PromptContributorProtocol):
    """Adapter for prompt contributor formats.

    This class wraps various prompt contribution formats and normalizes
    them to the PromptContributorProtocol interface. It supports:

    1. Dictionary-based hints (legacy format)
    2. Existing PromptContributorProtocol implementations
    3. Callable factories
    4. Mixed format inputs

    Example:
        # From dictionary (legacy format)
        adapter = PromptContributorAdapter.from_dict(
            task_hints={
                "edit": {"hint": "[EDIT] Read first...", "tool_budget": 5},
                "search": {"hint": "[SEARCH] Use grep...", "tool_budget": 8},
            },
            system_prompt_section="Always verify changes.",
        )

        # Wrap existing contributor
        adapter = PromptContributorAdapter.wrap(coding_prompt_contributor)

        # Use as protocol
        for task_type, hint in adapter.get_task_type_hints().items():
            print(f"{task_type}: {hint.hint}")
    """

    def __init__(
        self,
        task_hints: Dict[str, TaskTypeHint],
        system_prompt_section: str = "",
        grounding_rules: str = "",
        priority: int = 50,
    ):
        """Initialize the adapter.

        Args:
            task_hints: Dict mapping task types to TaskTypeHint
            system_prompt_section: System prompt section text
            grounding_rules: Grounding rules text
            priority: Priority for ordering (lower = first)
        """
        self._task_hints = task_hints
        self._system_prompt_section = system_prompt_section
        self._grounding_rules = grounding_rules
        self._priority = priority

    def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
        """Get task type hints.

        Returns:
            Dict mapping task types to TaskTypeHint
        """
        return self._task_hints.copy()

    def get_system_prompt_section(self) -> str:
        """Get system prompt section.

        Returns:
            System prompt section text
        """
        return self._system_prompt_section

    def get_grounding_rules(self) -> str:
        """Get grounding rules.

        Returns:
            Grounding rules text
        """
        return self._grounding_rules

    def get_priority(self) -> int:
        """Get priority.

        Returns:
            Priority value
        """
        return self._priority

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_dict(
        cls,
        task_hints: Optional[Dict[str, Any]] = None,
        system_prompt_section: str = "",
        grounding_rules: str = "",
        priority: int = 50,
    ) -> "PromptContributorAdapter":
        """Create adapter from dictionary format.

        Supports multiple hint formats:
        - TaskTypeHint instances
        - Dict with hint/tool_budget/priority_tools keys
        - String (just the hint text)

        Args:
            task_hints: Dict mapping task types to hints (various formats)
            system_prompt_section: System prompt section text
            grounding_rules: Grounding rules text
            priority: Priority for ordering

        Returns:
            Configured PromptContributorAdapter

        Example:
            adapter = PromptContributorAdapter.from_dict(
                task_hints={
                    "edit": {"hint": "[EDIT]...", "tool_budget": 5},
                    "search": TaskTypeHint(...),
                    "simple": "Just a hint string",
                }
            )
        """
        converted_hints: Dict[str, TaskTypeHint] = {}

        if task_hints:
            for task_type, hint_data in task_hints.items():
                if isinstance(hint_data, TaskTypeHint):
                    # Already a TaskTypeHint
                    converted_hints[task_type] = hint_data
                elif isinstance(hint_data, dict):
                    # Dict format
                    converted_hints[task_type] = TaskTypeHint(
                        task_type=task_type,
                        hint=hint_data.get("hint", ""),
                        tool_budget=hint_data.get("tool_budget"),
                        priority_tools=hint_data.get("priority_tools", []),
                    )
                elif isinstance(hint_data, str):
                    # Just a string hint
                    converted_hints[task_type] = TaskTypeHint(
                        task_type=task_type,
                        hint=hint_data,
                    )
                elif hasattr(hint_data, "hint"):
                    # Object with hint attribute (duck typing)
                    converted_hints[task_type] = TaskTypeHint(
                        task_type=task_type,
                        hint=getattr(hint_data, "hint", ""),
                        tool_budget=getattr(hint_data, "tool_budget", None),
                        priority_tools=getattr(hint_data, "priority_tools", []),
                    )

        return cls(
            task_hints=converted_hints,
            system_prompt_section=system_prompt_section,
            grounding_rules=grounding_rules,
            priority=priority,
        )

    @classmethod
    def wrap(
        cls,
        contributor: PromptContributorProtocol,
    ) -> "PromptContributorAdapter":
        """Wrap an existing contributor in an adapter.

        This is useful for adding additional processing or
        overriding specific methods.

        Args:
            contributor: Existing PromptContributorProtocol

        Returns:
            PromptContributorAdapter wrapping the contributor
        """
        return cls(
            task_hints=contributor.get_task_type_hints(),
            system_prompt_section="\n\n".join(
                contribution.text
                for contribution in collect_prompt_section_contributions(contributor)
                if contribution.text
            ),
            grounding_rules=contributor.get_grounding_rules(),
            priority=contributor.get_priority(),
        )

    @classmethod
    def from_callable(
        cls,
        hints_fn: Callable[[], Dict[str, TaskTypeHint]],
        prompt_fn: Optional[Callable[[], str]] = None,
        grounding_fn: Optional[Callable[[], str]] = None,
        priority: int = 50,
    ) -> "PromptContributorAdapter":
        """Create adapter from callable factories.

        Useful for lazy evaluation of hints.

        Args:
            hints_fn: Function returning task hints dict
            prompt_fn: Optional function returning system prompt
            grounding_fn: Optional function returning grounding rules
            priority: Priority for ordering

        Returns:
            PromptContributorAdapter
        """
        return cls(
            task_hints=hints_fn(),
            system_prompt_section=prompt_fn() if prompt_fn else "",
            grounding_rules=grounding_fn() if grounding_fn else "",
            priority=priority,
        )

    @classmethod
    def empty(cls) -> "PromptContributorAdapter":
        """Create an empty adapter.

        Returns:
            Empty PromptContributorAdapter
        """
        return cls(
            task_hints={},
            system_prompt_section="",
            grounding_rules="",
            priority=100,
        )

    # =========================================================================
    # Composition Methods
    # =========================================================================

    def with_hints(self, additional_hints: Dict[str, TaskTypeHint]) -> "PromptContributorAdapter":
        """Create new adapter with additional hints.

        Args:
            additional_hints: Hints to add (override existing)

        Returns:
            New PromptContributorAdapter with merged hints
        """
        merged = {**self._task_hints, **additional_hints}
        return PromptContributorAdapter(
            task_hints=merged,
            system_prompt_section=self._system_prompt_section,
            grounding_rules=self._grounding_rules,
            priority=self._priority,
        )

    def with_priority(self, priority: int) -> "PromptContributorAdapter":
        """Create new adapter with different priority.

        Args:
            priority: New priority value

        Returns:
            New PromptContributorAdapter with updated priority
        """
        return PromptContributorAdapter(
            task_hints=self._task_hints,
            system_prompt_section=self._system_prompt_section,
            grounding_rules=self._grounding_rules,
            priority=priority,
        )

    def merge(self, other: "PromptContributorAdapter") -> "PromptContributorAdapter":
        """Merge with another adapter.

        The other adapter's values take precedence.

        Args:
            other: Adapter to merge with

        Returns:
            New PromptContributorAdapter with merged values
        """
        merged_hints = {**self._task_hints, **other._task_hints}

        # Combine prompt sections
        sections = []
        if self._system_prompt_section:
            sections.append(self._system_prompt_section)
        if other._system_prompt_section:
            sections.append(other._system_prompt_section)
        merged_prompt = "\n\n".join(sections)

        # Use other's grounding if provided
        grounding = other._grounding_rules or self._grounding_rules

        # Use lower (higher priority) value
        priority = min(self._priority, other._priority)

        return PromptContributorAdapter(
            task_hints=merged_hints,
            system_prompt_section=merged_prompt,
            grounding_rules=grounding,
            priority=priority,
        )


# =============================================================================
# Composite Prompt Contributor
# =============================================================================


class CompositePromptContributor(PromptContributorProtocol):
    """Composite that merges multiple prompt contributors.

    Contributors are applied in priority order (lower = first),
    with later contributors overriding earlier ones for the same
    task types.

    Example:
        composite = CompositePromptContributor([
            coding_contributor,   # priority 10
            project_contributor,  # priority 50
            user_contributor,     # priority 90
        ])

        # Hints from coding_contributor are overridden by
        # project_contributor for same task types
        hints = composite.get_task_type_hints()
    """

    def __init__(self, contributors: List[PromptContributorProtocol]):
        """Initialize the composite.

        Args:
            contributors: List of prompt contributors
        """
        # Sort by priority (lower first)
        self._contributors = sorted(contributors, key=lambda c: c.get_priority())

    def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
        """Get merged task type hints.

        Returns:
            Merged dict (later contributors override)
        """
        merged: Dict[str, TaskTypeHint] = {}
        for contributor in self._contributors:
            merged.update(contributor.get_task_type_hints())
        return merged

    def get_system_prompt_section(self) -> str:
        """Get combined system prompt sections.

        Returns:
            Concatenated prompt sections
        """
        sections = []
        for contributor in self._contributors:
            for contribution in collect_prompt_section_contributions(contributor):
                if contribution.text:
                    sections.append(contribution.text)
        return "\n\n".join(sections)

    def get_grounding_rules(self) -> str:
        """Get first non-empty grounding rules.

        Returns:
            Grounding rules from highest priority contributor
        """
        for contributor in self._contributors:
            rules = contributor.get_grounding_rules()
            if rules:
                return rules
        return ""

    def get_priority(self) -> int:
        """Get minimum priority.

        Returns:
            Lowest priority value among contributors
        """
        if not self._contributors:
            return 50
        return min(c.get_priority() for c in self._contributors)

    def add(self, contributor: PromptContributorProtocol) -> None:
        """Add a contributor.

        Args:
            contributor: Contributor to add
        """
        self._contributors.append(contributor)
        self._contributors.sort(key=lambda c: c.get_priority())


# =============================================================================
# Factory Functions
# =============================================================================


def create_prompt_adapter(
    task_hints: Optional[Dict[str, Any]] = None,
    system_prompt: str = "",
    grounding: str = "",
    priority: int = 50,
) -> PromptContributorAdapter:
    """Create a prompt contributor adapter.

    Convenience function for creating adapters from various formats.

    Args:
        task_hints: Task hints in various formats
        system_prompt: System prompt section
        grounding: Grounding rules
        priority: Priority value

    Returns:
        Configured PromptContributorAdapter
    """
    return PromptContributorAdapter.from_dict(
        task_hints=task_hints,
        system_prompt_section=system_prompt,
        grounding_rules=grounding,
        priority=priority,
    )


def merge_contributors(
    *contributors: PromptContributorProtocol,
) -> CompositePromptContributor:
    """Merge multiple prompt contributors.

    Args:
        *contributors: Contributors to merge

    Returns:
        CompositePromptContributor
    """
    return CompositePromptContributor(list(contributors))


__all__ = [
    "LegacyTaskHint",
    "PromptContributorAdapter",
    "CompositePromptContributor",
    "create_prompt_adapter",
    "merge_contributors",
]
