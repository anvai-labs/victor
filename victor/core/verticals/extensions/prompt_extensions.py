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

"""Prompt Extensions - ISP-compliant composite for prompt-related protocols.

This module provides a focused extension for prompt-related vertical capabilities:
- Prompt contributors for system prompt sections and task hints
- Enrichment strategies for DSPy-like prompt optimization

This replaces the prompt-related parts of the monolithic VerticalExtensions class,
following Interface Segregation Principle (ISP).

Usage:
    from victor.core.verticals.extensions import PromptExtensions
    from victor.core.verticals.protocols import PromptContributorProtocol

    class CodingPromptContributor(PromptContributorProtocol):
        def get_task_type_hints(self):
            return {"edit": TaskTypeHint(hint="Read file first")}

        def get_system_prompt_section(self):
            return "You are an expert programmer."

    prompt_ext = PromptExtensions(
        prompt_contributors=[CodingPromptContributor()],
        enrichment_strategy=CodebaseEnrichmentStrategy(),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor.agent.prompt_section_registry import register_prompt_contributor_sections
from victor.core.vertical_types import TaskTypeHint
from victor.core.verticals.protocols.prompt_provider import (
    collect_prompt_section_contributions,
)


@dataclass
class PromptExtensions:
    """Focused extension for prompt-related vertical capabilities.

    Groups prompt contributors and enrichment strategies - the prompt-specific
    parts that were previously bundled in VerticalExtensions.

    Attributes:
        prompt_contributors: List of prompt contributors for system prompt sections
            and task-type-specific hints. Contributors are sorted by priority.
        enrichment_strategy: Optional enrichment strategy for DSPy-like prompt
            optimization. Can inject context from knowledge graphs, web search, etc.

    Example:
        prompt_ext = PromptExtensions(
            prompt_contributors=[
                CodingPromptContributor(),
                SecurityPromptContributor(),
            ],
            enrichment_strategy=CodebaseEnrichmentStrategy(),
        )

        # Get merged task hints from all contributors
        hints = prompt_ext.get_all_task_hints()

        # Build combined system prompt sections
        sections = prompt_ext.get_combined_system_prompt_sections()
    """

    prompt_contributors: List[Any] = field(default_factory=list)  # List[PromptContributorProtocol]
    enrichment_strategy: Optional[Any] = None  # EnrichmentStrategyProtocol

    def get_all_task_hints(self) -> Dict[str, TaskTypeHint]:
        """Merge task hints from all contributors.

        Contributors are sorted by priority, and later contributors
        override earlier ones for the same task type.

        Returns:
            Dict mapping task types to their hints
        """
        merged: Dict[str, TaskTypeHint] = {}
        sorted_contributors = sorted(
            self.prompt_contributors,
            key=lambda c: c.get_priority() if hasattr(c, "get_priority") else 50,
        )
        for contributor in sorted_contributors:
            if hasattr(contributor, "get_task_type_hints"):
                merged.update(contributor.get_task_type_hints())
        return merged

    def get_hint_for_task(self, task_type: str) -> Optional[TaskTypeHint]:
        """Get the hint for a specific task type.

        Args:
            task_type: The task type to get hint for (e.g., "edit", "analyze")

        Returns:
            TaskTypeHint if found, None otherwise
        """
        hints = self.get_all_task_hints()
        return hints.get(task_type.lower())

    def get_combined_system_prompt_sections(self) -> str:
        """Get combined system prompt sections from all contributors.

        Sections are sorted by contributor priority and concatenated
        with double newlines between them.

        Returns:
            Combined system prompt text
        """
        register_prompt_contributor_sections(self.prompt_contributors)
        sections: List[str] = []
        sorted_contributors = sorted(
            self.prompt_contributors,
            key=lambda c: c.get_priority() if hasattr(c, "get_priority") else 50,
        )
        for contributor in sorted_contributors:
            for contribution in collect_prompt_section_contributions(contributor):
                if contribution.text:
                    sections.append(contribution.text)
        return "\n\n".join(sections)

    def get_grounding_rules(self) -> str:
        """Get combined grounding rules from contributors.

        Returns the first non-empty grounding rules found (by priority).

        Returns:
            Grounding rules text or empty string
        """
        sorted_contributors = sorted(
            self.prompt_contributors,
            key=lambda c: c.get_priority() if hasattr(c, "get_priority") else 50,
        )
        for contributor in sorted_contributors:
            if hasattr(contributor, "get_grounding_rules"):
                rules = contributor.get_grounding_rules()
                if rules:
                    return rules
        return ""

    def has_enrichment(self) -> bool:
        """Check if an enrichment strategy is configured.

        Returns:
            True if enrichment_strategy is set
        """
        return self.enrichment_strategy is not None

    async def get_enrichments(
        self,
        prompt: str,
        context: Any,
    ) -> List[Any]:
        """Get prompt enrichments if enrichment strategy is configured.

        Args:
            prompt: The prompt to enrich
            context: EnrichmentContext with task metadata

        Returns:
            List of ContextEnrichment objects or empty list
        """
        if not self.enrichment_strategy:
            return []

        if hasattr(self.enrichment_strategy, "get_enrichments"):
            return await self.enrichment_strategy.get_enrichments(prompt, context)
        return []

    def get_enrichment_priority(self) -> int:
        """Get the priority of the enrichment strategy.

        Returns:
            Priority value (default 50) or 100 if no strategy
        """
        if self.enrichment_strategy and hasattr(self.enrichment_strategy, "get_priority"):
            return self.enrichment_strategy.get_priority()
        return 100  # Low priority if not configured

    def get_enrichment_token_allocation(self) -> float:
        """Get the token allocation fraction for enrichment.

        Returns:
            Float between 0.0 and 1.0 (default 0.0 if no strategy)
        """
        if self.enrichment_strategy and hasattr(self.enrichment_strategy, "get_token_allocation"):
            return self.enrichment_strategy.get_token_allocation()
        return 0.0

    def merge(self, other: "PromptExtensions") -> "PromptExtensions":
        """Merge with another PromptExtensions instance.

        Contributors are concatenated. Enrichment strategy from other
        overrides self if present.

        Args:
            other: Another PromptExtensions to merge from

        Returns:
            New PromptExtensions with merged content
        """
        # Merge contributors (deduplicate by instance)
        seen_ids = set(id(c) for c in self.prompt_contributors)
        merged_contributors = list(self.prompt_contributors)
        for c in other.prompt_contributors:
            if id(c) not in seen_ids:
                merged_contributors.append(c)
                seen_ids.add(id(c))

        # Take other's enrichment strategy if present
        enrichment = other.enrichment_strategy or self.enrichment_strategy

        return PromptExtensions(
            prompt_contributors=merged_contributors,
            enrichment_strategy=enrichment,
        )

    def __bool__(self) -> bool:
        """Return True if any content is present."""
        return bool(self.prompt_contributors or self.enrichment_strategy)


__all__ = ["PromptExtensions"]
