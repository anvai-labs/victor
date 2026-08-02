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

"""Prompt Extensions Protocol (ISP: Interface Segregation Principle).

This module provides a focused protocol for prompt-related extensions
as part of the VerticalExtensions ISP refactoring.

Following the Interface Segregation Principle (ISP), this protocol contains
ONLY prompt-related fields extracted from the larger VerticalExtensions
interface. This allows verticals to implement only the prompt extensions they
need without being forced to depend on unrelated interfaces.

Usage:
    from victor.core.verticals.protocols.focused.prompt_extensions import (
        PromptExtensionsProtocol,
    )

    class CodingPromptExtensions(PromptExtensionsProtocol):
        def __init__(self):
            self.prompt_contributors = [CodingPromptContributor()]
            self.enrichment_strategy = CodingEnrichmentStrategy()

Related Protocols:
    - PromptContributorProtocol: The underlying protocol for prompt contribution
    - EnrichmentStrategyProtocol: The underlying protocol for prompt enrichment
    - PromptExtensionsProtocol: Focused protocol for prompt-related extensions
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Protocol, runtime_checkable

from victor.core.verticals.protocols.enrichment import EnrichmentStrategyProtocol
from victor.core.verticals.protocols.prompt_provider import (
    PromptContributorProtocol,
    TaskTypeHint,
    collect_prompt_section_contributions,
)

# =============================================================================
# Prompt Extensions Protocol
# =============================================================================


@runtime_checkable
class PromptExtensionsProtocol(Protocol):
    """Protocol for prompt-related vertical extensions.

    This focused protocol contains ONLY prompt-related fields extracted
    from the larger VerticalExtensions interface. This follows the Interface
    Segregation Principle (ISP), allowing verticals to implement prompt
    extensions without depending on unrelated interfaces.

    The protocol provides access to:
    - prompt_contributors: List of prompt contributors for domain-specific
      task hints and system prompt sections
    - enrichment_strategy: Optional prompt enrichment strategy for DSPy-like
      auto prompt optimization with vertical-specific context

    Example:
        class CodingPromptExtensions(PromptExtensionsProtocol):
            def __init__(self):
                self.prompt_contributors = [CodingPromptContributor()]
                self.enrichment_strategy = CodingEnrichmentStrategy()

            def get_all_task_hints(self) -> Dict[str, TaskTypeHint]:
                merged = {}
                for contributor in sorted(
                    self.prompt_contributors,
                    key=lambda c: c.get_priority()
                ):
                    merged.update(contributor.get_task_type_hints())
                return merged
    """

    @property
    @abstractmethod
    def prompt_contributors(self) -> List[PromptContributorProtocol]:
        """Get the list of prompt contributors.

        Returns:
            List of prompt contributors for domain-specific task hints and
            system prompt sections. Empty list if none configured.
        """
        ...

    @property
    @abstractmethod
    def enrichment_strategy(self) -> EnrichmentStrategyProtocol | None:
        """Get the enrichment strategy for prompt optimization.

        Returns:
            Enrichment strategy for DSPy-like auto prompt optimization,
            or None if not configured.
        """
        ...

    def get_all_task_hints(self) -> Dict[str, TaskTypeHint]:
        """Merge task hints from all contributors.

        Later contributors override earlier ones for same task type.

        Returns:
            Merged dict of task type hints
        """
        merged: Dict[str, TaskTypeHint] = {}
        for contributor in sorted(self.prompt_contributors, key=lambda c: c.get_priority()):
            merged.update(contributor.get_task_type_hints())
        return merged

    def get_all_system_prompt_sections(self) -> List[str]:
        """Collect all system prompt sections from contributors.

        Returns:
            Combined list of system prompt sections (excluding empty strings)
        """
        sections = []
        for contributor in sorted(self.prompt_contributors, key=lambda c: c.get_priority()):
            for contribution in collect_prompt_section_contributions(contributor):
                if contribution.text:
                    sections.append(contribution.text)
        return sections


__all__ = [
    "PromptExtensionsProtocol",
]
