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

"""Complexity-based prompt hints for the enrichment pipeline.

This module provides prompt hints based on task complexity. These hints
are injected into prompts via the enrichment pipeline, following SRP
by keeping prompt engineering separate from classification.

Design Principles:
- SRP: Only handles prompt hint generation
- Classification is done by framework/task/complexity.py
- Hints flow through the enrichment pipeline, not hardcoded in classifier

Usage:
    from victor.framework.task import TaskComplexity
    from victor.framework.enrichment.strategies import ComplexityHintEnricher

    enricher = ComplexityHintEnricher()
    hint = enricher.get_hint(TaskComplexity.COMPLEX)
    # -> "[COMPLEX] Deep work needed. Examine code systematically..."
"""

from __future__ import annotations

from typing import Dict, Optional

from victor.framework.task.protocols import TaskComplexity

# Standard prompt hints per complexity level
COMPLEXITY_HINTS: Dict[TaskComplexity, str] = {
    TaskComplexity.SIMPLE: "[SIMPLE] Quick query. Focus on relevant tools. Answer concisely.",
    TaskComplexity.MEDIUM: "[MEDIUM] Moderate exploration. Be focused and efficient.",
    TaskComplexity.COMPLEX: "[COMPLEX] Deep work needed. Examine code systematically. Provide detailed answer.",
    TaskComplexity.GENERATION: "[GENERATE] Write code directly. Minimal exploration. Display or save as requested.",
    TaskComplexity.ACTION: "[ACTION] Execute task. Multiple tool calls allowed. Continue until complete.",
    TaskComplexity.ANALYSIS: "[ANALYSIS] Thorough exploration required. Examine all relevant modules. Comprehensive output.",
}

# Extended hints for models that benefit from more guidance (e.g., Ollama, LMStudio)
EXTENDED_COMPLEXITY_HINTS: Dict[TaskComplexity, str] = {
    TaskComplexity.SIMPLE: "[SIMPLE QUERY] Quick info retrieval. Focus on relevant tools.",
    TaskComplexity.MEDIUM: "[MEDIUM] Moderate exploration. Be focused and efficient.",
    TaskComplexity.COMPLEX: "[COMPLEX] Thorough work. Examine code systematically.",
    TaskComplexity.GENERATION: "[GENERATION] Direct code creation. Minimal exploration.",
    TaskComplexity.ACTION: "[ACTION] Multi-step execution. Continue until complete.",
    TaskComplexity.ANALYSIS: "[ANALYSIS] Comprehensive exploration. Examine all modules.",
}


class ComplexityHintEnricher:
    """Enricher that provides complexity-based hints for prompts.

    This class generates appropriate hints based on task complexity,
    with support for provider-specific hint styles.

    Example:
        enricher = ComplexityHintEnricher()
        hint = enricher.get_hint(TaskComplexity.COMPLEX)

        # For specific providers
        hint = enricher.get_hint(TaskComplexity.COMPLEX, provider="ollama")
    """

    # Providers that need extended hints
    EXTENDED_HINT_PROVIDERS = {"ollama", "lmstudio", "vllm"}

    # Providers that work well with standard hints
    STANDARD_HINT_PROVIDERS = {"anthropic", "openai", "google", "xai"}

    def __init__(
        self,
        hints: Optional[Dict[TaskComplexity, str]] = None,
        extended_hints: Optional[Dict[TaskComplexity, str]] = None,
    ) -> None:
        """Initialize the hint enricher.

        Args:
            hints: Custom standard hints (overrides COMPLEXITY_HINTS)
            extended_hints: Custom extended hints (overrides EXTENDED_COMPLEXITY_HINTS)
        """
        self.hints = hints or COMPLEXITY_HINTS.copy()
        self.extended_hints = extended_hints or EXTENDED_COMPLEXITY_HINTS.copy()

    def get_hint(
        self,
        complexity: TaskComplexity,
        extended: bool = False,
        provider: Optional[str] = None,
    ) -> str:
        """Get the appropriate hint for a complexity level.

        Args:
            complexity: The task complexity level
            extended: Whether to use extended hints
            provider: Optional provider name for provider-specific selection

        Returns:
            Prompt hint string for the complexity level
        """
        # Determine which hint set to use
        use_extended = extended

        if provider:
            provider_lower = provider.lower()
            if provider_lower in self.STANDARD_HINT_PROVIDERS:
                use_extended = False
            elif provider_lower in self.EXTENDED_HINT_PROVIDERS:
                use_extended = True

        # Return appropriate hint
        if use_extended:
            return self.extended_hints.get(complexity, self.hints.get(complexity, ""))
        return self.hints.get(complexity, "")

    def update_hint(self, complexity: TaskComplexity, hint: str, extended: bool = False) -> None:
        """Update a hint for a complexity level.

        Args:
            complexity: The complexity level to update
            hint: The new hint text
            extended: Whether to update extended hints
        """
        if extended:
            self.extended_hints[complexity] = hint
        else:
            self.hints[complexity] = hint


# Convenience function for quick access
def get_complexity_hint(
    complexity: TaskComplexity,
    extended: bool = False,
    provider: Optional[str] = None,
) -> str:
    """Get a complexity hint using the default enricher.

    Args:
        complexity: The task complexity level
        extended: Whether to use extended hints
        provider: Optional provider name

    Returns:
        Prompt hint string
    """
    return ComplexityHintEnricher().get_hint(complexity, extended, provider)
