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

"""Prompt section budget allocator for token-efficient prompt construction.

Based on arXiv research on prompt compression and token efficiency:
- Prompt Compression in the Wild (arXiv:2604.02985)
- AgentGate: Lightweight Structured Routing (arXiv:2604.06696)
- Runtime Burden Allocation (arXiv:2604.01235)

Optimizes prompt construction under token budget constraints by:
1. Scoring each prompt section by relevance to current context
2. Sorting by relevance × inverse_token_cost
3. Selecting sections until budget exhausted
4. Using edge model for relevance scoring (no LLM needed)
5. Caching selection decisions for similar contexts

Target: 2-3x token reduction with < 5% quality degradation.
"""

from __future__ import annotations

import hashlib
from victor.core.json_utils import json_dumps
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PromptSection(Enum):
    """Standard prompt sections for Victor agents."""

    # Core sections (always included)
    SYSTEM_ROLE = "system_role"
    TASK_DEFINITION = "task_definition"

    # Guidance sections (optional, can be pruned)
    GROUNDING_RULES = "grounding_rules"
    TOOL_GUIDANCE = "tool_guidance"
    COMPLETION_SIGNALING = "completion_signaling"
    OUTPUT_FORMAT = "output_format"
    ERROR_HANDLING = "error_handling"
    CONTEXT_AWARENESS = "context_awareness"

    # Enhancement sections (expensive, prune aggressively)
    FEW_SHOT_EXAMPLES = "few_shot_examples"
    RECOVERY_STRATEGIES = "recovery_strategies"
    OPTIMIZATION_HINTS = "optimization_hints"
    SYNTHESIS_RULES = "synthesis_rules"


_REGISTRY_SECTION_EQUIVALENTS: Dict[PromptSection, set[str]] = {
    PromptSection.GROUNDING_RULES: {"GROUNDING_RULES", "GROUNDING_RULES_EXTENDED"},
    PromptSection.TOOL_GUIDANCE: {
        "ASI_TOOL_EFFECTIVENESS_GUIDANCE",
        "PARALLEL_READ_GUIDANCE",
        "LARGE_FILE_PAGINATION_GUIDANCE",
    },
    PromptSection.COMPLETION_SIGNALING: {"COMPLETION_GUIDANCE"},
    PromptSection.FEW_SHOT_EXAMPLES: {"FEW_SHOT_EXAMPLES"},
    PromptSection.SYNTHESIS_RULES: {"INIT_SYNTHESIS_RULES"},
}


@dataclass
class SectionMetadata:
    """Metadata about a prompt section."""

    name: str
    content: str
    token_cost: int
    relevance_score: float = 0.5  # 0.0 to 1.0
    priority: int = 5  # 1=highest, 10=lowest
    last_used: float = 0.0
    usage_count: int = 0
    category: str = "optional"  # "core", "guidance", "enhancement"

    def value_score(self) -> float:
        """Calculate value score = relevance / cost.

        Higher is better - balances relevance and token cost.
        """
        if self.token_cost == 0:
            return self.relevance_score * 10.0  # Free sections get high score

        # Value = relevance × (1 / cost) × priority_weight
        # Lower priority number = higher priority
        priority_weight = 11.0 - self.priority
        return self.relevance_score * (1000.0 / self.token_cost) * (priority_weight / 10.0)


@dataclass
class SectionMeasurement:
    """Rolling observed token cost for a prompt section."""

    average_token_cost: float = 0.0
    sample_count: int = 0
    last_measured: float = 0.0

    def record(self, token_cost: int) -> None:
        """Update the rolling average with a newly observed token cost."""
        self.average_token_cost = ((self.average_token_cost * self.sample_count) + token_cost) / (
            self.sample_count + 1
        )
        self.sample_count += 1
        self.last_measured = time.time()


class PromptSectionBudgetAllocator:
    """Allocates prompt sections under token budget constraints.

    Uses a scoring algorithm to select which prompt sections to include:
    1. Core sections are always included (if budget permits)
    2. Guidance sections included if relevance > threshold
    3. Enhancement sections compete on value_score
    4. Sections sorted by value_score descending
    5. Select until budget exhausted

    Example:
        allocator = PromptSectionBudgetAllocator(max_tokens=8000)

        sections = {
            "grounding_rules": SectionMetadata(...),
            "tool_guidance": SectionMetadata(...),
        }

        selected = allocator.allocate(
            sections=sections,
            context={"task": "code_search", "user_query": "Find bugs"},
        )
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        min_tokens: int = 1000,
        core_section_threshold: float = 0.3,
        guidance_section_threshold: float = 0.6,
        enhancement_section_threshold: float = 0.8,
        cache_selections: bool = True,
        cache_ttl_hours: float = 1.0,
    ):
        """Initialize prompt section budget allocator.

        Args:
            max_tokens: Maximum token budget for prompt
            min_tokens: Minimum tokens to reserve for core content
            core_section_threshold: Min relevance for core sections
            guidance_section_threshold: Min relevance for guidance sections
            enhancement_section_threshold: Min relevance for enhancement sections
            cache_selections: Whether to cache selection decisions
            cache_ttl_hours: How long to cache selections (default: 1 hour)
        """
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.core_threshold = core_section_threshold
        self.guidance_threshold = guidance_section_threshold
        self.enhancement_threshold = enhancement_section_threshold
        self.cache_selections = cache_selections
        self.cache_ttl_seconds = cache_ttl_hours * 3600

        # Selection cache: context_hash -> {section_names, timestamp}
        self._selection_cache: Dict[str, Dict[str, Any]] = {}
        self._section_measurements: Dict[str, SectionMeasurement] = {}

    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Generate hash for caching selection decisions.

        Args:
            context: Decision context

        Returns:
            Hash string for cache key
        """
        # Normalize context for consistent hashing
        normalized = {
            k: sorted(v) if isinstance(v, list) else v
            for k, v in sorted(context.items())
            if k not in ["timestamp", "request_id"]
        }
        context_str = json_dumps(normalized, sort_keys=True, default=str)
        return hashlib.md5(context_str.encode()).hexdigest()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count (roughly 4 chars per token)
        """
        # Rough estimation: ~4 characters per token
        return len(text) // 4

    def _effective_token_cost(self, section_name: str, metadata: SectionMetadata) -> int:
        """Use measured token cost when available, else fall back to static metadata."""
        measurement = self._section_measurements.get(section_name)
        if measurement and measurement.average_token_cost > 0:
            return max(1, int(round(measurement.average_token_cost)))
        return metadata.token_cost

    def _value_score(self, section_name: str, metadata: SectionMetadata) -> float:
        """Calculate section value while respecting measured token costs."""
        effective_cost = self._effective_token_cost(section_name, metadata)
        if effective_cost == 0:
            return metadata.relevance_score * 10.0

        priority_weight = 11.0 - metadata.priority
        return metadata.relevance_score * (1000.0 / effective_cost) * (priority_weight / 10.0)

    def _score_section_relevance(
        self,
        section_name: str,
        context: Dict[str, Any],
    ) -> float:
        """Score section relevance to current context using edge model.

        Args:
            section_name: Name of the section (string or enum value)
            context: Decision context

        Returns:
            Relevance score 0.0 to 1.0
        """
        # For now, use keyword-based heuristics (no LLM needed)
        # In production, this could use edge model embeddings

        task = context.get("task", "").lower()
        user_query = context.get("user_query", "").lower()
        mode = context.get("mode", "").lower()

        # Normalize section_name to match enum values if it's an enum
        # Handle both string names and PromptSection enum values
        section_key = _normalize_prompt_section(section_name)

        # Keyword relevance mapping
        section_keywords = {
            PromptSection.GROUNDING_RULES: {
                "hallucinate": 0.9,
                "invent": 0.9,
                "file": 0.7,
                "tool": 0.6,
                "search": 0.5,
            },
            PromptSection.TOOL_GUIDANCE: {
                "tool": 0.9,
                "call": 0.8,
                "function": 0.7,
                "use": 0.6,
            },
            PromptSection.ERROR_HANDLING: {
                "error": 0.9,
                "fail": 0.8,
                "exception": 0.7,
                "try": 0.5,
                "catch": 0.5,
            },
            PromptSection.COMPLETION_SIGNALING: {
                "done": 0.8,
                "complete": 0.7,
                "finish": 0.7,
                "task": 0.6,
            },
            PromptSection.FEW_SHOT_EXAMPLES: {
                "example": 0.7,
                "sample": 0.7,
                "demonstrate": 0.6,
            },
            PromptSection.RECOVERY_STRATEGIES: {
                "retry": 0.7,
                "recover": 0.7,
                "fallback": 0.6,
                "alternative": 0.5,
            },
        }

        keywords = section_keywords.get(section_key, {})

        # Check keyword matches in context
        score = 0.5  # Base score
        for keyword, kw_score in keywords.items():
            if keyword in task or keyword in user_query:
                score = max(score, kw_score)

        # Boost score based on mode
        if mode == "bug":
            if section_key in [
                PromptSection.ERROR_HANDLING,
                PromptSection.RECOVERY_STRATEGIES,
            ]:
                score = min(1.0, score + 0.3)
        elif mode == "search":
            if section_key in [
                PromptSection.GROUNDING_RULES,
                PromptSection.TOOL_GUIDANCE,
            ]:
                score = min(1.0, score + 0.2)

        return score

    def allocate(
        self,
        sections: Dict[str, SectionMetadata],
        context: Dict[str, Any],
    ) -> List[str]:
        """Allocate prompt sections under budget constraints.

        Args:
            sections: Dict of section_name -> SectionMetadata
            context: Decision context (task, user_query, mode, etc.)

        Returns:
            List of selected section names in priority order
        """
        # Check cache
        if self.cache_selections:
            context_hash = self._hash_context(context)
            cached = self._selection_cache.get(context_hash)
            if cached and time.time() - cached["timestamp"] < self.cache_ttl_seconds:
                logger.debug(f"Using cached section selection: {cached['sections']}")
                return cached["sections"]

        # Update relevance scores based on context
        for section_name, metadata in sections.items():
            metadata.relevance_score = self._score_section_relevance(section_name, context)
            metadata.last_used = time.time()

        # Separate sections by category
        core_sections = []
        guidance_sections = []
        enhancement_sections = []

        for name, metadata in sections.items():
            if metadata.category == "core":
                core_sections.append((name, metadata))
            elif metadata.category == "guidance":
                guidance_sections.append((name, metadata))
            else:
                enhancement_sections.append((name, metadata))

        # Always include core sections if above threshold and budget permits
        selected: List[Tuple[str, SectionMetadata]] = []
        used_tokens = 0

        for name, metadata in core_sections:
            if metadata.relevance_score >= self.core_threshold:
                effective_cost = self._effective_token_cost(name, metadata)
                if used_tokens + effective_cost <= self.max_tokens:
                    selected.append((name, metadata))
                    used_tokens += effective_cost
                    metadata.usage_count += 1
                else:
                    logger.warning(
                        f"Core section '{name}' ({effective_cost} tokens) "
                        f"exceeds remaining budget {self.max_tokens - used_tokens}"
                    )
                    break

        # Add guidance sections if above threshold and budget permits
        for name, metadata in sorted(
            guidance_sections,
            key=lambda x: self._value_score(x[0], x[1]),
            reverse=True,
        ):
            if metadata.relevance_score >= self.guidance_threshold:
                effective_cost = self._effective_token_cost(name, metadata)
                if used_tokens + effective_cost <= self.max_tokens:
                    selected.append((name, metadata))
                    used_tokens += effective_cost
                    metadata.usage_count += 1
                else:
                    break

        # Add enhancement sections based on value_score
        remaining_budget = self.max_tokens - used_tokens
        if remaining_budget > self.min_tokens:
            for name, metadata in sorted(
                enhancement_sections,
                key=lambda x: self._value_score(x[0], x[1]),
                reverse=True,
            ):
                if metadata.relevance_score >= self.enhancement_threshold:
                    effective_cost = self._effective_token_cost(name, metadata)
                    if used_tokens + effective_cost <= self.max_tokens:
                        selected.append((name, metadata))
                        used_tokens += effective_cost
                        metadata.usage_count += 1
                    else:
                        break

        # Extract section names in selection order
        section_names = [name for name, _ in selected]

        # Cache selection
        if self.cache_selections:
            self._selection_cache[context_hash] = {
                "sections": section_names,
                "timestamp": time.time(),
                "tokens_used": used_tokens,
            }

            # Clean up old cache entries
            self._cleanup_cache()

        logger.info(
            f"Prompt budget allocator: selected {len(section_names)}/{len(sections)} sections, "
            f"{used_tokens}/{self.max_tokens} tokens used"
        )

        return section_names

    def _cleanup_cache(self) -> None:
        """Remove expired cache entries."""
        now = time.time()
        expired = [
            key
            for key, entry in self._selection_cache.items()
            if now - entry["timestamp"] > self.cache_ttl_seconds
        ]

        for key in expired:
            del self._selection_cache[key]

        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired cache entries")

    def record_section_measurements(self, measured_token_costs: Dict[str, int]) -> None:
        """Record observed token costs for prompt sections.

        New measurements clear the selection cache because section fit and value
        decisions may change once observed costs replace static estimates.
        """
        for section_name, token_cost in measured_token_costs.items():
            if token_cost <= 0:
                continue
            measurement = self._section_measurements.setdefault(section_name, SectionMeasurement())
            measurement.record(int(token_cost))

        if measured_token_costs:
            self._selection_cache.clear()

    def get_section_measurements(self) -> Dict[str, Dict[str, Any]]:
        """Return recorded prompt-section measurements for observability."""
        return {
            name: {
                "average_token_cost": measurement.average_token_cost,
                "sample_count": measurement.sample_count,
                "last_measured": measurement.last_measured,
            }
            for name, measurement in self._section_measurements.items()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get allocator statistics."""
        measurement_samples = sum(
            measurement.sample_count for measurement in self._section_measurements.values()
        )
        return {
            "cache_entries": len(self._selection_cache),
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "measured_sections": len(self._section_measurements),
            "measurement_samples": measurement_samples,
        }


# Convenience functions


def allocate_prompt_sections(
    sections: Dict[str, SectionMetadata],
    context: Dict[str, Any],
    max_tokens: int = 8000,
) -> List[str]:
    """Allocate prompt sections under budget (convenience function).

    Args:
        sections: Dict of section_name -> SectionMetadata
        context: Decision context
        max_tokens: Maximum token budget

    Returns:
        List of selected section names

    Example:
        sections = {
            "grounding_rules": SectionMetadata(
                name="grounding_rules",
                content=GROUNDING_RULES_TEXT,
                token_cost=150,
                priority=3,
                category="guidance",
            ),
        }

        selected = allocate_prompt_sections(
            sections=sections,
            context={"task": "search", "user_query": "Find functions"},
            max_tokens=8000,
        )
    """
    allocator = PromptSectionBudgetAllocator(max_tokens=max_tokens)
    return allocator.allocate(sections, context)


def create_section_metadata(
    name: str,
    content: str,
    category: str = "optional",
    priority: int = 5,
) -> SectionMetadata:
    """Create SectionMetadata with automatic token estimation.

    Args:
        name: Section name
        content: Section content
        category: Section category (core, guidance, enhancement)
        priority: Priority level (1=highest, 10=lowest)

    Returns:
        SectionMetadata with estimated token cost

    Example:
        section = create_section_metadata(
            name="grounding_rules",
            content=GROUNDING_RULES_TEXT,
            category="guidance",
            priority=3,
        )
    """
    allocator = PromptSectionBudgetAllocator()
    token_cost = allocator._estimate_tokens(content)

    return SectionMetadata(
        name=name,
        content=content,
        token_cost=token_cost,
        priority=priority,
        category=category,
    )


def create_section_metadata_from_registry(
    section_name: str | PromptSection,
    *,
    content: Optional[str] = None,
    category: Optional[str] = None,
    priority: Optional[int] = None,
) -> SectionMetadata:
    """Create allocator metadata from the unified prompt section registry.

    Args:
        section_name: Canonical registry section name or legacy PromptSection enum.
        content: Optional content override. Defaults to registry static text.
        category: Optional allocator category override.
        priority: Optional allocator priority override.

    Returns:
        SectionMetadata aligned to the unified prompt section registry.

    Raises:
        ValueError: If the section cannot be resolved through the registry bridge.
    """
    from victor.agent.prompt_section_registry import (
        SectionCategory,
        get_section_registry,
    )

    canonical_name = _resolve_registry_section_name(section_name)
    if not canonical_name:
        raise ValueError(f"Unknown prompt section for registry-backed metadata: {section_name}")

    section = get_section_registry().get(canonical_name)
    if section is None:
        raise ValueError(f"Section not registered: {canonical_name}")

    resolved_content = content if content is not None else section.default_text

    if category is None:
        if section.required:
            resolved_category = "core"
        elif section.category in {SectionCategory.FEW_SHOT, SectionCategory.SYNTHESIS}:
            resolved_category = "enhancement"
        else:
            resolved_category = "guidance"
    else:
        resolved_category = category

    resolved_priority = (
        priority if priority is not None else max(1, min(10, int(round(section.priority / 10.0))))
    )

    return create_section_metadata(
        name=section.name,
        content=resolved_content,
        category=resolved_category,
        priority=resolved_priority,
    )


def _normalize_prompt_section(section_name: str | PromptSection) -> str | PromptSection:
    """Normalize legacy or canonical prompt section names for allocator heuristics."""
    if isinstance(section_name, PromptSection):
        return section_name

    normalized = str(section_name or "").strip()
    if not normalized:
        return normalized

    for section in PromptSection:
        if section.value == normalized:
            return section

    registry_name = _resolve_registry_section_name(normalized)
    if not registry_name:
        return normalized

    for section, equivalents in _REGISTRY_SECTION_EQUIVALENTS.items():
        if registry_name in equivalents:
            return section

    return normalized


def _resolve_registry_section_name(section_name: str | PromptSection) -> Optional[str]:
    """Resolve a legacy or canonical section reference into a registry section name."""
    from victor.agent.prompt_section_registry import get_section_registry

    if isinstance(section_name, PromptSection):
        equivalents = _REGISTRY_SECTION_EQUIVALENTS.get(section_name, set())
        if len(equivalents) == 1:
            return next(iter(equivalents))
        if equivalents:
            return sorted(equivalents)[0]
        return None

    normalized = str(section_name or "").strip()
    if not normalized:
        return None

    registry_section = get_section_registry().get(normalized)
    if registry_section is not None:
        return registry_section.name

    for section in PromptSection:
        if section.value == normalized:
            return _resolve_registry_section_name(section)

    return None
