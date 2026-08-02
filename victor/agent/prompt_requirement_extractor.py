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

"""Dynamic Prompt Requirement Extractor.

Extracts explicit requirements from user prompts to dynamically adjust
agent behavior. This enables the agent to respect user-specified counts
for files, fixes, findings, etc. without hardcoding.

Design Principles:
    - Open/Closed: New requirement patterns can be added without modifying core logic
    - Single Responsibility: Only extracts requirements, doesn't enforce them
    - Dependency Inversion: Returns data structure, consumers decide how to use it

Example:
    extractor = PromptRequirementExtractor()
    requirements = extractor.extract("Review 9 files and provide top 3 fixes")
    # requirements.file_count = 9
    # requirements.fix_count = 3
    # requirements.tool_budget = 27 (9 * 3 buffer)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class PromptRequirements:
    """Extracted requirements from a user prompt.

    All counts are Optional - None means not specified in prompt.
    Consumers should fall back to defaults when None.

    Attributes:
        file_count: Number of files to read/review (e.g., "read 9 files")
        fix_count: Number of fixes to provide (e.g., "top 3 fixes")
        finding_count: Number of findings to report (e.g., "6-10 findings")
        step_count: Number of steps to perform (e.g., "5 steps")
        example_count: Number of examples to provide
        recommendation_count: Number of recommendations
        tool_budget: Computed tool budget based on requirements
        iteration_budget: Computed iteration budget
        buffer_multiplier: Multiplier applied to budgets (default 1.5)
        confidence: Confidence in extraction (0.0-1.0)
        raw_matches: Raw regex matches for debugging
    """

    file_count: Optional[int] = None
    fix_count: Optional[int] = None
    finding_count: Optional[int] = None
    step_count: Optional[int] = None
    example_count: Optional[int] = None
    recommendation_count: Optional[int] = None

    # Computed budgets
    tool_budget: Optional[int] = None
    iteration_budget: Optional[int] = None

    # Metadata
    buffer_multiplier: float = 1.5
    confidence: float = 0.0
    raw_matches: Dict[str, List[str]] = field(default_factory=dict)

    def compute_budgets(
        self,
        default_tool_budget: int = 50,
        default_iteration_budget: int = 100,
        min_tools_per_file: int = 3,
    ) -> None:
        """Compute dynamic budgets based on extracted requirements.

        Uses the largest requirement to set budgets with buffer.

        Args:
            default_tool_budget: Default if no requirements found
            default_iteration_budget: Default iteration limit
            min_tools_per_file: Minimum tools needed per file (read + process)
        """
        # Find the largest file-related requirement
        file_requirements = [c for c in [self.file_count, self.finding_count] if c is not None]

        if file_requirements:
            max_files = max(file_requirements)
            # Each file needs ~3 tools (read, maybe grep/glob, process)
            base_budget = max_files * min_tools_per_file
            self.tool_budget = int(base_budget * self.buffer_multiplier)
            self.iteration_budget = int(max_files * 5 * self.buffer_multiplier)
        else:
            self.tool_budget = default_tool_budget
            self.iteration_budget = default_iteration_budget

        # Ensure minimums
        self.tool_budget = max(self.tool_budget, default_tool_budget)
        self.iteration_budget = max(self.iteration_budget, default_iteration_budget)

    def has_explicit_requirements(self) -> bool:
        """Check if any explicit requirements were found."""
        return any(
            [
                self.file_count,
                self.fix_count,
                self.finding_count,
                self.step_count,
                self.example_count,
                self.recommendation_count,
            ]
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "file_count": self.file_count,
            "fix_count": self.fix_count,
            "finding_count": self.finding_count,
            "step_count": self.step_count,
            "example_count": self.example_count,
            "recommendation_count": self.recommendation_count,
            "tool_budget": self.tool_budget,
            "iteration_budget": self.iteration_budget,
            "confidence": self.confidence,
            "has_explicit": self.has_explicit_requirements(),
        }


# =============================================================================
# Requirement Patterns (OCP: Add new patterns without modifying extractor)
# =============================================================================


@dataclass
class RequirementPattern:
    """Pattern for extracting a specific requirement type.

    Attributes:
        name: Requirement name (e.g., "file_count")
        patterns: List of regex patterns to try
        extractor: Function to extract value from match
        priority: Higher priority patterns are tried first
    """

    name: str
    patterns: List[str]
    priority: int = 50

    def compile(self) -> List[Pattern]:
        """Compile all patterns."""
        return [re.compile(p, re.IGNORECASE) for p in self.patterns]


# Default patterns for common requirements
DEFAULT_PATTERNS: List[RequirementPattern] = [
    # File count patterns
    RequirementPattern(
        name="file_count",
        patterns=[
            r"(?:read|review|analyze|check|examine)\s+(?:the\s+)?(?:following\s+)?(\d+)\s+files?",
            r"(\d+)\s+files?\s+(?:to\s+)?(?:read|review|analyze|check)",
            r"(?:following|these|the)\s+(\d+)\s+files?",
            r"read\s+(?:the\s+)?files?\s*[:\-]?\s*(?:\n.*?){0,2}?(\d+)\s+files?",
            r"files?\s*\(\s*(\d+)\s*\)",
        ],
        priority=100,
    ),
    # File list detection (count items in a file list)
    RequirementPattern(
        name="file_list_count",
        patterns=[
            # Count lines starting with "- " after "files:" or "following files first:"
            r"(?:following\s+)?files?(?:\s+\w+)*\s*:\s*\n((?:\s*-\s*[^\n]+\n?)+)",
        ],
        priority=95,
    ),
    # Fix count patterns
    RequirementPattern(
        name="fix_count",
        patterns=[
            r"top\s+(\d+)\s+fix(?:es)?",
            r"(\d+)\s+(?:key\s+)?fix(?:es)?",
            r"provide\s+(\d+)\s+fix(?:es)?",
            r"suggest\s+(\d+)\s+fix(?:es)?",
            r"list\s+(\d+)\s+fix(?:es)?",
        ],
        priority=90,
    ),
    # Finding count patterns
    RequirementPattern(
        name="finding_count",
        patterns=[
            r"(\d+)\s*[-–to]+\s*(\d+)\s+findings?",  # "6-10 findings" -> takes max
            r"(\d+)\s+findings?\s+max",
            r"prioritize\s+(\d+)\s+findings?",
            r"top\s+(\d+)\s+findings?",
            r"(\d+)\s+(?:key\s+)?findings?",
            r"identify\s+(\d+)\s+(?:key\s+)?(?:issues?|problems?|gaps?)",
        ],
        priority=85,
    ),
    # Recommendation count patterns
    RequirementPattern(
        name="recommendation_count",
        patterns=[
            r"top\s+(\d+)\s+recommend(?:ations?)?",
            r"(\d+)\s+recommend(?:ations?)?",
            r"provide\s+(\d+)\s+suggest(?:ions?)?",
        ],
        priority=80,
    ),
    # Step count patterns
    RequirementPattern(
        name="step_count",
        patterns=[
            r"(\d+)\s+steps?",
            r"step\s+(\d+)",
            r"in\s+(\d+)\s+steps?",
        ],
        priority=70,
    ),
    # Example count patterns
    RequirementPattern(
        name="example_count",
        patterns=[
            r"(\d+)\s+examples?",
            r"provide\s+(\d+)\s+examples?",
            r"show\s+(\d+)\s+examples?",
        ],
        priority=60,
    ),
]


# =============================================================================
# Prompt Requirement Extractor
# =============================================================================


class PromptRequirementExtractor:
    """Extracts explicit requirements from user prompts.

    Uses configurable regex patterns to identify numeric requirements
    like file counts, fix counts, etc. Computes dynamic budgets based
    on extracted requirements.

    Example:
        extractor = PromptRequirementExtractor()
        req = extractor.extract("Review 9 files and provide top 3 fixes")
        print(req.file_count)  # 9
        print(req.fix_count)   # 3
        print(req.tool_budget) # 40 (9 * 3 * 1.5 buffer)
    """

    def __init__(
        self,
        patterns: Optional[List[RequirementPattern]] = None,
        buffer_multiplier: float = 1.5,
    ):
        """Initialize extractor.

        Args:
            patterns: Custom patterns (defaults to DEFAULT_PATTERNS)
            buffer_multiplier: Multiplier for computed budgets
        """
        self._patterns = patterns or DEFAULT_PATTERNS
        self._buffer_multiplier = buffer_multiplier
        self._compiled: Dict[str, List[Pattern]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile all regex patterns."""
        # Sort by priority (higher first)
        sorted_patterns = sorted(self._patterns, key=lambda p: -p.priority)
        for pattern in sorted_patterns:
            self._compiled[pattern.name] = pattern.compile()

    def extract(self, prompt: str) -> PromptRequirements:
        """Extract requirements from a prompt.

        Args:
            prompt: User prompt text

        Returns:
            PromptRequirements with extracted values and computed budgets
        """
        requirements = PromptRequirements(buffer_multiplier=self._buffer_multiplier)
        raw_matches: Dict[str, List[str]] = {}
        match_count = 0
        total_patterns = 0

        for name, compiled_patterns in self._compiled.items():
            for pattern in compiled_patterns:
                total_patterns += 1
                matches = pattern.findall(prompt)
                if matches:
                    raw_matches.setdefault(name, []).extend(
                        [str(m) if isinstance(m, str) else str(m[0]) for m in matches]
                    )
                    # Extract the first numeric value (pass pattern name for special handling)
                    value = self._extract_value(matches, pattern_name=name)
                    if value is not None:
                        # Map file_list_count to file_count
                        target_name = "file_count" if name == "file_list_count" else name
                        self._set_requirement(requirements, target_name, value)
                        match_count += 1
                        break  # First match wins for each requirement type

        requirements.raw_matches = raw_matches

        # Compute confidence based on match ratio
        if requirements.has_explicit_requirements():
            requirements.confidence = min(1.0, match_count / 3)  # 3+ matches = full confidence
        else:
            requirements.confidence = 0.0

        # Compute dynamic budgets
        requirements.compute_budgets()

        if requirements.has_explicit_requirements():
            logger.info(
                f"PromptRequirementExtractor: Extracted requirements: "
                f"files={requirements.file_count}, fixes={requirements.fix_count}, "
                f"findings={requirements.finding_count}, "
                f"tool_budget={requirements.tool_budget}, confidence={requirements.confidence:.2f}"
            )

        return requirements

    def _extract_value(self, matches: List, pattern_name: str = "") -> Optional[int]:
        """Extract numeric value from regex matches.

        Handles both simple matches and tuple matches (from groups).
        For range matches like "6-10", returns the max value.
        For file_list_count, counts the number of list items.

        Args:
            matches: List of regex matches
            pattern_name: Name of the pattern (for special handling)

        Returns:
            Extracted integer value or None
        """
        if not matches:
            return None

        match = matches[0]

        # Special handling for file list counting
        if pattern_name == "file_list_count" and isinstance(match, str):
            # Count lines starting with "-" in the matched content
            lines = [line.strip() for line in match.split("\n") if line.strip().startswith("-")]
            return len(lines) if lines else None

        # Handle tuple matches (multiple groups)
        if isinstance(match, tuple):
            # For ranges like "6-10", take the max
            values = [int(v) for v in match if v and v.isdigit()]
            return max(values) if values else None

        # Handle simple string match
        if isinstance(match, str) and match.isdigit():
            return int(match)

        return None

    def _set_requirement(
        self,
        requirements: PromptRequirements,
        name: str,
        value: int,
    ) -> None:
        """Set a requirement value on the requirements object.

        Args:
            requirements: Requirements object to update
            name: Requirement name
            value: Value to set
        """
        if hasattr(requirements, name):
            setattr(requirements, name, value)

    def add_pattern(self, pattern: RequirementPattern) -> None:
        """Add a custom pattern (OCP: extend without modifying).

        Args:
            pattern: New pattern to add
        """
        self._patterns.append(pattern)
        self._compiled[pattern.name] = pattern.compile()


# =============================================================================
# Singleton and Factory
# =============================================================================

_extractor: Optional[PromptRequirementExtractor] = None


def get_prompt_requirement_extractor() -> PromptRequirementExtractor:
    """Get global prompt requirement extractor singleton.

    Returns:
        PromptRequirementExtractor instance
    """
    global _extractor
    if _extractor is None:
        _extractor = PromptRequirementExtractor()
    return _extractor


def extract_prompt_requirements(prompt: str) -> PromptRequirements:
    """Convenience function to extract requirements from a prompt.

    Args:
        prompt: User prompt text

    Returns:
        PromptRequirements with extracted values
    """
    return get_prompt_requirement_extractor().extract(prompt)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PromptRequirements",
    "PromptRequirementExtractor",
    "RequirementPattern",
    "get_prompt_requirement_extractor",
    "extract_prompt_requirements",
    "DEFAULT_PATTERNS",
]
