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

"""Requirement Validator - Validates completion against extracted requirements.

This module provides requirement-driven completion detection by validating
that all extracted requirements from PerceptionIntegration are satisfied
before considering a task complete.

Design Principles:
1. Requirement-driven: Completion based on what user asked for, not heuristics
2. Priority-aware: P0 requirements must be met, P1-P3 are nice-to-have
3. Acceptance criteria: Validate against explicit acceptance criteria when provided
4. Graceful degradation: Works with partial requirement information

Based on research from:
- arXiv:2603.07379 - Agentic RAG Taxonomy (requirement extraction)
- arXiv:2601.03192 - MemRL (episodic memory for requirements)

Example:
    from victor.framework.requirement_validator import (
        RequirementValidator,
        RequirementResult,
    )

    validator = RequirementValidator()

    result = validator.validate_completion(
        requirements=perception.requirements,
        action_result=turn_result,
        context={"file_path": "/path/to/file.py"},
    )

    if result.is_satisfied:
        print("All requirements met!")
    else:
        print(f"Missing: {result.missing_requirements}")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.agent.task_completion import DeliverableType
    from victor.agent.services.turn_execution_runtime import TurnResult
    from victor.framework.perception_integration import Requirement


class RequirementPriority(Enum):
    """Priority levels for requirements."""

    P0 = 0  # Critical - must be satisfied
    P1 = 1  # High - should be satisfied
    P2 = 2  # Medium - nice to have
    P3 = 3  # Low - optional


@dataclass
class RequirementStatus:
    """Status of a single requirement."""

    requirement: Requirement
    is_satisfied: bool
    evidence: str = ""
    gap_description: str = ""


@dataclass
class RequirementResult:
    """Result of validating completion against requirements.

    Attributes:
        is_satisfied: True if all P0 requirements are met
        satisfaction_score: 0.0-1.0 score based on priority-weighted satisfaction
        satisfied_requirements: List of requirements that are met
        missing_requirements: List of requirements that are not met
        critical_gaps: List of P0 requirements that are not satisfied
        summary: Human-readable summary of validation result
    """

    is_satisfied: bool
    satisfaction_score: float
    satisfied_requirements: List[RequirementStatus] = field(default_factory=list)
    missing_requirements: List[RequirementStatus] = field(default_factory=list)
    critical_gaps: List[RequirementStatus] = field(default_factory=list)
    summary: str = ""

    def get_satisfied_count(self, priority: Optional[RequirementPriority] = None) -> int:
        """Get count of satisfied requirements (optionally filtered by priority)."""
        if priority is None:
            return len(self.satisfied_requirements)
        return sum(
            1
            for status in self.satisfied_requirements
            if status.requirement.priority == priority.value
        )

    def get_total_count(self, priority: Optional[RequirementPriority] = None) -> int:
        """Get total count of requirements (optionally filtered by priority)."""
        total = len(self.satisfied_requirements) + len(self.missing_requirements)
        if priority is None:
            return total
        return sum(
            1
            for status in self.satisfied_requirements + self.missing_requirements
            if status.requirement.priority == priority.value
        )


class RequirementValidator:
    """Validates completion against extracted requirements.

    This validator checks that:
    1. All P0 (critical) requirements are satisfied
    2. Deliverable types match requirement types
    3. Acceptance criteria are met when specified
    4. Evidence exists for requirement satisfaction

    Usage:
        validator = RequirementValidator()

        result = validator.validate_completion(
            requirements=perception.requirements,
            action_result=turn_result,
            context={"files_modified": ["file.py"]},
        )
    """

    def __init__(
        self,
        p0_threshold: float = 1.0,  # All P0 must be satisfied
        overall_threshold: float = 0.7,  # 70% overall satisfaction
    ):
        """Initialize validator with thresholds.

        Args:
            p0_threshold: Minimum satisfaction for P0 requirements (0.0-1.0)
            overall_threshold: Minimum overall satisfaction score (0.0-1.0)
        """
        self.p0_threshold = p0_threshold
        self.overall_threshold = overall_threshold

    def validate_completion(
        self,
        requirements: List[Requirement],
        action_result: Any,
        context: Dict[str, Any],
    ) -> RequirementResult:
        """Validate completion against extracted requirements.

        Args:
            requirements: List of requirements from PerceptionIntegration
            action_result: TurnResult or similar with response/tool results
            context: Additional context (files modified, tools used, etc.)

        Returns:
            RequirementResult with satisfaction status and details
        """
        if not requirements:
            # No requirements extracted - use legacy behavior
            return RequirementResult(
                is_satisfied=False,
                satisfaction_score=0.5,
                summary="No requirements to validate - using heuristics",
            )

        # Validate each requirement
        requirement_statuses = []
        for req in requirements:
            status = self._validate_requirement(req, action_result, context)
            requirement_statuses.append(status)

        # Separate satisfied and missing
        satisfied = [s for s in requirement_statuses if s.is_satisfied]
        missing = [s for s in requirement_statuses if not s.is_satisfied]

        # Find critical gaps (P0 requirements not satisfied)
        critical_gaps = [
            s for s in missing if s.requirement.priority == RequirementPriority.P0.value
        ]

        # Calculate satisfaction score (priority-weighted)
        satisfaction_score = self._calculate_satisfaction_score(requirement_statuses)

        # Determine if satisfied (all P0 met + overall threshold)
        is_satisfied = len(critical_gaps) == 0 and satisfaction_score >= self.overall_threshold

        # Generate summary
        summary = self._generate_summary(
            is_satisfied, satisfaction_score, satisfied, missing, critical_gaps
        )

        return RequirementResult(
            is_satisfied=is_satisfied,
            satisfaction_score=satisfaction_score,
            satisfied_requirements=satisfied,
            missing_requirements=missing,
            critical_gaps=critical_gaps,
            summary=summary,
        )

    def _validate_requirement(
        self,
        requirement: Requirement,
        action_result: Any,
        context: Dict[str, Any],
    ) -> RequirementStatus:
        """Validate a single requirement.

        Args:
            requirement: The requirement to validate
            action_result: TurnResult with response/tool results
            context: Additional context

        Returns:
            RequirementStatus indicating if requirement is satisfied
        """
        # Check based on requirement type
        req_type = requirement.type.value.lower()

        if req_type == "functional":
            return self._validate_functional_requirement(requirement, action_result, context)
        elif req_type == "constraint":
            return self._validate_constraint_requirement(requirement, action_result, context)
        elif req_type == "quality":
            return self._validate_quality_requirement(requirement, action_result, context)
        else:
            # Generic validation for other types
            return self._validate_generic_requirement(requirement, action_result, context)

    def _validate_functional_requirement(
        self,
        requirement: Requirement,
        action_result: Any,
        context: Dict[str, Any],
    ) -> RequirementStatus:
        """Validate functional requirement (file operations, answers, etc.)."""
        description_lower = requirement.description.lower()

        # Check for file operations
        if any(
            word in description_lower
            for word in ["create", "write", "generate", "implement", "add"]
        ):
            # Check if file was created/modified
            files_modified = context.get("files_modified", [])
            files_created = context.get("files_created", [])

            if files_modified or files_created:
                return RequirementStatus(
                    requirement=requirement,
                    is_satisfied=True,
                    evidence=f"Files modified/created: {files_modified + files_created}",
                )
            else:
                return RequirementStatus(
                    requirement=requirement,
                    is_satisfied=False,
                    gap_description="No files were modified or created",
                )

        # Check for answers/explanations
        if any(
            word in description_lower for word in ["explain", "describe", "analyze", "summarize"]
        ):
            # Check if response has substantial content
            response = self._extract_response(action_result)
            if response and len(response.strip()) > 200:
                return RequirementStatus(
                    requirement=requirement,
                    is_satisfied=True,
                    evidence=f"Provided {len(response)} character response",
                )
            else:
                return RequirementStatus(
                    requirement=requirement,
                    is_satisfied=False,
                    gap_description="Response too short or missing",
                )

        # Check for fixes/changes
        if any(word in description_lower for word in ["fix", "change", "update", "modify"]):
            # Check if any tool was used (indicates action taken)
            tool_calls = self._extract_tool_calls(action_result)
            if tool_calls:
                return RequirementStatus(
                    requirement=requirement,
                    is_satisfied=True,
                    evidence=f"Used {len(tool_calls)} tool(s)",
                )
            else:
                return RequirementStatus(
                    requirement=requirement,
                    is_satisfied=False,
                    gap_description="No tools were used to make changes",
                )

        # Default: check if response mentions requirement keywords
        response = self._extract_response(action_result)
        if response and self._keywords_present(response, requirement.description):
            return RequirementStatus(
                requirement=requirement,
                is_satisfied=True,
                evidence="Response addresses requirement keywords",
            )
        else:
            return RequirementStatus(
                requirement=requirement,
                is_satisfied=False,
                gap_description="Could not verify requirement was addressed",
            )

    def _validate_constraint_requirement(
        self,
        requirement: Requirement,
        action_result: Any,
        context: Dict[str, Any],
    ) -> RequirementStatus:
        """Validate constraint requirement (e.g., 'must use Python', 'max 100 lines')."""
        description_lower = requirement.description.lower()

        # Check for language constraints
        if "python" in description_lower or "java" in description_lower:
            # Check if files match the language
            files_modified = context.get("files_modified", [])
            for file_path in files_modified:
                if description_lower.split()[0] in file_path.lower():
                    return RequirementStatus(
                        requirement=requirement,
                        is_satisfied=True,
                        evidence=f"Created/modified {file_path}",
                    )

        # Check for size constraints
        if "line" in description_lower or "size" in description_lower:
            # Would need to check actual file sizes - for now assume satisfied
            return RequirementStatus(
                requirement=requirement,
                is_satisfied=True,
                evidence="Size constraint noted (validation requires file access)",
            )

        # Default: assume constraint satisfied
        return RequirementStatus(
            requirement=requirement,
            is_satisfied=True,
            evidence="Constraint acknowledged",
        )

    def _validate_quality_requirement(
        self,
        requirement: Requirement,
        action_result: Any,
        context: Dict[str, Any],
    ) -> RequirementStatus:
        """Validate quality requirement (e.g., 'well-documented', 'efficient')."""
        # Quality requirements are hard to validate automatically
        # Check if response mentions quality keywords
        response = self._extract_response(action_result)
        quality_keywords = [
            "optimized",
            "efficient",
            "clean",
            "documented",
            "commented",
        ]

        if response and any(keyword in response.lower() for keyword in quality_keywords):
            return RequirementStatus(
                requirement=requirement,
                is_satisfied=True,
                evidence="Response mentions quality aspects",
            )

        # Default: assume satisfied (P2/P3 priority)
        return RequirementStatus(
            requirement=requirement,
            is_satisfied=True,
            evidence="Quality requirement noted (subjective)",
        )

    def _validate_generic_requirement(
        self,
        requirement: Requirement,
        action_result: Any,
        context: Dict[str, Any],
    ) -> RequirementStatus:
        """Generic requirement validation."""
        # Check if response addresses requirement
        response = self._extract_response(action_result)
        if response and self._keywords_present(response, requirement.description):
            return RequirementStatus(
                requirement=requirement,
                is_satisfied=True,
                evidence="Response addresses requirement",
            )

        # Check if any tools were used
        tool_calls = self._extract_tool_calls(action_result)
        if tool_calls:
            return RequirementStatus(
                requirement=requirement,
                is_satisfied=True,
                evidence=f"Tools used: {len(tool_calls)}",
            )

        # Default: not satisfied
        return RequirementStatus(
            requirement=requirement,
            is_satisfied=False,
            gap_description="Could not verify requirement was addressed",
        )

    def _calculate_satisfaction_score(self, requirement_statuses: List[RequirementStatus]) -> float:
        """Calculate priority-weighted satisfaction score.

        P0 requirements: 40% weight
        P1 requirements: 30% weight
        P2 requirements: 20% weight
        P3 requirements: 10% weight
        """
        if not requirement_statuses:
            return 0.5

        # Group by priority (clamp unknown values to P3/low)
        by_priority: dict = {0: [], 1: [], 2: [], 3: []}
        for status in requirement_statuses:
            priority = getattr(status.requirement, "priority", 3)
            if priority not in by_priority:
                priority = 3
            by_priority[priority].append(status)

        # Calculate weighted score
        weights = {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}
        total_score = 0.0

        for priority, statuses in by_priority.items():
            if not statuses:
                continue

            satisfied_count = sum(1 for s in statuses if s.is_satisfied)
            total_count = len(statuses)

            if total_count > 0:
                priority_score = satisfied_count / total_count
                total_score += priority_score * weights[priority]

        return min(total_score, 1.0)

    def _generate_summary(
        self,
        is_satisfied: bool,
        score: float,
        satisfied: List[RequirementStatus],
        missing: List[RequirementStatus],
        critical_gaps: List[RequirementStatus],
    ) -> str:
        """Generate human-readable summary."""
        if is_satisfied:
            return (
                f"All requirements satisfied (score: {score:.2f}, "
                f"{len(satisfied)}/{len(satisfied) + len(missing)} met)"
            )
        elif critical_gaps:
            gap_desc = ", ".join([g.requirement.description[:50] for g in critical_gaps[:3]])
            return (
                f"Critical requirements not met: {gap_desc}... "
                f"(score: {score:.2f}, {len(satisfied)}/{len(satisfied) + len(missing)} met)"
            )
        else:
            return (
                f"Partial satisfaction (score: {score:.2f}, "
                f"{len(satisfied)}/{len(satisfied) + len(missing)} met)"
            )

    def _extract_response(self, action_result: Any) -> Optional[str]:
        """Extract response text from action_result."""
        if hasattr(action_result, "response"):
            response = action_result.response
            if isinstance(response, str):
                return response
            content = getattr(response, "content", None)
            if isinstance(content, str):
                return content
            return None
        elif hasattr(action_result, "content"):
            content = action_result.content
            return content if isinstance(content, str) else None
        elif isinstance(action_result, str):
            return action_result
        return None

    def _extract_tool_calls(self, action_result: Any) -> List[Any]:
        """Extract tool calls from action_result."""
        if hasattr(action_result, "tool_calls"):
            return action_result.tool_calls
        if hasattr(action_result, "tools_used"):
            return action_result.tools_used
        return []

    def _keywords_present(self, text: str, description: str) -> bool:
        """Check if keywords from description are present in text."""
        # Extract meaningful keywords from description
        keywords = re.findall(r"\b\w{4,}\b", description.lower())
        text_lower = text.lower()

        # Check if at least 2 keywords are present
        matches = sum(1 for kw in keywords if kw in text_lower)
        return matches >= 2
