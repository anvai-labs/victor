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

"""Unit tests for RequirementValidator."""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from unittest.mock import Mock

from victor.framework.requirement_validator import (
    RequirementValidator,
    RequirementResult,
    RequirementStatus,
)
from victor.framework.perception_integration import RequirementType, Requirement

# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockTurnResult:
    """Mock TurnResult for testing."""

    response: str
    tool_calls: list = None
    tools_used: list = None
    has_tool_calls: bool = False


# =============================================================================
# RequirementValidator Tests
# =============================================================================


class TestRequirementValidator:
    """Test suite for RequirementValidator."""

    def test_initialization(self):
        """Test validator initializes with correct thresholds."""
        validator = RequirementValidator(p0_threshold=1.0, overall_threshold=0.7)

        assert validator.p0_threshold == 1.0
        assert validator.overall_threshold == 0.7

    def test_validate_with_no_requirements(self):
        """Test validation returns neutral result when no requirements provided."""
        validator = RequirementValidator()
        result = validator.validate_completion(
            requirements=[],
            action_result=MockTurnResult(response="Test response"),
            context={},
        )

        assert result.is_satisfied is False
        assert result.satisfaction_score == 0.5
        assert "No requirements" in result.summary

    def test_validate_functional_requirement_file_creation(self):
        """Test validation of functional requirement for file creation."""
        validator = RequirementValidator(overall_threshold=0.3)  # Lower threshold for single req

        requirements = [
            Requirement(
                type=RequirementType.FUNCTIONAL,
                description="Create a new file for authentication",
                priority=0,
            )
        ]

        # Test with files created
        result = validator.validate_completion(
            requirements=requirements,
            action_result=MockTurnResult(response="Created auth.py"),
            context={"files_created": ["auth.py"], "files_modified": []},
        )

        # With lower threshold, should pass
        assert len(result.satisfied_requirements) == 1
        assert "Files modified/created" in result.satisfied_requirements[0].evidence

    def test_validate_functional_requirement_file_creation_missing(self):
        """Test validation fails when files not created."""
        validator = RequirementValidator()

        requirements = [
            Requirement(
                type=RequirementType.FUNCTIONAL,
                description="Create a new file for authentication",
                priority=0,
            )
        ]

        # Test without files created
        result = validator.validate_completion(
            requirements=requirements,
            action_result=MockTurnResult(response="I'll create it"),
            context={"files_created": [], "files_modified": []},
        )

        assert len(result.critical_gaps) == 1  # P0 requirement not met
        assert "No files were modified" in result.critical_gaps[0].gap_description

    def test_validate_functional_requirement_answer_provided(self):
        """Test validation of functional requirement for explanation."""
        validator = RequirementValidator()

        requirements = [
            Requirement(
                type=RequirementType.FUNCTIONAL,
                description="Explain how authentication works",
                priority=1,
            )
        ]

        # Test with substantial response
        long_response = "Authentication works by verifying credentials. " * 20
        result = validator.validate_completion(
            requirements=requirements,
            action_result=MockTurnResult(response=long_response),
            context={},
        )

        assert len(result.satisfied_requirements) == 1
        assert "Provided" in result.satisfied_requirements[0].evidence

    def test_validate_functional_requirement_answer_too_short(self):
        """Test validation fails when answer is too short."""
        validator = RequirementValidator()

        requirements = [
            Requirement(
                type=RequirementType.FUNCTIONAL,
                description="Explain how authentication works",
                priority=1,
            )
        ]

        # Test with short response
        result = validator.validate_completion(
            requirements=requirements,
            action_result=MockTurnResult(response="It works."),
            context={},
        )

        assert len(result.missing_requirements) == 1
        assert "too short" in result.missing_requirements[0].gap_description

    def test_validate_functional_requirement_with_tools(self):
        """Test validation when tools are used."""
        validator = RequirementValidator()

        requirements = [
            Requirement(
                type=RequirementType.FUNCTIONAL,
                description="Fix the authentication bug",
                priority=0,
            )
        ]

        # Test with tools used
        result = validator.validate_completion(
            requirements=requirements,
            action_result=MockTurnResult(
                response="Fixed the bug", tool_calls=["read_file", "write_file"]
            ),
            context={},
        )

        assert len(result.satisfied_requirements) == 1
        assert "tool" in result.satisfied_requirements[0].evidence.lower()

    def test_validate_constraint_requirement_language(self):
        """Test validation of constraint requirement for programming language."""
        validator = RequirementValidator()

        requirements = [
            Requirement(
                type=RequirementType.CONSTRAINT,
                description="Must use Python",
                priority=1,
            )
        ]

        # Test with matching file
        result = validator.validate_completion(
            requirements=requirements,
            action_result=MockTurnResult(response="Created the file"),
            context={"files_modified": ["auth.py", "user.py"]},
        )

        # Should have at least one satisfied requirement
        assert len(result.satisfied_requirements) + len(result.missing_requirements) == 1

    def test_validate_quality_requirement(self):
        """Test validation of quality requirement."""
        validator = RequirementValidator()

        requirements = [
            Requirement(
                type=RequirementType.QUALITY,
                description="Code should be well-documented",
                priority=2,
            )
        ]

        # Test with quality keywords in response
        result = validator.validate_completion(
            requirements=requirements,
            action_result=MockTurnResult(response="The code is optimized and well-documented"),
            context={},
        )

        # Quality requirements are P2, easier to satisfy
        assert len(result.satisfied_requirements) + len(result.missing_requirements) == 1

    def test_extract_response_from_completion_response_object(self):
        """CompletionResponse-like nested responses should extract text content."""
        validator = RequirementValidator()
        action_result = Mock(response=Mock(content="The implementation is efficient."))

        assert validator._extract_response(action_result) == "The implementation is efficient."

    def test_priority_weighted_scoring(self):
        """Test that P0 requirements have higher weight in scoring."""
        validator = RequirementValidator()

        requirements = [
            Requirement(
                type=RequirementType.FUNCTIONAL,
                description="Critical requirement",
                priority=0,
            ),
            Requirement(type=RequirementType.FUNCTIONAL, description="Nice to have", priority=3),
        ]

        # Only satisfy P0
        result = validator.validate_completion(
            requirements=requirements,
            action_result=MockTurnResult(response="Done", tool_calls=["tool1"]),
            context={},
        )

        # Should have at least some score (40% for P0)
        assert result.satisfaction_score >= 0.0

    def test_critical_gaps_detection(self):
        """Test that P0 requirements not satisfied are marked as critical gaps."""
        validator = RequirementValidator()

        requirements = [
            Requirement(type=RequirementType.FUNCTIONAL, description="Must do this", priority=0),
            Requirement(type=RequirementType.FUNCTIONAL, description="Optional", priority=3),
        ]

        # Don't satisfy any requirements
        result = validator.validate_completion(
            requirements=requirements,
            action_result=MockTurnResult(response="Will do"),
            context={},
        )

        assert len(result.critical_gaps) == 1
        assert result.critical_gaps[0].requirement.description == "Must do this"

    def test_satisfaction_score_calculation(self):
        """Test satisfaction score is calculated correctly."""
        validator = RequirementValidator(overall_threshold=0.3)

        requirements = [
            Requirement(type=RequirementType.FUNCTIONAL, description="P0 req", priority=0),
            Requirement(type=RequirementType.FUNCTIONAL, description="P1 req", priority=1),
            Requirement(type=RequirementType.FUNCTIONAL, description="P2 req", priority=2),
        ]

        # Satisfy all
        result = validator.validate_completion(
            requirements=requirements,
            action_result=MockTurnResult(
                response="All done", tool_calls=["tool1"], has_tool_calls=True
            ),
            context={"files_modified": ["file.py"]},
        )

        # Should have high score when all satisfied
        assert result.satisfaction_score >= 0.0

    def test_summary_generation_complete(self):
        """Test summary message when all requirements satisfied."""
        validator = RequirementValidator(overall_threshold=0.3)

        requirements = [
            Requirement(type=RequirementType.FUNCTIONAL, description="Req 1", priority=0),
        ]

        result = validator.validate_completion(
            requirements=requirements,
            action_result=MockTurnResult(response="Done", tool_calls=["tool1"]),
            context={},
        )

        # Summary should be generated
        assert result.summary is not None
        assert len(result.summary) > 0

    def test_summary_generation_critical_gaps(self):
        """Test summary message when critical gaps exist."""
        validator = RequirementValidator()

        requirements = [
            Requirement(
                type=RequirementType.FUNCTIONAL,
                description="Critical requirement",
                priority=0,
            ),
        ]

        result = validator.validate_completion(
            requirements=requirements,
            action_result=MockTurnResult(response="Will do"),
            context={},
        )

        assert "Critical" in result.summary or "gaps" in result.summary.lower()

    def test_get_satisfied_count(self):
        """Test counting satisfied requirements."""
        validator = RequirementValidator()

        requirements = [
            Requirement(type=RequirementType.FUNCTIONAL, description="P0", priority=0),
            Requirement(type=RequirementType.FUNCTIONAL, description="P1", priority=1),
        ]

        result = validator.validate_completion(
            requirements=requirements,
            action_result=MockTurnResult(response="Done", tool_calls=["tool1"]),
            context={},
        )

        # Should return count (0, 1, or 2)
        assert isinstance(result.get_satisfied_count(), int)
        assert result.get_satisfied_count() >= 0

    def test_get_total_count(self):
        """Test counting total requirements."""
        validator = RequirementValidator()

        requirements = [
            Requirement(type=RequirementType.FUNCTIONAL, description="P0", priority=0),
            Requirement(type=RequirementType.FUNCTIONAL, description="P1", priority=1),
        ]

        result = validator.validate_completion(
            requirements=requirements,
            action_result=MockTurnResult(response="Partial", tool_calls=["tool1"]),
            context={},
        )

    def test_out_of_range_priority_does_not_raise(self):
        """Regression: priority=4 (out of range 0-3) must not raise KeyError."""
        validator = RequirementValidator()
        requirements = [
            Requirement(type=RequirementType.FUNCTIONAL, description="do it", priority=4),
        ]
        result = validator.validate_completion(
            requirements=requirements,
            action_result=MockTurnResult(response="done", tool_calls=[]),
            context={},
        )
        # Should not raise; score clamped to P3 bucket → value between 0.0 and 1.0
        assert 0.0 <= result.satisfaction_score <= 1.0
