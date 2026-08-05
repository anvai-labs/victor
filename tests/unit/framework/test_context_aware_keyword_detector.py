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

"""Unit tests for ContextAwareKeywordDetector."""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field

from victor.framework.context_aware_keyword_detector import (
    ContextAwareKeywordDetector,
    CompletionSignal,
    CompletionIndicatorType,
)
from victor.framework.completion_scorer import TaskType

# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockRequirement:
    """Mock requirement for testing."""

    type: str
    description: str
    priority: int = 3


# =============================================================================
# ContextAwareKeywordDetector Tests
# =============================================================================


class TestContextAwareKeywordDetector:
    """Test suite for ContextAwareKeywordDetector."""

    def test_initialization(self):
        """Test detector initializes with task-specific patterns."""
        detector = ContextAwareKeywordDetector()

        # Should have patterns for all task types
        assert "code_generation" in detector.task_completion_patterns
        assert "testing" in detector.task_completion_patterns
        assert "debugging" in detector.task_completion_patterns

    def test_detect_continuation_request(self):
        """Test detection of continuation requests."""
        detector = ContextAwareKeywordDetector()

        response = "Would you like me to continue with the implementation?"

        signal = detector.detect_completion(
            response=response, task_type=TaskType.CODE_GENERATION, requirements=None
        )

        assert signal.is_continuation_request is True
        assert signal.has_completion_indicator is False
        assert signal.confidence == 0.3

    def test_detect_code_generation_completion(self):
        """Test detection of code generation completion."""
        detector = ContextAwareKeywordDetector()

        response = "Here is the implementation of the authentication system in the code above."

        signal = detector.detect_completion(
            response=response, task_type=TaskType.CODE_GENERATION, requirements=None
        )

        assert signal.has_completion_indicator is True
        assert signal.is_continuation_request is False
        assert signal.confidence > 0.8
        assert len(signal.evidence) > 0

    def test_detect_testing_completion(self):
        """Test detection of testing task completion."""
        detector = ContextAwareKeywordDetector()

        response = "All tests pass and the test suite is complete with 95% coverage."

        signal = detector.detect_completion(
            response=response, task_type=TaskType.TESTING, requirements=None
        )

        assert signal.has_completion_indicator is True
        assert any("tests pass" in evidence.lower() for evidence in signal.evidence)

    def test_detect_debugging_completion(self):
        """Test detection of debugging task completion."""
        detector = ContextAwareKeywordDetector()

        response = "The fix is applied and the issue is resolved. Root cause was a null pointer."

        signal = detector.detect_completion(
            response=response, task_type=TaskType.DEBUGGING, requirements=None
        )

        assert signal.has_completion_indicator is True
        assert "fix" in signal.evidence[0].lower() or "resolved" in signal.evidence[0].lower()

    def test_detect_search_completion(self):
        """Test detection of search task completion."""
        detector = ContextAwareKeywordDetector()

        response = "Found the authentication module in the src/auth directory and located the login function."

        signal = detector.detect_completion(
            response=response, task_type=TaskType.SEARCH, requirements=None
        )

        assert signal.has_completion_indicator is True
        assert signal.confidence > 0.7

    def test_detect_analysis_completion(self):
        """Test detection of analysis task completion."""
        detector = ContextAwareKeywordDetector()

        response = "In conclusion, the key findings show that the system performs well under load. To summarize, the architecture is scalable."

        signal = detector.detect_completion(
            response=response, task_type=TaskType.ANALYSIS, requirements=None
        )

        assert signal.has_completion_indicator is True
        assert any(
            "summary" in evidence.lower() or "conclusion" in evidence.lower()
            for evidence in signal.evidence
        )

    def test_detect_complete_code_blocks(self):
        """Test detection of complete code blocks."""
        detector = ContextAwareKeywordDetector()

        response = """
Here's the implementation:

```python
def authenticate(user, password):
    return verify(user, password)
```

The function above handles authentication.
"""

        signal = detector.detect_completion(
            response=response, task_type=TaskType.CODE_GENERATION, requirements=None
        )

        assert signal.has_complete_code is True
        assert CompletionIndicatorType.CODE in signal.indicator_types

    def test_detect_incomplete_code_blocks(self):
        """Test that incomplete code blocks are not detected as complete."""
        detector = ContextAwareKeywordDetector()

        response = "Here's the code:\n```python\ndef foo():"  # Missing closing

        signal = detector.detect_completion(
            response=response, task_type=TaskType.CODE_GENERATION, requirements=None
        )

        assert signal.has_complete_code is False

    def test_detect_structured_response(self):
        """Test detection of structured response."""
        detector = ContextAwareKeywordDetector()

        response = """
## Analysis Results

### Performance
- Response time: 100ms
- Throughput: 1000 req/s

### Errors
1. Timeout error
2. Memory leak
"""

        signal = detector.detect_completion(
            response=response, task_type=TaskType.ANALYSIS, requirements=None
        )

        assert signal.has_structure is True
        assert CompletionIndicatorType.STRUCTURE in signal.indicator_types

    def test_detect_with_requirements(self):
        """Test detection with requirements that are addressed."""
        detector = ContextAwareKeywordDetector()

        requirements = [
            MockRequirement(type="functional", description="Implement authentication system")
        ]

        response = "I've implemented the authentication system with secure password hashing."

        signal = detector.detect_completion(
            response=response,
            task_type=TaskType.CODE_GENERATION,
            requirements=requirements,
        )

        # Should detect that requirements are addressed
        assert len(signal.evidence) > 0
        assert any("requirement" in evidence.lower() for evidence in signal.evidence)

    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        detector = ContextAwareKeywordDetector()

        # Strong completion signals
        response = "Here is the implementation.\n```code```\n## Summary"

        signal = detector.detect_completion(
            response=response, task_type=TaskType.CODE_GENERATION, requirements=None
        )

        # Multiple signals should give high confidence
        assert signal.confidence > 0.8

    def test_empty_response(self):
        """Test detection with empty response."""
        detector = ContextAwareKeywordDetector()

        signal = detector.detect_completion(
            response="", task_type=TaskType.CODE_GENERATION, requirements=None
        )

        assert signal.has_completion_indicator is False
        assert signal.confidence == 0.0
        assert "No response" in signal.evidence[0]

    def test_unknown_task_type(self):
        """Test detection with unknown task type uses generic patterns."""
        detector = ContextAwareKeywordDetector()

        response = "Here is the solution to your problem."

        signal = detector.detect_completion(
            response=response, task_type=TaskType.UNKNOWN, requirements=None
        )

        # Should still detect generic completion patterns
        assert signal.has_completion_indicator is True

    def test_deliverable_references(self):
        """Test detection of deliverable references."""
        detector = ContextAwareKeywordDetector()

        response = "The code above implements the authentication logic as shown."

        signal = detector.detect_completion(
            response=response, task_type=TaskType.CODE_GENERATION, requirements=None
        )

        assert signal.has_completion_indicator is True
        assert any(
            "deliverable" in str(indicator).lower() or "explicit" in str(indicator).lower()
            for indicator in signal.indicator_types
        )

    def test_multiple_indicators(self):
        """Test detection with multiple completion indicators."""
        detector = ContextAwareKeywordDetector()

        response = """
## Summary

Here is the implementation of the fix:

```python
def fix():
    pass
```

The code above resolves the issue. In conclusion, the system is now working.
"""

        signal = detector.detect_completion(
            response=response, task_type=TaskType.DEBUGGING, requirements=None
        )

        # Should detect multiple indicator types
        assert len(signal.indicator_types) >= 3
        assert signal.confidence > 0.8

    def test_length_bonus_in_confidence(self):
        """Test that longer responses get confidence bonus."""
        detector = ContextAwareKeywordDetector()

        # Short response
        short_response = "Here is the code."
        short_signal = detector.detect_completion(
            response=short_response,
            task_type=TaskType.CODE_GENERATION,
            requirements=None,
        )

        # Long response
        long_response = "Here is the code. " * 100  # > 1000 chars
        long_signal = detector.detect_completion(
            response=long_response,
            task_type=TaskType.CODE_GENERATION,
            requirements=None,
        )

        # Long response should have higher or equal confidence
        assert long_signal.confidence >= short_signal.confidence

    def test_requirements_addressed_check(self):
        """Test that requirements are checked for being addressed."""
        detector = ContextAwareKeywordDetector()

        requirements = [
            MockRequirement(type="functional", description="Add authentication"),
            MockRequirement(type="functional", description="Create user model"),
        ]

        # Response that addresses both
        response = "I've added authentication and created the user model with fields."

        signal = detector.detect_completion(
            response=response,
            task_type=TaskType.CODE_GENERATION,
            requirements=requirements,
        )

        # Should detect that requirements were addressed
        assert len([e for e in signal.evidence if "requirement" in e.lower()]) > 0

    def test_requirements_not_addressed(self):
        """Test when response doesn't address requirements."""
        detector = ContextAwareKeywordDetector()

        requirements = [MockRequirement(type="functional", description="Add authentication system")]

        # Response that doesn't mention the requirement
        response = "I've created a simple file for testing."

        signal = detector.detect_completion(
            response=response,
            task_type=TaskType.CODE_GENERATION,
            requirements=requirements,
        )

        # Should not indicate requirements addressed
        assert not any("requirement" in e.lower() for e in signal.evidence)

    def test_task_type_specific_patterns_code_generation(self):
        """Test code generation specific patterns."""
        detector = ContextAwareKeywordDetector()

        patterns = detector._get_task_patterns(TaskType.CODE_GENERATION)

        assert "here is the implementation" in patterns
        assert "here's the code" in patterns
        assert "implemented the" in patterns

    def test_task_type_specific_patterns_debugging(self):
        """Test debugging specific patterns."""
        detector = ContextAwareKeywordDetector()

        patterns = detector._get_task_patterns(TaskType.DEBUGGING)

        assert "the fix is" in patterns
        assert "issue resolved" in patterns
        assert "bug fixed" in patterns

    def test_task_type_specific_patterns_testing(self):
        """Test testing specific patterns."""
        detector = ContextAwareKeywordDetector()

        patterns = detector._get_task_patterns(TaskType.TESTING)

        assert "tests pass" in patterns
        assert "all tests passing" in patterns
        assert "verified" in patterns
