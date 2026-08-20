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

"""Unit tests for the classification module.

Tests the unified classification patterns, NudgeEngine, and PatternMatcher
functionality including SWE-bench style bug fix detection patterns.
"""

import pytest
from victor.classification import (
    PATTERNS,
    ClassificationPattern,
    TaskType,
    TASK_TYPE_TO_COMPLEXITY,
    get_patterns_by_complexity,
    get_patterns_by_task_type,
    match_first_pattern,
    match_all_patterns,
    NudgeEngine,
    NudgeRule,
    PatternMatcher,
    get_nudge_engine,
    get_pattern_matcher,
)
from victor.classification.nudge_engine import reset_singletons
from victor.framework.task.protocols import TaskComplexity


@pytest.fixture(autouse=True)
def reset_classification_singletons():
    """Reset singletons before each test."""
    reset_singletons()
    yield
    reset_singletons()


class TestPatternRegistry:
    """Tests for the unified pattern registry."""

    def test_patterns_exist(self):
        """PATTERNS dictionary should have patterns."""
        assert len(PATTERNS) > 0

    def test_all_patterns_have_required_fields(self):
        """All patterns should have required fields."""
        for name, pattern in PATTERNS.items():
            assert isinstance(pattern.name, str)
            assert isinstance(pattern.regex, str)
            assert isinstance(pattern.semantic_intent, str)
            assert isinstance(pattern.task_type, TaskType)
            assert isinstance(pattern.complexity, TaskComplexity)
            assert 0.0 <= pattern.confidence <= 1.0

    def test_task_type_to_complexity_mapping(self):
        """TASK_TYPE_TO_COMPLEXITY should map all TaskTypes."""
        # Should have mappings for most TaskTypes
        assert len(TASK_TYPE_TO_COMPLEXITY) >= 20
        # Check some specific mappings
        assert TASK_TYPE_TO_COMPLEXITY[TaskType.BUG_FIX] == TaskComplexity.ACTION
        assert TASK_TYPE_TO_COMPLEXITY[TaskType.SEARCH] == TaskComplexity.SIMPLE
        assert TASK_TYPE_TO_COMPLEXITY[TaskType.REFACTOR] == TaskComplexity.COMPLEX


class TestPatternMatcherCore:
    """Core tests for PatternMatcher."""

    def test_match_first_pattern_function(self):
        """match_first_pattern should find first matching pattern."""
        # SWE-bench style issue
        result = match_first_pattern("### Description\n\nThe bug occurs when...")
        assert result is not None
        assert result.task_type == TaskType.BUG_FIX

    def test_match_all_patterns_function(self):
        """match_all_patterns should find all matching patterns."""
        result = match_all_patterns("fix the bug and run tests")
        assert len(result) >= 1

    def test_get_patterns_by_complexity(self):
        """get_patterns_by_complexity should filter correctly."""
        action_patterns = get_patterns_by_complexity(TaskComplexity.ACTION)
        for pattern in action_patterns:
            assert pattern.complexity == TaskComplexity.ACTION

    def test_get_patterns_by_task_type(self):
        """get_patterns_by_task_type should filter correctly."""
        bug_fix_patterns = get_patterns_by_task_type(TaskType.BUG_FIX)
        for pattern in bug_fix_patterns:
            assert pattern.task_type == TaskType.BUG_FIX


class TestPatternMatcherClass:
    """Tests for PatternMatcher class."""

    def test_match_returns_pattern(self):
        """match should return a ClassificationPattern."""
        matcher = PatternMatcher()
        result = matcher.match("refactor the authentication module")
        assert result is not None
        assert isinstance(result, ClassificationPattern)

    def test_match_returns_none_for_no_match(self):
        """match should return None when no pattern matches."""
        matcher = PatternMatcher()
        result = matcher.match("hello world")
        assert result is None

    def test_classify_fast_returns_tuple(self):
        """classify_fast should return tuple or None."""
        matcher = PatternMatcher()
        result = matcher.classify_fast("git commit -m 'message'")
        assert result is not None
        task_type, confidence, pattern_name = result
        assert isinstance(task_type, TaskType)
        assert isinstance(confidence, float)
        assert isinstance(pattern_name, str)

    def test_match_all_returns_sorted_list(self):
        """match_all should return sorted by priority."""
        matcher = PatternMatcher()
        result = matcher.match_all("fix the bug in authentication")
        assert isinstance(result, list)
        # Should be sorted by priority (descending)
        priorities = [p.priority for p in result]
        assert priorities == sorted(priorities, reverse=True)


class TestNudgeEngine:
    """Tests for NudgeEngine."""

    def test_apply_override_rule(self):
        """Override rules should always apply."""
        engine = NudgeEngine()
        result_type, confidence, rule_name = engine.apply(
            prompt="read and explain auth.py",
            embedding_result=TaskType.EDIT,
            embedding_confidence=0.9,
            scores={TaskType.EDIT: 0.9, TaskType.ANALYZE: 0.3},
        )
        assert result_type == TaskType.ANALYZE
        assert confidence == 1.0
        assert rule_name is not None

    def test_apply_no_nudge_when_no_match(self):
        """No nudge should be applied when no rule matches."""
        engine = NudgeEngine()
        result_type, confidence, rule_name = engine.apply(
            prompt="hello world",
            embedding_result=TaskType.GENERAL,
            embedding_confidence=0.5,
            scores={},
        )
        assert result_type == TaskType.GENERAL
        assert confidence == 0.5
        assert rule_name is None

    def test_get_matching_rules(self):
        """get_matching_rules should return all matching rules."""
        engine = NudgeEngine()
        rules = engine.get_matching_rules("fix the bug in the code")
        assert isinstance(rules, list)
        for rule in rules:
            assert isinstance(rule, NudgeRule)


class TestSWEBenchPatterns:
    """Tests for SWE-bench style bug fix detection patterns."""

    def test_github_issue_format(self):
        """GitHub issue format should be detected as BUG_FIX."""
        matcher = PatternMatcher()
        result = matcher.match("""
### Description

The world_to_pixel function fails to converge.

### Expected behavior

Should return pixel coordinates.

### Actual behavior

Raises NoConvergenceError.
""")
        assert result is not None
        assert result.task_type == TaskType.BUG_FIX

    def test_steps_to_reproduce(self):
        """Steps to Reproduce should be detected as BUG_FIX."""
        matcher = PatternMatcher()
        result = matcher.match("""
Steps to Reproduce:
1. Call the function
2. Pass invalid parameter
3. Error occurs
""")
        assert result is not None
        assert result.task_type == TaskType.BUG_FIX

    def test_traceback_error_pattern(self):
        """Python traceback should be detected as BUG_FIX."""
        matcher = PatternMatcher()
        result = matcher.match("""
Traceback (most recent call last):
  File "test.py", line 10
    raise ValueError("test")
ValueError: test
""")
        assert result is not None
        assert result.task_type == TaskType.BUG_FIX

    def test_fix_the_issue_pattern(self):
        """'Fix the issue' should be detected as BUG_FIX."""
        matcher = PatternMatcher()
        result = matcher.match("Please fix the issue with the authentication")
        assert result is not None
        assert result.task_type == TaskType.BUG_FIX

    def test_expected_vs_actual_pattern(self):
        """'expected X but got Y' should be detected as BUG_FIX."""
        matcher = PatternMatcher()
        result = matcher.match("expected 42 but got None instead")
        assert result is not None
        assert result.task_type == TaskType.BUG_FIX


class TestDevOpsPatterns:
    """Tests for DevOps task type patterns."""

    def test_kubernetes_pattern(self):
        """Kubernetes patterns should be detected."""
        matcher = PatternMatcher()
        result = matcher.match("create a kubernetes deployment")
        assert result is not None
        assert result.task_type in [TaskType.KUBERNETES, TaskType.INFRASTRUCTURE]

    def test_docker_pattern(self):
        """Docker patterns should be detected."""
        matcher = PatternMatcher()
        result = matcher.match("create a Dockerfile for the app")
        assert result is not None
        assert result.task_type in [TaskType.DOCKERFILE, TaskType.INFRASTRUCTURE]

    def test_terraform_pattern(self):
        """Terraform patterns should be detected."""
        matcher = PatternMatcher()
        result = matcher.match("write a terraform module")
        assert result is not None
        assert result.task_type in [TaskType.TERRAFORM, TaskType.INFRASTRUCTURE]


class TestCodingPatterns:
    """Tests for coding task type patterns."""

    def test_refactor_pattern(self):
        """Refactor patterns should be detected."""
        matcher = PatternMatcher()
        result = matcher.match("refactor the authentication module")
        assert result is not None
        assert result.task_type == TaskType.REFACTOR

    def test_debug_pattern(self):
        """Debug patterns should be detected."""
        matcher = PatternMatcher()
        result = matcher.match("debug the issue in the parser")
        assert result is not None
        assert result.task_type == TaskType.DEBUG

    def test_test_pattern(self):
        """Test patterns should be detected."""
        matcher = PatternMatcher()
        result = matcher.match("write unit tests for the validator")
        assert result is not None
        assert result.task_type == TaskType.TEST


class TestResearchPatterns:
    """Tests for research task type patterns."""

    def test_fact_check_pattern(self):
        """Fact check patterns should be detected."""
        matcher = PatternMatcher()
        result = matcher.match("fact-check the claim about performance")
        assert result is not None
        assert result.task_type == TaskType.FACT_CHECK

    def test_literature_review_pattern(self):
        """Literature review patterns should be detected."""
        matcher = PatternMatcher()
        result = matcher.match("conduct a literature review on AI safety")
        assert result is not None
        assert result.task_type == TaskType.LITERATURE_REVIEW


class TestGitActionPatterns:
    """Tests for Git action patterns."""

    def test_git_commit_pattern(self):
        """Git commit should be ACTION."""
        matcher = PatternMatcher()
        result = matcher.match("git commit -m 'fix bug'")
        assert result is not None
        assert result.task_type == TaskType.ACTION

    def test_git_status_pattern(self):
        """Git status should be SEARCH (read operation)."""
        matcher = PatternMatcher()
        result = matcher.match("git status")
        assert result is not None
        assert result.task_type == TaskType.SEARCH

    def test_run_tests_pattern(self):
        """Run tests should be ACTION."""
        matcher = PatternMatcher()
        result = matcher.match("run the tests")
        assert result is not None
        assert result.task_type == TaskType.ACTION


class TestSingletons:
    """Tests for singleton pattern."""

    def test_get_nudge_engine_returns_singleton(self):
        """get_nudge_engine should return the same instance."""
        e1 = get_nudge_engine()
        e2 = get_nudge_engine()
        assert e1 is e2

    def test_get_pattern_matcher_returns_singleton(self):
        """get_pattern_matcher should return the same instance."""
        m1 = get_pattern_matcher()
        m2 = get_pattern_matcher()
        assert m1 is m2
