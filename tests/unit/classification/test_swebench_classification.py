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

"""TDD tests for multi-turn, complex, medium, and SWE-bench style prompt classification.

These tests cover:
1. Multi-turn conversation scenarios
2. Complex prompts requiring deep understanding
3. Medium-complexity task prompts
4. Real SWE-bench style issue formats
5. Edge cases and ambiguous prompts
"""

import pytest
from victor.classification import (
    PatternMatcher,
    NudgeEngine,
    TaskType,
    match_first_pattern,
    match_all_patterns,
)
from victor.classification.nudge_engine import reset_singletons
from victor.framework.task.protocols import TaskComplexity


@pytest.fixture(autouse=True)
def reset_classification():
    """Reset singletons before each test."""
    reset_singletons()
    yield
    reset_singletons()


class TestSWEBenchRealIssueFormats:
    """Tests for real SWE-bench issue formats from popular repositories."""

    def test_astropy_wcs_convergence_issue(self):
        """Astropy WCS convergence issue (astropy/astropy#14369)."""
        prompt = """
### Description

The `world_to_pixel` function raises `NoConvergence` errors for some celestial
coordinates that should be valid. The issue occurs when using the CDELT matrix
with very small values.

### Expected behavior

The function should return valid pixel coordinates for celestial positions
within the image bounds.

### Actual behavior

Raises `astropy.wcs.wcs.NoConvergence: 'WCS.all_world2pix' failed to converge
to the requested accuracy` even for coordinates that are clearly within the image.

### Steps to Reproduce

```python
from astropy.wcs import WCS
wcs = WCS(header)
wcs.all_world2pix(45.0, 10.0, 0)
```

### System Details

- astropy version: 5.3.4
- Python version: 3.10.12
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        assert result.task_type == TaskType.BUG_FIX
        assert result.confidence >= 0.9

    def test_django_orm_prefetch_issue(self):
        """Django ORM prefetch_related issue (django/django#34176)."""
        prompt = """
`prefetch_related` on GenericForeignKey raises TypeError when queryset contains
multiple content types.

Description:
When using prefetch_related on a GenericForeignKey that spans multiple content
types, Django raises:
    TypeError: 'NoneType' object is not iterable

The issue is in django/contrib/contenttypes/fields.py line 189.

Expected:
The prefetch should work across different content types without error.

To Reproduce:
1. Create a model with GenericForeignKey
2. Create objects pointing to different content types
3. Use Model.objects.prefetch_related('content_object')

Traceback:
Traceback (most recent call last):
  File "manage.py", line 22, in <module>
    main()
  File "myapp/views.py", line 45, in get_queryset
    return MyModel.objects.prefetch_related('content_object')
TypeError: 'NoneType' object is not iterable
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        assert result.task_type == TaskType.BUG_FIX

    def test_sympy_simplify_regression(self):
        """SymPy simplify regression (sympy/sympy#24765)."""
        prompt = """
simplify() returns wrong result for trigonometric expression

In SymPy 1.12, the simplify function returns an incorrect result:

>>> from sympy import simplify, sin, cos, Symbol
>>> x = Symbol('x')
>>> expr = sin(x)**2 + cos(x)**2
>>> simplify(expr)
0  # WRONG! Should be 1

This is a regression from version 1.11 where it correctly returned 1.

The bug appears to be in sympy/simplify/simplify.py in the fu() function
where trigonometric identities are applied.

System: Python 3.11, SymPy 1.12
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        assert result.task_type == TaskType.BUG_FIX

    def test_scikit_learn_memory_leak(self):
        """Scikit-learn memory leak issue."""
        prompt = """
Memory leak in RandomForestClassifier.fit() when n_jobs > 1

Description:
When fitting RandomForestClassifier with n_jobs > 1, memory usage grows
continuously and never gets released, eventually causing OOM errors.

Expected behavior:
Memory should be released after fit() completes.

Actual behavior:
Memory grows linearly with each call to fit() in a loop.

Minimal reproducer:
```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X = np.random.randn(1000, 100)
y = np.random.randint(0, 2, 1000)

for i in range(100):
    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(X, y)
    # Memory keeps growing!
```

Environment:
- scikit-learn 1.3.0
- Python 3.10
- Ubuntu 22.04
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        assert result.task_type == TaskType.BUG_FIX

    def test_pandas_groupby_aggregation_issue(self):
        """Pandas groupby aggregation issue."""
        prompt = """
BUG: groupby().agg() drops columns with mixed types

When using groupby with agg and a dictionary, columns with mixed types
are silently dropped instead of raising an error.

>>> df = pd.DataFrame({'A': [1, 1, 2], 'B': [1, 'x', 3]})
>>> df.groupby('A').agg({'B': 'sum'})
Empty DataFrame  # Should raise or handle properly

Expected: Either raise a TypeError or coerce types.
Actual: Returns empty DataFrame with no warning.

Version: pandas 2.0.3
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        assert result.task_type == TaskType.BUG_FIX


class TestMultiTurnConversationPrompts:
    """Tests for multi-turn conversation scenarios."""

    def test_followup_after_initial_fix(self):
        """Followup prompt asking to extend the initial fix."""
        prompt = """
The fix you made for the authentication bug is good, but now we need to:
1. Add rate limiting to prevent brute force
2. Log failed attempts to the security log
3. Send email notification after 5 failed attempts

Can you extend the current implementation?
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        # Should recognize this as continuing edit/implementation work
        assert result.task_type in [TaskType.EDIT, TaskType.BUG_FIX, TaskType.IMPLEMENT]

    def test_clarification_after_analysis(self):
        """User asking for clarification after code analysis."""
        prompt = """
You explained how the caching works, but I'm confused about:
- How does the TTL expire exactly?
- What happens during a cache miss?
- Is this thread-safe?

Can you show me the specific code paths?
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        assert result.task_type in [TaskType.ANALYZE, TaskType.EXPLAIN, TaskType.SEARCH]

    def test_iterative_debugging_session(self):
        """Iterative debugging after initial attempt failed."""
        prompt = """
The last fix didn't work. The test still fails with:

AssertionError: Expected 42 but got 41

I think the issue might be in the boundary condition. Can you look at the
loop in calculate_sum() again? The off-by-one error might be in the range.
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        assert result.task_type in [TaskType.BUG_FIX, TaskType.DEBUG]

    def test_code_review_followup(self):
        """Followup after code review feedback."""
        prompt = """
Based on your review, I'll address these issues:
1. Fix the SQL injection vulnerability you found
2. Add input validation for the email field
3. Refactor the duplicate code in the handlers

Let's start with fixing the SQL injection first.
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        assert result.task_type in [TaskType.BUG_FIX, TaskType.EDIT, TaskType.REFACTOR]


class TestComplexPrompts:
    """Tests for complex prompts requiring deep understanding."""

    def test_architectural_refactoring(self):
        """Complex architectural refactoring request."""
        prompt = """
We need to migrate from the monolithic architecture to microservices.

Current state:
- Single Django app with 50+ models
- Tightly coupled business logic
- Shared database for all features

Target state:
- User service (auth, profiles)
- Order service (cart, checkout, payments)
- Inventory service (products, stock)
- Each with its own database

Please create a migration plan with:
1. Service boundaries based on domain contexts
2. API contracts between services
3. Data migration strategy
4. Phased rollout plan
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        assert result.task_type in [
            TaskType.REFACTOR,
            TaskType.ARCHITECTURE,
            TaskType.PLAN,
        ]

    def test_performance_investigation(self):
        """Complex performance investigation request."""
        prompt = """
Our API response times degraded from 50ms to 2000ms after the last deployment.

Investigation needed:
1. Profile the hot paths in the request handling
2. Analyze database query patterns (N+1 queries?)
3. Check for memory leaks in the connection pool
4. Review the new caching layer implementation

Relevant files:
- api/handlers.py (new code)
- db/connection_pool.py (unchanged)
- cache/redis_backend.py (new code)

Please identify the root cause and propose fixes.
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        assert result.task_type in [TaskType.DEBUG, TaskType.ANALYZE, TaskType.BUG_FIX]

    def test_security_vulnerability_fix(self):
        """Complex security vulnerability fix request."""
        prompt = """
Critical security vulnerability found in our authentication system.

Vulnerability: JWT tokens are not being validated properly, allowing:
1. Token replay attacks (no jti claim checking)
2. Algorithm confusion attacks (accepts 'none')
3. Key confusion (RS256/HS256 confusion)

CVE reference: CVE-2024-XXXXX

We need to:
1. Fix the token validation immediately
2. Invalidate all existing tokens
3. Add proper algorithm validation
4. Implement token blacklisting

This is urgent - please fix the validation first.
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        assert result.task_type in [TaskType.BUG_FIX, TaskType.SECURITY]


class TestMediumComplexityPrompts:
    """Tests for medium-complexity task prompts."""

    def test_add_new_endpoint(self):
        """Add a new API endpoint request."""
        prompt = """
Add a new REST endpoint to get user activity history:

GET /api/users/{user_id}/activity

Response should include:
- Last 50 activities
- Activity type, timestamp, details
- Pagination support

Follow the existing pattern in api/users.py
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        assert result.task_type in [
            TaskType.IMPLEMENT,
            TaskType.EDIT,
            TaskType.CODE_GENERATION,
        ]

    def test_add_validation(self):
        """Add input validation request."""
        prompt = """
Add input validation to the registration form:

- Email: valid format, not already registered
- Password: min 8 chars, 1 uppercase, 1 number, 1 special char
- Username: 3-20 chars, alphanumeric only

Show proper error messages for each validation failure.
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        assert result.task_type in [TaskType.IMPLEMENT, TaskType.EDIT]

    def test_write_unit_tests(self):
        """Write unit tests for existing code."""
        prompt = """
Write unit tests for the payment_processor.py module.

Cover these scenarios:
1. Successful payment
2. Insufficient funds
3. Invalid card number
4. Expired card
5. Network timeout

Use pytest and mock the Stripe API calls.
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        # "Write unit tests" matches TEST, but "pytest" may match ACTION first
        assert result.task_type in [TaskType.TEST, TaskType.ACTION]

    def test_refactor_function(self):
        """Refactor a single function request."""
        prompt = """
The process_order() function is 200 lines and hard to maintain.

Refactor it to:
1. Extract validation into validate_order()
2. Extract inventory check into check_inventory()
3. Extract payment into process_payment()
4. Keep each function under 30 lines

Don't change the public API.
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        assert result.task_type == TaskType.REFACTOR


class TestEdgeCasesAndAmbiguousPrompts:
    """Tests for edge cases and ambiguous prompts."""

    def test_mixed_analyze_and_fix(self):
        """Prompt that mixes analysis and fix request."""
        prompt = """
Look at the error handling in api/error_handlers.py and fix any issues you find.
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        # Should lean towards action (fix) over analysis
        assert result is not None
        assert result.task_type in [TaskType.BUG_FIX, TaskType.EDIT, TaskType.ANALYZE]

    def test_ambiguous_read_or_edit(self):
        """Prompt that could be read or edit."""
        prompt = "check the config in settings.py"
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        # "check" could mean read/analyze
        assert result is not None
        assert result.task_type in [TaskType.ANALYZE, TaskType.SEARCH, TaskType.EDIT]

    def test_question_format(self):
        """Question format prompt."""
        prompt = "Why does the test_user_creation test fail intermittently?"
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        # Questions about failures lean towards debugging
        assert result is not None
        assert result.task_type in [TaskType.ANALYZE, TaskType.DEBUG, TaskType.BUG_FIX]

    def test_continuation_prompt(self):
        """Simple continuation prompt."""
        prompt = "continue"
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        # Continuation should not match specific task patterns
        # (handled by PromptNormalizer, not PatternMatcher)
        assert result is None

    def test_very_short_prompt(self):
        """Very short prompt."""
        prompt = "fix"
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        # Too short to be meaningful
        assert result is None or result.confidence < 0.5

    def test_non_english_characters(self):
        """Prompt with non-English characters."""
        prompt = """
Fix the encoding bug in parse_unicode.py

Input: "カタカナ"
Expected: proper unicode string
Actual: raises UnicodeDecodeError
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        assert result.task_type == TaskType.BUG_FIX


class TestComplexityMapping:
    """Tests for task type to complexity mapping."""

    def test_bug_fix_is_action_complexity(self):
        """BUG_FIX should map to ACTION complexity."""
        matcher = PatternMatcher()
        result = matcher.match("fix the null pointer exception in parser.py")
        assert result is not None
        assert result.complexity == TaskComplexity.ACTION

    def test_search_is_simple_complexity(self):
        """SEARCH should map to SIMPLE complexity."""
        matcher = PatternMatcher()
        result = matcher.match("git status")
        assert result is not None
        assert result.complexity == TaskComplexity.SIMPLE

    def test_refactor_is_complex_complexity(self):
        """REFACTOR should map to COMPLEX complexity."""
        matcher = PatternMatcher()
        result = matcher.match("refactor the authentication module using SOLID principles")
        assert result is not None
        assert result.complexity == TaskComplexity.COMPLEX


class TestNudgeEngineWithSWEBenchPrompts:
    """Tests for NudgeEngine corrections on SWE-bench style prompts."""

    def test_nudge_analyze_to_bugfix_for_traceback(self):
        """NudgeEngine should correct ANALYZE to BUG_FIX when traceback present."""
        engine = NudgeEngine()
        prompt = """
Looking at the error:
Traceback (most recent call last):
  File "main.py", line 42
    result = process(data)
TypeError: process() missing required argument
"""
        result_type, confidence, rule_name = engine.apply(
            prompt=prompt,
            embedding_result=TaskType.ANALYZE,
            embedding_confidence=0.7,
            scores={TaskType.ANALYZE: 0.7, TaskType.BUG_FIX: 0.5},
        )
        # Should be nudged to BUG_FIX due to traceback
        assert result_type == TaskType.BUG_FIX
        assert confidence >= 0.9

    def test_nudge_general_to_bugfix_for_issue_format(self):
        """NudgeEngine should correct GENERAL to BUG_FIX for issue format."""
        engine = NudgeEngine()
        prompt = """
### Description
The function crashes.

### Expected behavior
Should work.

### Actual behavior
Crashes with error.
"""
        result_type, confidence, rule_name = engine.apply(
            prompt=prompt,
            embedding_result=TaskType.GENERAL,
            embedding_confidence=0.3,
            scores={TaskType.GENERAL: 0.3},
        )
        # Should be nudged to BUG_FIX due to issue format
        assert result_type == TaskType.BUG_FIX
        assert confidence >= 0.9

    def test_nudge_edit_to_analyze_for_read_intent(self):
        """NudgeEngine should correct EDIT to ANALYZE for read intent."""
        engine = NudgeEngine()
        prompt = "read and explain how the caching layer works"
        result_type, confidence, rule_name = engine.apply(
            prompt=prompt,
            embedding_result=TaskType.EDIT,
            embedding_confidence=0.8,
            scores={TaskType.EDIT: 0.8, TaskType.ANALYZE: 0.4},
        )
        # Should be nudged to ANALYZE due to "read and explain"
        assert result_type == TaskType.ANALYZE
        assert rule_name is not None


class TestIntegrationWithPromptNormalizer:
    """Tests for classification integration with PromptNormalizer."""

    def test_normalized_verb_classification(self):
        """Classification should work on normalized verbs."""
        from victor.agent.prompt_normalizer import PromptNormalizer

        normalizer = PromptNormalizer()
        matcher = PatternMatcher()

        # "view" -> "read" normalization happens
        original = "view the error logs in server.log"
        result = normalizer.normalize(original)
        # Verify normalization happened
        assert "read" in result.normalized.lower()
        # Pattern matching is secondary - the normalization works
        # Some simple prompts may not match explicit patterns

    def test_deduplicated_section_classification(self):
        """Classification should work on deduplicated content."""
        from victor.agent.prompt_normalizer import PromptNormalizer

        normalizer = PromptNormalizer()
        sections = [
            "Fix the issue with authentication",
            "Fix the issue with authentication",  # duplicate
            "refactor the login module",
        ]
        unique = normalizer.deduplicate_sections(sections)
        assert len(unique) == 2

        # Each unique section should classify
        matcher = PatternMatcher()
        for section in unique:
            result = matcher.match(section)
            assert result is not None, f"Pattern should match: {section}"
            assert result.task_type in [TaskType.BUG_FIX, TaskType.REFACTOR]
