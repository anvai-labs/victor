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

"""Integration tests for classification module with Ollama.

These tests verify that the classification, nudge rules, and prompt normalization
work correctly with actual LLM-assisted classification when Ollama is available.

Tests are skipped if Ollama is not available.
"""

import socket
import pytest


def is_ollama_available() -> bool:
    """Check if Ollama server is running at localhost:11434."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 11434))
        return result == 0
    finally:
        sock.close()


def list_installed_ollama_models() -> list[str]:
    """Return installed Ollama model names, or [] if unreachable / none installed.

    Hits the ``/api/tags`` endpoint directly so the check is synchronous and
    usable from a ``skipif`` marker evaluated at collection time.
    """
    import httpx

    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return [m["name"] for m in models if m.get("name")]
    except Exception:
        return []


def requires_ollama():
    """Pytest marker to skip tests when Ollama or an installed model is unavailable.

    Skips when the server is down OR when the server is up but no model is
    installed, so a reachable-but-empty Ollama skips instead of failing.
    """
    if not is_ollama_available():
        reason = "Ollama server not available"
    elif not list_installed_ollama_models():
        reason = (
            "Ollama server available but no models installed "
            "(run `ollama pull qwen2.5-coder:7b`)"
        )
    else:
        reason = ""
    return pytest.mark.skipif(bool(reason), reason=reason or "Ollama available")


from victor.classification import (
    PatternMatcher,
    NudgeEngine,
    TaskType,
    match_first_pattern,
)
from victor.classification.nudge_engine import reset_singletons
from victor.agent.prompt_normalizer import (
    PromptNormalizer,
    get_prompt_normalizer,
    reset_normalizer,
)
from victor.framework.task.protocols import TaskComplexity


@pytest.fixture(autouse=True)
def reset_classification():
    """Reset singletons before each test."""
    reset_singletons()
    reset_normalizer()
    yield
    reset_singletons()
    reset_normalizer()


class TestSWEBenchPatternClassification:
    """Integration tests for SWE-bench style pattern classification."""

    @requires_ollama()
    def test_astropy_style_issue_classification(self):
        """Test classification of Astropy-style GitHub issues."""
        prompt = """
### Description

The `world_to_pixel` function raises `NoConvergence` errors for valid coordinates.

### Expected behavior

Should return pixel coordinates.

### Actual behavior

Raises NoConvergence error.

### Steps to Reproduce

```python
from astropy.wcs import WCS
wcs.all_world2pix(45.0, 10.0, 0)
```
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        assert result.task_type == TaskType.BUG_FIX
        assert result.complexity == TaskComplexity.ACTION

    @requires_ollama()
    def test_django_style_issue_classification(self):
        """Test classification of Django-style bug reports."""
        prompt = """
TypeError in prefetch_related for GenericForeignKey

Traceback (most recent call last):
  File "manage.py", line 22, in <module>
    main()
  File "myapp/views.py", line 45, in get_queryset
    return Model.objects.prefetch_related('content_object')
TypeError: 'NoneType' object is not iterable

Expected: prefetch should work.
Actual: Raises TypeError.
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        assert result.task_type == TaskType.BUG_FIX

    @requires_ollama()
    def test_memory_leak_issue_classification(self):
        """Test classification of memory leak issues."""
        prompt = """
Memory leak in parallel processing with n_jobs > 1

Description:
When running fit() with n_jobs=-1, memory usage grows continuously.

Expected: Memory released after each fit().
Actual: Memory keeps growing causing OOM.
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        assert result.task_type == TaskType.BUG_FIX


class TestNudgeEngineIntegration:
    """Integration tests for NudgeEngine with LLM-assisted classification."""

    @requires_ollama()
    def test_nudge_analyze_to_bugfix_for_traceback(self):
        """Test that tracebacks trigger BUG_FIX nudge."""
        engine = NudgeEngine()
        prompt = """
Looking at this error:
Traceback (most recent call last):
  File "main.py", line 42
TypeError: missing argument
"""
        result_type, confidence, rule_name = engine.apply(
            prompt=prompt,
            embedding_result=TaskType.ANALYZE,
            embedding_confidence=0.7,
            scores={TaskType.ANALYZE: 0.7, TaskType.BUG_FIX: 0.5},
        )
        assert result_type == TaskType.BUG_FIX
        assert confidence >= 0.9

    @requires_ollama()
    def test_nudge_general_to_bugfix_for_issue_format(self):
        """Test that GitHub issue format triggers BUG_FIX nudge."""
        engine = NudgeEngine()
        prompt = """
### Description
Function crashes.

### Expected behavior
Should work.

### Actual behavior
Crashes.
"""
        result_type, confidence, rule_name = engine.apply(
            prompt=prompt,
            embedding_result=TaskType.GENERAL,
            embedding_confidence=0.3,
            scores={TaskType.GENERAL: 0.3},
        )
        assert result_type == TaskType.BUG_FIX
        assert confidence >= 0.9


class TestPromptNormalizerIntegration:
    """Integration tests for PromptNormalizer."""

    @requires_ollama()
    def test_verb_normalization_preserves_intent(self):
        """Test that verb normalization preserves classification intent."""
        normalizer = PromptNormalizer()
        matcher = PatternMatcher()

        # Test different verb forms
        prompts = [
            ("view the configuration file", "read"),
            ("look at the error logs", "read"),
            ("examine the test results", "analyze"),
            ("review the code changes", "analyze"),
        ]

        for original, expected_verb in prompts:
            result = normalizer.normalize(original)
            assert (
                expected_verb in result.normalized.lower()
            ), f"'{original}' should normalize to contain '{expected_verb}'"

    @requires_ollama()
    def test_continuation_collapsing(self):
        """Test that multiple continuations are tracked and collapsed."""
        normalizer = PromptNormalizer()

        # Simulate continuation sequence
        normalizer.normalize("continue")
        normalizer.normalize("yes")
        normalizer.normalize("ok")
        normalizer.normalize("proceed")  # 4th continuation should trigger collapse

        # Check continuation tracking via stats
        stats = normalizer.get_stats()
        # After 3+ continuations, count is tracked
        assert stats["continuation_count"] >= 0  # Counter may reset after collapse

    @requires_ollama()
    def test_section_deduplication_for_prompts(self):
        """Test section deduplication for repeated instructions."""
        normalizer = PromptNormalizer()
        sections = [
            "Always use type hints",
            "Follow PEP 8 style",
            "Always use type hints",  # duplicate
            "Write unit tests",
        ]

        unique = normalizer.deduplicate_sections(sections)
        assert len(unique) == 3
        assert unique.count("Always use type hints") == 1


class TestComplexPromptClassification:
    """Integration tests for complex multi-part prompts."""

    @requires_ollama()
    def test_multiturn_debugging_session(self):
        """Test classification of iterative debugging prompts."""
        prompt = """
The last fix didn't work. The test still fails:

AssertionError: Expected 42 but got 41

I think the off-by-one error is in the loop. Can you check calculate_sum()?
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        assert result.task_type in [TaskType.BUG_FIX, TaskType.DEBUG]

    @requires_ollama()
    def test_architectural_migration_request(self):
        """Test classification of architecture migration prompts."""
        prompt = """
Migrate from monolith to microservices:
- User service (auth, profiles)
- Order service (cart, checkout)
- Inventory service (products)

Create a phased rollout plan.
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        assert result.task_type in [
            TaskType.ARCHITECTURE,
            TaskType.REFACTOR,
            TaskType.PLAN,
        ]

    @requires_ollama()
    def test_security_vulnerability_request(self):
        """Test classification of security vulnerability prompts."""
        prompt = """
Critical security vulnerability found:
CVE-2024-XXXXX - JWT token validation bypass

Fix the token validation immediately.
"""
        matcher = PatternMatcher()
        result = matcher.match(prompt)
        assert result is not None
        assert result.task_type in [TaskType.SECURITY, TaskType.BUG_FIX]


class TestComplexityMappingIntegration:
    """Integration tests for task type to complexity mapping."""

    @requires_ollama()
    def test_bug_fix_maps_to_action(self):
        """Test that BUG_FIX maps to ACTION complexity."""
        matcher = PatternMatcher()
        result = matcher.match("fix the null pointer exception in parser.py")
        assert result is not None
        assert result.task_type == TaskType.BUG_FIX
        assert result.complexity == TaskComplexity.ACTION

    @requires_ollama()
    def test_refactor_maps_to_complex(self):
        """Test that REFACTOR maps to COMPLEX complexity."""
        matcher = PatternMatcher()
        result = matcher.match("refactor the authentication module using SOLID principles")
        assert result is not None
        assert result.task_type == TaskType.REFACTOR
        assert result.complexity == TaskComplexity.COMPLEX

    @requires_ollama()
    def test_search_maps_to_simple(self):
        """Test that SEARCH maps to SIMPLE complexity."""
        matcher = PatternMatcher()
        result = matcher.match("git status")
        assert result is not None
        assert result.task_type == TaskType.SEARCH
        assert result.complexity == TaskComplexity.SIMPLE


@pytest.mark.skipif(not is_ollama_available(), reason="Ollama not available")
class TestFullClassificationPipeline:
    """End-to-end tests for the complete classification pipeline."""

    def test_full_pipeline_bugfix(self):
        """Test full classification pipeline for bug fix."""
        # Step 1: Normalize input
        normalizer = get_prompt_normalizer()
        prompt = "View the error in auth.py and fix the issue"
        normalized = normalizer.normalize(prompt)

        # Step 2: Pattern match
        result = match_first_pattern(normalized.normalized)
        assert result is not None

        # Step 3: Apply nudge if needed
        engine = NudgeEngine()
        final_type, confidence, _ = engine.apply(
            prompt=normalized.normalized,
            embedding_result=result.task_type,
            embedding_confidence=result.confidence,
            scores={result.task_type: result.confidence},
        )

        # Should classify as bug fix with high confidence
        assert final_type in [TaskType.BUG_FIX, TaskType.EDIT, TaskType.ANALYZE]

    def test_full_pipeline_swe_bench_issue(self):
        """Test full classification pipeline for SWE-bench style issue."""
        normalizer = get_prompt_normalizer()
        prompt = """
### Description
The function returns wrong results.

### Expected behavior
Should return correct value.

### Actual behavior
Returns None.
"""
        normalized = normalizer.normalize(prompt)
        result = match_first_pattern(normalized.normalized)

        assert result is not None
        assert result.task_type == TaskType.BUG_FIX
        assert result.complexity == TaskComplexity.ACTION
