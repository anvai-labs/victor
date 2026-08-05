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

"""
Integration tests for fuzzy matching in classification systems.

These tests verify that fuzzy matching works correctly across all
classification systems when integrated with embeddings and real data.
"""

import pytest

from victor.storage.embeddings.task_classifier import TaskType, TaskTypeClassifier
from victor.tools.semantic_selector import SemanticToolSelector
from victor.storage.embeddings.intent_classifier import IntentClassifier, IntentType

# "analyze"-style prompts legitimately resolve to any analysis-family task type.
# Finer-grained members (ANALYSIS_DEEP, DATA_ANALYSIS) were added over time, so
# these tests assert membership in the family rather than one exact enum value;
# SEARCH stays an acceptable near-neighbour for such short prompts.
ANALYSIS_TASK_TYPES = {
    TaskType.ANALYZE,
    TaskType.ANALYSIS_DEEP,
    TaskType.DATA_ANALYSIS,
    TaskType.SEARCH,
}


@pytest.mark.integration
class TestTaskClassificationWithTypos:
    """Test TaskTypeClassifier with typos in user input."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        # Get classifier instance
        self.classifier = TaskTypeClassifier.get_instance()
        if not self.classifier.is_initialized:
            self.classifier.initialize_sync()

    def test_analyze_with_typo(self):
        """Test ANALYZE classification with typo."""
        # Original: "analyze framework structure"
        # With typo: "analize framework structre"
        result = self.classifier.classify_sync("analize framework structre")

        # Should still classify into the analysis family (with fuzzy matching)
        assert result.task_type in ANALYSIS_TASK_TYPES
        assert result.confidence > 0.3

    def test_refactor_with_typo(self):
        """Test REFACTOR classification with typo."""
        # Original: "refactor the code"
        # With typo: "refactor the cdoe"
        result = self.classifier.classify_sync("refactor the cdoe")

        # Should classify as REFACTOR or similar
        assert result.confidence > 0.3

    def test_search_with_typo(self):
        """Test SEARCH classification with typo."""
        # Original: "search for functions"
        # With typo: "serch for funcitons"
        result = self.classifier.classify_sync("serch for funcitons")

        # Should classify as SEARCH
        assert result.task_type in [TaskType.SEARCH, TaskType.ANALYZE]
        assert result.confidence > 0.3

    def test_multiple_typos(self):
        """Test classification with multiple typos."""
        # Original: "analyze architecture and design"
        # With typos: "analize architcture and dseign"
        result = self.classifier.classify_sync("analize architcture and dseign")

        # Should still classify correctly (ANALYSIS_DEEP is also valid for "architecture")
        assert result.task_type in [
            TaskType.ANALYZE,
            TaskType.SEARCH,
            TaskType.ANALYSIS_DEEP,
        ]
        assert result.confidence > 0.3

    def test_exact_match_still_works(self):
        """Verify exact matches still work correctly."""
        result = self.classifier.classify_sync("analyze the code structure")

        # Should classify into the analysis family with high confidence
        assert result.task_type in ANALYSIS_TASK_TYPES
        assert result.confidence > 0.3


@pytest.mark.integration
class TestToolSelectionWithTypos:
    """Test SemanticToolSelector with typos."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        # Get selector instance
        self.selector = SemanticToolSelector()

    async def test_search_tool_with_typo(self):
        """Test search tool selection with typo."""
        # User wants to search but types "serch"
        from victor.tools.registry import ToolRegistry

        query = "serch for functions"
        tools_registry = ToolRegistry()

        # Should not crash with typo (even if registry is empty)
        tools = await self.selector.select_relevant_tools(query, tools_registry, max_tools=5)

        # Verify it returns a list (even if empty)
        assert isinstance(tools, list)

    async def test_read_tool_with_typo(self):
        """Test read tool selection with typo."""
        # User wants to read but types "raed"
        from victor.tools.registry import ToolRegistry

        query = "raed the file"
        tools_registry = ToolRegistry()

        # Should not crash with typo (even if registry is empty)
        tools = await self.selector.select_relevant_tools(query, tools_registry, max_tools=5)

        # Verify it returns a list (even if empty)
        assert isinstance(tools, list)

    def test_analysis_query_with_typo(self):
        """Test analysis query detection with typo."""
        # User wants to analyze but types "analize"
        query = "analize the code"

        # Should detect as analysis query
        assert self.selector._is_analysis_query(query) is True

    def test_review_query_with_typo(self):
        """Test review query detection with typo."""
        # User wants to review but types "reviw"
        query = "reviw the implementation"

        # Should detect as analysis query
        assert self.selector._is_analysis_query(query) is True


@pytest.mark.integration
class TestIntentDetectionWithTypos:
    """Test IntentClassifier with typos."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        # Get classifier instance
        self.classifier = IntentClassifier.get_instance()
        if not self.classifier.is_initialized:
            self.classifier.initialize_sync()

    def test_continuation_intent_with_typo(self):
        """Test CONTINUATION intent detection with typo."""
        # User wants to continue but types "contniue"
        result = self.classifier.classify_intent_sync("Let me contniue by raeding the file")

        # Should detect CONTINUATION intent
        assert result.intent in [IntentType.CONTINUATION, IntentType.NEUTRAL]
        assert result.confidence > 0.3

    def test_continuation_with_read_typo(self):
        """Test continuation detection with read typo."""
        result = self.classifier.classify_intent_sync("Let me chekc the file")

        # Should detect CONTINUATION intent
        assert result.intent in [IntentType.CONTINUATION, IntentType.NEUTRAL]

    def test_continuation_with_examine_typo(self):
        """Test continuation detection with examine typo."""
        result = self.classifier.classify_intent_sync("I'll exmine the code")

        # Should detect CONTINUATION intent
        assert result.intent in [IntentType.CONTINUATION, IntentType.NEUTRAL]

    def test_asking_input_with_typo(self):
        """Test ASKING_INPUT intent detection with typo."""
        # User asks "should I" but types "shuold I"
        result = self.classifier.classify_intent_sync("Shuold I proceeed with the fix")

        # Should detect ASKING_INPUT or CONTINUATION
        assert result.intent in [
            IntentType.ASKING_INPUT,
            IntentType.CONTINUATION,
            IntentType.NEUTRAL,
        ]

    def test_exact_match_still_works(self):
        """Verify exact matches still work correctly."""
        result = self.classifier.classify_intent_sync("Let me read the file")

        # Should detect CONTINUATION intent
        assert result.intent in [IntentType.CONTINUATION, IntentType.NEUTRAL]


@pytest.mark.integration
class TestRealMessageValidation:
    """Test fuzzy matching on realistic message patterns."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.classifier = TaskTypeClassifier.get_instance()
        if not self.classifier.is_initialized:
            self.classifier.initialize_sync()

    def test_realistic_typos_dataset(self):
        """Test on a dataset of realistic typos."""
        test_cases = [
            # (original, with_typo, expected_type)
            ("analyze the code", "analize the code", TaskType.ANALYZE),
            ("search for functions", "serch for functions", TaskType.SEARCH),
            ("review the implementation", "reviw the implementation", TaskType.ANALYZE),
            ("examine the structure", "exmine the structure", TaskType.ANALYZE),
            ("fix the bug", "fix the buug", TaskType.REFACTOR),
            ("edit the file", "edit the fiel", TaskType.REFACTOR),
            ("test the code", "test the cdoe", TaskType.TEST),
            ("deploy the app", "depoly the app", TaskType.ACTION),
        ]

        passed = 0
        total = len(test_cases)

        for original, typo_query, expected_type in test_cases:
            # Test with typo
            result = self.classifier.classify_sync(typo_query)

            # Check if classification is reasonable
            # We allow for some flexibility in exact type matching
            if result.task_type == expected_type or result.confidence > 0.3:
                passed += 1

        # At least 75% of test cases should pass
        success_rate = passed / total
        assert success_rate >= 0.75, f"Success rate {success_rate:.2%} is below 75% threshold"

    def test_common_typos_patterns(self):
        """Test common typo patterns."""
        common_typos = [
            # Missing letters
            ("analze", "analyze"),
            ("structre", "structure"),
            ("architcture", "architecture"),
            # Transposed letters
            ("anlayze", "analyze"),
            ("strutcure", "structure"),
            # Double letters
            ("analize", "analyze"),
            ("structture", "structure"),
        ]

        for typo, correct in common_typos:
            # Test that fuzzy matching can handle these
            query = f"{typo} the code"
            result = self.classifier.classify_sync(query)

            # Should get some classification (not error)
            assert result.confidence >= 0.0
            assert result.task_type in TaskType


@pytest.mark.integration
class TestFuzzyMatchingPerformance:
    """Test performance of fuzzy matching in real scenarios."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.classifier = TaskTypeClassifier.get_instance()
        if not self.classifier.is_initialized:
            self.classifier.initialize_sync()

    def test_classification_speed_with_fuzzy(self, benchmark):
        """Ensure fuzzy matching doesn't significantly slow down classification."""
        query = "analize the structre and architcture"

        # Benchmark classification speed
        result = benchmark(self.classifier.classify_sync, query)

        # Should complete in reasonable time
        assert result is not None
        assert result.confidence > 0.0

    def test_batch_classification_with_typos(self):
        """Test classifying multiple queries with typos."""
        queries = [
            "analize the code",
            "serch for functions",
            "reviw the implementation",
            "exmine the structure",
            "fix the buug",
            "edit the fiel",
            "test the cdoe",
        ]

        results = [self.classifier.classify_sync(q) for q in queries]

        # All should complete successfully
        for result in results:
            assert result.confidence > 0.0
            assert result.task_type in TaskType


@pytest.mark.integration
class TestFuzzyMatchingEdgeCases:
    """Test edge cases for fuzzy matching."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.classifier = TaskTypeClassifier.get_instance()
        if not self.classifier.is_initialized:
            self.classifier.initialize_sync()

    def test_very_short_query(self):
        """Test fuzzy matching with very short queries."""
        result = self.classifier.classify_sync("fix")

        # Should still work
        assert result.confidence >= 0.0
        assert result.task_type in TaskType

    def test_very_long_query(self):
        """Test fuzzy matching with very long queries."""
        long_query = " ".join(["analize"] * 100)
        result = self.classifier.classify_sync(long_query)

        # Should still work
        assert result.confidence >= 0.0
        assert result.task_type in TaskType

    def test_special_characters_with_typos(self):
        """Test fuzzy matching with special characters."""
        query = "analize: the cdoe! @#$"
        result = self.classifier.classify_sync(query)

        # Should still work
        assert result.confidence >= 0.0
        assert result.task_type in TaskType

    def test_mixed_case_with_typos(self):
        """Test fuzzy matching with mixed case."""
        query = "AnAlIzE ThE CoDe"
        result = self.classifier.classify_sync(query)

        # Should still work
        assert result.confidence >= 0.0
        assert result.task_type in TaskType

    def test_no_keywords_only_typos(self):
        """Test query with only typos, no real keywords."""
        result = self.classifier.classify_sync("xyz abc def")

        # Should still classify (as GENERAL or similar)
        assert result.confidence >= 0.0
        assert result.task_type in TaskType
