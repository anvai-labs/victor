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

"""Tests for PromptRequirementExtractor."""

import pytest

from victor.agent.prompt_requirement_extractor import (
    PromptRequirementExtractor,
    PromptRequirements,
    RequirementPattern,
    extract_prompt_requirements,
    get_prompt_requirement_extractor,
)


class TestPromptRequirements:
    """Tests for PromptRequirements dataclass."""

    def test_default_values(self):
        """Test default values are None/empty."""
        req = PromptRequirements()
        assert req.file_count is None
        assert req.fix_count is None
        assert req.finding_count is None
        assert req.tool_budget is None
        assert req.confidence == 0.0

    def test_has_explicit_requirements_false(self):
        """Test has_explicit_requirements returns False when empty."""
        req = PromptRequirements()
        assert req.has_explicit_requirements() is False

    def test_has_explicit_requirements_true(self):
        """Test has_explicit_requirements returns True when set."""
        req = PromptRequirements(file_count=5)
        assert req.has_explicit_requirements() is True

        req = PromptRequirements(fix_count=3)
        assert req.has_explicit_requirements() is True

    def test_compute_budgets_with_file_count(self):
        """Test budget computation based on file count."""
        req = PromptRequirements(file_count=9)
        req.compute_budgets()

        # 9 files * 3 tools/file * 1.5 buffer = 40.5 -> 40
        assert req.tool_budget >= 40
        assert req.iteration_budget >= 67  # 9 * 5 * 1.5

    def test_compute_budgets_defaults_when_no_requirements(self):
        """Test default budgets when no requirements."""
        req = PromptRequirements()
        req.compute_budgets(default_tool_budget=50)

        assert req.tool_budget == 50

    def test_to_dict(self):
        """Test to_dict serialization."""
        req = PromptRequirements(file_count=5, fix_count=3)
        req.compute_budgets()
        d = req.to_dict()

        assert d["file_count"] == 5
        assert d["fix_count"] == 3
        assert d["has_explicit"] is True
        assert "tool_budget" in d


class TestPromptRequirementExtractor:
    """Tests for PromptRequirementExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return PromptRequirementExtractor()

    def test_extract_file_count_basic(self, extractor):
        """Test basic file count extraction."""
        req = extractor.extract("Read the following 9 files")
        assert req.file_count == 9

    def test_extract_file_count_variants(self, extractor):
        """Test various file count phrasings."""
        test_cases = [
            ("review 5 files", 5),
            ("analyze the 10 files", 10),
            ("check 3 files first", 3),
            ("examine these 7 files", 7),
        ]
        for prompt, expected in test_cases:
            req = extractor.extract(prompt)
            assert req.file_count == expected, f"Failed for: {prompt}"

    def test_extract_fix_count(self, extractor):
        """Test fix count extraction."""
        test_cases = [
            ("provide top 3 fixes", 3),
            ("suggest 5 fixes", 5),
            ("list 10 fixes", 10),
        ]
        for prompt, expected in test_cases:
            req = extractor.extract(prompt)
            assert req.fix_count == expected, f"Failed for: {prompt}"

    def test_extract_finding_count_range(self, extractor):
        """Test finding count with range (takes max)."""
        req = extractor.extract("Prioritize 6-10 findings")
        assert req.finding_count == 10  # Takes max of range

    def test_extract_finding_count_simple(self, extractor):
        """Test simple finding count."""
        req = extractor.extract("identify 5 key issues")
        assert req.finding_count == 5

    def test_extract_recommendation_count(self, extractor):
        """Test recommendation count extraction."""
        req = extractor.extract("provide top 5 recommendations")
        assert req.recommendation_count == 5

    def test_extract_combined_requirements(self, extractor):
        """Test extracting multiple requirements from one prompt."""
        prompt = "Review 9 files and provide top 3 fixes with 6-10 findings"
        req = extractor.extract(prompt)

        assert req.file_count == 9
        assert req.fix_count == 3
        assert req.finding_count == 10

    def test_real_audit_prompt(self, extractor):
        """Test with realistic audit prompt."""
        prompt = """You are auditing the Victor codebase. Focus on framework + vertical integration.

Requirements:
1) Read the following files first and cite them explicitly in your findings:
   - victor/framework/vertical_integration.py
   - victor/framework/step_handlers.py
   - victor/core/verticals/base.py
   - victor/core/verticals/protocols.py
   - victor/agent/orchestrator.py
   - victor/agent/tool_selection.py
   - victor/agent/tool_pipeline.py
   - victor/observability/event_bus.py
   - victor/workflows/registry.py
2) For each finding, include file path, what the code does, why it's a gap.
3) Prioritize 6-10 findings max, ordered by severity.
4) End with a short "Top 3 fixes" list."""

        req = extractor.extract(prompt)

        # Should extract finding range and fix count
        assert req.finding_count == 10  # "6-10 findings" -> takes max
        assert req.fix_count == 3  # "Top 3 fixes"
        assert req.has_explicit_requirements() is True

    def test_confidence_calculation(self, extractor):
        """Test confidence increases with more matches."""
        # Single match
        req1 = extractor.extract("read 5 files")
        assert req1.confidence > 0

        # Multiple matches
        req2 = extractor.extract("read 5 files and provide top 3 fixes with 10 findings")
        assert req2.confidence > req1.confidence

    def test_budget_computation(self, extractor):
        """Test dynamic budget computation."""
        req = extractor.extract("Review 9 files and provide top 3 fixes")

        assert req.tool_budget is not None
        assert req.tool_budget >= 40  # 9 files * 3 tools * 1.5 buffer
        assert req.iteration_budget is not None

    def test_no_requirements(self, extractor):
        """Test prompt with no explicit requirements."""
        req = extractor.extract("Help me fix this bug")

        assert req.file_count is None
        assert req.fix_count is None
        assert req.has_explicit_requirements() is False
        assert req.confidence == 0.0

    def test_custom_buffer_multiplier(self):
        """Test custom buffer multiplier."""
        extractor = PromptRequirementExtractor(buffer_multiplier=2.0)
        req = extractor.extract("read 10 files")

        # 10 files * 3 tools * 2.0 buffer = 60
        assert req.tool_budget >= 60

    def test_add_custom_pattern(self, extractor):
        """Test adding custom pattern."""
        # Add pattern for "items"
        extractor.add_pattern(
            RequirementPattern(
                name="item_count",
                patterns=[r"(\d+)\s+items?"],
                priority=50,
            )
        )

        # Since item_count isn't a field in PromptRequirements,
        # it won't be set, but we can verify the pattern matches
        req = extractor.extract("process 5 items")
        assert "item_count" in req.raw_matches


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_prompt_requirement_extractor_singleton(self):
        """Test singleton pattern."""
        ext1 = get_prompt_requirement_extractor()
        ext2 = get_prompt_requirement_extractor()
        assert ext1 is ext2

    def test_extract_prompt_requirements_function(self):
        """Test convenience function."""
        req = extract_prompt_requirements("read 5 files")
        assert req.file_count == 5


class TestIntegrationWithTaskTracker:
    """Integration tests with UnifiedTaskTracker."""

    def test_create_tracker_with_prompt_requirements(self):
        """Test tracker creation with prompt requirements."""
        from victor.agent.unified_task_tracker import (
            create_tracker_with_prompt_requirements,
        )

        tracker, task_type, details = create_tracker_with_prompt_requirements(
            "Review 9 files and provide top 3 fixes"
        )

        assert "prompt_requirements" in details
        assert details["prompt_requirements"]["file_count"] == 9
        assert details["prompt_requirements"]["fix_count"] == 3
        assert tracker._progress.has_prompt_requirements is True

    def test_dynamic_budget_applied(self):
        """Test that dynamic budget is applied to tracker."""
        from victor.agent.unified_task_tracker import (
            create_tracker_with_prompt_requirements,
        )

        tracker, _, details = create_tracker_with_prompt_requirements(
            "Review 20 files"  # Large file count should increase budget
        )

        # Budget should be increased from default
        assert tracker._progress.tool_budget >= 50  # At least default
        assert details["recommended_budget"] >= 50

    def test_soft_limits_enabled_with_requirements(self):
        """Test soft limits are enabled when prompt has requirements."""
        from victor.agent.unified_task_tracker import (
            create_tracker_with_prompt_requirements,
        )

        tracker, _, _ = create_tracker_with_prompt_requirements("Read 5 files and provide 3 fixes")

        assert tracker._progress.has_prompt_requirements is True
        # Soft limit buffer should allow 50% overage (set in should_stop)
