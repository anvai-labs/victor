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

"""Test structural analysis task classification fix.

This test verifies that structural analysis tasks are correctly classified
to prevent the legacy tool budget truncation issue.
"""

import pytest

from victor.agent.unified_classifier import UnifiedTaskClassifier, ClassifierTaskType


class TestStructuralAnalysisClassification:
    """Verify structural analysis task classification works correctly."""

    def test_structural_analysis_classification(self):
        """Verify 'structural analysis' is classified as ANALYSIS."""
        classifier = UnifiedTaskClassifier()
        result = classifier.classify("framework structural analysis")

        assert result.task_type == ClassifierTaskType.ANALYSIS, (
            f"Expected ANALYSIS but got {result.task_type}. "
            f"This would cause tool_budget=10 instead of 200."
        )
        assert result.is_analysis_task is True
        assert result.confidence >= 0.5  # Single keyword match gives 0.5 confidence

    def test_framework_analysis_classification(self):
        """Verify 'framework analysis' is classified as ANALYSIS."""
        classifier = UnifiedTaskClassifier()
        result = classifier.classify("analyze the framework architecture")

        assert result.task_type == ClassifierTaskType.ANALYSIS
        assert result.is_analysis_task is True

    def test_architecture_analysis_classification(self):
        """Verify 'architecture analysis' is classified as ANALYSIS."""
        classifier = UnifiedTaskClassifier()
        result = classifier.classify("system architecture review")

        assert result.task_type == ClassifierTaskType.ANALYSIS
        assert result.is_analysis_task is True

    def test_code_structure_review_classification(self):
        """Verify 'code structure review' is classified as ANALYSIS."""
        classifier = UnifiedTaskClassifier()
        result = classifier.classify("code structure review")

        assert result.task_type == ClassifierTaskType.ANALYSIS
        assert result.is_analysis_task is True

    def test_deep_dive_architecture_classification(self):
        """Verify 'deep dive into architecture' is classified as ANALYSIS."""
        classifier = UnifiedTaskClassifier()
        result = classifier.classify("deep dive into framework architecture")

        assert result.task_type == ClassifierTaskType.ANALYSIS
        assert result.is_analysis_task is True

    def test_comprehensive_analysis_classification(self):
        """Verify comprehensive analysis is classified as ANALYSIS."""
        classifier = UnifiedTaskClassifier()
        result = classifier.classify("comprehensive framework analysis")

        assert result.task_type == ClassifierTaskType.ANALYSIS
        assert result.is_analysis_task is True

    def test_analysis_with_context_boost(self):
        """Verify classification works with conversation history context."""
        classifier = UnifiedTaskClassifier()
        history = [
            {"role": "user", "content": "I need to analyze the framework"},
            {"role": "assistant", "content": "I'll help you with structural analysis"},
        ]
        result = classifier.classify_with_context("continue the structural analysis", history)

        # Context should maintain analysis classification
        assert result.task_type == ClassifierTaskType.ANALYSIS
        assert result.is_analysis_task is True
        # Context boost may be 0 if history doesn't strongly signal analysis

    def test_legacy_dict_compatibility(self):
        """Verify legacy dict format includes is_analysis_task=True."""
        classifier = UnifiedTaskClassifier()
        result = classifier.classify("framework structural analysis")
        legacy = result.to_legacy_dict()

        assert legacy.get("is_analysis_task") is True
        assert legacy.get("coarse_task_type") in ("analysis", "default")

    def test_tool_budget_recommendation(self):
        """Verify structural analysis gets adequate tool budget."""
        classifier = UnifiedTaskClassifier()
        result = classifier.classify("comprehensive framework structural analysis")

        # Analysis tasks should get higher tool budget
        assert result.recommended_tool_budget >= 50, (
            f"Tool budget {result.recommended_tool_budget} too low for analysis task. "
            f"Should be at least 50 to avoid truncation."
        )
