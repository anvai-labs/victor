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

"""Unit tests for keyword classification utilities."""

import pytest
from victor.framework.enrichment.keyword_classifier import (
    ANALYSIS_TYPES,
    INFRA_TYPES,
    RESEARCH_TYPES,
    KeywordClassifier,
    create_combined_classifier,
)


class TestKeywordConstants:
    """Tests for keyword constants."""

    def test_analysis_types_has_correlation(self):
        """Test ANALYSIS_TYPES contains correlation."""
        assert "correlation" in ANALYSIS_TYPES
        assert "correlation" in ANALYSIS_TYPES["correlation"]

    def test_analysis_types_has_regression(self):
        """Test ANALYSIS_TYPES contains regression."""
        assert "regression" in ANALYSIS_TYPES

    def test_analysis_types_has_visualization(self):
        """Test ANALYSIS_TYPES contains visualization."""
        assert "visualization" in ANALYSIS_TYPES
        assert "plot" in ANALYSIS_TYPES["visualization"]

    def test_infra_types_has_docker(self):
        """Test INFRA_TYPES contains docker."""
        assert "docker" in INFRA_TYPES
        assert "docker" in INFRA_TYPES["docker"]

    def test_infra_types_has_kubernetes(self):
        """Test INFRA_TYPES contains kubernetes."""
        assert "kubernetes" in INFRA_TYPES

    def test_research_types_has_fact_checking(self):
        """Test RESEARCH_TYPES contains fact_checking."""
        assert "fact_checking" in RESEARCH_TYPES


class TestKeywordClassifier:
    """Tests for KeywordClassifier class."""

    def test_classify_empty_text(self):
        """Test classification of empty text."""
        classifier = KeywordClassifier({"test": ["keyword"]})
        result = classifier.classify("")
        assert result == []

    def test_classify_single_match(self):
        """Test classification with single match."""
        classifier = KeywordClassifier({"analysis": ["analyze", "correlation"]})
        result = classifier.classify("I need to analyze the data")
        assert "analysis" in result

    def test_classify_multiple_matches(self):
        """Test classification with multiple category matches."""
        classifier = KeywordClassifier(
            {
                "correlation": ["correlation", "correlate"],
                "visualization": ["plot", "chart", "graph"],
            }
        )
        result = classifier.classify("Create a correlation plot")
        assert "correlation" in result
        assert "visualization" in result

    def test_classify_case_insensitive(self):
        """Test classification is case insensitive."""
        classifier = KeywordClassifier({"docker": ["Docker", "container"]})
        result = classifier.classify("Using DOCKER for deployment")
        assert "docker" in result

    def test_classify_no_match(self):
        """Test classification with no matches."""
        classifier = KeywordClassifier({"docker": ["docker", "container"]})
        result = classifier.classify("This is about Python")
        assert result == []

    def test_classify_with_scores_single(self):
        """Test scoring with single keyword match."""
        classifier = KeywordClassifier({"docker": ["docker", "container", "image"]})
        result = classifier.classify_with_scores("Using docker for deployment")
        assert "docker" in result
        assert result["docker"] == 1

    def test_classify_with_scores_multiple(self):
        """Test scoring with multiple keyword matches."""
        classifier = KeywordClassifier({"docker": ["docker", "container", "image"]})
        result = classifier.classify_with_scores("Docker builds container images")
        assert result["docker"] >= 2

    def test_classify_with_scores_empty(self):
        """Test scoring with no matches."""
        classifier = KeywordClassifier({"docker": ["docker"]})
        result = classifier.classify_with_scores("Python code")
        assert result == {}

    def test_top_category(self):
        """Test getting top category."""
        classifier = KeywordClassifier(
            {
                "docker": ["docker"],
                "kubernetes": ["kubernetes", "k8s", "pod", "deployment"],
            }
        )
        result = classifier.top_category("Deploy kubernetes pods to the deployment")
        assert result == "kubernetes"

    def test_top_category_no_match(self):
        """Test top_category with no matches."""
        classifier = KeywordClassifier({"docker": ["docker"]})
        result = classifier.top_category("Python code")
        assert result is None

    def test_has_category_true(self):
        """Test has_category returns True when match."""
        classifier = KeywordClassifier({"docker": ["docker", "container"]})
        assert classifier.has_category("Using docker", "docker")

    def test_has_category_false(self):
        """Test has_category returns False when no match."""
        classifier = KeywordClassifier({"docker": ["docker", "container"]})
        assert not classifier.has_category("Using python", "docker")

    def test_has_category_unknown(self):
        """Test has_category returns False for unknown category."""
        classifier = KeywordClassifier({"docker": ["docker"]})
        assert not classifier.has_category("Using docker", "unknown")


class TestCreateCombinedClassifier:
    """Tests for create_combined_classifier function."""

    def test_combine_two_dicts(self):
        """Test combining two keyword dicts."""
        dict1 = {"docker": ["docker"]}
        dict2 = {"python": ["python"]}

        classifier = create_combined_classifier(dict1, dict2)
        result = classifier.classify("Using docker and python")

        assert "docker" in result
        assert "python" in result

    def test_combine_overlapping_categories(self):
        """Test combining dicts with overlapping categories."""
        dict1 = {"config": ["yaml"]}
        dict2 = {"config": ["json"]}

        classifier = create_combined_classifier(dict1, dict2)

        # Should match both keywords for "config"
        result1 = classifier.classify("Using yaml config")
        result2 = classifier.classify("Using json config")

        assert "config" in result1
        assert "config" in result2

    def test_combine_all_types(self):
        """Test combining all predefined types."""
        classifier = create_combined_classifier(ANALYSIS_TYPES, INFRA_TYPES, RESEARCH_TYPES)

        result = classifier.classify("Deploy docker and run correlation analysis")

        assert "docker" in result
        assert "correlation" in result
