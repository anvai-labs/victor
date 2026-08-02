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

"""Keyword-based classification utilities for enrichment.

This module provides configurable keyword-based classification of text,
consolidating previously duplicated logic across verticals.

Example:
    classifier = KeywordClassifier(ANALYSIS_TYPES)
    types = classifier.classify("I need to run a correlation analysis")
    # Returns: ["correlation"]
"""

from typing import Dict, List, Set

# Data analysis type keywords
ANALYSIS_TYPES: Dict[str, List[str]] = {
    "correlation": [
        "correlation",
        "correlate",
        "covariance",
        "relationship",
        "r-squared",
    ],
    "regression": ["regression", "linear", "predict", "fit", "coefficients", "ols"],
    "clustering": ["cluster", "kmeans", "k-means", "grouping", "centroid", "dbscan"],
    "classification": [
        "classify",
        "classification",
        "logistic",
        "decision tree",
        "random forest",
    ],
    "time_series": ["time series", "forecast", "trend", "seasonal", "arima", "prophet"],
    "statistical_test": [
        "t-test",
        "chi-square",
        "anova",
        "p-value",
        "hypothesis",
        "significance",
    ],
    "visualization": [
        "plot",
        "chart",
        "graph",
        "visualize",
        "histogram",
        "scatter",
        "heatmap",
    ],
    "profiling": ["profile", "describe", "summary", "statistics", "distribution"],
}

# DevOps infrastructure type keywords
INFRA_TYPES: Dict[str, List[str]] = {
    "docker": ["docker", "container", "dockerfile", "compose", "image", "registry"],
    "kubernetes": [
        "kubernetes",
        "k8s",
        "kubectl",
        "pod",
        "deployment",
        "service",
        "helm",
    ],
    "terraform": ["terraform", "infrastructure as code", "iac", "tfstate", "provider"],
    "ci_cd": [
        "pipeline",
        "ci/cd",
        "github actions",
        "jenkins",
        "gitlab ci",
        "workflow",
    ],
    "aws": ["aws", "amazon", "s3", "ec2", "lambda", "cloudformation", "dynamodb"],
    "azure": ["azure", "arm", "blob", "aks", "azure functions"],
    "gcp": ["gcp", "google cloud", "gke", "bigquery", "cloud run"],
}

# Research task type keywords
RESEARCH_TYPES: Dict[str, List[str]] = {
    "literature_review": [
        "review",
        "survey",
        "literature",
        "state of the art",
        "related work",
    ],
    "fact_checking": ["fact check", "verify", "accurate", "claim", "source"],
    "summarization": ["summarize", "summary", "overview", "key points", "tldr"],
    "comparison": [
        "compare",
        "comparison",
        "versus",
        "vs",
        "difference",
        "similarities",
    ],
    "synthesis": ["synthesize", "combine", "integrate", "cross-reference"],
}


class KeywordClassifier:
    """Classify text based on keyword presence.

    Provides flexible, case-insensitive keyword matching for text
    classification across multiple categories.

    Example:
        classifier = KeywordClassifier({
            "urgent": ["asap", "urgent", "critical", "immediately"],
            "bug": ["bug", "error", "crash", "broken", "fix"],
        })
        categories = classifier.classify("This is an urgent bug fix")
        # ["urgent", "bug"]
    """

    def __init__(self, keyword_map: Dict[str, List[str]]) -> None:
        """Initialize with keyword mapping.

        Args:
            keyword_map: Dict mapping category name to list of keywords
        """
        self.keyword_map = keyword_map

    def classify(self, text: str) -> List[str]:
        """Classify text into matching categories.

        Args:
            text: Text to classify (case-insensitive matching)

        Returns:
            List of matching category names, in order of keyword_map
        """
        if not text:
            return []

        text_lower = text.lower()
        matches: List[str] = []

        for category, keywords in self.keyword_map.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    matches.append(category)
                    break  # Found match, move to next category

        return matches

    def classify_with_scores(self, text: str) -> Dict[str, int]:
        """Classify text with match count scores.

        Args:
            text: Text to classify

        Returns:
            Dict of category -> number of keyword matches
        """
        if not text:
            return {}

        text_lower = text.lower()
        scores: Dict[str, int] = {}

        for category, keywords in self.keyword_map.items():
            count = sum(1 for kw in keywords if kw.lower() in text_lower)
            if count > 0:
                scores[category] = count

        return scores

    def top_category(self, text: str) -> str | None:
        """Get the category with most keyword matches.

        Args:
            text: Text to classify

        Returns:
            Top matching category or None if no matches
        """
        scores = self.classify_with_scores(text)
        if not scores:
            return None
        return max(scores, key=scores.get)

    def has_category(self, text: str, category: str) -> bool:
        """Check if text matches a specific category.

        Args:
            text: Text to check
            category: Category to check for

        Returns:
            True if text matches the category
        """
        if category not in self.keyword_map:
            return False

        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in self.keyword_map[category])


def create_combined_classifier(*keyword_dicts: Dict[str, List[str]]) -> KeywordClassifier:
    """Create a classifier combining multiple keyword dictionaries.

    Args:
        *keyword_dicts: Keyword dictionaries to combine

    Returns:
        KeywordClassifier with all keywords merged

    Example:
        classifier = create_combined_classifier(ANALYSIS_TYPES, INFRA_TYPES)
    """
    combined: Dict[str, List[str]] = {}
    for kd in keyword_dicts:
        for category, keywords in kd.items():
            if category in combined:
                combined[category].extend(keywords)
            else:
                combined[category] = list(keywords)
    return KeywordClassifier(combined)


__all__ = [
    "ANALYSIS_TYPES",
    "INFRA_TYPES",
    "RESEARCH_TYPES",
    "KeywordClassifier",
    "create_combined_classifier",
]
