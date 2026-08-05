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

"""File pattern matching utilities for enrichment.

This module provides configurable file pattern matching for categorizing
files by type, consolidating previously duplicated logic across verticals.

Example:
    matcher = FilePatternMatcher(DEVOPS_PATTERNS)
    matches = matcher.match(["Dockerfile", "main.py", "deployment.yaml"])
    # Returns: {"docker": ["Dockerfile"], "kubernetes": ["deployment.yaml"]}
"""

import fnmatch
from typing import Dict, List

# DevOps infrastructure patterns
DEVOPS_PATTERNS: Dict[str, List[str]] = {
    "docker": [
        "Dockerfile",
        "Dockerfile.*",
        "docker-compose*.yml",
        "docker-compose*.yaml",
        ".dockerignore",
    ],
    "kubernetes": [
        "*.yaml",
        "*.yml",
        "kustomization.yaml",
        "Chart.yaml",
        "values.yaml",
        "templates/*.yaml",
    ],
    "terraform": ["*.tf", "*.tfvars", "terraform.tfstate*", ".terraform*"],
    "ci_cd": [
        ".github/workflows/*.yml",
        ".github/workflows/*.yaml",
        "Jenkinsfile",
        ".gitlab-ci.yml",
        ".circleci/config.yml",
        "azure-pipelines.yml",
        ".travis.yml",
    ],
    "ansible": [
        "playbook*.yml",
        "playbook*.yaml",
        "inventory*",
        "roles/*/tasks/*.yml",
        "ansible.cfg",
    ],
    "helm": ["Chart.yaml", "values*.yaml", "templates/*.yaml"],
}

# Data file patterns
DATA_PATTERNS: Dict[str, List[str]] = {
    "csv": ["*.csv"],
    "excel": ["*.xlsx", "*.xls", "*.xlsm"],
    "parquet": ["*.parquet", "*.pq"],
    "json": ["*.json", "*.jsonl", "*.ndjson"],
    "sql": ["*.sql"],
    "database": ["*.db", "*.sqlite", "*.sqlite3"],
    "arrow": ["*.arrow", "*.feather"],
    "avro": ["*.avro"],
}

# Code file patterns
CODE_PATTERNS: Dict[str, List[str]] = {
    "python": ["*.py", "*.pyi", "*.pyx"],
    "javascript": ["*.js", "*.jsx", "*.mjs", "*.cjs"],
    "typescript": ["*.ts", "*.tsx"],
    "rust": ["*.rs"],
    "go": ["*.go"],
    "java": ["*.java"],
    "cpp": ["*.cpp", "*.hpp", "*.cc", "*.hh", "*.c", "*.h"],
}


class FilePatternMatcher:
    """Match file paths against pattern categories.

    Provides flexible pattern matching with support for glob patterns
    and category-based file classification.

    Example:
        matcher = FilePatternMatcher({
            "docker": ["Dockerfile", "docker-compose*"],
            "config": ["*.yaml", "*.yml", "*.json"],
        })
        result = matcher.match(["Dockerfile", "config.yaml", "main.py"])
        # {"docker": ["Dockerfile"], "config": ["config.yaml"]}
    """

    def __init__(self, categories: Dict[str, List[str]]) -> None:
        """Initialize with pattern categories.

        Args:
            categories: Dict mapping category name to list of glob patterns
        """
        self.categories = categories

    def match(self, paths: List[str]) -> Dict[str, List[str]]:
        """Match paths to categories.

        Args:
            paths: List of file paths to match

        Returns:
            Dict of category -> list of matching paths.
            Only categories with matches are included.
        """
        result: Dict[str, List[str]] = {cat: [] for cat in self.categories}

        for path in paths:
            # Get just the filename for pattern matching
            filename = path.rsplit("/", 1)[-1] if "/" in path else path

            for category, patterns in self.categories.items():
                for pattern in patterns:
                    # Match against filename or full path
                    if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(path, pattern):
                        if path not in result[category]:
                            result[category].append(path)
                        break

        # Remove empty categories
        return {k: v for k, v in result.items() if v}

    def match_category(self, path: str) -> List[str]:
        """Get all categories a path matches.

        Args:
            path: Single file path

        Returns:
            List of matching category names
        """
        matches = self.match([path])
        return list(matches.keys())

    def has_match(self, paths: List[str], category: str) -> bool:
        """Check if any path matches a category.

        Args:
            paths: List of paths to check
            category: Category to check for

        Returns:
            True if any path matches the category
        """
        if category not in self.categories:
            return False

        for path in paths:
            filename = path.rsplit("/", 1)[-1] if "/" in path else path
            for pattern in self.categories[category]:
                if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(path, pattern):
                    return True
        return False


def create_combined_matcher(*pattern_dicts: Dict[str, List[str]]) -> FilePatternMatcher:
    """Create a matcher combining multiple pattern dictionaries.

    Args:
        *pattern_dicts: Pattern dictionaries to combine

    Returns:
        FilePatternMatcher with all patterns merged

    Example:
        matcher = create_combined_matcher(DEVOPS_PATTERNS, DATA_PATTERNS)
    """
    combined: Dict[str, List[str]] = {}
    for pd in pattern_dicts:
        for category, patterns in pd.items():
            if category in combined:
                combined[category].extend(patterns)
            else:
                combined[category] = list(patterns)
    return FilePatternMatcher(combined)


__all__ = [
    "DEVOPS_PATTERNS",
    "DATA_PATTERNS",
    "CODE_PATTERNS",
    "FilePatternMatcher",
    "create_combined_matcher",
]
