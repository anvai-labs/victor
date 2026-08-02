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

"""Unit tests for file pattern matching utilities."""

import pytest
from victor.framework.enrichment.file_patterns import (
    DEVOPS_PATTERNS,
    DATA_PATTERNS,
    CODE_PATTERNS,
    FilePatternMatcher,
    create_combined_matcher,
)


class TestPatternConstants:
    """Tests for pattern constants."""

    def test_devops_patterns_has_docker(self):
        """Test DEVOPS_PATTERNS contains docker."""
        assert "docker" in DEVOPS_PATTERNS
        assert "Dockerfile" in DEVOPS_PATTERNS["docker"]

    def test_devops_patterns_has_kubernetes(self):
        """Test DEVOPS_PATTERNS contains kubernetes."""
        assert "kubernetes" in DEVOPS_PATTERNS

    def test_devops_patterns_has_terraform(self):
        """Test DEVOPS_PATTERNS contains terraform."""
        assert "terraform" in DEVOPS_PATTERNS
        assert "*.tf" in DEVOPS_PATTERNS["terraform"]

    def test_data_patterns_has_csv(self):
        """Test DATA_PATTERNS contains csv."""
        assert "csv" in DATA_PATTERNS
        assert "*.csv" in DATA_PATTERNS["csv"]

    def test_data_patterns_has_json(self):
        """Test DATA_PATTERNS contains json."""
        assert "json" in DATA_PATTERNS

    def test_code_patterns_has_python(self):
        """Test CODE_PATTERNS contains python."""
        assert "python" in CODE_PATTERNS
        assert "*.py" in CODE_PATTERNS["python"]


class TestFilePatternMatcher:
    """Tests for FilePatternMatcher class."""

    def test_match_empty_paths(self):
        """Test matching empty path list."""
        matcher = FilePatternMatcher({"docker": ["Dockerfile"]})
        result = matcher.match([])
        assert result == {}

    def test_match_single_category(self):
        """Test matching files to single category."""
        matcher = FilePatternMatcher({"docker": ["Dockerfile", "docker-compose*"]})
        result = matcher.match(["Dockerfile", "main.py"])
        assert "docker" in result
        assert "Dockerfile" in result["docker"]
        assert "main.py" not in result.get("docker", [])

    def test_match_multiple_categories(self):
        """Test matching files to multiple categories."""
        matcher = FilePatternMatcher(
            {
                "docker": ["Dockerfile"],
                "python": ["*.py"],
            }
        )
        result = matcher.match(["Dockerfile", "main.py", "config.yaml"])
        assert "docker" in result
        assert "python" in result
        assert "Dockerfile" in result["docker"]
        assert "main.py" in result["python"]

    def test_match_glob_patterns(self):
        """Test glob pattern matching."""
        matcher = FilePatternMatcher({"yaml": ["*.yaml", "*.yml"]})
        result = matcher.match(["config.yaml", "settings.yml", "data.json"])
        assert "yaml" in result
        assert "config.yaml" in result["yaml"]
        assert "settings.yml" in result["yaml"]

    def test_match_full_path(self):
        """Test matching with full file paths."""
        matcher = FilePatternMatcher({"docker": ["Dockerfile"]})
        result = matcher.match(["src/docker/Dockerfile", "Dockerfile"])
        assert "docker" in result
        assert len(result["docker"]) == 2

    def test_excludes_empty_categories(self):
        """Test empty categories are excluded from result."""
        matcher = FilePatternMatcher(
            {
                "docker": ["Dockerfile"],
                "rust": ["*.rs"],
            }
        )
        result = matcher.match(["Dockerfile"])
        assert "docker" in result
        assert "rust" not in result

    def test_no_duplicates_in_category(self):
        """Test no duplicates within a category."""
        matcher = FilePatternMatcher({"config": ["*.yaml", "*.yml", "config*"]})
        result = matcher.match(["config.yaml"])
        assert result["config"].count("config.yaml") == 1

    def test_match_category_single_path(self):
        """Test match_category for single path."""
        matcher = FilePatternMatcher(
            {
                "docker": ["Dockerfile"],
                "python": ["*.py"],
            }
        )
        categories = matcher.match_category("Dockerfile")
        assert "docker" in categories
        assert "python" not in categories

    def test_has_match_true(self):
        """Test has_match returns True when match exists."""
        matcher = FilePatternMatcher({"docker": ["Dockerfile"]})
        assert matcher.has_match(["Dockerfile", "main.py"], "docker")

    def test_has_match_false(self):
        """Test has_match returns False when no match."""
        matcher = FilePatternMatcher({"docker": ["Dockerfile"]})
        assert not matcher.has_match(["main.py"], "docker")

    def test_has_match_unknown_category(self):
        """Test has_match returns False for unknown category."""
        matcher = FilePatternMatcher({"docker": ["Dockerfile"]})
        assert not matcher.has_match(["Dockerfile"], "unknown")


class TestCreateCombinedMatcher:
    """Tests for create_combined_matcher function."""

    def test_combine_two_dicts(self):
        """Test combining two pattern dicts."""
        dict1 = {"docker": ["Dockerfile"]}
        dict2 = {"python": ["*.py"]}

        matcher = create_combined_matcher(dict1, dict2)
        result = matcher.match(["Dockerfile", "main.py"])

        assert "docker" in result
        assert "python" in result

    def test_combine_overlapping_categories(self):
        """Test combining dicts with overlapping categories."""
        dict1 = {"config": ["*.yaml"]}
        dict2 = {"config": ["*.json"]}

        matcher = create_combined_matcher(dict1, dict2)
        result = matcher.match(["config.yaml", "config.json"])

        assert "config" in result
        assert "config.yaml" in result["config"]
        assert "config.json" in result["config"]

    def test_combine_all_pattern_types(self):
        """Test combining all predefined pattern types."""
        matcher = create_combined_matcher(DEVOPS_PATTERNS, DATA_PATTERNS, CODE_PATTERNS)
        result = matcher.match(["Dockerfile", "data.csv", "main.py"])

        assert "docker" in result
        assert "csv" in result
        assert "python" in result
