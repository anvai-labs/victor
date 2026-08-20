# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for ExplorationCoordinator."""

from pathlib import Path

import pytest

from victor.agent.services.exploration_runtime import (
    ExplorationCoordinator,
    ExplorationResult,
)


class TestExplorationCoordinator:
    """Tests for parallel exploration."""

    def test_legacy_module_reexports_service_runtime(self):
        """Legacy coordinator path should re-export service-owned exploration runtime."""
        from victor.agent.services.exploration_runtime import (
            ExplorationCoordinator as legacy_coordinator,
            ExplorationResult as legacy_result,
        )

        assert legacy_coordinator is ExplorationCoordinator
        assert legacy_result is ExplorationResult

    def test_extract_search_terms_from_camelcase(self):
        """Extracts CamelCase identifiers."""
        coord = ExplorationCoordinator()
        terms = coord._extract_search_terms("Fix the bug in SeparabilityMatrix for CompoundModel")
        assert "SeparabilityMatrix" in terms
        assert "CompoundModel" in terms

    def test_extract_search_terms_from_snake_case(self):
        """Extracts snake_case identifiers."""
        coord = ExplorationCoordinator()
        terms = coord._extract_search_terms(
            "The separability_matrix function returns wrong results"
        )
        assert "separability_matrix" in terms

    def test_extract_search_terms_from_backticks(self):
        """Extracts backtick-quoted terms."""
        coord = ExplorationCoordinator()
        terms = coord._extract_search_terms("Fix the `compute_matrix` function in `core.py`")
        assert "compute_matrix" in terms

    def test_extract_no_terms_from_generic(self):
        """Generic text returns empty terms."""
        coord = ExplorationCoordinator()
        terms = coord._extract_search_terms("fix the bug")
        assert len(terms) == 0

    def test_exploration_result_defaults(self):
        """ExplorationResult has sensible defaults."""
        result = ExplorationResult()
        assert result.file_paths == []
        assert result.summary == ""
        assert result.duration_seconds == 0.0
        assert result.tool_calls == 0

    @pytest.mark.asyncio
    async def test_explore_skips_when_no_terms(self):
        """Exploration returns early if no search terms extracted."""
        coord = ExplorationCoordinator()
        result = await coord.explore_parallel(
            "fix the bug",
            project_root=Path("/tmp/nonexistent"),
        )
        assert result.tool_calls == 0
        assert result.file_paths == []
