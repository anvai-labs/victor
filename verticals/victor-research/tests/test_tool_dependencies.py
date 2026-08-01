# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for research tool dependency configuration (YAML-backed)."""

from pathlib import Path

import pytest

import victor_research.tool_dependencies as td
from victor_research.tool_dependencies import (
    RESEARCH_REQUIRED_TOOLS,
    RESEARCH_TOOL_CLUSTERS,
    RESEARCH_TOOL_DEPENDENCIES,
    RESEARCH_TOOL_SEQUENCES,
    RESEARCH_TOOL_TRANSITIONS,
    ResearchToolDependencyProvider,
    get_provider,
)


class TestYamlConfig:
    def test_yaml_file_ships_with_package(self):
        yaml_path = Path(td.__file__).parent / "tool_dependencies.yaml"

        assert yaml_path.is_file()

    def test_legacy_constants_loaded_from_yaml(self):
        """Research eagerly derives legacy constants at import time."""
        assert RESEARCH_TOOL_TRANSITIONS, "transitions empty — YAML load failed"
        assert RESEARCH_TOOL_CLUSTERS, "clusters empty — YAML load failed"
        assert RESEARCH_TOOL_SEQUENCES, "sequences empty — YAML load failed"
        assert RESEARCH_TOOL_DEPENDENCIES, "dependencies empty — YAML load failed"
        assert RESEARCH_REQUIRED_TOOLS, "required tools empty — YAML load failed"

    def test_web_tools_are_required_for_research(self):
        assert "web_search" in RESEARCH_REQUIRED_TOOLS


class TestProviderContract:
    def test_provider_class_is_deprecated_but_functional(self):
        with pytest.warns(DeprecationWarning, match="ResearchToolDependencyProvider"):
            provider = ResearchToolDependencyProvider()

        deps = provider.get_dependencies()
        assert isinstance(deps, list)
        assert deps, "provider returned no dependencies"

    def test_entry_point_factory_returns_provider(self):
        with pytest.warns(DeprecationWarning):
            provider = get_provider()

        assert isinstance(provider, ResearchToolDependencyProvider)

    def test_provider_dependencies_match_legacy_constants(self):
        with pytest.warns(DeprecationWarning):
            provider = ResearchToolDependencyProvider()

        provider_tools = {dep.tool_name for dep in provider.get_dependencies()}
        legacy_tools = {dep.tool_name for dep in RESEARCH_TOOL_DEPENDENCIES}
        assert provider_tools == legacy_tools


class TestTransitionShapes:
    def test_transitions_map_tool_to_weighted_successors(self):
        for tool, successors in RESEARCH_TOOL_TRANSITIONS.items():
            assert isinstance(tool, str) and tool
            for entry in successors:
                next_tool, weight = entry
                assert isinstance(next_tool, str) and next_tool
                assert 0 < weight <= 1.0, f"{tool}->{next_tool} weight {weight}"

    def test_sequences_are_non_trivial(self):
        for name, sequence in RESEARCH_TOOL_SEQUENCES.items():
            assert len(sequence) >= 2, f"sequence {name} has fewer than 2 steps"
