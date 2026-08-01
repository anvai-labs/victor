# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for data analysis tool dependency configuration (YAML-backed)."""

from pathlib import Path

import pytest

import victor_dataanalysis.tool_dependencies as td
from victor_dataanalysis.tool_dependencies import (
    DataAnalysisToolDependencyProvider,
    get_provider,
)


class TestYamlConfig:
    def test_yaml_file_ships_with_package(self):
        yaml_path = Path(td.__file__).parent / "tool_dependencies.yaml"

        assert yaml_path.is_file()


class TestProviderContract:
    def test_provider_class_is_deprecated_but_functional(self):
        with pytest.warns(DeprecationWarning, match="DataAnalysisToolDependencyProvider"):
            provider = DataAnalysisToolDependencyProvider()

        deps = provider.get_dependencies()
        assert isinstance(deps, list)
        assert deps, "expected non-empty tool dependencies"

    def test_provider_vertical_name(self):
        with pytest.warns(DeprecationWarning):
            provider = DataAnalysisToolDependencyProvider()

        assert provider.vertical == "data_analysis"

    def test_required_tools_cover_analysis_basics(self):
        with pytest.warns(DeprecationWarning):
            provider = DataAnalysisToolDependencyProvider()

        assert provider.get_required_tools() == {"read", "write", "shell"}

    def test_entry_point_factory_returns_provider(self):
        with pytest.warns(DeprecationWarning):
            provider = get_provider()

        assert isinstance(provider, DataAnalysisToolDependencyProvider)


class TestDeprecatedConstants:
    def test_legacy_transitions_access_warns_and_resolves(self):
        with pytest.warns(DeprecationWarning):
            transitions = dict(td.DATA_ANALYSIS_TOOL_TRANSITIONS)

        assert transitions, "expected non-empty transitions"

    def test_legacy_required_tools_access_warns(self):
        with pytest.warns(DeprecationWarning):
            required = set(td.DATA_ANALYSIS_REQUIRED_TOOLS)

        assert required == {"read", "write", "shell"}

    def test_legacy_clusters_access_warns_and_resolves(self):
        with pytest.warns(DeprecationWarning):
            clusters = dict(td.DATA_ANALYSIS_TOOL_CLUSTERS)

        assert clusters, "expected non-empty clusters"
