# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for package exports and victor-vertical.toml metadata."""

import tomllib
from pathlib import Path

import pytest

import victor_dataanalysis

_PACKAGE_DIR = Path(victor_dataanalysis.__file__).parent


class TestPackageExports:
    def test_every_declared_export_resolves(self):
        for name in victor_dataanalysis.__all__:
            attr = getattr(victor_dataanalysis, name)
            assert attr is not None, f"{name} resolved to None"

    def test_core_exports_declared(self):
        for name in (
            "DataAnalysisAssistant",
            "DataAnalysisPromptContributor",
            "DataAnalysisModeConfigProvider",
            "DataAnalysisSafetyExtension",
            "DataAnalysisToolDependencyProvider",
            "DataAnalysisCapabilityProvider",
        ):
            assert name in victor_dataanalysis.__all__

    def test_enhanced_exports_declared(self):
        for name in (
            "DataAnalysisSafetyRules",
            "EnhancedDataAnalysisSafetyExtension",
            "DataAnalysisContext",
            "EnhancedDataAnalysisConversationManager",
        ):
            assert name in victor_dataanalysis.__all__

    def test_assistant_export_is_the_vertical_class(self):
        from victor_dataanalysis.assistant import DataAnalysisAssistant

        assert victor_dataanalysis.DataAnalysisAssistant is DataAnalysisAssistant


class TestVerticalPackageMetadata:
    @pytest.fixture
    def metadata(self) -> dict:
        toml_path = _PACKAGE_DIR / "victor-vertical.toml"
        return tomllib.loads(toml_path.read_text(encoding="utf-8"))

    def test_toml_ships_with_package(self):
        assert (_PACKAGE_DIR / "victor-vertical.toml").is_file()

    def test_required_fields_present(self, metadata):
        vertical = metadata["vertical"]

        assert vertical["name"] == "dataanalysis"
        assert vertical["version"]
        assert vertical["description"]
        assert vertical["authors"]
        assert vertical["license"] == "Apache-2.0"
        assert vertical["requires_victor"]

    def test_vertical_class_points_at_assistant(self, metadata):
        cls = metadata["vertical"]["class"]

        assert cls["module"] == "victor_dataanalysis.assistant"
        assert cls["class_name"] == "DataAnalysisAssistant"

    def test_declared_class_is_importable(self, metadata):
        import importlib

        cls = metadata["vertical"]["class"]
        module = importlib.import_module(cls["module"])

        assert hasattr(module, cls["class_name"])

    def test_provides_declarations_non_empty(self, metadata):
        cls = metadata["vertical"]["class"]

        assert cls["provides_tools"]
        assert cls["provides_workflows"]
        assert cls["provides_capabilities"]

    def test_declared_workflows_overlap_shipped_workflows(self, metadata):
        """The metadata's workflow claims must overlap the YAML-shipped set."""
        from victor_dataanalysis.workflows import DataAnalysisWorkflowProvider

        declared = set(metadata["vertical"]["class"]["provides_workflows"])
        shipped = set(DataAnalysisWorkflowProvider().get_workflow_names())

        assert declared & shipped, "no overlap between declared and shipped workflows"
