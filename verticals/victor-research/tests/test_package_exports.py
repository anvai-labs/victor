# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for lazy package exports and victor-vertical.toml metadata."""

import tomllib
import warnings
from pathlib import Path

import pytest

import victor_research

_PACKAGE_DIR = Path(victor_research.__file__).parent


class TestLazyExports:
    def test_every_declared_export_resolves(self):
        with warnings.catch_warnings():
            # ResearchToolDependencyProvider lazily builds a deprecated provider
            warnings.simplefilter("ignore", DeprecationWarning)
            for name in victor_research.__all__:
                attr = getattr(victor_research, name)
                assert attr is not None, f"{name} resolved to None"

    def test_exports_map_covers_all_but_tool_dependency_special_case(self):
        # ResearchToolDependencyProvider is special-cased in __getattr__
        assert set(victor_research.__all__) - set(victor_research._EXPORTS) == {
            "ResearchToolDependencyProvider"
        }

    def test_tool_dependency_export_returns_provider_instance(self):
        """The lazy export special case builds an instance via get_provider()."""
        with pytest.warns(DeprecationWarning):
            provider = victor_research.ResearchToolDependencyProvider

        assert type(provider).__name__ == "ResearchToolDependencyProvider"

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            victor_research.NotARealExport

    def test_assistant_export_is_the_vertical_class(self):
        from victor_research.assistant import ResearchAssistant

        assert victor_research.ResearchAssistant is ResearchAssistant


class TestVerticalPackageMetadata:
    @pytest.fixture
    def metadata(self) -> dict:
        toml_path = _PACKAGE_DIR / "victor-vertical.toml"
        return tomllib.loads(toml_path.read_text(encoding="utf-8"))

    def test_toml_ships_with_package(self):
        assert (_PACKAGE_DIR / "victor-vertical.toml").is_file()

    def test_required_fields_present(self, metadata):
        vertical = metadata["vertical"]

        assert vertical["name"] == "research"
        assert vertical["version"]
        assert vertical["description"]
        assert vertical["authors"]
        assert vertical["license"] == "Apache-2.0"
        assert vertical["requires_victor"]

    def test_vertical_class_points_at_assistant(self, metadata):
        cls = metadata["vertical"]["class"]

        assert cls["module"] == "victor_research.assistant"
        assert cls["class_name"] == "ResearchAssistant"

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

    def test_deep_research_workflow_declared(self, metadata):
        """The flagship workflow must be declared in package metadata."""
        cls = metadata["vertical"]["class"]

        assert "deep_research" in cls["provides_workflows"]
