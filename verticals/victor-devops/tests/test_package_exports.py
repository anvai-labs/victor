# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for lazy package exports and victor-vertical.toml metadata."""

import tomllib
from pathlib import Path

import pytest

import victor_devops

_PACKAGE_DIR = Path(victor_devops.__file__).parent


class TestLazyExports:
    def test_every_declared_export_resolves(self):
        for name in victor_devops.__all__:
            attr = getattr(victor_devops, name)
            assert attr is not None, f"{name} resolved to None"

    def test_exports_map_covers_all(self):
        assert set(victor_devops.__all__) == set(victor_devops._EXPORTS)

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            victor_devops.NotARealExport

    def test_assistant_export_is_the_vertical_class(self):
        from victor_devops.assistant import DevOpsAssistant

        assert victor_devops.DevOpsAssistant is DevOpsAssistant


class TestVerticalPackageMetadata:
    @pytest.fixture
    def metadata(self) -> dict:
        toml_path = _PACKAGE_DIR / "victor-vertical.toml"
        return tomllib.loads(toml_path.read_text(encoding="utf-8"))

    def test_toml_ships_with_package(self):
        assert (_PACKAGE_DIR / "victor-vertical.toml").is_file()

    def test_required_fields_present(self, metadata):
        vertical = metadata["vertical"]

        assert vertical["name"] == "devops"
        assert vertical["version"]
        assert vertical["description"]
        assert vertical["authors"]
        assert vertical["license"] == "Apache-2.0"
        assert vertical["requires_victor"]

    def test_vertical_class_points_at_assistant(self, metadata):
        cls = metadata["vertical"]["class"]

        assert cls["module"] == "victor_devops.assistant"
        assert cls["class_name"] == "DevOpsAssistant"

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
