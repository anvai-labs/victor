# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for validate_manifest() — written BEFORE implementation (TDD RED)."""

from __future__ import annotations


from victor_contracts.core.api_version import (
    CURRENT_API_VERSION,
    MIN_SUPPORTED_API_VERSION,
)
from victor_contracts.verticals.manifest import ExtensionManifest
from victor_contracts.verticals.protocols.base import VerticalBase
from victor_contracts.verticals.registration import register_vertical
from victor_contracts.verticals.validation import validate_manifest


class TestValidateManifestStructural:
    """Structural validity checks."""

    def test_valid_manifest_passes(self):
        @register_vertical(name="test-valid", version="1.0.0")
        class GoodVertical(VerticalBase):
            @classmethod
            def get_name(cls) -> str:
                return "test-valid"

            @classmethod
            def get_description(cls) -> str:
                return "A valid test vertical"

            @classmethod
            def get_tools(cls):
                return ["tool_a"]

        report = validate_manifest(GoodVertical)
        assert report.ok, f"Expected ok, got: {report.to_text()}"

    def test_missing_manifest_reports_error(self):
        # A plain class (not inheriting VerticalBase) has no manifest at all
        class NotAVertical:
            pass

        report = validate_manifest(NotAVertical)
        assert any(i.code == "missing_manifest" for i in report.issues)

    def test_empty_name_reports_error(self):
        class EmptyNameVertical(VerticalBase):
            _victor_manifest = ExtensionManifest(name="", version="1.0.0")

            @classmethod
            def get_name(cls) -> str:
                return ""

            @classmethod
            def get_description(cls) -> str:
                return "Empty name"

            @classmethod
            def get_tools(cls):
                return []

        report = validate_manifest(EmptyNameVertical)
        assert any(i.code == "empty_manifest_name" for i in report.issues)

    def test_api_version_above_current_reports_warning(self):
        class FutureVertical(VerticalBase):
            _victor_manifest = ExtensionManifest(
                name="future",
                version="1.0.0",
                api_version=CURRENT_API_VERSION + 1,
            )

            @classmethod
            def get_name(cls) -> str:
                return "future"

            @classmethod
            def get_description(cls) -> str:
                return "Future API"

            @classmethod
            def get_tools(cls):
                return []

        report = validate_manifest(FutureVertical)
        warnings = [i for i in report.issues if i.level == "warning"]
        assert any("api_version" in i.code for i in warnings)

    def test_api_version_below_minimum_reports_error(self):
        class OldVertical(VerticalBase):
            _victor_manifest = ExtensionManifest(
                name="old",
                version="1.0.0",
                api_version=MIN_SUPPORTED_API_VERSION - 1,
            )

            @classmethod
            def get_name(cls) -> str:
                return "old"

            @classmethod
            def get_description(cls) -> str:
                return "Old API"

            @classmethod
            def get_tools(cls):
                return []

        report = validate_manifest(OldVertical)
        errors = [i for i in report.issues if i.level == "error"]
        assert any("api_version" in i.code for i in errors)
