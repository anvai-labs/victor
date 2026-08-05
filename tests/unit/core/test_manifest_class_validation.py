# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0

"""Tests for manifest class reference validation.

CapabilityNegotiator should detect stale module/class references
in extension manifests at load time.
"""

from __future__ import annotations

from victor_contracts.verticals.manifest import ExtensionManifest


class TestManifestClassValidation:
    """Validate manifest module/class references at load time."""

    def test_valid_manifest_class_resolves(self):
        """Manifest with real module+class passes validation."""
        from victor.core.verticals.capability_negotiator import CapabilityNegotiator

        negotiator = CapabilityNegotiator()
        errors = negotiator.validate_manifest_class_references(
            module_path="victor.framework.events",
            class_name="EventType",
        )
        assert len(errors) == 0

    def test_stale_module_path_detected(self):
        """Nonexistent module path returns an error."""
        from victor.core.verticals.capability_negotiator import CapabilityNegotiator

        negotiator = CapabilityNegotiator()
        errors = negotiator.validate_manifest_class_references(
            module_path="victor.nonexistent.fake_module",
            class_name="FakeClass",
        )
        assert len(errors) == 1
        assert "module" in errors[0].lower()

    def test_stale_class_name_detected(self):
        """Module exists but class doesn't — returns an error."""
        from victor.core.verticals.capability_negotiator import CapabilityNegotiator

        negotiator = CapabilityNegotiator()
        errors = negotiator.validate_manifest_class_references(
            module_path="victor.framework.events",
            class_name="NonexistentClass12345",
        )
        assert len(errors) == 1
        assert "class" in errors[0].lower()

    def test_empty_references_pass(self):
        """Empty module/class skips validation."""
        from victor.core.verticals.capability_negotiator import CapabilityNegotiator

        negotiator = CapabilityNegotiator()
        errors = negotiator.validate_manifest_class_references(
            module_path="",
            class_name="",
        )
        assert len(errors) == 0
