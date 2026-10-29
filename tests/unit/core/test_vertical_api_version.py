"""Tests for vertical API version contract."""

import logging

from victor.core.verticals.base import VerticalBase, VerticalRegistry
from victor_contracts.core.api_version import (
    CURRENT_API_VERSION,
    MIN_SUPPORTED_API_VERSION,
)


def _make_vertical(name: str, api_version=None):
    """Create a minimal concrete vertical for testing."""
    attrs = {
        "name": name,
        "description": f"Test vertical {name}",
    }
    if api_version is not None:
        attrs["VERTICAL_API_VERSION"] = api_version

    cls = type(
        f"TestVertical_{name}",
        (VerticalBase,),
        {
            **attrs,
            "get_tools": classmethod(lambda cls: ["read"]),
            "get_system_prompt": classmethod(lambda cls: "test prompt"),
        },
    )
    return cls


class TestVerticalAPIVersion:
    """Verify API version contract for VerticalBase and VerticalRegistry."""

    def test_default_api_version_is_1(self):
        """VerticalBase should have VERTICAL_API_VERSION = 1 by default."""
        v = _make_vertical("ver_default")
        assert v.VERTICAL_API_VERSION == 1

    def test_vertical_without_version_warns(self, caplog):
        """A vertical without VERTICAL_API_VERSION should log a warning."""
        # Set VERTICAL_API_VERSION to None to simulate an old vertical
        v = _make_vertical("ver_no_version", api_version=None)
        # Override to None explicitly (inherited value is 1)
        v.VERTICAL_API_VERSION = None

        with caplog.at_level(logging.WARNING):
            result = VerticalRegistry._validate_external_vertical(v, "test_ep")

        assert result is True
        assert "no VERTICAL_API_VERSION" in caplog.text

    def test_vertical_too_old_rejected(self, caplog):
        """A vertical below MINIMUM_SUPPORTED_API_VERSION should be rejected."""
        v = _make_vertical("ver_old", api_version=0)

        original_min = VerticalRegistry.MINIMUM_SUPPORTED_API_VERSION
        try:
            VerticalRegistry.MINIMUM_SUPPORTED_API_VERSION = 1
            with caplog.at_level(logging.ERROR):
                result = VerticalRegistry._validate_external_vertical(v, "test_ep")
        finally:
            VerticalRegistry.MINIMUM_SUPPORTED_API_VERSION = original_min

        assert result is False
        assert "below minimum" in caplog.text

    def test_vertical_newer_rejected(self, caplog):
        """A vertical with version > CURRENT should be rejected."""
        v = _make_vertical("ver_newer", api_version=99)

        with caplog.at_level(logging.ERROR):
            result = VerticalRegistry._validate_external_vertical(v, "test_ep")

        assert result is False
        assert "newer than current" in caplog.text

    def test_matching_version_silent(self, caplog):
        """A vertical with matching version should produce no version warnings."""
        v = _make_vertical("ver_match", api_version=CURRENT_API_VERSION)

        with caplog.at_level(logging.DEBUG):
            result = VerticalRegistry._validate_external_vertical(v, "test_ep")

        assert result is True
        # No version-related warnings
        assert "VERTICAL_API_VERSION" not in caplog.text
        assert "below minimum" not in caplog.text
        assert "newer than current" not in caplog.text

    def test_registry_version_constants(self):
        """VerticalRegistry should expose SDK-aligned version constants."""
        assert VerticalRegistry.MINIMUM_SUPPORTED_API_VERSION == MIN_SUPPORTED_API_VERSION
        assert VerticalRegistry.CURRENT_API_VERSION == CURRENT_API_VERSION
        assert (
            VerticalRegistry.CURRENT_API_VERSION >= VerticalRegistry.MINIMUM_SUPPORTED_API_VERSION
        )
