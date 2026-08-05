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

"""Tests for capability negotiation system."""

import pytest

from victor.framework.capability_negotiation import (
    Version,
    CompatibilityStrategy,
    CapabilityFeature,
    CapabilityDeclaration,
    NegotiationStatus,
    NegotiationResult,
    CapabilityNegotiator,
    CapabilityNegotiationProtocol,
    negotiate_capabilities,
)


class TestVersion:
    """Tests for Version class."""

    def test_parse_major_only(self):
        """Test parsing major version only."""
        version = Version.parse("1")
        assert version.major == 1
        assert version.minor == 0
        assert version.patch == 0
        assert str(version) == "1"

    def test_parse_major_minor(self):
        """Test parsing major.minor version."""
        version = Version.parse("1.2")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 0
        assert str(version) == "1.2"

    def test_parse_major_minor_patch(self):
        """Test parsing major.minor.patch version."""
        version = Version.parse("1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert str(version) == "1.2.3"

    def test_version_equality(self):
        """Test version equality."""
        v1 = Version(1, 2, 3)
        v2 = Version(1, 2, 3)
        v3 = Version(1, 2, 4)

        assert v1 == v2
        assert not (v1 == v3)

    def test_version_comparison(self):
        """Test version comparison."""
        v1 = Version(1, 2, 3)
        v2 = Version(1, 2, 4)
        v3 = Version(2, 0, 0)

        assert v1 < v2
        assert v1 <= v2
        assert v2 > v1
        assert v2 >= v1
        assert v3 > v2

    def test_version_compatible_with_strict(self):
        """Test strict compatibility strategy."""
        v1 = Version(1, 2, 3)
        v2 = Version(1, 2, 3)
        v3 = Version(1, 2, 4)

        assert v1.is_compatible_with(v2, CompatibilityStrategy.STRICT)
        assert not v1.is_compatible_with(v3, CompatibilityStrategy.STRICT)

    def test_version_compatible_with_backward_compatible(self):
        """Test backward compatible strategy."""
        v1 = Version(1, 2, 0)
        v2 = Version(1, 0, 0)
        v3 = Version(1, 3, 0)
        v4 = Version(2, 0, 0)

        assert v1.is_compatible_with(v2, CompatibilityStrategy.BACKWARD_COMPATIBLE)
        assert v3.is_compatible_with(v1, CompatibilityStrategy.BACKWARD_COMPATIBLE)
        assert not v4.is_compatible_with(v1, CompatibilityStrategy.BACKWARD_COMPATIBLE)

    def test_version_compatible_with_minimum_version(self):
        """Test minimum version strategy."""
        v1 = Version(1, 5, 0)
        v2 = Version(1, 3, 0)
        v3 = Version(1, 6, 0)

        assert v1.is_compatible_with(v2, CompatibilityStrategy.MINIMUM_VERSION)
        assert not v1.is_compatible_with(v3, CompatibilityStrategy.MINIMUM_VERSION)


class TestCapabilityDeclaration:
    """Tests for CapabilityDeclaration class."""

    def test_create_capability(self):
        """Test creating a capability declaration."""
        cap = CapabilityDeclaration(
            name="tools",
            version=Version(1, 0, 0),
        )

        assert cap.name == "tools"
        assert cap.version == Version(1, 0, 0)
        assert cap.min_version == Version(1, 0, 0)

    def test_capability_with_features(self):
        """Test capability with features."""
        features = [
            CapabilityFeature("tool_list", required=True),
            CapabilityFeature("tool_filtering"),
        ]

        cap = CapabilityDeclaration(
            name="tools",
            version=Version(1, 0, 0),
            features=features,
        )

        assert len(cap.features) == 2
        assert cap.get_feature("tool_list") is not None
        assert cap.get_feature("tool_list").required is True
        assert cap.has_feature("tool_filtering") is True
        assert cap.has_feature("non_existent") is False

    def test_get_required_features(self):
        """Test getting required features."""
        features = [
            CapabilityFeature("required1", required=True),
            CapabilityFeature("optional1", required=False),
            CapabilityFeature("required2", required=True),
        ]

        cap = CapabilityDeclaration(
            name="test",
            version=Version(1, 0, 0),
            features=features,
        )

        required = cap.get_required_features()
        optional = cap.get_optional_features()

        assert len(required) == 2
        assert len(optional) == 1
        assert all(f.required for f in required)
        assert all(not f.required for f in optional)

    def test_capability_to_dict(self):
        """Test converting capability to dictionary."""
        cap = CapabilityDeclaration(
            name="tools",
            version=Version(1, 2, 3),
            features=[
                CapabilityFeature("tool_list", required=True),
            ],
        )

        data = cap.to_dict()

        assert data["name"] == "tools"
        assert data["version"] == "1.2.3"
        assert len(data["features"]) == 1
        assert data["features"][0]["name"] == "tool_list"
        assert data["features"][0]["required"] is True


class TestCapabilityNegotiator:
    """Tests for CapabilityNegotiator class."""

    def test_negotiate_compatible_versions(self):
        """Test negotiating compatible versions."""
        negotiator = CapabilityNegotiator()

        vertical_caps = {
            "tools": CapabilityDeclaration(
                name="tools",
                version=Version(1, 5, 0),
            )
        }

        orchestrator_caps = {
            "tools": CapabilityDeclaration(
                name="tools",
                version=Version(1, 0, 0),
                min_version=Version(1, 0, 0),
            )
        }

        results = negotiator.negotiate(vertical_caps, orchestrator_caps)

        assert "tools" in results
        result = results["tools"]
        assert result.is_success
        assert result.agreed_version == Version(1, 5, 0)
        assert result.status == NegotiationStatus.SUCCESS

    def test_negotiate_incompatible_versions(self):
        """Test negotiating incompatible versions."""
        negotiator = CapabilityNegotiator(strategy=CompatibilityStrategy.STRICT)

        vertical_caps = {
            "tools": CapabilityDeclaration(
                name="tools",
                version=Version(1, 0, 0),
                strategy=CompatibilityStrategy.STRICT,
            )
        }

        orchestrator_caps = {
            "tools": CapabilityDeclaration(
                name="tools",
                version=Version(2, 0, 0),
                strategy=CompatibilityStrategy.STRICT,
            )
        }

        results = negotiator.negotiate(vertical_caps, orchestrator_caps)

        assert "tools" in results
        result = results["tools"]
        assert not result.is_success
        assert result.status == NegotiationStatus.FAILURE

    def test_negotiate_with_fallback(self):
        """Test negotiating with fallback enabled."""
        negotiator = CapabilityNegotiator(enable_fallback=True)

        vertical_caps = {
            "tools": CapabilityDeclaration(
                name="tools",
                version=Version(2, 0, 0),
            )
        }

        orchestrator_caps = {
            "tools": CapabilityDeclaration(
                name="tools",
                version=Version(1, 0, 0),
                min_version=Version(1, 0, 0),
            )
        }

        results = negotiator.negotiate(vertical_caps, orchestrator_caps)

        result = results["tools"]
        # Should fallback to version 1.0.0 since they're different major versions
        assert result.is_success or result.has_fallback

    def test_negotiate_features(self):
        """Test negotiating with features."""
        negotiator = CapabilityNegotiator()

        vertical_caps = {
            "tools": CapabilityDeclaration(
                name="tools",
                version=Version(1, 0, 0),
                features=[
                    CapabilityFeature("tool_list", required=True),
                    CapabilityFeature("tool_filtering"),
                    CapabilityFeature("advanced_feature"),
                ],
            )
        }

        orchestrator_caps = {
            "tools": CapabilityDeclaration(
                name="tools",
                version=Version(1, 0, 0),
                features=[
                    CapabilityFeature("tool_list", required=True),
                    CapabilityFeature("tool_filtering"),
                ],
            )
        }

        results = negotiator.negotiate(vertical_caps, orchestrator_caps)

        result = results["tools"]
        assert result.is_success
        assert "tool_list" in result.supported_features
        assert "tool_filtering" in result.supported_features
        assert "advanced_feature" in result.unsupported_features

    def test_negotiate_missing_required_features(self):
        """Test negotiating with missing required features."""
        negotiator = CapabilityNegotiator(enable_fallback=False)

        vertical_caps = {
            "tools": CapabilityDeclaration(
                name="tools",
                version=Version(1, 0, 0),
                features=[
                    CapabilityFeature("tool_list", required=True),
                    CapabilityFeature("required_feature", required=True),
                ],
            )
        }

        orchestrator_caps = {
            "tools": CapabilityDeclaration(
                name="tools",
                version=Version(1, 0, 0),
                features=[
                    CapabilityFeature("tool_list", required=True),
                ],
            )
        }

        results = negotiator.negotiate(vertical_caps, orchestrator_caps)

        result = results["tools"]
        assert not result.is_success
        assert "required_feature" in result.missing_required_features

    def test_negotiate_missing_capability_on_one_side(self):
        """Test when capability is missing on one side."""
        negotiator = CapabilityNegotiator()

        vertical_caps = {
            "tools": CapabilityDeclaration(
                name="tools",
                version=Version(1, 0, 0),
            )
        }

        orchestrator_caps = {
            "tools": CapabilityDeclaration(
                name="tools",
                version=Version(1, 0, 0),
            ),
            "prompt": CapabilityDeclaration(
                name="prompt",
                version=Version(1, 0, 0),
            ),
        }

        results = negotiator.negotiate(vertical_caps, orchestrator_caps)

        # tools should succeed
        assert results["tools"].is_success

        # prompt should fail (missing on vertical side)
        assert not results["prompt"].is_success
        assert "not declared by both sides" in results["prompt"].error


class TestNegotiationResult:
    """Tests for NegotiationResult class."""

    def test_is_success_property(self):
        """Test is_success property."""
        result = NegotiationResult(
            capability_name="test",
            status=NegotiationStatus.SUCCESS,
            agreed_version=Version(1, 0, 0),
        )

        assert result.is_success is True

        result.status = NegotiationStatus.FAILURE
        assert result.is_success is False

    def test_has_fallback_property(self):
        """Test has_fallback property."""
        result = NegotiationResult(
            capability_name="test",
            status=NegotiationStatus.FALLBACK,
            agreed_version=Version(1, 0, 0),
        )

        assert result.has_fallback is True

        result.status = NegotiationStatus.SUCCESS
        assert result.has_fallback is False

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = NegotiationResult(
            capability_name="tools",
            status=NegotiationStatus.SUCCESS,
            agreed_version=Version(1, 2, 3),
            supported_features=["tool_list", "tool_filtering"],
            unsupported_features=["advanced"],
            missing_required_features=[],
            fallback_version=None,
        )

        data = result.to_dict()

        assert data["capability_name"] == "tools"
        assert data["status"] == "success"
        assert data["agreed_version"] == "1.2.3"
        assert "tool_list" in data["supported_features"]
        assert "advanced" in data["unsupported_features"]


class TestCapabilityNegotiationProtocol:
    """Tests for CapabilityNegotiationProtocol class."""

    def test_extract_vertical_capabilities(self):
        """Test extracting capabilities from vertical."""
        from victor_contracts.verticals.protocols.base import VerticalBase

        class TestVertical(VerticalBase):
            @classmethod
            def get_name(cls) -> str:
                return "test"

            @classmethod
            def get_description(cls) -> str:
                return "Test vertical"

            @classmethod
            def get_tools(cls) -> list[str]:
                return ["read", "write"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "You are a test assistant"

        caps = CapabilityNegotiationProtocol._get_vertical_capabilities(TestVertical)

        assert "tools" in caps
        assert "prompt" in caps
        assert caps["tools"].version == Version(1, 0, 0)

    def test_extract_orchestrator_capabilities(self):
        """Test extracting capabilities from orchestrator."""

        # Mock orchestrator
        class MockOrchestrator:
            def __init__(self):
                self._version = "1.5.0"

            def get_enabled_tools(self):
                return ["read", "write"]

            def get_system_prompt(self):
                return "Test prompt"

        orchestrator = MockOrchestrator()
        caps = CapabilityNegotiationProtocol._get_orchestrator_capabilities(orchestrator)

        assert "tools" in caps
        assert "prompt" in caps
        assert caps["tools"].version == Version(1, 5, 0)


class TestPublicAPI:
    """Tests for public API functions."""

    def test_negotiate_capabilities(self):
        """Test negotiate_capabilities public API."""
        from victor_contracts.verticals.protocols.base import VerticalBase

        class TestVertical(VerticalBase):
            version = "1.5.0"

            @classmethod
            def get_name(cls) -> str:
                return "test"

            @classmethod
            def get_description(cls) -> str:
                return "Test vertical"

            @classmethod
            def get_tools(cls) -> list[str]:
                return ["read", "write"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Test prompt"

        class MockOrchestrator:
            _version = "1.0.0"

            def get_enabled_tools(self):
                return ["read", "write"]

            def get_system_prompt(self):
                return "Test prompt"

        results = negotiate_capabilities(TestVertical, MockOrchestrator())

        assert "tools" in results
        assert "prompt" in results

        # Should succeed with backward compatible strategy
        assert results["tools"].is_success
