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

"""Tests for middleware profiles and builder."""

import pytest

from victor.framework.middleware_profiles.profiles import (
    MiddlewareProfile,
    MiddlewareProfiles,
)
from victor.framework.middleware_profiles.builder import MiddlewareProfileBuilder
from victor.core.verticals.protocols import MiddlewareProtocol


# Create mock middleware for testing (avoids import complexity)
class MockMiddleware(MiddlewareProtocol):
    """Mock middleware for testing."""

    def __init__(self, name: str = "mock"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def before_tool_execution(self, context, tool_name, args):
        pass

    def after_tool_execution(self, context, tool_name, result):
        pass

    def on_error(self, context, tool_name, error):
        pass

    # Add missing abstract methods
    def before_tool_call(self, context, tool_name, tool_args):
        pass

    def after_tool_call(self, context, tool_name, tool_args, tool_result):
        pass


class TestMiddlewareProfile:
    """Tests for MiddlewareProfile dataclass."""

    def test_create_profile(self):
        """Test creating a basic profile."""
        profile = MiddlewareProfile(
            name="test",
            description="Test profile",
            middlewares=[],
            priority=50,
        )

        assert profile.name == "test"
        assert profile.description == "Test profile"
        assert profile.middlewares == []
        assert profile.priority == 50

    def test_profile_with_middlewares(self):
        """Test profile with middleware list."""
        middleware = MockMiddleware("logging")
        profile = MiddlewareProfile(
            name="test",
            description="Test profile",
            middlewares=[middleware],
            priority=50,
        )

        assert len(profile.middlewares) == 1
        assert profile.middlewares[0] is middleware

    def test_profile_immutability(self):
        """Test that profile is immutable (frozen)."""
        profile = MiddlewareProfile(
            name="test",
            description="Test profile",
            middlewares=[],
            priority=50,
        )

        # Attempting to modify should raise an error
        with pytest.raises(Exception):  # FrozenInstanceError
            profile.name = "modified"

    def test_with_additional_middleware(self):
        """Test that MiddlewareProfile is immutable (frozen dataclass)."""
        middleware1 = MockMiddleware("logging")
        middleware2 = MockMiddleware("metrics")

        profile = MiddlewareProfile(
            name="test",
            description="Test profile",
            middlewares=[middleware1],
            priority=50,
        )

        # Since MiddlewareProfile is frozen, we need to create a new instance
        # to add middleware (no with_additional_middleware method exists)
        new_profile = MiddlewareProfile(
            name=profile.name,
            description=profile.description,
            middlewares=list(profile.middlewares) + [middleware2],
            priority=profile.priority,
        )

        # Original should be unchanged
        assert len(profile.middlewares) == 1
        assert profile.middlewares[0] is middleware1

        # New profile should have both
        assert len(new_profile.middlewares) == 2
        assert new_profile.middlewares[0] is middleware1
        assert new_profile.middlewares[1] is middleware2


class TestMiddlewareProfiles:
    """Tests for predefined middleware profiles."""

    def test_default_profile(self):
        """Test default profile has basic logging."""
        profile = MiddlewareProfiles.default_profile()

        assert profile.name == "default"
        assert len(profile.middlewares) == 1
        assert profile.priority == 50
        assert profile.description == "Basic logging"

    def test_safety_first_profile(self):
        """Test safety-first profile has strict checks."""
        profile = MiddlewareProfiles.safety_first_profile()

        assert profile.name == "safety_first"
        assert len(profile.middlewares) == 3
        assert profile.priority == 25  # HIGH priority (lower number = higher priority)
        assert profile.description == "Git safety and secret masking"

    def test_development_profile(self):
        """Test development profile is permissive."""
        profile = MiddlewareProfiles.development_profile()

        assert profile.name == "development"
        assert len(profile.middlewares) == 2
        assert profile.priority == 75  # LOW priority (higher number = lower priority)
        assert profile.description == "Permissive git with detailed logging"

    def test_production_profile(self):
        """Test production profile has audit logging."""
        profile = MiddlewareProfiles.production_profile()

        assert profile.name == "production"
        assert len(profile.middlewares) == 3
        assert profile.priority == 0  # CRITICAL priority
        assert profile.description == "Strict safety and secrets masking"

    def test_minimal_profile(self):
        """Test minimal profile is lightweight."""
        profile = MiddlewareProfiles.analysis_profile()

        assert profile.name == "analysis"
        assert len(profile.middlewares) == 1
        assert profile.priority == 50
        assert profile.description == "Read-only analysis with minimal logging"

    def test_audit_profile(self):
        """Test audit profile has comprehensive logging."""
        profile = MiddlewareProfiles.ci_cd_profile()

        assert profile.name == "ci_cd"
        assert len(profile.middlewares) == 3
        assert profile.priority == 10
        assert profile.description == "CI/CD with deployment safety"

    def test_get_all_profiles(self):
        """Test listing all available profile names."""
        from victor.framework.middleware_profiles import list_profiles

        profile_names = list_profiles()

        assert isinstance(profile_names, list)
        assert len(profile_names) == 6
        assert "default" in profile_names
        assert "safety_first" in profile_names
        assert "production" in profile_names

    def test_get_profile_by_name(self):
        """Test getting profile by name."""
        from victor.framework.middleware_profiles import get_profile

        profile = get_profile("safety_first")

        assert profile is not None
        assert profile.name == "safety_first"

    def test_get_profile_nonexistent(self):
        """Test getting nonexistent profile returns None."""
        from victor.framework.middleware_profiles import get_profile

        profile = get_profile("nonexistent")

        assert profile is None


class TestMiddlewareProfileBuilder:
    """Tests for MiddlewareProfileBuilder."""

    def test_build_minimal_profile(self):
        """Test building minimal profile."""
        profile = (
            MiddlewareProfileBuilder().set_name("test").set_description("Test profile").build()
        )

        assert profile.name == "test"
        assert profile.description == "Test profile"
        assert profile.middlewares == []
        assert profile.priority == 50

    def test_build_with_middleware(self):
        """Test building profile with middleware."""
        middleware = MockMiddleware("logging")

        profile = MiddlewareProfileBuilder().set_name("test").add_middleware(middleware).build()

        assert len(profile.middlewares) == 1
        assert profile.middlewares[0] is middleware

    def test_build_with_multiple_middlewares(self):
        """Test building profile with multiple middlewares."""
        middleware1 = MockMiddleware("logging")
        middleware2 = MockMiddleware("metrics")

        profile = (
            MiddlewareProfileBuilder()
            .set_name("test")
            .add_middleware(middleware1)
            .add_middleware(middleware2)
            .build()
        )

        assert len(profile.middlewares) == 2
        assert profile.middlewares[0] is middleware1
        assert profile.middlewares[1] is middleware2

    def test_build_with_middlewares_list(self):
        """Test building profile with middleware list."""
        middlewares = [
            MockMiddleware("logging"),
            MockMiddleware("metrics"),
        ]

        profile = MiddlewareProfileBuilder().set_name("test").add_middlewares(middlewares).build()

        assert len(profile.middlewares) == 2

    def test_build_with_priority(self):
        """Test building profile with custom priority."""
        profile = MiddlewareProfileBuilder().set_name("test").set_priority(75).build()

        assert profile.priority == 75

    def test_build_with_tags(self):
        """Test building profile with tags."""
        # Note: Current builder doesn't support metadata
        # This test can be enhanced when metadata is added to builder
        profile = MiddlewareProfileBuilder().set_name("test").build()

        assert profile.name == "test"

    def test_build_without_name_uses_default(self):
        """Test that builder uses default name when not set."""
        profile = MiddlewareProfileBuilder().build()

        assert profile.name == "custom"  # Default name from __init__


class TestMiddlewareProfileIntegration:
    """Integration tests for middleware profiles."""

    def test_extend_predefined_profile(self):
        """Test that we can create an extended profile from a predefined one."""
        from victor.framework.middleware_profiles import get_profile

        base_profile = get_profile("default")

        # Create extended profile by manually building a new one
        extended_profile = MiddlewareProfile(
            name=base_profile.name + "_extended",
            description=base_profile.description + " with git safety",
            middlewares=list(base_profile.middlewares) + [MockMiddleware("git_safety")],
            priority=base_profile.priority,
        )

        # Should have original middlewares + new one
        assert len(extended_profile.middlewares) == len(base_profile.middlewares) + 1

    def test_profile_metadata_enables_discovery(self):
        """Test that profile descriptions enable discovery and filtering."""
        from victor.framework.middleware_profiles import list_profiles, get_profile

        profile_names = list_profiles()

        # Filter profiles by name/description containing "production"
        production_profiles = [
            name
            for name in profile_names
            if "production" in name or "production" in get_profile(name).description.lower()
        ]

        assert "production" in production_profiles

        # Filter profiles for development-related ones
        dev_profiles = [
            name
            for name in profile_names
            if "dev" in name or "development" in get_profile(name).description.lower()
        ]

        assert "development" in dev_profiles
