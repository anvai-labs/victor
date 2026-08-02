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

"""Tests for victor.framework.middleware_profiles.builder module.

These tests verify the MiddlewareProfileBuilder that provides a fluent
interface for creating custom middleware profiles.
"""

import logging
import pytest

from victor.framework.middleware_profiles.builder import (
    MiddlewareProfileBuilder,
    create_profile,
)
from victor.framework.middleware_profiles.profiles import MiddlewareProfile
from victor.framework.middleware import (
    GitSafetyMiddleware,
    SecretMaskingMiddleware,
    LoggingMiddleware,
)

# =============================================================================
# MiddlewareProfileBuilder Basic Tests
# =============================================================================


class TestMiddlewareProfileBuilder:
    """Tests for MiddlewareProfileBuilder basic functionality."""

    def test_builder_initialization(self):
        """Builder should initialize with default values."""
        builder = MiddlewareProfileBuilder()

        assert builder._name == "custom"
        assert builder._description == "Custom middleware profile"
        assert builder._middlewares == []
        assert builder._priority == 50

    def test_builder_builds_empty_profile(self):
        """Builder should build profile with no middlewares."""
        builder = MiddlewareProfileBuilder()
        profile = builder.build()

        assert isinstance(profile, MiddlewareProfile)
        assert profile.name == "custom"
        assert profile.description == "Custom middleware profile"
        assert profile.middlewares == []
        assert profile.priority == 50


# =============================================================================
# Builder Method Chaining Tests
# =============================================================================


class TestBuilderMethodChaining:
    """Tests for builder fluent interface."""

    def test_set_name_returns_builder(self):
        """set_name() should return builder for chaining."""
        builder = MiddlewareProfileBuilder()
        result = builder.set_name("test_profile")

        assert result is builder
        assert builder._name == "test_profile"

    def test_set_description_returns_builder(self):
        """set_description() should return builder for chaining."""
        builder = MiddlewareProfileBuilder()
        result = builder.set_description("Test description")

        assert result is builder
        assert builder._description == "Test description"

    def test_add_middleware_returns_builder(self):
        """add_middleware() should return builder for chaining."""
        builder = MiddlewareProfileBuilder()
        middleware = LoggingMiddleware()

        result = builder.add_middleware(middleware)

        assert result is builder
        assert len(builder._middlewares) == 1

    def test_add_middlewares_returns_builder(self):
        """add_middlewares() should return builder for chaining."""
        builder = MiddlewareProfileBuilder()
        middlewares = [
            LoggingMiddleware(),
            SecretMaskingMiddleware(),
        ]

        result = builder.add_middlewares(middlewares)

        assert result is builder
        assert len(builder._middlewares) == 2

    def test_set_priority_returns_builder(self):
        """set_priority() should return builder for chaining."""
        builder = MiddlewareProfileBuilder()
        result = builder.set_priority(25)

        assert result is builder
        assert builder._priority == 25

    def test_remove_middleware_returns_builder(self):
        """remove_middleware() should return builder for chaining."""
        builder = MiddlewareProfileBuilder()
        builder.add_middleware(LoggingMiddleware())
        builder.add_middleware(SecretMaskingMiddleware())

        result = builder.remove_middleware(SecretMaskingMiddleware)

        assert result is builder
        assert len(builder._middlewares) == 1

    def test_clear_middlewares_returns_builder(self):
        """clear_middlewares() should return builder for chaining."""
        builder = MiddlewareProfileBuilder()
        builder.add_middleware(LoggingMiddleware())

        result = builder.clear_middlewares()

        assert result is builder
        assert len(builder._middlewares) == 0


# =============================================================================
# Builder Fluent Interface Tests
# =============================================================================


class TestBuilderFluentInterface:
    """Tests for complete fluent interface workflows."""

    def test_complete_builder_chain(self):
        """Builder should support complete fluent chain."""
        middleware1 = LoggingMiddleware(log_level=logging.DEBUG)
        middleware2 = SecretMaskingMiddleware()

        profile = (
            MiddlewareProfileBuilder()
            .set_name("custom_profile")
            .set_description("My custom profile")
            .add_middleware(middleware1)
            .add_middleware(middleware2)
            .set_priority(10)
            .build()
        )

        assert profile.name == "custom_profile"
        assert profile.description == "My custom profile"
        assert len(profile.middlewares) == 2
        assert profile.priority == 10

    def test_multiple_add_middleware_calls(self):
        """Builder should support multiple add_middleware() calls."""
        profile = (
            MiddlewareProfileBuilder()
            .add_middleware(LoggingMiddleware())
            .add_middleware(SecretMaskingMiddleware())
            .add_middleware(GitSafetyMiddleware(block_dangerous=True))
            .build()
        )

        assert len(profile.middlewares) == 3
        assert isinstance(profile.middlewares[0], LoggingMiddleware)
        assert isinstance(profile.middlewares[1], SecretMaskingMiddleware)
        assert isinstance(profile.middlewares[2], GitSafetyMiddleware)

    def test_add_middlewares_with_multiple(self):
        """Builder should support add_middlewares() with list."""
        middlewares = [
            LoggingMiddleware(),
            SecretMaskingMiddleware(),
            GitSafetyMiddleware(block_dangerous=False),
        ]

        profile = MiddlewareProfileBuilder().add_middlewares(middlewares).build()

        assert len(profile.middlewares) == 3

    def test_mixed_add_middleware_methods(self):
        """Builder should support mixed add_middleware methods."""
        profile = (
            MiddlewareProfileBuilder()
            .add_middleware(LoggingMiddleware())
            .add_middlewares(
                [
                    SecretMaskingMiddleware(),
                    GitSafetyMiddleware(),
                ]
            )
            .add_middleware(LoggingMiddleware(log_level=logging.INFO))
            .build()
        )

        assert len(profile.middlewares) == 4


# =============================================================================
# Builder Configuration Tests
# =============================================================================


class TestBuilderConfiguration:
    """Tests for builder configuration methods."""

    def test_set_name_updates_name(self):
        """set_name() should update profile name."""
        profile = MiddlewareProfileBuilder().set_name("test_name").build()

        assert profile.name == "test_name"

    def test_set_name_multiple_calls(self):
        """Multiple set_name() calls should use last value."""
        profile = MiddlewareProfileBuilder().set_name("first").set_name("second").build()

        assert profile.name == "second"

    def test_set_description_updates_description(self):
        """set_description() should update profile description."""
        profile = MiddlewareProfileBuilder().set_description("Test description").build()

        assert profile.description == "Test description"

    def test_set_priority_updates_priority(self):
        """set_priority() should update profile priority."""
        profile = MiddlewareProfileBuilder().set_priority(100).build()

        assert profile.priority == 100

    def test_set_priority_zero(self):
        """set_priority() should accept 0 (highest priority)."""
        profile = MiddlewareProfileBuilder().set_priority(0).build()

        assert profile.priority == 0

    def test_set_priority_negative(self):
        """set_priority() should accept negative values."""
        profile = MiddlewareProfileBuilder().set_priority(-10).build()

        assert profile.priority == -10


# =============================================================================
# Builder Middleware Management Tests
# =============================================================================


class TestBuilderMiddlewareManagement:
    """Tests for builder middleware management."""

    def test_add_middleware_appends(self):
        """add_middleware() should append to middlewares list."""
        profile = (
            MiddlewareProfileBuilder()
            .add_middleware(LoggingMiddleware(log_level=logging.DEBUG))
            .add_middleware(LoggingMiddleware(log_level=logging.INFO))
            .build()
        )

        assert len(profile.middlewares) == 2
        assert profile.middlewares[0]._log_level == logging.DEBUG
        assert profile.middlewares[1]._log_level == logging.INFO

    def test_add_middlewares_extends(self):
        """add_middlewares() should extend middlewares list."""
        profile = (
            MiddlewareProfileBuilder()
            .add_middleware(LoggingMiddleware())
            .add_middlewares(
                [
                    SecretMaskingMiddleware(),
                    GitSafetyMiddleware(),
                ]
            )
            .build()
        )

        assert len(profile.middlewares) == 3

    def test_remove_middleware_by_type(self):
        """remove_middleware() should remove middleware by type."""
        profile = (
            MiddlewareProfileBuilder()
            .add_middleware(LoggingMiddleware())
            .add_middleware(SecretMaskingMiddleware())
            .add_middleware(GitSafetyMiddleware())
            .remove_middleware(SecretMaskingMiddleware)
            .build()
        )

        assert len(profile.middlewares) == 2
        assert not any(isinstance(m, SecretMaskingMiddleware) for m in profile.middlewares)

    def test_remove_middleware_removes_all_of_type(self):
        """remove_middleware() should remove all middlewares of type."""
        profile = (
            MiddlewareProfileBuilder()
            .add_middleware(LoggingMiddleware(log_level=logging.DEBUG))
            .add_middleware(LoggingMiddleware(log_level=logging.INFO))
            .add_middleware(SecretMaskingMiddleware())
            .remove_middleware(LoggingMiddleware)
            .build()
        )

        assert len(profile.middlewares) == 1
        assert isinstance(profile.middlewares[0], SecretMaskingMiddleware)

    def test_remove_middleware_nonexistent_type(self):
        """remove_middleware() should handle nonexistent type gracefully."""
        profile = (
            MiddlewareProfileBuilder()
            .add_middleware(LoggingMiddleware())
            .remove_middleware(SecretMaskingMiddleware)
            .build()
        )

        assert len(profile.middlewares) == 1

    def test_clear_middlewares_removes_all(self):
        """clear_middlewares() should remove all middlewares."""
        profile = (
            MiddlewareProfileBuilder()
            .add_middleware(LoggingMiddleware())
            .add_middleware(SecretMaskingMiddleware())
            .add_middleware(GitSafetyMiddleware())
            .clear_middlewares()
            .build()
        )

        assert len(profile.middlewares) == 0

    def test_clear_middlewares_empty_list(self):
        """clear_middlewares() should handle empty list."""
        profile = MiddlewareProfileBuilder().clear_middlewares().build()

        assert len(profile.middlewares) == 0


# =============================================================================
# Builder Build Tests
# =============================================================================


class TestBuilderBuild:
    """Tests for builder build() method."""

    def test_build_creates_new_profile_each_time(self):
        """build() should create new profile on each call."""
        builder = MiddlewareProfileBuilder()
        builder.add_middleware(LoggingMiddleware())

        profile1 = builder.build()
        profile2 = builder.build()

        assert profile1 is not profile2
        assert profile1.name == profile2.name
        assert len(profile1.middlewares) == len(profile2.middlewares)

    def test_build_does_not_modify_builder_state(self):
        """build() should not modify builder state."""
        builder = MiddlewareProfileBuilder()
        builder.add_middleware(LoggingMiddleware())

        profile1 = builder.build()
        builder.add_middleware(SecretMaskingMiddleware())
        profile2 = builder.build()

        assert len(profile1.middlewares) == 1
        assert len(profile2.middlewares) == 2
        assert len(builder._middlewares) == 2

    def test_build_copies_middlewares(self):
        """build() should copy middlewares list."""
        builder = MiddlewareProfileBuilder()
        middleware = LoggingMiddleware()
        builder.add_middleware(middleware)

        profile = builder.build()

        # Profile middlewares should be a copy
        assert profile.middlewares is not builder._middlewares
        assert len(profile.middlewares) == len(builder._middlewares)


# =============================================================================
# Builder Edge Cases Tests
# =============================================================================


class TestBuilderEdgeCases:
    """Tests for builder edge cases."""

    def test_build_with_no_middlewares(self):
        """Builder should build profile with no middlewares."""
        profile = MiddlewareProfileBuilder().set_name("empty_profile").build()

        assert len(profile.middlewares) == 0
        assert profile.name == "empty_profile"

    def test_build_with_empty_string_name(self):
        """Builder should accept empty string name."""
        profile = MiddlewareProfileBuilder().set_name("").build()

        assert profile.name == ""

    def test_build_with_empty_string_description(self):
        """Builder should accept empty string description."""
        profile = MiddlewareProfileBuilder().set_description("").build()

        assert profile.description == ""

    def test_multiple_builds_with_different_config(self):
        """Builder should support multiple builds with different configs."""
        builder = MiddlewareProfileBuilder()

        profile1 = builder.set_name("profile1").build()
        profile2 = builder.set_name("profile2").build()

        assert profile1.name == "profile1"
        assert profile2.name == "profile2"

    def test_builder_reuse_after_build(self):
        """Builder should be reusable after build()."""
        builder = MiddlewareProfileBuilder()

        profile1 = builder.add_middleware(LoggingMiddleware()).build()
        builder.clear_middlewares()

        profile2 = builder.add_middleware(SecretMaskingMiddleware()).build()

        assert len(profile1.middlewares) == 1
        assert len(profile2.middlewares) == 1
        assert isinstance(profile1.middlewares[0], LoggingMiddleware)
        assert isinstance(profile2.middlewares[0], SecretMaskingMiddleware)


# =============================================================================
# create_profile() Function Tests
# =============================================================================


class TestCreateProfileFunction:
    """Tests for create_profile() convenience function."""

    def test_create_profile_basic(self):
        """create_profile() should create profile with basic params."""
        profile = create_profile(
            name="test",
            description="Test profile",
        )

        assert profile.name == "test"
        assert profile.description == "Test profile"
        assert profile.middlewares == []
        assert profile.priority == 50

    def test_create_profile_with_middlewares(self):
        """create_profile() should create profile with middlewares."""
        middlewares = [
            LoggingMiddleware(),
            SecretMaskingMiddleware(),
        ]

        profile = create_profile(
            name="test",
            middlewares=middlewares,
        )

        assert len(profile.middlewares) == 2
        assert isinstance(profile.middlewares[0], LoggingMiddleware)
        assert isinstance(profile.middlewares[1], SecretMaskingMiddleware)

    def test_create_profile_with_priority(self):
        """create_profile() should create profile with priority."""
        profile = create_profile(
            name="test",
            priority=25,
        )

        assert profile.priority == 25

    def test_create_profile_all_parameters(self):
        """create_profile() should create profile with all parameters."""
        middlewares = [LoggingMiddleware()]

        profile = create_profile(
            name="custom",
            description="Custom profile",
            middlewares=middlewares,
            priority=10,
        )

        assert profile.name == "custom"
        assert profile.description == "Custom profile"
        assert len(profile.middlewares) == 1
        assert profile.priority == 10

    def test_create_profile_defaults(self):
        """create_profile() should use defaults when params omitted."""
        profile = create_profile()

        assert profile.name == "custom"
        assert profile.description == "Custom middleware profile"
        assert profile.middlewares == []
        assert profile.priority == 50

    def test_create_profile_none_middlewares(self):
        """create_profile() should handle None middlewares."""
        profile = create_profile(
            name="test",
            middlewares=None,
        )

        assert profile.middlewares == []


# =============================================================================
# Builder Integration Tests
# =============================================================================


class TestBuilderIntegration:
    """Integration tests for builder with real middleware."""

    def test_build_safety_profile_similar_to_predefined(self):
        """Builder should create profile similar to safety_first."""
        custom_profile = (
            MiddlewareProfileBuilder()
            .set_name("custom_safety")
            .set_description("Custom safety profile")
            .add_middleware(
                GitSafetyMiddleware(
                    block_dangerous=True,
                    warn_on_risky=True,
                    protected_branches={"production", "staging", "main"},
                )
            )
            .add_middleware(
                SecretMaskingMiddleware(
                    replacement="[REDACTED]",
                    mask_in_arguments=True,
                )
            )
            .add_middleware(
                LoggingMiddleware(
                    log_level=logging.DEBUG,
                    include_arguments=True,
                    sanitize_arguments=True,
                )
            )
            .set_priority(25)
            .build()
        )

        # Verify structure matches safety_first profile
        assert len(custom_profile.middlewares) == 3
        middleware_types = [type(m).__name__ for m in custom_profile.middlewares]

        assert "GitSafetyMiddleware" in middleware_types
        assert "SecretMaskingMiddleware" in middleware_types
        assert "LoggingMiddleware" in middleware_types
        assert custom_profile.priority == 25

    def test_build_custom_profile_with_mixed_config(self):
        """Builder should create profile with mixed middleware config."""
        profile = (
            MiddlewareProfileBuilder()
            .set_name("mixed")
            .add_middleware(
                LoggingMiddleware(
                    log_level=logging.INFO,
                    include_arguments=True,
                    sanitize_arguments=False,
                )
            )
            .add_middleware(
                SecretMaskingMiddleware(
                    replacement="[XXX]",
                    mask_in_arguments=False,
                )
            )
            .set_priority(40)
            .build()
        )

        assert len(profile.middlewares) == 2
        assert profile.priority == 40

        logging_mw = next(m for m in profile.middlewares if isinstance(m, LoggingMiddleware))
        assert logging_mw._log_level == logging.INFO
        assert logging_mw._sanitize_arguments is False

        secret_mw = next(m for m in profile.middlewares if isinstance(m, SecretMaskingMiddleware))
        assert secret_mw._replacement == "[XXX]"


__all__ = [
    "TestMiddlewareProfileBuilder",
    "TestBuilderMethodChaining",
    "TestBuilderFluentInterface",
    "TestBuilderConfiguration",
    "TestBuilderMiddlewareManagement",
    "TestBuilderBuild",
    "TestBuilderEdgeCases",
    "TestCreateProfileFunction",
    "TestBuilderIntegration",
]
