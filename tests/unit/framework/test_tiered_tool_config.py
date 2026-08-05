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

"""Tests for tiered tool configuration canonical contract.

These tests verify that the TieredConfigStepHandler correctly handles
the canonical get_tiered_tool_config() API.

Workstream D: OpenAI Codex feedback fixes.
"""

import pytest
from typing import Dict, List, Optional, Set
from unittest.mock import MagicMock

from victor.core.vertical_types import TieredToolConfig
from victor.framework.step_handlers import (
    TieredConfigStepHandler,
    get_tiered_config,
)

# =============================================================================
# Test Fixtures
# =============================================================================


class MockVerticalWithTieredToolConfig:
    """Mock vertical that implements get_tiered_tool_config()."""

    name = "mock_with_config"

    @classmethod
    def get_tiered_tool_config(cls) -> TieredToolConfig:
        return TieredToolConfig(
            mandatory={"read", "ls"},
            vertical_core={"write", "edit"},
            semantic_pool={"grep", "shell"},
        )


class MockVerticalWithNeither:
    """Mock vertical that implements neither method."""

    name = "mock_with_neither"


class MockVerticalWithNoneReturn:
    """Mock vertical where get_tiered_tool_config returns None."""

    name = "mock_with_none"

    @classmethod
    def get_tiered_tool_config(cls) -> Optional[TieredToolConfig]:
        return None


# =============================================================================
# Tests for get_tiered_config helper function
# =============================================================================


class TestTieredToolConfigCanonicalContract:
    """Test get_tiered_config canonical behavior."""

    def test_fallback_chain_prefers_get_tiered_tool_config(self):
        """get_tiered_config should prefer get_tiered_tool_config when available."""
        config = get_tiered_config(MockVerticalWithTieredToolConfig)

        assert config is not None
        assert "read" in config.mandatory
        assert "ls" in config.mandatory
        assert "write" in config.vertical_core

    def test_fallback_chain_returns_none_when_neither(self):
        """get_tiered_config should return None when vertical has neither method."""
        config = get_tiered_config(MockVerticalWithNeither)

        assert config is None

    def test_fallback_chain_with_none_return(self):
        """get_tiered_config should return None if canonical method returns None."""
        config = get_tiered_config(MockVerticalWithNoneReturn)

        assert config is None


# =============================================================================
# Tests for TieredConfigStepHandler
# =============================================================================


class TestTieredConfigStepHandler:
    """Test TieredConfigStepHandler with canonical contract."""

    def test_handler_properties(self):
        """Handler should have correct name and order."""
        handler = TieredConfigStepHandler()

        assert handler.name == "tiered_config"
        assert handler.order == 15  # After tools (10), before prompt (20)

    def test_handler_applies_get_tiered_tool_config(self):
        """Handler should apply config from get_tiered_tool_config."""
        handler = TieredConfigStepHandler()
        context = MagicMock()
        result = MagicMock()
        orchestrator = MagicMock()

        handler._do_apply(
            orchestrator,
            MockVerticalWithTieredToolConfig,
            context,
            result,
        )

        # Verify context.apply_tiered_config was called
        context.apply_tiered_config.assert_called_once()
        config_arg = context.apply_tiered_config.call_args[0][0]
        assert "read" in config_arg.mandatory

    def test_handler_handles_neither_method(self):
        """Handler should handle verticals with neither method gracefully."""
        handler = TieredConfigStepHandler()
        context = MagicMock()
        result = MagicMock()
        orchestrator = MagicMock()

        # Should not raise
        handler._do_apply(
            orchestrator,
            MockVerticalWithNeither,
            context,
            result,
        )

        # Verify context.apply_tiered_config was NOT called
        context.apply_tiered_config.assert_not_called()


# =============================================================================
# Tests for protocol isinstance checks
# =============================================================================


class TestProtocolIsinstanceCheck:
    """Test that we use isinstance instead of hasattr for protocol checks."""

    def test_get_tiered_config_does_not_use_hasattr_directly(self):
        """get_tiered_config should use safe attribute access patterns."""
        # This is a documentation test - the actual check happens in coverage
        # The implementation should use hasattr + callable check or getattr
        # rather than raw hasattr which can return True for non-callable attrs

        # Create a mock with a non-callable attribute
        class BadVertical:
            name = "bad"
            get_tiered_tool_config = "not a method"  # This is a string, not callable

        # get_tiered_config should handle this gracefully
        config = get_tiered_config(BadVertical)
        # Should return None since the attribute is not callable
        assert config is None

    def test_callable_check_for_methods(self):
        """get_tiered_config should verify methods are callable."""

        # Properties on classes are descriptors, not callable when accessed on the class
        # So we test that callable class methods work correctly
        class VerticalWithClassmethod:
            name = "classmethod_vertical"

            @classmethod
            def get_tiered_tool_config(cls) -> TieredToolConfig:
                return TieredToolConfig(
                    mandatory={"from_classmethod"},
                    vertical_core=set(),
                    semantic_pool=set(),
                )

        # Classmethod should work
        config = get_tiered_config(VerticalWithClassmethod)
        assert config is not None
        assert "from_classmethod" in config.mandatory

    def test_validates_return_type(self):
        """get_tiered_config should validate return type has mandatory attr."""

        class BadReturnVertical:
            name = "bad_return"

            @classmethod
            def get_tiered_tool_config(cls):
                # Returns something that doesn't look like TieredToolConfig
                return {"not": "a config"}

        # Should return None since return doesn't have mandatory attribute
        config = get_tiered_config(BadReturnVertical)
        assert config is None
