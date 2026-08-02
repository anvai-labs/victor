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

"""Tests for ModeAwareMixin and related classes."""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from victor.protocols.mode_aware import (
    IModeController,
    ModeAwareMixin,
    ModeInfo,
    create_mode_aware_mixin,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_mode_controller():
    """Create a mock mode controller in BUILD mode."""
    mock = MagicMock()
    mock.current_mode.value = "BUILD"
    mock.config.allow_all_tools = True
    mock.config.exploration_multiplier = 2.0
    mock.config.sandbox_dir = None
    mock.config.allow_sandbox_edits = False
    mock.config.allowed_tools = set()
    mock.config.disallowed_tools = set()
    mock.is_tool_allowed.return_value = True
    mock.get_tool_priority.return_value = 1.0
    mock.get_system_prompt_addition.return_value = "BUILD mode: Full access."
    return mock


@pytest.fixture
def plan_mode_controller():
    """Create a mock mode controller in PLAN mode."""
    mock = MagicMock()
    mock.current_mode.value = "PLAN"
    mock.config.allow_all_tools = False
    mock.config.exploration_multiplier = 2.5
    mock.config.sandbox_dir = "/tmp/sandbox"
    mock.config.allow_sandbox_edits = True
    mock.config.allowed_tools = {"read_file", "list_directory", "semantic_search"}
    mock.config.disallowed_tools = {"shell", "write_file"}
    mock.is_tool_allowed.side_effect = lambda t: t not in {"shell", "write_file"}
    mock.get_tool_priority.return_value = 1.2
    mock.get_system_prompt_addition.return_value = "PLAN mode: Read-only with sandbox."
    return mock


@pytest.fixture
def explore_mode_controller():
    """Create a mock mode controller in EXPLORE mode."""
    mock = MagicMock()
    mock.current_mode.value = "EXPLORE"
    mock.config.allow_all_tools = False
    mock.config.exploration_multiplier = 3.0
    mock.config.sandbox_dir = "/tmp/notes"
    mock.config.allow_sandbox_edits = True
    mock.config.allowed_tools = {"read_file", "list_directory"}
    mock.config.disallowed_tools = {"shell", "write_file", "edit_files"}
    mock.is_tool_allowed.side_effect = lambda t: t in {"read_file", "list_directory"}
    mock.get_tool_priority.return_value = 1.5
    mock.get_system_prompt_addition.return_value = "EXPLORE mode: Read-only exploration."
    return mock


class TestComponent(ModeAwareMixin):
    """Test class that uses ModeAwareMixin."""

    def __init__(self):
        self._mode_controller = None
        self._mode_info_cache = None


# =============================================================================
# ModeInfo Tests
# =============================================================================


class TestModeInfo:
    """Tests for ModeInfo dataclass."""

    def test_default_creation(self):
        """Test creating ModeInfo with defaults."""
        info = ModeInfo()
        assert info.name == "BUILD"
        assert info.allow_all_tools is True
        assert info.exploration_multiplier == 1.0
        assert info.sandbox_dir is None
        assert info.allowed_tools == set()
        assert info.disallowed_tools == set()

    def test_post_init_sets_empty_sets(self):
        """Test __post_init__ converts None to empty sets."""
        info = ModeInfo(name="PLAN", allowed_tools=None, disallowed_tools=None)
        assert info.allowed_tools == set()
        assert info.disallowed_tools == set()

    def test_with_explicit_values(self):
        """Test creating ModeInfo with explicit values."""
        info = ModeInfo(
            name="PLAN",
            allow_all_tools=False,
            exploration_multiplier=2.5,
            sandbox_dir="/tmp/sandbox",
            allowed_tools={"read_file"},
            disallowed_tools={"shell"},
        )
        assert info.name == "PLAN"
        assert info.allow_all_tools is False
        assert info.exploration_multiplier == 2.5
        assert info.sandbox_dir == "/tmp/sandbox"
        assert "read_file" in info.allowed_tools
        assert "shell" in info.disallowed_tools

    def test_default_factory(self):
        """Test ModeInfo.default() factory method."""
        info = ModeInfo.default()
        assert info.name == "BUILD"
        assert info.allow_all_tools is True
        assert info.exploration_multiplier == 2.0
        assert info.sandbox_dir is None


# =============================================================================
# ModeAwareMixin Tests - No Controller
# =============================================================================


class TestModeAwareMixinNoController:
    """Tests for ModeAwareMixin when mode controller is unavailable."""

    def test_mode_controller_returns_none_when_unavailable(self):
        """Test that mode_controller can be set to None manually."""
        component = TestComponent()

        # Set mode_controller to None directly (it's a cached property)
        component.__dict__["mode_controller"] = None
        assert component.mode_controller is None

    def test_current_mode_name_defaults_to_build(self):
        """Test current_mode_name returns BUILD when controller unavailable."""
        component = TestComponent()
        component.__dict__["mode_controller"] = None
        assert component.current_mode_name == "BUILD"

    def test_is_build_mode_false_when_no_controller(self):
        """Test is_build_mode returns False when controller unavailable."""
        component = TestComponent()
        component.__dict__["mode_controller"] = None
        # Conservative default - don't assume build mode
        assert component.is_build_mode is False

    def test_is_plan_mode_false_when_no_controller(self):
        """Test is_plan_mode returns False when controller unavailable."""
        component = TestComponent()
        component.__dict__["mode_controller"] = None
        assert component.is_plan_mode is False

    def test_is_explore_mode_false_when_no_controller(self):
        """Test is_explore_mode returns False when controller unavailable."""
        component = TestComponent()
        component.__dict__["mode_controller"] = None
        assert component.is_explore_mode is False

    def test_exploration_multiplier_default(self):
        """Test exploration_multiplier returns 1.0 when controller unavailable."""
        component = TestComponent()
        component.__dict__["mode_controller"] = None
        assert component.exploration_multiplier == 1.0

    def test_sandbox_dir_none_when_no_controller(self):
        """Test sandbox_dir returns None when controller unavailable."""
        component = TestComponent()
        component.__dict__["mode_controller"] = None
        assert component.sandbox_dir is None

    def test_allow_sandbox_edits_false_when_no_controller(self):
        """Test allow_sandbox_edits returns False when controller unavailable."""
        component = TestComponent()
        component.__dict__["mode_controller"] = None
        assert component.allow_sandbox_edits is False

    def test_is_tool_allowed_by_mode_defaults_true(self):
        """Test is_tool_allowed_by_mode returns True when controller unavailable."""
        component = TestComponent()
        component.__dict__["mode_controller"] = None
        assert component.is_tool_allowed_by_mode("shell") is True

    def test_get_tool_mode_priority_defaults_to_one(self):
        """Test get_tool_mode_priority returns 1.0 when controller unavailable."""
        component = TestComponent()
        component.__dict__["mode_controller"] = None
        assert component.get_tool_mode_priority("read_file") == 1.0

    def test_get_mode_info_returns_default(self):
        """Test get_mode_info returns default ModeInfo when controller unavailable."""
        component = TestComponent()
        component.__dict__["mode_controller"] = None
        info = component.get_mode_info()
        assert info.name == "BUILD"
        assert info.allow_all_tools is True
        assert info.exploration_multiplier == 2.0

    def test_get_mode_system_prompt_empty_when_no_controller(self):
        """Test get_mode_system_prompt returns empty string when controller unavailable."""
        component = TestComponent()
        component.__dict__["mode_controller"] = None
        assert component.get_mode_system_prompt() == ""


# =============================================================================
# ModeAwareMixin Tests - With Controller
# =============================================================================


class TestModeAwareMixinWithController:
    """Tests for ModeAwareMixin with a mode controller."""

    def test_set_mode_controller(self, mock_mode_controller):
        """Test setting mode controller explicitly."""
        component = TestComponent()
        component.set_mode_controller(mock_mode_controller)
        assert component._mode_controller is mock_mode_controller

    def test_set_mode_controller_invalidates_cache(self, mock_mode_controller):
        """Test that set_mode_controller clears mode info cache."""
        component = TestComponent()
        component._mode_info_cache = ModeInfo(name="OLD")
        component.set_mode_controller(mock_mode_controller)
        assert component._mode_info_cache is None

    def test_current_mode_name_build(self, mock_mode_controller):
        """Test current_mode_name in BUILD mode."""
        component = TestComponent()
        component.set_mode_controller(mock_mode_controller)
        assert component.current_mode_name == "BUILD"

    def test_current_mode_name_plan(self, plan_mode_controller):
        """Test current_mode_name in PLAN mode."""
        component = TestComponent()
        component.set_mode_controller(plan_mode_controller)
        assert component.current_mode_name == "PLAN"

    def test_current_mode_name_explore(self, explore_mode_controller):
        """Test current_mode_name in EXPLORE mode."""
        component = TestComponent()
        component.set_mode_controller(explore_mode_controller)
        assert component.current_mode_name == "EXPLORE"

    def test_is_build_mode_true(self, mock_mode_controller):
        """Test is_build_mode returns True in BUILD mode."""
        component = TestComponent()
        component.set_mode_controller(mock_mode_controller)
        assert component.is_build_mode is True

    def test_is_build_mode_false_in_plan(self, plan_mode_controller):
        """Test is_build_mode returns False in PLAN mode."""
        component = TestComponent()
        component.set_mode_controller(plan_mode_controller)
        assert component.is_build_mode is False

    def test_is_plan_mode(self, plan_mode_controller):
        """Test is_plan_mode in PLAN mode."""
        component = TestComponent()
        component.set_mode_controller(plan_mode_controller)
        assert component.is_plan_mode is True

    def test_is_explore_mode(self, explore_mode_controller):
        """Test is_explore_mode in EXPLORE mode."""
        component = TestComponent()
        component.set_mode_controller(explore_mode_controller)
        assert component.is_explore_mode is True

    def test_exploration_multiplier_build(self, mock_mode_controller):
        """Test exploration_multiplier in BUILD mode."""
        component = TestComponent()
        component.set_mode_controller(mock_mode_controller)
        assert component.exploration_multiplier == 2.0

    def test_exploration_multiplier_plan(self, plan_mode_controller):
        """Test exploration_multiplier in PLAN mode."""
        component = TestComponent()
        component.set_mode_controller(plan_mode_controller)
        assert component.exploration_multiplier == 2.5

    def test_exploration_multiplier_explore(self, explore_mode_controller):
        """Test exploration_multiplier in EXPLORE mode."""
        component = TestComponent()
        component.set_mode_controller(explore_mode_controller)
        assert component.exploration_multiplier == 3.0

    def test_sandbox_dir_none_in_build(self, mock_mode_controller):
        """Test sandbox_dir is None in BUILD mode."""
        component = TestComponent()
        component.set_mode_controller(mock_mode_controller)
        assert component.sandbox_dir is None

    def test_sandbox_dir_set_in_plan(self, plan_mode_controller):
        """Test sandbox_dir is set in PLAN mode."""
        component = TestComponent()
        component.set_mode_controller(plan_mode_controller)
        assert component.sandbox_dir == "/tmp/sandbox"

    def test_allow_sandbox_edits_false_in_build(self, mock_mode_controller):
        """Test allow_sandbox_edits is False in BUILD mode."""
        component = TestComponent()
        component.set_mode_controller(mock_mode_controller)
        assert component.allow_sandbox_edits is False

    def test_allow_sandbox_edits_true_in_plan(self, plan_mode_controller):
        """Test allow_sandbox_edits is True in PLAN mode."""
        component = TestComponent()
        component.set_mode_controller(plan_mode_controller)
        assert component.allow_sandbox_edits is True

    def test_is_tool_allowed_by_mode(self, plan_mode_controller):
        """Test is_tool_allowed_by_mode delegates to controller."""
        component = TestComponent()
        component.set_mode_controller(plan_mode_controller)
        assert component.is_tool_allowed_by_mode("read_file") is True
        assert component.is_tool_allowed_by_mode("shell") is False

    def test_get_tool_mode_priority(self, plan_mode_controller):
        """Test get_tool_mode_priority delegates to controller."""
        component = TestComponent()
        component.set_mode_controller(plan_mode_controller)
        assert component.get_tool_mode_priority("read_file") == 1.2


# =============================================================================
# ModeInfo Caching Tests
# =============================================================================


class TestModeInfoCaching:
    """Tests for ModeInfo caching in ModeAwareMixin."""

    def test_get_mode_info_caches_result(self, mock_mode_controller):
        """Test that get_mode_info caches the result."""
        component = TestComponent()
        component.set_mode_controller(mock_mode_controller)

        info1 = component.get_mode_info()
        info2 = component.get_mode_info()

        assert info1 is info2  # Same object returned

    def test_get_mode_info_refresh_forces_reload(self, mock_mode_controller):
        """Test that get_mode_info(refresh=True) reloads."""
        component = TestComponent()
        component.set_mode_controller(mock_mode_controller)

        info1 = component.get_mode_info()
        # Change controller state
        mock_mode_controller.current_mode.value = "PLAN"
        mock_mode_controller.config.allow_all_tools = False

        info2 = component.get_mode_info(refresh=True)

        assert info2.name == "PLAN"
        assert info2.allow_all_tools is False

    def test_get_mode_info_captures_all_fields(self, plan_mode_controller):
        """Test that get_mode_info captures all config fields."""
        component = TestComponent()
        component.set_mode_controller(plan_mode_controller)

        info = component.get_mode_info()

        assert info.name == "PLAN"
        assert info.allow_all_tools is False
        assert info.exploration_multiplier == 2.5
        assert info.sandbox_dir == "/tmp/sandbox"
        assert "read_file" in info.allowed_tools
        assert "shell" in info.disallowed_tools


# =============================================================================
# System Prompt Tests
# =============================================================================


class TestModeSystemPrompt:
    """Tests for mode system prompt functionality."""

    def test_get_mode_system_prompt_build(self, mock_mode_controller):
        """Test system prompt in BUILD mode."""
        component = TestComponent()
        component.set_mode_controller(mock_mode_controller)
        prompt = component.get_mode_system_prompt()
        assert "BUILD mode" in prompt

    def test_get_mode_system_prompt_plan(self, plan_mode_controller):
        """Test system prompt in PLAN mode."""
        component = TestComponent()
        component.set_mode_controller(plan_mode_controller)
        prompt = component.get_mode_system_prompt()
        assert "PLAN mode" in prompt

    def test_get_mode_system_prompt_explore(self, explore_mode_controller):
        """Test system prompt in EXPLORE mode."""
        component = TestComponent()
        component.set_mode_controller(explore_mode_controller)
        prompt = component.get_mode_system_prompt()
        assert "EXPLORE mode" in prompt


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateModeAwareMixin:
    """Tests for create_mode_aware_mixin factory function."""

    def test_creates_instance(self):
        """Test that factory creates ModeAwareMixin instance."""
        instance = create_mode_aware_mixin()
        assert isinstance(instance, ModeAwareMixin)

    def test_instance_has_default_properties(self):
        """Test that created instance has expected default behavior."""
        instance = create_mode_aware_mixin()
        # Mode controller not set, should use defaults
        assert instance.current_mode_name == "BUILD"


# =============================================================================
# IModeController Protocol Tests
# =============================================================================


class TestIModeControllerProtocol:
    """Tests for IModeController protocol compliance."""

    def test_protocol_runtime_checkable(self):
        """Test that IModeController is runtime checkable."""

        @dataclass
        class SimpleController:
            current_mode: MagicMock
            config: MagicMock

            def is_tool_allowed(self, tool_name: str) -> bool:
                return True

            def get_tool_priority(self, tool_name: str) -> float:
                return 1.0

        controller = SimpleController(
            current_mode=MagicMock(value="BUILD"),
            config=MagicMock(),
        )
        assert isinstance(controller, IModeController)

    def test_protocol_has_required_methods(self):
        """Test that IModeController defines required methods."""
        # Check protocol has expected attributes
        assert hasattr(IModeController, "current_mode")
        assert hasattr(IModeController, "config")
        assert hasattr(IModeController, "is_tool_allowed")
        assert hasattr(IModeController, "get_tool_priority")
