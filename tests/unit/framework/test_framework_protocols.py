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

"""Tests for framework protocol implementations.

Tests for:
- ModeAwareMixin
- ToolAccessController
- BudgetManager
- PathResolver
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import os

# =============================================================================
# ModeAwareMixin Tests
# =============================================================================


class TestModeAwareMixin:
    """Tests for ModeAwareMixin."""

    def test_mode_info_default(self):
        """Test ModeInfo default values."""
        from victor.protocols.mode_aware import ModeInfo

        info = ModeInfo.default()
        assert info.name == "BUILD"
        assert info.allow_all_tools is True
        assert info.exploration_multiplier == 2.0

    def test_mixin_without_controller(self):
        """Test mixin behavior when mode controller not available."""
        from victor.protocols.mode_aware import ModeAwareMixin

        class TestClass(ModeAwareMixin):
            pass

        obj = TestClass()

        # Mock get_mode_controller at the source to simulate unavailable mode controller
        with patch("victor.agent.mode_controller.get_mode_controller") as mock_get:
            mock_get.side_effect = RuntimeError("Mode controller not initialized")

            # Clear any cached controller
            obj._mode_controller = None
            if "mode_controller" in obj.__dict__:
                del obj.__dict__["mode_controller"]

            # With no controller, should return safe/conservative defaults
            # Note: current_mode_name returns BUILD as default name
            assert obj.current_mode_name == "BUILD"
            # is_build_mode returns False when controller unavailable (conservative)
            # This ensures stage filtering and other protections remain active
            assert obj.is_build_mode is False
            # exploration_multiplier returns 1.0 when controller is None
            assert obj.exploration_multiplier == 1.0

    def test_mixin_with_mock_controller(self):
        """Test mixin with mock mode controller."""
        from victor.protocols.mode_aware import ModeAwareMixin

        class TestClass(ModeAwareMixin):
            pass

        # Create mock controller
        mock_controller = MagicMock()
        mock_controller.current_mode.value = "PLAN"
        mock_controller.config.allow_all_tools = False
        mock_controller.config.exploration_multiplier = 2.5
        mock_controller.config.sandbox_dir = ".victor/sandbox"
        mock_controller.config.allowed_tools = {"read_file", "list_directory"}
        mock_controller.config.disallowed_tools = {"shell", "bash"}
        mock_controller.is_tool_allowed.return_value = True
        mock_controller.get_tool_priority.return_value = 1.2

        obj = TestClass()
        obj.set_mode_controller(mock_controller)

        assert obj.current_mode_name == "PLAN"
        assert obj.is_build_mode is False
        assert obj.is_plan_mode is True
        assert obj.exploration_multiplier == 2.5
        assert obj.sandbox_dir == ".victor/sandbox"

    def test_mode_info_snapshot(self):
        """Test get_mode_info returns snapshot."""
        from victor.protocols.mode_aware import ModeAwareMixin

        class TestClass(ModeAwareMixin):
            pass

        mock_controller = MagicMock()
        mock_controller.current_mode.value = "EXPLORE"
        mock_controller.config.allow_all_tools = False
        mock_controller.config.exploration_multiplier = 3.0
        mock_controller.config.sandbox_dir = None
        mock_controller.config.allowed_tools = {"read_file"}
        mock_controller.config.disallowed_tools = set()

        obj = TestClass()
        obj.set_mode_controller(mock_controller)

        info = obj.get_mode_info()
        assert info.name == "EXPLORE"
        assert info.exploration_multiplier == 3.0


# =============================================================================
# ToolAccessController Tests
# =============================================================================


class TestToolAccessController:
    """Tests for ToolAccessController."""

    def test_controller_creation(self):
        """Test controller can be created."""
        from victor.agent.tool_access_controller import (
            ToolAccessController,
            create_tool_access_controller,
        )

        controller = create_tool_access_controller()
        assert controller is not None
        assert len(controller.layers) > 0

    def test_check_access_no_context(self):
        """Test access check without context defaults to allowing."""
        from victor.agent.tool_access_controller import create_tool_access_controller

        controller = create_tool_access_controller()
        decision = controller.check_access("read_file")

        # Should be allowed with no restrictions
        assert decision.allowed is True
        assert decision.tool_name == "read_file"

    def test_check_access_with_context(self):
        """Test access check with context."""
        from victor.agent.tool_access_controller import create_tool_access_controller
        from victor.agent.protocols import ToolAccessContext

        controller = create_tool_access_controller()
        context = ToolAccessContext(
            current_mode="BUILD",
            session_enabled_tools={"read_file", "write_file", "shell"},
        )

        decision = controller.check_access("read_file", context)
        assert decision.allowed is True

    def test_filter_tools(self):
        """Test filtering a list of tools."""
        from victor.agent.tool_access_controller import create_tool_access_controller
        from victor.agent.protocols import ToolAccessContext

        controller = create_tool_access_controller()
        tools = ["read_file", "write_file", "shell", "code_search"]

        allowed, denials = controller.filter_tools(tools)
        assert len(allowed) == len(tools)  # All allowed without restrictions

    def test_safety_layer_sandbox_mode(self):
        """Test safety layer blocks dangerous tools in sandbox mode."""
        from victor.agent.tool_access_controller import (
            SafetyLayer,
            create_tool_access_controller,
        )

        # Create controller with sandbox mode enabled
        controller = create_tool_access_controller(sandbox_mode=True)

        decision = controller.check_access("shell")
        assert decision.allowed is False
        assert decision.source == "safety"

    def test_explain_decision(self):
        """Test explanation generation."""
        from victor.agent.tool_access_controller import create_tool_access_controller

        controller = create_tool_access_controller()
        explanation = controller.explain_decision("read_file")

        assert "read_file" in explanation
        assert "ALLOWED" in explanation

    def test_layer_precedence(self):
        """Test layers are checked in correct order."""
        from victor.agent.tool_access_controller import ToolAccessController
        from victor.agent.protocols import AccessPrecedence

        controller = ToolAccessController()

        # Verify layers are sorted by precedence
        precedences = [layer.PRECEDENCE for layer in controller.layers]
        assert precedences == sorted(precedences)


# =============================================================================
# BudgetManager Tests
# =============================================================================


class TestBudgetManager:
    """Tests for BudgetManager."""

    def test_budget_creation(self):
        """Test budget manager creation."""
        from victor.agent.budget_manager import create_budget_manager
        from victor.agent.protocols import BudgetConfig

        config = BudgetConfig(
            base_tool_calls=20,
            base_iterations=40,
            base_exploration=10,
            base_action=15,
        )
        manager = create_budget_manager(config=config)
        assert manager is not None

    def test_consume_budget(self):
        """Test budget consumption."""
        from victor.agent.budget_manager import create_budget_manager
        from victor.agent.protocols import BudgetType, BudgetConfig

        config = BudgetConfig(base_exploration=5)
        manager = create_budget_manager(config=config)

        # Consume budget
        for i in range(5):
            assert manager.consume(BudgetType.EXPLORATION) is True

        # Budget exhausted
        assert manager.is_exhausted(BudgetType.EXPLORATION) is True

    def test_multiplier_composition(self):
        """Test multiplier composition."""
        from victor.agent.budget_manager import create_budget_manager
        from victor.agent.protocols import BudgetType, BudgetConfig

        config = BudgetConfig(base_exploration=10)
        manager = create_budget_manager(config=config)

        # Set multipliers
        manager.set_model_multiplier(1.5)  # 1.5x
        manager.set_mode_multiplier(2.0)  # 2.0x
        # Combined: 10 * 1.5 * 2.0 = 30

        status = manager.get_status(BudgetType.EXPLORATION)
        assert status.effective_maximum == 30
        assert status.model_multiplier == 1.5
        assert status.mode_multiplier == 2.0

    def test_record_tool_call(self):
        """Test recording tool calls."""
        from victor.agent.budget_manager import create_budget_manager
        from victor.agent.protocols import BudgetType, BudgetConfig

        config = BudgetConfig(base_exploration=10, base_action=10)
        manager = create_budget_manager(config=config)

        # Read operation should consume exploration budget
        manager.record_tool_call("read_file", is_write_operation=False)
        assert manager.get_status(BudgetType.EXPLORATION).current == 1
        assert manager.get_status(BudgetType.ACTION).current == 0

        # Write operation should consume action budget
        manager.record_tool_call("write_file", is_write_operation=True)
        assert manager.get_status(BudgetType.EXPLORATION).current == 1
        assert manager.get_status(BudgetType.ACTION).current == 1

    def test_auto_detect_write_tools(self):
        """Test auto-detection of write tools."""
        from victor.agent.budget_manager import create_budget_manager, is_write_tool
        from victor.agent.protocols import BudgetType, BudgetConfig

        # Test write tool detection
        assert is_write_tool("write_file") is True
        assert is_write_tool("edit_files") is True
        assert is_write_tool("shell") is True
        assert is_write_tool("read_file") is False
        assert is_write_tool("code_search") is False

    def test_reset_budget(self):
        """Test budget reset."""
        from victor.agent.budget_manager import create_budget_manager
        from victor.agent.protocols import BudgetType, BudgetConfig

        config = BudgetConfig(base_exploration=5)
        manager = create_budget_manager(config=config)

        # Consume some budget
        manager.consume(BudgetType.EXPLORATION, amount=3)
        assert manager.get_status(BudgetType.EXPLORATION).current == 3

        # Reset
        manager.reset(BudgetType.EXPLORATION)
        assert manager.get_status(BudgetType.EXPLORATION).current == 0

    def test_prompt_budget_info(self):
        """Test prompt budget info generation."""
        from victor.agent.budget_manager import create_budget_manager
        from victor.agent.protocols import BudgetConfig

        config = BudgetConfig(base_tool_calls=30, base_exploration=10)
        manager = create_budget_manager(config=config)
        manager.set_mode_multiplier(2.0)

        info = manager.get_prompt_budget_info()
        assert "tool_budget" in info
        assert "exploration_budget" in info
        assert info["mode_multiplier"] == 2.0


# =============================================================================
# PathResolver Tests
# =============================================================================


class TestPathResolver:
    """Tests for PathResolver."""

    def test_resolver_creation(self):
        """Test resolver creation."""
        from victor.protocols.path_resolver import create_path_resolver

        resolver = create_path_resolver()
        assert resolver is not None
        assert resolver.cwd is not None

    def test_resolve_existing_file(self):
        """Test resolving existing file."""
        from victor.protocols.path_resolver import PathResolver

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("# test")

            resolver = PathResolver(cwd=Path(tmpdir))
            result = resolver.resolve("test.py")

            assert result.exists is True
            assert result.is_file is True
            # Use resolve() on both to handle symlinks (e.g., /var -> /private/var on macOS)
            assert result.resolved_path.resolve() == test_file.resolve()

    def test_resolve_directory(self):
        """Test resolving directory."""
        from victor.protocols.path_resolver import PathResolver

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a subdirectory
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()

            resolver = PathResolver(cwd=Path(tmpdir))
            result = resolver.resolve_directory("subdir")

            assert result.exists is True
            assert result.is_directory is True

    def test_normalize_cwd_prefix(self):
        """Test normalization of cwd prefix."""
        from victor.protocols.path_resolver import PathResolver

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create project/utils/test.py
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir()
            utils_dir = project_dir / "utils"
            utils_dir.mkdir()
            test_file = utils_dir / "test.py"
            test_file.write_text("# test")

            # Resolver's cwd is project/
            resolver = PathResolver(cwd=project_dir)

            # LLM might use "project/utils/test.py" even though cwd is project/
            result = resolver.resolve("project/utils/test.py")

            assert result.exists is True
            assert result.was_normalized is True
            assert "utils/test.py" in str(result.resolved_path)

    def test_resolve_nonexistent_file(self):
        """Test resolving nonexistent file raises error."""
        from victor.protocols.path_resolver import PathResolver

        with tempfile.TemporaryDirectory() as tmpdir:
            resolver = PathResolver(cwd=Path(tmpdir))

            with pytest.raises(FileNotFoundError):
                resolver.resolve("nonexistent.py")

    def test_resolve_nonexistent_optional(self):
        """Test resolving nonexistent file with must_exist=False."""
        from victor.protocols.path_resolver import PathResolver

        with tempfile.TemporaryDirectory() as tmpdir:
            resolver = PathResolver(cwd=Path(tmpdir))
            result = resolver.resolve("nonexistent.py", must_exist=False)

            assert result.exists is False

    def test_suggest_similar(self):
        """Test similar path suggestions."""
        from victor.protocols.path_resolver import PathResolver

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files
            (Path(tmpdir) / "test_models.py").write_text("# models")
            (Path(tmpdir) / "test_views.py").write_text("# views")

            resolver = PathResolver(cwd=Path(tmpdir))
            suggestions = resolver.suggest_similar("test_modls.py")  # Typo

            assert len(suggestions) > 0
            # Should suggest test_models.py
            assert any("model" in s for s in suggestions)

    def test_file_vs_directory_error(self):
        """Test appropriate errors for file vs directory mismatch."""
        from victor.protocols.path_resolver import PathResolver

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory
            subdir = Path(tmpdir) / "mydir"
            subdir.mkdir()

            resolver = PathResolver(cwd=Path(tmpdir))

            # Should raise IsADirectoryError when trying to read as file
            with pytest.raises(IsADirectoryError):
                resolver.resolve_file("mydir")

    def test_caching(self):
        """Test resolution caching."""
        from victor.protocols.path_resolver import PathResolver

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("# test")

            resolver = PathResolver(cwd=Path(tmpdir))

            # First resolution
            result1 = resolver.resolve("test.py")
            # Second resolution (should use cache)
            result2 = resolver.resolve("test.py")

            assert result1.resolved_path == result2.resolved_path


# =============================================================================
# Integration Tests
# =============================================================================


class TestFrameworkIntegration:
    """Integration tests for framework components."""

    def test_mode_aware_budget_manager(self):
        """Test BudgetManager with mode awareness."""
        from victor.agent.budget_manager import BudgetManager
        from victor.agent.protocols import BudgetType, BudgetConfig

        config = BudgetConfig(base_exploration=10)
        manager = BudgetManager(config=config)

        # Mock the mode controller
        mock_controller = MagicMock()
        mock_controller.current_mode.value = "PLAN"
        mock_controller.config.allow_all_tools = False
        mock_controller.config.exploration_multiplier = 2.5

        manager.set_mode_controller(mock_controller)
        manager.update_from_mode()

        # Mode multiplier should be applied
        status = manager.get_status(BudgetType.EXPLORATION)
        assert status.effective_maximum == 25  # 10 * 2.5

    def test_tool_access_with_budget_check(self):
        """Test combining tool access and budget checks."""
        from victor.agent.tool_access_controller import create_tool_access_controller
        from victor.agent.budget_manager import create_budget_manager
        from victor.agent.protocols import BudgetType, BudgetConfig

        # Create both controllers
        access_controller = create_tool_access_controller()
        budget_manager = create_budget_manager(config=BudgetConfig(base_exploration=3))

        tools = ["read_file", "code_search", "list_directory"]

        for tool in tools:
            # Check access
            decision = access_controller.check_access(tool)
            if decision.allowed:
                # Record tool call
                budget_manager.record_tool_call(tool)

        # Should have consumed 3 exploration budget
        assert budget_manager.get_status(BudgetType.EXPLORATION).current == 3
        assert budget_manager.is_exhausted(BudgetType.EXPLORATION) is True
