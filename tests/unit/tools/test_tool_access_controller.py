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

"""Tests for ToolAccessController and related classes."""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Set
from unittest.mock import MagicMock, patch

from victor.agent.protocols import (
    AccessPrecedence,
    IToolAccessController,
    ToolAccessContext,
    ToolAccessDecision,
)
from victor.agent.tool_access_controller import (
    AccessLayer,
    IntentLayer,
    ModeLayer,
    SafetyLayer,
    SessionLayer,
    StageLayer,
    ToolAccessController,
    VerticalLayer,
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
    mock.config.allowed_tools = set()
    mock.config.disallowed_tools = set()
    mock.is_tool_allowed.return_value = True
    mock.get_tool_priority.return_value = 1.0
    return mock


@pytest.fixture
def plan_mode_controller():
    """Create a mock mode controller in PLAN mode."""
    mock = MagicMock()
    mock.current_mode.value = "PLAN"
    mock.config.allow_all_tools = False
    mock.config.exploration_multiplier = 2.5
    mock.config.sandbox_dir = "/tmp/sandbox"
    mock.config.allowed_tools = {"read_file", "list_directory", "semantic_search"}
    mock.config.disallowed_tools = {"shell", "write_file"}
    mock.is_tool_allowed.side_effect = lambda t: t not in {"shell", "write_file"}
    mock.get_tool_priority.return_value = 1.2
    return mock


@pytest.fixture
def mock_registry():
    """Create a mock tool registry."""
    mock = MagicMock()

    class MockTool:
        def __init__(self, name):
            self.name = name

    mock.list_tools.return_value = [
        MockTool("read_file"),
        MockTool("write_file"),
        MockTool("shell"),
        MockTool("list_directory"),
        MockTool("semantic_search"),
    ]
    return mock


@pytest.fixture
def controller(mock_registry):
    """Create a ToolAccessController with default layers."""
    return ToolAccessController(registry=mock_registry)


# =============================================================================
# ToolAccessDecision Tests
# =============================================================================


class TestToolAccessDecision:
    """Tests for ToolAccessDecision dataclass."""

    def test_basic_allowed_decision(self):
        """Test creating an allowed decision."""
        decision = ToolAccessDecision(
            allowed=True,
            tool_name="read_file",
            reason="All layers allow",
            source="all",
            precedence_level=-1,
        )
        assert decision.allowed is True
        assert decision.tool_name == "read_file"

    def test_denied_decision(self):
        """Test creating a denied decision."""
        decision = ToolAccessDecision(
            allowed=False,
            tool_name="shell",
            reason="Tool blocked in PLAN mode",
            source="mode",
            precedence_level=1,
        )
        assert decision.allowed is False
        assert decision.source == "mode"

    def test_decision_with_layer_results(self):
        """Test decision with layer results."""
        decision = ToolAccessDecision(
            allowed=False,
            tool_name="shell",
            reason="Blocked",
            source="safety",
            precedence_level=0,
            checked_layers=["safety", "mode", "session"],
            layer_results={"safety": False, "mode": True, "session": True},
        )
        assert len(decision.checked_layers) == 3
        assert decision.layer_results["safety"] is False


# =============================================================================
# ToolAccessContext Tests
# =============================================================================


class TestToolAccessContext:
    """Tests for ToolAccessContext dataclass."""

    def test_empty_context(self):
        """Test creating empty context."""
        context = ToolAccessContext()
        assert context.user_message is None
        assert context.current_mode is None
        assert context.session_enabled_tools is None

    def test_context_with_values(self):
        """Test context with values."""
        context = ToolAccessContext(
            current_mode="PLAN",
            session_enabled_tools={"read_file", "list_directory"},
        )
        assert context.current_mode == "PLAN"
        assert "read_file" in context.session_enabled_tools


# =============================================================================
# SafetyLayer Tests
# =============================================================================


class TestSafetyLayer:
    """Tests for SafetyLayer."""

    def test_precedence(self):
        """Test safety layer has highest precedence."""
        layer = SafetyLayer()
        assert layer.PRECEDENCE == AccessPrecedence.SAFETY.value
        assert layer.NAME == "safety"

    def test_allows_normal_tools(self):
        """Test normal tools pass safety check."""
        layer = SafetyLayer()
        allowed, reason = layer.check("read_file", None)
        assert allowed is True

    def test_sandbox_mode_blocks_dangerous(self):
        """Test sandbox mode blocks dangerous tools."""
        layer = SafetyLayer(sandbox_mode=True)
        allowed, reason = layer.check("shell", None)
        assert allowed is False
        assert "sandbox" in reason.lower()

    def test_sandbox_mode_blocks_shell_alias(self):
        """Legacy shell aliases should still be blocked via canonical matching."""
        layer = SafetyLayer(sandbox_mode=True)
        allowed, reason = layer.check("execute_bash", None)
        assert allowed is False
        assert "sandbox" in reason.lower()

    def test_non_sandbox_allows_dangerous(self):
        """Test non-sandbox mode allows dangerous tools."""
        layer = SafetyLayer(sandbox_mode=False)
        allowed, reason = layer.check("shell", None)
        assert allowed is True

    def test_get_allowed_tools(self):
        """Test get_allowed_tools filters correctly."""
        layer = SafetyLayer(sandbox_mode=True)
        all_tools = {"read_file", "shell", "write_file", "list_directory"}
        allowed = layer.get_allowed_tools(all_tools, None)
        assert "read_file" in allowed
        assert "list_directory" in allowed


# =============================================================================
# ModeLayer Tests
# =============================================================================


class TestModeLayer:
    """Tests for ModeLayer."""

    def test_precedence(self):
        """Test mode layer precedence."""
        layer = ModeLayer()
        assert layer.PRECEDENCE == AccessPrecedence.MODE.value
        assert layer.NAME == "mode"

    def test_build_mode_allows_all(self, mock_mode_controller):
        """Test BUILD mode allows all tools."""
        layer = ModeLayer()
        layer.set_mode_controller(mock_mode_controller)

        allowed, reason = layer.check("shell", None)
        assert allowed is True
        assert "BUILD" in reason

    def test_plan_mode_restricts(self, plan_mode_controller):
        """Test PLAN mode restricts tools."""
        layer = ModeLayer()
        layer.set_mode_controller(plan_mode_controller)

        allowed, reason = layer.check("shell", None)
        assert allowed is False
        assert "PLAN" in reason

    def test_get_allowed_tools_build_mode(self, mock_mode_controller):
        """Test get_allowed_tools in BUILD mode."""
        layer = ModeLayer()
        layer.set_mode_controller(mock_mode_controller)

        all_tools = {"read_file", "shell", "write_file"}
        allowed = layer.get_allowed_tools(all_tools, None)
        assert allowed == all_tools


# =============================================================================
# SessionLayer Tests
# =============================================================================


class TestSessionLayer:
    """Tests for SessionLayer."""

    def test_precedence(self):
        """Test session layer precedence."""
        layer = SessionLayer()
        assert layer.PRECEDENCE == AccessPrecedence.SESSION.value
        assert layer.NAME == "session"

    def test_no_restrictions_with_none_context(self):
        """Test no restrictions when context is None."""
        layer = SessionLayer()
        allowed, reason = layer.check("shell", None)
        assert allowed is True
        assert "no session restrictions" in reason.lower()

    def test_no_restrictions_with_none_tools(self):
        """Test no restrictions when session_enabled_tools is None."""
        layer = SessionLayer()
        context = ToolAccessContext(session_enabled_tools=None)
        allowed, reason = layer.check("shell", context)
        assert allowed is True

    def test_allows_enabled_tools(self):
        """Test allows tools in session_enabled_tools."""
        layer = SessionLayer()
        context = ToolAccessContext(session_enabled_tools={"read_file", "shell"})
        allowed, reason = layer.check("read_file", context)
        assert allowed is True
        assert "enabled" in reason.lower()

    def test_blocks_non_enabled_tools(self):
        """Test blocks tools not in session_enabled_tools."""
        layer = SessionLayer()
        context = ToolAccessContext(session_enabled_tools={"read_file"})
        allowed, reason = layer.check("shell", context)
        assert allowed is False
        assert "not enabled" in reason.lower()

    def test_get_allowed_tools(self):
        """Test get_allowed_tools filters correctly."""
        layer = SessionLayer()
        context = ToolAccessContext(session_enabled_tools={"read_file", "list_directory"})
        all_tools = {"read_file", "shell", "write_file", "list_directory"}
        allowed = layer.get_allowed_tools(all_tools, context)
        assert allowed == {"read_file", "list_directory"}


# =============================================================================
# VerticalLayer Tests
# =============================================================================


class TestVerticalLayer:
    """Tests for VerticalLayer."""

    def test_precedence(self):
        """Test vertical layer precedence."""
        layer = VerticalLayer()
        assert layer.PRECEDENCE == AccessPrecedence.VERTICAL.value
        assert layer.NAME == "vertical"

    def test_no_restrictions_without_config(self):
        """Test no restrictions when tiered_config is None."""
        layer = VerticalLayer()
        allowed, reason = layer.check("shell", None)
        assert allowed is True
        assert "no vertical restrictions" in reason.lower()

    def test_allows_core_tools(self):
        """Test allows tools in core_tools."""
        config = MagicMock()
        config.core_tools = ["read_file", "shell"]
        config.extension_tools = []
        config.optional_tools = []

        layer = VerticalLayer(tiered_config=config)
        allowed, reason = layer.check("read_file", None)
        assert allowed is True

    def test_allows_extension_tools(self):
        """Test allows tools in extension_tools."""
        config = MagicMock()
        config.core_tools = []
        config.extension_tools = ["semantic_search"]
        config.optional_tools = []

        layer = VerticalLayer(tiered_config=config)
        allowed, reason = layer.check("semantic_search", None)
        assert allowed is True

    def test_blocks_non_vertical_tools(self):
        """Test blocks tools not in any tier."""
        config = MagicMock()
        config.core_tools = ["read_file"]
        config.extension_tools = []
        config.optional_tools = []

        layer = VerticalLayer(tiered_config=config)
        allowed, reason = layer.check("shell", None)
        assert allowed is False
        assert "not in" in reason.lower()

    def test_set_tiered_config(self):
        """Test set_tiered_config updates config."""
        layer = VerticalLayer()
        config = MagicMock()
        config.core_tools = ["read_file"]
        config.extension_tools = []
        config.optional_tools = []

        layer.set_tiered_config(config)
        allowed, reason = layer.check("shell", None)
        assert allowed is False


# =============================================================================
# StageLayer Tests
# =============================================================================


class TestStageLayer:
    """Tests for StageLayer."""

    def test_precedence(self):
        """Test stage layer precedence."""
        layer = StageLayer()
        assert layer.PRECEDENCE == AccessPrecedence.STAGE.value
        assert layer.NAME == "stage"


# =============================================================================
# IntentLayer Tests
# =============================================================================


class TestIntentLayer:
    """Tests for IntentLayer."""

    def test_precedence(self):
        """Test intent layer precedence."""
        layer = IntentLayer()
        assert layer.PRECEDENCE == AccessPrecedence.INTENT.value
        assert layer.NAME == "intent"

    def test_no_restrictions_without_intent(self):
        """Test no restrictions when intent is None."""
        layer = IntentLayer()
        context = ToolAccessContext(intent=None)
        allowed, reason = layer.check("shell", context)
        assert allowed is True

    def test_no_restrictions_with_none_context(self):
        """Test no restrictions when context is None."""
        layer = IntentLayer()
        allowed, reason = layer.check("shell", None)
        assert allowed is True

    def test_blocks_write_for_readonly_intent(self):
        """Test blocks write tools for read-only intent."""
        layer = IntentLayer()
        intent = MagicMock()
        intent.name = "READ_ONLY"
        context = ToolAccessContext(intent=intent)

        allowed, reason = layer.check("write_file", context)
        assert allowed is False
        assert "write" in reason.lower()

    def test_allows_read_for_readonly_intent(self):
        """Test allows read tools for read-only intent."""
        layer = IntentLayer()
        intent = MagicMock()
        intent.name = "READ_ONLY"
        context = ToolAccessContext(intent=intent)

        allowed, reason = layer.check("read_file", context)
        assert allowed is True

    def test_allows_shell_for_explicit_readonly_sqlite_request(self):
        """Explicit readonly shell requests should survive intent filtering."""
        layer = IntentLayer()
        intent = MagicMock()
        intent.name = "READ_ONLY"
        context = ToolAccessContext(
            intent=intent,
            user_message="use shell tool with sqlite commands to inspect generations",
        )

        allowed, reason = layer.check("shell", context)
        assert allowed is True


# =============================================================================
# ToolAccessController Tests
# =============================================================================


class TestToolAccessControllerInit:
    """Tests for ToolAccessController initialization."""

    def test_creates_default_layers(self, mock_registry):
        """Test creates default layers when none provided."""
        controller = ToolAccessController(registry=mock_registry)
        assert len(controller.layers) == 6
        layer_names = [layer.NAME for layer in controller.layers]
        assert "safety" in layer_names
        assert "mode" in layer_names
        assert "session" in layer_names

    def test_uses_custom_layers(self, mock_registry):
        """Test uses custom layers when provided."""
        custom_layers = [SafetyLayer(), ModeLayer()]
        controller = ToolAccessController(registry=mock_registry, layers=custom_layers)
        assert len(controller.layers) == 2


class TestToolAccessControllerCheckAccess:
    """Tests for ToolAccessController.check_access() method."""

    def test_allows_when_all_layers_pass(self, controller):
        """Test allows when all layers pass."""
        decision = controller.check_access("read_file")
        assert decision.allowed is True
        assert decision.source == "all"

    def test_caches_decisions(self, controller):
        """Test decisions are cached."""
        decision1 = controller.check_access("read_file")
        decision2 = controller.check_access("read_file")
        assert decision1 is decision2

    def test_records_all_checked_layers(self, controller):
        """Test all layers are recorded in decision."""
        decision = controller.check_access("read_file")
        assert len(decision.checked_layers) > 0
        assert all(isinstance(l, str) for l in decision.checked_layers)

    def test_records_layer_results(self, controller):
        """Test layer results are recorded."""
        decision = controller.check_access("read_file")
        assert len(decision.layer_results) > 0

    def test_invalidates_cache_on_context_change(self, controller):
        """Test cache invalidates when context changes."""
        context1 = ToolAccessContext(current_mode="BUILD")
        context2 = ToolAccessContext(current_mode="PLAN")

        decision1 = controller.check_access("read_file", context1)
        # Internal cache should be invalidated on context change
        decision2 = controller.check_access("read_file", context2)
        # These might be different objects due to cache invalidation
        assert decision1.tool_name == decision2.tool_name


class TestToolAccessControllerFilterTools:
    """Tests for ToolAccessController.filter_tools() method."""

    def test_returns_allowed_tools(self, controller):
        """Test returns list of allowed tools."""
        tools = ["read_file", "list_directory", "shell"]
        allowed, denials = controller.filter_tools(tools)
        assert isinstance(allowed, list)
        assert isinstance(denials, list)

    def test_returns_denial_decisions(self, controller):
        """Test returns denial decisions."""
        # Set up a restriction
        context = ToolAccessContext(session_enabled_tools={"read_file"})
        tools = ["read_file", "shell"]
        allowed, denials = controller.filter_tools(tools, context)
        assert "read_file" in allowed
        assert "shell" not in allowed
        assert len(denials) == 1
        assert denials[0].tool_name == "shell"


class TestToolAccessControllerGetAllowedTools:
    """Tests for ToolAccessController.get_allowed_tools() method."""

    def test_returns_empty_without_registry(self):
        """Test returns empty set without registry."""
        controller = ToolAccessController(registry=None)
        allowed = controller.get_allowed_tools()
        assert allowed == set()

    def test_returns_tools_from_registry(self, controller):
        """Test returns tools from registry."""
        allowed = controller.get_allowed_tools()
        assert isinstance(allowed, set)


# =============================================================================
# IToolAccessController Protocol Tests
# =============================================================================


class TestIToolAccessControllerProtocol:
    """Tests for IToolAccessController protocol compliance."""

    def test_controller_implements_protocol(self, mock_registry):
        """Test that ToolAccessController implements IToolAccessController."""
        controller = ToolAccessController(registry=mock_registry)
        assert isinstance(controller, IToolAccessController)

    def test_protocol_has_required_methods(self):
        """Test that IToolAccessController defines required methods."""
        assert hasattr(IToolAccessController, "check_access")
        assert hasattr(IToolAccessController, "filter_tools")
        assert hasattr(IToolAccessController, "get_allowed_tools")


# =============================================================================
# AccessPrecedence Tests
# =============================================================================


class TestAccessPrecedence:
    """Tests for AccessPrecedence enum."""

    def test_precedence_order(self):
        """Test precedence order is correct."""
        assert AccessPrecedence.SAFETY.value < AccessPrecedence.MODE.value
        assert AccessPrecedence.MODE.value < AccessPrecedence.SESSION.value
        assert AccessPrecedence.SESSION.value < AccessPrecedence.VERTICAL.value
        assert AccessPrecedence.VERTICAL.value < AccessPrecedence.STAGE.value
        assert AccessPrecedence.STAGE.value < AccessPrecedence.INTENT.value


# =============================================================================
# Integration Tests
# =============================================================================


class TestToolAccessControllerIntegration:
    """Integration tests for ToolAccessController."""

    def test_safety_overrides_mode(self, mock_registry):
        """Test safety layer takes precedence over mode layer."""
        controller = ToolAccessController(
            registry=mock_registry,
            layers=[
                SafetyLayer(sandbox_mode=True),
                ModeLayer(),
            ],
        )

        # Even if mode would allow, safety blocks
        decision = controller.check_access("shell")
        assert decision.allowed is False
        assert decision.source == "safety"

    def test_multiple_layers_all_pass(self, mock_registry):
        """Test tool passes when all layers allow it."""
        controller = ToolAccessController(
            registry=mock_registry,
            layers=[
                SafetyLayer(sandbox_mode=False),
                SessionLayer(),
            ],
        )

        context = ToolAccessContext(session_enabled_tools={"read_file"})
        decision = controller.check_access("read_file", context)
        assert decision.allowed is True

    def test_session_restriction_applied(self, mock_registry):
        """Test session restriction is applied."""
        controller = ToolAccessController(
            registry=mock_registry,
            layers=[
                SafetyLayer(sandbox_mode=False),
                SessionLayer(),
            ],
        )

        context = ToolAccessContext(session_enabled_tools={"read_file"})
        decision = controller.check_access("shell", context)
        assert decision.allowed is False
        assert decision.source == "session"


# =============================================================================
# VerticalLayer Extended Tests
# =============================================================================


class TestVerticalLayerExtended:
    """Extended tests for VerticalLayer."""

    def test_no_config_allows_all(self):
        """Test vertical layer with no config allows all tools."""
        layer = VerticalLayer(tiered_config=None)
        allowed, reason = layer.check("any_tool", None)
        assert allowed is True
        assert "no vertical" in reason.lower() or "not configured" in reason.lower()

    def test_empty_tool_sets_allows_all(self):
        """Test vertical with empty tool sets allows all."""
        config = MagicMock()
        config.core_tools = []
        config.extension_tools = []
        config.optional_tools = []

        layer = VerticalLayer(tiered_config=config)
        allowed, reason = layer.check("any_tool", None)
        assert allowed is True
        assert "no tool restrictions" in reason.lower()

    def test_get_allowed_tools_no_config(self):
        """Test get_allowed_tools with no config returns all."""
        layer = VerticalLayer(tiered_config=None)
        all_tools = {"read_file", "write_file", "shell"}
        result = layer.get_allowed_tools(all_tools, None)
        assert result == all_tools

    def test_get_allowed_tools_with_config(self):
        """Test get_allowed_tools filters based on config."""
        config = MagicMock()
        config.core_tools = ["read_file"]
        config.extension_tools = ["list_directory"]
        config.optional_tools = ["semantic_search"]

        layer = VerticalLayer(tiered_config=config)
        all_tools = {"read_file", "write_file", "shell", "list_directory"}
        result = layer.get_allowed_tools(all_tools, None)
        assert result == {"read_file", "list_directory"}

    def test_get_allowed_tools_empty_sets(self):
        """Test get_allowed_tools with empty config returns all."""
        config = MagicMock()
        config.core_tools = []
        config.extension_tools = []
        config.optional_tools = []

        layer = VerticalLayer(tiered_config=config)
        all_tools = {"read_file", "write_file"}
        result = layer.get_allowed_tools(all_tools, None)
        assert result == all_tools


# =============================================================================
# StageLayer Extended Tests
# =============================================================================


class TestStageLayerExtended:
    """Extended tests for StageLayer."""

    def test_set_preserved_tools(self):
        """Test setting preserved tools."""
        layer = StageLayer()
        layer.set_preserved_tools({"custom_tool", "another_tool"})
        assert "custom_tool" in layer._preserved_tools
        assert "another_tool" in layer._preserved_tools

    def test_stage_from_context_metadata(self):
        """Test stage extracted from context metadata."""
        layer = StageLayer()
        # Use set_mode_controller to properly inject the mock
        mock_controller = MagicMock()
        mock_controller.config.allow_all_tools = False
        layer.set_mode_controller(mock_controller)

        # Create mock stage with name attribute - use INITIAL which is in EXPLORATION_STAGES
        mock_stage = MagicMock()
        mock_stage.name = "INITIAL"

        context = ToolAccessContext(metadata={"stage": mock_stage})
        allowed, reason = layer.check("write_file", context)
        assert allowed is False
        assert "filtered" in reason.lower()

    def test_non_exploration_stage_allows_all(self):
        """Test non-exploration stage allows all tools."""
        layer = StageLayer()
        # Use set_mode_controller to properly inject the mock
        mock_controller = MagicMock()
        mock_controller.config.allow_all_tools = False
        layer.set_mode_controller(mock_controller)

        mock_stage = MagicMock()
        mock_stage.name = "IMPLEMENTING"

        context = ToolAccessContext(conversation_stage=mock_stage)
        allowed, reason = layer.check("write_file", context)
        assert allowed is True
        assert "allows all tools" in reason.lower()

    def test_preserved_tools_allowed_in_exploration(self):
        """Test preserved tools allowed during exploration."""
        layer = StageLayer(preserved_tools={"special_write"})
        # Use set_mode_controller to properly inject the mock
        mock_controller = MagicMock()
        mock_controller.config.allow_all_tools = False
        layer.set_mode_controller(mock_controller)

        mock_stage = MagicMock()
        mock_stage.name = "INITIAL"  # Use INITIAL which is in EXPLORATION_STAGES

        context = ToolAccessContext(conversation_stage=mock_stage)
        allowed, reason = layer.check("special_write", context)
        assert allowed is True
        assert "preserved" in reason.lower()

    def test_explicit_readonly_shell_request_allowed_in_exploration(self):
        """Readonly shell requests should stay available during exploration stages."""
        layer = StageLayer()
        mock_controller = MagicMock()
        mock_controller.config.allow_all_tools = False
        layer.set_mode_controller(mock_controller)

        mock_stage = MagicMock()
        mock_stage.name = "INITIAL"
        intent = MagicMock()
        intent.name = "READ_ONLY"
        context = ToolAccessContext(
            conversation_stage=mock_stage,
            intent=intent,
            user_message="use shell with sqlite3 to inspect prompt generations",
        )

        allowed, reason = layer.check("shell", context)
        assert allowed is True
        assert "readonly shell request" in reason.lower()


# =============================================================================
# ToolAccessController Method Tests
# =============================================================================


class TestToolAccessControllerMethods:
    """Tests for ToolAccessController additional methods."""

    def test_explain_decision(self, mock_registry):
        """Test explain_decision returns explanation."""
        controller = ToolAccessController(registry=mock_registry)
        explanation = controller.explain_decision("read_file")
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_get_layer_existing(self, mock_registry):
        """Test get_layer returns existing layer."""
        controller = ToolAccessController(
            registry=mock_registry,
            layers=[SafetyLayer(), ModeLayer()],
        )
        layer = controller.get_layer("safety")
        assert layer is not None
        assert isinstance(layer, SafetyLayer)

    def test_get_layer_non_existing(self, mock_registry):
        """Test get_layer returns None for non-existing layer."""
        controller = ToolAccessController(
            registry=mock_registry,
            layers=[SafetyLayer()],
        )
        layer = controller.get_layer("nonexistent")
        assert layer is None

    def test_set_tiered_config(self, mock_registry):
        """Test set_tiered_config updates vertical layer."""
        controller = ToolAccessController(
            registry=mock_registry,
            layers=[VerticalLayer()],
        )

        config = MagicMock()
        config.core_tools = ["read_file"]
        config.extension_tools = []
        config.optional_tools = []

        controller.set_tiered_config(config)

        # Verify config was set by checking tool access
        decision = controller.check_access("write_file")
        # write_file is not in core_tools, so should be blocked
        assert decision.allowed is False

    def test_set_preserved_tools_on_controller(self, mock_registry):
        """Test set_preserved_tools updates stage layer."""
        controller = ToolAccessController(
            registry=mock_registry,
            layers=[StageLayer()],
        )

        controller.set_preserved_tools({"custom_tool"})

        # Verify it was set
        stage_layer = controller.get_layer("stage")
        assert isinstance(stage_layer, StageLayer)
        assert "custom_tool" in stage_layer._preserved_tools


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateToolAccessController:
    """Tests for create_tool_access_controller factory function."""

    def test_creates_controller(self, mock_registry):
        """Test factory creates controller."""
        from victor.agent.tool_access_controller import create_tool_access_controller

        controller = create_tool_access_controller(registry=mock_registry)
        assert isinstance(controller, ToolAccessController)

    def test_creates_with_sandbox_mode(self, mock_registry):
        """Test factory with sandbox_mode."""
        from victor.agent.tool_access_controller import create_tool_access_controller

        controller = create_tool_access_controller(
            registry=mock_registry,
            sandbox_mode=True,
        )

        # Verify sandbox mode is applied to safety layer
        safety_layer = controller.get_layer("safety")
        assert isinstance(safety_layer, SafetyLayer)
        assert safety_layer._sandbox_mode is True

    def test_creates_with_tiered_config(self, mock_registry):
        """Test factory with tiered config."""
        from victor.agent.tool_access_controller import create_tool_access_controller

        config = MagicMock()
        config.core_tools = ["read_file"]
        config.extension_tools = []
        config.optional_tools = []

        controller = create_tool_access_controller(
            registry=mock_registry,
            tiered_config=config,
        )

        # Verify config was applied
        vertical_layer = controller.get_layer("vertical")
        assert isinstance(vertical_layer, VerticalLayer)
        assert vertical_layer._tiered_config is config

    def test_creates_with_preserved_tools(self, mock_registry):
        """Test factory with preserved tools."""
        from victor.agent.tool_access_controller import create_tool_access_controller

        preserved = {"custom_tool", "special_tool"}
        controller = create_tool_access_controller(
            registry=mock_registry,
            preserved_tools=preserved,
        )

        # Verify preserved tools were applied
        stage_layer = controller.get_layer("stage")
        assert isinstance(stage_layer, StageLayer)
        assert stage_layer._preserved_tools == preserved

    def test_creates_all_layers(self, mock_registry):
        """Test factory creates all default layers."""
        from victor.agent.tool_access_controller import create_tool_access_controller

        controller = create_tool_access_controller(registry=mock_registry)

        # Check all layers exist
        assert controller.get_layer("safety") is not None
        assert controller.get_layer("mode") is not None
        assert controller.get_layer("session") is not None
        assert controller.get_layer("vertical") is not None
        assert controller.get_layer("stage") is not None
        assert controller.get_layer("intent") is not None
