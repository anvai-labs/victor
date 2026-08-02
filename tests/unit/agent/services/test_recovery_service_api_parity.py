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

"""TDD tests for RecoveryService API parity with StreamingRecoveryCoordinator.

This test file validates that RecoveryService has complete API parity with
the deprecated StreamingRecoveryCoordinator, enabling safe migration.

Migration strategy:
1. Validate RecoveryService has all required methods
2. Validate RecoveryService produces same results when has_native_streaming_runtime() is True
3. Migrate calling sites from StreamingRecoveryCoordinator to RecoveryService
4. Remove recovery_compat.py
"""

import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Any, Dict, List, Optional

import pytest


class TestRecoveryServiceAPIParity:
    """Validate RecoveryService has complete API parity with StreamingRecoveryCoordinator."""

    def test_recovery_service_has_all_required_methods(self):
        """RecoveryService must have all methods called by orchestrator."""
        from victor.agent.services.recovery_service import RecoveryService

        service = RecoveryService()

        # Methods called by orchestrator and other components
        required_methods = [
            # Streaming recovery methods
            "check_natural_completion",
            "handle_empty_response",
            "get_recovery_fallback_message",
            "check_tool_budget",
            "truncate_tool_calls",
            "filter_blocked_tool_calls",
            "check_blocked_threshold",
            "check_force_action",
            # Integration methods
            "handle_recovery_with_integration",
            "apply_recovery_action",
            # Runtime binding
            "bind_runtime_components",
            "has_native_streaming_runtime",
        ]

        for method_name in required_methods:
            assert hasattr(service, method_name), f"RecoveryService missing method: {method_name}"
            assert callable(getattr(service, method_name)), f"{method_name} must be callable"

    def test_recovery_service_has_native_streaming_runtime_returns_false_initially(
        self,
    ):
        """has_native_streaming_runtime() must return False before binding components."""
        from victor.agent.services.recovery_service import RecoveryService

        service = RecoveryService()
        assert (
            service.has_native_streaming_runtime() is False
        ), "RecoveryService should not have native streaming runtime until components are bound"

    def test_recovery_service_has_native_streaming_runtime_returns_true_after_binding(
        self,
    ):
        """has_native_streaming_runtime() must return True after binding required components."""
        from victor.agent.services.recovery_service import RecoveryService

        service = RecoveryService()

        # Create mock components
        mock_streaming_handler = Mock()
        mock_settings = Mock()
        mock_unified_tracker = Mock()
        mock_recovery_integration = Mock()

        # Bind required components
        service.bind_runtime_components(
            streaming_handler=mock_streaming_handler,
            settings=mock_settings,
            unified_tracker=mock_unified_tracker,
            recovery_integration=mock_recovery_integration,
        )

        assert (
            service.has_native_streaming_runtime() is True
        ), "RecoveryService should have native streaming runtime after binding required components"

    def test_bind_runtime_components_accepts_all_collaborators(self):
        """bind_runtime_components() must accept all optional collaborators."""
        from victor.agent.services.recovery_service import RecoveryService

        service = RecoveryService()

        # All possible collaborators
        all_components = {
            "recovery_coordinator": Mock(),
            "recovery_handler": Mock(),
            "recovery_integration": Mock(),
            "streaming_handler": Mock(),
            "context_compactor": Mock(),
            "unified_tracker": Mock(),
            "settings": Mock(),
            "event_bus": Mock(),
            "presentation": Mock(),
        }

        # Should not raise any errors
        service.bind_runtime_components(**all_components)

        # Verify components are stored
        assert service._recovery_coordinator is not None
        assert service._recovery_handler is not None
        assert service._recovery_integration is not None
        assert service._streaming_handler is not None
        assert service._context_compactor is not None
        assert service._unified_tracker is not None
        assert service._settings is not None
        assert service._event_bus is not None
        assert service._presentation is not None


class TestRecoveryServiceDelegationBehavior:
    """Validate RecoveryService delegation behavior when native runtime is enabled."""

    @pytest.fixture
    def recovery_service_with_native_runtime(self):
        """Create RecoveryService with native streaming runtime enabled."""
        from victor.agent.services.recovery_service import RecoveryService

        service = RecoveryService()

        # Create mock components
        mock_streaming_handler = Mock()
        mock_settings = Mock()
        mock_unified_tracker = Mock()
        mock_recovery_integration = Mock()
        mock_context_compactor = Mock()
        mock_presentation = Mock()
        mock_event_bus = Mock()

        # Configure mocks
        mock_presentation.icon = Mock(return_value="icon")
        mock_settings.recovery_blocked_consecutive_threshold = 3
        mock_settings.recovery_blocked_total_threshold = 5

        # Bind components
        service.bind_runtime_components(
            streaming_handler=mock_streaming_handler,
            settings=mock_settings,
            unified_tracker=mock_unified_tracker,
            recovery_integration=mock_recovery_integration,
            context_compactor=mock_context_compactor,
            presentation=mock_presentation,
            event_bus=mock_event_bus,
        )

        return service, {
            "streaming_handler": mock_streaming_handler,
            "settings": mock_settings,
            "unified_tracker": mock_unified_tracker,
            "recovery_integration": mock_recovery_integration,
            "context_compactor": mock_context_compactor,
            "presentation": mock_presentation,
            "event_bus": mock_event_bus,
        }

    def test_check_natural_completion_uses_native_runtime(
        self, recovery_service_with_native_runtime
    ):
        """check_natural_completion() should use streaming_handler when native runtime is enabled."""
        service, mocks = recovery_service_with_native_runtime

        # Setup mock
        mock_result = Mock()
        mock_result.chunks = []
        mocks["streaming_handler"].check_natural_completion = Mock(return_value=mock_result)

        # Create mock context
        mock_ctx = Mock()
        mock_ctx.streaming_context = Mock()

        # Call method
        result = service.check_natural_completion(
            mock_ctx, has_tool_calls=False, content_length=100
        )

        # Verify delegation to streaming_handler
        mocks["streaming_handler"].check_natural_completion.assert_called_once_with(
            mock_ctx.streaming_context, False, 100
        )

    def test_handle_empty_response_uses_native_runtime(self, recovery_service_with_native_runtime):
        """handle_empty_response() should use streaming_handler when native runtime is enabled."""
        service, mocks = recovery_service_with_native_runtime

        # Setup mock
        mock_result = Mock()
        mock_result.chunks = [Mock(content="error")]
        mock_ctx = Mock()
        mock_ctx.streaming_context = Mock()
        mock_ctx.streaming_context.force_completion = False
        mocks["streaming_handler"].handle_empty_response = Mock(return_value=mock_result)

        # Call method
        chunk, should_complete = service.handle_empty_response(mock_ctx)

        # Verify delegation to streaming_handler
        mocks["streaming_handler"].handle_empty_response.assert_called_once()
        assert should_complete is False

    def test_check_tool_budget_uses_native_runtime(self, recovery_service_with_native_runtime):
        """check_tool_budget() should use streaming_handler when native runtime is enabled."""
        service, mocks = recovery_service_with_native_runtime

        # Setup mock - no budget warning
        mocks["streaming_handler"].check_tool_budget = Mock(return_value=None)

        # Create mock context
        mock_ctx = Mock()

        # Call method
        result = service.check_tool_budget(mock_ctx, warning_threshold=250)

        # Verify delegation to streaming_handler
        mocks["streaming_handler"].check_tool_budget.assert_called_once_with(
            mock_ctx.streaming_context,
            250,
            0.8,
            5,
        )
        assert result is None

    def test_filter_blocked_tool_calls_uses_native_runtime(
        self, recovery_service_with_native_runtime
    ):
        """filter_blocked_tool_calls() should use streaming_handler when native runtime is enabled."""
        service, mocks = recovery_service_with_native_runtime

        # Setup mock
        tool_calls = [{"name": "test_tool", "args": {}}]
        mocks["unified_tracker"].is_blocked_after_warning = Mock(return_value=False)
        mocks["streaming_handler"].filter_blocked_tool_calls = Mock(
            return_value=(tool_calls, [], 0)
        )

        # Create mock context
        mock_ctx = Mock()

        # Call method
        filtered, chunks, count = service.filter_blocked_tool_calls(mock_ctx, tool_calls)

        # Verify delegation to streaming_handler
        mocks["streaming_handler"].filter_blocked_tool_calls.assert_called_once()
        assert filtered == tool_calls
        assert count == 0

    def test_check_blocked_threshold_uses_native_runtime(
        self, recovery_service_with_native_runtime
    ):
        """check_blocked_threshold() should use streaming_handler when native runtime is enabled."""
        service, mocks = recovery_service_with_native_runtime

        # Setup mock - no threshold exceeded
        mocks["streaming_handler"].check_blocked_threshold = Mock(return_value=None)

        # Create mock context
        mock_ctx = Mock()

        # Call method
        result = service.check_blocked_threshold(mock_ctx, all_blocked=True)

        # Verify delegation to streaming_handler
        mocks["streaming_handler"].check_blocked_threshold.assert_called_once()
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_recovery_with_integration_uses_native_runtime(
        self, recovery_service_with_native_runtime
    ):
        """handle_recovery_with_integration() should use recovery_integration when available."""
        from victor.agent.orchestrator_recovery import OrchestratorRecoveryAction

        service, mocks = recovery_service_with_native_runtime

        # Setup mock
        mocks["recovery_integration"].enabled = True
        expected_action = OrchestratorRecoveryAction(action="continue", reason="test")
        mocks["recovery_integration"].handle_response = AsyncMock(return_value=expected_action)

        # Create mock context
        mock_ctx = Mock()
        mock_ctx.provider_name = "anthropic"
        mock_ctx.model = "claude-opus-4-6"
        mock_ctx.tool_calls_used = 5
        mock_ctx.tool_budget = 100
        mock_ctx.streaming_context = Mock()
        mock_ctx.streaming_context.total_iterations = 1
        mock_ctx.streaming_context.max_total_iterations = 200
        mock_ctx.temperature = 0.7
        mock_ctx.last_quality_score = 0.8
        mock_ctx.unified_task_type = Mock(value="analysis")
        mock_ctx.is_analysis_task = True
        mock_ctx.is_action_task = False

        # Call method
        result = await service.handle_recovery_with_integration(
            mock_ctx,
            full_content="test",
            tool_calls=None,
            mentioned_tools=None,
        )

        # Verify delegation to recovery_integration
        mocks["recovery_integration"].handle_response.assert_called_once()
        assert result.action == "continue"

    def test_apply_recovery_action_handles_continue(self, recovery_service_with_native_runtime):
        """apply_recovery_action() should return None for continue action."""
        from victor.agent.orchestrator_recovery import OrchestratorRecoveryAction

        service, mocks = recovery_service_with_native_runtime

        action = OrchestratorRecoveryAction(action="continue", reason="test")
        mock_ctx = Mock()

        result = service.apply_recovery_action(action, mock_ctx)

        assert result is None

    def test_apply_recovery_action_handles_retry_with_message(
        self, recovery_service_with_native_runtime
    ):
        """apply_recovery_action() should add user message for retry action."""
        from victor.agent.orchestrator_recovery import OrchestratorRecoveryAction

        service, mocks = recovery_service_with_native_runtime

        action = OrchestratorRecoveryAction(
            action="retry",
            reason="test",
            message="Please try again",
            new_temperature=0.8,
        )
        mock_ctx = Mock()
        mock_ctx.temperature = 0.7
        message_adder = Mock()

        result = service.apply_recovery_action(action, mock_ctx, message_adder=message_adder)

        assert result is None
        message_adder.assert_called_once_with("user", "Please try again")

    def test_apply_recovery_action_handles_force_summary(
        self, recovery_service_with_native_runtime
    ):
        """apply_recovery_action() should set force_completion for force_summary action."""
        from victor.agent.orchestrator_recovery import OrchestratorRecoveryAction

        service, mocks = recovery_service_with_native_runtime

        action = OrchestratorRecoveryAction(action="force_summary", reason="test")
        mock_ctx = Mock()
        mock_ctx.streaming_context = Mock()
        mock_ctx.streaming_context.force_completion = False
        message_adder = Mock()

        result = service.apply_recovery_action(action, mock_ctx, message_adder=message_adder)

        assert result is None
        assert mock_ctx.streaming_context.force_completion is True
        message_adder.assert_called_once()

    def test_apply_recovery_action_handles_abort(self, recovery_service_with_native_runtime):
        """apply_recovery_action() should return StreamChunk for abort action."""
        from victor.agent.orchestrator_recovery import OrchestratorRecoveryAction
        from victor.providers.base import StreamChunk

        service, mocks = recovery_service_with_native_runtime

        action = OrchestratorRecoveryAction(
            action="abort", reason="Critical error", failure_type="auth"
        )
        mock_ctx = Mock()
        mock_ctx.iteration = 5

        result = service.apply_recovery_action(action, mock_ctx)

        assert isinstance(result, StreamChunk)
        assert result.is_final is True
        assert "aborted" in result.content.lower()

    def test_truncate_tool_calls_enforces_budget(self, recovery_service_with_native_runtime):
        """truncate_tool_calls() should enforce budget limit."""
        service, mocks = recovery_service_with_native_runtime

        tool_calls = [{"name": f"tool_{i}", "args": {}} for i in range(10)]

        truncated, was_truncated = service.truncate_tool_calls(Mock(), tool_calls, max_calls=5)

        assert len(truncated) == 5
        assert was_truncated is True

    def test_truncate_tool_calls_no_truncation_needed(self, recovery_service_with_native_runtime):
        """truncate_tool_calls() should not truncate when under budget."""
        service, mocks = recovery_service_with_native_runtime

        tool_calls = [{"name": f"tool_{i}", "args": {}} for i in range(3)]

        truncated, was_truncated = service.truncate_tool_calls(Mock(), tool_calls, max_calls=5)

        assert len(truncated) == 3
        assert was_truncated is False


class TestRecoveryServiceFallbackBehavior:
    """Validate RecoveryService fallback when native runtime is disabled."""

    def test_check_natural_completion_falls_back_to_coordinator(self):
        """check_natural_completion() should fall back to recovery_coordinator when native runtime disabled."""
        from victor.agent.services.recovery_service import RecoveryService

        service = RecoveryService()

        # Create mock recovery_coordinator but don't bind native runtime
        mock_coordinator = Mock()
        mock_result = Mock(content="result", is_final=True)
        mock_coordinator.check_natural_completion = Mock(return_value=mock_result)

        service.bind_runtime_components(recovery_coordinator=mock_coordinator)

        mock_ctx = Mock()

        # Call method
        result = service.check_natural_completion(mock_ctx, False, 100)

        # Verify fallback to coordinator
        mock_coordinator.check_natural_completion.assert_called_once_with(mock_ctx, False, 100)

    def test_handle_empty_response_falls_back_to_coordinator(self):
        """handle_empty_response() should fall back to recovery_coordinator when native runtime disabled."""
        from victor.agent.services.recovery_service import RecoveryService

        service = RecoveryService()

        # Create mock recovery_coordinator
        mock_coordinator = Mock()
        mock_coordinator.handle_empty_response = Mock(return_value=(Mock(content="err"), False))

        service.bind_runtime_components(recovery_coordinator=mock_coordinator)

        mock_ctx = Mock()

        # Call method
        result, should_complete = service.handle_empty_response(mock_ctx)

        # Verify fallback to coordinator
        mock_coordinator.handle_empty_response.assert_called_once_with(mock_ctx)
        assert should_complete is False


class TestRecoveryServiceUtilities:
    """Test RecoveryService utility methods."""

    def test_get_recovery_fallback_message_returns_default(self):
        """get_recovery_fallback_message() should return default message when no coordinator bound."""
        from victor.agent.services.recovery_service import RecoveryService

        service = RecoveryService()
        mock_ctx = Mock()

        message = service.get_recovery_fallback_message(mock_ctx)

        assert "apologize" in message.lower() or "summary" in message.lower()

    def test_get_recovery_fallback_message_delegates_to_coordinator(self):
        """get_recovery_fallback_message() should delegate to coordinator when bound."""
        from victor.agent.services.recovery_service import RecoveryService

        service = RecoveryService()

        mock_coordinator = Mock()
        mock_coordinator.get_recovery_fallback_message = Mock(return_value="coordinator fallback")
        service.bind_runtime_components(recovery_coordinator=mock_coordinator)

        mock_ctx = Mock()

        message = service.get_recovery_fallback_message(mock_ctx)

        assert message == "coordinator fallback"

    def test_check_force_action_returns_false_none(self):
        """check_force_action() should return (False, None) by default."""
        from victor.agent.services.recovery_service import RecoveryService

        service = RecoveryService()
        mock_ctx = Mock()

        should_force, action_type = service.check_force_action(mock_ctx)

        assert should_force is False
        assert action_type is None
