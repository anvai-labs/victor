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

"""Unit tests for chat runtime protocol boundaries during service-first migration.

Tests:
- Protocol importability and structure
- runtime_checkable isinstance checks
- explicit legacy ChatCoordinator shim construction with mock orchestrator
- Protocol completeness (key attributes declared)
"""

import pytest
from typing import Protocol
from unittest.mock import MagicMock

# =============================================================================
# Protocol importability and structure
# =============================================================================


class TestProtocolImports:
    """Verify canonical service-hosted runtime protocols are importable."""

    def test_chat_context_protocol_importable(self):
        from victor.agent.services.protocols.chat_runtime import ChatContextProtocol

        assert issubclass(ChatContextProtocol, Protocol)

    def test_tool_context_protocol_importable(self):
        from victor.agent.services.protocols.chat_runtime import ToolContextProtocol

        assert issubclass(ToolContextProtocol, Protocol)

    def test_provider_context_protocol_importable(self):
        from victor.agent.services.protocols.chat_runtime import ProviderContextProtocol

        assert issubclass(ProviderContextProtocol, Protocol)

    def test_chat_orchestrator_protocol_importable(self):
        from victor.agent.services.protocols.chat_runtime import (
            ChatOrchestratorProtocol,
        )

        assert issubclass(ChatOrchestratorProtocol, Protocol)

    def test_chat_runtime_helper_access_protocol_importable(self):
        from victor.agent.services.protocols.chat_runtime import (
            ChatRuntimeHelperAccessProtocol,
        )

        assert issubclass(ChatRuntimeHelperAccessProtocol, Protocol)

    def test_planning_context_protocol_importable(self):
        from victor.agent.services.protocols.chat_runtime import PlanningContextProtocol

        assert issubclass(PlanningContextProtocol, Protocol)

    def test_chat_orchestrator_protocol_is_runtime_checkable(self):
        from victor.agent.services.protocols.chat_runtime import (
            ChatOrchestratorProtocol,
        )

        # runtime_checkable sets _is_runtime_protocol on the class
        assert getattr(ChatOrchestratorProtocol, "_is_runtime_protocol", False)

    def test_importable_from_service_protocol_package(self):
        from victor.agent.services.protocols import (
            ChatContextProtocol,
            ChatOrchestratorProtocol,
            ChatRuntimeHelperAccessProtocol,
            ProviderContextProtocol,
            ToolContextProtocol,
        )

        assert ChatContextProtocol is not None
        assert ChatOrchestratorProtocol is not None
        assert ChatRuntimeHelperAccessProtocol is not None
        assert ProviderContextProtocol is not None
        assert ToolContextProtocol is not None

    def test_importable_from_legacy_coordinator_module_warns(self):
        with pytest.warns(
            DeprecationWarning,
            match="victor.agent.coordinators.chat_protocols.ChatContextProtocol is deprecated",
        ):
            from victor.agent.coordinators.chat_protocols import (
                ChatContextProtocol,
            )

        with pytest.warns(
            DeprecationWarning,
            match=(
                "victor.agent.coordinators.chat_protocols.ChatOrchestratorProtocol is deprecated"
            ),
        ):
            from victor.agent.coordinators.chat_protocols import (
                ChatOrchestratorProtocol,
            )

        with pytest.warns(
            DeprecationWarning,
            match=(
                "victor.agent.coordinators.chat_protocols.ProviderContextProtocol is deprecated"
            ),
        ):
            from victor.agent.coordinators.chat_protocols import (
                ProviderContextProtocol,
            )

        with pytest.warns(
            DeprecationWarning,
            match="victor.agent.coordinators.chat_protocols.ToolContextProtocol is deprecated",
        ):
            from victor.agent.coordinators.chat_protocols import (
                ToolContextProtocol,
            )

        assert ChatContextProtocol is not None
        assert ChatOrchestratorProtocol is not None
        assert ProviderContextProtocol is not None
        assert ToolContextProtocol is not None

    def test_importable_from_coordinators_package(self):
        with pytest.warns(
            DeprecationWarning,
            match="victor.agent.coordinators.ChatContextProtocol is deprecated",
        ):
            from victor.agent.coordinators import (
                ChatContextProtocol,
            )

        with pytest.warns(
            DeprecationWarning,
            match="victor.agent.coordinators.ChatOrchestratorProtocol is deprecated",
        ):
            from victor.agent.coordinators import (
                ChatOrchestratorProtocol,
            )

        with pytest.warns(
            DeprecationWarning,
            match="victor.agent.coordinators.ProviderContextProtocol is deprecated",
        ):
            from victor.agent.coordinators import (
                ProviderContextProtocol,
            )

        with pytest.warns(
            DeprecationWarning,
            match="victor.agent.coordinators.ToolContextProtocol is deprecated",
        ):
            from victor.agent.coordinators import (
                ToolContextProtocol,
            )

        assert ChatContextProtocol is not None
        assert ChatOrchestratorProtocol is not None
        assert ProviderContextProtocol is not None
        assert ToolContextProtocol is not None


# =============================================================================
# Protocol completeness checks
# =============================================================================


class TestProtocolCompleteness:
    """Verify protocols declare the expected key attributes."""

    def test_chat_context_declares_conversation(self):
        from victor.agent.services.protocols.chat_runtime import ChatContextProtocol

        annotations = {}
        for cls in ChatContextProtocol.__mro__:
            annotations.update(getattr(cls, "__annotations__", {}))
        assert "conversation" in annotations
        assert "messages" in annotations
        assert "settings" in annotations
        assert "_system_added" in annotations
        assert "_cumulative_token_usage" in annotations

    def test_chat_context_declares_add_message(self):
        from victor.agent.services.protocols.chat_runtime import ChatContextProtocol

        assert hasattr(ChatContextProtocol, "add_message")

    def test_tool_context_declares_tool_members(self):
        from victor.agent.services.protocols.chat_runtime import ToolContextProtocol

        annotations = {}
        for cls in ToolContextProtocol.__mro__:
            annotations.update(getattr(cls, "__annotations__", {}))
        assert "tool_selector" in annotations
        assert "tool_budget" in annotations
        assert "tool_calls_used" in annotations
        assert "use_semantic_selection" in annotations

    def test_tool_context_declares_execute_tool_calls(self):
        from victor.agent.services.protocols.chat_runtime import ToolContextProtocol

        assert hasattr(ToolContextProtocol, "execute_tool_calls")

    def test_composite_protocol_declares_create_recovery_context(self):
        from victor.agent.services.protocols.chat_runtime import (
            ChatOrchestratorProtocol,
        )

        assert hasattr(ChatOrchestratorProtocol, "create_recovery_context")

    def test_planning_context_declares_planning_runtime_members(self):
        from victor.agent.services.protocols.chat_runtime import PlanningContextProtocol

        assert hasattr(PlanningContextProtocol, "profile")
        assert hasattr(PlanningContextProtocol, "planning_model_override")
        assert hasattr(PlanningContextProtocol, "_system_prompt_override")
        assert hasattr(PlanningContextProtocol, "set_system_prompt")

    def test_chat_runtime_helper_access_protocol_declares_canonical_helper_getters(
        self,
    ):
        from victor.agent.services.protocols.chat_runtime import (
            ChatRuntimeHelperAccessProtocol,
        )

        assert hasattr(ChatRuntimeHelperAccessProtocol, "_get_planning_chat_runtime")
        assert hasattr(ChatRuntimeHelperAccessProtocol, "_get_context_limit_runtime")

    def test_provider_context_declares_provider_members(self):
        from victor.agent.services.protocols.chat_runtime import ProviderContextProtocol

        annotations = {}
        for cls in ProviderContextProtocol.__mro__:
            annotations.update(getattr(cls, "__annotations__", {}))
        assert "provider" in annotations
        assert "model" in annotations
        assert "temperature" in annotations
        assert "max_tokens" in annotations

    def test_provider_context_declares_check_cancellation(self):
        from victor.agent.services.protocols.chat_runtime import ProviderContextProtocol

        assert hasattr(ProviderContextProtocol, "_check_cancellation")

    def test_composite_protocol_inherits_all_sub_protocols(self):
        from victor.agent.services.protocols.chat_runtime import (
            ChatContextProtocol,
            ChatOrchestratorProtocol,
            ProviderContextProtocol,
            ToolContextProtocol,
        )

        # Check MRO for sub-protocol inheritance (sub-protocols are not
        # runtime_checkable, so issubclass() won't work)
        mro = ChatOrchestratorProtocol.__mro__
        assert ChatContextProtocol in mro
        assert ToolContextProtocol in mro
        assert ProviderContextProtocol in mro

    def test_composite_protocol_declares_task_members(self):
        from victor.agent.services.protocols.chat_runtime import (
            ChatOrchestratorProtocol,
        )

        annotations = {}
        for cls in ChatOrchestratorProtocol.__mro__:
            annotations.update(getattr(cls, "__annotations__", {}))
        assert "task_classifier" in annotations
        assert "task_coordinator" in annotations
        assert "unified_tracker" in annotations

    def test_composite_protocol_declares_recovery_members(self):
        from victor.agent.services.protocols.chat_runtime import (
            ChatOrchestratorProtocol,
        )

        annotations = {}
        for cls in ChatOrchestratorProtocol.__mro__:
            annotations.update(getattr(cls, "__annotations__", {}))
        assert "_recovery_coordinator" in annotations
        assert "_recovery_integration" in annotations
        assert hasattr(ChatOrchestratorProtocol, "_record_runtime_intelligence_outcome")

    def test_composite_protocol_declares_presentation_members(self):
        from victor.agent.services.protocols.chat_runtime import (
            ChatOrchestratorProtocol,
        )

        annotations = {}
        for cls in ChatOrchestratorProtocol.__mro__:
            annotations.update(getattr(cls, "__annotations__", {}))
        assert "sanitizer" in annotations
        assert "response_completer" in annotations
        assert "debug_logger" in annotations
        assert "_chunk_generator" in annotations


# =============================================================================
# Mock isinstance validation
# =============================================================================


class TestChatCoordinatorWithMock:
    """Verify protocol isinstance checks work correctly."""

    def test_mock_missing_attribute_fails_isinstance(self):
        """A mock missing a protocol attribute fails isinstance."""
        from victor.agent.services.protocols.chat_runtime import (
            ChatOrchestratorProtocol,
        )

        class Incomplete:
            pass

        obj = Incomplete()
        assert not isinstance(obj, ChatOrchestratorProtocol)
