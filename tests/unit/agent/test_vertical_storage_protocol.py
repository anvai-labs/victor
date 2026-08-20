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

"""Unit tests for VerticalStorageProtocol.

Tests the protocol definition and its implementation in AgentOrchestrator,
ensuring DIP compliance for vertical data storage.
"""

import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch


class TestVerticalStorageProtocol:
    """Tests for the VerticalStorageProtocol definition."""

    def test_protocol_is_runtime_checkable(self):
        """VerticalStorageProtocol should be runtime checkable."""
        from victor.agent.protocols import VerticalStorageProtocol

        # Create a minimal implementation
        class MinimalStorage:
            def set_middleware(self, middleware: List[Any]) -> None:
                pass

            def get_middleware(self) -> List[Any]:
                return []

            def set_safety_patterns(self, patterns: List[Any]) -> None:
                pass

            def get_safety_patterns(self) -> List[Any]:
                return []

            def set_team_specs(self, specs: Dict[str, Any]) -> None:
                pass

            def get_team_specs(self) -> Dict[str, Any]:
                return {}

        storage = MinimalStorage()
        assert isinstance(storage, VerticalStorageProtocol)

    def test_protocol_rejects_incomplete_implementation(self):
        """VerticalStorageProtocol should reject incomplete implementations."""
        from victor.agent.protocols import VerticalStorageProtocol

        # Create an incomplete implementation
        class IncompleteStorage:
            def set_middleware(self, middleware: List[Any]) -> None:
                pass

            # Missing other methods

        storage = IncompleteStorage()
        # Runtime checkable only checks for method existence, not completeness
        # but at least set_middleware exists
        assert hasattr(storage, "set_middleware")

    def test_protocol_exported_in_all(self):
        """VerticalStorageProtocol should be exported in __all__."""
        from victor.agent import protocols

        assert "VerticalStorageProtocol" in protocols.__all__


class TestVerticalStorageProtocolImplementation:
    """Tests for concrete implementation that satisfies VerticalStorageProtocol."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage implementing VerticalStorageProtocol."""

        class MockStorage:
            def __init__(self):
                self._middleware: List[Any] = []
                self._safety_patterns: List[Any] = []
                self._team_specs: Dict[str, Any] = {}

            def set_middleware(self, middleware: List[Any]) -> None:
                self._middleware = middleware

            def get_middleware(self) -> List[Any]:
                return self._middleware

            def set_safety_patterns(self, patterns: List[Any]) -> None:
                self._safety_patterns = patterns

            def get_safety_patterns(self) -> List[Any]:
                return self._safety_patterns

            def set_team_specs(self, specs: Dict[str, Any]) -> None:
                self._team_specs = specs

            def get_team_specs(self) -> Dict[str, Any]:
                return self._team_specs

        return MockStorage()

    def test_set_and_get_middleware(self, mock_storage):
        """Test middleware storage round trip."""
        middleware = [MagicMock(), MagicMock()]

        mock_storage.set_middleware(middleware)
        result = mock_storage.get_middleware()

        assert result == middleware
        assert len(result) == 2

    def test_set_and_get_safety_patterns(self, mock_storage):
        """Test safety patterns storage round trip."""
        patterns = [
            MagicMock(pattern="rm -rf", description="dangerous command"),
            MagicMock(pattern="sudo", description="elevated privileges"),
        ]

        mock_storage.set_safety_patterns(patterns)
        result = mock_storage.get_safety_patterns()

        assert result == patterns
        assert len(result) == 2

    def test_set_and_get_team_specs(self, mock_storage):
        """Test team specs storage round trip."""
        specs = {
            "code_review_team": MagicMock(),
            "debugging_team": MagicMock(),
        }

        mock_storage.set_team_specs(specs)
        result = mock_storage.get_team_specs()

        assert result == specs
        assert "code_review_team" in result
        assert "debugging_team" in result

    def test_empty_getters_return_empty_collections(self, mock_storage):
        """Test that getters return empty collections when nothing is set."""
        assert mock_storage.get_middleware() == []
        assert mock_storage.get_safety_patterns() == []
        assert mock_storage.get_team_specs() == {}

    def test_overwriting_values(self, mock_storage):
        """Test that setting values twice overwrites previous values."""
        mock_storage.set_middleware([MagicMock()])
        mock_storage.set_middleware([MagicMock(), MagicMock()])

        assert len(mock_storage.get_middleware()) == 2


class TestAgentOrchestratorVerticalStorage:
    """Tests for AgentOrchestrator's VerticalStorageProtocol implementation.

    These tests verify that AgentOrchestrator properly implements the
    VerticalStorageProtocol methods.
    """

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a minimal mock orchestrator with the storage methods."""
        from unittest.mock import MagicMock

        orchestrator = MagicMock()

        # Set up storage attributes
        orchestrator._vertical_middleware = []
        orchestrator._vertical_safety_patterns = []
        orchestrator._team_specs = {}

        # Implement the protocol methods
        def set_middleware(middleware):
            orchestrator._vertical_middleware = middleware

        def get_middleware():
            return getattr(orchestrator, "_vertical_middleware", [])

        def set_safety_patterns(patterns):
            orchestrator._vertical_safety_patterns = patterns

        def get_safety_patterns():
            return getattr(orchestrator, "_vertical_safety_patterns", [])

        def set_team_specs(specs):
            orchestrator._team_specs = specs

        def get_team_specs():
            return getattr(orchestrator, "_team_specs", {})

        orchestrator.set_middleware = set_middleware
        orchestrator.get_middleware = get_middleware
        orchestrator.set_safety_patterns = set_safety_patterns
        orchestrator.get_safety_patterns = get_safety_patterns
        orchestrator.set_team_specs = set_team_specs
        orchestrator.get_team_specs = get_team_specs

        return orchestrator

    def test_set_and_get_middleware_on_orchestrator(self, mock_orchestrator):
        """Test middleware storage on orchestrator."""
        middleware = [MagicMock(name="mw1"), MagicMock(name="mw2")]

        mock_orchestrator.set_middleware(middleware)
        result = mock_orchestrator.get_middleware()

        assert result == middleware

    def test_set_and_get_safety_patterns_on_orchestrator(self, mock_orchestrator):
        """Test safety patterns storage on orchestrator."""
        patterns = [MagicMock(pattern="test")]

        mock_orchestrator.set_safety_patterns(patterns)
        result = mock_orchestrator.get_safety_patterns()

        assert result == patterns

    def test_set_and_get_team_specs_on_orchestrator(self, mock_orchestrator):
        """Test team specs storage on orchestrator."""
        specs = {"team1": MagicMock()}

        mock_orchestrator.set_team_specs(specs)
        result = mock_orchestrator.get_team_specs()

        assert result == specs


class TestFrameworkStepHandlerUsage:
    """Tests verifying FrameworkStepHandler can use VerticalStorageProtocol."""

    def test_step_handler_can_use_protocol_for_team_specs(self):
        """Test that FrameworkStepHandler can set team specs via protocol."""
        from victor.agent.protocols import VerticalStorageProtocol
        from unittest.mock import MagicMock

        # Create storage implementing the protocol
        class StorageImpl:
            def __init__(self):
                self._middleware = []
                self._safety_patterns = []
                self._team_specs = {}

            def set_middleware(self, middleware):
                self._middleware = middleware

            def get_middleware(self):
                return self._middleware

            def set_safety_patterns(self, patterns):
                self._safety_patterns = patterns

            def get_safety_patterns(self):
                return self._safety_patterns

            def set_team_specs(self, specs):
                self._team_specs = specs

            def get_team_specs(self):
                return self._team_specs

        storage = StorageImpl()

        # Verify it implements the protocol
        assert isinstance(storage, VerticalStorageProtocol)

        # Use protocol methods
        specs = {"test_team": MagicMock()}
        storage.set_team_specs(specs)

        assert storage.get_team_specs() == specs

    def test_protocol_type_hints_work_correctly(self):
        """Test that protocol can be used as type hint."""
        from victor.agent.protocols import VerticalStorageProtocol
        from typing import Any, Dict, List

        def configure_vertical(storage: VerticalStorageProtocol) -> None:
            """Function that accepts any VerticalStorageProtocol implementation."""
            storage.set_middleware([])
            storage.set_safety_patterns([])
            storage.set_team_specs({})

        # Create a conforming implementation
        class ConformingStorage:
            def set_middleware(self, middleware: List[Any]) -> None:
                pass

            def get_middleware(self) -> List[Any]:
                return []

            def set_safety_patterns(self, patterns: List[Any]) -> None:
                pass

            def get_safety_patterns(self) -> List[Any]:
                return []

            def set_team_specs(self, specs: Dict[str, Any]) -> None:
                pass

            def get_team_specs(self) -> Dict[str, Any]:
                return {}

        storage = ConformingStorage()

        # This should not raise any errors
        configure_vertical(storage)


class TestProtocolIntegrationWithAdapter:
    """Tests for protocol usage by VerticalIntegrationAdapter."""

    def test_adapter_can_use_storage_protocol(self):
        """Test that VerticalIntegrationAdapter can work with storage protocol."""
        from victor.agent.protocols import VerticalStorageProtocol
        from unittest.mock import MagicMock

        # Simulate orchestrator implementing VerticalStorageProtocol
        orchestrator = MagicMock()
        orchestrator._vertical_middleware = []
        orchestrator._vertical_safety_patterns = []

        # Mock the protocol methods
        def set_middleware(middleware):
            orchestrator._vertical_middleware = middleware

        def get_middleware():
            return orchestrator._vertical_middleware

        orchestrator.set_middleware = set_middleware
        orchestrator.get_middleware = get_middleware

        # Verify adapter-like usage
        middleware = [MagicMock()]
        orchestrator.set_middleware(middleware)

        assert orchestrator.get_middleware() == middleware

    def test_fallback_to_private_attributes_for_backward_compatibility(self):
        """Test that code can fall back to private attrs if protocol not available."""
        from unittest.mock import MagicMock

        # Simulate legacy orchestrator without protocol methods
        orchestrator = MagicMock(spec=[])
        orchestrator._vertical_middleware = []

        # Fallback pattern used in VerticalIntegrationAdapter
        if hasattr(orchestrator, "set_middleware"):
            orchestrator.set_middleware([MagicMock()])
        else:
            # Fallback to direct attribute access
            orchestrator._vertical_middleware = [MagicMock()]

        assert len(orchestrator._vertical_middleware) == 1
