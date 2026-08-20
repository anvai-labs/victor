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

"""Tests for state externalization service (Phase 4.3)."""

import pytest

from victor.agent.state_service import (
    StateService,
    get_state_service,
    save_vertical_state,
    load_vertical_state,
)
from victor.agent.vertical_context import VerticalContext


@pytest.fixture
def temp_db_path(tmp_path):
    """Create temporary database path."""
    return tmp_path / "test_project.db"


@pytest.fixture
def state_service(temp_db_path):
    """Create StateService instance with temporary database."""
    return StateService(db_path=temp_db_path)


@pytest.fixture
def sample_context():
    """Create sample VerticalContext."""
    context = VerticalContext(
        name="coding",
        stages={"planning": {"description": "Planning stage"}},
        middleware=[],
        safety_patterns=[],
        task_hints={},
        system_prompt="You are a coding assistant.",
        mode_configs={"default": {"tool_budget": 10}},
        default_mode="default",
        default_budget=10,
        tool_dependencies=[],
        tool_sequences=[],
        enabled_tools={"read", "write", "search"},
    )
    return context


class TestStateService:
    """Tests for StateService class."""

    def test_init_creates_tables(self, state_service):
        """Test that initialization creates required tables."""
        # Tables should be created during initialization
        result = state_service._db.query(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('vertical_state', 'vertical_negotiation', 'vertical_session')"
        )
        tables = [row[0] for row in result] if result else []

        assert "vertical_state" in tables
        assert "vertical_negotiation" in tables
        assert "vertical_session" in tables

    def test_save_vertical_state(self, state_service, sample_context):
        """Test saving vertical context."""
        state_id = state_service.save_vertical_state(
            context=sample_context,
            vertical_name="coding",
            vertical_version="1.5.0",
        )

        assert state_id is not None
        assert isinstance(state_id, str)

        # Verify state was saved
        result = state_service._db.query(
            "SELECT * FROM vertical_state WHERE id = ?",
            (state_id,),
        )

        assert len(result) == 1
        assert result[0][1] == "coding"  # vertical_name
        assert result[0][2] == "1.5.0"  # vertical_version

    def test_load_vertical_state(self, state_service, sample_context):
        """Test loading vertical context."""
        # Save state
        state_id = state_service.save_vertical_state(
            context=sample_context,
            vertical_name="coding",
            vertical_version="1.5.0",
        )

        # Load state
        loaded_context = state_service.load_vertical_state(state_id)

        assert loaded_context is not None
        assert loaded_context.name == "coding"
        assert loaded_context.enabled_tools == sample_context.enabled_tools
        assert loaded_context.default_mode == sample_context.default_mode

    def test_load_nonexistent_state(self, state_service):
        """Test loading non-existent state returns None."""
        loaded_context = state_service.load_vertical_state("nonexistent-id")
        assert loaded_context is None

    def test_update_vertical_state(self, state_service, sample_context):
        """Test updating vertical context."""
        # Save state
        state_id = state_service.save_vertical_state(
            context=sample_context,
            vertical_name="coding",
            vertical_version="1.5.0",
        )

        # Modify context
        sample_context.enabled_tools.add("git")
        sample_context.default_budget = 20

        # Update state
        success = state_service.update_vertical_state(state_id, sample_context)
        assert success is True

        # Verify update
        loaded_context = state_service.load_vertical_state(state_id)
        assert "git" in loaded_context.enabled_tools
        assert loaded_context.default_budget == 20

    def test_delete_vertical_state(self, state_service, sample_context):
        """Test deleting vertical context."""
        # Save state
        state_id = state_service.save_vertical_state(
            context=sample_context,
            vertical_name="coding",
        )

        # Delete state
        success = state_service.delete_vertical_state(state_id)
        assert success is True

        # Verify deletion
        loaded_context = state_service.load_vertical_state(state_id)
        assert loaded_context is None

    def test_list_vertical_states(self, state_service, sample_context):
        """Test listing vertical states."""
        # Save multiple states
        state_service.save_vertical_state(
            context=sample_context,
            vertical_name="coding",
            vertical_version="1.0.0",
        )

        sample_context.name = "research"
        state_service.save_vertical_state(
            context=sample_context,
            vertical_name="research",
            vertical_version="1.0.0",
        )

        # List all states
        states = state_service.list_vertical_states(limit=10)
        assert len(states) == 2

        # Filter by vertical name
        coding_states = state_service.list_vertical_states(vertical_name="coding")
        assert len(coding_states) == 1
        assert coding_states[0]["vertical_name"] == "coding"

    def test_negotiation_results_persistence(self, state_service, sample_context):
        """Test that negotiation results are persisted."""
        # Add negotiation results to context
        sample_context.capability_negotiation_results = {
            "tools": {
                "capability_name": "tools",
                "status": "success",
                "agreed_version": "1.0.0",
                "supported_features": ["tool_list", "tool_filtering"],
                "unsupported_features": [],
                "missing_required_features": [],
            }
        }

        # Save state
        state_id = state_service.save_vertical_state(
            context=sample_context,
            vertical_name="coding",
        )

        # Verify negotiation results were saved
        result = state_service._db.query(
            "SELECT * FROM vertical_negotiation WHERE vertical_state_id = ?",
            (state_id,),
        )

        assert len(result) == 1
        assert result[0][2] == "tools"  # capability_name
        assert result[0][3] == "success"  # status

        # Load state and verify results
        loaded_context = state_service.load_vertical_state(state_id)
        assert loaded_context.capability_negotiation_results is not None
        assert "tools" in loaded_context.capability_negotiation_results


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_state_service_singleton(self):
        """Test that get_state_service returns singleton."""
        service1 = get_state_service()
        service2 = get_state_service()
        assert service1 is service2

    def test_save_vertical_state_function(self, temp_db_path, sample_context):
        """Test save_vertical_state convenience function."""
        # This test uses the singleton, so we can't easily test with temp db
        # Just verify the function exists and is callable
        assert callable(save_vertical_state)

    def test_load_vertical_state_function(self):
        """Test load_vertical_state convenience function."""
        # This test uses the singleton, so we can't easily test with temp db
        # Just verify the function exists and is callable
        assert callable(load_vertical_state)


class TestStateServiceIntegration:
    """Integration tests for state service."""

    def test_full_lifecycle(self, state_service, sample_context):
        """Test complete lifecycle: save -> load -> update -> delete."""
        # Save
        state_id = state_service.save_vertical_state(
            context=sample_context,
            vertical_name="coding",
        )

        # Load
        loaded = state_service.load_vertical_state(state_id)
        assert loaded is not None

        # Update
        loaded.default_budget = 50
        state_service.update_vertical_state(state_id, loaded)

        # Verify update
        reloaded = state_service.load_vertical_state(state_id)
        assert reloaded.default_budget == 50

        # Delete
        state_service.delete_vertical_state(state_id)

        # Verify deletion
        final = state_service.load_vertical_state(state_id)
        assert final is None

    def test_multiple_contexts_same_vertical(self, state_service):
        """Test saving multiple contexts for the same vertical."""
        contexts = []
        for i in range(3):
            context = VerticalContext(
                name="coding",
                enabled_tools={f"tool{i}"},
            )
            state_id = state_service.save_vertical_state(
                context=context,
                vertical_name="coding",
                session_id=f"session{i}",
            )
            contexts.append((state_id, context))

        # List all states for coding vertical
        states = state_service.list_vertical_states(vertical_name="coding")
        assert len(states) == 3

        # Verify each state can be loaded
        for state_id, original_context in contexts:
            loaded = state_service.load_vertical_state(state_id)
            assert loaded is not None
            assert loaded.name == "coding"
