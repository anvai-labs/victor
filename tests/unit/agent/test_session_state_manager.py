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

"""Tests for SessionStateManager.

Tests the session state management functionality including:
- ExecutionState dataclass serialization
- SessionFlags dataclass serialization
- Tool call tracking and budget management
- File observation and read tracking
- Failed tool signature tracking
- Token usage tracking
- Checkpoint/restore operations
- Session summary generation
"""

import pytest

from victor.agent.session_state_manager import (
    ExecutionState,
    SessionFlags,
    SessionStateManager,
    create_session_state_manager,
)


class TestExecutionState:
    """Tests for ExecutionState dataclass."""

    def test_default_initialization(self):
        """Test default values for ExecutionState."""
        state = ExecutionState()

        assert state.tool_calls_used == 0
        assert state.observed_files == set()
        assert state.executed_tools == []
        assert state.failed_tool_signatures == set()
        assert state.read_files_session == set()
        assert state.required_files == []
        assert state.required_outputs == []
        assert state.token_usage["prompt_tokens"] == 0
        assert state.token_usage["completion_tokens"] == 0
        assert state.token_usage["total_tokens"] == 0

    def test_to_dict(self):
        """Test ExecutionState serialization."""
        state = ExecutionState(
            tool_calls_used=5,
            observed_files={"file1.py", "file2.py"},
            executed_tools=["read", "edit", "read"],
            failed_tool_signatures={("edit", "abc123")},
            read_files_session={"file1.py"},
            required_files=["main.py"],
            required_outputs=["analysis"],
            token_usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 20,
            },
        )

        data = state.to_dict()

        assert data["tool_calls_used"] == 5
        assert set(data["observed_files"]) == {"file1.py", "file2.py"}
        assert data["executed_tools"] == ["read", "edit", "read"]
        assert data["failed_tool_signatures"] == [["edit", "abc123"]]
        assert data["read_files_session"] == ["file1.py"]
        assert data["required_files"] == ["main.py"]
        assert data["required_outputs"] == ["analysis"]
        assert data["token_usage"]["prompt_tokens"] == 100

    def test_from_dict(self):
        """Test ExecutionState deserialization."""
        data = {
            "tool_calls_used": 10,
            "observed_files": ["a.py", "b.py"],
            "executed_tools": ["read", "write"],
            "failed_tool_signatures": [["write", "xyz789"]],
            "read_files_session": ["a.py"],
            "required_files": ["a.py", "b.py"],
            "required_outputs": [],
            "token_usage": {
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }

        state = ExecutionState.from_dict(data)

        assert state.tool_calls_used == 10
        assert state.observed_files == {"a.py", "b.py"}
        assert state.executed_tools == ["read", "write"]
        assert state.failed_tool_signatures == {("write", "xyz789")}
        assert state.read_files_session == {"a.py"}
        assert state.required_files == ["a.py", "b.py"]
        assert state.token_usage["prompt_tokens"] == 200

    def test_from_dict_with_defaults(self):
        """Test ExecutionState deserialization with missing fields."""
        data = {"tool_calls_used": 3}

        state = ExecutionState.from_dict(data)

        assert state.tool_calls_used == 3
        assert state.observed_files == set()
        assert state.executed_tools == []
        assert state.token_usage["prompt_tokens"] == 0


class TestSessionFlags:
    """Tests for SessionFlags dataclass."""

    def test_default_initialization(self):
        """Test default values for SessionFlags."""
        flags = SessionFlags()

        assert flags.system_added is False
        assert flags.all_files_read_nudge_sent is False
        assert flags.tool_capability_warned is False
        assert flags.consecutive_blocked_attempts == 0
        assert flags.total_blocked_attempts == 0

    def test_to_dict(self):
        """Test SessionFlags serialization."""
        flags = SessionFlags(
            system_added=True,
            all_files_read_nudge_sent=True,
            tool_capability_warned=False,
            consecutive_blocked_attempts=2,
            total_blocked_attempts=5,
        )

        data = flags.to_dict()

        assert data["system_added"] is True
        assert data["all_files_read_nudge_sent"] is True
        assert data["tool_capability_warned"] is False
        assert data["consecutive_blocked_attempts"] == 2
        assert data["total_blocked_attempts"] == 5

    def test_from_dict(self):
        """Test SessionFlags deserialization."""
        data = {
            "system_added": True,
            "all_files_read_nudge_sent": False,
            "tool_capability_warned": True,
            "consecutive_blocked_attempts": 1,
            "total_blocked_attempts": 3,
        }

        flags = SessionFlags.from_dict(data)

        assert flags.system_added is True
        assert flags.all_files_read_nudge_sent is False
        assert flags.tool_capability_warned is True
        assert flags.consecutive_blocked_attempts == 1
        assert flags.total_blocked_attempts == 3

    def test_from_dict_with_defaults(self):
        """Test SessionFlags deserialization with missing fields."""
        data = {}

        flags = SessionFlags.from_dict(data)

        assert flags.system_added is False
        assert flags.consecutive_blocked_attempts == 0


class TestSessionStateManager:
    """Tests for SessionStateManager."""

    def test_initialization(self):
        """Test manager initialization with default budget."""
        manager = SessionStateManager()

        assert manager.tool_budget == 200
        assert manager.tool_calls_used == 0
        assert len(manager.observed_files) == 0
        assert len(manager.executed_tools) == 0

    def test_initialization_with_custom_budget(self):
        """Test manager initialization with custom tool budget."""
        manager = SessionStateManager(tool_budget=50)

        assert manager.tool_budget == 50

    def test_tool_budget_setter(self):
        """Test setting tool budget."""
        manager = SessionStateManager(tool_budget=100)
        manager.tool_budget = 150

        assert manager.tool_budget == 150

    def test_tool_budget_minimum(self):
        """Test that tool budget cannot go below 1."""
        manager = SessionStateManager(tool_budget=100)
        manager.tool_budget = 0

        assert manager.tool_budget == 1

    # =========================================================================
    # Tool Call Tracking Tests
    # =========================================================================

    def test_record_tool_call(self):
        """Test recording a tool call."""
        manager = SessionStateManager()

        manager.record_tool_call("read_file", {"path": "/src/main.py"})

        assert "read_file" in manager.executed_tools
        assert "/src/main.py" in manager.observed_files

    def test_record_tool_call_tracks_multiple_files(self):
        """Test that tool calls with file paths are tracked."""
        manager = SessionStateManager()

        manager.record_tool_call("read", {"file_path": "/a.py"})
        manager.record_tool_call("edit", {"filepath": "/b.py"})
        manager.record_tool_call("write", {"file": "/c.py"})

        assert "/a.py" in manager.observed_files
        assert "/b.py" in manager.observed_files
        assert "/c.py" in manager.observed_files

    def test_increment_tool_calls(self):
        """Test incrementing tool calls counter."""
        manager = SessionStateManager()

        result = manager.increment_tool_calls()
        assert result == 1
        assert manager.tool_calls_used == 1

        result = manager.increment_tool_calls(5)
        assert result == 6
        assert manager.tool_calls_used == 6

    def test_is_budget_exhausted_false(self):
        """Test budget not exhausted."""
        manager = SessionStateManager(tool_budget=10)
        manager.increment_tool_calls(5)

        assert manager.is_budget_exhausted() is False

    def test_is_budget_exhausted_true(self):
        """Test budget exhausted."""
        manager = SessionStateManager(tool_budget=10)
        manager.increment_tool_calls(10)

        assert manager.is_budget_exhausted() is True

    def test_is_budget_exhausted_over(self):
        """Test budget over-exhausted."""
        manager = SessionStateManager(tool_budget=10)
        manager.increment_tool_calls(15)

        assert manager.is_budget_exhausted() is True

    def test_get_remaining_budget(self):
        """Test remaining budget calculation."""
        manager = SessionStateManager(tool_budget=100)

        assert manager.get_remaining_budget() == 100

        manager.increment_tool_calls(30)
        assert manager.get_remaining_budget() == 70

        manager.increment_tool_calls(80)
        assert manager.get_remaining_budget() == 0  # Can't go negative

    # =========================================================================
    # Failed Tool Signature Tests
    # =========================================================================

    def test_check_failed_signature_not_found(self):
        """Test checking for a signature that hasn't failed."""
        manager = SessionStateManager()

        assert manager.check_failed_signature("edit", "abc123") is False

    def test_add_and_check_failed_signature(self):
        """Test adding and checking failed signatures."""
        manager = SessionStateManager()

        manager.add_failed_signature("edit", "abc123")

        assert manager.check_failed_signature("edit", "abc123") is True
        assert manager.check_failed_signature("edit", "xyz789") is False
        assert manager.check_failed_signature("write", "abc123") is False

    def test_check_and_record_failed_first_time(self):
        """Test check_and_record_failed for first-time failure."""
        manager = SessionStateManager()

        result = manager.check_and_record_failed("edit", {"path": "/a.py"})

        assert result is False  # Not previously failed
        # Now it should be recorded
        args_hash = manager._hash_args({"path": "/a.py"})
        assert manager.check_failed_signature("edit", args_hash) is True

    def test_check_and_record_failed_repeated(self):
        """Test check_and_record_failed for repeated failure."""
        manager = SessionStateManager()

        # First call records the failure
        manager.check_and_record_failed("edit", {"path": "/a.py"})

        # Second call detects it
        result = manager.check_and_record_failed("edit", {"path": "/a.py"})

        assert result is True  # Previously failed

    def test_hash_args_consistency(self):
        """Test that arg hashing is consistent."""
        manager = SessionStateManager()

        hash1 = manager._hash_args({"a": 1, "b": 2})
        hash2 = manager._hash_args({"b": 2, "a": 1})  # Different order

        assert hash1 == hash2  # Should be the same after sorting

    # =========================================================================
    # File Read Tracking Tests
    # =========================================================================

    def test_record_file_read(self):
        """Test recording file reads."""
        manager = SessionStateManager()

        manager.record_file_read("/src/main.py")
        manager.record_file_read("/src/utils.py")

        read_files = manager.get_read_files()
        assert "/src/main.py" in read_files
        assert "/src/utils.py" in read_files
        assert "/src/main.py" in manager.observed_files

    def test_get_read_files_returns_copy(self):
        """Test that get_read_files returns a copy."""
        manager = SessionStateManager()
        manager.record_file_read("/a.py")

        files = manager.get_read_files()
        files.add("/b.py")  # Modify the copy

        assert "/b.py" not in manager.get_read_files()

    # =========================================================================
    # Task Requirements Tests
    # =========================================================================

    def test_set_task_requirements(self):
        """Test setting task requirements."""
        manager = SessionStateManager()

        manager.set_task_requirements(
            required_files=["main.py", "utils.py"],
            required_outputs=["analysis", "report"],
        )

        assert manager.execution_state.required_files == ["main.py", "utils.py"]
        assert manager.execution_state.required_outputs == ["analysis", "report"]

    def test_check_all_files_read_empty(self):
        """Test check_all_files_read with no requirements."""
        manager = SessionStateManager()

        assert manager.check_all_files_read() is False

    def test_check_all_files_read_incomplete(self):
        """Test check_all_files_read with partial completion."""
        manager = SessionStateManager()
        manager.set_task_requirements(required_files=["a.py", "b.py", "c.py"])
        manager.record_file_read("a.py")
        manager.record_file_read("b.py")

        assert manager.check_all_files_read() is False

    def test_check_all_files_read_complete(self):
        """Test check_all_files_read with all files read."""
        manager = SessionStateManager()
        manager.set_task_requirements(required_files=["a.py", "b.py"])
        manager.record_file_read("a.py")
        manager.record_file_read("b.py")

        assert manager.check_all_files_read() is True

    def test_check_all_files_read_extra_files(self):
        """Test check_all_files_read with extra files read."""
        manager = SessionStateManager()
        manager.set_task_requirements(required_files=["a.py"])
        manager.record_file_read("a.py")
        manager.record_file_read("b.py")  # Extra file

        assert manager.check_all_files_read() is True

    # =========================================================================
    # Session Flags Tests
    # =========================================================================

    def test_system_added_flag(self):
        """Test system_added flag management."""
        manager = SessionStateManager()

        assert manager.is_system_added() is False

        manager.mark_system_added()

        assert manager.is_system_added() is True

    def test_all_files_read_nudge(self):
        """Test all_files_read nudge functionality."""
        manager = SessionStateManager()
        manager.set_task_requirements(required_files=["a.py"])
        manager.record_file_read("a.py")

        # Should send nudge since all files read and not sent yet
        assert manager.should_send_all_files_read_nudge() is True

        manager.mark_all_files_read_nudge_sent()

        # Should not send again
        assert manager.should_send_all_files_read_nudge() is False

    def test_tool_capability_warned_flag(self):
        """Test tool_capability_warned flag."""
        manager = SessionStateManager()

        assert manager.is_tool_capability_warned() is False

        manager.mark_tool_capability_warned()

        assert manager.is_tool_capability_warned() is True

    def test_blocked_attempts_tracking(self):
        """Test blocked attempts tracking."""
        manager = SessionStateManager()

        count = manager.record_blocked_attempt()
        assert count == 1

        count = manager.record_blocked_attempt()
        assert count == 2

        consecutive, total = manager.get_blocked_attempts()
        assert consecutive == 2
        assert total == 2

        manager.reset_blocked_attempts()
        consecutive, total = manager.get_blocked_attempts()
        assert consecutive == 0
        assert total == 2  # Total preserved

    # =========================================================================
    # Token Usage Tests
    # =========================================================================

    def test_get_token_usage_initial(self):
        """Test initial token usage."""
        manager = SessionStateManager()

        usage = manager.get_token_usage()

        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0

    def test_update_token_usage(self):
        """Test updating token usage."""
        manager = SessionStateManager()

        manager.update_token_usage(prompt_tokens=100, completion_tokens=50)

        usage = manager.get_token_usage()
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150

    def test_update_token_usage_accumulates(self):
        """Test that token usage accumulates."""
        manager = SessionStateManager()

        manager.update_token_usage(prompt_tokens=100, completion_tokens=50)
        manager.update_token_usage(prompt_tokens=200, completion_tokens=100)

        usage = manager.get_token_usage()
        assert usage["prompt_tokens"] == 300
        assert usage["completion_tokens"] == 150
        assert usage["total_tokens"] == 450

    def test_update_token_usage_with_cache(self):
        """Test updating token usage with cache tokens."""
        manager = SessionStateManager()

        manager.update_token_usage(
            prompt_tokens=100,
            completion_tokens=50,
            cache_creation_input_tokens=20,
            cache_read_input_tokens=30,
        )

        usage = manager.get_token_usage()
        assert usage["cache_creation_input_tokens"] == 20
        assert usage["cache_read_input_tokens"] == 30

    def test_reset_token_usage(self):
        """Test resetting token usage."""
        manager = SessionStateManager()
        manager.update_token_usage(prompt_tokens=100, completion_tokens=50)

        manager.reset_token_usage()

        usage = manager.get_token_usage()
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0

    def test_get_token_usage_returns_copy(self):
        """Test that get_token_usage returns a copy."""
        manager = SessionStateManager()
        manager.update_token_usage(prompt_tokens=100)

        usage = manager.get_token_usage()
        usage["prompt_tokens"] = 999

        assert manager.get_token_usage()["prompt_tokens"] == 100

    # =========================================================================
    # Checkpoint/Restore Tests
    # =========================================================================

    def test_get_checkpoint_state(self):
        """Test checkpoint state serialization."""
        manager = SessionStateManager(tool_budget=100)
        manager.increment_tool_calls(5)
        manager.record_file_read("/a.py")
        manager.mark_system_added()
        manager.update_token_usage(prompt_tokens=50)

        state = manager.get_checkpoint_state()

        assert state["tool_budget"] == 100
        assert state["execution_state"]["tool_calls_used"] == 5
        assert "/a.py" in state["execution_state"]["read_files_session"]
        assert state["session_flags"]["system_added"] is True
        assert state["execution_state"]["token_usage"]["prompt_tokens"] == 50

    def test_apply_checkpoint_state(self):
        """Test checkpoint state restoration."""
        manager = SessionStateManager(tool_budget=50)

        state = {
            "tool_budget": 200,
            "execution_state": {
                "tool_calls_used": 10,
                "observed_files": ["x.py"],
                "executed_tools": ["read"],
                "failed_tool_signatures": [],
                "read_files_session": ["x.py"],
                "required_files": [],
                "required_outputs": [],
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 0,
                    "total_tokens": 100,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
            "session_flags": {
                "system_added": True,
                "all_files_read_nudge_sent": False,
                "tool_capability_warned": False,
                "consecutive_blocked_attempts": 0,
                "total_blocked_attempts": 0,
            },
        }

        manager.apply_checkpoint_state(state)

        assert manager.tool_budget == 200
        assert manager.tool_calls_used == 10
        assert "x.py" in manager.observed_files
        assert manager.is_system_added() is True
        assert manager.get_token_usage()["prompt_tokens"] == 100

    def test_checkpoint_roundtrip(self):
        """Test full checkpoint save/restore cycle."""
        manager1 = SessionStateManager(tool_budget=150)
        manager1.increment_tool_calls(25)
        manager1.record_tool_call("read", {"path": "/src/main.py"})
        manager1.record_file_read("/src/main.py")
        manager1.add_failed_signature("edit", "hash123")
        manager1.mark_system_added()
        manager1.record_blocked_attempt()
        manager1.update_token_usage(prompt_tokens=500, completion_tokens=200)

        state = manager1.get_checkpoint_state()

        manager2 = SessionStateManager()
        manager2.apply_checkpoint_state(state)

        assert manager2.tool_budget == manager1.tool_budget
        assert manager2.tool_calls_used == manager1.tool_calls_used
        assert manager2.observed_files == manager1.observed_files
        assert manager2.is_system_added() == manager1.is_system_added()
        assert manager2.check_failed_signature("edit", "hash123") is True
        assert manager2.get_token_usage() == manager1.get_token_usage()

    # =========================================================================
    # Reset Tests
    # =========================================================================

    def test_reset(self):
        """Test full reset."""
        manager = SessionStateManager(tool_budget=100)
        manager.increment_tool_calls(50)
        manager.record_file_read("/a.py")
        manager.mark_system_added()
        manager.update_token_usage(prompt_tokens=500)

        manager.reset()

        assert manager.tool_budget == 100  # Budget preserved
        assert manager.tool_calls_used == 0
        assert len(manager.observed_files) == 0
        assert manager.is_system_added() is False
        assert manager.get_token_usage()["prompt_tokens"] == 0

    def test_reset_preserves_token_usage(self):
        """Test reset with token usage preservation."""
        manager = SessionStateManager()
        manager.update_token_usage(prompt_tokens=500)
        manager.increment_tool_calls(10)

        manager.reset(preserve_token_usage=True)

        assert manager.tool_calls_used == 0
        assert manager.get_token_usage()["prompt_tokens"] == 500

    def test_reset_for_new_turn(self):
        """Test partial reset for new turn."""
        manager = SessionStateManager()
        manager.record_file_read("/a.py")
        manager.record_blocked_attempt()
        manager.record_blocked_attempt()
        manager.mark_all_files_read_nudge_sent()
        manager.increment_tool_calls(10)

        manager.reset_for_new_turn()

        # These should be reset
        assert len(manager.get_read_files()) == 0
        consecutive, _ = manager.get_blocked_attempts()
        assert consecutive == 0
        assert manager.should_send_all_files_read_nudge() is False  # No required files
        assert manager.tool_calls_used == 0  # Reset for new turn (FIX: now properly resets)
        assert "/a.py" in manager.observed_files  # Observed files preserved

    # =========================================================================
    # Session Summary Tests
    # =========================================================================

    def test_get_session_summary(self):
        """Test session summary generation."""
        manager = SessionStateManager(tool_budget=100)
        manager.increment_tool_calls(30)
        manager.record_tool_call("read", {"path": "/a.py"})
        manager.record_tool_call("read", {"path": "/b.py"})
        manager.record_tool_call("edit", {"path": "/a.py"})
        manager.record_file_read("/a.py")
        manager.add_failed_signature("edit", "hash1")
        manager.set_task_requirements(required_files=["/a.py"])
        manager.mark_system_added()
        manager.record_blocked_attempt()
        manager.update_token_usage(prompt_tokens=1000, completion_tokens=500)

        summary = manager.get_session_summary()

        assert summary["tool_budget"] == 100
        assert summary["tool_calls_used"] == 30
        assert summary["tool_calls_remaining"] == 70
        assert summary["budget_exhausted"] is False
        assert summary["files_observed"] == 2
        assert summary["files_read"] == 1
        assert summary["unique_tools_used"] == 2  # read, edit
        assert summary["total_tool_executions"] == 3
        assert summary["failed_signatures"] == 1
        assert summary["required_files_count"] == 1
        assert summary["all_files_read"] is True
        assert summary["system_added"] is True
        assert summary["total_blocked_attempts"] == 1
        assert summary["token_usage"]["prompt_tokens"] == 1000

    # =========================================================================
    # Repr Test
    # =========================================================================

    def test_repr(self):
        """Test string representation."""
        manager = SessionStateManager(tool_budget=100)
        manager.increment_tool_calls(25)
        manager.record_file_read("/a.py")
        manager.record_tool_call("read", {})

        repr_str = repr(manager)

        assert "SessionStateManager" in repr_str
        assert "25/100" in repr_str


class TestCreateSessionStateManager:
    """Tests for factory function."""

    def test_create_with_defaults(self):
        """Test factory with default values."""
        manager = create_session_state_manager()

        assert manager.tool_budget == 200
        assert manager.tool_calls_used == 0

    def test_create_with_custom_budget(self):
        """Test factory with custom budget."""
        manager = create_session_state_manager(tool_budget=50)

        assert manager.tool_budget == 50

    def test_create_with_initial_state(self):
        """Test factory with initial state."""
        initial_state = {
            "tool_budget": 150,
            "execution_state": {
                "tool_calls_used": 20,
                "observed_files": [],
                "executed_tools": [],
                "failed_tool_signatures": [],
                "read_files_session": [],
                "required_files": [],
                "required_outputs": [],
                "token_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
            "session_flags": {
                "system_added": False,
                "all_files_read_nudge_sent": False,
                "tool_capability_warned": False,
                "consecutive_blocked_attempts": 0,
                "total_blocked_attempts": 0,
            },
        }

        manager = create_session_state_manager(initial_state=initial_state)

        assert manager.tool_budget == 150
        assert manager.tool_calls_used == 20
