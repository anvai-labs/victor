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

"""Tests for victor.tools.context module."""

from pathlib import Path

import pytest

from victor.tools.context import (
    DEFAULT_PERMISSIONS,
    FULL_PERMISSIONS,
    Permission,
    SAFE_PERMISSIONS,
    STANDARD_PERMISSIONS,
    ToolExecutionContext,
    create_context,
    create_full_access_context,
    create_readonly_context,
)


class TestPermission:
    """Tests for Permission enum."""

    def test_all_permissions_defined(self):
        """Test all expected permissions exist."""
        assert Permission.READ_FILES
        assert Permission.WRITE_FILES
        assert Permission.EXECUTE_COMMANDS
        assert Permission.NETWORK_ACCESS
        assert Permission.GIT_OPERATIONS
        assert Permission.ADMIN_OPERATIONS
        assert Permission.DATABASE_ACCESS
        assert Permission.SENSITIVE_DATA

    def test_permission_sets(self):
        """Test predefined permission sets."""
        assert Permission.READ_FILES in DEFAULT_PERMISSIONS
        assert Permission.READ_FILES in SAFE_PERMISSIONS
        assert Permission.WRITE_FILES not in SAFE_PERMISSIONS

        assert Permission.READ_FILES in STANDARD_PERMISSIONS
        assert Permission.WRITE_FILES in STANDARD_PERMISSIONS
        assert Permission.EXECUTE_COMMANDS in STANDARD_PERMISSIONS

        assert len(FULL_PERMISSIONS) == len(Permission)


class TestToolExecutionContext:
    """Tests for ToolExecutionContext dataclass."""

    def test_create_minimal(self):
        """Test creating with minimal fields."""
        context = ToolExecutionContext(
            session_id="test123",
            workspace_root=Path("/project"),
        )
        assert context.session_id == "test123"
        assert context.workspace_root == Path("/project")
        assert context.current_stage == "INITIAL"
        assert context.tool_budget_total == 25

    def test_create_full(self):
        """Test creating with all fields."""
        context = ToolExecutionContext(
            session_id="test123",
            workspace_root=Path("/project"),
            conversation_history=[{"role": "user", "content": "hi"}],
            current_stage="EXECUTING",
            tool_budget_total=50,
            tool_budget_used=10,
            provider_name="anthropic",
            model_name="claude-3",
            user_permissions={Permission.READ_FILES, Permission.WRITE_FILES},
            vertical="coding",
        )
        assert context.tool_budget_total == 50
        assert context.tool_budget_used == 10
        assert context.vertical == "coding"

    def test_budget_remaining(self):
        """Test tool_budget_remaining property."""
        context = ToolExecutionContext(
            session_id="test",
            workspace_root=Path("."),
            tool_budget_total=25,
            tool_budget_used=10,
        )
        assert context.tool_budget_remaining == 15

    def test_budget_remaining_zero(self):
        """Test budget remaining when exhausted."""
        context = ToolExecutionContext(
            session_id="test",
            workspace_root=Path("."),
            tool_budget_total=25,
            tool_budget_used=30,  # Over budget
        )
        assert context.tool_budget_remaining == 0

    def test_budget_exhausted(self):
        """Test budget_exhausted property."""
        context = ToolExecutionContext(
            session_id="test",
            workspace_root=Path("."),
            tool_budget_total=25,
            tool_budget_used=25,
        )
        assert context.budget_exhausted is True

    def test_budget_not_exhausted(self):
        """Test budget_exhausted when budget remains."""
        context = ToolExecutionContext(
            session_id="test",
            workspace_root=Path("."),
            tool_budget_total=25,
            tool_budget_used=10,
        )
        assert context.budget_exhausted is False

    def test_use_budget_success(self):
        """Test use_budget when budget available."""
        context = ToolExecutionContext(
            session_id="test",
            workspace_root=Path("."),
            tool_budget_total=25,
            tool_budget_used=10,
        )
        result = context.use_budget(5)
        assert result is True
        assert context.tool_budget_used == 15

    def test_use_budget_failure(self):
        """Test use_budget when insufficient budget."""
        context = ToolExecutionContext(
            session_id="test",
            workspace_root=Path("."),
            tool_budget_total=25,
            tool_budget_used=20,
        )
        result = context.use_budget(10)
        assert result is False
        assert context.tool_budget_used == 20  # Unchanged

    def test_reset_budget(self):
        """Test reset_budget method."""
        context = ToolExecutionContext(
            session_id="test",
            workspace_root=Path("."),
            tool_budget_total=25,
            tool_budget_used=15,
        )
        context.reset_budget()
        assert context.tool_budget_used == 0


class TestPermissionChecks:
    """Tests for permission checking methods."""

    def test_can_read(self):
        """Test can_read property."""
        context = ToolExecutionContext(
            session_id="test",
            workspace_root=Path("."),
            user_permissions={Permission.READ_FILES},
        )
        assert context.can_read is True

    def test_can_write(self):
        """Test can_write property."""
        context_yes = ToolExecutionContext(
            session_id="test",
            workspace_root=Path("."),
            user_permissions={Permission.WRITE_FILES},
        )
        context_no = ToolExecutionContext(
            session_id="test",
            workspace_root=Path("."),
            user_permissions={Permission.READ_FILES},
        )
        assert context_yes.can_write is True
        assert context_no.can_write is False

    def test_can_execute(self):
        """Test can_execute property."""
        context = ToolExecutionContext(
            session_id="test",
            workspace_root=Path("."),
            user_permissions={Permission.EXECUTE_COMMANDS},
        )
        assert context.can_execute is True

    def test_can_network(self):
        """Test can_network property."""
        context = ToolExecutionContext(
            session_id="test",
            workspace_root=Path("."),
            user_permissions={Permission.NETWORK_ACCESS},
        )
        assert context.can_network is True

    def test_can_git(self):
        """Test can_git property."""
        context = ToolExecutionContext(
            session_id="test",
            workspace_root=Path("."),
            user_permissions={Permission.GIT_OPERATIONS},
        )
        assert context.can_git is True

    def test_has_permission(self):
        """Test has_permission method."""
        context = ToolExecutionContext(
            session_id="test",
            workspace_root=Path("."),
            user_permissions={Permission.READ_FILES, Permission.WRITE_FILES},
        )
        assert context.has_permission(Permission.READ_FILES) is True
        assert context.has_permission(Permission.EXECUTE_COMMANDS) is False

    def test_has_all_permissions(self):
        """Test has_all_permissions method."""
        context = ToolExecutionContext(
            session_id="test",
            workspace_root=Path("."),
            user_permissions={Permission.READ_FILES, Permission.WRITE_FILES},
        )
        assert context.has_all_permissions({Permission.READ_FILES}) is True
        assert context.has_all_permissions({Permission.READ_FILES, Permission.WRITE_FILES}) is True
        assert (
            context.has_all_permissions({Permission.READ_FILES, Permission.EXECUTE_COMMANDS})
            is False
        )

    def test_has_any_permission(self):
        """Test has_any_permission method."""
        context = ToolExecutionContext(
            session_id="test",
            workspace_root=Path("."),
            user_permissions={Permission.READ_FILES},
        )
        assert context.has_any_permission({Permission.READ_FILES, Permission.WRITE_FILES}) is True
        assert (
            context.has_any_permission({Permission.EXECUTE_COMMANDS, Permission.NETWORK_ACCESS})
            is False
        )


class TestFileStateTracking:
    """Tests for file state tracking methods."""

    def test_mark_file_modified(self):
        """Test mark_file_modified method."""
        context = ToolExecutionContext(
            session_id="test",
            workspace_root=Path("."),
        )
        context.mark_file_modified("/path/to/file.py")
        assert "/path/to/file.py" in context.modified_files

    def test_mark_file_created(self):
        """Test mark_file_created method."""
        context = ToolExecutionContext(
            session_id="test",
            workspace_root=Path("."),
        )
        context.mark_file_created("/path/to/new.py")
        assert "/path/to/new.py" in context.created_files

    def test_cache_file_content(self):
        """Test cache_file_content method."""
        context = ToolExecutionContext(
            session_id="test",
            workspace_root=Path("."),
        )
        context.cache_file_content("/path/file.py", "print('hello')")
        assert context.get_cached_content("/path/file.py") == "print('hello')"

    def test_get_cached_content_missing(self):
        """Test get_cached_content for missing file."""
        context = ToolExecutionContext(
            session_id="test",
            workspace_root=Path("."),
        )
        assert context.get_cached_content("/missing/file.py") is None


class TestBackwardCompatibility:
    """Tests for to_dict/from_dict backward compatibility."""

    def test_to_dict(self):
        """Test to_dict method."""
        context = ToolExecutionContext(
            session_id="test123",
            workspace_root=Path("/project"),
            current_stage="EXECUTING",
            tool_budget_total=50,
            tool_budget_used=10,
            provider_name="anthropic",
            model_name="claude-3",
            user_permissions=STANDARD_PERMISSIONS,
            vertical="coding",
        )
        d = context.to_dict()

        assert d["session_id"] == "test123"
        assert d["workspace_root"] == "/project"
        assert d["current_stage"] == "EXECUTING"
        assert d["tool_budget_remaining"] == 40
        assert d["can_read"] is True
        assert d["can_write"] is True
        assert d["vertical"] == "coding"

    def test_from_dict(self):
        """Test from_dict method."""
        d = {
            "session_id": "test123",
            "workspace_root": "/project",
            "current_stage": "PLANNING",
            "tool_budget_total": 30,
            "tool_budget_used": 5,
            "can_read": True,
            "can_write": True,
            "can_execute": False,
            "vertical": "devops",
        }
        context = ToolExecutionContext.from_dict(d)

        assert context.session_id == "test123"
        assert context.workspace_root == Path("/project")
        assert context.current_stage == "PLANNING"
        assert context.tool_budget_total == 30
        assert context.can_read is True
        assert context.can_write is True
        assert context.can_execute is False
        assert context.vertical == "devops"

    def test_from_dict_empty(self):
        """Test from_dict with minimal data."""
        d = {}
        context = ToolExecutionContext.from_dict(d)
        assert context.session_id == ""
        assert context.workspace_root == Path(".")
        assert context.current_stage == "INITIAL"

    def test_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        original = ToolExecutionContext(
            session_id="test123",
            workspace_root=Path("/project"),
            current_stage="EXECUTING",
            tool_budget_total=50,
            tool_budget_used=10,
            provider_name="anthropic",
            vertical="coding",
            user_permissions=STANDARD_PERMISSIONS,
        )

        d = original.to_dict()
        restored = ToolExecutionContext.from_dict(d)

        assert restored.session_id == original.session_id
        assert restored.current_stage == original.current_stage
        assert restored.tool_budget_total == original.tool_budget_total
        assert restored.vertical == original.vertical

    def test_metadata_preserved(self):
        """Test that unknown fields become metadata."""
        d = {
            "session_id": "test",
            "workspace_root": ".",
            "custom_field": "custom_value",
            "another_field": 42,
        }
        context = ToolExecutionContext.from_dict(d)
        assert context.metadata["custom_field"] == "custom_value"
        assert context.metadata["another_field"] == 42

    def test_repr(self):
        """Test __repr__ method."""
        context = ToolExecutionContext(
            session_id="test123",
            workspace_root=Path("/project"),
            current_stage="EXECUTING",
            tool_budget_total=25,
            tool_budget_used=10,
        )
        r = repr(context)
        assert "test123" in r
        assert "EXECUTING" in r
        assert "15/25" in r


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_context(self):
        """Test create_context factory."""
        context = create_context(
            session_id="test",
            workspace_root=Path("/project"),
            budget=30,
            vertical="coding",
        )
        assert context.session_id == "test"
        assert context.tool_budget_total == 30
        assert context.vertical == "coding"
        # Standard permissions by default
        assert context.can_read is True
        assert context.can_write is True

    def test_create_readonly_context(self):
        """Test create_readonly_context factory."""
        context = create_readonly_context(
            session_id="test",
            workspace_root=Path("/project"),
            budget=100,
        )
        assert context.can_read is True
        assert context.can_write is False
        assert context.can_execute is False
        assert context.current_stage == "EXPLORE"
        assert context.tool_budget_total == 100

    def test_create_full_access_context(self):
        """Test create_full_access_context factory."""
        context = create_full_access_context(
            session_id="test",
            workspace_root=Path("/project"),
        )
        assert context.can_read is True
        assert context.can_write is True
        assert context.can_execute is True
        assert context.can_network is True
        assert context.can_git is True
        assert context.has_permission(Permission.ADMIN_OPERATIONS) is True
        assert context.tool_budget_total == 100


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Test __all__ exports."""
        from victor.tools import context

        # Check main classes
        assert hasattr(context, "ToolExecutionContext")
        assert hasattr(context, "Permission")

        # Check permission sets
        assert hasattr(context, "DEFAULT_PERMISSIONS")
        assert hasattr(context, "SAFE_PERMISSIONS")
        assert hasattr(context, "STANDARD_PERMISSIONS")
        assert hasattr(context, "FULL_PERMISSIONS")

        # Check factory functions
        assert hasattr(context, "create_context")
        assert hasattr(context, "create_readonly_context")
        assert hasattr(context, "create_full_access_context")
