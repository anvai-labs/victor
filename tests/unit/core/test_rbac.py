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

"""Tests for RBAC module - achieving 70%+ coverage."""

import asyncio
import pytest
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

from victor.security.auth.rbac import (
    Permission,
    Role,
    User,
    RBACManager,
    require_permission,
    get_rbac_manager,
    set_rbac_manager,
    get_permission_for_access_mode,
)


class TestPermission:
    """Tests for Permission enum."""

    def test_permission_values(self):
        """Test permission value strings."""
        assert Permission.READ.value == "read"
        assert Permission.WRITE.value == "write"
        assert Permission.EXECUTE.value == "execute"
        assert Permission.NETWORK.value == "network"
        assert Permission.ADMIN.value == "admin"
        assert Permission.TOOL_MANAGE.value == "tool_manage"
        assert Permission.USER_MANAGE.value == "user_manage"

    def test_from_access_mode_readonly(self):
        """Test conversion from READONLY AccessMode."""
        from victor.tools.base import AccessMode

        perm = Permission.from_access_mode(AccessMode.READONLY)
        assert perm == Permission.READ

    def test_from_access_mode_write(self):
        """Test conversion from WRITE AccessMode."""
        from victor.tools.base import AccessMode

        perm = Permission.from_access_mode(AccessMode.WRITE)
        assert perm == Permission.WRITE

    def test_from_access_mode_execute(self):
        """Test conversion from EXECUTE AccessMode."""
        from victor.tools.base import AccessMode

        perm = Permission.from_access_mode(AccessMode.EXECUTE)
        assert perm == Permission.EXECUTE

    def test_from_access_mode_network(self):
        """Test conversion from NETWORK AccessMode."""
        from victor.tools.base import AccessMode

        perm = Permission.from_access_mode(AccessMode.NETWORK)
        assert perm == Permission.NETWORK

    def test_from_access_mode_mixed(self):
        """Test conversion from MIXED AccessMode."""
        from victor.tools.base import AccessMode

        perm = Permission.from_access_mode(AccessMode.MIXED)
        assert perm == Permission.ADMIN


class TestRole:
    """Tests for Role dataclass."""

    def test_basic_role_creation(self):
        """Test creating a basic role."""
        role = Role("test_role", frozenset({Permission.READ}))
        assert role.name == "test_role"
        assert Permission.READ in role.permissions
        assert role.allowed_categories == frozenset()
        assert role.denied_tools == frozenset()

    def test_role_with_categories(self):
        """Test role with category restrictions."""
        role = Role(
            "dev",
            frozenset({Permission.READ, Permission.WRITE}),
            allowed_categories=frozenset({"filesystem", "git"}),
        )
        assert role.can_access_category("filesystem")
        assert role.can_access_category("git")
        assert not role.can_access_category("network")

    def test_role_empty_categories_allows_all(self):
        """Test that empty categories allows all."""
        role = Role("admin", frozenset({Permission.ADMIN}))
        assert role.can_access_category("filesystem")
        assert role.can_access_category("network")
        assert role.can_access_category("anything")

    def test_role_denied_tools(self):
        """Test role with denied tools."""
        role = Role(
            "limited",
            frozenset({Permission.READ}),
            denied_tools=frozenset({"dangerous_tool"}),
        )
        assert not role.can_use_tool("dangerous_tool", "filesystem")
        assert role.can_use_tool("safe_tool", "filesystem")

    def test_role_has_permission_basic(self):
        """Test has_permission for basic permissions."""
        role = Role("viewer", frozenset({Permission.READ}))
        assert role.has_permission(Permission.READ)
        assert not role.has_permission(Permission.WRITE)

    def test_role_admin_grants_all(self):
        """Test that ADMIN permission grants all other permissions."""
        role = Role("admin", frozenset({Permission.ADMIN}))
        assert role.has_permission(Permission.READ)
        assert role.has_permission(Permission.WRITE)
        assert role.has_permission(Permission.EXECUTE)
        assert role.has_permission(Permission.NETWORK)
        assert role.has_permission(Permission.ADMIN)

    def test_role_immutability(self):
        """Test that role fields are properly frozen."""
        # Pass mutable sets - they should be converted to frozensets
        role = Role(
            "test",
            {Permission.READ},  # type: ignore
            allowed_categories={"cat1"},  # type: ignore
            denied_tools={"tool1"},  # type: ignore
        )
        assert isinstance(role.permissions, frozenset)
        assert isinstance(role.allowed_categories, frozenset)
        assert isinstance(role.denied_tools, frozenset)

    def test_can_use_tool_with_category_restriction(self):
        """Test can_use_tool respects category restrictions."""
        role = Role(
            "dev",
            frozenset({Permission.READ}),
            allowed_categories=frozenset({"filesystem"}),
        )
        assert role.can_use_tool("read_file", "filesystem")
        assert not role.can_use_tool("web_search", "network")


class TestUser:
    """Tests for User dataclass."""

    def test_basic_user_creation(self):
        """Test creating a basic user."""
        user = User("alice")
        assert user.name == "alice"
        assert user.roles == set()
        assert user.metadata == {}

    def test_user_with_roles(self):
        """Test user with assigned roles."""
        admin_role = Role("admin", frozenset({Permission.ADMIN}))
        user = User("alice", roles={admin_role})
        assert admin_role in user.roles

    def test_user_with_metadata(self):
        """Test user with metadata."""
        user = User("bob", metadata={"email": "bob@example.com"})
        assert user.metadata["email"] == "bob@example.com"

    def test_get_effective_permissions_single_role(self):
        """Test effective permissions with single role."""
        role = Role("viewer", frozenset({Permission.READ}))
        user = User("test", roles={role})
        perms = user.get_effective_permissions()
        assert Permission.READ in perms
        assert Permission.WRITE not in perms

    def test_get_effective_permissions_multiple_roles(self):
        """Test effective permissions with multiple roles."""
        viewer = Role("viewer", frozenset({Permission.READ}))
        writer = Role("writer", frozenset({Permission.WRITE}))
        user = User("test", roles={viewer, writer})
        perms = user.get_effective_permissions()
        assert Permission.READ in perms
        assert Permission.WRITE in perms
        assert Permission.EXECUTE not in perms

    def test_has_permission_true(self):
        """Test has_permission returns True when user has permission."""
        role = Role("admin", frozenset({Permission.ADMIN}))
        user = User("alice", roles={role})
        assert user.has_permission(Permission.ADMIN)
        assert user.has_permission(Permission.READ)  # ADMIN grants all

    def test_has_permission_false(self):
        """Test has_permission returns False when user lacks permission."""
        role = Role("viewer", frozenset({Permission.READ}))
        user = User("bob", roles={role})
        assert not user.has_permission(Permission.WRITE)

    def test_can_use_tool_with_permission(self):
        """Test can_use_tool with proper permission."""
        role = Role("dev", frozenset({Permission.READ, Permission.WRITE}))
        user = User("dev_user", roles={role})
        assert user.can_use_tool("edit_file", "filesystem", Permission.WRITE)

    def test_can_use_tool_without_permission(self):
        """Test can_use_tool without permission."""
        role = Role("viewer", frozenset({Permission.READ}))
        user = User("viewer_user", roles={role})
        assert not user.can_use_tool("edit_file", "filesystem", Permission.WRITE)

    def test_can_use_tool_with_category_restriction(self):
        """Test can_use_tool with category restrictions."""
        role = Role(
            "fs_only",
            frozenset({Permission.READ}),
            allowed_categories=frozenset({"filesystem"}),
        )
        user = User("fs_user", roles={role})
        assert user.can_use_tool("read_file", "filesystem", Permission.READ)
        assert not user.can_use_tool("web_search", "network", Permission.READ)


class TestRBACManager:
    """Tests for RBACManager class."""

    def test_default_initialization(self):
        """Test default RBACManager initialization."""
        rbac = RBACManager()
        assert rbac.enabled is True
        assert rbac._default_role_name == "viewer"
        assert rbac._allow_unknown_users is True
        # Check predefined roles are loaded
        assert rbac.get_role("admin") is not None
        assert rbac.get_role("developer") is not None
        assert rbac.get_role("viewer") is not None

    def test_disabled_initialization(self):
        """Test disabled RBACManager."""
        rbac = RBACManager(enabled=False)
        assert rbac.enabled is False

    def test_custom_default_role(self):
        """Test custom default role."""
        rbac = RBACManager(default_role="admin")
        assert rbac._default_role_name == "admin"

    def test_enable_disable(self):
        """Test enable/disable methods."""
        rbac = RBACManager(enabled=False)
        assert not rbac.enabled
        rbac.enable()
        assert rbac.enabled
        rbac.disable()
        assert not rbac.enabled

    def test_add_and_get_role(self):
        """Test adding and retrieving roles."""
        rbac = RBACManager()
        custom_role = Role("custom", frozenset({Permission.READ, Permission.NETWORK}))
        rbac.add_role(custom_role)
        retrieved = rbac.get_role("custom")
        assert retrieved is not None
        assert retrieved.name == "custom"
        assert Permission.NETWORK in retrieved.permissions

    def test_get_role_not_found(self):
        """Test getting non-existent role."""
        rbac = RBACManager()
        assert rbac.get_role("nonexistent") is None

    def test_add_and_get_user(self):
        """Test adding and retrieving users."""
        rbac = RBACManager()
        admin_role = rbac.get_role("admin")
        user = User("alice", roles={admin_role})
        rbac.add_user(user)
        retrieved = rbac.get_user("alice")
        assert retrieved is not None
        assert retrieved.name == "alice"

    def test_get_user_creates_default(self):
        """Test getting unknown user creates default user."""
        rbac = RBACManager(allow_unknown_users=True, default_role="viewer")
        user = rbac.get_user("unknown_user")
        assert user is not None
        assert user.name == "unknown_user"
        # Should have viewer role
        assert any(r.name == "viewer" for r in user.roles)

    def test_get_user_denies_unknown(self):
        """Test getting unknown user when not allowed."""
        rbac = RBACManager(allow_unknown_users=False)
        user = rbac.get_user("unknown_user")
        assert user is None

    def test_set_current_user(self):
        """Test setting and getting current user."""
        rbac = RBACManager()
        rbac.set_current_user("alice")
        assert rbac.get_current_user() == "alice"
        rbac.set_current_user(None)
        assert rbac.get_current_user() is None

    def test_check_permission_enabled(self):
        """Test permission check when enabled."""
        rbac = RBACManager()
        admin_role = rbac.get_role("admin")
        user = User("alice", roles={admin_role})
        rbac.add_user(user)
        assert rbac.check_permission("alice", Permission.ADMIN)
        assert rbac.check_permission("alice", Permission.READ)

    def test_check_permission_disabled(self):
        """Test permission check when disabled with fail_open."""
        rbac = RBACManager(enabled=False, fail_open=True)
        # All permissions pass when disabled and fail_open=True
        assert rbac.check_permission("anyone", Permission.ADMIN)

    def test_check_permission_denied(self):
        """Test permission denied."""
        rbac = RBACManager(allow_unknown_users=False)
        assert not rbac.check_permission("unknown", Permission.ADMIN)

    def test_check_tool_access_enabled(self):
        """Test tool access check when enabled."""
        from victor.tools.base import AccessMode

        rbac = RBACManager()
        dev_role = rbac.get_role("developer")
        user = User("dev", roles={dev_role})
        rbac.add_user(user)
        # Developer has READ, WRITE, EXECUTE
        assert rbac.check_tool_access("dev", "edit_file", "filesystem", AccessMode.WRITE)
        assert not rbac.check_tool_access("dev", "web_search", "network", AccessMode.NETWORK)

    def test_check_tool_access_disabled(self):
        """Test tool access check when disabled with fail_open."""
        from victor.tools.base import AccessMode

        rbac = RBACManager(enabled=False, fail_open=True)
        assert rbac.check_tool_access("anyone", "any_tool", "any_cat", AccessMode.MIXED)

    def test_check_current_user_tool_access(self):
        """Test tool access check for current user."""
        from victor.tools.base import AccessMode

        rbac = RBACManager()
        admin_role = rbac.get_role("admin")
        user = User("alice", roles={admin_role})
        rbac.add_user(user)
        rbac.set_current_user("alice")
        assert rbac.check_current_user_tool_access("any_tool", "any_cat", AccessMode.MIXED)

    def test_check_current_user_tool_access_no_user(self):
        """Test tool access check with no current user."""
        from victor.tools.base import AccessMode

        rbac = RBACManager(allow_unknown_users=True)
        # No current user set
        assert rbac.check_current_user_tool_access("tool", "cat", AccessMode.READONLY)

    def test_load_from_dict(self):
        """Test loading configuration from dict."""
        config = {
            "enabled": True,
            "default_role": "developer",
            "allow_unknown_users": False,
            "roles": {
                "custom_role": {
                    "permissions": ["READ", "WRITE"],
                    "tool_categories": ["filesystem"],
                    "denied_tools": ["dangerous"],
                }
            },
            "users": {
                "test_user": {
                    "roles": ["custom_role"],
                    "metadata": {"team": "engineering"},
                }
            },
        }
        rbac = RBACManager()
        rbac.load_from_dict(config)
        assert rbac._default_role_name == "developer"
        assert rbac._allow_unknown_users is False
        role = rbac.get_role("custom_role")
        assert role is not None
        assert Permission.READ in role.permissions
        assert "filesystem" in role.allowed_categories
        user = rbac.get_user("test_user")
        assert user is not None
        assert user.metadata["team"] == "engineering"

    def test_load_from_yaml(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
rbac:
  enabled: true
  default_role: viewer
  roles:
    test_role:
      permissions:
        - READ
        - WRITE
  users:
    yaml_user:
      roles:
        - test_role
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            rbac = RBACManager()
            rbac.load_from_yaml(yaml_path)
            role = rbac.get_role("test_role")
            assert role is not None
            user = rbac.get_user("yaml_user")
            assert user is not None
        finally:
            yaml_path.unlink()

    def test_get_stats(self):
        """Test getting RBAC statistics."""
        rbac = RBACManager()
        stats = rbac.get_stats()
        assert "enabled" in stats
        assert "roles_count" in stats
        assert "users_count" in stats
        assert "default_role" in stats
        assert "allow_unknown_users" in stats
        assert "current_user" in stats
        # Predefined roles should be counted
        assert stats["roles_count"] >= 4

    def test_thread_safety(self):
        """Test thread-safe concurrent access."""
        rbac = RBACManager()
        errors = []

        def worker(i):
            try:
                role = Role(f"role_{i}", frozenset({Permission.READ}))
                rbac.add_role(role)
                user = User(f"user_{i}", roles={role})
                rbac.add_user(user)
                rbac.check_permission(f"user_{i}", Permission.READ)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestRequirePermissionDecorator:
    """Tests for require_permission decorator."""

    def test_sync_function_allowed(self):
        """Test sync function when permission is granted."""
        rbac = RBACManager()
        admin_role = rbac.get_role("admin")
        user = User("alice", roles={admin_role})
        rbac.add_user(user)

        @require_permission(Permission.READ)
        def read_operation():
            return "success"

        result = read_operation(_rbac_manager=rbac, _rbac_user="alice")
        assert result == "success"

    def test_sync_function_denied(self):
        """Test sync function when permission is denied."""
        rbac = RBACManager()
        viewer_role = rbac.get_role("viewer")
        user = User("bob", roles={viewer_role})
        rbac.add_user(user)

        @require_permission(Permission.ADMIN)
        def admin_operation():
            return "success"

        with pytest.raises(PermissionError):
            admin_operation(_rbac_manager=rbac, _rbac_user="bob")

    def test_async_function_allowed(self):
        """Test async function when permission is granted."""
        rbac = RBACManager()
        admin_role = rbac.get_role("admin")
        user = User("alice", roles={admin_role})
        rbac.add_user(user)

        @require_permission(Permission.WRITE)
        async def async_write():
            return "written"

        # Use new event loop to avoid pollution from other tests
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(async_write(_rbac_manager=rbac, _rbac_user="alice"))
            assert result == "written"
        finally:
            loop.close()

    def test_async_function_denied(self):
        """Test async function when permission is denied."""
        rbac = RBACManager()
        viewer_role = rbac.get_role("viewer")
        user = User("bob", roles={viewer_role})
        rbac.add_user(user)

        @require_permission(Permission.EXECUTE)
        async def async_execute():
            return "executed"

        # Use new event loop to avoid pollution from other tests
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(PermissionError):
                loop.run_until_complete(async_execute(_rbac_manager=rbac, _rbac_user="bob"))
        finally:
            loop.close()

    def test_no_rbac_manager_passes(self):
        """Test function passes when no RBAC manager provided."""

        @require_permission(Permission.ADMIN)
        def operation():
            return "success"

        result = operation()
        assert result == "success"

    def test_disabled_rbac_passes(self):
        """Test function passes when RBAC is disabled."""
        rbac = RBACManager(enabled=False)

        @require_permission(Permission.ADMIN)
        def operation():
            return "success"

        result = operation(_rbac_manager=rbac, _rbac_user="anyone")
        assert result == "success"


class TestGlobalFunctions:
    """Tests for global RBAC functions."""

    def test_get_rbac_manager_singleton(self):
        """Test get_rbac_manager returns singleton."""
        # Reset global state
        set_rbac_manager(None)

        manager1 = get_rbac_manager()
        manager2 = get_rbac_manager()
        assert manager1 is manager2

    def test_set_rbac_manager(self):
        """Test setting global RBAC manager."""
        custom_manager = RBACManager(enabled=False)
        set_rbac_manager(custom_manager)
        retrieved = get_rbac_manager()
        assert retrieved is custom_manager
        assert retrieved.enabled is False

    def test_get_permission_for_access_mode(self):
        """Test get_permission_for_access_mode function."""
        from victor.tools.base import AccessMode

        assert get_permission_for_access_mode(AccessMode.READONLY) == Permission.READ
        assert get_permission_for_access_mode(AccessMode.WRITE) == Permission.WRITE
        assert get_permission_for_access_mode(AccessMode.EXECUTE) == Permission.EXECUTE
        assert get_permission_for_access_mode(AccessMode.NETWORK) == Permission.NETWORK
        assert get_permission_for_access_mode(AccessMode.MIXED) == Permission.ADMIN


class TestPredefinedRoles:
    """Tests for predefined roles."""

    def test_admin_role(self):
        """Test admin predefined role."""
        rbac = RBACManager()
        admin = rbac.get_role("admin")
        assert admin is not None
        assert Permission.ADMIN in admin.permissions

    def test_developer_role(self):
        """Test developer predefined role."""
        rbac = RBACManager()
        dev = rbac.get_role("developer")
        assert dev is not None
        assert Permission.READ in dev.permissions
        assert Permission.WRITE in dev.permissions
        assert Permission.EXECUTE in dev.permissions
        assert Permission.NETWORK not in dev.permissions

    def test_operator_role(self):
        """Test operator predefined role."""
        rbac = RBACManager()
        operator = rbac.get_role("operator")
        assert operator is not None
        assert Permission.READ in operator.permissions
        assert Permission.EXECUTE in operator.permissions
        assert Permission.NETWORK in operator.permissions
        assert Permission.WRITE not in operator.permissions

    def test_viewer_role(self):
        """Test viewer predefined role."""
        rbac = RBACManager()
        viewer = rbac.get_role("viewer")
        assert viewer is not None
        assert Permission.READ in viewer.permissions
        assert len(viewer.permissions) == 1


class TestEdgeCases:
    """Edge case tests for RBAC."""

    def test_user_with_no_roles(self):
        """Test user with no roles has no permissions."""
        user = User("empty_user")
        assert user.get_effective_permissions() == set()
        assert not user.has_permission(Permission.READ)

    def test_role_with_no_permissions(self):
        """Test role with no permissions."""
        role = Role("empty_role", frozenset())
        assert not role.has_permission(Permission.READ)

    def test_load_from_dict_with_invalid_permission(self):
        """Test loading config with invalid permission name."""
        config = {
            "roles": {
                "invalid": {
                    "permissions": ["READ", "INVALID_PERM", "WRITE"],
                }
            }
        }
        rbac = RBACManager()
        rbac.load_from_dict(config)
        role = rbac.get_role("invalid")
        assert role is not None
        # Only valid permissions should be loaded
        assert Permission.READ in role.permissions
        assert Permission.WRITE in role.permissions

    def test_load_from_dict_with_missing_role_for_user(self):
        """Test loading config where user references non-existent role."""
        config = {
            "users": {
                "orphan": {
                    "roles": ["nonexistent_role"],
                }
            }
        }
        rbac = RBACManager()
        rbac.load_from_dict(config)
        user = rbac.get_user("orphan")
        assert user is not None
        assert len(user.roles) == 0  # No roles assigned

    def test_check_permission_unknown_user_disallowed(self):
        """Test permission check for unknown user when not allowed."""
        rbac = RBACManager(allow_unknown_users=False)
        assert not rbac.check_permission("unknown", Permission.READ)

    def test_concurrent_role_modification(self):
        """Test concurrent role modifications."""
        rbac = RBACManager()
        errors = []

        def add_roles():
            try:
                for i in range(100):
                    role = Role(
                        f"concurrent_role_{threading.current_thread().name}_{i}",
                        frozenset({Permission.READ}),
                    )
                    rbac.add_role(role)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_roles) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
