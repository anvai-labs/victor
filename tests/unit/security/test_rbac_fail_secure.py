from __future__ import annotations

"""Tests for RBAC fail_open / fail-secure behaviour."""


from victor.security.auth.rbac import Permission, RBACManager


class TestRBACFailOpen:
    """Verify fail_open controls behaviour when RBAC is disabled."""

    def test_fail_open_true_allows_when_disabled(self):
        """fail_open=True should allow access when RBAC is disabled."""
        rbac = RBACManager(enabled=False, fail_open=True)
        assert rbac.check_permission("alice", Permission.WRITE) is True

    def test_fail_open_false_denies_when_disabled(self):
        """fail_open=False (default) must deny access when RBAC is disabled."""
        rbac = RBACManager(enabled=False, fail_open=False)
        assert rbac.check_permission("alice", Permission.WRITE) is False

    def test_default_fail_open_is_false(self):
        """Default fail_open should be False (fail-secure)."""
        rbac = RBACManager(enabled=False)
        assert rbac.check_permission("alice", Permission.WRITE) is False

    def test_enabled_rbac_ignores_fail_open(self):
        """When RBAC is enabled, fail_open has no effect."""
        rbac = RBACManager(enabled=True, fail_open=False)
        from victor.security.auth.rbac import Role, User

        role = Role("dev", frozenset({Permission.READ, Permission.WRITE}))
        rbac.add_role(role)
        rbac.add_user(User("alice", roles={role}))

        assert rbac.check_permission("alice", Permission.WRITE) is True
        assert rbac.check_permission("alice", Permission.ADMIN) is False


class TestRBACFailOpenToolAccess:
    """Verify fail_open controls check_tool_access when RBAC is disabled."""

    def test_fail_open_true_allows_tool_access(self):
        rbac = RBACManager(enabled=False, fail_open=True)
        from victor.tools.base import AccessMode

        assert rbac.check_tool_access("alice", "shell", "execute", AccessMode.EXECUTE) is True

    def test_fail_open_false_denies_tool_access(self):
        rbac = RBACManager(enabled=False, fail_open=False)
        from victor.tools.base import AccessMode

        assert rbac.check_tool_access("alice", "shell", "execute", AccessMode.EXECUTE) is False
