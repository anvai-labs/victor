# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Role-Based Access Control (RBAC) for Victor tool execution.

This module provides enterprise-grade RBAC that integrates with Victor's
existing AccessMode system for tool security.

Design Principles:
- Permission model aligned with AccessMode enum (READONLY, WRITE, EXECUTE, NETWORK, MIXED)
- Category-based permission derivation (not string matching on tool names)
- YAML-configurable roles and users
- Decorator support for permission checks
- Thread-safe for concurrent access

Example Configuration (in profiles.yaml or rbac.yaml):
    rbac:
      enabled: true
      default_role: viewer
      roles:
        admin:
          permissions: [READ, WRITE, EXECUTE, NETWORK, ADMIN]
        developer:
          permissions: [READ, WRITE, EXECUTE]
          tool_categories: [filesystem, git, testing]
        viewer:
          permissions: [READ]
      users:
        alice:
          roles: [admin]
        bob:
          roles: [developer]
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.tools.base import AccessMode, BaseTool

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Permissions aligned with AccessMode for consistent security model.

    These permissions map directly to the tool AccessMode system:
    - READ: Can execute readonly tools (no side effects)
    - WRITE: Can execute tools that modify files
    - EXECUTE: Can execute shell commands/code
    - NETWORK: Can make external network calls
    - ADMIN: Full access including dangerous operations

    Additional permissions for fine-grained control:
    - TOOL_MANAGE: Can enable/disable tools
    - USER_MANAGE: Can manage users and roles
    """

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    ADMIN = "admin"
    TOOL_MANAGE = "tool_manage"
    USER_MANAGE = "user_manage"

    @classmethod
    def from_access_mode(cls, access_mode: "AccessMode") -> "Permission":
        """Convert AccessMode to corresponding Permission.

        Args:
            access_mode: Tool's access mode

        Returns:
            Required permission for that access mode
        """
        from victor.tools.base import AccessMode

        mapping = {
            AccessMode.READONLY: cls.READ,
            AccessMode.WRITE: cls.WRITE,
            AccessMode.EXECUTE: cls.EXECUTE,
            AccessMode.NETWORK: cls.NETWORK,
            AccessMode.MIXED: cls.ADMIN,  # MIXED requires full access
        }
        return mapping.get(access_mode, cls.READ)


def get_permission_for_access_mode(access_mode: "AccessMode") -> Permission:
    """Get the required permission for a given AccessMode.

    This is the primary integration point between RBAC and the tool system.

    Args:
        access_mode: Tool's AccessMode

    Returns:
        Required Permission
    """
    return Permission.from_access_mode(access_mode)


@dataclass(frozen=True)
class Role:
    """Immutable role with permissions and optional category restrictions.

    Roles can be:
    - Global: Permissions apply to all tools
    - Category-scoped: Permissions only apply to specific tool categories

    Example:
        admin_role = Role("admin", {Permission.ADMIN})
        dev_role = Role("developer", {Permission.READ, Permission.WRITE},
                        allowed_categories={"filesystem", "git"})
    """

    name: str
    permissions: FrozenSet[Permission] = field(default_factory=frozenset)
    allowed_categories: FrozenSet[str] = field(default_factory=frozenset)
    denied_tools: FrozenSet[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate and convert mutable sets to frozensets."""
        # Ensure frozensets for immutability
        if not isinstance(self.permissions, frozenset):
            object.__setattr__(self, "permissions", frozenset(self.permissions))
        if not isinstance(self.allowed_categories, frozenset):
            object.__setattr__(self, "allowed_categories", frozenset(self.allowed_categories))
        if not isinstance(self.denied_tools, frozenset):
            object.__setattr__(self, "denied_tools", frozenset(self.denied_tools))

    def has_permission(self, permission: Permission) -> bool:
        """Check if role has a specific permission.

        Args:
            permission: Permission to check

        Returns:
            True if role has the permission
        """
        # ADMIN grants all permissions
        if Permission.ADMIN in self.permissions:
            return True
        return permission in self.permissions

    def can_access_category(self, category: str) -> bool:
        """Check if role can access a tool category.

        Args:
            category: Tool category name

        Returns:
            True if role can access tools in this category
        """
        # Empty allowed_categories means all categories are allowed
        if not self.allowed_categories:
            return True
        return category in self.allowed_categories

    def can_use_tool(self, tool_name: str, category: str) -> bool:
        """Check if role can use a specific tool.

        Args:
            tool_name: Tool name
            category: Tool's category

        Returns:
            True if role can use this tool
        """
        if tool_name in self.denied_tools:
            return False
        return self.can_access_category(category)


@dataclass
class User:
    """User with assigned roles.

    Users inherit permissions from all their roles.
    Effective permissions are the union of all role permissions.

    Example:
        admin = Role("admin", {Permission.ADMIN})
        viewer = Role("viewer", {Permission.READ})
        user = User("alice", roles={admin, viewer})
    """

    name: str
    roles: Set[Role] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_effective_permissions(self) -> Set[Permission]:
        """Get all permissions from all assigned roles.

        Returns:
            Union of all role permissions
        """
        permissions: Set[Permission] = set()
        for role in self.roles:
            permissions.update(role.permissions)
        return permissions

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission.

        Args:
            permission: Permission to check

        Returns:
            True if any role grants this permission
        """
        return any(role.has_permission(permission) for role in self.roles)

    def can_use_tool(self, tool_name: str, category: str, required_permission: Permission) -> bool:
        """Check if user can use a specific tool.

        Args:
            tool_name: Tool name
            category: Tool's category
            required_permission: Permission required by the tool

        Returns:
            True if user can use this tool
        """
        for role in self.roles:
            if role.has_permission(required_permission) and role.can_use_tool(tool_name, category):
                return True
        return False


class RBACManager:
    """Thread-safe RBAC manager for Victor.

    Manages roles, users, and permission checks. Can be configured
    from YAML files or programmatically.

    Thread Safety:
        All public methods are thread-safe via internal locking.

    Example:
        rbac = RBACManager()
        rbac.add_role(Role("admin", {Permission.ADMIN}))
        rbac.add_user(User("alice", roles={rbac.get_role("admin")}))

        if rbac.check_tool_access("alice", "shell", "execute", AccessMode.EXECUTE):
            # Allow execution
    """

    # Predefined roles for convenience
    PREDEFINED_ROLES: Dict[str, Role] = {
        "admin": Role("admin", frozenset({Permission.ADMIN})),
        "developer": Role(
            "developer",
            frozenset({Permission.READ, Permission.WRITE, Permission.EXECUTE}),
        ),
        "operator": Role(
            "operator",
            frozenset({Permission.READ, Permission.EXECUTE, Permission.NETWORK}),
        ),
        "viewer": Role("viewer", frozenset({Permission.READ})),
    }

    def __init__(
        self,
        enabled: bool = True,
        default_role: str = "viewer",
        allow_unknown_users: bool = True,
        fail_open: bool = False,
    ):
        """Initialize RBAC manager.

        Args:
            enabled: If False, behaviour depends on fail_open
            default_role: Role assigned to unknown users
            allow_unknown_users: If True, unknown users get default role
            fail_open: If True, disabled RBAC allows all access.
                If False (default), disabled RBAC denies all access.
        """
        self._enabled = enabled
        self._default_role_name = default_role
        self._allow_unknown_users = allow_unknown_users
        self._fail_open = fail_open
        self._lock = threading.RLock()

        # Initialize with predefined roles
        self._roles: Dict[str, Role] = dict(self.PREDEFINED_ROLES)
        self._users: Dict[str, User] = {}

        # Current user context (set per-request)
        self._current_user: Optional[str] = None

    @property
    def enabled(self) -> bool:
        """Check if RBAC is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable RBAC checks."""
        with self._lock:
            self._enabled = True
            logger.info("RBAC enabled")

    def disable(self) -> None:
        """Disable RBAC checks (all permissions granted)."""
        with self._lock:
            self._enabled = False
            logger.info("RBAC disabled")

    def set_current_user(self, username: Optional[str]) -> None:
        """Set the current user context for permission checks.

        Args:
            username: Current user's name or None to clear
        """
        with self._lock:
            self._current_user = username

    def get_current_user(self) -> Optional[str]:
        """Get the current user context."""
        with self._lock:
            return self._current_user

    def add_role(self, role: Role) -> None:
        """Add or update a role.

        Args:
            role: Role to add
        """
        with self._lock:
            self._roles[role.name] = role
            logger.debug(f"Added role: {role.name}")

    def get_role(self, name: str) -> Optional[Role]:
        """Get a role by name.

        Args:
            name: Role name

        Returns:
            Role or None if not found
        """
        with self._lock:
            return self._roles.get(name)

    def add_user(self, user: User) -> None:
        """Add or update a user.

        Args:
            user: User to add
        """
        with self._lock:
            self._users[user.name] = user
            logger.debug(f"Added user: {user.name}")

    def get_user(self, name: str) -> Optional[User]:
        """Get a user by name.

        Args:
            name: Username

        Returns:
            User or None if not found
        """
        with self._lock:
            user = self._users.get(name)

            # Create default user if allowed
            if user is None and self._allow_unknown_users:
                default_role = self._roles.get(self._default_role_name)
                if default_role:
                    user = User(name, roles={default_role})
                    logger.debug(
                        f"Created default user '{name}' with role '{self._default_role_name}'"
                    )

            return user

    def check_permission(self, username: str, permission: Permission) -> bool:
        """Check if a user has a specific permission.

        Args:
            username: Username to check
            permission: Required permission

        Returns:
            True if permitted
        """
        if not self._enabled:
            if self._fail_open:
                return True
            logger.warning(
                "RBAC disabled with fail_open=False; denying '%s' for '%s'",
                permission.value,
                username,
            )
            return False

        with self._lock:
            user = self.get_user(username)
            if user is None:
                logger.warning(f"Unknown user '{username}' denied permission '{permission.value}'")
                return False

            result = user.has_permission(permission)
            if not result:
                logger.debug(f"User '{username}' denied permission '{permission.value}'")
            return result

    def check_tool_access(
        self,
        username: str,
        tool_name: str,
        category: str,
        access_mode: "AccessMode",
    ) -> bool:
        """Check if a user can access a specific tool.

        This is the primary method for tool execution permission checks.

        Args:
            username: Username
            tool_name: Tool name
            category: Tool's category
            access_mode: Tool's AccessMode

        Returns:
            True if user can use the tool
        """
        if not self._enabled:
            if self._fail_open:
                return True
            logger.warning(
                "RBAC disabled with fail_open=False; denying tool '%s' for '%s'",
                tool_name,
                username,
            )
            return False

        required_permission = Permission.from_access_mode(access_mode)

        with self._lock:
            user = self.get_user(username)
            if user is None:
                logger.warning(f"Unknown user '{username}' denied access to tool '{tool_name}'")
                return False

            result = user.can_use_tool(tool_name, category, required_permission)
            if not result:
                logger.debug(
                    f"User '{username}' denied access to tool '{tool_name}' "
                    f"(requires {required_permission.value})"
                )
            return result

    def check_current_user_tool_access(
        self,
        tool_name: str,
        category: str,
        access_mode: "AccessMode",
    ) -> bool:
        """Check if current user can access a tool.

        Uses the user set via set_current_user().

        Args:
            tool_name: Tool name
            category: Tool's category
            access_mode: Tool's AccessMode

        Returns:
            True if current user can use the tool
        """
        if not self._enabled:
            if self._fail_open:
                return True
            logger.warning(
                "RBAC disabled with fail_open=False; denying tool '%s' for current user",
                tool_name,
            )
            return False

        username = self.get_current_user()
        if username is None:
            # No user context - use default behavior
            return self._allow_unknown_users

        return self.check_tool_access(username, tool_name, category, access_mode)

    def load_from_dict(self, config: Dict[str, Any]) -> None:
        """Load RBAC configuration from a dictionary.

        Args:
            config: Configuration dict (from YAML)
        """
        with self._lock:
            # Load settings
            self._enabled = config.get("enabled", True)
            self._default_role_name = config.get("default_role", "viewer")
            self._allow_unknown_users = config.get("allow_unknown_users", True)

            # Load roles
            for role_name, role_data in config.get("roles", {}).items():
                permissions_list = role_data.get("permissions", [])
                permissions = frozenset(
                    Permission(p.lower())
                    for p in permissions_list
                    if hasattr(Permission, p.upper())
                )
                allowed_categories = frozenset(role_data.get("tool_categories", []))
                denied_tools = frozenset(role_data.get("denied_tools", []))

                role = Role(
                    name=role_name,
                    permissions=permissions,
                    allowed_categories=allowed_categories,
                    denied_tools=denied_tools,
                )
                self._roles[role_name] = role

            # Load users
            for user_name, user_data in config.get("users", {}).items():
                role_names = user_data.get("roles", [])
                roles = {self._roles[rn] for rn in role_names if rn in self._roles}
                metadata = user_data.get("metadata", {})

                user = User(name=user_name, roles=roles, metadata=metadata)
                self._users[user_name] = user

            logger.info(f"Loaded RBAC config: {len(self._roles)} roles, {len(self._users)} users")

    def load_from_yaml(self, path: Path) -> None:
        """Load RBAC configuration from a YAML file.

        Args:
            path: Path to YAML config file
        """
        import yaml

        with open(path, "r") as f:
            config = yaml.safe_load(f)

        rbac_config = config.get("rbac", config)
        self.load_from_dict(rbac_config)

    def get_stats(self) -> Dict[str, Any]:
        """Get RBAC statistics.

        Returns:
            Dict with role/user counts and status
        """
        with self._lock:
            return {
                "enabled": self._enabled,
                "roles_count": len(self._roles),
                "users_count": len(self._users),
                "default_role": self._default_role_name,
                "allow_unknown_users": self._allow_unknown_users,
                "current_user": self._current_user,
            }


def require_permission(permission: Permission) -> Callable:
    """Decorator to require a permission for a function.

    Example:
        @require_permission(Permission.WRITE)
        async def write_file(path: str, content: str):
            ...

    Args:
        permission: Required permission

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get RBAC manager from context if available
            rbac = kwargs.pop("_rbac_manager", None)
            username = kwargs.pop("_rbac_user", None)

            if rbac and rbac.enabled and username:
                if not rbac.check_permission(username, permission):
                    raise PermissionError(
                        f"User '{username}' lacks permission '{permission.value}' "
                        f"for {func.__name__}"
                    )

            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            rbac = kwargs.pop("_rbac_manager", None)
            username = kwargs.pop("_rbac_user", None)

            if rbac and rbac.enabled and username:
                if not rbac.check_permission(username, permission):
                    raise PermissionError(
                        f"User '{username}' lacks permission '{permission.value}' "
                        f"for {func.__name__}"
                    )

            return func(*args, **kwargs)

        # Return appropriate wrapper
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Singleton instance for convenience
_global_rbac: Optional[RBACManager] = None


def get_rbac_manager() -> RBACManager:
    """Get or create the global RBAC manager.

    Returns:
        Global RBACManager instance
    """
    global _global_rbac
    if _global_rbac is None:
        _global_rbac = RBACManager()
    return _global_rbac


def set_rbac_manager(manager: RBACManager) -> None:
    """Set the global RBAC manager.

    Args:
        manager: RBACManager to use globally
    """
    global _global_rbac
    _global_rbac = manager
