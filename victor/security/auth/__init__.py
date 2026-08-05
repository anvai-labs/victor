# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Authentication and Authorization module for Victor.

This module provides:
- Role-Based Access Control (RBAC) for tool execution
- Permission system integrated with tool AccessMode
- User/Role management with YAML configuration support
"""

from victor.security.auth.rbac import (
    Permission,
    Role,
    User,
    RBACManager,
    require_permission,
    get_permission_for_access_mode,
)

__all__ = [
    "Permission",
    "Role",
    "User",
    "RBACManager",
    "require_permission",
    "get_permission_for_access_mode",
]
