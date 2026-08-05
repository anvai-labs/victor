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

"""Typed Tool Execution Context.

This module provides a typed context for tool execution, replacing
the Dict[str, Any] _exec_ctx parameter with a strongly-typed dataclass.

Benefits:
- Type safety and IDE autocompletion
- Clear documentation of available context fields
- Permission checking helpers
- Budget tracking methods
- Backward compatibility with dict-based contexts

Example:
    from victor.tools.context import ToolExecutionContext, Permission

    # Create typed context
    context = ToolExecutionContext(
        session_id="abc123",
        workspace_root=Path("/project"),
        user_permissions={Permission.READ_FILES, Permission.WRITE_FILES},
        tool_budget_total=25,
    )

    # Check permissions
    if context.can_write:
        await edit_file(...)

    # Track budget
    if context.use_budget(1):
        result = await run_tool(...)

    # Backward compatibility
    old_dict = context.to_dict()
    new_context = ToolExecutionContext.from_dict(old_dict)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from victor.tools.cache_manager import ToolCacheManager, CacheNamespace


class Permission(Enum):
    """Permissions for tool execution.

    Defines what operations a tool is allowed to perform.
    Used for safety enforcement and action authorization.

    Attributes:
        READ_FILES: Can read files from filesystem
        WRITE_FILES: Can create/modify files
        EXECUTE_COMMANDS: Can run shell commands
        NETWORK_ACCESS: Can make network requests
        GIT_OPERATIONS: Can run git commands
        ADMIN_OPERATIONS: Can perform admin actions
        DATABASE_ACCESS: Can query databases
        SENSITIVE_DATA: Can access sensitive data
    """

    READ_FILES = auto()
    WRITE_FILES = auto()
    EXECUTE_COMMANDS = auto()
    NETWORK_ACCESS = auto()
    GIT_OPERATIONS = auto()
    ADMIN_OPERATIONS = auto()
    DATABASE_ACCESS = auto()
    SENSITIVE_DATA = auto()


# Default permission sets for common scenarios
DEFAULT_PERMISSIONS: Set[Permission] = {Permission.READ_FILES}

SAFE_PERMISSIONS: Set[Permission] = {
    Permission.READ_FILES,
}

STANDARD_PERMISSIONS: Set[Permission] = {
    Permission.READ_FILES,
    Permission.WRITE_FILES,
    Permission.EXECUTE_COMMANDS,
    Permission.GIT_OPERATIONS,
}

FULL_PERMISSIONS: Set[Permission] = set(Permission)


@dataclass
class ToolExecutionContext:
    """Typed context for tool execution.

    Replaces Dict[str, Any] _exec_ctx parameter with a strongly-typed
    dataclass that provides type safety, documentation, and helpers.

    Attributes:
        session_id: Unique session identifier
        workspace_root: Root directory for file operations
        conversation_history: List of prior messages
        current_stage: Conversation stage (PLANNING, EXECUTING, etc.)
        tool_budget_total: Total tool budget for session
        tool_budget_used: Tools used so far
        provider_name: LLM provider name
        model_name: Model name
        provider_capabilities: Provider-specific capability flags
        user_permissions: Set of allowed permissions
        open_files: Dict of path -> content for open files
        modified_files: Set of modified file paths
        created_files: Set of created file paths
        vertical: Vertical context (coding, devops, etc.)
        metadata: Additional metadata for extensibility
        settings: Application settings reference
    """

    # Required fields
    session_id: str
    workspace_root: Path

    # Conversation state
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_stage: str = "INITIAL"

    # Budget tracking
    tool_budget_total: int = 25
    tool_budget_used: int = 0

    # Provider info
    provider_name: str = ""
    model_name: str = ""
    provider_capabilities: Dict[str, Any] = field(default_factory=dict)

    # Permissions
    user_permissions: Set[Permission] = field(default_factory=lambda: {Permission.READ_FILES})

    # File state tracking
    open_files: Dict[str, str] = field(default_factory=dict)  # path -> content
    modified_files: Set[str] = field(default_factory=set)
    created_files: Set[str] = field(default_factory=set)

    # Vertical context
    vertical: Optional[str] = None

    # Extensibility
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Settings reference (optional, for backwards compatibility)
    settings: Optional[Any] = None

    # ==========================================================================
    # DI Fields (for dependency injection)
    # ==========================================================================

    # Cache manager for tool-level caching (replaces module-level caches)
    cache_manager: Optional["ToolCacheManager"] = None

    # Path resolver for consistent path handling
    path_resolver: Optional[Any] = None

    # Logger for tool-specific logging
    logger: Optional[logging.Logger] = None

    # ==========================================================================
    # Budget Management
    # ==========================================================================

    @property
    def tool_budget_remaining(self) -> int:
        """Get remaining tool budget.

        Returns:
            Number of tool calls remaining
        """
        return max(0, self.tool_budget_total - self.tool_budget_used)

    @property
    def budget_exhausted(self) -> bool:
        """Check if budget is exhausted.

        Returns:
            True if no budget remaining
        """
        return self.tool_budget_remaining <= 0

    def use_budget(self, amount: int = 1) -> bool:
        """Use tool budget.

        Args:
            amount: Budget to consume (default 1)

        Returns:
            True if budget was available and consumed, False if insufficient
        """
        if self.tool_budget_used + amount > self.tool_budget_total:
            return False
        self.tool_budget_used += amount
        return True

    def reset_budget(self) -> None:
        """Reset used budget to zero."""
        self.tool_budget_used = 0

    # ==========================================================================
    # Permission Checks
    # ==========================================================================

    @property
    def can_read(self) -> bool:
        """Check if reading files is permitted."""
        return Permission.READ_FILES in self.user_permissions

    @property
    def can_write(self) -> bool:
        """Check if writing files is permitted."""
        return Permission.WRITE_FILES in self.user_permissions

    @property
    def can_execute(self) -> bool:
        """Check if executing commands is permitted."""
        return Permission.EXECUTE_COMMANDS in self.user_permissions

    @property
    def can_network(self) -> bool:
        """Check if network access is permitted."""
        return Permission.NETWORK_ACCESS in self.user_permissions

    @property
    def can_git(self) -> bool:
        """Check if git operations are permitted."""
        return Permission.GIT_OPERATIONS in self.user_permissions

    def has_permission(self, permission: Permission) -> bool:
        """Check if a specific permission is granted.

        Args:
            permission: Permission to check

        Returns:
            True if permission is granted
        """
        return permission in self.user_permissions

    def has_all_permissions(self, permissions: Set[Permission]) -> bool:
        """Check if all specified permissions are granted.

        Args:
            permissions: Set of permissions to check

        Returns:
            True if all permissions are granted
        """
        return permissions.issubset(self.user_permissions)

    def has_any_permission(self, permissions: Set[Permission]) -> bool:
        """Check if any of the specified permissions are granted.

        Args:
            permissions: Set of permissions to check

        Returns:
            True if at least one permission is granted
        """
        return bool(permissions & self.user_permissions)

    # ==========================================================================
    # File State Tracking
    # ==========================================================================

    def mark_file_modified(self, path: str) -> None:
        """Mark a file as modified.

        Args:
            path: Path to the modified file
        """
        self.modified_files.add(path)

    def mark_file_created(self, path: str) -> None:
        """Mark a file as created.

        Args:
            path: Path to the created file
        """
        self.created_files.add(path)

    def cache_file_content(self, path: str, content: str) -> None:
        """Cache file content for later use.

        Args:
            path: Path to the file
            content: File content
        """
        self.open_files[path] = content

    def get_cached_content(self, path: str) -> Optional[str]:
        """Get cached file content.

        Args:
            path: Path to the file

        Returns:
            Cached content or None
        """
        return self.open_files.get(path)

    # ==========================================================================
    # DI Accessors (with fallback to global state)
    # ==========================================================================

    def get_cache(self, namespace: str) -> "CacheNamespace":
        """Get a namespaced cache.

        Uses injected cache_manager if available, otherwise falls back
        to global cache manager.

        Args:
            namespace: Cache namespace name (e.g., "code_search_index")

        Returns:
            CacheNamespace for the requested namespace
        """
        if self.cache_manager is not None:
            return self.cache_manager.get_namespace(namespace)

        # Fallback to global cache manager
        from victor.tools.cache_manager import get_tool_cache_manager

        return get_tool_cache_manager().get_namespace(namespace)

    def get_path_resolver(self) -> Any:
        """Get the path resolver.

        Uses injected path_resolver if available, otherwise falls back
        to global path resolver.

        Returns:
            PathResolver instance
        """
        if self.path_resolver is not None:
            return self.path_resolver

        # Fallback to global path resolver
        from victor.tools.filesystem import get_path_resolver

        return get_path_resolver()

    def get_logger(self, name: str = "victor.tools") -> logging.Logger:
        """Get a logger for the tool.

        Uses injected logger if available, otherwise creates one.

        Args:
            name: Logger name (default: victor.tools)

        Returns:
            Logger instance
        """
        if self.logger is not None:
            return self.logger

        return logging.getLogger(name)

    @property
    def index_cache(self) -> "CacheNamespace":
        """Get code search index cache (convenience property)."""
        return self.get_cache("code_search_index")

    @property
    def file_content_cache(self) -> "CacheNamespace":
        """Get file content cache (convenience property)."""
        return self.get_cache("file_content")

    @property
    def connection_pool(self) -> "CacheNamespace":
        """Get database connection pool (convenience property)."""
        return self.get_cache("database_connections")

    # ==========================================================================
    # Backward Compatibility
    # ==========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for backward compatibility.

        Returns:
            Dict representation of context
        """
        result = {
            "session_id": self.session_id,
            "workspace_root": str(self.workspace_root),
            "conversation_history": self.conversation_history,
            "current_stage": self.current_stage,
            "tool_budget_total": self.tool_budget_total,
            "tool_budget_used": self.tool_budget_used,
            "tool_budget_remaining": self.tool_budget_remaining,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "can_read": self.can_read,
            "can_write": self.can_write,
            "can_execute": self.can_execute,
            "can_network": self.can_network,
            "can_git": self.can_git,
            "vertical": self.vertical,
            "modified_files": list(self.modified_files),
            "created_files": list(self.created_files),
        }

        # Include settings if available
        if self.settings is not None:
            result["settings"] = self.settings

        # Include metadata
        result.update(self.metadata)

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolExecutionContext":
        """Create from dict for backward compatibility.

        Args:
            data: Dict with context data

        Returns:
            ToolExecutionContext instance
        """
        # Extract known fields
        session_id = data.get("session_id", "")
        workspace_root = Path(data.get("workspace_root", "."))

        # Build permission set from boolean flags
        permissions: Set[Permission] = set()
        if data.get("can_read", True):
            permissions.add(Permission.READ_FILES)
        if data.get("can_write", False):
            permissions.add(Permission.WRITE_FILES)
        if data.get("can_execute", False):
            permissions.add(Permission.EXECUTE_COMMANDS)
        if data.get("can_network", False):
            permissions.add(Permission.NETWORK_ACCESS)
        if data.get("can_git", False):
            permissions.add(Permission.GIT_OPERATIONS)

        # Known field names to exclude from metadata
        known_fields = {
            "session_id",
            "workspace_root",
            "conversation_history",
            "current_stage",
            "tool_budget_total",
            "tool_budget_used",
            "tool_budget_remaining",
            "provider_name",
            "model_name",
            "provider_capabilities",
            "user_permissions",
            "open_files",
            "modified_files",
            "created_files",
            "vertical",
            "can_read",
            "can_write",
            "can_execute",
            "can_network",
            "can_git",
            "settings",
        }

        # Extract metadata (unknown fields)
        metadata = {k: v for k, v in data.items() if k not in known_fields}

        return cls(
            session_id=session_id,
            workspace_root=workspace_root,
            conversation_history=data.get("conversation_history", []),
            current_stage=data.get("current_stage", "INITIAL"),
            tool_budget_total=data.get("tool_budget_total", 25),
            tool_budget_used=data.get("tool_budget_used", 0),
            provider_name=data.get("provider_name", ""),
            model_name=data.get("model_name", ""),
            provider_capabilities=data.get("provider_capabilities", {}),
            user_permissions=permissions,
            open_files=data.get("open_files", {}),
            modified_files=set(data.get("modified_files", [])),
            created_files=set(data.get("created_files", [])),
            vertical=data.get("vertical"),
            metadata=metadata,
            settings=data.get("settings"),
        )

    def __repr__(self) -> str:
        return (
            f"ToolExecutionContext(session={self.session_id!r}, "
            f"stage={self.current_stage!r}, "
            f"budget={self.tool_budget_remaining}/{self.tool_budget_total})"
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_context(
    session_id: str,
    workspace_root: Path,
    *,
    permissions: Optional[Set[Permission]] = None,
    budget: int = 25,
    vertical: Optional[str] = None,
    settings: Optional[Any] = None,
) -> ToolExecutionContext:
    """Create a tool execution context with common defaults.

    Args:
        session_id: Unique session identifier
        workspace_root: Root directory for operations
        permissions: Permission set (default: STANDARD_PERMISSIONS)
        budget: Tool budget (default: 25)
        vertical: Vertical context (optional)
        settings: Settings reference (optional)

    Returns:
        Configured ToolExecutionContext
    """
    return ToolExecutionContext(
        session_id=session_id,
        workspace_root=workspace_root,
        user_permissions=permissions or STANDARD_PERMISSIONS,
        tool_budget_total=budget,
        vertical=vertical,
        settings=settings,
    )


def create_readonly_context(
    session_id: str,
    workspace_root: Path,
    *,
    budget: int = 50,
    vertical: Optional[str] = None,
) -> ToolExecutionContext:
    """Create a read-only tool execution context.

    For explore/analysis modes where file modifications are not allowed.

    Args:
        session_id: Unique session identifier
        workspace_root: Root directory
        budget: Tool budget (default: 50 for exploration)
        vertical: Vertical context (optional)

    Returns:
        Read-only ToolExecutionContext
    """
    return ToolExecutionContext(
        session_id=session_id,
        workspace_root=workspace_root,
        user_permissions=SAFE_PERMISSIONS,
        tool_budget_total=budget,
        current_stage="EXPLORE",
        vertical=vertical,
    )


def create_full_access_context(
    session_id: str,
    workspace_root: Path,
    *,
    budget: int = 100,
    vertical: Optional[str] = None,
    settings: Optional[Any] = None,
) -> ToolExecutionContext:
    """Create a full-access tool execution context.

    For trusted scenarios where all permissions are granted.

    Args:
        session_id: Unique session identifier
        workspace_root: Root directory
        budget: Tool budget (default: 100)
        vertical: Vertical context (optional)
        settings: Settings reference (optional)

    Returns:
        Full-access ToolExecutionContext
    """
    return ToolExecutionContext(
        session_id=session_id,
        workspace_root=workspace_root,
        user_permissions=FULL_PERMISSIONS,
        tool_budget_total=budget,
        vertical=vertical,
        settings=settings,
    )


__all__ = [
    # Main class
    "ToolExecutionContext",
    # Permission enum
    "Permission",
    # Permission sets
    "DEFAULT_PERMISSIONS",
    "SAFE_PERMISSIONS",
    "STANDARD_PERMISSIONS",
    "FULL_PERMISSIONS",
    # Factory functions
    "create_context",
    "create_readonly_context",
    "create_full_access_context",
]
