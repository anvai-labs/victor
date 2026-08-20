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

"""Fine-grained permission hierarchy for tool execution.

Provides a three-tier permission model (ReadOnly → WorkspaceWrite → DangerFullAccess)
with per-tool permission requirements and authorization logic, integrated
with Victor's existing AccessMode/DangerLevel metadata via facade mappings.

The permission hierarchy controls which tools an agent can use based on the
active permission mode. Tools declare their minimum required permission level,
and the PermissionPolicy enforces access control with optional user prompting
for escalation.

Example:
    policy = PermissionPolicy(PermissionMode.WORKSPACE_WRITE)
    result = policy.authorize("shell", {"cmd": "ls"})
    if result.allowed:
        # Execute tool
    elif result.needs_prompt:
        # Ask user for permission escalation
    else:
        # Denied with result.reason
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

from victor.tools.core_tool_aliases import canonicalize_core_tool_name

logger = logging.getLogger(__name__)


class PermissionMode(IntEnum):
    """Permission levels ordered by increasing privilege.

    The integer ordering allows comparison: a mode with a higher value
    includes all privileges of lower modes.
    """

    READ_ONLY = 1
    WORKSPACE_WRITE = 2
    DANGER_FULL_ACCESS = 3

    @classmethod
    def from_string(cls, value: str) -> PermissionMode:
        """Parse a permission mode from a string identifier.

        Args:
            value: String like "read-only", "workspace-write", "danger-full-access".

        Returns:
            Corresponding PermissionMode.

        Raises:
            ValueError: If string is not a valid mode.
        """
        mapping = {
            "read-only": cls.READ_ONLY,
            "readonly": cls.READ_ONLY,
            "workspace-write": cls.WORKSPACE_WRITE,
            "workspacewrite": cls.WORKSPACE_WRITE,
            "danger-full-access": cls.DANGER_FULL_ACCESS,
            "dangerfullaccess": cls.DANGER_FULL_ACCESS,
        }
        normalized = value.lower().strip()
        if normalized not in mapping:
            raise ValueError(
                f"Unknown permission mode '{value}'. "
                f"Valid modes: read-only, workspace-write, danger-full-access"
            )
        return mapping[normalized]

    @classmethod
    def from_access_mode(cls, access_mode: Any) -> PermissionMode:
        """Map a victor.tools.enums.AccessMode to PermissionMode.

        This bridges the existing tool metadata system with the
        permission hierarchy, avoiding duplicate per-tool mappings.

        Args:
            access_mode: An AccessMode enum value.

        Returns:
            Corresponding PermissionMode.
        """
        name = getattr(access_mode, "value", str(access_mode)).lower()
        mapping = {
            "readonly": cls.READ_ONLY,
            "network": cls.READ_ONLY,
            "write": cls.WORKSPACE_WRITE,
            "execute": cls.DANGER_FULL_ACCESS,
            "mixed": cls.DANGER_FULL_ACCESS,
        }
        return mapping.get(name, cls.DANGER_FULL_ACCESS)

    @classmethod
    def from_danger_level(cls, danger_level: Any) -> PermissionMode:
        """Map a victor.tools.enums.DangerLevel to PermissionMode.

        Args:
            danger_level: A DangerLevel enum value.

        Returns:
            Corresponding PermissionMode.
        """
        name = getattr(danger_level, "value", str(danger_level)).lower()
        mapping = {
            "safe": cls.READ_ONLY,
            "low": cls.WORKSPACE_WRITE,
            "medium": cls.WORKSPACE_WRITE,
            "high": cls.DANGER_FULL_ACCESS,
            "critical": cls.DANGER_FULL_ACCESS,
        }
        return mapping.get(name, cls.DANGER_FULL_ACCESS)

    def __str__(self) -> str:
        names = {
            self.READ_ONLY: "read-only",
            self.WORKSPACE_WRITE: "workspace-write",
            self.DANGER_FULL_ACCESS: "danger-full-access",
        }
        return names.get(self, "unknown")


class AuthorizationDecision:
    """Result of a permission authorization check."""

    __slots__ = ("allowed", "needs_prompt", "reason")

    def __init__(
        self,
        allowed: bool,
        needs_prompt: bool = False,
        reason: str = "",
    ) -> None:
        self.allowed = allowed
        self.needs_prompt = needs_prompt
        self.reason = reason

    @classmethod
    def allow(cls) -> AuthorizationDecision:
        """Create an allow decision."""
        return cls(allowed=True)

    @classmethod
    def deny(cls, reason: str) -> AuthorizationDecision:
        """Create a deny decision with a reason."""
        return cls(allowed=False, reason=reason)

    @classmethod
    def prompt(cls, reason: str) -> AuthorizationDecision:
        """Create a decision that requires user confirmation."""
        return cls(allowed=False, needs_prompt=True, reason=reason)

    def __bool__(self) -> bool:
        return self.allowed

    def __repr__(self) -> str:
        if self.allowed:
            return "AuthorizationDecision(ALLOW)"
        if self.needs_prompt:
            return f"AuthorizationDecision(PROMPT: {self.reason})"
        return f"AuthorizationDecision(DENY: {self.reason})"


@runtime_checkable
class PermissionPrompter(Protocol):
    """Protocol for prompting the user to approve permission escalation."""

    async def prompt_for_permission(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        required_mode: PermissionMode,
        active_mode: PermissionMode,
    ) -> bool:
        """Ask the user whether to allow an escalated permission.

        Args:
            tool_name: Name of the tool requesting access.
            tool_input: Arguments being passed to the tool.
            required_mode: The permission level the tool requires.
            active_mode: The current active permission level.

        Returns:
            True if user approves, False otherwise.
        """
        ...


# Default tool permission requirements
DEFAULT_TOOL_PERMISSIONS: Dict[str, PermissionMode] = {
    # Read-only tools
    "read": PermissionMode.READ_ONLY,
    "glob": PermissionMode.READ_ONLY,
    "glob_search": PermissionMode.READ_ONLY,
    "grep": PermissionMode.READ_ONLY,
    "grep_search": PermissionMode.READ_ONLY,
    "search": PermissionMode.READ_ONLY,
    "code_search": PermissionMode.READ_ONLY,
    "symbol": PermissionMode.READ_ONLY,
    "web_fetch": PermissionMode.READ_ONLY,
    "web_search": PermissionMode.READ_ONLY,
    "sleep": PermissionMode.READ_ONLY,
    "tool_search": PermissionMode.READ_ONLY,
    "structured_output": PermissionMode.READ_ONLY,
    # Workspace-write tools
    "write": PermissionMode.WORKSPACE_WRITE,
    "edit": PermissionMode.WORKSPACE_WRITE,
    "file_editor": PermissionMode.WORKSPACE_WRITE,
    "patch": PermissionMode.WORKSPACE_WRITE,
    "notebook_edit": PermissionMode.WORKSPACE_WRITE,
    "git": PermissionMode.WORKSPACE_WRITE,
    "git_tool": PermissionMode.WORKSPACE_WRITE,
    "config": PermissionMode.WORKSPACE_WRITE,
    "todo_write": PermissionMode.WORKSPACE_WRITE,
    "scaffold": PermissionMode.WORKSPACE_WRITE,
    "refactor": PermissionMode.WORKSPACE_WRITE,
    # Danger/full-access tools
    "shell": PermissionMode.DANGER_FULL_ACCESS,
    "code_executor": PermissionMode.DANGER_FULL_ACCESS,
    "repl": PermissionMode.DANGER_FULL_ACCESS,
    "docker": PermissionMode.DANGER_FULL_ACCESS,
    "database": PermissionMode.DANGER_FULL_ACCESS,
    "agent": PermissionMode.DANGER_FULL_ACCESS,
    "pipeline": PermissionMode.DANGER_FULL_ACCESS,
}


class PermissionPolicy:
    """Enforces permission-based access control for tool execution.

    The policy compares the active permission mode against each tool's required
    permission level. When the active mode is insufficient, it can either deny
    outright or prompt the user for escalation (if a prompter is available).
    """

    def __init__(
        self,
        active_mode: PermissionMode = PermissionMode.WORKSPACE_WRITE,
        tool_requirements: Optional[Dict[str, PermissionMode]] = None,
        allow_all: bool = False,
    ) -> None:
        """Initialize the permission policy.

        Args:
            active_mode: The current permission level for the session.
            tool_requirements: Per-tool permission overrides. Falls back to
                DEFAULT_TOOL_PERMISSIONS for tools not specified.
            allow_all: If True, bypass all permission checks (for trusted contexts).
        """
        self._active_mode = active_mode
        self._tool_requirements = dict(DEFAULT_TOOL_PERMISSIONS)
        if tool_requirements:
            self._tool_requirements.update(
                {
                    canonicalize_core_tool_name(name): required
                    for name, required in tool_requirements.items()
                }
            )
        self._allow_all = allow_all

    @property
    def active_mode(self) -> PermissionMode:
        """Return the current active permission mode."""
        return self._active_mode

    @active_mode.setter
    def active_mode(self, mode: PermissionMode) -> None:
        """Update the active permission mode."""
        self._active_mode = mode

    def get_required_permission(self, tool_name: str) -> PermissionMode:
        """Get the required permission level for a tool.

        Args:
            tool_name: Name of the tool.

        Returns:
            Required PermissionMode. Defaults to DANGER_FULL_ACCESS for unknown tools.
        """
        tool_name = canonicalize_core_tool_name(tool_name)
        return self._tool_requirements.get(tool_name, PermissionMode.DANGER_FULL_ACCESS)

    def register_tool_permission(self, tool_name: str, required: PermissionMode) -> None:
        """Register or update the permission requirement for a tool.

        Args:
            tool_name: Name of the tool.
            required: Required permission level.
        """
        self._tool_requirements[canonicalize_core_tool_name(tool_name)] = required

    def sync_from_tool_metadata(self) -> int:
        """Derive tool permissions from ToolMetadataRegistry's AccessMode.

        Reads each tool's AccessMode from the existing metadata system
        and maps it to PermissionMode, avoiding duplicate per-tool
        hardcoded mappings. Existing explicit overrides are preserved.

        Returns:
            Number of tools synced from metadata.
        """
        count = 0
        try:
            from victor.tools.metadata_registry import ToolMetadataRegistry

            registry = ToolMetadataRegistry.get_instance()
            for name, meta in registry.get_all().items():
                if name in self._tool_requirements:
                    continue  # Explicit override takes precedence
                access_mode = getattr(meta, "access_mode", None)
                if access_mode is not None:
                    self._tool_requirements[name] = PermissionMode.from_access_mode(access_mode)
                    count += 1
        except (ImportError, Exception):
            logger.debug("Could not sync from ToolMetadataRegistry")
        return count

    def authorize(
        self,
        tool_name: str,
        tool_input: Optional[Dict[str, Any]] = None,
    ) -> AuthorizationDecision:
        """Check if a tool call is authorized under the current policy.

        Args:
            tool_name: Name of the tool being called.
            tool_input: Arguments being passed to the tool.

        Returns:
            AuthorizationDecision indicating allow, deny, or needs_prompt.
        """
        if self._allow_all:
            return AuthorizationDecision.allow()

        tool_name = canonicalize_core_tool_name(tool_name)
        required = self.get_required_permission(tool_name)

        if self._active_mode >= required:
            return AuthorizationDecision.allow()

        # Need escalation - check if prompting is possible
        if self._active_mode == PermissionMode.WORKSPACE_WRITE and (
            required == PermissionMode.DANGER_FULL_ACCESS
        ):
            return AuthorizationDecision.prompt(
                f"Tool '{tool_name}' requires '{required}' permission, "
                f"but active mode is '{self._active_mode}'. "
                f"User approval needed."
            )

        return AuthorizationDecision.deny(
            f"Tool '{tool_name}' requires '{required}' permission, "
            f"but active mode is '{self._active_mode}'."
        )

    def get_allowed_tools(self) -> list[str]:
        """Return tool names that are allowed under the current mode.

        Returns:
            List of tool names whose required permission <= active mode.
        """
        return [
            name
            for name, required in self._tool_requirements.items()
            if self._active_mode >= required
        ]

    def get_denied_tools(self) -> list[str]:
        """Return tool names that are denied under the current mode.

        Returns:
            List of tool names whose required permission > active mode.
        """
        return [
            name
            for name, required in self._tool_requirements.items()
            if self._active_mode < required
        ]
