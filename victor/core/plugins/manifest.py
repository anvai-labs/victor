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

"""External plugin manifest and subprocess-based plugin system.

This module provides support for external plugins defined via a plugin.json
manifest file (plugin.json). External plugins execute
tools as subprocesses with JSON I/O, enabling language-agnostic extensibility.

Manifest format (plugin.json):
    {
        "name": "my-plugin",
        "version": "1.0.0",
        "description": "Description",
        "defaultEnabled": false,
        "permissions": ["read", "write", "execute"],
        "hooks": {
            "PreToolUse": ["./hooks/pre.sh"],
            "PostToolUse": ["./hooks/post.sh"]
        },
        "lifecycle": {
            "Init": ["./lifecycle/init.sh"],
            "Shutdown": ["./lifecycle/shutdown.sh"]
        },
        "tools": [
            {
                "name": "tool_name",
                "description": "What it does",
                "inputSchema": {"type": "object", ...},
                "command": "./tools/script.sh",
                "args": ["--flag"],
                "requiredPermission": "workspace-write"
            }
        ],
        "commands": [
            {
                "name": "cmd_name",
                "description": "What it does",
                "command": "./commands/script.sh"
            }
        ]
    }
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

VALID_PERMISSIONS = frozenset({"read", "write", "execute"})
VALID_REQUIRED_PERMISSIONS = frozenset({"read-only", "workspace-write", "danger-full-access"})


class PluginKind(Enum):
    """Plugin origin type."""

    BUILTIN = "builtin"
    BUNDLED = "bundled"
    EXTERNAL = "external"


@dataclass(frozen=True)
class PluginToolSpec:
    """Specification for a tool provided by an external plugin."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    command: str
    args: List[str] = field(default_factory=list)
    required_permission: str = "workspace-write"


@dataclass(frozen=True)
class PluginCommandSpec:
    """Specification for a command provided by an external plugin."""

    name: str
    description: str
    command: str


@dataclass(frozen=True)
class PluginHooksSpec:
    """Hook command lists for pre/post tool use."""

    pre_tool_use: List[str] = field(default_factory=list)
    post_tool_use: List[str] = field(default_factory=list)

    def merged_with(self, other: PluginHooksSpec) -> PluginHooksSpec:
        """Merge hook specs from another plugin."""
        return PluginHooksSpec(
            pre_tool_use=list(self.pre_tool_use) + list(other.pre_tool_use),
            post_tool_use=list(self.post_tool_use) + list(other.post_tool_use),
        )


@dataclass(frozen=True)
class PluginLifecycleSpec:
    """Lifecycle command lists for init/shutdown."""

    init: List[str] = field(default_factory=list)
    shutdown: List[str] = field(default_factory=list)


class ManifestValidationError(Exception):
    """Raised when a plugin manifest is invalid."""

    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__(f"Plugin manifest validation failed: {'; '.join(errors)}")


@dataclass
class PluginManifest:
    """Parsed and validated plugin manifest from plugin.json."""

    name: str
    version: str
    description: str
    default_enabled: bool = False
    permissions: List[str] = field(default_factory=list)
    hooks: PluginHooksSpec = field(default_factory=PluginHooksSpec)
    lifecycle: PluginLifecycleSpec = field(default_factory=PluginLifecycleSpec)
    tools: List[PluginToolSpec] = field(default_factory=list)
    commands: List[PluginCommandSpec] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: Path) -> PluginManifest:
        """Load and validate a manifest from a plugin.json file.

        Args:
            path: Path to plugin.json file.

        Returns:
            Validated PluginManifest.

        Raises:
            ManifestValidationError: If manifest is invalid.
            FileNotFoundError: If file does not exist.
        """
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data, plugin_root=path.parent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], plugin_root: Optional[Path] = None) -> PluginManifest:
        """Parse and validate manifest from a dictionary.

        Args:
            data: Raw manifest data.
            plugin_root: Root directory of the plugin (for path validation).

        Returns:
            Validated PluginManifest.

        Raises:
            ManifestValidationError: If manifest is invalid.
        """
        errors: List[str] = []

        # Required string fields
        name = data.get("name", "")
        version = data.get("version", "")
        description = data.get("description", "")
        if not name:
            errors.append("'name' field is required and cannot be empty")
        if not version:
            errors.append("'version' field is required and cannot be empty")
        if not description:
            errors.append("'description' field is required and cannot be empty")

        # Permissions validation
        permissions = data.get("permissions", [])
        seen_perms: set[str] = set()
        for perm in permissions:
            if perm not in VALID_PERMISSIONS:
                errors.append(f"Invalid permission '{perm}' (valid: {VALID_PERMISSIONS})")
            if perm in seen_perms:
                errors.append(f"Duplicate permission '{perm}'")
            seen_perms.add(perm)

        # Hooks
        hooks_data = data.get("hooks", {})
        hooks = PluginHooksSpec(
            pre_tool_use=hooks_data.get("PreToolUse", []),
            post_tool_use=hooks_data.get("PostToolUse", []),
        )

        # Lifecycle
        lifecycle_data = data.get("lifecycle", {})
        lifecycle = PluginLifecycleSpec(
            init=lifecycle_data.get("Init", []),
            shutdown=lifecycle_data.get("Shutdown", []),
        )

        # Tools validation
        tools: List[PluginToolSpec] = []
        seen_tool_names: set[str] = set()
        for tool_data in data.get("tools", []):
            tool_name = tool_data.get("name", "")
            if not tool_name:
                errors.append("Tool 'name' is required and cannot be empty")
                continue
            if tool_name in seen_tool_names:
                errors.append(f"Duplicate tool name '{tool_name}'")
            seen_tool_names.add(tool_name)

            tool_desc = tool_data.get("description", "")
            if not tool_desc:
                errors.append(f"Tool '{tool_name}' missing 'description'")

            input_schema = tool_data.get("inputSchema", {})
            if not isinstance(input_schema, dict):
                errors.append(f"Tool '{tool_name}' inputSchema must be a JSON object")

            command = tool_data.get("command", "")
            if not command:
                errors.append(f"Tool '{tool_name}' missing 'command'")
            elif plugin_root and _is_path_command(command):
                resolved = plugin_root / command
                if not resolved.exists():
                    errors.append(f"Tool '{tool_name}' command path '{command}' does not exist")

            req_perm = tool_data.get("requiredPermission", "workspace-write")
            if req_perm not in VALID_REQUIRED_PERMISSIONS:
                errors.append(f"Tool '{tool_name}' invalid requiredPermission '{req_perm}'")

            tools.append(
                PluginToolSpec(
                    name=tool_name,
                    description=tool_desc,
                    input_schema=input_schema,
                    command=command,
                    args=tool_data.get("args", []),
                    required_permission=req_perm,
                )
            )

        # Commands validation
        commands: List[PluginCommandSpec] = []
        seen_cmd_names: set[str] = set()
        for cmd_data in data.get("commands", []):
            cmd_name = cmd_data.get("name", "")
            if not cmd_name:
                errors.append("Command 'name' is required and cannot be empty")
                continue
            if cmd_name in seen_cmd_names:
                errors.append(f"Duplicate command name '{cmd_name}'")
            seen_cmd_names.add(cmd_name)

            cmd_desc = cmd_data.get("description", "")
            command = cmd_data.get("command", "")
            if not command:
                errors.append(f"Command '{cmd_name}' missing 'command'")

            commands.append(
                PluginCommandSpec(
                    name=cmd_name,
                    description=cmd_desc,
                    command=command,
                )
            )

        if errors:
            raise ManifestValidationError(errors)

        return cls(
            name=name,
            version=version,
            description=description,
            default_enabled=data.get("defaultEnabled", False),
            permissions=permissions,
            hooks=hooks,
            lifecycle=lifecycle,
            tools=tools,
            commands=commands,
        )


def sanitize_plugin_id(name: str, marketplace: str) -> str:
    """Create a sanitized plugin ID from name and marketplace.

    Args:
        name: Plugin name.
        marketplace: Plugin marketplace/origin (builtin, bundled, external).

    Returns:
        Sanitized plugin ID in format "name@marketplace".
    """
    sanitized = re.sub(r"[/\\@:]", "-", name)
    return f"{sanitized}@{marketplace}"


def _is_path_command(command: str) -> bool:
    """Check if a command string looks like a file path vs a literal command."""
    return command.startswith("./") or command.startswith("../") or command.startswith("/")
