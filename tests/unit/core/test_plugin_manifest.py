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

"""Tests for external plugin manifest parsing and validation."""

import json

import pytest

from victor.core.plugins.manifest import (
    ManifestValidationError,
    PluginKind,
    PluginManifest,
    sanitize_plugin_id,
)


class TestPluginManifest:
    def test_valid_manifest_minimal(self):
        data = {
            "name": "test-plugin",
            "version": "1.0.0",
            "description": "A test plugin",
        }
        m = PluginManifest.from_dict(data)
        assert m.name == "test-plugin"
        assert m.version == "1.0.0"
        assert m.description == "A test plugin"
        assert m.tools == []
        assert m.commands == []
        assert not m.default_enabled

    def test_valid_manifest_with_tools(self):
        data = {
            "name": "test-plugin",
            "version": "1.0.0",
            "description": "A test plugin",
            "tools": [
                {
                    "name": "my_tool",
                    "description": "Does stuff",
                    "inputSchema": {"type": "object"},
                    "command": "echo",
                }
            ],
        }
        m = PluginManifest.from_dict(data)
        assert len(m.tools) == 1
        assert m.tools[0].name == "my_tool"
        assert m.tools[0].description == "Does stuff"
        assert m.tools[0].command == "echo"
        assert m.tools[0].input_schema == {"type": "object"}
        assert m.tools[0].required_permission == "workspace-write"

    def test_missing_required_fields(self):
        with pytest.raises(ManifestValidationError) as exc:
            PluginManifest.from_dict({})
        assert len(exc.value.errors) >= 3

    def test_missing_name_only(self):
        with pytest.raises(ManifestValidationError) as exc:
            PluginManifest.from_dict({"version": "1", "description": "d"})
        assert any("name" in e for e in exc.value.errors)

    def test_invalid_permission(self):
        data = {
            "name": "p",
            "version": "1",
            "description": "d",
            "permissions": ["invalid"],
        }
        with pytest.raises(ManifestValidationError) as exc:
            PluginManifest.from_dict(data)
        assert any("invalid" in e.lower() or "permission" in e.lower() for e in exc.value.errors)

    def test_valid_permissions(self):
        data = {
            "name": "p",
            "version": "1",
            "description": "d",
            "permissions": ["read", "write", "execute"],
        }
        m = PluginManifest.from_dict(data)
        assert m.permissions == ["read", "write", "execute"]

    def test_duplicate_permissions(self):
        data = {
            "name": "p",
            "version": "1",
            "description": "d",
            "permissions": ["read", "read"],
        }
        with pytest.raises(ManifestValidationError) as exc:
            PluginManifest.from_dict(data)
        assert any("duplicate" in e.lower() for e in exc.value.errors)

    def test_duplicate_tool_names(self):
        data = {
            "name": "p",
            "version": "1",
            "description": "d",
            "tools": [
                {"name": "t", "description": "d", "inputSchema": {}, "command": "echo"},
                {"name": "t", "description": "d", "inputSchema": {}, "command": "echo"},
            ],
        }
        with pytest.raises(ManifestValidationError) as exc:
            PluginManifest.from_dict(data)
        assert any("duplicate" in e.lower() for e in exc.value.errors)

    def test_tool_missing_command(self):
        data = {
            "name": "p",
            "version": "1",
            "description": "d",
            "tools": [
                {"name": "t", "description": "d", "inputSchema": {}},
            ],
        }
        with pytest.raises(ManifestValidationError) as exc:
            PluginManifest.from_dict(data)
        assert any("command" in e.lower() for e in exc.value.errors)

    def test_from_file(self, tmp_path):
        data = {"name": "fp", "version": "1.0", "description": "file test"}
        p = tmp_path / "plugin.json"
        p.write_text(json.dumps(data))
        m = PluginManifest.from_file(p)
        assert m.name == "fp"
        assert m.version == "1.0"

    def test_from_file_not_found(self, tmp_path):
        p = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            PluginManifest.from_file(p)

    def test_hooks_parsing(self):
        data = {
            "name": "hp",
            "version": "1",
            "description": "hooks",
            "hooks": {"PreToolUse": ["cmd1"], "PostToolUse": ["cmd2"]},
        }
        m = PluginManifest.from_dict(data)
        assert m.hooks.pre_tool_use == ["cmd1"]
        assert m.hooks.post_tool_use == ["cmd2"]

    def test_lifecycle_parsing(self):
        data = {
            "name": "lp",
            "version": "1",
            "description": "lifecycle",
            "lifecycle": {"Init": ["init.sh"], "Shutdown": ["shutdown.sh"]},
        }
        m = PluginManifest.from_dict(data)
        assert m.lifecycle.init == ["init.sh"]
        assert m.lifecycle.shutdown == ["shutdown.sh"]

    def test_default_enabled(self):
        data = {
            "name": "p",
            "version": "1",
            "description": "d",
            "defaultEnabled": True,
        }
        m = PluginManifest.from_dict(data)
        assert m.default_enabled

    def test_commands_parsing(self):
        data = {
            "name": "p",
            "version": "1",
            "description": "d",
            "commands": [
                {"name": "cmd1", "description": "first", "command": "./cmd1.sh"},
            ],
        }
        m = PluginManifest.from_dict(data)
        assert len(m.commands) == 1
        assert m.commands[0].name == "cmd1"
        assert m.commands[0].command == "./cmd1.sh"

    def test_tool_required_permission(self):
        data = {
            "name": "p",
            "version": "1",
            "description": "d",
            "tools": [
                {
                    "name": "t",
                    "description": "d",
                    "inputSchema": {},
                    "command": "echo",
                    "requiredPermission": "read-only",
                }
            ],
        }
        m = PluginManifest.from_dict(data)
        assert m.tools[0].required_permission == "read-only"

    def test_tool_invalid_required_permission(self):
        data = {
            "name": "p",
            "version": "1",
            "description": "d",
            "tools": [
                {
                    "name": "t",
                    "description": "d",
                    "inputSchema": {},
                    "command": "echo",
                    "requiredPermission": "admin",
                }
            ],
        }
        with pytest.raises(ManifestValidationError):
            PluginManifest.from_dict(data)


class TestPluginKind:
    def test_values(self):
        assert PluginKind.BUILTIN.value == "builtin"
        assert PluginKind.BUNDLED.value == "bundled"
        assert PluginKind.EXTERNAL.value == "external"


class TestSanitizePluginId:
    def test_basic(self):
        assert sanitize_plugin_id("my-plugin", "external") == "my-plugin@external"

    def test_special_chars_removed(self):
        result = sanitize_plugin_id("my/plugin@v2:latest", "bundled")
        name_part = result.split("@")[0]
        assert "/" not in name_part
        assert ":" not in name_part
        # The @ in the original name gets replaced too
        assert result.endswith("@bundled")

    def test_slashes_replaced(self):
        result = sanitize_plugin_id("org/repo", "external")
        assert result == "org-repo@external"

    def test_backslash_replaced(self):
        result = sanitize_plugin_id("path\\plugin", "builtin")
        assert result == "path-plugin@builtin"
