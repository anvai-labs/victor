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

"""Tests for the fine-grained permission hierarchy."""

import pytest

from victor.security.permissions import (
    AuthorizationDecision,
    DEFAULT_TOOL_PERMISSIONS,
    PermissionMode,
    PermissionPolicy,
)
from victor.tools.enums import (
    AccessMode,
    CostTier,
    DangerLevel,
    ExecutionCategory,
    Priority,
)
from victor.tools.metadata_registry import ToolMetadataRegistry


class TestPermissionMode:
    def test_ordering(self):
        assert (
            PermissionMode.READ_ONLY
            < PermissionMode.WORKSPACE_WRITE
            < PermissionMode.DANGER_FULL_ACCESS
        )

    def test_from_string(self):
        assert PermissionMode.from_string("read-only") == PermissionMode.READ_ONLY
        assert PermissionMode.from_string("workspace-write") == PermissionMode.WORKSPACE_WRITE
        assert PermissionMode.from_string("danger-full-access") == PermissionMode.DANGER_FULL_ACCESS
        with pytest.raises(ValueError):
            PermissionMode.from_string("invalid")

    def test_from_string_alternate_forms(self):
        assert PermissionMode.from_string("readonly") == PermissionMode.READ_ONLY
        assert PermissionMode.from_string("workspacewrite") == PermissionMode.WORKSPACE_WRITE
        assert PermissionMode.from_string("dangerfullaccess") == PermissionMode.DANGER_FULL_ACCESS

    def test_from_string_case_insensitive(self):
        assert PermissionMode.from_string("READ-ONLY") == PermissionMode.READ_ONLY
        assert PermissionMode.from_string("Workspace-Write") == PermissionMode.WORKSPACE_WRITE

    def test_from_access_mode(self):
        from victor.tools.enums import AccessMode

        assert PermissionMode.from_access_mode(AccessMode.READONLY) == PermissionMode.READ_ONLY
        assert PermissionMode.from_access_mode(AccessMode.WRITE) == PermissionMode.WORKSPACE_WRITE
        assert (
            PermissionMode.from_access_mode(AccessMode.EXECUTE) == PermissionMode.DANGER_FULL_ACCESS
        )
        assert PermissionMode.from_access_mode(AccessMode.NETWORK) == PermissionMode.READ_ONLY
        assert (
            PermissionMode.from_access_mode(AccessMode.MIXED) == PermissionMode.DANGER_FULL_ACCESS
        )

    def test_from_danger_level(self):
        from victor.tools.enums import DangerLevel

        assert PermissionMode.from_danger_level(DangerLevel.SAFE) == PermissionMode.READ_ONLY
        assert PermissionMode.from_danger_level(DangerLevel.LOW) == PermissionMode.WORKSPACE_WRITE
        assert (
            PermissionMode.from_danger_level(DangerLevel.MEDIUM) == PermissionMode.WORKSPACE_WRITE
        )
        assert (
            PermissionMode.from_danger_level(DangerLevel.HIGH) == PermissionMode.DANGER_FULL_ACCESS
        )
        assert (
            PermissionMode.from_danger_level(DangerLevel.CRITICAL)
            == PermissionMode.DANGER_FULL_ACCESS
        )

    def test_str_representation(self):
        assert str(PermissionMode.READ_ONLY) == "read-only"
        assert str(PermissionMode.WORKSPACE_WRITE) == "workspace-write"
        assert str(PermissionMode.DANGER_FULL_ACCESS) == "danger-full-access"


class TestAuthorizationDecision:
    def test_allow(self):
        d = AuthorizationDecision.allow()
        assert d.allowed
        assert not d.needs_prompt
        assert bool(d)

    def test_deny(self):
        d = AuthorizationDecision.deny("reason")
        assert not d.allowed
        assert d.reason == "reason"
        assert not bool(d)

    def test_prompt(self):
        d = AuthorizationDecision.prompt("needs approval")
        assert not d.allowed
        assert d.needs_prompt
        assert d.reason == "needs approval"

    def test_repr_allow(self):
        d = AuthorizationDecision.allow()
        assert "ALLOW" in repr(d)

    def test_repr_deny(self):
        d = AuthorizationDecision.deny("forbidden")
        assert "DENY" in repr(d)
        assert "forbidden" in repr(d)

    def test_repr_prompt(self):
        d = AuthorizationDecision.prompt("escalation needed")
        assert "PROMPT" in repr(d)
        assert "escalation needed" in repr(d)


class TestPermissionPolicy:
    def teardown_method(self):
        ToolMetadataRegistry.reset_instance()

    def test_readonly_allows_read_tools(self):
        policy = PermissionPolicy(PermissionMode.READ_ONLY)
        assert policy.authorize("read").allowed
        assert policy.authorize("grep").allowed
        assert not policy.authorize("write").allowed
        assert not policy.authorize("bash").allowed

    def test_workspace_write_allows_write_tools(self):
        policy = PermissionPolicy(PermissionMode.WORKSPACE_WRITE)
        assert policy.authorize("read").allowed
        assert policy.authorize("write").allowed
        assert policy.authorize("edit").allowed
        # bash needs escalation (DANGER required, WORKSPACE active)
        result = policy.authorize("bash")
        assert not result.allowed
        assert result.needs_prompt

    def test_danger_allows_everything(self):
        policy = PermissionPolicy(PermissionMode.DANGER_FULL_ACCESS)
        assert policy.authorize("bash").allowed
        assert policy.authorize("docker").allowed
        assert policy.authorize("read").allowed

    def test_allow_all_bypasses(self):
        policy = PermissionPolicy(PermissionMode.READ_ONLY, allow_all=True)
        assert policy.authorize("bash").allowed
        assert policy.authorize("docker").allowed

    def test_unknown_tool_defaults_to_danger(self):
        policy = PermissionPolicy(PermissionMode.WORKSPACE_WRITE)
        result = policy.authorize("unknown_exotic_tool")
        # Unknown tool requires DANGER_FULL_ACCESS, WORKSPACE_WRITE active -> prompt
        assert result.needs_prompt

    def test_unknown_tool_denied_in_readonly(self):
        policy = PermissionPolicy(PermissionMode.READ_ONLY)
        result = policy.authorize("unknown_exotic_tool")
        # Unknown tool requires DANGER_FULL_ACCESS, READ_ONLY active -> deny (no prompt)
        assert not result.allowed
        assert not result.needs_prompt

    def test_register_tool_permission(self):
        policy = PermissionPolicy(PermissionMode.READ_ONLY)
        policy.register_tool_permission("my_tool", PermissionMode.READ_ONLY)
        assert policy.authorize("my_tool").allowed

    def test_get_allowed_tools(self):
        policy = PermissionPolicy(PermissionMode.READ_ONLY)
        allowed = policy.get_allowed_tools()
        assert "read" in allowed
        assert "grep" in allowed
        assert "bash" not in allowed

    def test_get_denied_tools(self):
        policy = PermissionPolicy(PermissionMode.READ_ONLY)
        denied = policy.get_denied_tools()
        assert "shell" in denied
        assert "docker" in denied
        assert "read" not in denied

    def test_get_required_permission(self):
        policy = PermissionPolicy(PermissionMode.READ_ONLY)
        assert policy.get_required_permission("read") == PermissionMode.READ_ONLY
        assert policy.get_required_permission("write") == PermissionMode.WORKSPACE_WRITE
        assert policy.get_required_permission("bash") == PermissionMode.DANGER_FULL_ACCESS
        # Unknown defaults to DANGER_FULL_ACCESS
        assert policy.get_required_permission("unknown") == PermissionMode.DANGER_FULL_ACCESS

    def test_active_mode_property(self):
        policy = PermissionPolicy(PermissionMode.READ_ONLY)
        assert policy.active_mode == PermissionMode.READ_ONLY
        policy.active_mode = PermissionMode.DANGER_FULL_ACCESS
        assert policy.active_mode == PermissionMode.DANGER_FULL_ACCESS

    def test_custom_tool_requirements(self):
        custom = {"my_tool": PermissionMode.READ_ONLY}
        policy = PermissionPolicy(PermissionMode.READ_ONLY, tool_requirements=custom)
        assert policy.authorize("my_tool").allowed
        # Default tools still work
        assert policy.authorize("read").allowed

    def test_default_tool_permissions_not_empty(self):
        assert len(DEFAULT_TOOL_PERMISSIONS) > 0
        assert "read" in DEFAULT_TOOL_PERMISSIONS
        assert "shell" in DEFAULT_TOOL_PERMISSIONS

    def test_sync_from_tool_metadata_uses_singleton_registry(self):
        registry = ToolMetadataRegistry.get_instance()

        class MetadataTool:
            name = "generated_write_tool"
            description = "Generated write tool"
            category = "testing"
            keywords = ["write"]
            stages = []
            priority = Priority.MEDIUM
            access_mode = AccessMode.WRITE
            danger_level = DangerLevel.SAFE
            cost_tier = CostTier.FREE
            aliases = set()
            mandatory_keywords = []
            task_types = []
            progress_params = []
            execution_category = ExecutionCategory.WRITE

        registry.refresh_from_tools([MetadataTool()], force=True)

        policy = PermissionPolicy(PermissionMode.READ_ONLY)
        synced = policy.sync_from_tool_metadata()

        assert synced == 1
        assert (
            policy.get_required_permission("generated_write_tool") == PermissionMode.WORKSPACE_WRITE
        )
