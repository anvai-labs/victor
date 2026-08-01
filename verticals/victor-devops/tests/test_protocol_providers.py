# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for victor.extension.protocols implementations in victor-devops."""

from victor_devops.protocols import (
    DevOpsHookProvider,
    DevOpsPermissionProvider,
    DevOpsPromptProvider,
    DevOpsSafetyProvider,
    DevOpsSandboxProvider,
    DevOpsToolProvider,
    DevOpsToolSelectionStrategy,
    DevOpsWorkflowProvider,
)


class TestToolProvider:
    def test_tool_list_contract(self):
        tools = DevOpsToolProvider().get_tools()

        assert len(tools) == len(set(tools)), "duplicate tools"
        for expected in ("read", "write", "shell", "docker_build", "kubectl_apply"):
            assert expected in tools

    def test_covers_all_devops_domains(self):
        tools = set(DevOpsToolProvider().get_tools())

        assert tools & {"docker_build", "docker_run", "docker_compose"}, "no docker tools"
        assert tools & {"kubectl_apply", "kubectl_get"}, "no k8s tools"
        assert tools & {"terraform_apply", "terraform_plan"}, "no IaC tools"
        assert tools & {"aws_cli", "gcloud_cli", "azure_cli"}, "no cloud tools"


class TestToolSelectionStrategy:
    def test_stage_specific_tools(self):
        strategy = DevOpsToolSelectionStrategy()

        assert "terraform_apply" in strategy.get_tools_for_stage("apply", "deploy")
        assert "prometheus_query" in strategy.get_tools_for_stage("monitor", "deploy")

    def test_unknown_stage_falls_back_to_read_shell(self):
        strategy = DevOpsToolSelectionStrategy()

        assert strategy.get_tools_for_stage("unknown_stage", "any") == ["read", "shell"]


class TestSafetyProvider:
    def test_dangerous_bash_patterns_registered(self):
        provider = DevOpsSafetyProvider()

        patterns = [p["pattern"] for p in provider.get_bash_patterns()]
        assert "terraform destroy" in patterns
        assert "kubectl delete" in patterns
        assert "rm -rf /" in patterns

    def test_extensions_include_self(self):
        provider = DevOpsSafetyProvider()

        assert provider in provider.get_extensions()

    def test_tool_restrictions_block_auto_approve(self):
        restrictions = DevOpsSafetyProvider().get_tool_restrictions()

        assert "-auto-approve" in restrictions["terraform_apply"]


class TestPromptProvider:
    def test_system_prompt_sections(self):
        sections = DevOpsPromptProvider().get_system_prompt_sections()

        assert {"role", "expertise", "safety", "best_practices"} <= set(sections)
        assert "DevOps" in sections["role"]

    def test_task_type_hints_have_budgets(self):
        hints = DevOpsPromptProvider().get_task_type_hints()

        for task_type, hint in hints.items():
            assert hint["hint"], f"{task_type} hint empty"
            assert hint["tool_budget"] > 0


class TestWorkflowProviderProtocol:
    def test_workflow_lookup_roundtrip(self):
        provider = DevOpsWorkflowProvider()

        names = provider.list_workflows()
        assert "deploy_service" in names
        workflow = provider.get_workflow("deploy_service")
        assert workflow["stages"] == ["assess", "plan", "apply", "verify"]

    def test_unknown_workflow_returns_none(self):
        assert DevOpsWorkflowProvider().get_workflow("nope") is None


class TestSandboxProvider:
    def test_sandbox_allows_docker_socket(self):
        config = DevOpsSandboxProvider().get_sandbox_config()

        assert config["enabled"] is True
        assert "/var/run/docker.sock" in config["allowed_mounts"]

    def test_deployment_tools_escape_network_isolation(self):
        overrides = DevOpsSandboxProvider().get_tool_sandbox_overrides()

        assert overrides["terraform_apply"]["network_isolation"] is False


class TestPermissionProvider:
    def test_default_mode_is_workspace_write(self):
        assert DevOpsPermissionProvider().get_permission_mode() == "workspace-write"

    def test_destructive_tools_require_danger_mode(self):
        permissions = DevOpsPermissionProvider().get_tool_permissions()

        for tool in ("terraform_destroy", "kubectl_delete", "shell"):
            assert permissions[tool] == "danger-full-access", tool

    def test_readonly_tools_stay_readonly(self):
        permissions = DevOpsPermissionProvider().get_tool_permissions()

        for tool in ("read", "grep", "terraform_plan", "kubectl_get"):
            assert permissions[tool] == "read-only", tool

    def test_escalation_rules_auto_approve_downgrades_only(self):
        rules = DevOpsPermissionProvider().get_permission_escalation_rules()

        assert rules
        for rule in rules:
            # every auto-approved rule must move toward read-only
            if rule["auto_approve"]:
                assert rule["to_mode"] == "read-only"


class TestHookProvider:
    def test_hooks_are_lists(self):
        provider = DevOpsHookProvider()

        assert provider.get_pre_tool_hooks() == []
        assert provider.get_post_tool_hooks() == []
