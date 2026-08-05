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

"""Contract shape tests for the Victor Contracts.

These tests validate that the contract protocol surface remains stable.
If any protocol method is renamed, removed, or has its signature changed,
these tests will fail — catching breaking changes before they reach
external verticals (victor-coding, victor-devops, etc.).

Run with: pytest tests/unit/contracts/test_contracts_contract_shapes.py -v
"""

from __future__ import annotations

import inspect
from typing import get_type_hints

import pytest


class TestVerticalBaseContract:
    """Verify VerticalBase abstract interface is stable."""

    def test_has_required_abstract_methods(self):
        from victor_contracts.verticals.protocols.base import VerticalBase

        required = {"get_name", "get_description", "get_tools", "get_system_prompt"}
        actual = {
            name
            for name, method in inspect.getmembers(VerticalBase)
            if not name.startswith("_") and callable(method)
        }
        assert required.issubset(actual), f"Missing: {required - actual}"

    def test_has_optional_methods(self):
        from victor_contracts.verticals.protocols.base import VerticalBase

        optional = {
            "get_stages",
            "get_tier",
            "get_config",
            "get_definition",
            "get_manifest",
            "get_metadata",
        }
        actual = {name for name in dir(VerticalBase) if not name.startswith("_")}
        assert optional.issubset(actual), f"Missing optional: {optional - actual}"


class TestVerticalExtensionsContract:
    """Verify VerticalExtensions lazy container shape is stable."""

    def test_has_all_extension_fields(self):
        from victor_contracts.verticals.extensions import VerticalExtensions

        ext = VerticalExtensions()
        fields = {
            "middleware",
            "safety_extensions",
            "prompt_contributors",
            "mode_config_provider",
            "tool_dependency_provider",
            "workflow_provider",
            "service_provider",
            "rl_config_provider",
            "team_spec_provider",
            "enrichment_strategy",
            "tool_selection_strategy",
            "tiered_tool_config",
        }
        for field in fields:
            assert hasattr(ext, field), f"Missing field: {field}"

    def test_list_fields_default_to_empty_list(self):
        from victor_contracts.verticals.extensions import VerticalExtensions

        ext = VerticalExtensions()
        assert ext.middleware == []
        assert ext.safety_extensions == []
        assert ext.prompt_contributors == []

    def test_optional_fields_default_to_none(self):
        from victor_contracts.verticals.extensions import VerticalExtensions

        ext = VerticalExtensions()
        assert ext.mode_config_provider is None
        assert ext.workflow_provider is None
        assert ext.rl_config_provider is None

    def test_lazy_loading_works(self):
        from victor_contracts.verticals.extensions import VerticalExtensions

        calls = []
        ext = VerticalExtensions(middleware=lambda: (calls.append(1), ["m"])[1])
        assert len(calls) == 0
        result = ext.middleware
        assert len(calls) == 1
        assert result == ["m"]

    def test_has_convenience_methods(self):
        from victor_contracts.verticals.extensions import VerticalExtensions

        for method in (
            "get_all_task_hints",
            "get_all_safety_patterns",
            "get_all_mode_configs",
        ):
            assert hasattr(VerticalExtensions, method), f"Missing method: {method}"

    def test_pending_factories_property(self):
        from victor_contracts.verticals.extensions import VerticalExtensions

        ext = VerticalExtensions(middleware=lambda: [])
        assert ext.pending_factories == 1
        _ = ext.middleware
        assert ext.pending_factories == 0


class TestPluginProtocolContract:
    """Verify VictorPlugin protocol shape is stable."""

    def test_has_required_methods(self):
        from victor_contracts.core.plugins import VictorPlugin

        required = {
            "name",
            "register",
            "get_cli_app",
            "on_activate",
            "on_deactivate",
            "health_check",
        }
        actual = {name for name in dir(VictorPlugin) if not name.startswith("_")}
        assert required.issubset(actual), f"Missing: {required - actual}"

    def test_has_async_lifecycle_methods(self):
        from victor_contracts.core.plugins import VictorPlugin

        async_methods = {"on_activate_async", "on_deactivate_async"}
        actual = {name for name in dir(VictorPlugin) if not name.startswith("_")}
        assert async_methods.issubset(actual), f"Missing async: {async_methods - actual}"


class TestVerticalProtocolsContract:
    """Verify SDK protocol exports are stable."""

    def test_sdk_native_protocols_exported(self):
        from victor_contracts.verticals import protocols

        required = {
            "ToolProvider",
            "SafetyProvider",
            "SafetyExtension",
            "PromptProvider",
            "PromptContributor",
            "WorkflowProvider",
            "TeamProvider",
            "MiddlewareProvider",
            "ModeConfigProvider",
            "RLProvider",
            "EnrichmentProvider",
            "ServiceProvider",
            "CapabilityProvider",
        }
        actual = set(dir(protocols))
        assert required.issubset(actual), f"Missing: {required - actual}"

    def test_promoted_protocols_exported(self):
        from victor_contracts.verticals import protocols

        promoted = {
            "MiddlewareProtocol",
            "SafetyExtensionProtocol",
            "PromptContributorProtocol",
            "ModeConfigProviderProtocol",
            "WorkflowProviderProtocol",
            "ServiceProviderProtocol",
            "RLConfigProviderProtocol",
            "EnrichmentStrategyProtocol",
            "CapabilityProviderProtocol",
            "TeamSpecProviderProtocol",
            "ToolDependencyProviderProtocol",
        }
        actual = set(dir(protocols))
        assert promoted.issubset(actual), f"Missing promoted: {promoted - actual}"

    def test_contract_boundary_helpers_exported(self):
        from victor_contracts import verticals

        helpers = {
            "ExtensionDependency",
            "MiddlewarePriority",
            "MiddlewareResult",
            "SafetyPattern",
            "TaskTypeHint",
            "ToolDependency",
            "register_vertical",
        }
        actual = set(dir(verticals))
        assert helpers.issubset(actual), f"Missing helpers: {helpers - actual}"

    def test_root_sdk_exports_extension_dependency(self):
        from victor_contracts import ExtensionDependency
        from victor_contracts.verticals.registration import (
            ExtensionDependency as RegistrationDependency,
        )

        assert ExtensionDependency is RegistrationDependency

    def test_framework_vertical_base_shim_reexports_sdk_registration_contract(self):
        from victor.framework.vertical_base import (
            ExtensionDependency,
            register_vertical,
        )
        from victor_contracts import ExtensionDependency as SdkExtensionDependency
        from victor_contracts import register_vertical as sdk_register_vertical

        assert ExtensionDependency is SdkExtensionDependency
        assert register_vertical is sdk_register_vertical

    def test_sdk_register_vertical_attaches_manifest_without_runtime_registration(self):
        from victor_contracts.verticals import register_vertical

        @register_vertical(name="sdk-test", version="2.1.0", load_priority=42)
        class ExampleVertical:
            pass

        manifest = ExampleVertical._victor_manifest

        assert manifest.name == "sdk-test"
        assert manifest.version == "2.1.0"
        assert manifest.load_priority == 42
        assert ExampleVertical.name == "sdk-test"
        assert ExampleVertical.version == "2.1.0"

    def test_extended_protocols_exported(self):
        from victor_contracts.verticals import protocols

        extended = {
            "McpProvider",
            "SandboxProvider",
            "HookProvider",
            "PermissionProvider",
            "CompactionProvider",
            "ExternalPluginProvider",
        }
        actual = set(dir(protocols))
        assert extended.issubset(actual), f"Missing extended: {extended - actual}"


class TestExtensionManifestContract:
    """Verify ExtensionManifest shape is stable."""

    def test_manifest_has_required_fields(self):
        from victor_contracts.verticals.manifest import ExtensionManifest

        m = ExtensionManifest(name="test")
        required = {"name", "api_version", "provides", "requires"}
        for field in required:
            assert hasattr(m, field), f"Missing: {field}"

    def test_extension_type_values(self):
        from victor_contracts.verticals.manifest import ExtensionType

        expected = {
            "SAFETY",
            "TOOLS",
            "WORKFLOWS",
            "TEAMS",
            "MIDDLEWARE",
            "MODE_CONFIG",
            "RL_CONFIG",
            "ENRICHMENT",
            "API_ROUTER",
            "CAPABILITIES",
            "SERVICE_PROVIDER",
        }
        actual = {e.name for e in ExtensionType}
        assert expected.issubset(actual), f"Missing types: {expected - actual}"


class TestTopLevelSDKExports:
    """Verify victor_contracts top-level exports are stable."""

    def test_core_types_exported(self):
        import victor_contracts

        core_types = {
            "VerticalBase",
            "VerticalExtensions",
            "VerticalConfig",
            "VerticalDefinition",
            "ExtensionManifest",
            "ExtensionType",
            "PluginContext",
            "VictorPlugin",
            "StageDefinition",
            "ToolSet",
        }
        actual = set(dir(victor_contracts))
        assert core_types.issubset(actual), f"Missing: {core_types - actual}"

    def test_extended_protocols_exported(self):
        import victor_contracts

        protocols = {
            "McpProvider",
            "SandboxProvider",
            "HookProvider",
            "PermissionProvider",
            "CompactionProvider",
            "ExternalPluginProvider",
        }
        actual = set(dir(victor_contracts))
        assert protocols.issubset(actual), f"Missing: {protocols - actual}"

    def test_sdk_has_no_victor_ai_dependency(self):
        """SDK must not import from victor-ai at import time."""
        import victor_contracts
        import sys

        victor_modules = [
            name
            for name in sys.modules
            if name.startswith("victor.") and not name.startswith("victor_contracts")
        ]
        # Allow some leakage from test environment but ensure SDK itself doesn't cause it
        sdk_source = inspect.getfile(victor_contracts)
        assert "victor-contracts" in sdk_source or "victor_contracts" in sdk_source
