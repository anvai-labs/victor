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

"""Tests for shared framework vertical integration service."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from victor.core.verticals.base import VerticalBase
from victor.framework.vertical_integration import IntegrationResult
from victor.framework.vertical_service import (
    apply_vertical_configuration,
    clear_vertical_integration_pipeline_cache,
    get_vertical_integration_pipeline,
)
from victor_contracts import VerticalBase as SdkVerticalBase
from victor_contracts.constants import CapabilityIds


class DummyVertical:
    """Minimal vertical type for service-level tests."""

    name = "dummy"


class DefinitionOnlyVertical(VerticalBase):
    """Vertical that supports the definition-layer contract."""

    name = "definition_only"
    description = "Definition only test vertical"
    version = "1.0.0"

    @classmethod
    def get_tools(cls):
        return ["read", "write"]

    @classmethod
    def get_system_prompt(cls):
        return "Definition-only prompt"


class SdkDefinitionOnlyVertical(SdkVerticalBase):
    """contract-only vertical that relies on runtime adaptation."""

    name = "sdk_definition_only"
    description = "SDK definition only test vertical"
    version = "2.0.0"

    @classmethod
    def get_name(cls):
        return cls.name

    @classmethod
    def get_description(cls):
        return cls.description

    @classmethod
    def get_tools(cls):
        return ["read", "write"]

    @classmethod
    def get_system_prompt(cls):
        return "SDK definition-only prompt"

    @classmethod
    def customize_config(cls, config):
        return config.with_metadata(runtime_adapter="sdk")


class CapabilityRequirementVertical(VerticalBase):
    """Vertical that declares SDK capability requirements."""

    name = "capability_requirements"
    description = "Capability requirement test vertical"
    version = "1.0.0"

    @classmethod
    def get_tools(cls):
        return ["read", "write", "edit", "grep"]

    @classmethod
    def get_system_prompt(cls):
        return "Capability requirement prompt"

    @classmethod
    def get_capability_requirements(cls):
        return [
            CapabilityIds.FILE_OPS,
            {"capability_id": "not_registered_capability"},
        ]


class LegacyConfigOnlyVertical(VerticalBase):
    """Vertical using standard VerticalBase protocol."""

    name = "legacy_config_only"
    description = "Legacy config only vertical"
    version = "1.0.0"

    @classmethod
    def get_tools(cls):
        return ["read", "write"]

    @classmethod
    def get_system_prompt(cls):
        return "Legacy config only prompt"


class VerticalWithServiceProvider(VerticalBase):
    """Test vertical exposing service-provider extensions."""

    name = "service_vertical"
    description = "Service provider test vertical"
    version = "1.0.0"

    @classmethod
    def get_tools(cls):
        return ["read"]

    @classmethod
    def get_system_prompt(cls):
        return "Service vertical prompt"

    @classmethod
    def get_extensions(cls):
        provider = MagicMock()
        provider.get_required_services.return_value = ["svc_required"]
        provider.get_optional_services.return_value = ["svc_optional"]
        return SimpleNamespace(
            service_provider=provider,
            middleware=None,
            safety_extensions=None,
            prompt_contributors=None,
            mode_config_provider=None,
            tool_dependency_provider=None,
            enrichment_strategy=None,
            tool_selection_strategy=None,
        )


class StubOrchestrator:
    """Minimal orchestrator with required public ports for integration."""

    def __init__(self):
        self.settings = MagicMock()
        self._container = MagicMock()
        self._enabled_tools = set()
        self._vertical_context = None
        self.prompt_builder = SimpleNamespace(set_custom_prompt=lambda _prompt: None)

    def get_service_container(self):
        return self._container

    def set_enabled_tools(self, tools):
        self._enabled_tools = set(tools)

    def set_vertical_context(self, context):
        self._vertical_context = context


class TestVerticalService:
    """Tests for framework-level vertical integration service."""

    def test_get_vertical_integration_pipeline_returns_singleton(self):
        """Service should reuse one pipeline instance by default."""
        p1 = get_vertical_integration_pipeline(reset=True)
        p2 = get_vertical_integration_pipeline()
        assert p1 is p2

    def test_get_vertical_integration_pipeline_thread_safe_singleton(self):
        """Concurrent access should return one singleton pipeline instance."""
        get_vertical_integration_pipeline(reset=True)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(get_vertical_integration_pipeline) for _ in range(24)]
            instances = [f.result() for f in futures]

        assert len({id(instance) for instance in instances}) == 1

    def test_apply_vertical_configuration_uses_shared_pipeline(self):
        """apply_vertical_configuration should delegate to shared pipeline."""
        orchestrator = MagicMock()
        expected = IntegrationResult(success=True, vertical_name="dummy")

        with patch(
            "victor.framework.vertical_service.get_vertical_integration_pipeline"
        ) as mock_get:
            pipeline = MagicMock()
            pipeline.apply.return_value = expected
            mock_get.return_value = pipeline

            result = apply_vertical_configuration(orchestrator, DummyVertical, source="sdk")

        assert result is expected
        pipeline.apply.assert_called_once_with(orchestrator, DummyVertical)

    def test_apply_vertical_configuration_supports_definition_only_verticals(self):
        """Shared application should create context through the direct vertical protocol."""
        orchestrator = StubOrchestrator()
        get_vertical_integration_pipeline(reset=True)

        result = apply_vertical_configuration(
            orchestrator,
            DefinitionOnlyVertical,
            source="sdk",
        )

        assert result.success is True
        assert result.context is not None
        assert result.context.config is not None
        assert result.context.config.system_prompt == "Definition-only prompt"
        assert result.context.config.tools.tools == {"read", "write"}

    def test_apply_vertical_configuration_supports_sdk_only_verticals(self):
        """contract-only verticals should be adapted into runtime verticals transparently."""
        orchestrator = StubOrchestrator()
        get_vertical_integration_pipeline(reset=True)

        result = apply_vertical_configuration(
            orchestrator,
            SdkDefinitionOnlyVertical,
            source="sdk",
        )

        assert result.success is True
        assert result.vertical_name == "sdk_definition_only"
        assert result.context is not None
        assert result.context.config is not None
        assert result.context.config.system_prompt == "SDK definition-only prompt"
        assert result.context.config.tools.tools == {"read", "write"}
        assert result.context.config.metadata["runtime_adapter"] == "sdk"

    def test_apply_vertical_configuration_records_capability_requirement_diagnostics(
        self,
    ):
        """Capability requirements should be resolved and surfaced during application."""
        orchestrator = StubOrchestrator()
        get_vertical_integration_pipeline(reset=True)

        result = apply_vertical_configuration(
            orchestrator,
            CapabilityRequirementVertical,
            source="sdk",
        )

        assert result.success is True
        assert result.context is not None
        resolutions = result.context.get_capability_config("sdk_capability_resolutions")
        assert resolutions is not None
        assert resolutions[0]["capability_id"] == CapabilityIds.FILE_OPS
        assert resolutions[0]["available"] is True
        assert resolutions[1]["capability_id"] == "not_registered_capability"
        assert resolutions[1]["available"] is False
        assert any("not_registered_capability" in warning for warning in result.warnings)

    def test_apply_vertical_configuration_supports_legacy_config_only_verticals(self):
        """VerticalBase verticals should work with the direct protocol path."""
        orchestrator = StubOrchestrator()
        get_vertical_integration_pipeline(reset=True)

        result = apply_vertical_configuration(
            orchestrator,
            LegacyConfigOnlyVertical,
            source="sdk",
        )

        assert result.success is True
        assert result.vertical_name == "legacy_config_only"
        assert result.context is not None
        assert result.context.config is not None
        assert result.context.config.system_prompt == "Legacy config only prompt"
        assert result.context.config.tools.tools == {"read", "write"}

    def test_clear_vertical_integration_pipeline_cache_delegates_to_pipeline(self):
        """Cache clear helper should delegate to shared pipeline clear_cache()."""
        with patch(
            "victor.framework.vertical_service.get_vertical_integration_pipeline"
        ) as mock_get:
            pipeline = MagicMock()
            mock_get.return_value = pipeline

            clear_vertical_integration_pipeline_cache()

        pipeline.clear_cache.assert_called_once_with()

    def test_cli_and_sdk_paths_share_activation_helper_idempotently(self):
        """Both source paths should use same activation helper with idempotent behavior."""
        orchestrator = StubOrchestrator()
        get_vertical_integration_pipeline(reset=True)

        with patch(
            "victor.core.verticals.vertical_loader.activate_vertical_services",
            side_effect=[
                SimpleNamespace(services_registered=True),
                SimpleNamespace(services_registered=False),
            ],
        ) as mock_activate:
            cli_result = apply_vertical_configuration(
                orchestrator,
                VerticalWithServiceProvider,
                source="cli",
            )
            sdk_result = apply_vertical_configuration(
                orchestrator,
                VerticalWithServiceProvider,
                source="sdk",
            )

        expected_args = (
            orchestrator.get_service_container(),
            orchestrator.settings,
            "service_vertical",
        )
        assert mock_activate.call_count == 2
        assert mock_activate.call_args_list[0].args == expected_args
        assert mock_activate.call_args_list[1].args == expected_args
        assert cli_result.success is True
        assert sdk_result.success is True
        assert any("Registered 2 vertical services" in msg for msg in cli_result.info)
        assert any("already registered" in msg for msg in sdk_result.info)
