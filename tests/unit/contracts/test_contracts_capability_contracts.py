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

"""Compatibility tests for promoted SDK capability contracts."""

from __future__ import annotations

import sys
from types import ModuleType


def test_sdk_file_operations_capability_is_zero_dependency_contract() -> None:
    from victor_contracts.capabilities import (
        FileOperation,
        FileOperationsCapability,
        FileOperationType,
    )

    capability = FileOperationsCapability(
        operations=[
            FileOperation(FileOperationType.READ, "read"),
            FileOperation(FileOperationType.SEARCH, "grep"),
            FileOperation(FileOperationType.WRITE, "write", required=False),
        ]
    )

    assert capability.get_tools() == {"read", "grep"}
    assert capability.get_tool_list() == ["read", "grep"]


def test_sdk_prompt_contribution_capability_returns_serializable_hints() -> None:
    from victor_contracts.capabilities import (
        PromptContribution,
        PromptContributionCapability,
    )

    capability = PromptContributionCapability(
        contributions=[
            PromptContribution(
                name="read_first",
                task_type="edit",
                hint="Read before editing",
                tool_budget=5,
            ),
            PromptContribution(
                name="search_first",
                task_type="search",
                hint="Search before reading",
                tool_budget=4,
            ),
        ]
    )

    assert capability.get_task_hints() == {
        "edit": {"hint": "Read before editing", "tool_budget": 5},
        "search": {"hint": "Search before reading", "tool_budget": 4},
    }


def test_root_sdk_exports_promoted_capability_contracts() -> None:
    from victor_contracts import (
        FileOperation,
        FileOperationsCapability,
        FileOperationType,
        PromptContribution,
        PromptContributionCapability,
    )

    assert FileOperation is not None
    assert FileOperationsCapability is not None
    assert FileOperationType is not None
    assert PromptContribution is not None
    assert PromptContributionCapability is not None


def test_framework_file_operations_capability_reuses_sdk_contract() -> None:
    from victor.framework.capabilities.file_operations import (
        FileOperation,
        FileOperationsCapability,
        FileOperationType,
    )
    from victor_contracts.capabilities import (
        FileOperation as SdkFileOperation,
        FileOperationsCapability as SdkFileOperationsCapability,
        FileOperationType as SdkFileOperationType,
    )

    assert FileOperation is SdkFileOperation
    assert FileOperationType is SdkFileOperationType
    assert FileOperationsCapability is SdkFileOperationsCapability


def test_framework_prompt_contribution_capability_extends_sdk_contract() -> None:
    from victor.framework.capabilities.prompt_contributions import (
        PromptContribution,
        PromptContributionCapability,
    )
    from victor_contracts.capabilities import (
        PromptContribution as SdkPromptContribution,
        PromptContributionCapability as SdkPromptContributionCapability,
    )

    capability = PromptContributionCapability(
        contributions=[
            PromptContribution(
                name="verify",
                task_type="edit",
                hint="Verify after editing",
                tool_budget=6,
            )
        ]
    )

    assert PromptContribution is SdkPromptContribution
    assert issubclass(PromptContributionCapability, SdkPromptContributionCapability)
    assert capability.get_task_hints() == {
        "edit": {"hint": "Verify after editing", "tool_budget": 6}
    }


def test_core_vertical_types_reuse_sdk_middleware_contracts() -> None:
    from victor.core.vertical_types import MiddlewarePriority, MiddlewareResult
    from victor_contracts.verticals import (
        MiddlewarePriority as SdkMiddlewarePriority,
        MiddlewareResult as SdkMiddlewareResult,
    )

    assert MiddlewarePriority is SdkMiddlewarePriority
    assert MiddlewareResult is SdkMiddlewareResult


def test_core_task_type_hint_reuses_sdk_contract() -> None:
    from victor.core.vertical_types import TaskTypeHint
    from victor_contracts.verticals import TaskTypeHint as SdkTaskTypeHint

    hint = TaskTypeHint(task_type="edit", hint="Read before editing")

    assert TaskTypeHint is SdkTaskTypeHint
    assert hint.tool_budget is None
    assert hint.priority_tools == []


def test_core_stage_definition_exposes_sdk_compatible_fields() -> None:
    from victor.core.vertical_types import StageDefinition
    from victor_contracts import (
        StageDefinition as SdkStageDefinition,
        normalize_stage_definition,
    )

    stage = StageDefinition(
        name="EXECUTION",
        tools={"read", "edit"},
        keywords=["fix", "implement"],
        next_stages={"VERIFICATION"},
    )

    assert StageDefinition is SdkStageDefinition
    assert stage.description == ""
    assert stage.tools == {"read", "edit"}
    assert stage.required_tools == []
    assert stage.optional_tools == ["edit", "read"]
    assert stage.allow_custom_tools is True
    assert stage.to_dict()["tools"] == ["edit", "read"]

    normalized = normalize_stage_definition("EXECUTION", stage)
    assert normalized.name == "EXECUTION"
    assert normalized.optional_tools == ["edit", "read"]
    assert normalized.tools == {"read", "edit"}


def test_core_tiered_tool_config_populates_sdk_alias_fields() -> None:
    from victor.core.vertical_types import TieredToolConfig

    config = TieredToolConfig(
        mandatory={"read", "ls"},
        vertical_core={"write"},
        semantic_pool={"grep"},
    )

    assert config.basic_tools == ["ls", "read"]
    assert config.standard_tools == ["write"]
    assert config.advanced_tools == ["grep"]
    assert config.get_tools_for_tier("basic") == ["ls", "read"]
    assert config.get_tools_for_tier("standard") == ["ls", "read", "write"]
    assert config.get_max_tier_for_tools(["read", "ls", "write", "grep"]).value == "advanced"


def test_core_tiered_tool_config_accepts_sdk_tier_shape() -> None:
    from victor.core.vertical_types import TieredToolConfig

    config = TieredToolConfig(
        basic_tools=["read"],
        standard_tools=["write"],
        advanced_tools=["grep"],
    )

    assert config.mandatory == {"read"}
    assert config.vertical_core == {"write"}
    assert config.semantic_pool == {"grep"}
    assert config.get_base_tools() == {"read", "write"}


def test_core_vertical_config_exposes_sdk_compatible_helpers() -> None:
    from victor.core.verticals.base import VerticalConfig
    from victor.core.vertical_types import StageDefinition
    from victor.framework.tools import ToolSet

    config = VerticalConfig(
        name="coding",
        description="Coding assistant",
        tools=ToolSet.from_tools(["read", "write"]),
        system_prompt="You are an expert engineer.",
        stages={"EXECUTION": StageDefinition(name="EXECUTION", tools={"read", "write"})},
    )

    assert config.get_tool_names() == ["read", "write"]
    assert config.get_stage_names() == ["EXECUTION"]
    assert config.tier.value == "standard"
    assert config.extensions == {}
    payload = config.to_dict()
    assert payload["name"] == "coding"
    assert payload["description"] == "Coding assistant"
    assert payload["tools"] == ["read", "write"]


def test_core_vertical_config_with_metadata_and_extension_returns_copy() -> None:
    from victor.core.verticals.base import VerticalConfig
    from victor.framework.tools import ToolSet

    config = VerticalConfig(
        name="research",
        description="Research assistant",
        tools=ToolSet.from_tools(["read"]),
        system_prompt="Research mode",
        metadata={"source": "base"},
    )

    updated = config.with_metadata(version="2.0.0")
    extended = config.with_extension("prompt_provider", {"enabled": True})

    assert updated is not config
    assert updated.metadata == {"source": "base", "version": "2.0.0"}
    assert config.metadata == {"source": "base"}
    assert extended is not config
    assert extended.extensions == {"prompt_provider": {"enabled": True}}
    assert config.extensions == {}


def test_framework_toolset_converts_to_sdk_toolset() -> None:
    from victor.framework.tools import ToolSet
    from victor_contracts import ToolSet as SdkToolSet

    toolset = ToolSet.from_categories(["core"]).include("custom_tool")
    sdk_toolset = toolset.to_sdk_toolset(description="Runtime bridge", tier="advanced")

    assert isinstance(sdk_toolset, SdkToolSet)
    assert set(sdk_toolset.names) >= {"read", "write", "shell", "custom_tool"}
    assert sdk_toolset.description == "Runtime bridge"
    assert sdk_toolset.tier.value == "advanced"
    assert toolset.names == sorted(toolset.get_tool_names())


def test_framework_toolset_accepts_sdk_toolset_input() -> None:
    from victor.framework.tools import ToolSet
    from victor_contracts import ToolSet as SdkToolSet

    sdk_toolset = SdkToolSet(names=["read", "write"], description="SDK config")
    runtime_toolset = ToolSet.from_sdk_toolset(sdk_toolset)

    assert runtime_toolset.get_tool_names() == {"read", "write"}
    assert runtime_toolset.names == ["read", "write"]


def test_sdk_vertical_base_exposes_lazy_extension_container() -> None:
    from victor_contracts import VerticalBase, VerticalExtensions

    class _SdkVertical(VerticalBase):
        name = "sdk_lazy"
        description = "SDK lazy vertical"

        @classmethod
        def get_name(cls) -> str:
            return cls.name

        @classmethod
        def get_description(cls) -> str:
            return cls.description

        @classmethod
        def get_tools(cls) -> list[str]:
            return ["read"]

        @classmethod
        def get_system_prompt(cls) -> str:
            return "sdk"

        @classmethod
        def get_middleware(cls) -> list[object]:
            return ["middleware"]

        @classmethod
        def get_service_provider(cls) -> object:
            return {"provider": "service"}

    _SdkVertical.clear_config_cache(clear_all=True)
    extensions = _SdkVertical.get_extensions(use_cache=False)

    assert isinstance(extensions, VerticalExtensions)
    assert extensions.middleware == ["middleware"]
    assert extensions.service_provider == {"provider": "service"}


def test_sdk_vertical_base_extension_factory_uses_sdk_local_cache() -> None:
    from victor_contracts import VerticalBase

    module_name = "tests.sdk_fake_extension_module"
    fake_module = ModuleType(module_name)

    class DemoServiceProvider:
        def __init__(self) -> None:
            self.kind = "demo"

    fake_module.DemoServiceProvider = DemoServiceProvider
    sys.modules[module_name] = fake_module

    class DemoVertical(VerticalBase):
        name = "demo"
        description = "Demo vertical"

        @classmethod
        def get_name(cls) -> str:
            return cls.name

        @classmethod
        def get_description(cls) -> str:
            return cls.description

        @classmethod
        def get_tools(cls) -> list[str]:
            return ["read"]

        @classmethod
        def get_system_prompt(cls) -> str:
            return "demo"

        @classmethod
        def get_service_provider(cls):
            return cls._get_extension_factory("service_provider", module_name)

    try:
        DemoVertical.clear_config_cache(clear_all=True)
        first = DemoVertical.get_service_provider()
        second = DemoVertical.get_service_provider()

        assert isinstance(first, DemoServiceProvider)
        assert first is second
        assert first.kind == "demo"
    finally:
        sys.modules.pop(module_name, None)


def test_sdk_vertical_base_clear_config_cache_resets_extension_instances() -> None:
    from victor_contracts import VerticalBase

    created = []

    class CachedVertical(VerticalBase):
        name = "cached"
        description = "Cached vertical"

        @classmethod
        def get_name(cls) -> str:
            return cls.name

        @classmethod
        def get_description(cls) -> str:
            return cls.description

        @classmethod
        def get_tools(cls) -> list[str]:
            return ["read"]

        @classmethod
        def get_system_prompt(cls) -> str:
            return "cached"

        @classmethod
        def get_middleware(cls) -> list[object]:
            return cls._get_cached_extension(
                "middleware",
                lambda: created.append(object()) or [created[-1]],
            )

    CachedVertical.clear_config_cache(clear_all=True)
    first = CachedVertical.get_middleware()
    CachedVertical.clear_config_cache()
    second = CachedVertical.get_middleware()

    assert len(created) == 2
    assert first is not second


def test_sdk_vertical_base_get_config_accepts_use_cache_flag() -> None:
    from victor_contracts import VerticalBase

    class ConfigVertical(VerticalBase):
        name = "config"
        description = "Config vertical"
        builds = 0

        @classmethod
        def get_name(cls) -> str:
            return cls.name

        @classmethod
        def get_description(cls) -> str:
            return cls.description

        @classmethod
        def get_tools(cls) -> list[str]:
            cls.builds += 1
            return ["read"]

        @classmethod
        def get_system_prompt(cls) -> str:
            return "config"

    ConfigVertical.clear_config_cache(clear_all=True)
    first = ConfigVertical.get_config()
    cached_builds = ConfigVertical.builds
    second = ConfigVertical.get_config()

    assert first is second
    assert ConfigVertical.builds == cached_builds

    rebuilt = ConfigVertical.get_config(use_cache=False)

    assert rebuilt is not second
    assert ConfigVertical.builds > cached_builds
