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

"""Runtime adapters for contract-only vertical classes.

This module bridges SDK-pure vertical definitions into the core runtime without
forcing those external packages to inherit from victor.core.verticals.base.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Type, cast

from victor.core.vertical_types import StageDefinition
from victor.core.verticals.base import VerticalBase, VerticalConfig

if TYPE_CHECKING:
    from victor.framework.tools import ToolSet
from victor_contracts.core.types import (
    StageDefinition as SdkStageDefinition,
    ToolSet as SdkToolSet,
    VerticalConfig as SdkVerticalConfig,
    normalize_stage_definition,
)
from victor_contracts.verticals.protocols.base import VerticalBase as SdkVerticalBase


def _to_runtime_stage(stage_name: str, stage: Any) -> StageDefinition:
    """Convert an SDK/declarative stage into the core runtime shape."""

    return normalize_stage_definition(stage_name, stage)


def _to_runtime_stages(stages: Dict[str, Any]) -> Dict[str, StageDefinition]:
    """Convert a stage mapping to the core runtime shape."""

    return {
        stage_name: _to_runtime_stage(stage_name, stage) for stage_name, stage in stages.items()
    }


def _to_runtime_toolset(tools: Any) -> ToolSet:
    """Convert SDK/declarative tools into the runtime ToolSet."""

    # Deferred: victor.core must not import victor.framework at module import
    # time (core -> framework layering; circular via framework._api, which
    # imports victor.core.verticals). Unmasked by the lazy victor/__init__.
    from victor.framework.tools import ToolSet

    if isinstance(tools, ToolSet):
        return tools
    if isinstance(tools, SdkToolSet):
        return ToolSet.from_sdk_toolset(tools)
    if isinstance(tools, list):
        return ToolSet.from_tools(tools)
    raise TypeError(f"Unsupported tool configuration for runtime adaptation: {type(tools)!r}")


class VerticalRuntimeAdapter(VerticalBase):
    """Adapter that wraps an SDK-pure vertical class with core runtime behavior."""

    _sdk_vertical_cls: ClassVar[Type[SdkVerticalBase]]
    _adapter_cache: ClassVar[Dict[Type[SdkVerticalBase], Type[VerticalBase]]] = {}
    _adapter_lock: ClassVar[threading.RLock] = threading.RLock()

    @classmethod
    def adapt(cls, vertical_cls: Type[SdkVerticalBase]) -> Type[VerticalBase]:
        """Return a core-runtime-compatible class for the given vertical."""

        if issubclass(vertical_cls, VerticalBase):
            return cast(Type[VerticalBase], vertical_cls)

        with cls._adapter_lock:
            cached = cls._adapter_cache.get(vertical_cls)
            if cached is not None:
                return cached

            adapter_cls = cast(
                Type[VerticalBase],
                type(
                    vertical_cls.__name__,
                    (cls,),
                    {
                        "__module__": getattr(vertical_cls, "__module__", cls.__module__),
                        "__doc__": getattr(vertical_cls, "__doc__", cls.__doc__),
                        "_sdk_vertical_cls": vertical_cls,
                        "name": getattr(vertical_cls, "name", vertical_cls.get_name()),
                        "description": getattr(
                            vertical_cls,
                            "description",
                            vertical_cls.get_description(),
                        ),
                        "version": getattr(vertical_cls, "version", vertical_cls.get_version()),
                        "VERTICAL_API_VERSION": getattr(
                            vertical_cls,
                            "VERTICAL_API_VERSION",
                            VerticalBase.VERTICAL_API_VERSION,
                        ),
                    },
                ),
            )
            cls._adapter_cache[vertical_cls] = adapter_cls
            return adapter_cls

    @classmethod
    def get_name(cls) -> str:
        return cls._sdk_vertical_cls.get_name()

    @classmethod
    def get_description(cls) -> str:
        return cls._sdk_vertical_cls.get_description()

    @classmethod
    def get_version(cls) -> str:
        return cls._sdk_vertical_cls.get_version()

    @classmethod
    def get_tools(cls) -> List[str]:
        return list(cls._sdk_vertical_cls.get_tools())

    @classmethod
    def get_system_prompt(cls) -> str:
        return cls._sdk_vertical_cls.get_system_prompt()

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        return _to_runtime_stages(cls._sdk_vertical_cls.get_stages())

    @classmethod
    def get_tier(cls) -> Any:
        return cls._sdk_vertical_cls.get_tier()

    @classmethod
    def get_tool_requirements(cls) -> List[Any]:
        return list(cls._sdk_vertical_cls.get_tool_requirements())

    @classmethod
    def get_capability_requirements(cls) -> List[Any]:
        return list(cls._sdk_vertical_cls.get_capability_requirements())

    @classmethod
    def get_prompt_templates(cls) -> Dict[str, Any]:
        return dict(cls._sdk_vertical_cls.get_prompt_templates())

    @classmethod
    def get_task_type_hints(cls) -> Dict[str, Any]:
        return dict(cls._sdk_vertical_cls.get_task_type_hints())

    @classmethod
    def get_prompt_metadata(cls) -> Any:
        return cls._sdk_vertical_cls.get_prompt_metadata()

    @classmethod
    def get_team_declarations(cls) -> Dict[str, Any]:
        return dict(cls._sdk_vertical_cls.get_team_declarations())

    @classmethod
    def get_default_team(cls) -> Optional[str]:
        return cls._sdk_vertical_cls.get_default_team()

    @classmethod
    def get_team_metadata(cls) -> Any:
        return cls._sdk_vertical_cls.get_team_metadata()

    @classmethod
    def get_initial_stage(cls) -> Optional[str]:
        return cls._sdk_vertical_cls.get_initial_stage()

    @classmethod
    def get_workflow_spec(cls) -> Dict[str, Any]:
        return dict(cls._sdk_vertical_cls.get_workflow_spec())

    @classmethod
    def get_workflow_metadata(cls) -> Any:
        return cls._sdk_vertical_cls.get_workflow_metadata()

    @classmethod
    def get_provider_hints(cls) -> Dict[str, Any]:
        return dict(cls._sdk_vertical_cls.get_provider_hints())

    @classmethod
    def get_evaluation_criteria(cls) -> List[str]:
        return list(cls._sdk_vertical_cls.get_evaluation_criteria())

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        return dict(cls._sdk_vertical_cls.get_metadata())

    @classmethod
    def get_manifest(cls) -> Any:
        return cls._sdk_vertical_cls.get_manifest()

    @classmethod
    def get_definition(cls) -> Any:
        return cls._sdk_vertical_cls.get_definition()

    @classmethod
    def get_extensions(
        cls,
        *,
        use_cache: bool = True,
        strict: Optional[bool] = None,
    ) -> Any:
        explicit_extensions = getattr(cls._sdk_vertical_cls, "get_extensions", None)
        if callable(explicit_extensions):
            return explicit_extensions(use_cache=use_cache, strict=strict)
        return super().get_extensions(use_cache=use_cache, strict=strict)

    @classmethod
    def register_tools(cls, registry: Any) -> None:
        explicit_register = getattr(cls._sdk_vertical_cls, "register_tools", None)
        if callable(explicit_register):
            explicit_register(registry)

    @classmethod
    def customize_config(cls, config: VerticalConfig) -> VerticalConfig:
        explicit_customize = getattr(cls._sdk_vertical_cls, "customize_config", None)
        if callable(explicit_customize):
            return explicit_customize(config)
        return config

    @classmethod
    def get_config(cls, *, use_cache: bool = True) -> VerticalConfig:
        """Build a core runtime config from the wrapped SDK vertical."""

        cache_key = cls._config_cache_key()

        if use_cache:
            with cls._config_cache_lock:
                cached = cls._config_cache.get(cache_key)
                if cached is not None:
                    ts = cls._config_cache_timestamps.get(cache_key, 0.0)
                    if time.time() - ts < cls._config_cache_ttl:
                        return cached
                    cls._config_cache.pop(cache_key, None)
                    cls._config_cache_timestamps.pop(cache_key, None)

        sdk_config = cls._sdk_vertical_cls.get_config()
        if isinstance(sdk_config, VerticalConfig):
            runtime_config = sdk_config
        elif isinstance(sdk_config, SdkVerticalConfig):
            runtime_config = VerticalConfig(
                name=sdk_config.name or cls.get_name(),
                description=sdk_config.description or cls.get_description(),
                tools=_to_runtime_toolset(sdk_config.tools),
                system_prompt=sdk_config.system_prompt,
                stages=(
                    _to_runtime_stages(sdk_config.stages) if sdk_config.stages else cls.get_stages()
                ),
                provider_hints=cls.get_provider_hints(),
                evaluation_criteria=cls.get_evaluation_criteria(),
                metadata={
                    "vertical_name": cls.get_name(),
                    "vertical_version": cls.get_version(),
                    "description": cls.get_description(),
                    **cls.get_metadata(),
                    **sdk_config.metadata,
                },
                tier=sdk_config.tier,
                extensions=dict(sdk_config.extensions),
            )
        else:
            raise TypeError(
                f"Unsupported config returned by SDK vertical '{cls.get_name()}': "
                f"{type(sdk_config)!r}"
            )

        runtime_config = cls.customize_config(runtime_config)

        with cls._config_cache_lock:
            cls._config_cache[cache_key] = runtime_config
            cls._config_cache_timestamps[cache_key] = time.time()
        return runtime_config


def ensure_runtime_vertical(vertical_cls: Type[Any]) -> Type[VerticalBase]:
    """Return a core-runtime-compatible vertical class."""

    if not isinstance(vertical_cls, type):
        raise TypeError(f"Expected a vertical class, got {type(vertical_cls)!r}")

    if issubclass(vertical_cls, VerticalBase):
        return vertical_cls
    if issubclass(vertical_cls, SdkVerticalBase):
        return VerticalRuntimeAdapter.adapt(vertical_cls)

    raise TypeError(f"Unsupported vertical class: {vertical_cls!r}")


__all__ = [
    "VerticalRuntimeAdapter",
    "ensure_runtime_vertical",
]
