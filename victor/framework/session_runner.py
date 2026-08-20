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

"""Framework-owned session preparation for CLI chat surfaces.

This module keeps runtime setup and execution-mode selection in the
framework layer so UI entrypoints stay focused on parsing and rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.framework.agent import Agent
    from victor.framework.client import VictorClient
    from victor.framework.session_config import SessionConfig
    from victor.core.container import ServiceContainer


def create_victor_client(
    config: "SessionConfig",
    *,
    container: Optional["ServiceContainer"] = None,
) -> "VictorClient":
    """Create the canonical framework client for a prepared session config."""
    from victor.framework.client import VictorClient

    return VictorClient(config, container=container)


@dataclass(frozen=True)
class PreparedSessionState:
    """Normalized session state returned to clients before execution."""

    config: "SessionConfig"
    show_reasoning: bool
    use_streaming: bool


class FrameworkSessionRunner:
    """Prepare session state and bootstrap framework client execution."""

    def __init__(
        self,
        settings: Any,
        config: "SessionConfig",
        *,
        client_factory: Optional[Callable[["SessionConfig"], "VictorClient"]] = None,
    ) -> None:
        self._settings = settings
        self._config = config
        self._client_factory = client_factory

    @property
    def settings(self) -> Any:
        """Return the mutable settings instance receiving session overrides."""
        return self._settings

    def prepare_state(
        self,
        *,
        one_shot_mode: bool,
        stream: bool,
        show_reasoning: bool,
    ) -> PreparedSessionState:
        """Normalize session mode and return execution state for the client."""
        config = self._normalize_config(one_shot_mode=one_shot_mode)
        config.apply_to_settings(self._settings)

        return PreparedSessionState(
            config=config,
            show_reasoning=self._resolve_show_reasoning(show_reasoning),
            use_streaming=bool(stream and not bool(config.planning_enabled)),
        )

    def validate_configuration(self) -> Any:
        """Validate the current settings via the canonical framework path."""
        from victor.config.validation import validate_configuration

        return validate_configuration(self._settings)

    def validate_default_model(self) -> tuple[bool, Optional[str]]:
        """Validate the resolved default model for the prepared session."""
        from victor.config.settings import validate_default_model

        return validate_default_model(self._settings)

    def create_client(
        self,
        config: Optional["SessionConfig"] = None,
    ) -> "VictorClient":
        """Create a framework client for the prepared session config."""
        factory = self._client_factory or create_victor_client
        return factory(config or self._config)

    async def initialize_client(
        self,
        client: "VictorClient",
        *,
        planning_model: Optional[str] = None,
    ) -> "Agent":
        """Initialize a client and apply agent-level planning overrides."""
        agent = await client.initialize()
        if planning_model:
            agent._planning_model_override = planning_model
        return agent

    async def start_embedding_preload(self, client: "VictorClient") -> None:
        """Start framework-managed embedding preload for the session."""
        await client.start_embedding_preload()

    def apply_agent_mode(self, mode: Optional[str]) -> None:
        """Switch the global agent mode when explicitly requested."""
        if not mode:
            return

        from victor.agent.mode_controller import AgentMode, get_mode_controller

        controller = get_mode_controller()
        controller.switch_mode(AgentMode(mode))

    def _normalize_config(self, *, one_shot_mode: bool) -> "SessionConfig":
        if self._config.one_shot_mode is one_shot_mode:
            return self._config
        return replace(self._config, one_shot_mode=one_shot_mode)

    def _resolve_show_reasoning(self, show_reasoning: bool) -> bool:
        if show_reasoning:
            return True

        provider_settings = getattr(self._settings, "provider", None)
        provider = getattr(provider_settings, "default_provider", None)
        model = getattr(provider_settings, "default_model", None)

        try:
            from victor.agent.tool_calling.capabilities import ModelCapabilityLoader

            caps = ModelCapabilityLoader().get_capabilities(provider, model)
        except Exception:
            return show_reasoning

        return bool(caps and getattr(caps, "thinking_mode", False))


__all__ = ["FrameworkSessionRunner", "PreparedSessionState", "create_victor_client"]
