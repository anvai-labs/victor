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

"""Configuration routes: /model/switch, /mode/switch, /models, /providers, /capabilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from victor.integrations.api.fastapi_server import (
    SwitchModelRequest,
    SwitchModeRequest,
    SwitchProfileRequest,
)

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create configuration routes bound to *server*."""
    router = APIRouter(tags=["Configuration"])

    @router.post("/model/switch")
    async def switch_model(request: SwitchModelRequest) -> JSONResponse:
        """Switch AI model."""
        from dataclasses import replace

        current = getattr(server, "_session_config", None)
        if current is not None:
            from victor.framework.session_config import ProviderOverrideConfig

            await server.update_session_config(
                replace(
                    current,
                    provider_override=ProviderOverrideConfig.from_cli(
                        provider=request.provider,
                        model=request.model,
                    ),
                )
            )
        try:
            from victor.agent.model_switcher import get_model_switcher

            get_model_switcher().switch(request.provider, request.model)
        except Exception:
            logger.debug("Model switcher update skipped", exc_info=True)
        return JSONResponse({"success": True, "provider": request.provider, "model": request.model})

    @router.post("/mode/switch")
    async def switch_mode(request: SwitchModeRequest) -> JSONResponse:
        """Switch agent mode."""
        from dataclasses import replace

        from victor.framework.runtime_discovery import CANONICAL_AGENT_MODES

        if request.mode not in CANONICAL_AGENT_MODES:
            return JSONResponse(
                {
                    "success": False,
                    "error": f"Unsupported mode '{request.mode}'",
                    "modes": list(CANONICAL_AGENT_MODES),
                },
                status_code=400,
            )

        current = getattr(server, "_session_config", None)
        if current is not None:
            await server.update_session_config(replace(current, mode=request.mode))
        try:
            from victor.agent.mode_controller import AgentMode, get_mode_controller

            get_mode_controller().switch_mode(AgentMode(request.mode))
        except Exception:
            logger.debug("Mode controller switch skipped", exc_info=True)
        return JSONResponse({"success": True, "mode": request.mode})

    @router.post("/profile/switch")
    async def switch_profile(request: SwitchProfileRequest) -> JSONResponse:
        """Switch the active runtime profile."""
        from dataclasses import replace

        settings = getattr(server, "_settings", None)
        profiles = settings.load_profiles() if settings is not None else {}
        if request.profile not in profiles:
            return JSONResponse(
                {
                    "success": False,
                    "error": f"Unknown profile '{request.profile}'",
                    "profiles": sorted(profiles.keys()),
                },
                status_code=404,
            )

        current = getattr(server, "_session_config", None)
        if current is not None:
            from victor.framework.session_config import ProviderOverrideConfig

            await server.update_session_config(
                replace(
                    current,
                    agent_profile=request.profile,
                    provider_override=ProviderOverrideConfig(),
                )
            )
        return JSONResponse({"success": True, "profile": request.profile})

    @router.get("/config/effective")
    async def get_effective_config() -> JSONResponse:
        """Return the effective runtime config for clients."""
        from victor.framework.runtime_discovery import effective_runtime_config

        return JSONResponse(
            effective_runtime_config(
                getattr(server, "_settings", None),
                getattr(server, "_session_config", None),
            )
        )

    @router.get("/profiles")
    async def list_profiles() -> JSONResponse:
        """List configured Victor runtime profiles."""
        from victor.framework.runtime_discovery import list_runtime_profiles

        return JSONResponse(
            {
                "profiles": [
                    profile.to_dict()
                    for profile in list_runtime_profiles(getattr(server, "_settings", None))
                ]
            }
        )

    @router.get("/modes")
    async def list_modes() -> JSONResponse:
        """List supported agent modes."""
        from victor.framework.runtime_discovery import list_runtime_modes

        return JSONResponse({"modes": list_runtime_modes()})

    @router.get("/models")
    async def list_models() -> JSONResponse:
        """List available models."""
        from victor.agent.model_switcher import get_model_switcher

        switcher = get_model_switcher()
        models = switcher.get_available_models()

        return JSONResponse(
            {
                "models": [
                    {
                        "provider": m.provider,
                        "model_id": m.model_id,
                        "display_name": m.display_name,
                        "is_local": m.is_local,
                    }
                    for m in models
                ]
            }
        )

    @router.get("/providers")
    async def list_providers() -> JSONResponse:
        """List available LLM providers."""
        try:
            from victor.providers.registry import get_provider_registry

            registry = get_provider_registry()
            providers_info = []

            for provider_name in registry.list_providers():
                try:
                    provider = registry.get(provider_name)
                    providers_info.append(
                        {
                            "name": provider_name,
                            "display_name": provider_name.replace("_", " ").title(),
                            "is_local": provider_name in ("ollama", "lmstudio", "vllm"),
                            "configured": (
                                provider.is_configured()
                                if hasattr(provider, "is_configured")
                                else True
                            ),
                            "supports_tools": (
                                provider.supports_tools()
                                if hasattr(provider, "supports_tools")
                                else False
                            ),
                            "supports_streaming": (
                                provider.supports_streaming()
                                if hasattr(provider, "supports_streaming")
                                else True
                            ),
                        }
                    )
                except Exception as e:
                    logger.debug(f"Provider {provider_name} not available: {e}")
                    providers_info.append(
                        {
                            "name": provider_name,
                            "display_name": provider_name.replace("_", " ").title(),
                            "is_local": provider_name in ("ollama", "lmstudio", "vllm"),
                            "configured": False,
                            "supports_tools": False,
                            "supports_streaming": True,
                        }
                    )

            return JSONResponse({"providers": providers_info})

        except Exception as e:
            logger.exception("List providers error")
            return JSONResponse({"providers": [], "error": str(e)})

    @router.get("/capabilities")
    async def get_capabilities(
        vertical: Optional[str] = Query(None, description="Filter by vertical"),
    ) -> JSONResponse:
        """Discover all Victor capabilities."""
        try:
            from victor.ui.commands.capabilities import get_capability_discovery

            discovery = get_capability_discovery()

            if vertical:
                manifest = discovery.discover_by_vertical(vertical)
                return JSONResponse(manifest)
            else:
                manifest = discovery.discover_all()
                return JSONResponse(manifest.to_dict())

        except Exception:
            logger.exception("Capabilities discovery error")
            return JSONResponse(
                {"error": "Internal server error", "capabilities": {}}, status_code=500
            )

    @router.get("/capabilities/recommend")
    async def get_capability_recommendations(
        task_type: str = Query(..., description="Task type or short task label"),
        complexity: str = Query(..., description="Task complexity, e.g. low, medium, high"),
        mode: str = Query("build", description="Mode policy to apply"),
        vertical: Optional[str] = Query(None, description="Filter by vertical"),
    ) -> JSONResponse:
        """Recommend teams/workflows from shared framework capability catalogs."""
        try:
            from victor.ui.commands.capabilities import get_capability_discovery

            discovery = get_capability_discovery()
            payload = discovery.recommend_for_task(
                task_type=task_type,
                complexity=complexity,
                mode=mode,
                vertical=vertical,
            )
            return JSONResponse(payload)

        except Exception:
            logger.exception("Capability recommendation error")
            return JSONResponse(
                {"error": "Internal server error", "recommendations": []},
                status_code=500,
            )

    return router
