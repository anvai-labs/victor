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

"""FastAPI router plugin loading for vertical API extensions.

External packages can register routers via entry points:

    [project.entry-points."victor.api_routers"]
    coding = "victor_coding.api.router_provider:get_fastapi_router_provider"

Provider contract:
- Callable with keyword `workspace_root` returning routers, or
- Object exposing `get_fastapi_routers(workspace_root=...)`.

Each returned router can be:
- APIRouterRegistration instance
- APIRouter instance (empty prefix)
- (APIRouter, "/prefix") tuple
"""

from __future__ import annotations

import logging
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)

API_ROUTER_ENTRY_POINT_GROUP = "victor.api_routers"


@dataclass(frozen=True)
class APIRouterRegistration:
    """Router registration descriptor."""

    router: Any
    prefix: str = ""
    entry_point_name: str = ""
    entry_point_value: str = ""


class FastAPIRouterProvider(Protocol):
    """Protocol for external FastAPI router providers."""

    def get_fastapi_routers(self, *, workspace_root: str) -> Sequence[Any]:
        """Return FastAPI router registrations."""


def load_fastapi_router_registrations(
    *,
    workspace_root: str,
    entry_point_group: str = API_ROUTER_ENTRY_POINT_GROUP,
) -> List[APIRouterRegistration]:
    """Load FastAPI router registrations from installed entry points."""
    registrations: List[APIRouterRegistration] = []

    try:
        from victor.framework.entry_point_registry import get_entry_point_registry

        registry = get_entry_point_registry()
        group_obj = registry.get_group(entry_point_group)

        if not group_obj:
            return registrations

        entry_points_list = list(group_obj.entry_points.values())
    except Exception as exc:
        logger.debug("Failed to discover API router entry points: %s", exc)
        return registrations

    for ep, loaded in entry_points_list:
        try:
            # Load entry point if not already loaded
            provider = loaded if loaded is not False else ep.load()
            provider = provider() if isinstance(provider, type) else provider
            raw = _invoke_provider(provider, workspace_root=workspace_root)
            normalized = _normalize_router_items(
                raw,
                entry_point_name=ep.name,
                entry_point_value=ep.value,
            )
            registrations.extend(normalized)
            if normalized:
                logger.info(
                    "Loaded %d FastAPI router(s) from entry point '%s' (%s)",
                    len(normalized),
                    ep.name,
                    ep.value,
                )
        except Exception as exc:
            logger.warning(
                "Failed to load API router provider '%s' from entry point '%s': %s",
                ep.name,
                ep.value,
                exc,
            )

    return registrations


def _invoke_provider(provider: Any, *, workspace_root: str) -> Iterable[Any]:
    """Invoke a provider object and return router-like items."""
    if hasattr(provider, "get_fastapi_routers"):
        result = provider.get_fastapi_routers(workspace_root=workspace_root)
    elif callable(provider):
        result = provider(workspace_root=workspace_root)
    else:
        raise TypeError("Provider must be callable or implement get_fastapi_routers()")

    if result is None:
        return []
    if hasattr(result, "routes"):
        return [result]
    if isinstance(result, IterableABC) and not isinstance(result, (str, bytes)):
        return result
    return [result]


def _normalize_router_items(
    raw_items: Iterable[Any],
    *,
    entry_point_name: str,
    entry_point_value: str,
) -> List[APIRouterRegistration]:
    """Normalize provider output into APIRouterRegistration objects."""
    normalized: List[APIRouterRegistration] = []

    for item in raw_items:
        if isinstance(item, APIRouterRegistration):
            normalized.append(item)
            continue

        router: Optional[Any] = None
        prefix = ""

        if isinstance(item, tuple):
            if len(item) != 2:
                raise TypeError("Router tuple must be (router, prefix)")
            router, prefix = item
        else:
            router = item

        if router is None or not hasattr(router, "routes"):
            raise TypeError("Router item must be an APIRouter-like object")

        normalized.append(
            APIRouterRegistration(
                router=router,
                prefix=str(prefix or ""),
                entry_point_name=entry_point_name,
                entry_point_value=entry_point_value,
            )
        )

    return normalized


__all__ = [
    "API_ROUTER_ENTRY_POINT_GROUP",
    "APIRouterRegistration",
    "FastAPIRouterProvider",
    "load_fastapi_router_registrations",
]
