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

"""HTTP API Module for Victor.

Provides REST API endpoints for IDE integrations (VS Code, JetBrains, etc.)
and external tool access.

Components:
- VictorFastAPIServer: The single canonical FastAPI-based server (requires fastapi)
- APIMiddlewareStack: Authentication, rate limiting, and CORS middleware
- EventBridge: Real-time event streaming to WebSocket clients
- Router plugins: Entry-point based FastAPI router extensions

The legacy aiohttp-based ``VictorAPIServer`` was removed; FastAPI is now the sole HTTP
server and its routers (``victor/integrations/api/routes/``) own the full API surface.
"""

from typing import TYPE_CHECKING

# Lazy imports for optional dependencies
if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer
    from victor.integrations.api.middleware import (
        APIMiddlewareStack,
        RateLimitConfig,
        AuthConfig,
        TokenBucket,
    )
    from victor.integrations.api.event_bridge import (
        EventBroadcaster,
        BridgeEvent,
        BridgeEventType,
    )
    from victor.integrations.api.router_plugins import (
        APIRouterRegistration,
        FastAPIRouterProvider,
        load_fastapi_router_registrations,
    )


def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name == "VictorFastAPIServer":
        from victor.integrations.api.fastapi_server import VictorFastAPIServer

        return VictorFastAPIServer
    elif name in ("APIMiddlewareStack", "RateLimitConfig", "AuthConfig", "TokenBucket"):
        from victor.integrations.api import middleware

        return getattr(middleware, name)
    elif name in ("EventBroadcaster", "BridgeEvent", "BridgeEventType"):
        from victor.integrations.api import event_bridge

        return getattr(event_bridge, name)
    elif name in (
        "APIRouterRegistration",
        "FastAPIRouterProvider",
        "load_fastapi_router_registrations",
    ):
        from victor.integrations.api import router_plugins

        return getattr(router_plugins, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Server
    "VictorFastAPIServer",
    # Middleware
    "APIMiddlewareStack",
    "RateLimitConfig",
    "AuthConfig",
    "TokenBucket",
    # Event Bridge
    "EventBroadcaster",
    "BridgeEvent",
    "BridgeEventType",
    # Router plugins
    "APIRouterRegistration",
    "FastAPIRouterProvider",
    "load_fastapi_router_registrations",
]
