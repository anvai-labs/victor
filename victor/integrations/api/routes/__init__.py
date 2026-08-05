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

"""APIRouter modules for the Victor FastAPI server.

Each module exports a ``create_router(server)`` factory that returns an
``APIRouter`` instance with routes bound to the shared server state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from fastapi import APIRouter

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

from victor.integrations.api.routes.system_routes import (
    create_router as create_system_router,
)
from victor.integrations.api.routes.chat_routes import (
    create_router as create_chat_router,
)
from victor.integrations.api.routes.search_routes import (
    create_router as create_search_router,
)
from victor.integrations.api.routes.config_routes import (
    create_router as create_config_router,
)
from victor.integrations.api.routes.tool_routes import (
    create_router as create_tool_router,
)
from victor.integrations.api.routes.conversation_routes import (
    create_router as create_conversation_router,
)
from victor.integrations.api.routes.git_routes import create_router as create_git_router
from victor.integrations.api.routes.terminal_routes import (
    create_router as create_terminal_router,
)
from victor.integrations.api.routes.workspace_routes import (
    create_router as create_workspace_router,
)
from victor.integrations.api.routes.workflow_routes import (
    create_router as create_workflow_router,
)
from victor.integrations.api.routes.rl_routes import create_router as create_rl_router
from victor.integrations.api.routes.agent_routes import (
    create_router as create_agent_router,
)
from victor.integrations.api.routes.plan_routes import (
    create_router as create_plan_router,
)
from victor.integrations.api.routes.team_routes import (
    create_router as create_team_router,
)
from victor.integrations.api.routes.mcp_routes import create_router as create_mcp_router
from victor.integrations.api.routes.lsp_routes import create_router as create_lsp_router
from victor.integrations.api.routes.websocket_routes import (
    create_router as create_websocket_router,
)
from victor.integrations.api.routes.observability_routes import (
    router as observability_router,
)

_ROUTER_FACTORIES = [
    create_system_router,
    create_chat_router,
    create_search_router,
    create_config_router,
    create_tool_router,
    create_conversation_router,
    create_git_router,
    create_terminal_router,
    create_workspace_router,
    create_workflow_router,
    create_rl_router,
    create_agent_router,
    create_plan_router,
    create_team_router,
    create_mcp_router,
    create_lsp_router,
    create_websocket_router,
]

# Direct routers (not factory functions)
_DIRECT_ROUTERS = [
    observability_router,
]


def create_all_routers(server: "VictorFastAPIServer") -> List[APIRouter]:
    """Create all route routers bound to the given server instance."""
    routers = [factory(server) for factory in _ROUTER_FACTORIES]
    # Add direct routers (not factory functions)
    routers.extend(_DIRECT_ROUTERS)
    return routers


__all__ = [
    "create_all_routers",
    "create_system_router",
    "create_chat_router",
    "create_search_router",
    "create_config_router",
    "create_tool_router",
    "create_conversation_router",
    "create_git_router",
    "create_terminal_router",
    "create_workspace_router",
    "create_workflow_router",
    "create_rl_router",
    "create_agent_router",
    "create_plan_router",
    "create_team_router",
    "create_mcp_router",
    "create_lsp_router",
    "create_websocket_router",
]
