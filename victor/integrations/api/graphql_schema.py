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

"""GraphQL schema for Victor API.

Provides a strawberry-graphql schema layer over the existing FastAPI server,
exposing queries, mutations, and subscriptions for non-Python consumers
and IDE integrations.

Requires: pip install victor[api]  (includes strawberry-graphql[fastapi])
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional

import strawberry
from strawberry.scalars import JSON

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)


# =============================================================================
# GraphQL Types
# =============================================================================


@strawberry.type
class HealthType:
    """Health check response."""

    status: str
    version: str


@strawberry.type
class StatusType:
    """Server status response."""

    connected: bool
    mode: str
    provider: str
    model: str
    workspace: str


@strawberry.type
class ChatMessageType:
    """A single chat message."""

    role: str
    content: str


@strawberry.type
class ChatResponseType:
    """Response from a chat mutation."""

    role: str
    content: str
    tool_calls: Optional[JSON] = None


@strawberry.type
class CompletionResponseType:
    """Response from a completion mutation."""

    completions: list[str]
    error: Optional[str] = None
    latency_ms: Optional[float] = None


@strawberry.type
class SearchResultType:
    """A code search result."""

    file: str
    line: int
    content: str
    score: float


@strawberry.type
class ToolInfoType:
    """Information about an available tool."""

    name: str
    description: str
    category: str


@strawberry.type
class ProviderInfoType:
    """Information about an LLM provider."""

    name: str
    models: list[str]


@strawberry.type
class AgentEventType:
    """A real-time agent execution event."""

    type: str
    content: str
    tool_name: Optional[str] = None
    error: Optional[str] = None
    timestamp: float = 0.0


# =============================================================================
# Input Types
# =============================================================================


@strawberry.input
class ChatMessageInput:
    """Input type for chat messages."""

    role: str
    content: str


# =============================================================================
# Query Resolver
# =============================================================================


def _make_query(server: VictorFastAPIServer) -> type:
    """Create Query class bound to a server instance."""

    @strawberry.type
    class Query:
        @strawberry.field
        async def health(self) -> HealthType:
            return HealthType(status="healthy", version="0.5.1")

        @strawberry.field
        async def status(self) -> StatusType:
            try:
                orchestrator = await server._get_orchestrator()
                provider_name = "unknown"
                model_name = "unknown"
                mode = "chat"

                if orchestrator.provider:
                    provider_name = getattr(orchestrator.provider, "name", "unknown")
                    model_name = getattr(orchestrator.provider, "model", "unknown")

                if (
                    hasattr(orchestrator, "adaptive_controller")
                    and orchestrator.adaptive_controller
                ):
                    mode = orchestrator.adaptive_controller.current_mode.value

                return StatusType(
                    connected=True,
                    mode=mode,
                    provider=provider_name,
                    model=model_name,
                    workspace=server.workspace_root,
                )
            except Exception:
                return StatusType(
                    connected=False,
                    mode="unknown",
                    provider="unknown",
                    model="unknown",
                    workspace=server.workspace_root,
                )

        @strawberry.field
        async def tools(self) -> list[ToolInfoType]:
            try:
                from victor.tools.registry import ToolRegistry

                registry = ToolRegistry()
                result = []
                for tool in registry.list_tools():
                    category = server._get_tool_category(tool.name)
                    result.append(
                        ToolInfoType(
                            name=tool.name,
                            description=tool.description or "",
                            category=category,
                        )
                    )
                return result
            except Exception:
                return []

        @strawberry.field
        async def providers(self) -> list[ProviderInfoType]:
            try:
                from victor.providers.registry import get_provider_registry

                registry = get_provider_registry()
                result = []
                for name in registry.list_providers():
                    result.append(ProviderInfoType(name=name, models=[]))
                return result
            except Exception:
                return []

        @strawberry.field
        async def models(self) -> list[str]:
            try:
                from victor.agent.model_switcher import get_model_switcher

                switcher = get_model_switcher()
                return [m.model_id for m in switcher.get_available_models()]
            except Exception:
                return []

        @strawberry.field
        async def rl_stats(self) -> JSON:
            try:
                from victor.framework.rl.coordinator import get_rl_coordinator

                coordinator = get_rl_coordinator()
                learner = coordinator.get_learner("model_selector")
                if not learner:
                    return {"error": "Model selector learner not available"}
                rankings = learner.get_provider_rankings()
                return {
                    "strategy": learner.strategy.value,
                    "epsilon": round(learner.epsilon, 3),
                    "total_selections": learner._total_selections,
                    "top_provider": rankings[0]["provider"] if rankings else None,
                }
            except Exception as e:
                return {"error": str(e)}

    return Query


# =============================================================================
# Mutation Resolver
# =============================================================================


def _make_mutation(server: VictorFastAPIServer) -> type:
    """Create Mutation class bound to a server instance."""

    @strawberry.type
    class Mutation:
        @strawberry.mutation
        async def chat(self, messages: list[ChatMessageInput]) -> ChatResponseType:
            if not messages:
                return ChatResponseType(role="assistant", content="", tool_calls=None)

            orchestrator = await server._get_orchestrator()
            response = await orchestrator.chat(messages[-1].content)

            content = getattr(response, "content", None) or ""
            tool_calls = getattr(response, "tool_calls", None) or []

            return ChatResponseType(
                role="assistant",
                content=content,
                tool_calls=tool_calls if tool_calls else None,
            )

        @strawberry.mutation
        async def switch_model(self, provider: str, model: str) -> bool:
            try:
                from victor.agent.model_switcher import get_model_switcher

                switcher = get_model_switcher()
                switcher.switch(provider, model)
                return True
            except Exception:
                return False

        @strawberry.mutation
        async def reset_conversation(self) -> bool:
            try:
                if server._orchestrator:
                    server._orchestrator.reset_conversation()
                return True
            except Exception:
                return False

    return Mutation


# =============================================================================
# Subscription Resolver
# =============================================================================


def _make_subscription(server: VictorFastAPIServer) -> type:
    """Create Subscription class bound to a server instance."""

    @strawberry.type
    class Subscription:
        @strawberry.subscription
        async def agent_events(self) -> AsyncGenerator[AgentEventType, None]:
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            client_id = f"graphql-sub-{id(queue)}"

            async def enqueue(message: str) -> None:
                import json

                try:
                    data = json.loads(message)
                    await queue.put(data)
                except Exception:
                    pass

            if server._event_bridge:
                server._event_bridge._broadcaster.add_client(client_id, enqueue)

            try:
                while True:
                    data = await queue.get()
                    yield AgentEventType(
                        type=data.get("type", "unknown"),
                        content=data.get("content", ""),
                        tool_name=data.get("tool_name"),
                        error=data.get("error"),
                        timestamp=data.get("timestamp", time.time()),
                    )
            finally:
                if server._event_bridge:
                    server._event_bridge._broadcaster.remove_client(client_id)

    return Subscription


# =============================================================================
# Schema Factory
# =============================================================================


def create_graphql_schema(server: VictorFastAPIServer) -> strawberry.Schema:
    """Create a strawberry GraphQL schema wired to a VictorFastAPIServer instance.

    Args:
        server: The VictorFastAPIServer instance to wrap

    Returns:
        A strawberry.Schema with query, mutation, and subscription resolvers
    """
    query_cls = _make_query(server)
    mutation_cls = _make_mutation(server)
    subscription_cls = _make_subscription(server)

    return strawberry.Schema(
        query=query_cls,
        mutation=mutation_cls,
        subscription=subscription_cls,
    )
