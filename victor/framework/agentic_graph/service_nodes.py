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

"""Service provider nodes for StateGraph integration (Phase 3 consolidation).

This module provides nodes that access services through ExecutionContext,
enabling clean service injection into StateGraph-based execution.

Key Components:
- chat_service_node: Access chat/LLM service
- tool_service_node: Access tool execution service
- context_service_node: Access context/retrieval service
- provider_service_node: Access LLM provider service

Pattern:
    StateGraph Node (pure function)
        ├─ Receives ExecutionContext as an explicit dependency
        ├─ Calls service via ServiceAccessor
        └─ Returns updated state with service results
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.framework.agentic_graph._state_utils import normalize_state_argument
from victor.framework.agentic_graph.state import AgenticLoopStateModel

if TYPE_CHECKING:
    from victor.runtime.context import RuntimeExecutionContext

logger = logging.getLogger(__name__)


def _get_service_accessor(
    execution_context: Optional[RuntimeExecutionContext],
) -> Optional[Any]:
    """Get ServiceAccessor from an explicit execution context."""
    return execution_context.services if execution_context else None


def _get_prompt_orchestrator(execution_context: RuntimeExecutionContext) -> Any:
    """Get PromptOrchestrator from its explicit run-scoped context."""
    prompt_orchestrator = execution_context.prompt_orchestrator
    if prompt_orchestrator is None:
        raise RuntimeError("RuntimeExecutionContext has no prompt_orchestrator collaborator")
    return prompt_orchestrator


# =============================================================================
# Chat Service Node
# =============================================================================


@normalize_state_argument()
async def chat_service_node(
    state: AgenticLoopStateModel,
    message: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    execution_context: Optional[RuntimeExecutionContext] = None,
) -> AgenticLoopStateModel:
    """Chat service node: Access LLM chat service.

    This node provides access to the chat/LLM service through
    the ExecutionContext's ServiceAccessor.

    Args:
        state: Canonical agentic loop state. Legacy dict/COW inputs are
            normalized before the node body runs.
        message: Optional message to send (uses state.query if None)
        conversation_history: Optional conversation history

    Returns:
        Updated state with chat response
    """
    # Get message from parameter or state
    query = message or state.query
    if not query:
        return state

    # Get service accessor
    services = _get_service_accessor(execution_context)
    if not services or not services.chat:
        logger.warning("Chat service not available")
        return state

    try:
        # Call chat service
        response = await services.chat.chat(
            message=query,
            conversation_history=conversation_history or list(state.conversation_history or []),
        )

        # Store response in state
        context = dict(state.context or {})
        context["chat_response"] = response
        if hasattr(response, "content"):
            context["last_response"] = response.content
        elif isinstance(response, str):
            context["last_response"] = response

        state = state.model_copy(update={"context": context})
        logger.info("Chat service node: Generated response")

    except Exception as e:
        logger.warning(f"Chat service node failed: {e}")
        # Continue without chat response - not critical

    return state


# =============================================================================
# Tool Service Node
# =============================================================================


@normalize_state_argument()
async def tool_service_node(
    state: AgenticLoopStateModel,
    tool_name: str,
    tool_args: Optional[Dict[str, Any]] = None,
    execution_context: Optional[RuntimeExecutionContext] = None,
) -> AgenticLoopStateModel:
    """Tool service node: Execute tool via tool service.

    This node provides access to tool execution through the
    ExecutionContext's ServiceAccessor.

    Args:
        state: Canonical agentic loop state. Legacy dict/COW inputs are
            normalized before the node body runs.
        tool_name: Name of tool to execute
        tool_args: Arguments for the tool

    Returns:
        Updated state with tool result
    """
    # Get service accessor
    services = _get_service_accessor(execution_context)
    if not services or not services.tool:
        logger.warning(f"Tool service not available for {tool_name}")
        return state

    try:
        # Execute tool
        result = await services.tool.execute_tool(
            tool_name=tool_name,
            arguments=tool_args or {},
        )

        # Store result in state
        context = dict(state.context or {})
        tool_results = list(context.get("tool_results") or [])

        tool_result = {
            "tool": tool_name,
            "arguments": tool_args or {},
            "result": result,
        }
        tool_results.append(tool_result)
        context["tool_results"] = tool_results
        context["last_tool_result"] = tool_result

        state = state.model_copy(update={"context": context})
        logger.info(f"Tool service node: Executed {tool_name}")

    except Exception as e:
        logger.warning(f"Tool service node failed for {tool_name}: {e}")
        # Store error in state
        context = dict(state.context or {})
        tool_results = list(context.get("tool_results") or [])

        tool_result = {
            "tool": tool_name,
            "arguments": tool_args or {},
            "error": str(e),
        }
        tool_results.append(tool_result)
        context["tool_results"] = tool_results

        state = state.model_copy(update={"context": context})

    return state


# =============================================================================
# Context Service Node
# =============================================================================


@normalize_state_argument()
async def context_service_node(
    state: AgenticLoopStateModel,
    query: Optional[str] = None,
    max_results: int = 10,
    execution_context: Optional[RuntimeExecutionContext] = None,
) -> AgenticLoopStateModel:
    """Context service node: Retrieve relevant context.

    This node provides access to context/retrieval service through
    the ExecutionContext's ServiceAccessor.

    Args:
        state: Canonical agentic loop state. Legacy dict/COW inputs are
            normalized before the node body runs.
        query: Query for context retrieval (uses state.query if None)
        max_results: Maximum number of context items to retrieve

    Returns:
        Updated state with retrieved context
    """
    # Get query from parameter or state
    search_query = query or state.query
    if not search_query:
        return state

    # Get service accessor
    services = _get_service_accessor(execution_context)
    if not services or not services.context:
        logger.warning("Context service not available")
        return state

    try:
        # Retrieve context
        context_result = await services.context.retrieve_context(
            query=search_query,
            max_results=max_results,
            session_id=state.context.get("session_id") if state.context else None,
        )

        # Store in state
        context = dict(state.context or {})
        context["retrieved_context"] = context_result

        # Extract items from context result
        # Handle both callable .items() method and direct .items attribute
        if hasattr(context_result, "items") and callable(context_result.items):
            # Context result has an items() method (dict-like or result object)
            items_result = context_result.items()
            if isinstance(items_result, list):
                context["context_items"] = items_result[:max_results]
            else:
                context["context_items"] = list(items_result)[:max_results]
        elif hasattr(context_result, "items") and isinstance(context_result.items, list):
            # Context result has a direct .items list attribute
            context["context_items"] = context_result.items[:max_results]
        elif isinstance(context_result, list):
            context["context_items"] = context_result[:max_results]
        else:
            context["context_items"] = []

        state = state.model_copy(update={"context": context})
        logger.info(
            f"Context service node: Retrieved {len(context.get('context_items', []))} items"
        )

    except Exception as e:
        logger.warning(f"Context service node failed: {e}")
        # Continue without context - not critical

    return state


# =============================================================================
# Provider Service Node
# =============================================================================


@normalize_state_argument()
async def provider_service_node(
    state: AgenticLoopStateModel,
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    execution_context: Optional[RuntimeExecutionContext] = None,
) -> AgenticLoopStateModel:
    """Provider service node: Access LLM provider service.

    This node provides access to the LLM provider service through
    the ExecutionContext's ServiceAccessor.

    Args:
        state: Canonical agentic loop state. Legacy dict/COW inputs are
            normalized before the node body runs.
        provider_name: Optional provider name override
        model_name: Optional model name override

    Returns:
        Updated state with provider information
    """
    # Get service accessor
    services = _get_service_accessor(execution_context)
    if not services or not services.provider:
        logger.warning("Provider service not available")
        return state

    try:
        # Get current provider info
        provider_info = await services.provider.get_provider_info()

        # Apply overrides if provided
        if provider_name:
            provider_info["provider"] = provider_name
        if model_name:
            provider_info["model"] = model_name

        # Store in state
        context = dict(state.context or {})
        context["provider_info"] = provider_info
        context["current_provider"] = provider_info.get("provider", "")
        context["current_model"] = provider_info.get("model", "")

        state = state.model_copy(update={"context": context})
        logger.info(
            f"Provider service node: {provider_info.get('provider')}/{provider_info.get('model')}"
        )

    except Exception as e:
        logger.warning(f"Provider service node failed: {e}")
        # Continue without provider info - not critical

    return state


@normalize_state_argument()
async def prompt_service_node(
    state: AgenticLoopStateModel,
    base_prompt: Optional[str] = None,
    *,
    builder_type: str = "framework",
    constraints: Optional[Any] = None,
    vertical: str = "coding",
    execution_context: RuntimeExecutionContext,
) -> AgenticLoopStateModel:
    """Prompt facade node: build a system prompt through PromptOrchestrator.

    This node provides a framework-first entry point for prompt construction.
    It uses the PromptOrchestrator supplied by RuntimeExecutionContext. The
    composition root owns shared-service discovery; this node never reaches
    for a module-global collaborator.
    """
    prompt_orchestrator = _get_prompt_orchestrator(execution_context)
    context = dict(state.context or {})

    prompt_overlay_name = "framework.agentic_graph.system_prompt"
    existing_prompt_overlays = context.get("prompt_overlays")
    raw_existing_overlays = (
        existing_prompt_overlays
        if isinstance(existing_prompt_overlays, list)
        else [existing_prompt_overlays]
    )
    if context.get("system_prompt") or any(
        isinstance(overlay, dict) and overlay.get("name") == prompt_overlay_name
        for overlay in raw_existing_overlays
    ):
        return state

    provider = str(
        context.get("current_provider")
        or context.get("provider")
        or context.get("provider_name")
        or ""
    )
    model = str(context.get("current_model") or context.get("model") or "")
    task_type = str(context.get("task_type") or "default")
    resolved_base_prompt = (
        base_prompt if base_prompt is not None else context.get("base_prompt", "")
    )

    activated = False
    try:
        if constraints is not None:
            activated = bool(prompt_orchestrator.activate_constraints(constraints, vertical))

        prompt = prompt_orchestrator.build_system_prompt(
            builder_type=builder_type,
            provider=provider,
            model=model,
            task_type=task_type,
            base_prompt=resolved_base_prompt,
        )

        from victor.agent.prompt_pipeline import prompt_overlay_runtime_overrides

        existing_overlays = context.get("prompt_overlays")
        if isinstance(existing_overlays, list):
            prompt_overlays = list(existing_overlays)
        elif existing_overlays:
            prompt_overlays = [existing_overlays]
        else:
            prompt_overlays = []
        prompt_overlays.extend(
            prompt_overlay_runtime_overrides(
                prompt_overlay_name,
                prompt,
            )["prompt_overlays"]
        )
        context["prompt_overlays"] = prompt_overlays
        context["system_prompt_builder_type"] = builder_type
        if constraints is not None:
            context["constraints_activated"] = activated

        state = state.model_copy(update={"context": context})
        logger.info("Prompt service node: Built system prompt")
    except Exception as e:
        logger.warning(f"Prompt service node failed: {e}")
    finally:
        if constraints is not None and activated:
            try:
                prompt_orchestrator.deactivate_constraints()
            except Exception as exc:
                logger.debug("Prompt service node cleanup failed: %s", exc)

    return state
