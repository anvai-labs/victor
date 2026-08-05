# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Service-owned context and iteration limit runtime helper."""

from __future__ import annotations

import logging
from typing import Any, Optional

from victor.core.errors import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from victor.providers.base import Message, StreamChunk

logger = logging.getLogger(__name__)


class ContextLimitRuntime:
    """Handle context overflow and hard iteration limits for service chat paths."""

    def __init__(self, runtime_host: Any) -> None:
        self._runtime = runtime_host

    async def handle_limits(
        self,
        user_message: str,
        max_total_iterations: int,
        max_context: int,
        total_iterations: int,
        last_quality_score: float,
    ) -> tuple[bool, Optional[StreamChunk]]:
        """Handle context overflow and hard iteration limits."""
        runtime = self._runtime

        if runtime._check_context_overflow(max_context):
            logger.warning("Context overflow detected. Attempting smart compaction...")
            removed = runtime._conversation_controller.smart_compact_history(
                current_query=user_message
            )
            if removed > 0:
                logger.info("Smart compaction removed %s messages", removed)
                runtime._conversation_controller.inject_compaction_context()
                return (
                    False,
                    StreamChunk(
                        content=f"\n[context] Compacted history ({removed} messages) to continue.\n"
                    ),
                )

            if runtime._check_context_overflow(max_context):
                return await self._handle_context_overflow(last_quality_score)

        if total_iterations > max_total_iterations:
            return await self._handle_iteration_limit(
                max_total_iterations=max_total_iterations,
                last_quality_score=last_quality_score,
            )

        return False, None

    async def _handle_context_overflow(
        self,
        last_quality_score: float,
    ) -> tuple[bool, StreamChunk]:
        """Force a final summary when context overflow cannot be compacted away."""
        runtime = self._runtime
        logger.warning("Still overflowing after compaction. Forcing completion.")
        completion_prompt = runtime._get_thinking_disabled_prompt(
            "Context limit reached. Summarize in 2-3 sentences."
        )
        recent_messages = (
            runtime.messages[-8:] if len(runtime.messages) > 8 else runtime.messages[:]
        )
        completion_messages = recent_messages + [Message(role="user", content=completion_prompt)]

        try:
            response = await runtime.provider.chat(
                messages=completion_messages,
                model=runtime.model,
                temperature=runtime.temperature,
                max_tokens=min(runtime.max_tokens, 1024),
                tools=None,
            )
            if response and response.content:
                sanitized = runtime.sanitizer.sanitize(response.content)
                if sanitized:
                    from victor.agent.conversation.types import (
                        MESSAGE_SOURCE_METADATA_KEY,
                        MessageSource,
                    )

                    runtime.add_message(
                        "assistant",
                        sanitized,
                        metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_RESPONSE.value},
                    )
                    runtime._record_runtime_intelligence_outcome(
                        success=True,
                        quality_score=last_quality_score,
                        user_satisfied=True,
                        completed=True,
                    )
                    return True, StreamChunk(content=sanitized, is_final=True)
        except Exception as exc:
            logger.warning("Final response after context overflow failed: %s", exc)

        runtime._record_runtime_intelligence_outcome(
            success=True,
            quality_score=last_quality_score,
            user_satisfied=True,
            completed=True,
        )
        return True, StreamChunk(content="", is_final=True)

    async def _handle_iteration_limit(
        self,
        *,
        max_total_iterations: int,
        last_quality_score: float,
    ) -> tuple[bool, StreamChunk]:
        """Force a final summary when the hard iteration budget is exceeded."""
        runtime = self._runtime
        logger.warning(
            "Hard iteration limit reached (%s). Forcing completion.",
            max_total_iterations,
        )
        iteration_prompt = runtime._get_thinking_disabled_prompt(
            "Max iterations reached. Summarize key findings in 3-4 sentences. "
            "Do NOT attempt any more tool calls."
        )
        recent_messages = (
            runtime.messages[-10:] if len(runtime.messages) > 10 else runtime.messages[:]
        )
        completion_messages = recent_messages + [Message(role="user", content=iteration_prompt)]

        chunk = StreamChunk(
            content=(
                f"\n[tool] {runtime._presentation.icon('warning', with_color=False)} "
                f"Maximum iterations ({max_total_iterations}) reached. Providing summary.\n"
            )
        )

        try:
            response = await runtime.provider.chat(
                messages=completion_messages,
                model=runtime.model,
                temperature=runtime.temperature,
                max_tokens=min(runtime.max_tokens, 1024),
                tools=None,
            )
            if response and response.content:
                sanitized = runtime.sanitizer.sanitize(response.content)
                if sanitized:
                    from victor.agent.conversation.types import (
                        MESSAGE_SOURCE_METADATA_KEY,
                        MessageSource,
                    )

                    runtime.add_message(
                        "assistant",
                        sanitized,
                        metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_RESPONSE.value},
                    )
                    chunk = StreamChunk(content=sanitized, is_final=True)
                    runtime._record_runtime_intelligence_outcome(
                        success=True,
                        quality_score=last_quality_score,
                        user_satisfied=True,
                        completed=True,
                    )
                    return True, chunk
        except (ProviderRateLimitError, ProviderTimeoutError) as exc:
            logger.error("Rate limit/timeout during final response: %s", exc)
            chunk = StreamChunk(
                content="Rate limited or timeout. Please retry in a moment.\n",
                is_final=True,
            )
        except ProviderAuthError as exc:
            logger.error("Auth error during final response: %s", exc)
            chunk = StreamChunk(
                content="Authentication error. Check API credentials.\n",
                is_final=True,
            )
        except (ConnectionError, TimeoutError) as exc:
            logger.error("Network error during final response: %s", exc)
            chunk = StreamChunk(content="Network error. Check connection.\n", is_final=True)
        except Exception:
            logger.exception("Unexpected error during final response generation")
            chunk = StreamChunk(
                content="Unable to generate final summary due to iteration limit.\n",
                is_final=True,
            )

        runtime._record_runtime_intelligence_outcome(
            success=True,
            quality_score=last_quality_score,
            user_satisfied=True,
            completed=True,
        )
        return True, chunk if chunk else StreamChunk(content="", is_final=True)
