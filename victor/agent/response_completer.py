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

"""Response completion handling for agentic tool loops.

This module provides utilities for ensuring complete responses after
tool execution, handling errors gracefully, and generating appropriate
user-facing messages.

Design Principles:
- Ensures no empty responses after tool failures
- Provides retry logic for generating responses
- Separates response completion logic from orchestration
- Reuses existing provider infrastructure
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from victor.core.errors import ProviderRateLimitError
from victor.providers.base import BaseProvider, Message

logger = logging.getLogger(__name__)


class CompletionStatus(Enum):
    """Status of response completion attempt."""

    SUCCESS = "success"
    PARTIAL = "partial"  # Some content but incomplete
    EMPTY = "empty"  # No content generated
    ERROR = "error"  # Error during generation
    TIMEOUT = "timeout"  # Timed out waiting for response


@dataclass
class CompletionConfig:
    """Configuration for response completion."""

    max_retries: int = 3
    retry_temperature_increment: float = 0.1
    min_response_length: int = 10
    max_recovery_attempts: int = 3
    force_response_on_error: bool = True
    error_response_template: str = (
        "I encountered an issue: {error}\n\n" "Based on what I found before the error:\n{context}"
    )


@dataclass
class ToolFailureContext:
    """Context about tool failures for error response generation."""

    failed_tools: List[Dict[str, Any]] = field(default_factory=list)
    successful_tools: List[Dict[str, Any]] = field(default_factory=list)
    last_error: Optional[str] = None
    files_examined: List[str] = field(default_factory=list)
    partial_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionResult:
    """Result of a response completion attempt."""

    status: CompletionStatus
    content: str = ""
    retries_used: int = 0
    error: Optional[str] = None
    recovery_context: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        """Check if response is complete."""
        return self.status == CompletionStatus.SUCCESS and len(self.content) > 10


class ResponseCompleter:
    """Ensures complete responses after tool execution.

    This class handles the complexity of generating meaningful responses
    after tool calls, including error recovery and retry logic.

    Example:
        completer = ResponseCompleter(provider, config)
        result = await completer.ensure_response(
            messages, tool_results, failure_context
        )
    """

    def __init__(
        self,
        provider: BaseProvider,
        config: Optional[CompletionConfig] = None,
    ):
        """Initialize response completer.

        Args:
            provider: LLM provider for generating responses
            config: Completion configuration
        """
        self.provider = provider
        self.config = config or CompletionConfig()

    async def ensure_response(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        failure_context: Optional[ToolFailureContext] = None,
        current_content: str = "",
    ) -> CompletionResult:
        """Ensure a complete response is generated.

        Args:
            messages: Current conversation messages
            model: Model identifier
            temperature: Base temperature
            max_tokens: Maximum tokens to generate
            tool_results: Results from tool execution
            failure_context: Context about any tool failures
            current_content: Content already generated

        Returns:
            CompletionResult with status and content
        """
        # If we already have sufficient content, return it
        if current_content and len(current_content.strip()) >= self.config.min_response_length:
            return CompletionResult(
                status=CompletionStatus.SUCCESS,
                content=current_content,
            )

        # Check if we have tool failures that need addressing
        if failure_context and failure_context.failed_tools:
            return await self._handle_tool_failures(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                failure_context=failure_context,
            )

        # Try to generate a complete response
        return await self._generate_response_with_retry(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def _handle_tool_failures(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        failure_context: ToolFailureContext,
    ) -> CompletionResult:
        """Generate a helpful response when tools have failed.

        Args:
            messages: Conversation messages
            model: Model identifier
            temperature: Base temperature
            max_tokens: Maximum tokens
            failure_context: Context about failures

        Returns:
            CompletionResult with error response
        """
        # Build context from what succeeded
        context_parts = []

        if failure_context.successful_tools:
            context_parts.append(
                f"Successfully executed: {', '.join(t.get('name', 'unknown') for t in failure_context.successful_tools)}"
            )

        if failure_context.files_examined:
            context_parts.append(f"Files examined: {', '.join(failure_context.files_examined[:5])}")

        if failure_context.partial_results:
            context_parts.append("Partial results were obtained before the failure.")

        context = "\n".join(context_parts) if context_parts else "No prior results available."

        # Build error description
        error_descriptions = []
        for failed in failure_context.failed_tools:
            tool_name = failed.get("name", "unknown")
            error = failed.get("error", "Unknown error")
            error_descriptions.append(f"- {tool_name}: {error}")

        error_summary = "\n".join(error_descriptions)

        # Add a system message prompting for error response
        error_prompt = (
            f"The following tool(s) failed:\n{error_summary}\n\n"
            f"Prior context:\n{context}\n\n"
            "Please provide a helpful response to the user explaining what happened "
            "and suggesting alternatives if applicable."
        )

        # Create modified messages with error prompt
        modified_messages = list(messages) + [Message(role="system", content=error_prompt)]

        # Generate response
        try:
            response = await self.provider.chat(
                messages=modified_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=None,  # No tools - force text response
            )

            if response and response.content and len(response.content.strip()) > 10:
                return CompletionResult(
                    status=CompletionStatus.SUCCESS,
                    content=response.content,
                    recovery_context=context,
                )

        except Exception as e:
            logger.warning(f"Failed to generate error response: {e}")

        # Fallback to template response
        fallback_content = self.config.error_response_template.format(
            error=error_summary,
            context=context,
        )

        return CompletionResult(
            status=CompletionStatus.PARTIAL,
            content=fallback_content,
            error=failure_context.last_error,
            recovery_context=context,
        )

    async def _generate_response_with_retry(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> CompletionResult:
        """Generate response with retry logic for empty responses.

        Uses progressively higher temperature and simplified prompts
        to encourage response generation.

        Args:
            messages: Conversation messages
            model: Model identifier
            temperature: Base temperature
            max_tokens: Maximum tokens

        Returns:
            CompletionResult with status and content
        """
        recovery_prompts = [
            "Please provide your response now based on the information gathered.",
            "Summarize your findings in a clear, helpful response.",
            "What is your answer to the user's question?",
        ]

        # Cap rate-limit waits so we don't sleep indefinitely on a misbehaving
        # provider. 60s covers the common 15-30s 429 cooldowns with headroom.
        _MAX_RATE_LIMIT_WAIT_SECONDS = 60

        for attempt in range(self.config.max_recovery_attempts):
            current_temp = min(
                temperature + (attempt * self.config.retry_temperature_increment),
                1.0,
            )

            # Add recovery prompt if not first attempt
            modified_messages = list(messages)
            if attempt > 0:
                modified_messages.append(
                    Message(
                        role="system",
                        content=recovery_prompts[min(attempt - 1, len(recovery_prompts) - 1)],
                    )
                )

            try:
                response = await self.provider.chat(
                    messages=modified_messages,
                    model=model,
                    temperature=current_temp,
                    max_tokens=max_tokens,
                    tools=None,  # Force text response
                )

                if response and response.content:
                    content = response.content.strip()
                    if len(content) >= self.config.min_response_length:
                        return CompletionResult(
                            status=CompletionStatus.SUCCESS,
                            content=content,
                            retries_used=attempt,
                        )

                logger.debug(f"Recovery attempt {attempt + 1}: insufficient response")

            except ProviderRateLimitError as e:
                # The provider (or its circuit breaker) is in cooldown. Retrying
                # in a tight loop just re-hits the suppression check and burns
                # all our recovery attempts in milliseconds. Sleep for the
                # provider-reported cooldown before trying again — and bail out
                # entirely if it's longer than we're willing to wait or we're
                # already on the last attempt.
                wait_seconds = getattr(e, "retry_after", None) or 15
                remaining_attempts = self.config.max_recovery_attempts - attempt - 1
                if wait_seconds > _MAX_RATE_LIMIT_WAIT_SECONDS or remaining_attempts <= 0:
                    logger.warning(
                        "Recovery attempt %d hit rate limit (retry_after=%ss); " "giving up: %s",
                        attempt + 1,
                        wait_seconds,
                        e,
                    )
                    return CompletionResult(
                        status=CompletionStatus.EMPTY,
                        content="",
                        retries_used=attempt + 1,
                        error=f"Provider rate-limited; gave up after waiting bound exceeded ({wait_seconds}s)",
                    )
                logger.warning(
                    "Recovery attempt %d hit rate limit; sleeping %ss before retry",
                    attempt + 1,
                    wait_seconds,
                )
                await asyncio.sleep(wait_seconds)

            except Exception as e:
                logger.warning(f"Recovery attempt {attempt + 1} failed: {e}")

        # All attempts failed
        return CompletionResult(
            status=CompletionStatus.EMPTY,
            content="",
            retries_used=self.config.max_recovery_attempts,
            error="Failed to generate response after multiple attempts",
        )

    def format_tool_failure_message(
        self,
        failure_context: ToolFailureContext,
    ) -> str:
        """Format a user-friendly message about tool failures.

        Args:
            failure_context: Context about failures

        Returns:
            Formatted error message
        """
        parts = []

        if failure_context.failed_tools:
            parts.append("I encountered some issues while processing your request:\n")
            for failed in failure_context.failed_tools:
                tool_name = failed.get("name", "unknown")
                error = failed.get("error", "Unknown error")
                parts.append(f"• {tool_name}: {error}")

        if failure_context.successful_tools:
            parts.append(
                f"\nHowever, I was able to successfully use: "
                f"{', '.join(t.get('name', 'unknown') for t in failure_context.successful_tools)}"
            )

        if failure_context.files_examined:
            parts.append(f"\nFiles examined: {', '.join(failure_context.files_examined[:5])}")

        return "\n".join(parts)


def create_response_completer(
    provider: BaseProvider,
    max_retries: int = 3,
    force_response: bool = True,
) -> ResponseCompleter:
    """Factory function to create a response completer.

    Args:
        provider: LLM provider
        max_retries: Maximum retry attempts
        force_response: Whether to force response on error

    Returns:
        Configured ResponseCompleter
    """
    config = CompletionConfig(
        max_retries=max_retries,
        force_response_on_error=force_response,
    )
    return ResponseCompleter(provider=provider, config=config)


class AgenticLoop:
    """Manages the agentic tool-use loop for complete responses.

    This class encapsulates the logic for running multiple iterations
    of model -> tool calls -> response until completion.

    Example:
        loop = AgenticLoop(orchestrator, config)
        response = await loop.run(user_message)
    """

    @dataclass
    class Config:
        """Configuration for agentic loop."""

        max_iterations: int = 10
        force_response_after_failures: int = 2
        continue_on_tool_failure: bool = True
        require_final_response: bool = True

    def __init__(
        self,
        provider: BaseProvider,
        completer: ResponseCompleter,
        config: Optional["AgenticLoop.Config"] = None,
    ):
        """Initialize agentic loop.

        Args:
            provider: LLM provider
            completer: Response completer for ensuring responses
            config: Loop configuration
        """
        self.provider = provider
        self.completer = completer
        self.config = config or self.Config()

    async def should_continue(
        self,
        iteration: int,
        tool_calls: Optional[List[Dict[str, Any]]],
        failure_context: ToolFailureContext,
        content: str,
    ) -> Tuple[bool, str]:
        """Determine if the loop should continue.

        Args:
            iteration: Current iteration number
            tool_calls: Tool calls from last response
            failure_context: Context about failures
            content: Content generated so far

        Returns:
            Tuple of (should_continue, reason)
        """
        # Stop if max iterations reached
        if iteration >= self.config.max_iterations:
            return False, "max_iterations_reached"

        # Stop if no tool calls and we have content
        if not tool_calls and content:
            return False, "complete_response"

        # Stop if too many failures
        if len(failure_context.failed_tools) >= self.config.force_response_after_failures:
            return False, "too_many_failures"

        # Continue if there are tool calls to process
        if tool_calls:
            return True, "has_tool_calls"

        return False, "no_tool_calls"
