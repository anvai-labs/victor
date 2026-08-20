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

"""Base class for query enhancement strategies.

Provides common functionality for LLM-based enhancement strategies:
- Lazy provider initialization
- Prompt template management with domain customization
- Graceful fallback handling
- LLM calling with error handling
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

from victor.protocols.query_enhancement import (
    EnhancedQuery,
    EnhancementContext,
    EnhancementTechnique,
)

if TYPE_CHECKING:
    from victor.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class BaseQueryEnhancementStrategy(ABC):
    """Abstract base class for query enhancement strategies.

    Provides:
    - Lazy LLM provider initialization
    - Domain-specific prompt template registration
    - Fallback handling when LLM unavailable
    - Caching integration hooks

    Subclasses must implement:
    - name: Strategy name
    - technique: Enhancement technique enum
    - _register_default_templates(): Register domain templates
    - _enhance_impl(): Core enhancement logic
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        timeout: float = 10.0,
    ):
        """Initialize strategy.

        Args:
            provider: LLM provider name (e.g., "ollama", "anthropic")
            model: Model name for enhancement
            temperature: LLM temperature (lower = more focused)
            timeout: Request timeout in seconds
        """
        self._provider_name = provider
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._provider_instance: Optional["BaseProvider"] = None

        # Domain-specific prompt templates
        self._prompt_templates: Dict[str, str] = {}
        self._register_default_templates()

    @property
    @abstractmethod
    def name(self) -> str:
        """Return strategy name."""
        pass

    @property
    @abstractmethod
    def technique(self) -> EnhancementTechnique:
        """Return the enhancement technique."""
        pass

    @property
    def requires_llm(self) -> bool:
        """Return whether this strategy requires LLM access."""
        return True

    @abstractmethod
    def _register_default_templates(self) -> None:
        """Register default prompt templates for each domain.

        Called during __init__. Subclasses should populate
        self._prompt_templates with domain-specific templates.

        Example:
            self._prompt_templates["financial"] = "Rewrite for SEC filings: {query}"
            self._prompt_templates["code"] = "Rewrite for code search: {query}"
            self._prompt_templates["general"] = "Rewrite query: {query}"
        """
        pass

    @abstractmethod
    async def _enhance_impl(
        self,
        query: str,
        context: EnhancementContext,
        llm_response: Optional[str],
    ) -> EnhancedQuery:
        """Implementation-specific enhancement logic.

        Args:
            query: Original query
            context: Enhancement context
            llm_response: LLM response (None if LLM unavailable)

        Returns:
            Enhanced query result
        """
        pass

    def register_template(self, domain: str, template: str) -> None:
        """Register a domain-specific prompt template.

        Args:
            domain: Domain identifier
            template: Prompt template with {query} and {context} placeholders
        """
        self._prompt_templates[domain] = template

    def get_prompt_template(self, domain: str) -> str:
        """Get domain-specific prompt template.

        Falls back to "general" if domain not registered.

        Args:
            domain: Domain identifier

        Returns:
            Prompt template string
        """
        return self._prompt_templates.get(domain, self._prompt_templates.get("general", ""))

    async def _get_provider(self) -> Optional["BaseProvider"]:
        """Get or create LLM provider instance (lazy initialization)."""
        if self._provider_instance is not None:
            return self._provider_instance

        try:
            from victor.config.settings import load_settings
            from victor.providers.registry import ProviderRegistry

            settings = load_settings()

            provider_name = self._provider_name or settings.provider.default_provider or "ollama"
            model = self._model or settings.provider.default_model

            self._provider_instance = ProviderRegistry.create(provider_name)
            if not self._model:
                self._model = model
            return self._provider_instance

        except Exception as e:
            logger.warning(f"Failed to create provider: {e}")
            return None

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM with prompt and return response.

        Args:
            prompt: The prompt to send to LLM

        Returns:
            LLM response text, or None if call failed
        """
        provider = await self._get_provider()
        if not provider:
            return None

        if not self._model:
            logger.warning("No model configured for query enhancement")
            return None

        try:
            from victor.providers.base import Message

            messages = [Message(role="user", content=prompt)]
            response = await provider.chat(
                messages=messages,
                model=self._model,
                temperature=self._temperature,
            )

            # Extract content from response (handles different response types)
            if hasattr(response, "content"):
                return response.content.strip()
            elif hasattr(response, "message"):
                return response.message.get("content", "").strip()
            return str(response).strip()

        except Exception as e:
            logger.warning(f"LLM call failed for {self.name}: {e}")
            return None

    async def enhance(
        self,
        query: str,
        context: EnhancementContext,
    ) -> EnhancedQuery:
        """Enhance a query using this strategy.

        Args:
            query: Original query
            context: Enhancement context

        Returns:
            Enhanced query result
        """
        llm_response = None

        if self.requires_llm:
            # Build prompt from template
            template = self.get_prompt_template(context.domain)
            if template:
                formatted_context = self._format_context(context)
                prompt = template.format(query=query, context=formatted_context)
                llm_response = await self._call_llm(prompt)

        return await self._enhance_impl(query, context, llm_response)

    def _format_context(self, context: EnhancementContext) -> str:
        """Format enhancement context for prompt insertion.

        Args:
            context: Enhancement context

        Returns:
            Formatted context string for prompt
        """
        parts = []

        if context.entity_metadata:
            entity_lines = []
            for entity in context.entity_metadata[:5]:  # Max 5 entities
                name = entity.get("name", "")
                ticker = entity.get("ticker", "")
                if name:
                    line = f"  - {name}"
                    if ticker:
                        line += f" ({ticker})"
                    entity_lines.append(line)
            if entity_lines:
                parts.append("Entities:\n" + "\n".join(entity_lines))

        if context.task_type:
            parts.append(f"Task type: {context.task_type}")

        if context.conversation_history:
            history = context.conversation_history[-3:]  # Last 3 messages
            parts.append("Recent context:\n" + "\n".join(f"  - {h[:100]}" for h in history))

        return "\n".join(parts) if parts else "No additional context."

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(technique={self.technique.value})"
