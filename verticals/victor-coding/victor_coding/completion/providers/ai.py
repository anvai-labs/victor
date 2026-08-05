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

"""AI-powered completion provider.

Uses LLMs to provide intelligent code completions including
multi-line suggestions (Copilot-style).
"""

import logging
import time
from typing import Any, AsyncIterator, Optional

from victor_coding.completion.protocol import (
    CompletionCapabilities,
    CompletionItem,
    CompletionItemKind,
    CompletionList,
    CompletionParams,
    InlineCompletionItem,
    InlineCompletionList,
    InlineCompletionParams,
)
from victor_coding.completion.provider import StreamingCompletionProvider

logger = logging.getLogger(__name__)


# FIM (Fill-In-the-Middle) prompt templates per language
FIM_TEMPLATES = {
    "default": {
        "prefix": "<PRE>",
        "suffix": "<SUF>",
        "middle": "<MID>",
        "format": "{prefix}{pre}{suffix}{suf}{middle}",
    },
    "codellama": {
        "prefix": "<PRE>",
        "suffix": " <SUF>",
        "middle": " <MID>",
        "format": "{prefix} {pre}{suffix}{suf}{middle}",
    },
    "starcoder": {
        "prefix": "<fim_prefix>",
        "suffix": "<fim_suffix>",
        "middle": "<fim_middle>",
        "format": "{prefix}{pre}{suffix}{suf}{middle}",
    },
    "deepseek": {
        "prefix": "<｜fim▁begin｜>",
        "suffix": "<｜fim▁hole｜>",
        "middle": "<｜fim▁end｜>",
        "format": "{prefix}{pre}{suffix}{suf}{middle}",
    },
    "qwen": {
        "prefix": "<|fim_prefix|>",
        "suffix": "<|fim_suffix|>",
        "middle": "<|fim_middle|>",
        "format": "{prefix}{pre}{suffix}{suf}{middle}",
    },
}


class AICompletionProvider(StreamingCompletionProvider):
    """AI-powered completion provider using LLMs.

    Supports:
    - Fill-in-the-middle (FIM) completions
    - Multi-line inline suggestions
    - Streaming completions
    - Multiple model backends (Ollama, OpenAI, Anthropic)
    """

    def __init__(
        self,
        priority: int = 60,
        provider: Optional[Any] = None,
        model: Optional[str] = None,
        max_context_lines: int = 100,
        fim_template: str = "default",
    ):
        """Initialize the AI completion provider.

        Args:
            priority: Provider priority (default 60 - medium-high)
            provider: LLM provider instance
            model: Model name to use
            max_context_lines: Maximum context lines to include
            fim_template: FIM template name or custom template dict
        """
        super().__init__(priority=priority)
        self._provider = provider
        self._model = model
        self._max_context_lines = max_context_lines

        # Set FIM template
        if isinstance(fim_template, dict):
            self._fim_template = fim_template
        else:
            self._fim_template = FIM_TEMPLATES.get(fim_template, FIM_TEMPLATES["default"])

    @property
    def name(self) -> str:
        return "ai"

    def _get_base_capabilities(self) -> CompletionCapabilities:
        """Get base capabilities."""
        return CompletionCapabilities(
            supports_completion=True,
            supports_inline_completion=True,
            supports_resolve=False,
            supports_snippets=False,
            supports_multi_line=True,
            max_context_lines=self._max_context_lines,
            supported_languages=[],  # All languages
        )

    def _get_provider(self) -> Optional[Any]:
        """Get or lazy-load the LLM provider."""
        if self._provider is not None:
            return self._provider

        try:
            from victor_contracts.provider_runtime import ProviderRegistry

            registry = ProviderRegistry()
            # Try to get Ollama for local completions
            self._provider = registry.get_provider("ollama")
            if self._provider is None:
                # Fall back to any available provider
                self._provider = registry.get_default_provider()
            return self._provider
        except ImportError:
            logger.debug("Provider registry not available")
            return None

    async def provide_completions(self, params: CompletionParams) -> CompletionList:
        """Provide AI-generated completions.

        For standard completions, generates short single-line suggestions.
        """
        provider = self._get_provider()
        if provider is None:
            return CompletionList(is_incomplete=False, items=[])

        try:
            start_time = time.time()

            # Build FIM prompt
            prompt = self._build_fim_prompt(
                prefix=params.prefix,
                suffix="",  # No suffix for standard completions
                max_lines=20,  # Shorter context for quick completions
            )

            # Generate completion
            response = await provider.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=64,  # Short completions
                temperature=0.0,
                stop=["\n", "```"],  # Single line
            )

            elapsed_ms = (time.time() - start_time) * 1000
            completion_text = self._extract_completion(response)

            if not completion_text:
                return CompletionList(is_incomplete=False, items=[])

            # Create completion item
            item = CompletionItem(
                label=(
                    completion_text[:50] + "..." if len(completion_text) > 50 else completion_text
                ),
                kind=CompletionItemKind.TEXT,
                insert_text=completion_text,
                detail="AI suggestion",
                provider=self.name,
                confidence=0.8,
                latency_ms=elapsed_ms,
            )

            return CompletionList(is_incomplete=False, items=[item])

        except Exception as e:
            logger.warning(f"AI completion failed: {e}")
            return CompletionList(is_incomplete=False, items=[])

    async def provide_inline_completions(
        self, params: InlineCompletionParams
    ) -> InlineCompletionList:
        """Provide AI-generated inline completions.

        Generates multi-line Copilot-style suggestions.
        """
        provider = self._get_provider()
        if provider is None:
            return InlineCompletionList(items=[])

        try:
            start_time = time.time()

            # Build FIM prompt with full context
            prompt = self._build_fim_prompt(
                prefix=params.prefix,
                suffix=params.suffix,
                max_lines=self._max_context_lines,
            )

            # Generate completion
            response = await provider.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                stop=params.stop_sequences or ["```", "<|endoftext|>"],
            )

            elapsed_ms = (time.time() - start_time) * 1000
            completion_text = self._extract_completion(response)

            if not completion_text:
                return InlineCompletionList(items=[])

            # Clean up the completion
            completion_text = self._clean_completion(completion_text, params.suffix)

            tokens_used = getattr(response, "usage", {}).get("completion_tokens", 0)

            item = InlineCompletionItem(
                insert_text=completion_text,
                provider=self.name,
                confidence=0.9,
                is_complete=True,
                tokens_used=tokens_used,
                latency_ms=elapsed_ms,
            )

            return InlineCompletionList(items=[item])

        except Exception as e:
            logger.warning(f"AI inline completion failed: {e}")
            return InlineCompletionList(items=[])

    async def stream_inline_completion(self, params: InlineCompletionParams) -> AsyncIterator[str]:
        """Stream inline completion tokens.

        Args:
            params: Inline completion parameters

        Yields:
            Completion tokens as they're generated
        """
        provider = self._get_provider()
        if provider is None:
            return

        try:
            # Build FIM prompt
            prompt = self._build_fim_prompt(
                prefix=params.prefix,
                suffix=params.suffix,
                max_lines=self._max_context_lines,
            )

            # Stream completion
            async for chunk in provider.stream_chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                stop=params.stop_sequences or ["```", "<|endoftext|>"],
            ):
                if chunk.content:
                    yield chunk.content

        except Exception as e:
            logger.warning(f"AI streaming completion failed: {e}")

    def _build_fim_prompt(
        self,
        prefix: str,
        suffix: str,
        max_lines: int,
    ) -> str:
        """Build Fill-In-the-Middle prompt.

        Args:
            prefix: Text before cursor
            suffix: Text after cursor
            max_lines: Maximum context lines

        Returns:
            FIM formatted prompt
        """
        # Truncate prefix to max lines (keep end)
        prefix_lines = prefix.split("\n")
        if len(prefix_lines) > max_lines:
            prefix = "\n".join(prefix_lines[-max_lines:])

        # Truncate suffix to max lines (keep beginning)
        suffix_lines = suffix.split("\n")
        if len(suffix_lines) > max_lines:
            suffix = "\n".join(suffix_lines[:max_lines])

        # Format with template
        template = self._fim_template
        return template["format"].format(
            prefix=template["prefix"],
            pre=prefix,
            suffix=template["suffix"],
            suf=suffix,
            middle=template["middle"],
        )

    def _extract_completion(self, response: Any) -> str:
        """Extract completion text from provider response.

        Args:
            response: Provider response object

        Returns:
            Extracted completion text
        """
        if hasattr(response, "content"):
            return response.content or ""
        if hasattr(response, "text"):
            return response.text or ""
        if isinstance(response, dict):
            return response.get("content", response.get("text", ""))
        return str(response) if response else ""

    def _clean_completion(self, completion: str, suffix: str) -> str:
        """Clean up completion text.

        Removes FIM tokens, trailing whitespace, and overlapping suffix.

        Args:
            completion: Raw completion text
            suffix: Original suffix text

        Returns:
            Cleaned completion
        """
        # Remove FIM tokens
        for template in FIM_TEMPLATES.values():
            for key in ["prefix", "suffix", "middle"]:
                token = template[key]
                completion = completion.replace(token, "")

        # Remove common end tokens
        end_tokens = [
            "<|endoftext|>",
            "</s>",
            "<|im_end|>",
            "```",
            "<|end|>",
        ]
        for token in end_tokens:
            if completion.endswith(token):
                completion = completion[: -len(token)]

        # Trim trailing whitespace
        completion = completion.rstrip()

        # Remove overlapping suffix
        if suffix:
            suffix_start = suffix.lstrip()[:50]  # First 50 chars of suffix
            if suffix_start and suffix_start in completion:
                idx = completion.find(suffix_start)
                completion = completion[:idx]

        return completion

    def set_model(self, model: str) -> None:
        """Set the model to use for completions.

        Args:
            model: Model name
        """
        self._model = model

        # Update FIM template based on model
        model_lower = model.lower()
        if "codellama" in model_lower:
            self._fim_template = FIM_TEMPLATES["codellama"]
        elif "starcoder" in model_lower:
            self._fim_template = FIM_TEMPLATES["starcoder"]
        elif "deepseek" in model_lower:
            self._fim_template = FIM_TEMPLATES["deepseek"]
        elif "qwen" in model_lower:
            self._fim_template = FIM_TEMPLATES["qwen"]

    def set_provider(self, provider: Any) -> None:
        """Set the LLM provider.

        Args:
            provider: Provider instance
        """
        self._provider = provider
