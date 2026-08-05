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

"""Completion manager for orchestrating completion operations.

Provides a high-level API for IDE integration following the
Facade pattern.
"""

import logging
import time
from pathlib import Path
from typing import AsyncIterator, Optional

from victor_coding.completion.protocol import (
    CompletionCapabilities,
    CompletionContext,
    CompletionItem,
    CompletionList,
    CompletionMetrics,
    CompletionParams,
    CompletionTriggerKind,
    InlineCompletionList,
    InlineCompletionParams,
    Position,
)
from victor_coding.completion.provider import (
    BaseCompletionProvider,
    StreamingCompletionProvider,
)
from victor_coding.completion.registry import (
    CompletionProviderRegistry,
    get_completion_registry,
)

logger = logging.getLogger(__name__)


class CompletionManager:
    """High-level manager for code completion operations.

    Orchestrates completion providers and provides a unified API
    for IDE integration. Handles:
    - Provider selection and fallback
    - Metrics collection
    - Caching and debouncing
    - Language detection
    """

    def __init__(
        self,
        registry: Optional[CompletionProviderRegistry] = None,
    ):
        """Initialize the completion manager.

        Args:
            registry: Provider registry (uses global if not provided)
        """
        self._registry = registry or get_completion_registry()
        self._metrics = CompletionMetrics()
        self._last_request_time: float = 0
        self._debounce_ms: float = 50  # Debounce interval

    @property
    def metrics(self) -> CompletionMetrics:
        """Get completion metrics."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self._metrics = CompletionMetrics()

    def get_capabilities(self) -> CompletionCapabilities:
        """Get aggregated capabilities from all providers.

        Returns:
            Combined capabilities
        """
        return self._registry.get_aggregated_capabilities()

    def get_providers(self) -> list[BaseCompletionProvider]:
        """Get all registered providers.

        Returns:
            List of providers sorted by priority
        """
        return self._registry.get_all_providers()

    async def complete(
        self,
        file_path: Path,
        line: int,
        character: int,
        trigger_kind: CompletionTriggerKind = CompletionTriggerKind.INVOKED,
        trigger_character: Optional[str] = None,
        file_content: Optional[str] = None,
        language: Optional[str] = None,
        max_results: int = 20,
    ) -> CompletionList:
        """Get completions at a position.

        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character position (0-indexed)
            trigger_kind: How completion was triggered
            trigger_character: Character that triggered completion
            file_content: Full file content (reads file if not provided)
            language: Language identifier (auto-detects if not provided)
            max_results: Maximum number of results

        Returns:
            CompletionList with suggestions
        """
        start_time = time.time()
        self._metrics.total_requests += 1

        try:
            # Read file content if not provided
            if file_content is None:
                if file_path.exists():
                    file_content = file_path.read_text()
                else:
                    file_content = ""

            # Auto-detect language
            if language is None:
                language = self._detect_language(file_path, file_content)

            # Extract prefix (text before cursor on current line)
            lines = file_content.split("\n")
            prefix = ""
            if 0 <= line < len(lines):
                prefix = lines[line][:character]

            # Build params
            params = CompletionParams(
                file_path=file_path,
                position=Position(line=line, character=character),
                context=CompletionContext(
                    trigger_kind=trigger_kind,
                    trigger_character=trigger_character,
                ),
                prefix=prefix,
                file_content=file_content,
                language=language,
                max_results=max_results,
            )

            # Get providers for this language
            providers = self._registry.get_providers_for_language(language)
            if not providers:
                logger.warning(f"No completion providers for language: {language}")
                return CompletionList(is_incomplete=False, items=[])

            # Collect completions from all providers
            all_items: list[CompletionItem] = []
            is_incomplete = False

            for provider in providers:
                try:
                    result = await provider.provide_completions(params)
                    all_items.extend(result.items)
                    is_incomplete |= result.is_incomplete
                except Exception as e:
                    logger.warning(f"Provider {provider.name} failed: {e}")
                    continue

            # Sort by confidence and label
            all_items.sort(key=lambda item: (-item.confidence, item.sort_text or item.label))

            # Limit results
            if len(all_items) > max_results:
                all_items = all_items[:max_results]
                is_incomplete = True

            self._metrics.successful_requests += 1
            return CompletionList(is_incomplete=is_incomplete, items=all_items)

        except Exception as e:
            logger.error(f"Completion failed: {e}")
            self._metrics.failed_requests += 1
            return CompletionList(is_incomplete=False, items=[])

        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            self._metrics.total_latency_ms += elapsed_ms

    async def complete_inline(
        self,
        file_path: Path,
        line: int,
        character: int,
        file_content: Optional[str] = None,
        language: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> InlineCompletionList:
        """Get inline (ghost text) completions.

        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character position (0-indexed)
            file_content: Full file content
            language: Language identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            InlineCompletionList with suggestions
        """
        start_time = time.time()
        self._metrics.total_requests += 1

        try:
            # Read file content if not provided
            if file_content is None:
                if file_path.exists():
                    file_content = file_path.read_text()
                else:
                    file_content = ""

            # Auto-detect language
            if language is None:
                language = self._detect_language(file_path, file_content)

            # Split content at cursor position
            lines = file_content.split("\n")
            prefix_lines = lines[:line]
            suffix_lines = lines[line + 1 :] if line + 1 < len(lines) else []

            current_line = lines[line] if line < len(lines) else ""
            prefix = "\n".join(prefix_lines)
            if prefix:
                prefix += "\n"
            prefix += current_line[:character]

            suffix = current_line[character:]
            if suffix_lines:
                suffix += "\n" + "\n".join(suffix_lines)

            # Build params
            params = InlineCompletionParams(
                file_path=file_path,
                position=Position(line=line, character=character),
                prefix=prefix,
                suffix=suffix,
                file_content=file_content,
                language=language,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Get providers with inline completion support
            providers = self._registry.get_providers_with_capability("supports_inline_completion")
            providers = [p for p in providers if p.supports_language(language)]

            if not providers:
                logger.debug(f"No inline completion providers for: {language}")
                return InlineCompletionList(items=[])

            # Try providers in priority order
            for provider in providers:
                try:
                    result = await provider.provide_inline_completions(params)
                    if result.items:
                        self._metrics.successful_requests += 1

                        # Track token usage
                        for item in result.items:
                            self._metrics.total_tokens_used += item.tokens_used

                        return result
                except Exception as e:
                    logger.warning(f"Inline provider {provider.name} failed: {e}")
                    continue

            self._metrics.successful_requests += 1
            return InlineCompletionList(items=[])

        except Exception as e:
            logger.error(f"Inline completion failed: {e}")
            self._metrics.failed_requests += 1
            return InlineCompletionList(items=[])

        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            self._metrics.total_latency_ms += elapsed_ms

    async def stream_inline_completion(
        self,
        file_path: Path,
        line: int,
        character: int,
        file_content: Optional[str] = None,
        language: Optional[str] = None,
        max_tokens: int = 256,
    ) -> AsyncIterator[str]:
        """Stream inline completion tokens.

        Args:
            file_path: Path to the file
            line: Line number
            character: Character position
            file_content: Full file content
            language: Language identifier
            max_tokens: Maximum tokens

        Yields:
            Completion text tokens
        """
        # Read file content if not provided
        if file_content is None:
            if file_path.exists():
                file_content = file_path.read_text()
            else:
                file_content = ""

        # Auto-detect language
        if language is None:
            language = self._detect_language(file_path, file_content)

        # Split content
        lines = file_content.split("\n")
        prefix_lines = lines[:line]
        suffix_lines = lines[line + 1 :] if line + 1 < len(lines) else []

        current_line = lines[line] if line < len(lines) else ""
        prefix = "\n".join(prefix_lines)
        if prefix:
            prefix += "\n"
        prefix += current_line[:character]

        suffix = current_line[character:]
        if suffix_lines:
            suffix += "\n" + "\n".join(suffix_lines)

        params = InlineCompletionParams(
            file_path=file_path,
            position=Position(line=line, character=character),
            prefix=prefix,
            suffix=suffix,
            file_content=file_content,
            language=language,
            max_tokens=max_tokens,
        )

        # Find streaming provider
        for provider in self._registry.get_all_providers():
            if not provider.enabled:
                continue
            if not provider.supports_language(language):
                continue
            if not isinstance(provider, StreamingCompletionProvider):
                continue

            caps = provider.get_capabilities()
            if not caps.supports_streaming:
                continue

            try:
                async for token in provider.stream_inline_completion(params):
                    yield token
                return
            except Exception as e:
                logger.warning(f"Streaming provider {provider.name} failed: {e}")
                continue

    async def resolve_completion(self, item: CompletionItem) -> CompletionItem:
        """Resolve additional details for a completion item.

        Args:
            item: The item to resolve

        Returns:
            Resolved item with additional details
        """
        if not item.provider:
            return item

        provider = self._registry.get_provider(item.provider)
        if provider is None:
            return item

        try:
            return await provider.resolve_completion(item)
        except Exception as e:
            logger.warning(f"Resolution failed for {item.provider}: {e}")
            return item

    def _detect_language(self, file_path: Path, content: str) -> str:
        """Detect language from file path and content.

        Args:
            file_path: Path to the file
            content: File content

        Returns:
            Language identifier
        """
        # Try by extension
        ext = file_path.suffix.lower()
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".r": "r",
            ".sql": "sql",
            ".sh": "bash",
            ".bash": "bash",
            ".zsh": "zsh",
            ".fish": "fish",
            ".md": "markdown",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".xml": "xml",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".less": "less",
        }

        if ext in ext_map:
            return ext_map[ext]

        # Try shebang
        if content.startswith("#!"):
            first_line = content.split("\n")[0]
            if "python" in first_line:
                return "python"
            if "node" in first_line:
                return "javascript"
            if "ruby" in first_line:
                return "ruby"
            if "bash" in first_line or "sh" in first_line:
                return "bash"

        return "text"

    def register_provider(self, provider: BaseCompletionProvider) -> None:
        """Register a completion provider.

        Args:
            provider: The provider to register
        """
        self._registry.register(provider)

    def unregister_provider(self, name: str) -> bool:
        """Unregister a provider by name.

        Args:
            name: Provider name

        Returns:
            True if found and removed
        """
        return self._registry.unregister(name)


# Global manager singleton
_completion_manager: Optional[CompletionManager] = None


def get_completion_manager() -> CompletionManager:
    """Get the global completion manager.

    Returns:
        The singleton manager instance
    """
    global _completion_manager
    if _completion_manager is None:
        _completion_manager = CompletionManager()
    return _completion_manager


def reset_completion_manager() -> None:
    """Reset the global completion manager.

    Useful for testing.
    """
    global _completion_manager
    _completion_manager = None
