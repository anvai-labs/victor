# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# Licensed under the Apache License, Version 2.0 (the "License").

"""Response delivery through the runtime's existing chunk and sanitizer components."""

from dataclasses import dataclass
from typing import Protocol

from victor.providers.base import StreamChunk


class ContentChunkGenerator(Protocol):
    """Content emission, including the configured generator's metadata and effects."""

    def generate_content_chunk(self, content: str, is_final: bool = False) -> StreamChunk: ...


class ChatSanitizer(Protocol):
    """Text cleanup and garbage detection needed by streaming delivery."""

    def sanitize(self, text: str) -> str: ...

    def strip_markup(self, text: str) -> str: ...

    def is_garbage_content(self, text: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class ChatDelivery:
    """Bind existing components without retaining a facade or copying turn state.

    Only terminal recovery permits missing components. Ordinary content emission
    and cleanup require their configured component; component errors propagate.
    """

    chunks: ContentChunkGenerator | None = None
    sanitizer: ChatSanitizer | None = None

    def content_chunk(self, content: str, *, is_final: bool = False) -> StreamChunk:
        """Preserve the configured generator's output and side effects."""
        if self.chunks is None:
            raise TypeError("Chat delivery requires a content chunk generator")
        return self.chunks.generate_content_chunk(content, is_final=is_final)

    def final_marker_chunk(self) -> StreamChunk:
        """Use the configured marker factory, or the legacy empty final marker."""
        factory = getattr(self.chunks, "generate_final_marker_chunk", None)
        if callable(factory):
            return factory()
        return StreamChunk(content="", is_final=True)

    def sanitize(self, text: str, *, optional: bool = False) -> str:
        """Clean text, allowing the existing terminal recovery identity fallback."""
        if optional and not self.sanitizer:
            return text
        if self.sanitizer is None:
            raise TypeError("Chat delivery requires a sanitizer")
        return self.sanitizer.sanitize(text)

    def strip_markup(self, text: str, *, optional: bool = False) -> str:
        """Recover plain text with the same terminal-only identity fallback."""
        if optional and not self.sanitizer:
            return text
        if self.sanitizer is None:
            raise TypeError("Chat delivery requires a sanitizer")
        return self.sanitizer.strip_markup(text)

    def is_garbage_content(self, text: str) -> bool:
        """Delegate garbage classification to the configured sanitizer."""
        if self.sanitizer is None:
            raise TypeError("Chat delivery requires a sanitizer")
        return self.sanitizer.is_garbage_content(text)
