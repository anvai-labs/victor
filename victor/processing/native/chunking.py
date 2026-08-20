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

"""Document chunking, type detection, and token counting with native acceleration."""

import re
from typing import List

from victor.processing.native._base import _NATIVE_AVAILABLE, _native


def chunk_by_sentences(text: str, chunk_size: int = 1344, overlap: int = 128) -> List[str]:
    """Chunk text by sentence boundaries with overlap.

    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    if _NATIVE_AVAILABLE:
        return _native.chunk_by_sentences(text, chunk_size, overlap)

    # Pure Python fallback
    # Simple sentence splitting
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:]
            else:
                current_chunk = ""
        current_chunk += " " + sentence if current_chunk else sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def chunk_by_chars(text: str, chunk_size: int = 1344, overlap: int = 128) -> List[str]:
    """Chunk text by character count with overlap.

    Args:
        text: Text to chunk
        chunk_size: Chunk size in characters
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    if _NATIVE_AVAILABLE:
        return _native.chunk_by_chars(text, chunk_size, overlap)

    # Pure Python fallback
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap if overlap < end else 0

    return chunks


def chunk_by_paragraphs(text: str, chunk_size: int = 1344, overlap: int = 128) -> List[str]:
    """Chunk text by paragraph boundaries with overlap.

    Args:
        text: Text to chunk
        chunk_size: Target chunk size
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    if _NATIVE_AVAILABLE:
        return _native.chunk_by_paragraphs(text, chunk_size, overlap)

    # Pure Python fallback
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:]
            else:
                current_chunk = ""
        current_chunk += "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def detect_doc_type(source: str) -> str:
    """Detect document type from file extension.

    Args:
        source: File path or name

    Returns:
        Document type string
    """
    if _NATIVE_AVAILABLE:
        return _native.detect_doc_type(source)

    # Pure Python fallback
    source_lower = source.lower()
    extensions = {
        ".py": "code",
        ".js": "code",
        ".ts": "code",
        ".java": "code",
        ".go": "code",
        ".rs": "code",
        ".c": "code",
        ".cpp": "code",
        ".md": "markdown",
        ".markdown": "markdown",
        ".rst": "markdown",
        ".html": "html",
        ".htm": "html",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".xml": "xml",
        ".csv": "csv",
        ".txt": "text",
    }
    for ext, doc_type in extensions.items():
        if source_lower.endswith(ext):
            return doc_type
    return "text"


def count_tokens_approx(text: str) -> int:
    """Count approximate tokens in text.

    Fast method suitable for quick estimates. Accuracy is ~80-90%.
    For exact counting, use count_tokens() if tiktoken is available.

    Args:
        text: Text to count

    Returns:
        Approximate token count
    """
    if _NATIVE_AVAILABLE:
        return _native.count_tokens_approx(text)

    # Pure Python fallback (words + punctuation)
    words = len(re.findall(r"\w+", text))
    punctuation = len(re.findall(r"[^\w\s]", text))
    return words + punctuation


# Cache for tiktoken tokenizer singleton
_tiktoken_cache = None


def _get_tiktoken_encoder():
    """Get tiktoken encoder (cached singleton).

    Returns None if tiktoken is not installed.

    Returns:
        tiktoken Encoding object or None
    """
    global _tiktoken_cache
    if _tiktoken_cache is not None:
        return _tiktoken_cache

    try:
        import tiktoken

        # Use cl100k_base (GPT-4 tokenizer) as default - works well for most models
        _tiktoken_cache = tiktoken.get_encoding("cl100k_base")
        return _tiktoken_cache
    except ImportError:
        return None


def count_tokens(text: str) -> int:
    """Count exact tokens in text using tiktoken.

    Falls back to count_tokens_approx() if tiktoken is not available.

    Args:
        text: Text to count

    Returns:
        Exact token count (using tiktoken) or approximate if unavailable
    """
    encoder = _get_tiktoken_encoder()
    if encoder is not None:
        # tiktoken exact counting
        tokens = encoder.encode(text)
        return len(tokens)
    else:
        # Fallback to approximation
        return count_tokens_approx(text)


def count_tokens_batch(texts: List[str]) -> List[int]:
    """Count tokens for multiple texts efficiently.

    Uses tiktoken batch encoding if available, otherwise falls back
    to count_tokens_approx() for each text.

    Args:
        texts: List of texts to count

    Returns:
        List of token counts (same order as input)
    """
    encoder = _get_tiktoken_encoder()
    if encoder is not None:
        # tiktoken batch encoding - more efficient than individual calls
        try:
            # Encode all texts at once
            all_tokens = encoder.encode_batch(texts)
            return [len(tokens) for tokens in all_tokens]
        except Exception:
            # Fallback to individual encoding if batch fails
            return [count_tokens(text) for text in texts]
    else:
        # Fallback to approximation
        return [count_tokens_approx(text) for text in texts]
