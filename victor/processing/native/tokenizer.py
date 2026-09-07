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

"""Token counting functions with native acceleration.

Provides exact and approximate token counting using Rust BPE tokenizer
when available, falling back to tiktoken or word-based estimation.
"""

from typing import List

from victor.processing.native._base import _NATIVE_AVAILABLE, _native

# Try tiktoken for Python fallback
try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

# Cached tiktoken encoder for fallback
_tiktoken_encoder = None

# Cached native BPE tokenizer (co-design review item 17a). Built lazily from
# tiktoken's cl100k_base ranks so the exact-count path runs real BPE in Rust
# instead of the heuristic ``count_tokens_fast`` it was misrouted to.
_bpe_tokenizer = None
_bpe_tokenizer_failed = False


def _get_tiktoken_encoder():
    """Get or create cached tiktoken encoder."""
    global _tiktoken_encoder
    if _tiktoken_encoder is None and _TIKTOKEN_AVAILABLE:
        _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_encoder


def _get_bpe_tokenizer():
    """Get or create the cached native BpeTokenizer from tiktoken's ranks.

    Returns None when the native module lacks BpeTokenizer (stale wheel) or
    tiktoken is unavailable — callers fall through to the existing chain.
    Construction failures are remembered so we don't retry per call.

    Known divergence from tiktoken (documented in the Rust source): the
    pre-tokenizer's ``\\s+(?!\\S)`` negative lookahead is approximated as
    ``\\s+$`` (the Rust regex crate has no lookahead), so runs of 2+
    whitespace *before a word* can split differently — e.g. ``"a\\t\\tb"``
    counts one fewer token than tiktoken. Token *counts* on ordinary prose
    are unaffected; only mid-string multi-whitespace runs diverge.
    """
    global _bpe_tokenizer, _bpe_tokenizer_failed
    if _bpe_tokenizer is not None or _bpe_tokenizer_failed:
        return _bpe_tokenizer
    if not (_NATIVE_AVAILABLE and hasattr(_native, "BpeTokenizer")):
        _bpe_tokenizer_failed = True
        return None
    if not _TIKTOKEN_AVAILABLE:
        _bpe_tokenizer_failed = True
        return None
    try:
        encoder = _get_tiktoken_encoder()
        if encoder is None:
            _bpe_tokenizer_failed = True
            return None
        _bpe_tokenizer = _native.BpeTokenizer(
            "cl100k_base",
            list(encoder._mergeable_ranks.items()),
            list(encoder._special_tokens.items()),
        )
    except Exception:
        _bpe_tokenizer_failed = True
        _bpe_tokenizer = None
    return _bpe_tokenizer


def count_tokens(text: str) -> int:
    """Count tokens in text using exact BPE tokenization.

    Uses the native BpeTokenizer (real BPE over tiktoken's cl100k_base
    ranks) when both the native extension and tiktoken are available, so the
    count matches tiktoken exactly outside the documented whitespace-run
    divergence (see ``_get_bpe_tokenizer``). Falls back to tiktoken, then to
    word-based estimation.

    Args:
        text: Text to count tokens for

    Returns:
        Number of tokens
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "BpeTokenizer"):
        tokenizer = _get_bpe_tokenizer()
        if tokenizer is not None:
            return tokenizer.count_tokens(text)

    # Pure Python fallback using tiktoken
    encoder = _get_tiktoken_encoder()
    if encoder is not None:
        return len(encoder.encode(text))

    # Last resort: word-based estimation (~1.3 tokens per word)
    return len(text.split()) * 13 // 10


def count_tokens_fast(text: str) -> int:
    """Count tokens using fast approximate method.

    Optimized for speed over accuracy. Uses Rust native counting
    when available, otherwise falls back to word-based estimation.

    Args:
        text: Text to count tokens for

    Returns:
        Approximate number of tokens
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "count_tokens_fast"):
        return _native.count_tokens_fast(text)

    # Pure Python fallback: word-based estimation (~1.3 tokens per word)
    return len(text.split()) * 13 // 10


def count_tokens_batch(texts: List[str]) -> List[int]:
    """Count tokens for multiple texts in batch.

    More efficient than calling count_tokens() in a loop when
    Rust extensions are available (amortizes FFI overhead). Uses the same
    exact BpeTokenizer as count_tokens() when available (the native batch
    method is rayon-parallel), so batch and single counts agree.

    Args:
        texts: List of texts to count tokens for

    Returns:
        List of token counts, one per input text
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "BpeTokenizer"):
        tokenizer = _get_bpe_tokenizer()
        if tokenizer is not None and hasattr(tokenizer, "count_tokens_batch"):
            return tokenizer.count_tokens_batch(texts)
    # Prefer a single native crossing for the whole batch — this is what
    # actually amortises the FFI overhead (a per-item loop does not).
    if _NATIVE_AVAILABLE and hasattr(_native, "count_tokens_fast_batch"):
        return _native.count_tokens_fast_batch(texts)
    if _NATIVE_AVAILABLE and hasattr(_native, "count_tokens_fast"):
        return [_native.count_tokens_fast(text) for text in texts]

    # Pure Python fallback (tiktoken's encode_batch avoids a Python-level loop)
    encoder = _get_tiktoken_encoder()
    if encoder is not None:
        return [len(ids) for ids in encoder.encode_batch(texts)]

    # Last resort: word-based estimation
    return [len(text.split()) * 13 // 10 for text in texts]
