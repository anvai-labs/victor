# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Versioned feature extraction for the edge classifier (FEP-0012).

Hashing-trick features (char + word n-grams) projected into a fixed-size space.
Deterministic **across processes** (uses hashlib, not Python's randomized
``hash()``) so a model trained offline consumes the exact same features at
online inference time, and the per-project RL delta updates the right hashes.

The whole config — hash space, n-gram sizes, normalization — is pinned by
``FEATURE_SPEC_VERSION``. A model trained on spec vN only consumes vN features;
if the spec changes, old deltas are invalidated (migrated/zeroed). This is the
robustness contract for offline training / online inference / online RL all
agreeing on the feature space.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict

# Bump when the feature config below changes. Old artifacts/deltas are
# invalidated by a version mismatch (see EdgeClassifierModel).
FEATURE_SPEC_VERSION = "1"

# Fixed feature space (hashing trick). 2^18 = 262144 dims keeps collisions rare
# while staying small enough to ship as sparse weights (~KB–low MB).
HASH_SPACE = 1 << 18

_CHAR_NGRAMS = (2, 3, 4)
_WORD_NGRAMS = (1, 2)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _hash_token(token: str) -> int:
    """Deterministic hash of a token into [0, HASH_SPACE).

    blake2b (digest_size=4) is deterministic across processes and platforms,
    unlike Python's ``hash()`` which is salted per process.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % HASH_SPACE


def extract_features(text: str) -> Dict[int, float]:
    """Extract hashing-trick features from ``text``.

    Produces a sparse ``{feature_hash: count}`` map combining lowercased
    character n-grams (2–4) and word n-grams (1–2). Empty/whitespace text
    yields an empty map (the caller falls back to the heuristic).

    Args:
        text: The decision input text (message excerpt, response tail, error
            message, …).

    Returns:
        Sparse feature map ``{hash: weight}``.
    """
    if not text:
        return {}

    lowered = text.lower()
    features: Dict[int, float] = {}

    # Word n-grams (joined so multi-word grams hash as one token).
    words = _WORD_RE.findall(lowered)
    for n in _WORD_NGRAMS:
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i : i + n])
            h = _hash_token("w" + gram)  # namespace prefix avoids char/word collisions
            features[h] = features.get(h, 0.0) + 1.0

    # Character n-grams over the whole lowered string (catches morphology /
    # punctuation patterns the word tokenizer would split).
    for n in _CHAR_NGRAMS:
        for i in range(len(lowered) - n + 1):
            gram = lowered[i : i + n]
            h = _hash_token("c" + gram)
            features[h] = features.get(h, 0.0) + 1.0

    return features
