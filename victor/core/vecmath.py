# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Pure-Python vector math for numpy-optional fallbacks.

Stdlib-only by design: this module sits on paths that must work in a bare
``pip install victor-ai`` (no numpy). The numbers it produces match the numpy
formulations they replace to within float64-vs-float32 precision; callers that
need hardware throughput should route to ``victor_native`` (primary engine) or
numpy (optional accelerator), never here — these are correctness fallbacks for
small vectors (embedding dims ≤ ~1.5k, corpora ≤ ~2k).

Semantics are inherited verbatim from the numpy fallbacks this module replaced:
- norms carry a 1e-9 epsilon so zero vectors yield 0.0 similarity, not NaN;
- dimension mismatches raise ``ValueError``;
- empty corpora yield ``[]``.
"""

from __future__ import annotations

import math
from typing import List, Sequence

_NORM_EPS = 1e-9


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors (eps-damped norms, like numpy path).

    Raises:
        ValueError: If the vectors have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(f"Vectors must have same length: {len(a)} vs {len(b)}")
    if len(a) == 0:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    return dot / ((math.sqrt(norm_a) + _NORM_EPS) * (math.sqrt(norm_b) + _NORM_EPS))


def batch_cosine(query: Sequence[float], corpus: Sequence[Sequence[float]]) -> List[float]:
    """Cosine similarity of one query against every corpus row.

    Raises:
        ValueError: If a corpus row's dimension differs from the query's.
    """
    if not corpus:
        return []
    query_norm = math.sqrt(sum(x * x for x in query)) + _NORM_EPS
    results: List[float] = []
    for row in corpus:
        if len(row) != len(query):
            raise ValueError(
                f"Dimension mismatch: query has {len(query)} dims, " f"corpus has {len(row)} dims"
            )
        dot = 0.0
        row_norm = 0.0
        for x, y in zip(query, row):
            dot += x * y
            row_norm += y * y
        results.append(dot / (query_norm * (math.sqrt(row_norm) + _NORM_EPS)))
    return results


def normalize_rows(vectors: Sequence[Sequence[float]]) -> List[List[float]]:
    """Scale each row to unit length (eps-damped), like the numpy formulation."""
    if not vectors:
        return []
    normalized: List[List[float]] = []
    for row in vectors:
        norm = math.sqrt(sum(x * x for x in row)) + _NORM_EPS
        normalized.append([x / norm for x in row])
    return normalized


def softmax(scores: Sequence[float]) -> List[float]:
    """Numerically stable softmax (max-shifted, ``math.exp``)."""
    if not scores:
        return []
    peak = max(scores)
    exps = [math.exp(s - peak) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def argmax(scores: Sequence[float]) -> int:
    """Index of the largest value (first index on ties, like ``np.argmax``)."""
    best = 0
    for i in range(1, len(scores)):
        if scores[i] > scores[best]:
            best = i
    return best
