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

"""Unit tests for victor/core/vecmath.py (stdlib-only vector math).

Mirrored location for scripts/ci/select_changed_tests.py: changed source
``victor/core/vecmath.py`` maps here.
"""

from __future__ import annotations

import math

import pytest

from victor.core.vecmath import argmax, batch_cosine, cosine, normalize_rows, softmax


def test_softmax_and_argmax_match_numpy():
    np = pytest.importorskip("numpy")
    scores = [0.3, -1.2, 2.4, 0.0]
    pure = softmax(scores)
    expected = np.exp(np.array(scores) - max(scores))
    expected = (expected / expected.sum()).tolist()
    assert pure == pytest.approx(expected, rel=1e-12)
    assert argmax(scores) == int(np.argmax(scores))


def test_softmax_empty():
    assert softmax([]) == []


def test_argmax_first_index_on_ties():
    assert argmax([1.0, 2.0, 2.0]) == 1
    assert argmax([2.0, 2.0]) == 0  # np.argmax convention: first max


def test_vecmath_cosine_matches_closed_form():
    a = [3.0, 4.0]
    b = [4.0, 3.0]
    expected = (3 * 4 + 4 * 3) / ((5 + 1e-9) * (5 + 1e-9))
    assert cosine(a, b) == pytest.approx(expected, rel=1e-12)
    assert math.isfinite(cosine([0.0, 0.0], [1.0, 1.0]))


def test_vecmath_batch_cosine_semantics():
    assert batch_cosine([1.0, 0.0], []) == []
    with pytest.raises(ValueError, match="Dimension mismatch"):
        batch_cosine([1.0, 0.0], [[1.0, 0.0, 1.0]])
    assert normalize_rows([]) == []


def test_vecmath_cosine_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        cosine([1.0], [1.0, 2.0])
