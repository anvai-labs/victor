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

"""Tool-result cache: numpy-optional import safety and normalize semantics.

The cache imports numpy under a guard (its embedding features only run with
the embeddings extra); the module must import cleanly on a bare install, and
its normalize helper must behave identically with and without numpy.
"""

from __future__ import annotations

import math

import pytest

import victor.agent.tool_result_cache as trc


def test_module_imports_and_reports_numpy_availability():
    assert isinstance(trc.NUMPY_AVAILABLE, bool)
    # The guard resolved one way or the other; both are valid import outcomes.
    if trc.NUMPY_AVAILABLE:
        assert trc.np is not None
    else:
        assert trc.np is None


def test_normalize_matches_manual_cosine_preprocessing():
    if trc.np is None:
        pytest.skip("numpy not installed in this environment")
    cache = trc.ToolResultCache(embedding_service=object())
    vec = cache._normalize([3.0, 4.0])
    norm = math.sqrt(sum(x * x for x in vec))
    assert norm == pytest.approx(1.0, abs=1e-6)
    # Zero vector is returned unchanged (no div-by-zero).
    zero = cache._normalize([0.0, 0.0])
    assert list(zero) == [0.0, 0.0]


def test_normalize_pure_path_matches_numpy_path(monkeypatch):
    """With numpy hidden, the pure-Python normalize must agree with numpy's."""
    np = pytest.importorskip("numpy")
    cache = trc.ToolResultCache(embedding_service=object())
    numpy_result = cache._normalize(np.array([3.0, 4.0], dtype="float32"))

    monkeypatch.setattr(trc, "np", None)
    pure_result = list(cache._normalize([3.0, 4.0]))

    assert pure_result == pytest.approx([float(x) for x in numpy_result], rel=1e-6)
