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

"""Tests for victor/processing/native/_base.py: the numpy accelerator probe
and the PYTHON_PREFERRED dispatch policy.

Mirrored location for scripts/ci/select_changed_tests.py: changed source
``victor/processing/native/_base.py`` maps here.
"""

from __future__ import annotations

import pytest

from victor.processing.native import _base as native_base


@pytest.fixture(autouse=True)
def _reset_numpy_probe():
    native_base.reset_numpy_probe_for_tests()
    yield
    native_base.reset_numpy_probe_for_tests()


def test_probe_is_cached_and_resettable(monkeypatch):
    calls = []
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def counting_find_spec(name):
        if name == "numpy":
            calls.append(name)
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", counting_find_spec)
    native_base.reset_numpy_probe_for_tests()
    first = native_base.numpy_accelerator_available()
    second = native_base.numpy_accelerator_available()
    assert first is second
    assert len(calls) == 1  # cached: exactly one spec lookup
    native_base.reset_numpy_probe_for_tests()
    native_base.numpy_accelerator_available()
    assert len(calls) == 2


def test_probe_treats_poisoned_numpy_as_absent(monkeypatch):
    """The bare-install simulation (sys.modules['numpy'] = None) makes
    find_spec raise ValueError — the probe must report absent, not crash."""
    import sys

    monkeypatch.setitem(sys.modules, "numpy", None)
    native_base.reset_numpy_probe_for_tests()
    assert native_base.numpy_accelerator_available() is False


def test_preferred_ops_dispatch_to_native_without_numpy(monkeypatch):
    """The PYTHON_PREFERRED batch ops route to native when numpy is absent."""
    monkeypatch.setattr(native_base, "numpy_accelerator_available", lambda: False)
    monkeypatch.setattr(native_base, "_NATIVE_AVAILABLE", True)
    for op in ("batch_cosine_similarity", "similarity_matrix"):
        assert native_base.should_use_native_for_operation(op) is True
    # Non-preferred ops always native when available.
    assert native_base.should_use_native_for_operation("cosine_similarity") is True


def test_preferred_ops_keep_numpy_when_present(monkeypatch):
    """numpy-installed environments keep the BLAS batch path (no regression)."""
    if not native_base.numpy_accelerator_available():
        pytest.skip("numpy not installed in this environment")
    monkeypatch.setattr(native_base, "_NATIVE_AVAILABLE", True)
    for op in ("batch_cosine_similarity", "similarity_matrix"):
        assert native_base.should_use_native_for_operation(op) is False


def test_policy_without_native_is_always_false(monkeypatch):
    monkeypatch.setattr(native_base, "_NATIVE_AVAILABLE", False)
    monkeypatch.setattr(native_base, "numpy_accelerator_available", lambda: False)
    assert native_base.should_use_native_for_operation("batch_cosine_similarity") is False
