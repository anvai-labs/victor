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


def test_probe_tolerates_find_spec_errors(monkeypatch):
    """find_spec raises ValueError on a poisoned sys.modules entry; the probe
    must treat that as 'absent' rather than crash the dispatch path."""
    import importlib.util

    def boom(name):
        if name == "numpy":
            raise ValueError("poisoned")
        return importlib.util.find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", boom)
    native_base.reset_numpy_probe_for_tests()
    assert native_base.numpy_accelerator_available() is False


class TestNativeVersionGuard:
    """evaluate_native_version + import-time drift check.

    Regression context (2026-11-06): a venv silently kept victor_native 0.15.0
    under victor-ai 0.26.x; the stale wheel carried a dedup livelock that pinned
    a chat session at 100% CPU. These tests pin the guard that makes such drift
    loud instead of silent.
    """

    def test_identical_versions_ok(self):
        assert native_base.evaluate_native_version("0.26.2", "0.26.2") == "ok"

    def test_patch_drift_detected(self):
        assert native_base.evaluate_native_version("0.26.1", "0.26.2") == "patch-drift"

    def test_stated_major_minor_drift_is_stale(self):
        # The exact incident shape: 0.15.0 wheel under 0.26.x Python code.
        assert native_base.evaluate_native_version("0.15.0", "0.26.2") == "stale"

    def test_newer_native_is_forward_compatible(self):
        assert native_base.evaluate_native_version("0.27.0", "0.26.2") == "newer"

    def test_missing_or_unparseable_versions_are_unknown(self):
        assert native_base.evaluate_native_version(None, "0.26.2") == "unknown"
        assert native_base.evaluate_native_version("0.15.0", None) == "unknown"
        assert native_base.evaluate_native_version("dev", "0.26.2") == "unknown"

    def test_dev_suffix_on_patch_still_classifies(self):
        # "0.26.2-dev" has parseable major.minor; classification must not
        # bail to "unknown" just because the patch carries a suffix.
        assert native_base.evaluate_native_version("0.15.0", "0.26.2-dev") == "stale"
        assert native_base.evaluate_native_version("0.26.2-dev", "0.26.2") == "patch-drift"

    def test_skip_check_escape_hatch(self):
        assert native_base.evaluate_native_version("0.15.0", "0.26.2", skip_check=True) == "skipped"

    def test_import_time_check_flags_stale_wheel(self, monkeypatch, caplog):
        """A stale victor_native at import time records 'stale' and warns loudly."""
        import importlib
        import sys
        import types

        fake = types.ModuleType("victor_native")
        fake.__version__ = "0.15.0"
        monkeypatch.setitem(sys.modules, "victor_native", fake)
        monkeypatch.setattr("importlib.metadata.version", lambda name: "0.26.2", raising=False)
        monkeypatch.delenv("VICTOR_NATIVE_SKIP_VERSION_CHECK", raising=False)

        with caplog.at_level("WARNING", logger="victor.processing.native._base"):
            reloaded = importlib.reload(native_base)

        try:
            status = reloaded.get_native_version_status()
            assert status["state"] == "stale"
            assert status["native_version"] == "0.15.0"
            assert status["package_version"] == "0.26.2"
            assert any(
                "stale native wheel" in r.message for r in caplog.records
            ), "stale wheel must produce a loud warning"
        finally:
            importlib.reload(native_base)  # restore real module binding

    def test_import_time_check_silent_when_matched(self, monkeypatch, caplog):
        """Matching versions record 'ok' without any warning."""
        import importlib
        import sys
        import types

        real_version = getattr(native_base._native, "__version__", None)
        fake = types.ModuleType("victor_native")
        fake.__version__ = real_version or "0.0.0"
        monkeypatch.setitem(sys.modules, "victor_native", fake)
        monkeypatch.setattr(
            "importlib.metadata.version",
            lambda name: (real_version or "0.0.0") if name == "victor-ai" else None,
            raising=False,
        )

        with caplog.at_level("WARNING", logger="victor.processing.native._base"):
            reloaded = importlib.reload(native_base)

        try:
            assert reloaded.get_native_version_status()["state"] in {"ok", "unknown"}
            assert not any(
                "stale native wheel" in r.message for r in caplog.records
            ), "matched versions must not warn"
        finally:
            importlib.reload(native_base)
