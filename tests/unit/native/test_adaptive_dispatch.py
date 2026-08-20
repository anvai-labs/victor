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

"""Gap 3: adaptive accelerator dispatch (close the measurement→dispatch loop).

Two layers:
- Pure-Python tests of OperationStats EWMA / preferred_backend (always run).
- Integration tests through get_preferred_backend() (skip when the native
  extension is unavailable, since adaptive only operates when a Rust backend is
  a real choice).
"""

import pytest

from victor.native.observability import (
    NativeMetrics,
    OperationStats,
    get_operation_stats,
    reset_native_metrics,
)
from victor.processing.native import is_native_available
from victor.processing.native.accelerator import (
    calibrate_operation,
    get_preferred_backend,
    set_accelerator_preference,
)

requires_native = pytest.mark.skipif(
    not is_native_available(), reason="adaptive dispatch needs the native backend"
)


@pytest.fixture(autouse=True)
def _isolate_metrics():
    """Reset the shared metrics sink + any accelerator overrides between tests."""
    import victor.processing.native.accelerator as accel

    reset_native_metrics()
    accel._accelerator_overrides.clear()
    yield
    reset_native_metrics()
    accel._accelerator_overrides.clear()


# =============================================================================
# OperationStats EWMA — pure Python (always runs)
# =============================================================================


class TestOperationStatsAdaptive:
    def test_preferred_backend_none_without_enough_samples(self):
        stats = OperationStats()
        for _ in range(10):
            stats.record(1.0, used_rust=True)
            stats.record(5.0, used_rust=False)
        assert stats.preferred_backend(min_samples=20) is None

    def test_preferred_backend_picks_faster(self):
        stats = OperationStats()
        for _ in range(25):
            stats.record(1.0, used_rust=True)  # rust fast
            stats.record(5.0, used_rust=False)  # python slow
        assert stats.preferred_backend(min_samples=20) == "rust"

    def test_preferred_backend_flips_when_python_faster(self):
        stats = OperationStats()
        for _ in range(25):
            stats.record(8.0, used_rust=True)  # rust slow
            stats.record(0.5, used_rust=False)  # python fast
        assert stats.preferred_backend(min_samples=20) == "python"

    def test_update_ewma_false_does_not_populate_ewma(self):
        """Best-effort observers (update_ewma=False) must not pollute EWMA."""
        stats = OperationStats()
        for _ in range(50):
            stats.record(9.0, used_rust=True, update_ewma=False)
            stats.record(9.0, used_rust=False, update_ewma=False)
        assert stats.rust_ewma_samples == 0
        assert stats.python_ewma_samples == 0
        assert stats.preferred_backend(min_samples=1) is None
        # Aggregate counters still updated.
        assert stats.calls == 100

    def test_to_dict_includes_ewma_fields(self):
        d = OperationStats().to_dict()
        for key in (
            "rust_ewma_ms",
            "python_ewma_ms",
            "rust_ewma_samples",
            "python_ewma_samples",
        ):
            assert key in d


# =============================================================================
# get_preferred_backend integration (native-gated)
# =============================================================================


@requires_native
class TestAdaptiveDispatchIntegration:
    def test_cold_start_falls_through_to_default(self):
        """No observations → adaptive returns None → default branch (rust)."""
        assert get_preferred_backend("token_counting") == "rust"

    def test_adaptive_picks_faster_backend(self):
        m = NativeMetrics.get_instance()
        for _ in range(25):
            m.record_call("token_counting", 1.0, used_rust=True)
            m.record_call("token_counting", 5.0, used_rust=False)
        assert get_preferred_backend("token_counting") == "rust"

    def test_adaptive_self_corrects_to_python(self):
        """An op the default sends to Rust self-corrects to Python when Python
        is observed faster — the core value of closing the loop."""
        m = NativeMetrics.get_instance()
        for _ in range(25):
            m.record_call("token_counting", 8.0, used_rust=True)
            m.record_call("token_counting", 0.5, used_rust=False)
        assert get_preferred_backend("token_counting") == "python"

    def test_user_override_beats_adaptive(self):
        m = NativeMetrics.get_instance()
        for _ in range(25):
            m.record_call("token_counting", 1.0, used_rust=True)
            m.record_call("token_counting", 5.0, used_rust=False)
        try:
            set_accelerator_preference("token_counting", "python")
            assert get_preferred_backend("token_counting") == "python"
        finally:
            set_accelerator_preference("token_counting", "auto")

    def test_disabled_flag_skips_adaptive(self, monkeypatch):
        """When the setting is disabled, adaptive is skipped → default branch."""
        m = NativeMetrics.get_instance()
        for _ in range(25):
            m.record_call("token_counting", 8.0, used_rust=True)
            m.record_call("token_counting", 0.5, used_rust=False)
        monkeypatch.setattr(
            "victor.processing.native.accelerator._get_adaptive_settings",
            lambda: {"enabled": False, "min_samples": 20, "alpha": 0.3},
        )
        # Disabled → falls through static benchmark/default = rust.
        assert get_preferred_backend("token_counting") == "rust"

    def test_calibrate_operation_seeds_ewma(self):
        """calibrate_operation records known-backend samples into the sink."""

        def rust_fn(_):
            pass

        def slow_python_fn(_):
            pass

        result = calibrate_operation(
            "token_counting",
            rust_fn=rust_fn,
            python_fn=slow_python_fn,
            inputs=["a", "b", "c"],
            rounds=30,
        )
        assert result["rust_ms_avg"] >= 0.0
        assert result["python_ms_avg"] >= 0.0
        stats = get_operation_stats("token_counting")
        assert stats is not None
        assert stats.rust_ewma_samples > 0
        assert stats.python_ewma_samples > 0
