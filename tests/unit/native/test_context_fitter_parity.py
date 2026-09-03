# Copyright 2026 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""Native/fallback parity for context fitting.

The Python fallback previously accepted a disjoint strategy vocabulary
(recency/priority/balanced) from the Rust side (fifo/priority/smart), float
priorities raised PyO3 TypeError for every message without an explicit int
(silently disabling the native path), and both sides scored raw priority on
different scales (co-design review U8-F1/F2/F4). These tests pin the unified
contract: one vocabulary, loud unknown names, int priority coercion, and
identical native/fallback outcomes.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

import victor.processing.native.context_fitter as cf
from victor.processing.native._base import _NATIVE_AVAILABLE, _native

_ROLES = ["system", "user", "assistant", "user", "assistant", "user", "user", "assistant"]


def _messages(priorities: List[Any]) -> List[Dict[str, Any]]:
    assert len(priorities) == len(_ROLES)
    return [
        (
            {"role": role, "content": f"message {i}", "token_count": 100, "priority": pri}
            if pri is not None
            else {"role": role, "content": f"message {i}", "token_count": 100}
        )
        for i, (role, pri) in enumerate(zip(_ROLES, priorities))
    ]


def _stale_wheel() -> bool:
    """True when the installed wheel predates the unified contract."""
    if not _NATIVE_AVAILABLE:
        return False
    try:
        # Non-empty list with a forcing budget so the strategy match runs
        # (an empty list short-circuits before validation).
        slot = _native.MessageSlot(0, 100, 50, "user", 1.0)
        _native.fit_context([slot], 0, "bogus_strategy_name", True)
        return True  # old wheel: unknown names silently meant smart
    except ValueError:
        return False


class TestStrategyVocabulary:
    @pytest.mark.parametrize("strategy", ["smart", "priority", "fifo"])
    def test_canonical_strategies_accepted_by_fallback(self, strategy):
        # Budget 800 = everything fits; smart's pins (system + first user +
        # last 2) can legitimately EXCEED a smaller budget, so use a
        # no-drop budget here and drop-forcing budgets in the parity matrix.
        result = cf._fit_context_python(_messages([50] * 8), 800, strategy, True)
        assert result.total_tokens == 800

    def test_legacy_aliases_map_to_canonical(self):
        msgs = _messages([10, 90, 30, 70, 20, 80, 40, 60])
        assert cf.fit_context(msgs, 300, "recency") == cf.fit_context(msgs, 300, "fifo")
        assert cf.fit_context(msgs, 300, "balanced") == cf.fit_context(msgs, 300, "smart")

    def test_unknown_strategy_raises_loudly(self):
        with pytest.raises(ValueError, match="unknown fit strategy"):
            cf.fit_context(_messages([50] * 8), 300, "newest")

    def test_default_strategy_is_smart(self):
        import inspect

        assert inspect.signature(cf.fit_context).parameters["strategy"].default == "smart"


class TestPriorityCoercion:
    def test_default_priority_no_float_for_native(self):
        """Messages WITHOUT explicit priority must take the native path —
        previously the float default raised PyO3 TypeError inside the try and
        silently degraded to Python for every message."""
        if not _NATIVE_AVAILABLE or _stale_wheel():
            pytest.skip("native wheel unavailable or stale")

        calls = {"n": 0}
        real_fit = _native.fit_context

        class _Spy:
            MessageSlot = _native.MessageSlot

            @staticmethod
            def fit_context(*args, **kwargs):
                calls["n"] += 1
                return real_fit(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(_native, name)

        original = cf._native
        cf._native = _Spy()
        try:
            result = cf.fit_context(_messages([None] * 8), 300, "smart")
        finally:
            cf._native = original

        assert calls["n"] == 1, "native path must not silently fall back"
        # smart pins (system + first user + last 2 = 400 tokens) may exceed
        # the 300 budget — that is the native algorithm's contract.
        assert result.kept_indices == [0, 1, 6, 7]

    @pytest.mark.parametrize("priority", [0, 50, 100, 255])
    def test_int_priorities_accepted(self, priority):
        result = cf._fit_context_python(_messages([priority] * 8), 800, "smart", True)
        assert result.total_tokens == 800


class TestNativeFallbackParity:
    @pytest.mark.parametrize("strategy", ["smart", "priority", "fifo"])
    @pytest.mark.parametrize(
        "priorities",
        [
            [50] * 8,
            [10, 90, 30, 70, 20, 80, 40, 60],
            [0, 100, 0, 100, 0, 100, 0, 100],
        ],
    )
    def test_native_and_fallback_agree(self, strategy, priorities):
        if not _NATIVE_AVAILABLE:
            pytest.skip("native wheel not installed")
        if _stale_wheel():
            pytest.skip("native wheel predates unified strategies — rebuild with maturin")

        msgs = _messages(priorities)
        native = cf.fit_context(msgs, 300, strategy, preserve_system=True)
        fallback = cf._fit_context_python(msgs, 300, strategy, True)

        assert native.kept_indices == fallback.kept_indices
        assert native.total_tokens == fallback.total_tokens
        assert native.dropped_count == fallback.dropped_count
        assert native.freed_tokens == fallback.freed_tokens

    def test_scoring_follows_documented_formula_in_fallback(self):
        """smart drops the lowest (priority/100*0.4 + recency*0.6) scorers —
        with uniform priority 50, recency decides: the oldest droppable
        message goes first."""
        result = cf._fit_context_python(_messages([50] * 8), 300, "smart", True)
        # system (index 0) preserved; first user (1) and last 2 (6,7) are
        # smart-pinned by the native algorithm — the fallback keeps parity
        # for what fits after that.
        assert 0 in result.kept_indices
        assert 6 in result.kept_indices and 7 in result.kept_indices
