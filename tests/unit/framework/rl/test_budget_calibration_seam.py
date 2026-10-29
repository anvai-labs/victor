"""TDD tests for the RL budget-calibration application seam.

Phase 3: a pure function that composes reader -> calibrator -> overlay and is
gated by the settings toggle (default OFF = unchanged behavior). This is the
single entry point a session-bootstrap site would call.

Gating contract:
* tool_budget_calibration_enabled=False (default) -> baseline returned unchanged.
* enabled but low confidence -> baseline returned unchanged.
* enabled with sufficient confidence -> calibrated overlay applied.
"""

from __future__ import annotations

from typing import Any, Dict


from victor.config.settings import Settings
from victor.config.tool_settings import ToolSettings
from victor.framework.rl.budget_calibration_seam import apply_budget_calibration

# ---------------------------------------------------------------------------
# Test doubles (same idiom as test_budget_signal_reader.py)
# ---------------------------------------------------------------------------


class _Row:
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)


class _RoutingCursor:
    def __init__(self, tool_rows, outcome_rows):
        self._tool = tool_rows
        self._outcome = outcome_rows

    def execute(self, query: str, params: Any = ()):
        self._current = self._tool if "rl_tool_q" in query else self._outcome
        return self

    def fetchall(self):
        return [_Row(r) for r in self._current]


class FakeDB:
    def __init__(self, tool_rows, outcome_rows):
        self._tool = tool_rows
        self._outcome = outcome_rows

    def cursor(self):
        return _RoutingCursor(self._tool, self._outcome)


# Reliable signals: many samples, mixed Q.
GOOD_TOOLS = [
    {
        "tool_name": "read",
        "q_value": 0.92,
        "selection_count": 6189,
        "success_count": 5855,
    },
    {
        "tool_name": "shell",
        "q_value": 0.36,
        "selection_count": 271,
        "success_count": 120,
    },
]
# 1303 rows, ~78% failure (matches observed decision_outcome).
FAILING_OUTCOMES = [{"success": 0, "quality_score": 0.0, "attributed_reward": 0.0}] * 1015 + [
    {"success": 1, "quality_score": 0.9, "attributed_reward": 1.0}
] * 288


def _settings(enabled: bool = False, min_conf: float = 0.5, budget: int = 100) -> Settings:
    """Build a Settings instance with controllable calibration flags."""
    s = Settings()
    s.tools = ToolSettings(
        tool_call_budget=budget,
        tool_budget_calibration_enabled=enabled,
        tool_budget_calibration_min_confidence=min_conf,
    )
    return s


# ---------------------------------------------------------------------------
# Gating contract
# ---------------------------------------------------------------------------


class TestGatingContract:
    def test_disabled_returns_baseline_unchanged(self):
        base = _settings(enabled=False)
        db = FakeDB(GOOD_TOOLS, FAILING_OUTCOMES)
        result = apply_budget_calibration(base, db=db)  # type: ignore[arg-type]
        assert result is base.tools  # identity: no work done, no overlay

    def test_enabled_low_confidence_returns_baseline_unchanged(self):
        # Only decisions, no tool signals -> calibrator returns low confidence.
        base = _settings(enabled=True, min_conf=0.95)
        db = FakeDB([], FAILING_OUTCOMES)
        result = apply_budget_calibration(base, db=db)  # type: ignore[arg-type]
        assert result is base.tools

    def test_enabled_high_confidence_applies_overlay(self):
        base = _settings(enabled=True, min_conf=0.4, budget=100)
        db = FakeDB(GOOD_TOOLS, FAILING_OUTCOMES)
        result = apply_budget_calibration(base, db=db)  # type: ignore[arg-type]
        assert result is not base.tools  # new overlay instance
        # 78% failure tightens the budget below baseline.
        assert result.tool_call_budget <= base.tools.tool_call_budget

    def test_overlay_is_immutable(self):
        base = _settings(enabled=True, min_conf=0.4, budget=100)
        db = FakeDB(GOOD_TOOLS, FAILING_OUTCOMES)
        original_budget = base.tools.tool_call_budget
        apply_budget_calibration(base, db=db)  # type: ignore[arg-type]
        # Baseline never mutated.
        assert base.tools.tool_call_budget == original_budget

    def test_explicit_override_short_circuits_before_db_read(self):
        """FEP-0002 precedence: an explicit override wins absolutely.

        Even with calibration enabled + high-confidence signals present, the
        seam returns the baseline by identity (no DB read, no overlay) when
        explicit_override=True. Verified via a DB that raises if read.
        """
        base = _settings(enabled=True, min_conf=0.0, budget=100)

        class ExplodingDB:
            def cursor(self):
                raise AssertionError("explicit_override must short-circuit before any DB read")

        result = apply_budget_calibration(
            base, db=ExplodingDB(), explicit_override=True  # type: ignore[arg-type]
        )
        assert result is base.tools  # identity, no overlay

    def test_explicit_override_false_uses_normal_pipeline(self):
        """Default (explicit_override=False) must not change existing behavior."""
        base = _settings(enabled=True, min_conf=0.4, budget=100)
        db = FakeDB(GOOD_TOOLS, FAILING_OUTCOMES)
        result = apply_budget_calibration(
            base, db=db, explicit_override=False  # type: ignore[arg-type]
        )
        assert result is not base.tools  # overlay applied as usual


# ---------------------------------------------------------------------------
# Cold-start / error safety
# ---------------------------------------------------------------------------


class TestColdStartSafety:
    def test_empty_db_returns_baseline(self):
        base = _settings(enabled=True, min_conf=0.0)
        db = FakeDB([], [])
        result = apply_budget_calibration(base, db=db)  # type: ignore[arg-type]
        # No signal -> confidence 0.0; with min_conf=0.0 the equality is < not <=,
        # so baseline is still retained (no calibration to apply).
        assert result.tool_call_budget == base.tools.tool_call_budget

    def test_db_exception_returns_baseline(self):
        base = _settings(enabled=True, min_conf=0.0)

        class ExplodingDB:
            def cursor(self):
                raise RuntimeError("db unavailable")

        result = apply_budget_calibration(base, db=ExplodingDB())  # type: ignore[arg-type]
        assert result is base.tools
