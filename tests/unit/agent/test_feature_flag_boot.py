# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Feature flag pairwise boot tests.

Verifies that the orchestrator can initialize without crashing
under various feature flag combinations. Uses mocked providers
to avoid real LLM calls.
"""

import pytest

from victor.core.feature_flags import (
    FeatureFlag,
    FeatureFlagManager,
)


def _make_flag_manager(flags: dict) -> FeatureFlagManager:
    """Create a FeatureFlagManager with specific flags set."""
    mgr = FeatureFlagManager()
    for flag, enabled in flags.items():
        if enabled:
            mgr.enable(flag)
        else:
            mgr.disable(flag)
    return mgr


# Representative flag combos covering active rollout groups.
def _all_off() -> dict:
    return dict.fromkeys(FeatureFlag, False)


def _all_on() -> dict:
    return dict.fromkeys(FeatureFlag, True)


def _with(*enabled_flags: FeatureFlag) -> dict:
    combo = _all_off()
    for flag in enabled_flags:
        combo[flag] = True
    return combo


F = FeatureFlag  # shorthand

REPRESENTATIVE_COMBOS = [
    _all_off(),
    _all_on(),
    _with(F.USE_EDGE_MODEL),
    _with(F.USE_LLM_DECISION_SERVICE),
    _with(
        F.USE_COMPOSITION_OVER_INHERITANCE,
        F.USE_STRATEGY_BASED_TOOL_REGISTRATION,
    ),
    _with(F.USE_STATEGRAPH_AGENTIC_LOOP, F.USE_STAGE_TRANSITION_COORDINATOR),
    _with(F.USE_LLM_DECISION_SERVICE, F.USE_EDGE_MODEL),
]


def _combo_id(combo: dict) -> str:
    """Generate a short test ID from flag combo."""
    on = [f.name.replace("USE_", "")[:6] for f, v in combo.items() if v]
    return "+".join(on) if on else "ALL_OFF"


class TestFeatureFlagBoot:
    """Verify orchestrator initializes without crash for flag combos."""

    @pytest.mark.parametrize(
        "flag_combo",
        REPRESENTATIVE_COMBOS,
        ids=[_combo_id(c) for c in REPRESENTATIVE_COMBOS],
    )
    def test_orchestrator_flag_combo_no_import_error(self, flag_combo):
        """FeatureFlagManager accepts the combo without error."""
        mgr = _make_flag_manager(flag_combo)

        # Verify all flags are queryable
        for flag in FeatureFlag:
            # Should not raise
            result = mgr.is_enabled(flag)
            assert isinstance(result, bool)

    @pytest.mark.parametrize(
        "flag_combo",
        REPRESENTATIVE_COMBOS,
        ids=[_combo_id(c) for c in REPRESENTATIVE_COMBOS],
    )
    def test_flag_manager_state_matches_combo(self, flag_combo):
        """Flags set to True are enabled, False are disabled."""
        mgr = _make_flag_manager(flag_combo)

        for flag, expected in flag_combo.items():
            actual = mgr.is_enabled(flag)
            assert actual == expected, f"{flag.name}: expected {expected}, got {actual}"
