# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for SessionConfig.__post_init__ validators — Wave J."""

import pytest

from victor.framework.session_config import SessionConfig


class TestSessionConfigValidation:
    """SessionConfig: tool_budget, max_iterations, and threshold range validators."""

    def test_default_config_passes_validation(self):
        """Default SessionConfig should pass all validators."""
        config = SessionConfig()
        assert config is not None
        assert config.tool_budget is None
        assert config.max_iterations is None

    def test_tool_budget_below_one_raises(self):
        """tool_budget must be >= 1 if set."""
        with pytest.raises(ValueError, match="tool_budget must be >= 1"):
            SessionConfig(tool_budget=0)

        with pytest.raises(ValueError, match="tool_budget must be >= 1"):
            SessionConfig(tool_budget=-5)

    def test_tool_budget_one_passes(self):
        """tool_budget=1 should be allowed (minimum valid value)."""
        config = SessionConfig(tool_budget=1)
        assert config.tool_budget == 1

    def test_max_iterations_below_one_raises(self):
        """max_iterations must be >= 1 if set."""
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            SessionConfig(max_iterations=0)

        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            SessionConfig(max_iterations=-10)

    def test_max_iterations_one_passes(self):
        """max_iterations=1 should be allowed (minimum valid value)."""
        config = SessionConfig(max_iterations=1)
        assert config.max_iterations == 1

    def test_compaction_threshold_out_of_range_raises(self):
        """compaction.threshold must be in [0.0, 1.0] if set."""
        from victor.framework.session_config import CompactionConfig

        with pytest.raises(ValueError, match="compaction.threshold must be in \\[0.0, 1.0\\]"):
            SessionConfig(compaction=CompactionConfig(threshold=-0.1))

        with pytest.raises(ValueError, match="compaction.threshold must be in \\[0.0, 1.0\\]"):
            SessionConfig(compaction=CompactionConfig(threshold=1.5))

    def test_compaction_threshold_bounds_pass(self):
        """compaction.threshold at 0.0 and 1.0 should be allowed."""
        from victor.framework.session_config import CompactionConfig

        config_low = SessionConfig(compaction=CompactionConfig(threshold=0.0))
        assert config_low.compaction.threshold == 0.0

        config_high = SessionConfig(compaction=CompactionConfig(threshold=1.0))
        assert config_high.compaction.threshold == 1.0

    def test_bayesian_thresholds_out_of_range_raise(self):
        """bayesian.simple_threshold and complex_threshold must be in [0.0, 1.0]."""
        from victor.framework.bayesian_config import BayesianConfig

        with pytest.raises(ValueError, match="bayesian.simple_threshold must be in \\[0.0, 1.0\\]"):
            SessionConfig(bayesian=BayesianConfig(simple_threshold=-0.1))

        with pytest.raises(
            ValueError, match="bayesian.complex_threshold must be in \\[0.0, 1.0\\]"
        ):
            SessionConfig(bayesian=BayesianConfig(complex_threshold=1.5))

    def test_bayesian_thresholds_bounds_pass(self):
        """bayesian thresholds at 0.0 and 1.0 should be allowed."""
        from victor.framework.bayesian_config import BayesianConfig

        config_low = SessionConfig(bayesian=BayesianConfig(simple_threshold=0.0))
        assert config_low.bayesian.simple_threshold == 0.0

        config_high = SessionConfig(bayesian=BayesianConfig(complex_threshold=1.0))
        assert config_high.bayesian.complex_threshold == 1.0

    def test_from_cli_flags_with_invalid_tool_budget_raises(self):
        """from_cli_flags should reject tool_budget < 1."""
        with pytest.raises(ValueError, match="tool_budget must be >= 1"):
            SessionConfig.from_cli_flags(tool_budget=0)

    def test_from_cli_flags_with_invalid_max_iterations_raises(self):
        """from_cli_flags should reject max_iterations < 1."""
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            SessionConfig.from_cli_flags(max_iterations=-1)

    def test_from_cli_flags_with_invalid_compaction_threshold_raises(self):
        """from_cli_flags should reject compaction_threshold outside [0, 1]."""
        with pytest.raises(ValueError, match="compaction.threshold must be in \\[0.0, 1.0\\]"):
            SessionConfig.from_cli_flags(compaction_threshold=1.5)

    def test_from_cli_flags_with_invalid_bayesian_thresholds_raises(self):
        """from_cli_flags should reject bayesian thresholds outside [0, 1]."""
        with pytest.raises(ValueError, match="bayesian.simple_threshold must be in \\[0.0, 1.0\\]"):
            SessionConfig.from_cli_flags(simple_threshold=1.1)

        with pytest.raises(
            ValueError, match="bayesian.complex_threshold must be in \\[0.0, 1.0\\]"
        ):
            SessionConfig.from_cli_flags(complex_threshold=-0.5)


class TestSessionConfigApplyToolBudget:
    """SessionConfig.tool_budget must actually land on settings.tools (TDD RED).

    Prior to this fix, ``tool_budget`` was validated but never written by
    ``apply_to_settings``; an explicit ``--tool-budget`` override was silently
    dropped (verified: full method body lines 521-687 never reference
    ``self.tool_budget``). This is also the prerequisite for the FEP-0002
    calibration precedence test (explicit overrides must win).
    """

    def _make_settings(self):
        from victor.config.settings import Settings

        s = Settings()
        # Sanity: canonical consumer field is populated from defaults.
        assert s.tools is not None
        return s

    def test_tool_budget_none_leaves_baseline_untouched(self):
        """No explicit override -> baseline tools.tool_call_budget unchanged."""
        config = SessionConfig()  # tool_budget is None
        settings = self._make_settings()
        before = settings.tools.tool_call_budget
        config.apply_to_settings(settings)
        assert settings.tools.tool_call_budget == before

    def test_tool_budget_applied_to_settings_tools(self):
        """An explicit tool_budget must be written to settings.tools."""
        config = SessionConfig(tool_budget=42)
        settings = self._make_settings()
        baseline = settings.tools.tool_call_budget
        config.apply_to_settings(settings)
        # The whole point: the override must take effect.
        assert settings.tools.tool_call_budget == 42
        assert settings.tools.tool_call_budget != baseline or baseline == 42

    def test_tool_budget_applied_immutably(self):
        """Override lands on settings.tools (consistent with existing
        apply_to_settings writes, which use object.__setattr__ on the nested
        group). The new-immutable-instance guarantee belongs to the
        calibration seam, not SessionConfig."""
        config = SessionConfig(tool_budget=7)
        settings = self._make_settings()
        config.apply_to_settings(settings)
        assert settings.tools.tool_call_budget == 7
