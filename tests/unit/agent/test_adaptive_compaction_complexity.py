"""TDD tests for AdaptiveCompactionThreshold complexity integration — Wave 6.

Verifies that HIGH task complexity reduces the effective compaction threshold,
MEDIUM/LOW complexity leaves it unchanged, and None complexity is a no-op.
"""

from __future__ import annotations

from unittest.mock import MagicMock


class TestCompactorConfigHighComplexityFactor:
    def test_compactor_config_has_high_complexity_factor(self):
        from victor.agent.context_compactor import CompactorConfig

        config = CompactorConfig()
        assert hasattr(config, "high_complexity_threshold_factor")
        assert 0 < config.high_complexity_threshold_factor <= 1.0

    def test_high_complexity_factor_default_is_less_than_one(self):
        from victor.agent.context_compactor import CompactorConfig

        config = CompactorConfig()
        assert config.high_complexity_threshold_factor < 1.0

    def test_high_complexity_factor_configurable(self):
        from victor.agent.context_compactor import CompactorConfig

        config = CompactorConfig(high_complexity_threshold_factor=0.70)
        assert config.high_complexity_threshold_factor == 0.70


class TestShouldCompactWithComplexity:
    """ContextCompactor.should_compact() accepts optional task_complexity."""

    def _make_compactor(self, proactive_threshold=0.80):
        from victor.agent.context_compactor import ContextCompactor, CompactorConfig

        config = CompactorConfig(
            proactive_threshold=proactive_threshold,
            enable_proactive=True,
        )
        compactor = ContextCompactor.__new__(ContextCompactor)
        compactor.config = config

        # Mock controller with adjustable utilization
        metrics = MagicMock()
        metrics.is_overflow_risk = False
        metrics.utilization = 0.75  # below threshold

        controller = MagicMock()
        controller.get_context_metrics.return_value = metrics
        compactor.controller = controller
        compactor._metrics = metrics  # convenience

        return compactor

    def test_should_compact_accepts_task_complexity_kwarg(self):
        compactor = self._make_compactor()
        result = compactor.should_compact(task_complexity=None)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_high_complexity_reduces_effective_threshold(self):
        """With high complexity, a utilization that was below threshold should now trigger."""
        compactor = self._make_compactor(proactive_threshold=0.80)
        # default factor = 0.85 → effective = 0.80 * 0.85 = 0.68
        # utilization = 0.75 → above 0.68 → should compact
        compactor._metrics.utilization = 0.75
        should, trigger = compactor.should_compact(task_complexity="high")
        assert should is True

    def test_medium_complexity_leaves_threshold_unchanged(self):
        """With medium complexity, utilization below threshold should not trigger."""
        compactor = self._make_compactor(proactive_threshold=0.80)
        compactor._metrics.utilization = 0.75  # below 0.80
        should, trigger = compactor.should_compact(task_complexity="medium")
        assert should is False

    def test_low_complexity_leaves_threshold_unchanged(self):
        compactor = self._make_compactor(proactive_threshold=0.80)
        compactor._metrics.utilization = 0.75  # below 0.80
        should, trigger = compactor.should_compact(task_complexity="low")
        assert should is False

    def test_none_complexity_is_noop(self):
        compactor = self._make_compactor(proactive_threshold=0.80)
        compactor._metrics.utilization = 0.75
        should, trigger = compactor.should_compact(task_complexity=None)
        assert should is False

    def test_high_complexity_factor_configurable(self):
        """Custom high_complexity_threshold_factor is used."""
        from victor.agent.context_compactor import CompactorConfig

        compactor = self._make_compactor(proactive_threshold=0.80)
        compactor.config = CompactorConfig(
            proactive_threshold=0.80,
            enable_proactive=True,
            high_complexity_threshold_factor=0.90,  # mild factor
        )
        # utilization=0.75, threshold=0.80*0.90=0.72 → 0.75 > 0.72 → compact
        compactor._metrics.utilization = 0.75
        should, trigger = compactor.should_compact(task_complexity="high")
        assert should is True

    def test_overflow_risk_always_triggers_regardless_of_complexity(self):
        compactor = self._make_compactor(proactive_threshold=0.80)
        compactor._metrics.is_overflow_risk = True
        compactor._metrics.utilization = 0.10  # way below threshold
        should, trigger = compactor.should_compact(task_complexity=None)
        assert should is True
