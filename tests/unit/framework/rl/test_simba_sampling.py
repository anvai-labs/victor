"""Tests for SIMBA challenge-focused trace sampling (DSPy-inspired)."""

from unittest.mock import MagicMock


class TestSIMBAChallengeSampling:
    def _make_trace(self, success=True, score=0.8, failures=None, has_errors=False):
        t = MagicMock()
        t.success = success
        t.completion_score = score
        t.tool_failures = failures or {}
        details = []
        if has_errors:
            d = MagicMock()
            d.error_detail = "old_str not found"
            details.append(d)
        t.tool_call_details = details
        return t

    def test_function_exists(self):
        from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner

        assert hasattr(PromptOptimizerLearner, "_select_challenging_traces")

    def test_recovery_traces_preferred(self):
        from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner

        learner = MagicMock(spec=PromptOptimizerLearner)
        # Recovery: success + failures
        recovery = self._make_trace(success=True, score=0.7, failures={"edit_mismatch": 2})
        easy = self._make_trace(success=True, score=0.95)
        traces = [easy, recovery, easy, easy]

        result = PromptOptimizerLearner._select_challenging_traces(learner, traces, max_traces=2)
        # Recovery trace should be in result
        assert recovery in result

    def test_high_failure_traces_boosted(self):
        from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner

        learner = MagicMock(spec=PromptOptimizerLearner)
        high_fail = self._make_trace(
            success=False, score=0.3, failures={"edit_mismatch": 5}, has_errors=True
        )
        low_fail = self._make_trace(success=False, score=0.4, failures={"timeout": 1})
        traces = [low_fail, high_fail]

        result = PromptOptimizerLearner._select_challenging_traces(learner, traces, max_traces=2)
        assert result[0] == high_fail  # Higher challenge first

    def test_easy_successes_included(self):
        from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner

        learner = MagicMock(spec=PromptOptimizerLearner)
        hard = self._make_trace(success=False, score=0.2, failures={"edit_mismatch": 3})
        easy = self._make_trace(success=True, score=0.95)
        traces = [hard] * 15 + [easy] * 10

        result = PromptOptimizerLearner._select_challenging_traces(learner, traces, max_traces=10)
        # Should include some easy traces for contrast
        easy_count = sum(1 for t in result if t.success and t.completion_score > 0.9)
        assert easy_count >= 1

    def test_under_limit_returns_all(self):
        from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner

        learner = MagicMock(spec=PromptOptimizerLearner)
        traces = [self._make_trace() for _ in range(5)]
        result = PromptOptimizerLearner._select_challenging_traces(learner, traces, max_traces=20)
        assert len(result) == 5
