"""Guard tests for judge-identity pinning (ADR-011, FINDINGS checklist item 3).

The contract: rubric/hybrid completion gating is honored only for calibrated
judge models; anything else downgrades to "enhanced" loudly. The heuristic
fallback (measured α=−0.092) must never gate alone, so a missing model also
downgrades.
"""

from types import SimpleNamespace

import pytest

from victor.agent.services import judge_calibration_gate as gate
from victor.agent.services.judge_calibration_gate import (
    DEFAULT_CALIBRATED_JUDGE_MODELS,
    resolve_gated_completion_strategy,
)


@pytest.fixture(autouse=True)
def _reset_warn_dedupe():
    gate._warned_models.clear()
    yield
    gate._warned_models.clear()


def _settings(models=None):
    if models is None:
        return SimpleNamespace(agent=SimpleNamespace())
    return SimpleNamespace(agent=SimpleNamespace(rubric_judge_calibrated_models=models))


class TestCalibratedJudgePassthrough:
    def test_default_calibrated_models_pass(self):
        for model in DEFAULT_CALIBRATED_JUDGE_MODELS:
            assert resolve_gated_completion_strategy("rubric", model, _settings()) == "rubric"

    def test_match_is_case_insensitive(self):
        assert resolve_gated_completion_strategy("rubric", "Gemma4:31B", _settings()) == "rubric"

    def test_hybrid_passes_for_calibrated_judge(self):
        assert resolve_gated_completion_strategy("hybrid", "llama3.3:70b", _settings()) == "hybrid"

    def test_configured_list_overrides_default(self):
        settings = _settings(models=["my-custom-judge:8b"])
        assert (
            resolve_gated_completion_strategy("rubric", "my-custom-judge:8b", settings) == "rubric"
        )
        # And the defaults are no longer honored once a list is configured.
        assert resolve_gated_completion_strategy("rubric", "gemma4:31b", settings) == "enhanced"


class TestUncalibratedJudgeDowngrade:
    def test_uncalibrated_model_downgrades_to_enhanced(self, caplog):
        with caplog.at_level("WARNING"):
            result = resolve_gated_completion_strategy("rubric", "qwen2.5:0.5b", _settings())
        assert result == "enhanced"
        assert "not in the calibrated set" in caplog.text

    def test_missing_model_downgrades(self):
        # No provider model → the rubric judge cannot exist; the heuristic
        # fallback must never gate alone (ADR-011).
        assert resolve_gated_completion_strategy("rubric", None, _settings()) == "enhanced"
        assert resolve_gated_completion_strategy("rubric", "", _settings()) == "enhanced"

    def test_hybrid_downgrades_for_uncalibrated_judge(self):
        assert (
            resolve_gated_completion_strategy("hybrid", "qwen2.5:0.5b", _settings()) == "enhanced"
        )

    def test_warning_deduplicated_per_model(self, caplog):
        with caplog.at_level("WARNING"):
            resolve_gated_completion_strategy("rubric", "qwen2.5:0.5b", _settings())
            resolve_gated_completion_strategy("rubric", "qwen2.5:0.5b", _settings())
        assert caplog.text.count("not in the calibrated set") == 1


class TestNonJudgeStrategiesUntouched:
    @pytest.mark.parametrize("strategy", ["enhanced", "legacy", "anything-else"])
    def test_passthrough(self, strategy):
        assert resolve_gated_completion_strategy(strategy, "unknown", _settings()) == strategy


class TestEscapeHatch:
    def test_env_bypass_allows_uncalibrated(self, monkeypatch):
        monkeypatch.setenv("VICTOR_RUBRIC_JUDGE_ALLOW_UNCALIBRATED", "1")
        assert resolve_gated_completion_strategy("rubric", "qwen2.5:0.5b", _settings()) == "rubric"

    def test_env_bypass_off_values_do_not_bypass(self, monkeypatch):
        monkeypatch.setenv("VICTOR_RUBRIC_JUDGE_ALLOW_UNCALIBRATED", "0")
        assert (
            resolve_gated_completion_strategy("rubric", "qwen2.5:0.5b", _settings()) == "enhanced"
        )


class TestSettingsDefault:
    def test_agent_settings_default_matches_findings_gate_passers(self):
        from victor.config.groups.agent_config import AgentSettings

        assert set(AgentSettings().rubric_judge_calibrated_models) == set(
            DEFAULT_CALIBRATED_JUDGE_MODELS
        )
