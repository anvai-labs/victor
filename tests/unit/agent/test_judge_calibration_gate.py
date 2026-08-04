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
        assert resolve_gated_completion_strategy("rubric", "Llama3.3:70B", _settings()) == "rubric"

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


class TestResolveEffectiveCompletionStrategy:
    def test_settings_default_flows_through_gate(self, monkeypatch):
        monkeypatch.delenv("VICTOR_COMPLETION_STRATEGY", raising=False)
        settings = SimpleNamespace(agent=SimpleNamespace(completion_strategy="rubric"))
        from victor.agent.services.judge_calibration_gate import (
            resolve_completion_strategy,
        )

        assert resolve_completion_strategy(settings, "llama3.3:70b") == "rubric"
        assert resolve_completion_strategy(settings, "qwen2.5:0.5b") == "enhanced"

    def test_env_override_still_gated(self, monkeypatch):
        monkeypatch.setenv("VICTOR_COMPLETION_STRATEGY", "rubric")
        from victor.agent.services.judge_calibration_gate import (
            resolve_completion_strategy,
        )

        assert resolve_completion_strategy(None, "llama3.3:70b") == "rubric"
        assert resolve_completion_strategy(None, "uncalibrated:1b") == "enhanced"

    def test_missing_settings_defaults_enhanced(self, monkeypatch):
        monkeypatch.delenv("VICTOR_COMPLETION_STRATEGY", raising=False)
        from victor.agent.services.judge_calibration_gate import (
            resolve_completion_strategy,
        )

        assert resolve_completion_strategy(None, None) == "enhanced"


class TestCompletionJudgeBackend:
    """FEP-0030 Phase 1: the judge model is resolved from agent.completion_judge,
    independent of the session model. Default 'session-model' = historical behavior."""

    def _resolve(self):
        from victor.agent.services.judge_calibration_gate import (
            resolve_completion_judge_model,
        )

        return resolve_completion_judge_model

    def test_default_backend_is_session_model(self, monkeypatch):
        monkeypatch.delenv("VICTOR_COMPLETION_JUDGE", raising=False)
        resolve = self._resolve()
        # No completion_judge configured → judge is the session model (unchanged behavior).
        assert resolve(SimpleNamespace(agent=SimpleNamespace()), "llama3.3:70b") == "llama3.3:70b"
        assert resolve(None, "qwen2.5:0.5b") == "qwen2.5:0.5b"

    def test_enhanced_backend_forces_none(self, monkeypatch):
        monkeypatch.delenv("VICTOR_COMPLETION_JUDGE", raising=False)
        resolve = self._resolve()
        settings = SimpleNamespace(agent=SimpleNamespace(completion_judge="enhanced"))
        assert resolve(settings, "llama3.3:70b") is None

    def test_enhanced_backend_downgrades_rubric_even_for_calibrated_session(self, monkeypatch):
        # The whole point: even a calibrated session model gets enhanced when the
        # backend is explicitly 'enhanced'.
        monkeypatch.delenv("VICTOR_COMPLETION_JUDGE", raising=False)
        monkeypatch.delenv("VICTOR_COMPLETION_STRATEGY", raising=False)
        from victor.agent.services.judge_calibration_gate import resolve_completion_strategy

        settings = SimpleNamespace(
            agent=SimpleNamespace(completion_strategy="rubric", completion_judge="enhanced")
        )
        assert resolve_completion_strategy(settings, "llama3.3:70b") == "enhanced"

    def test_phase2_backends_forward_compat_to_session_model(self, monkeypatch):
        monkeypatch.delenv("VICTOR_COMPLETION_JUDGE", raising=False)
        resolve = self._resolve()
        for backend in ("llm:llama3.3:70b", "classifier:/models/j.npz"):
            settings = SimpleNamespace(agent=SimpleNamespace(completion_judge=backend))
            # Not yet wired → falls back to the session model, never breaks the session.
            assert resolve(settings, "llama3.3:70b") == "llama3.3:70b"

    def test_env_override_beats_settings(self, monkeypatch):
        monkeypatch.setenv("VICTOR_COMPLETION_JUDGE", "enhanced")
        resolve = self._resolve()
        settings = SimpleNamespace(agent=SimpleNamespace(completion_judge="session-model"))
        assert resolve(settings, "llama3.3:70b") is None

    def test_agent_settings_default(self):
        from victor.config.groups.agent_config import AgentSettings

        assert AgentSettings().completion_judge == "session-model"


class TestLlmJudgeBackend:
    """FEP-0030 Phase 2: the decoupled 'llm:<model>@<endpoint>' judge backend."""

    def test_parse_completion_judge_forms(self):
        from victor.agent.services.judge_calibration_gate import parse_completion_judge

        assert parse_completion_judge("session-model") == ("session-model", None, None)
        assert parse_completion_judge("enhanced") == ("enhanced", None, None)
        # model may contain colons; endpoint is after the LAST '@'
        assert parse_completion_judge("llm:llama3.3:70b@http://h:11434") == (
            "llm",
            "llama3.3:70b",
            "http://h:11434",
        )
        assert parse_completion_judge("llm:gemma4:31b") == ("llm", "gemma4:31b", None)
        assert parse_completion_judge("classifier:/m/j") == ("classifier", "/m/j", None)
        assert parse_completion_judge("nonsense") == ("unknown", None, None)

    def test_llm_backend_pin_checks_judge_model_not_session(self, monkeypatch):
        # THE Phase 2 win: an uncalibrated CHAT model still gates rubric because
        # the decoupled judge (llama3.3:70b) is calibrated.
        monkeypatch.delenv("VICTOR_COMPLETION_JUDGE", raising=False)
        monkeypatch.delenv("VICTOR_COMPLETION_STRATEGY", raising=False)
        from victor.agent.services.judge_calibration_gate import (
            resolve_completion_judge_model,
            resolve_completion_strategy,
        )

        settings = SimpleNamespace(
            agent=SimpleNamespace(
                completion_strategy="rubric",
                completion_judge="llm:llama3.3:70b@http://host:11434",
            )
        )
        # Session model is uncalibrated qwen — but the judge model is the calibrated llama.
        assert resolve_completion_judge_model(settings, "qwen2.5:0.5b") == "llama3.3:70b"
        assert resolve_completion_strategy(settings, "qwen2.5:0.5b") == "rubric"

    def test_build_judge_complete_fn_session_model(self):
        from unittest.mock import MagicMock

        from victor.agent.services.judge_calibration_gate import build_judge_complete_fn

        pctx = SimpleNamespace(provider=MagicMock(), model="qwen2.5:0.5b")
        fn = build_judge_complete_fn(SimpleNamespace(agent=SimpleNamespace()), pctx)
        assert fn is not None  # session model → a complete_fn

    def test_build_judge_complete_fn_enhanced_is_none(self):
        from victor.agent.services.judge_calibration_gate import build_judge_complete_fn

        pctx = SimpleNamespace(provider=object(), model="m")
        settings = SimpleNamespace(agent=SimpleNamespace(completion_judge="enhanced"))
        assert build_judge_complete_fn(settings, pctx) is None

    def test_build_judge_complete_fn_llm_creates_side_provider(self, monkeypatch):
        from unittest.mock import MagicMock, patch

        from victor.agent.services.judge_calibration_gate import build_judge_complete_fn

        settings = SimpleNamespace(
            agent=SimpleNamespace(completion_judge="llm:llama3.3:70b@http://host:11434")
        )
        pctx = SimpleNamespace(provider=MagicMock(), model="qwen2.5:0.5b")
        with patch(
            "victor.providers.registry.ProviderRegistry.create", return_value=MagicMock()
        ) as create:
            fn = build_judge_complete_fn(settings, pctx)
        assert fn is not None
        create.assert_called_once_with("ollama", base_url="http://host:11434")

    def test_build_judge_complete_fn_side_provider_failure_falls_back(self, monkeypatch):
        from unittest.mock import MagicMock, patch

        from victor.agent.services.judge_calibration_gate import build_judge_complete_fn

        settings = SimpleNamespace(agent=SimpleNamespace(completion_judge="llm:x@http://down"))
        pctx = SimpleNamespace(provider=MagicMock(), model="m")
        with patch(
            "victor.providers.registry.ProviderRegistry.create",
            side_effect=RuntimeError("unreachable"),
        ):
            assert build_judge_complete_fn(settings, pctx) is None


class TestBackendRobustness:
    def test_non_string_completion_judge_defaults_to_session_model(self, monkeypatch):
        # Mocked/misconfigured settings (e.g. an auto-MagicMock) must not reach
        # parse_completion_judge — coerce to the safe default.
        from unittest.mock import MagicMock

        from victor.agent.services.judge_calibration_gate import (
            resolve_completion_judge_model,
        )

        monkeypatch.delenv("VICTOR_COMPLETION_JUDGE", raising=False)
        settings = SimpleNamespace(agent=SimpleNamespace(completion_judge=MagicMock()))
        assert resolve_completion_judge_model(settings, "llama3.3:70b") == "llama3.3:70b"

    def test_empty_completion_judge_defaults_to_session_model(self, monkeypatch):
        from victor.agent.services.judge_calibration_gate import (
            resolve_completion_judge_model,
        )

        monkeypatch.delenv("VICTOR_COMPLETION_JUDGE", raising=False)
        settings = SimpleNamespace(agent=SimpleNamespace(completion_judge="   "))
        assert resolve_completion_judge_model(settings, "qwen") == "qwen"
