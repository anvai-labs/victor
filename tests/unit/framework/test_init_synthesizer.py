# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for InitSynthesizer — TDD."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.framework.init_synthesizer import (
    InitSynthesizer,
    SYNTHESIS_RULES,
    SYNTHESIS_PROMPT,
    _build_synthesis_prompt,
    _classify_init_failure,
    _init_provider_timeout,
)


class TestInitProviderTimeout:
    """Adaptive init-synthesis timeout resolution."""

    def test_local_provider_gets_large_budget(self):
        assert _init_provider_timeout("ollama") == 600

    def test_cloud_provider_gets_default(self):
        assert _init_provider_timeout("zai") == 300

    def test_never_reduces_configured_timeout(self):
        assert _init_provider_timeout("zai", configured=900) == 900
        assert _init_provider_timeout("ollama", configured=120) == 600

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("VICTOR_INIT_SYNTH_TIMEOUT", "42")
        assert _init_provider_timeout("ollama", configured=900) == 42

    def test_invalid_env_override_ignored(self, monkeypatch):
        monkeypatch.setenv("VICTOR_INIT_SYNTH_TIMEOUT", "not-a-number")
        assert _init_provider_timeout("zai") == 300


class TestClassifyInitFailure:
    """Failure categorization for actionable template-fallback logging."""

    def test_timeout(self):
        assert _classify_init_failure(Exception("Request timed out after 300s")) == "timeout"

    def test_connection(self):
        assert _classify_init_failure(OSError("connection refused")) == "connection"

    def test_empty_output(self):
        assert _classify_init_failure(Exception("returned empty content")) == "empty_output"

    def test_other(self):
        assert _classify_init_failure(ValueError("weird")) == "ValueError"


@pytest.fixture(autouse=True)
def _stub_expensive_init_enrichment(monkeypatch):
    """Keep synthesize() unit tests isolated from repo-scale graph/doc discovery."""
    monkeypatch.setattr(
        InitSynthesizer,
        "_pre_synthesis_discovery",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        InitSynthesizer,
        "_enrich_with_project_docs",
        staticmethod(lambda base_content: base_content),
    )


class TestInitSynthesizer:
    """Core synthesize() method tests."""

    @pytest.mark.asyncio
    async def test_synthesize_with_agent_uses_provider_directly(self):
        """When agent is provided, synthesize() calls agent.provider.chat() directly.

        Bypasses AgenticLoop — single LLM call: prompt → markdown.
        Uses the already-initialized provider (same credential path as victor chat).
        """
        mock_provider = AsyncMock()
        mock_provider.name = "test-provider"
        mock_provider.chat.return_value = MagicMock(content="# init.md\n\nProject overview.")

        mock_agent = MagicMock()
        mock_agent.provider = mock_provider
        mock_agent.model = "test-model"

        synthesizer = InitSynthesizer()
        result = await synthesizer.synthesize("raw data here", agent=mock_agent)

        mock_provider.chat.assert_called_once()
        call_kwargs = mock_provider.chat.call_args
        messages = call_kwargs[1].get("messages") or call_kwargs[0][0]
        assert any("raw data here" in str(m) for m in messages)
        assert result == "# init.md\n\nProject overview."

    @pytest.mark.asyncio
    async def test_synthesize_without_agent_creates_fresh(self):
        """When no agent, synthesize() uses direct provider call via _run_with_fresh_agent."""
        with patch.object(
            InitSynthesizer, "_run_with_fresh_agent", return_value="# Generated init.md"
        ) as mock_fresh:
            synthesizer = InitSynthesizer()
            result = await synthesizer.synthesize(
                "raw data", provider="ollama", model="qwen3-coder:30b"
            )

            mock_fresh.assert_called_once()
            assert result == "# Generated init.md"

    @pytest.mark.asyncio
    async def test_synthesize_cleans_code_fences(self):
        """Output wrapped in ``` gets cleaned."""
        mock_provider = AsyncMock()
        mock_provider.name = "test-provider"
        mock_provider.chat.return_value = MagicMock(
            content="```markdown\n# Project\n\nOverview.\n```"
        )
        mock_agent = MagicMock()
        mock_agent.provider = mock_provider
        mock_agent.model = None

        synthesizer = InitSynthesizer()
        result = await synthesizer.synthesize("raw data", agent=mock_agent)

        assert result == "# Project\n\nOverview."
        assert "```" not in result

    @pytest.mark.asyncio
    async def test_synthesize_with_agent_omits_none_model(self):
        """Provider calls should not receive model=None as an explicit override."""
        mock_provider = AsyncMock()
        mock_provider.name = "test-provider"
        mock_provider.chat.return_value = MagicMock(content="# init.md\n\nProject overview.")

        mock_agent = MagicMock()
        mock_agent.provider = mock_provider
        mock_agent.model = None

        synthesizer = InitSynthesizer()
        await synthesizer.synthesize("raw data here", agent=mock_agent)

        assert "model" not in mock_provider.chat.call_args.kwargs

    @pytest.mark.asyncio
    async def test_synthesize_with_ollama_agent_resolves_model_before_chat(self):
        """Reused Ollama providers should preflight a model when agent.model is None."""
        mock_provider = MagicMock()
        mock_provider.name = "ollama"
        mock_provider.base_url = "http://localhost:11434"
        mock_provider.list_models = AsyncMock(return_value=[{"name": "qwen3:8b"}])
        mock_provider.chat = AsyncMock(return_value=MagicMock(content="# Result"))

        mock_agent = MagicMock()
        mock_agent.provider = mock_provider
        mock_agent.model = None

        synthesizer = InitSynthesizer()
        mock_settings = MagicMock()
        mock_settings.default_provider = None
        mock_settings.default_model = None
        mock_settings.load_profiles.return_value = {}
        mock_settings.provider = MagicMock(default_provider=None, default_model=None)
        mock_settings.get_provider_settings.return_value = {}

        with patch("victor.config.settings.load_settings", return_value=mock_settings):
            result = await synthesizer.synthesize("raw data here", agent=mock_agent)

        assert result == "# Result"
        assert mock_provider.chat.call_args.kwargs["model"] == "qwen3:8b"

    @pytest.mark.asyncio
    async def test_synthesize_prompt_contains_base_content(self):
        """The synthesis prompt includes the base_content."""
        mock_provider = AsyncMock()
        mock_provider.name = "test-provider"
        mock_provider.chat.return_value = MagicMock(content="# Result")
        mock_agent = MagicMock()
        mock_agent.provider = mock_provider
        mock_agent.model = None

        synthesizer = InitSynthesizer()
        await synthesizer.synthesize("UNIQUE_BASE_DATA_12345", agent=mock_agent)

        call_kwargs = mock_provider.chat.call_args
        messages = call_kwargs[1].get("messages") or call_kwargs[0][0]
        assert any("UNIQUE_BASE_DATA_12345" in str(m) for m in messages)

    @pytest.mark.asyncio
    async def test_synthesize_propagates_provider_failure(self):
        """If provider.chat() raises, synthesis failure propagates to the caller."""
        mock_provider = AsyncMock()
        mock_provider.name = "test-provider"
        mock_provider.chat.side_effect = RuntimeError("LLM failed")
        mock_agent = MagicMock()
        mock_agent.provider = mock_provider
        mock_agent.model = None

        synthesizer = InitSynthesizer()
        with pytest.raises(RuntimeError, match="LLM failed"):
            await synthesizer.synthesize("data", agent=mock_agent)

    @pytest.mark.asyncio
    async def test_preflight_provider_ollama_raises_on_unreachable_server(self):
        """Unreachable local Ollama should fail fast before chat retries."""
        mock_provider = MagicMock()
        mock_provider.base_url = "http://localhost:11434"
        mock_provider.list_models = AsyncMock(side_effect=RuntimeError("connect failed"))

        synthesizer = InitSynthesizer()

        with pytest.raises(Exception, match="Ollama server unavailable"):
            await synthesizer._preflight_provider("ollama", mock_provider, "qwen3:8b")

    @pytest.mark.asyncio
    async def test_preflight_provider_ollama_picks_first_available_model(self):
        """If no model is configured, Ollama preflight should choose one."""
        mock_provider = MagicMock()
        mock_provider.base_url = "http://localhost:11434"
        mock_provider.list_models = AsyncMock(
            return_value=[{"name": "qwen3:8b"}, {"name": "gemma4:31b"}]
        )

        synthesizer = InitSynthesizer()
        result = await synthesizer._preflight_provider("ollama", mock_provider, None)

        assert result == "qwen3:8b"

    @pytest.mark.asyncio
    async def test_run_with_fresh_agent_uses_resolved_default_model(self):
        """Fresh provider path should resolve the configured default model."""
        mock_provider = MagicMock()
        mock_provider.list_models = AsyncMock(return_value=[{"name": "profile-default"}])
        mock_provider.chat = AsyncMock(return_value=MagicMock(content="# init"))
        mock_provider.close = AsyncMock()
        mock_provider.base_url = "http://localhost:11434"
        mock_create = MagicMock(return_value=mock_provider)

        mock_settings = MagicMock()
        mock_settings.default_provider = None
        mock_settings.default_model = None
        mock_settings.load_profiles.return_value = {
            "default": MagicMock(provider="ollama", model="profile-default")
        }
        mock_settings.provider = MagicMock(
            default_provider="ollama",
            default_model="local-default",
        )
        mock_settings.get_provider_settings.return_value = {}

        with patch("victor.providers.registry.ProviderRegistry.create", mock_create):
            with patch("victor.config.settings.load_settings", return_value=mock_settings):
                synthesizer = InitSynthesizer()
                result = await synthesizer._run_with_fresh_agent("prompt", "ollama", None)

        assert result == "# init"
        assert mock_provider.chat.call_args.kwargs["model"] == "profile-default"
        # Local providers (ollama) get a generous init-synthesis timeout so a
        # large local model isn't cut off mid-generation; retries stay at 0.
        mock_create.assert_called_once_with("ollama", timeout=600, max_retries=0)

    def test_resolve_provider_selection_prefers_default_profile(self):
        """Init synthesis should honor the default profile before provider defaults."""
        mock_settings = MagicMock()
        mock_settings.default_provider = None
        mock_settings.default_model = None
        mock_settings.load_profiles.return_value = {
            "default": MagicMock(provider="ollama", model="gemma4:31b")
        }
        mock_settings.provider = MagicMock(
            default_provider="ollama",
            default_model="qwen3.5:27b-q4_K_M",
        )

        with patch("victor.config.settings.load_settings", return_value=mock_settings):
            synthesizer = InitSynthesizer()
            provider, model = synthesizer._resolve_provider_selection(None, None)

        assert provider == "ollama"
        assert model == "gemma4:31b"

    def test_resolve_provider_request_routes_zai_coding_profile_to_coding_suffix(self):
        """ZAI coding profiles should preserve coding endpoint routing at provider init."""
        mock_settings = MagicMock()
        mock_settings.default_provider = None
        mock_settings.default_model = None
        mock_settings.load_profiles.return_value = {
            "zai-coding": MagicMock(provider="zai", model="glm-5.1")
        }
        mock_settings.provider = MagicMock(default_provider=None, default_model=None)

        with patch("victor.config.settings.load_settings", return_value=mock_settings):
            synthesizer = InitSynthesizer()
            provider, model, provider_init_model = synthesizer._resolve_provider_request(
                "zai-coding",
                None,
            )

        assert provider == "zai"
        assert model == "glm-5.1"
        assert provider_init_model == "glm-5.1:coding"

    def test_resolve_provider_bootstrap_uses_profile_provider_settings(self):
        """Init bootstrap should preserve profile extras through provider settings resolution."""
        profile = SimpleNamespace(
            provider="zai",
            model="glm-5.1",
            temperature=0.4,
            max_tokens=8192,
            __pydantic_extra__={"coding_plan": True},
        )
        mock_settings = MagicMock()
        mock_settings.load_profiles.return_value = {"zai-coding": profile}
        mock_settings.get_provider_settings.return_value = {
            "base_url": "https://api.z.ai/api/coding/paas/v4/",
            "coding_plan": True,
        }

        with patch("victor.config.settings.load_settings", return_value=mock_settings):
            bootstrap = InitSynthesizer._resolve_provider_bootstrap("zai-coding", None)

        assert bootstrap.provider_name == "zai"
        assert bootstrap.request_model == "glm-5.1"
        assert bootstrap.temperature == 0.4
        assert bootstrap.max_tokens == 8192
        assert bootstrap.provider_init_kwargs["base_url"].endswith("/api/coding/paas/v4/")
        assert bootstrap.provider_init_kwargs["coding_plan"] is True
        assert bootstrap.provider_init_kwargs["max_retries"] == 0
        assert "model" not in bootstrap.provider_init_kwargs

    def test_resolve_provider_bootstrap_threads_profile_account(self):
        """The profile's credential-identity ``account`` threads into provider
        settings so init resolves the profile's OWN key/endpoint (account-scoped
        resolution) rather than the provider default."""
        profile = SimpleNamespace(
            provider="anthropic",
            model="glm-5.2",
            account="kimi-k3-anthropic",
            temperature=0.4,
            max_tokens=8192,
            __pydantic_extra__={},
        )
        mock_settings = MagicMock()
        mock_settings.load_profiles.return_value = {"kimi-anthropic": profile}
        mock_settings.get_provider_settings.return_value = {"api_key": "sk-moonshot"}

        with patch("victor.config.settings.load_settings", return_value=mock_settings):
            InitSynthesizer._resolve_provider_bootstrap("kimi-anthropic", None)

        called = mock_settings.get_provider_settings.call_args
        assert called.args[0] == "anthropic"
        assert called.kwargs.get("account_name") == "kimi-k3-anthropic"

    def test_resolve_local_fallback_selection_avoids_non_ollama_default_model(self):
        """Local fallback should not reuse a remote provider's default model name."""
        mock_settings = MagicMock()
        mock_settings.default_provider = "openai"
        mock_settings.default_model = "gpt-5"
        mock_settings.load_profiles.return_value = {}
        mock_settings.provider = MagicMock(default_provider="openai", default_model="gpt-5")

        with patch("victor.config.settings.load_settings", return_value=mock_settings):
            synthesizer = InitSynthesizer()
            fallback = synthesizer._resolve_local_fallback_selection(exclude_provider="zai")

        assert fallback == ("ollama", None)

    @pytest.mark.asyncio
    async def test_reused_provider_rate_limit_falls_back_to_local_ollama(self):
        """Rate-limited remote init synthesis should retry once on local Ollama."""
        from victor.core.errors import ProviderRateLimitError

        mock_provider = MagicMock()
        mock_provider.name = "zai"
        mock_provider.chat = AsyncMock(
            side_effect=ProviderRateLimitError("Insufficient balance", provider="zai")
        )

        synthesizer = InitSynthesizer()
        with patch.object(
            InitSynthesizer,
            "_resolve_local_fallback_selection",
            return_value=("ollama", "qwen3:8b"),
        ):
            with patch.object(
                InitSynthesizer,
                "_run_with_fresh_agent",
                AsyncMock(return_value="# local init"),
            ) as mock_fallback:
                result = await synthesizer._call_initialized_provider(
                    "prompt",
                    mock_provider,
                    "glm-5.1",
                )

        assert result == "# local init"
        mock_fallback.assert_awaited_once_with(
            "prompt",
            "ollama",
            "qwen3:8b",
            allow_local_fallback=False,
        )


class TestInitSynthesizerToolsFallback:
    """synthesize_with_tools() fallback mode."""

    @pytest.mark.asyncio
    async def test_tools_fallback_with_agent(self):
        """Fallback uses agent.chat() with tools prompt."""
        # Use a real ``async def chat`` rather than AsyncMock: on Python
        # 3.10/3.11 ``inspect.iscoroutinefunction(AsyncMock())`` returns False
        # (AsyncMock coroutine detection was only fixed in 3.12), so
        # synthesize_with_tools took its "agent exposes no async chat/run"
        # branch and returned "" — a version-dependent CI failure. A real
        # coroutine function is detected consistently on all versions.
        captured = {}

        class _FakeAgent:
            async def chat(self, prompt, **kwargs):
                captured["prompt"] = prompt
                return MagicMock(content="# Fallback init.md")

        synthesizer = InitSynthesizer()
        result = await synthesizer.synthesize_with_tools(agent=_FakeAgent())

        assert result == "# Fallback init.md"
        assert "overview" in captured["prompt"].lower() or "analyze" in captured["prompt"].lower()

    @pytest.mark.asyncio
    async def test_tools_fallback_without_agent(self):
        """Fallback creates Agent with vertical=coding for tool access."""
        mock_result = MagicMock(success=True, content="# Tools init.md")

        with patch("victor.framework.agent.Agent") as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.create = AsyncMock(return_value=mock_agent_instance)

            synthesizer = InitSynthesizer()
            result = await synthesizer.synthesize_with_tools(provider="ollama", model="qwen3:8b")

            # Should create with vertical="coding" for tool access
            create_kwargs = MockAgent.create.call_args[1]
            assert create_kwargs.get("vertical") == "coding"
            assert result == "# Tools init.md"


class TestInitSynthesizerEvolvableSection:
    """INIT_SYNTHESIS_RULES is registered as an evolvable section."""

    def test_init_synthesis_rules_in_evolvable_sections(self):
        """INIT_SYNTHESIS_RULES must be in PromptOptimizerLearner.EVOLVABLE_SECTIONS."""
        from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner

        assert "INIT_SYNTHESIS_RULES" in PromptOptimizerLearner.EVOLVABLE_SECTIONS

    def test_init_section_uses_gepa_only(self):
        """INIT_SYNTHESIS_RULES uses GEPA only — MIPROv2/CoT irrelevant for single-shot."""
        from victor.framework.rl.learners.prompt_optimizer import (
            PromptOptimizerLearner,
        )

        mock_db = MagicMock()
        mock_db.execute.return_value = MagicMock(fetchall=MagicMock(return_value=[]))
        mock_db.executescript = MagicMock()
        mock_db.commit = MagicMock()

        learner = PromptOptimizerLearner("test", mock_db)
        strategies = learner._extra_strategies.get("INIT_SYNTHESIS_RULES", [])

        # Single-shot: GEPA only (MIPROv2/CoT add tool patterns, not useful)
        assert len(strategies) == 1
        assert type(strategies[0]).__name__ == "GEPAStrategy"

    def test_build_synthesis_prompt_preserves_base_content(self):
        """Frame always keeps {base_content} safe regardless of rules content."""
        custom_rules = "RULES:\n- Custom rule 1\n- Custom rule 2"
        prompt = _build_synthesis_prompt("MY_RAW_DATA", rules=custom_rules)
        assert "MY_RAW_DATA" in prompt
        assert "Custom rule 1" in prompt
        assert "init.md" in prompt  # Frame intro preserved

    def test_synthesis_prompt_backward_compat(self):
        """SYNTHESIS_PROMPT constant still works with .format(base_content=...)."""
        result = SYNTHESIS_PROMPT.format(base_content="test data")
        assert "test data" in result
        assert "RULES:" in result

    def test_static_rules_keep_tool_targeting_guidance(self):
        """Promoted baseline rules should keep the useful GEPA hardening."""
        assert "Verify tool argument types and output structure" in SYNTHESIS_RULES
        assert "targeted graph/code_search queries" in SYNTHESIS_RULES
        assert "pagination or incremental reads" in SYNTHESIS_RULES

    def test_prompt_optimization_identity_prefers_active_agent_provider_and_model(self):
        synthesizer = InitSynthesizer()
        provider = SimpleNamespace(name="zai")
        agent = SimpleNamespace(provider=provider, model="glm-5.1")

        resolved_provider, resolved_model = synthesizer._resolve_prompt_optimization_identity(
            agent=agent,
            provider="zai-coding",
            model=None,
        )

        assert resolved_provider == "zai"
        assert resolved_model == "glm-5.1"


class TestInitSynthesizerGEPAWiring:
    """Tests for GEPA-evolved rules retrieval and usage."""

    @pytest.mark.asyncio
    async def test_uses_evolved_rules_when_available(self):
        """When GEPA has evolved rules, synthesize() injects them into frame."""
        evolved_rules = "RULES:\n- Improved rule 1\n- Improved rule 2"

        mock_provider = AsyncMock()
        mock_provider.name = "test-provider"
        mock_provider.chat.return_value = MagicMock(content="# Evolved result")
        mock_agent = MagicMock()
        mock_agent.provider = mock_provider
        mock_agent.model = None

        synthesizer = InitSynthesizer()
        with patch.object(InitSynthesizer, "_get_evolved_rules", return_value=evolved_rules):
            result = await synthesizer.synthesize("raw data", agent=mock_agent)

        call_kwargs = mock_provider.chat.call_args
        messages = call_kwargs[1].get("messages") or call_kwargs[0][0]
        prompt_sent = str(messages)
        assert "Improved rule 1" in prompt_sent
        assert "raw data" in prompt_sent  # base_content always present (frame)
        assert result == "# Evolved result"

    @pytest.mark.asyncio
    async def test_falls_back_to_static_when_no_evolution(self):
        """When GEPA has no evolved rules, uses static SYNTHESIS_RULES."""
        mock_provider = AsyncMock()
        mock_provider.name = "test-provider"
        mock_provider.chat.return_value = MagicMock(content="# Static result")
        mock_agent = MagicMock()
        mock_agent.provider = mock_provider
        mock_agent.model = None

        synthesizer = InitSynthesizer()
        with patch.object(InitSynthesizer, "_get_evolved_rules", return_value=None):
            result = await synthesizer.synthesize("raw data", agent=mock_agent)

        call_kwargs = mock_provider.chat.call_args
        messages = call_kwargs[1].get("messages") or call_kwargs[0][0]
        prompt_sent = str(messages)
        assert "RULES:" in prompt_sent
        assert "raw data" in prompt_sent
        assert result == "# Static result"

    @pytest.mark.asyncio
    async def test_evolved_rules_always_preserve_base_content(self):
        """Even with evolved rules, {base_content} is always in the prompt."""
        evolved_rules = "RULES:\n- Any content here"

        mock_provider = AsyncMock()
        mock_provider.name = "test-provider"
        mock_provider.chat.return_value = MagicMock(content="# Result")
        mock_agent = MagicMock()
        mock_agent.provider = mock_provider
        mock_agent.model = None

        synthesizer = InitSynthesizer()
        with patch.object(InitSynthesizer, "_get_evolved_rules", return_value=evolved_rules):
            await synthesizer.synthesize("UNIQUE_DATA_XYZ", agent=mock_agent)

        call_kwargs = mock_provider.chat.call_args
        messages = call_kwargs[1].get("messages") or call_kwargs[0][0]
        prompt_sent = str(messages)
        assert "UNIQUE_DATA_XYZ" in prompt_sent  # Frame always includes data

    def test_get_evolved_rules_gates_on_settings(self):
        """_get_evolved_rules returns None when optimization disabled."""
        mock_settings = MagicMock()
        mock_settings.prompt_optimization.enabled = False

        with patch("victor.config.settings.get_settings", return_value=mock_settings):
            result = InitSynthesizer._get_evolved_rules("ollama")

        assert result is None

    def test_get_evolved_rules_gates_on_section_strategies(self):
        """_get_evolved_rules returns None when section strategies are empty."""
        mock_po = MagicMock()
        mock_po.enabled = True
        mock_po.get_strategies_for_section.return_value = []

        mock_settings = MagicMock()
        mock_settings.prompt_optimization = mock_po

        with patch("victor.config.settings.get_settings", return_value=mock_settings):
            result = InitSynthesizer._get_evolved_rules("ollama")

        assert result is None


class TestInitQualityScoring:
    """Tests for init output quality signal logging."""

    def test_quality_score_perfect_output(self):
        """Perfect output with all sections scores high."""
        mock_usage = MagicMock()
        result = (
            "# Project Overview\nA project.\n"
            "# System Flow\nUser → Core → Output.\n"
            "# Package Layout\n| Path | Description |\n"
            "# Key Entry Points\n| Component | Path |\n"
            "# Architecture Patterns\n- Pattern A.\n"
            "# Development Commands\n```bash\nmake test\n```\n"
            "# Dependencies\nCore: pydantic.\n"
            "# Configuration\nSettings via .env.\n"
            "# Codebase Scale\n100K lines.\n"
        )
        # Pad to ~80 lines
        result += "\n" * 70

        InitSynthesizer._log_init_quality(mock_usage, result)

        call_data = mock_usage.log_event.call_args[0][1]
        assert call_data["sections_found"] == 9
        assert call_data["sections_total"] == 9
        assert call_data["section_score"] == 1.0
        assert 60 <= call_data["line_count"] <= 100
        assert call_data["length_score"] == 1.0
        assert call_data["quality_score"] == pytest.approx(1.0, abs=0.01)

    def test_quality_score_missing_sections(self):
        """Output missing sections scores lower."""
        mock_usage = MagicMock()
        # Only 2 of 9 sections
        result = "# Project Overview\nA project.\n# Dependencies\nCore: pydantic.\n"

        InitSynthesizer._log_init_quality(mock_usage, result)

        call_data = mock_usage.log_event.call_args[0][1]
        assert call_data["sections_found"] == 2
        assert call_data["section_score"] == pytest.approx(2 / 9, abs=0.01)
        assert call_data["quality_score"] < 0.5

    def test_quality_score_empty_output(self):
        """Empty output gets minimal quality score."""
        mock_usage = MagicMock()
        InitSynthesizer._log_init_quality(mock_usage, "")

        call_data = mock_usage.log_event.call_args[0][1]
        assert call_data["sections_found"] == 0
        assert call_data["quality_score"] < 0.3
