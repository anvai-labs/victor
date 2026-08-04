class TestFromProfileEndpointFallback:
    """profiles.yaml uses `endpoint:`; from_profile must honor it as base_url.

    Regression guard for the 96/96 empty-response calibration run: the profile
    endpoint silently dropped and the provider called its localhost default.
    """

    def test_endpoint_extra_field_reaches_provider(self, monkeypatch, tmp_path):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        from victor.evaluation.agent_adapter import VictorAgentAdapter

        profile = SimpleNamespace(
            provider="ollama",
            model="test-model",
            temperature=0.6,
            max_tokens=1024,
            endpoint="http://gpu-host:11434",
            api_key=None,
        )
        settings = MagicMock()
        settings.load_profiles.return_value = {"default": profile}

        captured = {}

        def fake_create(name, **kwargs):
            captured.update(kwargs, provider=name)
            return MagicMock()

        with (
            patch("victor.evaluation.agent_adapter.load_settings", return_value=settings),
            patch("victor.providers.registry.ProviderRegistry.create", side_effect=fake_create),
            patch("victor.agent.orchestrator.AgentOrchestrator", return_value=MagicMock()),
        ):
            VictorAgentAdapter.from_profile("default")

        assert captured.get("base_url") == "http://gpu-host:11434"

    def test_explicit_base_url_still_wins(self, monkeypatch):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        from victor.evaluation.agent_adapter import VictorAgentAdapter

        profile = SimpleNamespace(
            provider="ollama",
            model="test-model",
            temperature=0.6,
            max_tokens=1024,
            endpoint="http://gpu-host:11434",
            api_key=None,
        )
        settings = MagicMock()
        settings.load_profiles.return_value = {"default": profile}
        captured = {}

        def fake_create(name, **kwargs):
            captured.update(kwargs)
            return MagicMock()

        with (
            patch("victor.evaluation.agent_adapter.load_settings", return_value=settings),
            patch("victor.providers.registry.ProviderRegistry.create", side_effect=fake_create),
            patch("victor.agent.orchestrator.AgentOrchestrator", return_value=MagicMock()),
        ):
            VictorAgentAdapter.from_profile("default", base_url="http://override:11434")

        assert captured.get("base_url") == "http://override:11434"
