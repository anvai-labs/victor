"""Test legacy environment variable mapping."""


class TestLegacyEnvironmentVariables:
    """Test that legacy flat environment variables work correctly."""

    def test_legacy_flat_env_vars(self, monkeypatch):
        """Test that old-style flat env vars are mapped to nested structure."""
        # Set old-style env vars
        monkeypatch.setenv("VICTOR_DEFAULT_PROVIDER", "anthropic")
        monkeypatch.setenv("VICTOR_DEFAULT_MODEL", "claude-sonnet-4")
        monkeypatch.setenv("VICTOR_DEFAULT_TEMPERATURE", "0.5")

        from victor.config.settings import Settings

        settings = Settings()

        # Verify legacy env vars were mapped to nested structure
        assert settings.provider.default_provider == "anthropic"
        assert settings.provider.default_model == "claude-sonnet-4"
        assert settings.provider.default_temperature == 0.5

    def test_new_nested_env_vars(self, monkeypatch):
        """Test that new-style nested env vars work."""
        # Set new-style env vars
        monkeypatch.setenv("VICTOR_PROVIDER__DEFAULT_PROVIDER", "ollama")
        monkeypatch.setenv("VICTOR_PROVIDER__DEFAULT_MODEL", "qwen3-coder:30b")
        monkeypatch.setenv("VICTOR_PROVIDER__DEFAULT_TEMPERATURE", "0.8")

        from victor.config.settings import Settings

        settings = Settings()

        # Verify new env vars work
        assert settings.provider.default_provider == "ollama"
        assert settings.provider.default_model == "qwen3-coder:30b"
        assert settings.provider.default_temperature == 0.8

    def test_precedence_new_over_old(self, monkeypatch):
        """Test that new-style env vars take precedence over old-style."""
        # Set both old and new style
        monkeypatch.setenv("VICTOR_DEFAULT_PROVIDER", "ollama")  # Old
        monkeypatch.setenv("VICTOR_PROVIDER__DEFAULT_PROVIDER", "anthropic")  # New

        from victor.config.settings import Settings

        settings = Settings()

        # New style should win
        assert settings.provider.default_provider == "anthropic"

    def test_tool_selection_legacy_env_vars(self, monkeypatch):
        """Test legacy env vars for tool selection config."""
        monkeypatch.setenv("VICTOR_USE_SEMANTIC_TOOL_SELECTION", "false")
        monkeypatch.setenv("VICTOR_FALLBACK_MAX_TOOLS", "15")

        from victor.config.settings import Settings

        settings = Settings()

        assert settings.tool_selection.use_semantic_tool_selection is False
        assert settings.tool_selection.fallback_max_tools == 15

    def test_embedding_legacy_env_vars(self, monkeypatch):
        """Test legacy env vars for embedding config."""
        monkeypatch.setenv("VICTOR_UNIFIED_EMBEDDING_MODEL", "custom-model")
        monkeypatch.setenv("VICTOR_EMBEDDING_PROVIDER", "ollama")

        from victor.config.settings import Settings

        settings = Settings()

        assert settings.embedding.unified_embedding_model == "custom-model"
        assert settings.embedding.embedding_provider == "ollama"

    def test_analytics_legacy_env_vars(self, monkeypatch):
        """Test legacy env vars for analytics config."""
        monkeypatch.setenv("VICTOR_ANALYTICS_ENABLED", "true")
        monkeypatch.setenv("VICTOR_SHOW_TOKEN_COUNT", "true")

        from victor.config.settings import Settings

        settings = Settings()

        assert settings.analytics.analytics_enabled is True
        assert settings.analytics.show_token_count is True

    def test_removed_service_rollout_env_vars_no_longer_map(self, monkeypatch):
        """Removed service-rollout env vars should not populate nested settings."""
        monkeypatch.setenv("VICTOR_USE_NEW_CHAT_SERVICE", "true")

        from victor.config.settings import Settings

        settings = Settings()

        assert not hasattr(settings.feature_flags, "use_new_chat_service")

    def test_legacy_runtime_intelligence_env_var_maps_to_canonical_pipeline_field(
        self, monkeypatch
    ):
        """Legacy runtime gate env var should populate the canonical nested field."""
        monkeypatch.setenv("VICTOR_INTELLIGENT_PIPELINE_ENABLED", "false")

        from victor.config.settings import Settings

        settings = Settings()

        assert settings.pipeline.runtime_intelligence_enabled is False
