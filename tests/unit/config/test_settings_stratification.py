"""Tests for Settings nested config group access.

Verifies that:
1. Nested group access works (settings.provider.default_provider)
2. Environment variables map correctly to nested groups
3. Each nested model has correct defaults independently
4. Direct initialization maps flat parameters to nested groups
"""

import pytest

from victor.config.settings import (
    EventSettings,
    PipelineSettings,
    ProviderSettings,
    ResilienceSettings,
    SearchSettings,
    SecuritySettings,
    Settings,
    ToolSettings,
)


class TestNestedAccess:
    """Nested config group access works correctly."""

    def test_nested_provider_access(self):
        s = Settings(default_provider="anthropic")
        assert s.provider.default_provider == "anthropic"

    def test_nested_provider_model(self):
        s = Settings(default_model="gpt-4")
        assert s.provider.default_model == "gpt-4"

    def test_nested_provider_api_key(self):
        s = Settings(anthropic_api_key="sk-test")
        assert s.provider.anthropic_api_key.get_secret_value() == "sk-test"

    def test_nested_tools_access(self):
        s = Settings(tool_retry_max_attempts=5)
        assert s.tools.tool_retry_max_attempts == 5

    def test_nested_tools_deduplication(self):
        s = Settings(enable_tool_deduplication=False)
        assert s.tool_selection.enable_tool_deduplication is False

    def test_nested_search_access(self):
        s = Settings(codebase_dimension=768)
        assert s.search.codebase_dimension == 768

    def test_nested_search_extra_config_access(self):
        s = Settings(codebase_embedding_extra_config={"table_name": "custom"})
        assert s.search.codebase_embedding_extra_config == {"table_name": "custom"}

    def test_nested_search_chunking_access(self):
        s = Settings(codebase_chunking_strategy="symbol_span", codebase_chunk_size=256)
        assert s.search.codebase_chunking_strategy == "symbol_span"
        assert s.search.codebase_chunk_size == 256

    def test_nested_resilience_access(self):
        s = Settings(circuit_breaker_timeout=120.0)
        assert s.resilience.circuit_breaker_timeout == 120.0

    def test_nested_security_access(self):
        s = Settings(write_approval_mode="all_writes")
        assert s.security.write_approval_mode == "all_writes"

    def test_nested_security_secret_access(self):
        s = Settings(server_api_key="server-token", server_session_secret="session-secret")
        assert s.server.server_api_key.get_secret_value() == "server-token"
        assert s.server.server_session_secret.get_secret_value() == "session-secret"

    def test_nested_events_access(self):
        s = Settings(event_queue_maxsize=5000)
        assert s.events.event_queue_maxsize == 5000

    def test_nested_pipeline_access(self):
        s = Settings(max_exploration_iterations=100)
        assert s.pipeline.max_exploration_iterations == 100

    def test_nested_pipeline_runtime_intelligence_access(self):
        s = Settings(runtime_intelligence_enabled=False)
        assert s.pipeline.runtime_intelligence_enabled is False

    def test_nested_pipeline_legacy_runtime_intelligence_alias(self):
        s = Settings(pipeline={"intelligent_pipeline_enabled": False})
        assert s.pipeline.runtime_intelligence_enabled is False


class TestNestedModelDefaults:
    """Nested config groups work independently with correct defaults."""

    def test_provider_settings_defaults(self):
        ps = ProviderSettings()
        assert ps.default_provider == "ollama"
        assert ps.default_model == "qwen3.5:27b-q4_K_M"
        assert ps.default_temperature == 0.6  # ADR-013 default flip
        assert ps.default_max_tokens == 4096
        assert ps.anthropic_api_key is None
        assert ps.ollama_base_url == "http://localhost:11434"
        assert ps.lmstudio_max_vram_gb == 48.0

    def test_tool_settings_defaults(self):
        ts = ToolSettings()
        assert ts.tool_retry_enabled is True
        assert ts.tool_retry_max_attempts == 3
        assert ts.fallback_max_tools == 8
        assert ts.use_semantic_tool_selection is True
        assert ts.embedding_provider == "sentence-transformers"
        assert ts.tool_validation_mode == "lenient"

    def test_search_settings_defaults(self):
        ss = SearchSettings()
        assert ss.codebase_vector_store == "lancedb"
        assert ss.codebase_dimension == 384
        assert ss.codebase_structural_indexing_enabled is False
        assert ss.codebase_chunking_strategy == "victor_codegraph"
        assert ss.codebase_chunk_size == 500
        assert ss.codebase_chunk_overlap == 50
        assert ss.codebase_embedding_extra_config == {}
        assert ss.codebase_graph_writer_mode == "off"
        assert ss.semantic_similarity_threshold == 0.25
        assert ss.enable_hybrid_search is False

    def test_resilience_settings_defaults(self):
        rs = ResilienceSettings()
        assert rs.resilience_enabled is True
        assert rs.circuit_breaker_failure_threshold == 5
        assert rs.retry_max_attempts == 3
        assert rs.rate_limiting_enabled is True
        assert rs.rate_limit_requests_per_minute == 50

    def test_security_settings_defaults(self):
        ss = SecuritySettings()
        assert ss.airgapped_mode is False
        assert ss.write_approval_mode == "risky_only"
        assert ss.code_executor_network_disabled is True
        assert ss.headless_mode is False

    def test_event_settings_defaults(self):
        es = EventSettings()
        assert es.event_backend_type == "in_memory"
        assert es.event_delivery_guarantee == "at_most_once"
        assert es.event_queue_maxsize == 10000
        assert es.event_queue_overflow_policy == "drop_newest"

    def test_pipeline_settings_defaults(self):
        ps = PipelineSettings()
        assert ps.runtime_intelligence_enabled is True
        assert ps.max_exploration_iterations == 200
        assert ps.recovery_empty_response_threshold == 5
        assert ps.session_idle_timeout == 180


class TestDeprecatedFieldsRemoved:
    """Deprecated eventbus_* fields are removed."""

    def test_eventbus_fields_not_on_settings(self):
        """Verify deprecated eventbus_* fields are no longer defined on Settings."""
        settings_fields = set(Settings.model_fields.keys())
        deprecated_fields = {
            "eventbus_backend",
            "eventbus_queue_maxsize",
            "eventbus_backpressure_strategy",
            "eventbus_sampling_enabled",
            "eventbus_sampling_default_rate",
            "eventbus_batching_enabled",
            "eventbus_batch_size",
            "eventbus_batch_flush_interval_ms",
        }
        overlap = settings_fields & deprecated_fields
        assert not overlap, f"Deprecated fields still on Settings: {overlap}"


class TestAllNestedGroupsPopulated:
    """All 7 nested groups are populated after construction."""

    def test_all_groups_exist(self):
        s = Settings()
        assert isinstance(s.provider, ProviderSettings)
        assert isinstance(s.tools, ToolSettings)
        assert isinstance(s.search, SearchSettings)
        assert isinstance(s.resilience, ResilienceSettings)
        assert isinstance(s.security, SecuritySettings)
        assert isinstance(s.events, EventSettings)
        assert isinstance(s.pipeline, PipelineSettings)
