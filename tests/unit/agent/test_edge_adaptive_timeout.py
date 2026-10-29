"""TDD tests for edge model adaptive timeouts and cache optimization."""

from victor.agent.edge_model import EdgeModelConfig


class TestAdaptiveTimeout:
    """EdgeModelConfig should support per-task-type timeouts."""

    def test_config_has_timeout_by_task(self):
        config = EdgeModelConfig()
        assert hasattr(config, "timeout_by_task")
        assert isinstance(config.timeout_by_task, dict)

    def test_classification_timeout_shorter(self):
        config = EdgeModelConfig()
        assert config.timeout_by_task.get("classification", 4000) <= 3000

    def test_tool_selection_timeout_longer(self):
        config = EdgeModelConfig()
        assert config.timeout_by_task.get("tool_selection", 4000) >= 5000

    def test_default_timeout_preserved(self):
        config = EdgeModelConfig()
        assert config.timeout_ms == 4000

    def test_get_timeout_for_task(self):
        config = EdgeModelConfig()
        assert config.get_timeout_for_task("classification") <= 3000
        assert config.get_timeout_for_task("tool_selection") >= 5000
        assert config.get_timeout_for_task("unknown_task") == config.timeout_ms


class TestCacheTTLByTask:
    """Classification decisions should have longer cache TTL."""

    def test_config_has_cache_ttl_by_task(self):
        config = EdgeModelConfig()
        assert hasattr(config, "cache_ttl_by_task")

    def test_classification_cache_longer(self):
        config = EdgeModelConfig()
        assert config.cache_ttl_by_task.get("classification", 120) >= 300

    def test_tool_selection_cache_default(self):
        config = EdgeModelConfig()
        assert config.cache_ttl_by_task.get("tool_selection", 120) == 120
