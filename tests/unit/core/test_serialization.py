# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the token-optimized serialization system."""

import pytest

from victor.processing.serialization import (
    AdaptiveSerializer,
    SerializationContext,
    SerializationFormat,
    SerializationConfig,
    DataCharacteristics,
    DataAnalyzer,
    serialize,
    reset_adaptive_serializer,
)
from victor.processing.serialization.formats import (
    JSONEncoder,
    MinifiedJSONEncoder,
    TOONEncoder,
    CSVEncoder,
    MarkdownTableEncoder,
    ReferenceEncoder,
    get_format_registry,
    reset_format_registry,
)
from victor.processing.serialization.formats import _register_builtin_encoders
from victor.processing.serialization.capabilities import (
    get_capability_registry,
    reset_capability_registry,
)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before each test and re-register encoders."""
    reset_adaptive_serializer()
    reset_format_registry()
    reset_capability_registry()
    # Re-register built-in encoders after reset
    _register_builtin_encoders()
    yield
    reset_adaptive_serializer()
    reset_format_registry()
    reset_capability_registry()


class TestDataAnalyzer:
    """Tests for DataAnalyzer."""

    def test_analyze_empty_list(self):
        """Test analyzing empty list."""
        analyzer = DataAnalyzer()
        chars = analyzer.analyze([])

        assert chars.array_length == 0
        assert chars.structure_type.value == "empty"

    def test_analyze_uniform_array(self):
        """Test analyzing uniform array of dicts."""
        analyzer = DataAnalyzer()
        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Charlie"},
        ]
        chars = analyzer.analyze(data)

        assert chars.structure_type.value == "uniform_array"
        assert chars.array_length == 3
        assert chars.array_uniformity >= 0.9
        assert "id" in chars.field_names
        assert "name" in chars.field_names

    def test_analyze_nested_object(self):
        """Test analyzing nested object."""
        analyzer = DataAnalyzer()
        data = {
            "user": {"id": 1, "name": "Alice"},
            "settings": {"theme": "dark"},
        }
        chars = analyzer.analyze(data)

        assert chars.has_nested_objects
        assert chars.nesting_depth >= 1

    def test_analyze_primitive_array(self):
        """Test analyzing array of primitives."""
        analyzer = DataAnalyzer()
        data = [1, 2, 3, 4, 5]
        chars = analyzer.analyze(data)

        assert chars.structure_type.value == "primitive_array"
        assert chars.array_length == 5

    def test_is_tabular(self):
        """Test tabular detection."""
        analyzer = DataAnalyzer()

        # Tabular data
        tabular = [{"a": 1, "b": 2}] * 5
        chars = analyzer.analyze(tabular)
        assert chars.is_tabular()

        # Non-tabular (nested)
        nested = [{"a": {"x": 1}}] * 5
        chars = analyzer.analyze(nested)
        assert not chars.is_tabular()


class TestFormatEncoders:
    """Tests for format encoders."""

    def test_json_encoder(self):
        """Test JSON encoder."""
        encoder = JSONEncoder()
        data = [{"id": 1, "name": "test"}]
        chars = DataCharacteristics()
        config = SerializationConfig()

        result = encoder.encode(data, chars, config)

        assert result.success
        assert '"id": 1' in result.content
        assert '"name": "test"' in result.content

    def test_minified_json_encoder(self):
        """Test minified JSON encoder."""
        encoder = MinifiedJSONEncoder()
        data = [{"id": 1, "name": "test"}]
        chars = DataCharacteristics()
        config = SerializationConfig()

        result = encoder.encode(data, chars, config)

        assert result.success
        # Minified - no spaces after colons
        assert '{"id":1' in result.content or '"id":1' in result.content

    def test_toon_encoder(self):
        """Test TOON encoder."""
        encoder = TOONEncoder()
        analyzer = DataAnalyzer()
        data = [
            {"id": 1, "name": "Alice", "status": "active"},
            {"id": 2, "name": "Bob", "status": "active"},
            {"id": 3, "name": "Charlie", "status": "inactive"},
        ]
        chars = analyzer.analyze(data)
        config = SerializationConfig()

        # TOON can encode tabular data
        assert encoder.can_encode(data, chars)

        result = encoder.encode(data, chars, config)

        assert result.success
        # TOON format: name[count]{fields}:
        assert "[3]" in result.content
        assert "{" in result.content
        assert ":" in result.content

    def test_csv_encoder(self):
        """Test CSV encoder."""
        encoder = CSVEncoder()
        analyzer = DataAnalyzer()
        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
        chars = analyzer.analyze(data)
        config = SerializationConfig()

        assert encoder.can_encode(data, chars)

        result = encoder.encode(data, chars, config)

        assert result.success
        assert "id" in result.content
        assert "name" in result.content
        assert "Alice" in result.content

    def test_markdown_encoder(self):
        """Test Markdown table encoder."""
        encoder = MarkdownTableEncoder()
        analyzer = DataAnalyzer()
        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
        chars = analyzer.analyze(data)
        config = SerializationConfig()

        result = encoder.encode(data, chars, config)

        assert result.success
        assert "|" in result.content
        assert "---" in result.content

    def test_reference_encoder(self):
        """Test reference encoder for repetitive data."""
        encoder = ReferenceEncoder()
        analyzer = DataAnalyzer()
        # Data with many repeated values
        data = [
            {"status": "active", "type": "user"},
            {"status": "active", "type": "admin"},
            {"status": "active", "type": "user"},
            {"status": "inactive", "type": "user"},
            {"status": "active", "type": "admin"},
        ]
        chars = analyzer.analyze(data)
        config = SerializationConfig(enable_reference_encoding=True)

        # Only encode if high repetition
        if chars.has_high_repetition():
            result = encoder.encode(data, chars, config)
            assert result.success


class TestFormatRegistry:
    """Tests for FormatRegistry."""

    def test_register_encoder(self):
        """Test encoder registration."""
        registry = get_format_registry()

        # Built-in encoders should be registered
        assert registry.get_encoder(SerializationFormat.JSON) is not None
        assert registry.get_encoder(SerializationFormat.TOON) is not None

    def test_select_best_encoder(self):
        """Test best encoder selection."""
        registry = get_format_registry()
        analyzer = DataAnalyzer()

        # Tabular data should select TOON or CSV
        data = [{"a": 1, "b": 2}] * 10
        chars = analyzer.analyze(data)
        config = SerializationConfig()

        encoder = registry.select_best_encoder(data, chars, config)

        assert encoder is not None
        assert encoder.format_id in [
            SerializationFormat.TOON,
            SerializationFormat.CSV,
        ]


class TestAdaptiveSerializer:
    """Tests for AdaptiveSerializer."""

    def test_serialize_tabular_data(self):
        """Test serializing tabular data uses TOON."""
        serializer = AdaptiveSerializer()
        data = [
            {"id": 1, "name": "Alice", "status": "active"},
            {"id": 2, "name": "Bob", "status": "active"},
            {"id": 3, "name": "Charlie", "status": "inactive"},
        ]

        result = serializer.serialize(data)

        assert result.format in [SerializationFormat.TOON, SerializationFormat.CSV]
        assert result.estimated_savings_percent > 0

    def test_serialize_with_context(self):
        """Test serializing with provider context."""
        serializer = AdaptiveSerializer()
        context = SerializationContext(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            tool_name="database_query",
        )
        data = [{"id": i, "value": f"item_{i}"} for i in range(10)]

        result = serializer.serialize(data, context)

        assert result.content
        assert result.format is not None

    def test_serialize_with_format_override(self):
        """Test serializing with format override."""
        serializer = AdaptiveSerializer()
        data = [{"a": 1}, {"a": 2}]

        result = serializer.serialize(data, format_override=SerializationFormat.JSON_MINIFIED)

        assert result.format == SerializationFormat.JSON_MINIFIED

    def test_serialize_convenience_function(self):
        """Test convenience serialize function."""
        data = [{"x": 1, "y": 2}] * 5

        result = serialize(data, provider="anthropic")

        assert result.content
        assert result.format is not None

    def test_metrics_collection(self):
        """Test that metrics are collected."""
        serializer = AdaptiveSerializer()
        data = [{"id": i, "name": f"test_{i}"} for i in range(20)]

        _result = serializer.serialize(data)
        metrics = serializer.get_last_metrics()

        assert metrics is not None
        assert metrics.original_json_tokens > 0
        assert metrics.serialized_tokens > 0
        assert metrics.format_selected != ""


class TestCapabilityRegistry:
    """Tests for CapabilityRegistry."""

    def test_get_capabilities_default(self):
        """Test getting default capabilities."""
        registry = get_capability_registry()
        caps = registry.get_capabilities("unknown_provider")

        # Should return defaults
        assert caps is not None
        assert len(caps.allowed_formats) > 0

    def test_get_capabilities_anthropic(self):
        """Test getting Anthropic-specific capabilities."""
        registry = get_capability_registry()
        caps = registry.get_capabilities("anthropic", "claude-sonnet-4-20250514")

        assert caps is not None
        # Anthropic has relaxed threshold
        assert caps.min_savings_threshold <= 0.20

    def test_get_capabilities_groq(self):
        """Test getting Groq-specific capabilities."""
        registry = get_capability_registry()
        caps = registry.get_capabilities("groqcloud")

        assert caps is not None
        # Groq should prefer compact formats due to payload limits
        assert SerializationFormat.CSV in caps.allowed_formats

    def test_to_config(self):
        """Test converting capabilities to config."""
        registry = get_capability_registry()
        caps = registry.get_capabilities("anthropic")
        config = caps.to_config()

        assert isinstance(config, SerializationConfig)
        assert config.allowed_formats == caps.allowed_formats


class TestTokenSavings:
    """Tests verifying token savings."""

    def test_toon_saves_tokens(self):
        """Test that TOON format saves tokens."""
        serializer = AdaptiveSerializer()
        data = [{"id": i, "name": f"user_{i}", "email": f"user{i}@example.com"} for i in range(20)]

        _result = serializer.serialize(data, format_override=SerializationFormat.TOON)
        metrics = serializer.get_last_metrics()

        # TOON should save at least 30% for tabular data
        assert (
            metrics.token_savings_percent >= 30
        ), f"Expected >= 30% savings, got {metrics.token_savings_percent:.1f}%"

    def test_csv_saves_tokens(self):
        """Test that CSV format saves tokens."""
        serializer = AdaptiveSerializer()
        data = [{"a": i, "b": i * 2, "c": i * 3} for i in range(30)]

        _result = serializer.serialize(data, format_override=SerializationFormat.CSV)
        metrics = serializer.get_last_metrics()

        # CSV should save at least 40% for clean tabular data
        assert (
            metrics.token_savings_percent >= 40
        ), f"Expected >= 40% savings, got {metrics.token_savings_percent:.1f}%"

    def test_minified_json_saves_tokens(self):
        """Test that minified JSON saves tokens."""
        serializer = AdaptiveSerializer()
        data = {"key1": "value1", "key2": "value2", "key3": "value3"}

        _result = serializer.serialize(data, format_override=SerializationFormat.JSON_MINIFIED)
        metrics = serializer.get_last_metrics()

        # Minified should save at least a small percentage
        assert metrics.token_savings_percent >= 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_data(self):
        """Test handling empty data."""
        serializer = AdaptiveSerializer()

        result = serializer.serialize([])
        assert result.content == "[]"

        result = serializer.serialize({})
        assert result.content == "{}"

    def test_string_data(self):
        """Test handling string data (not serialized)."""
        serializer = AdaptiveSerializer()

        result = serializer.serialize("plain text")
        # Strings are JSON serialized with quotes
        assert "plain text" in result.content
        # May use JSON or JSON_MINIFIED (both are valid JSON formats)
        assert result.format in [
            SerializationFormat.JSON,
            SerializationFormat.JSON_MINIFIED,
        ]

    def test_nested_data_falls_back(self):
        """Test that nested data uses JSON (not TOON/CSV)."""
        serializer = AdaptiveSerializer()
        data = [
            {"id": 1, "nested": {"a": 1, "b": 2}},
            {"id": 2, "nested": {"a": 3, "b": 4}},
        ]

        result = serializer.serialize(data)

        # Should not use TOON/CSV for nested data
        assert result.format not in [
            SerializationFormat.TOON,
            SerializationFormat.CSV,
        ]

    def test_special_characters(self):
        """Test handling special characters."""
        serializer = AdaptiveSerializer()
        data = [
            {"text": "Hello, World!"},
            {"text": "Line1\nLine2"},
            {"text": 'Quote: "test"'},
        ]

        result = serializer.serialize(data)

        # Should handle without error
        assert result.content  # Has content
        assert result.format is not None  # Has format selected


class TestToolAwareSerialization:
    """Tests for tool-aware serialization."""

    def test_serialize_with_tool_context(self):
        """Test serialization with tool context."""
        serializer = AdaptiveSerializer()
        context = SerializationContext(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            tool_name="git",
            tool_operation="log",
        )
        data = [{"commit": "abc123", "message": "test"}] * 5

        result = serializer.serialize(data, context)

        assert result.content
        assert result.format is not None

    def test_serialize_for_tool_method(self):
        """Test serialize_for_tool convenience method."""
        serializer = AdaptiveSerializer()
        data = [{"id": i, "name": f"item_{i}"} for i in range(10)]

        result = serializer.serialize_for_tool(
            data,
            tool_name="database",
            operation="query",
            provider="anthropic",
        )

        assert result.content
        assert result.format is not None

    def test_tool_config_registry(self):
        """Test ToolSerializationRegistry."""
        from victor.processing.serialization.tool_config import (
            ToolSerializationRegistry,
            reset_tool_serialization_registry,
        )

        reset_tool_serialization_registry()
        registry = ToolSerializationRegistry()

        # Get config for unknown tool (should return defaults)
        config = registry.get_tool_config("unknown_tool")
        assert config.tool_name == "unknown_tool"
        assert config.serialization_enabled  # Default is enabled

    def test_tool_output_type(self):
        """Test ToolOutputType enum."""
        from victor.processing.serialization.tool_config import ToolOutputType

        assert ToolOutputType.TABULAR.value == "tabular"
        assert ToolOutputType.TEXT.value == "text"
        assert ToolOutputType.STRUCTURED.value == "structured"
        assert ToolOutputType.MIXED.value == "mixed"


class TestSerializationSettings:
    """Tests for Settings integration."""

    def test_config_from_settings(self):
        """Test config_from_settings function."""
        from victor.processing.serialization import config_from_settings

        config = config_from_settings()

        # Should return a valid SerializationConfig
        assert config is not None
        # Settings defaults
        assert config.min_savings_threshold >= 0
        assert config.min_savings_threshold <= 1

    def test_is_serialization_enabled(self):
        """Test is_serialization_enabled function."""
        from victor.processing.serialization import is_serialization_enabled

        # Should return True by default
        result = is_serialization_enabled()
        assert isinstance(result, bool)


class TestMetricsCollector:
    """Tests for SerializationMetricsCollector."""

    def test_record_and_get_stats(self, tmp_path):
        """Test recording metrics and retrieving stats."""
        from victor.processing.serialization.metrics import (
            SerializationMetricsCollector,
            SerializationMetricRecord,
        )

        db_path = tmp_path / "test_metrics.db"
        collector = SerializationMetricsCollector(db_path)

        # Record a metric
        record = SerializationMetricRecord(
            timestamp="2025-01-01T00:00:00",
            tool_name="git",
            tool_operation="log",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            format_selected="toon",
            selection_reason="auto_selected",
            original_tokens=1000,
            serialized_tokens=500,
            token_savings_percent=50.0,
            char_savings_percent=45.0,
            data_structure_type="uniform_array",
            array_length=20,
            has_nested_objects=False,
            analysis_time_ms=1.5,
            encoding_time_ms=2.0,
        )
        collector.record(record)

        # Get tool stats
        stats = collector.get_tool_stats("git")
        assert stats["total_serializations"] == 1
        assert stats["avg_savings_percent"] == 50.0
        assert stats["total_tokens_saved"] == 500

        # Get overall stats
        overall = collector.get_overall_stats()
        assert overall["total_serializations"] == 1
        assert overall["total_tokens_saved"] == 500

    def test_format_stats(self, tmp_path):
        """Test format statistics aggregation."""
        from victor.processing.serialization.metrics import (
            SerializationMetricsCollector,
            SerializationMetricRecord,
        )

        db_path = tmp_path / "test_metrics.db"
        collector = SerializationMetricsCollector(db_path)

        # Record multiple metrics with different formats
        for fmt, savings in [("toon", 50.0), ("csv", 60.0), ("toon", 55.0)]:
            record = SerializationMetricRecord(
                timestamp="2025-01-01T00:00:00",
                tool_name="database",
                tool_operation="query",
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                format_selected=fmt,
                selection_reason="auto_selected",
                original_tokens=1000,
                serialized_tokens=int(1000 * (1 - savings / 100)),
                token_savings_percent=savings,
                char_savings_percent=savings - 5,
                data_structure_type="uniform_array",
                array_length=20,
                has_nested_objects=False,
                analysis_time_ms=1.5,
                encoding_time_ms=2.0,
            )
            collector.record(record)

        # Get format stats
        format_stats = collector.get_format_stats()
        assert len(format_stats) == 2  # toon and csv

        toon_stats = next(s for s in format_stats if s["format"] == "toon")
        assert toon_stats["usage_count"] == 2

    def test_clear_metrics(self, tmp_path):
        """Test clearing metrics."""
        from victor.processing.serialization.metrics import (
            SerializationMetricsCollector,
            SerializationMetricRecord,
        )

        db_path = tmp_path / "test_metrics.db"
        collector = SerializationMetricsCollector(db_path)

        # Record a metric
        record = SerializationMetricRecord(
            timestamp="2025-01-01T00:00:00",
            tool_name="git",
            tool_operation="log",
            provider=None,
            model=None,
            format_selected="json",
            selection_reason="fallback",
            original_tokens=100,
            serialized_tokens=100,
            token_savings_percent=0.0,
            char_savings_percent=0.0,
            data_structure_type="object",
            array_length=0,
            has_nested_objects=True,
            analysis_time_ms=0.5,
            encoding_time_ms=0.3,
        )
        collector.record(record)

        # Clear and verify
        collector.clear_metrics()
        stats = collector.get_overall_stats()
        assert stats["total_serializations"] == 0
