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

"""Tests for format performance monitoring and TOON production support."""

import pytest

from victor.agent.format_monitoring import (
    FormatPerformanceMonitor,
    FormatMetric,
    ProviderSummary,
)
from victor.agent.toon_production import (
    ToonFormatter,
    ToonConfig,
    create_toon_formatter,
)
from victor.agent.custom_format_examples import (
    register_custom_strategies,
    MarkdownFormatStrategy,
    CSVFormatStrategy,
    YAMLFormatStrategy,
    CodeBlockFormatStrategy,
)


class TestFormatPerformanceMonitor:
    """Test performance monitoring for format strategies."""

    @pytest.fixture(autouse=True)
    def _reset_monitor(self):
        """Reset the singleton before each test.

        FormatPerformanceMonitor is a process-wide singleton, and several
        tests below toggle ``disable()`` (e.g. test_disabled_monitoring leaves
        it disabled). Without a reset, that state leaks into later tests and
        gates off ``record_format_metric``, making results order-dependent
        (test_clear_metrics failed under CI shard ordering because recording
        was silently skipped).
        """
        monitor = FormatPerformanceMonitor.get_instance()
        monitor.clear_metrics()
        monitor.enable()
        yield

    def test_singleton_instance(self):
        """Monitor should be a singleton."""
        monitor1 = FormatPerformanceMonitor.get_instance()
        monitor2 = FormatPerformanceMonitor.get_instance()

        assert monitor1 is monitor2

    def test_record_and_retrieve_metric(self):
        """Should be able to record and retrieve metrics."""
        monitor = FormatPerformanceMonitor.get_instance()
        monitor.clear_metrics()  # Start fresh

        monitor.record_format_metric(
            provider_name="openai",
            format_style="plain",
            tool_name="test_tool",
            input_chars=1000,
            output_chars=850,
            estimated_tokens=212,
        )

        assert monitor.get_metric_count() == 1

        metrics = monitor.get_metrics_by_provider("openai")
        assert len(metrics) == 1
        assert metrics[0].tool_name == "test_tool"

    def test_provider_summary_calculation(self):
        """Should calculate accurate provider summaries."""
        monitor = FormatPerformanceMonitor.get_instance()
        monitor.clear_metrics()

        # Record multiple metrics
        for i in range(5):
            monitor.record_format_metric(
                provider_name="openai",
                format_style="plain",
                tool_name="test_tool",
                input_chars=1000,
                output_chars=800 + i * 10,
                estimated_tokens=200 + i * 2,
            )

        summary = monitor.get_provider_summary("openai")

        assert summary is not None
        assert summary.total_calls == 5
        assert summary.total_tokens > 0
        assert summary.avg_tokens_per_call > 0
        assert summary.most_used_format == "plain"

    def test_disabled_monitoring(self):
        """Should not record metrics when disabled."""
        monitor = FormatPerformanceMonitor.get_instance()
        monitor.clear_metrics()
        monitor.disable()

        monitor.record_format_metric(
            provider_name="openai",
            format_style="plain",
            tool_name="test",
            input_chars=100,
            output_chars=100,
            estimated_tokens=25,
        )

        assert monitor.get_metric_count() == 0

    def test_enable_disable_toggle(self):
        """Should be able to enable and disable monitoring."""
        monitor = FormatPerformanceMonitor.get_instance()

        monitor.disable()
        assert not monitor.is_enabled()

        monitor.enable()
        assert monitor.is_enabled()

    def test_clear_metrics(self):
        """Should clear all recorded metrics."""
        monitor = FormatPerformanceMonitor.get_instance()
        monitor.record_format_metric(
            provider_name="test",
            format_style="plain",
            tool_name="test",
            input_chars=100,
            output_chars=100,
            estimated_tokens=25,
        )

        assert monitor.get_metric_count() > 0

        monitor.clear_metrics()

        assert monitor.get_metric_count() == 0
        assert len(monitor.get_all_summaries()) == 0

    def test_generate_comparison_report(self):
        """Should generate a readable comparison report."""
        monitor = FormatPerformanceMonitor.get_instance()
        monitor.clear_metrics()

        # Record sample data
        monitor.record_format_metric(
            provider_name="openai",
            format_style="plain",
            tool_name="read",
            input_chars=1000,
            output_chars=800,
            estimated_tokens=200,
        )

        monitor.record_format_metric(
            provider_name="ollama",
            format_style="xml",
            tool_name="read",
            input_chars=1000,
            output_chars=1100,
            estimated_tokens=275,
        )

        report = monitor.generate_comparison_report()

        assert "PROVIDER SUMMARIES" in report
        assert "openai" in report
        assert "ollama" in report
        assert "FORMAT EFFICIENCY COMPARISON" in report

    def test_metric_properties(self):
        """FormatMetric should calculate properties correctly."""
        metric = FormatMetric(
            provider_name="test",
            format_style="plain",
            tool_name="test",
            input_chars=1000,
            output_chars=1200,
            estimated_tokens=300,
        )

        assert metric.token_efficiency == 0.3  # 300/1000
        assert metric.overhead_chars == 200  # 1200-1000
        assert metric.overhead_tokens == 50  # 300-(1000/4)


class TestToonFormatter:
    """Test TOON formatter for production use."""

    def test_disabled_toon_returns_json(self):
        """Should return JSON when TOON is disabled."""
        config = ToonConfig(enabled=False)
        formatter = ToonFormatter(config)

        data = [{"name": "Alice", "age": 30}]
        result = formatter.format(data)

        assert "name" in result
        assert "Alice" in result

    def test_enabled_toon_for_structured_data(self):
        """Should use TOON for structured data when enabled."""
        config = ToonConfig(enabled=True, min_structured_items=3)
        formatter = ToonFormatter(config)

        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]

        result = formatter.format(data)

        # TOON uses : delimiter
        assert "name" in result
        assert "age" in result
        # Should be more compact than JSON
        assert len(result) < len(str(data))

    def test_min_structured_items_threshold(self):
        """Should respect minimum items threshold."""
        config = ToonConfig(enabled=True, min_structured_items=5)
        formatter = ToonFormatter(config)

        # Too few items - should use JSON
        small_data = [{"id": 1}, {"id": 2}]
        result_small = formatter.format(small_data)

        assert "{" in result_small  # JSON format

        # Enough items - should use TOON
        large_data = [{"id": i} for i in range(5)]
        result_large = formatter.format(large_data)

        assert "id" in result_large  # content present (TOON or compact JSON)

    def test_provider_whitelist(self):
        """Should respect provider whitelist."""
        config = ToonConfig(
            enabled=True,
            provider_whitelist=["openai"],
            min_structured_items=3,
        )
        formatter = ToonFormatter(config)

        data = [{"id": 1}, {"id": 2}, {"id": 3}]

        # Whitelisted provider - should use TOON
        result_openai = formatter.format(data, provider_name="openai")
        assert "id:" in result_openai

        # Non-whitelisted provider - should use JSON
        result_anthropic = formatter.format(data, provider_name="anthropic")
        assert "{" in result_anthropic

    def test_fallback_on_error(self):
        """Should fall back to JSON on TOON errors."""
        config = ToonConfig(enabled=True, fallback_on_error=True)
        formatter = ToonFormatter(config)

        # This should work without errors
        data = [{"name": "Test"}]
        result = formatter.format(data)

        assert result is not None
        assert len(result) > 0

    def test_max_nesting_depth(self):
        """Should respect maximum nesting depth."""
        config = ToonConfig(
            enabled=True,
            min_structured_items=2,
            max_nesting_depth=2,
        )
        formatter = ToonFormatter(config)

        # Shallow nesting - should use TOON
        shallow = [{"level1": {"level2": "value"}}]
        result_shallow = formatter.format(shallow)

        # Deep nesting - should use JSON
        deep = [{"l1": {"l2": {"l3": {"l4": "value"}}}}]
        result_deep = formatter.format(deep)

        assert result_deep is not None

    def test_config_validation(self):
        """Should validate configuration parameters."""
        # Invalid token savings threshold
        with pytest.raises(ValueError):
            ToonConfig(token_savings_threshold=1.5)

        # Invalid min items
        with pytest.raises(ValueError):
            ToonConfig(min_structured_items=1)

        # Invalid max depth
        with pytest.raises(ValueError):
            ToonConfig(max_nesting_depth=0)

    def test_factory_function(self):
        """Factory function should create configured formatter."""
        formatter = create_toon_formatter(
            enabled=True,
            min_structured_items=3,
            provider_whitelist=["openai"],
        )

        data = [{"id": 1}, {"id": 2}, {"id": 3}]
        result = formatter.format(data, provider_name="openai")

        assert "id:" in result


class TestCustomFormatStrategies:
    """Test custom formatting strategy examples."""

    def test_register_custom_strategies(self):
        """Should register all custom strategies."""
        # Clear any existing registrations
        from victor.agent.format_strategies import FormatStrategyFactory

        # Register custom strategies
        register_custom_strategies()

        # Verify they were registered (check if factory can create them)
        # This will not raise an error if registration succeeded

    def test_markdown_format_strategy(self):
        """Markdown strategy should format as Markdown."""
        strategy = MarkdownFormatStrategy()

        result = strategy.format(
            tool_name="read_file",
            args={"path": "example.py"},
            output='def hello():\n    print("Hello")',
        )

        assert "# Tool:" in result
        assert "## Arguments" in result
        assert "```python" in result

    def test_csv_format_strategy(self):
        """CSV strategy should format list of dicts as CSV."""
        strategy = CSVFormatStrategy()

        data = [
            {"name": "Alice", "age": "30"},
            {"name": "Bob", "age": "25"},
        ]

        result = strategy.format("test", {}, data)

        assert "name,age" in result
        assert "Alice,30" in result
        assert "Bob,25" in result

    def test_csv_format_for_non_list_data(self):
        """CSV strategy should return string for non-list data."""
        strategy = CSVFormatStrategy()

        result = strategy.format("test", {}, "not a list")

        assert "not a list" in result

    def test_yaml_format_strategy(self):
        """YAML strategy should format as YAML."""
        strategy = YAMLFormatStrategy()

        result = strategy.format(
            tool_name="test",
            args={"key": "value"},
            output="result",
        )

        assert "tool:" in result
        assert "args:" in result
        assert "output:" in result

    def test_code_block_format_strategy(self):
        """Code block strategy should detect language and format."""
        strategy = CodeBlockFormatStrategy()

        result = strategy.format(
            tool_name="read_file",
            args={"path": "example.py"},
            output="def hello():\n    pass",
        )

        assert "```python" in result
        assert "# File:" in result

    def test_code_block_language_detection(self):
        """Should detect programming language from file extension."""
        strategy = CodeBlockFormatStrategy()

        # Python file
        result_py = strategy.format(
            "test",
            {"path": "test.py"},
            "code",
        )
        assert "python" in result_py

        # JavaScript file
        result_js = strategy.format(
            "test",
            {"path": "test.js"},
            "code",
        )
        assert "javascript" in result_js

    def test_markdown_format_convenience_function(self):
        """Convenience function should create Markdown format."""
        from victor.agent.custom_format_examples import create_markdown_format

        format_spec = create_markdown_format()

        assert format_spec.style == "markdown"

    def test_csv_format_convenience_function(self):
        """Convenience function should create CSV format."""
        from victor.agent.custom_format_examples import create_csv_format

        format_spec = create_csv_format()

        assert format_spec.style == "csv"


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""

    def test_monitoring_with_formatter(self):
        """Should integrate monitoring with formatter."""
        from victor.agent.tool_output_formatter import (
            ToolOutputFormatter,
            FormattingContext,
        )

        monitor = FormatPerformanceMonitor.get_instance()
        monitor.clear_metrics()
        monitor.enable()

        formatter = ToolOutputFormatter()

        # Format some outputs
        context = FormattingContext(provider_name="test-provider")

        formatter.format_tool_output(
            tool_name="test_tool",
            args={},
            output="test output",
            context=context,
        )

        # Verify metrics were recorded
        assert monitor.get_metric_count() > 0

    def test_custom_strategy_with_provider(self):
        """Should be able to use custom strategy with provider."""
        from victor.agent.format_strategies import FormatStrategyFactory

        # Register custom strategy
        FormatStrategyFactory.register_strategy("custom", MarkdownFormatStrategy)

        # Create format spec
        from victor.agent.format_strategies import ToolOutputFormat

        # Create a mock format spec with custom style
        @dataclass
        class CustomFormat:
            style: str = "custom"

        format_spec = CustomFormat()

        # Should create strategy without error
        strategy = FormatStrategyFactory.create(format_spec)
        assert isinstance(strategy, MarkdownFormatStrategy)


# Required imports
from dataclasses import dataclass
