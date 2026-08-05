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

"""Performance monitoring for tool output formatting.

This module provides tracking and reporting of token usage and efficiency
metrics for different formatting strategies across providers.

Features:
- Track token usage by provider and format
- Calculate efficiency improvements (XML vs Plain vs TOON)
- Generate performance reports
- Identify optimization opportunities
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FormatMetric:
    """Single formatting operation metric.

    Attributes:
        provider_name: Provider identifier
        format_style: Format style used (plain, xml, toon)
        tool_name: Tool that was executed
        input_chars: Character count of input
        output_chars: Character count of formatted output
        estimated_tokens: Estimated token count
        timestamp: When the formatting occurred
    """

    provider_name: str
    format_style: str
    tool_name: str
    input_chars: int
    output_chars: int
    estimated_tokens: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def token_efficiency(self) -> float:
        """Calculate tokens per character ratio."""
        if self.input_chars == 0:
            return 0.0
        return self.estimated_tokens / self.input_chars

    @property
    def overhead_chars(self) -> int:
        """Calculate formatting overhead in characters."""
        return self.output_chars - self.input_chars

    @property
    def overhead_tokens(self) -> int:
        """Calculate formatting overhead in tokens."""
        return self.estimated_tokens - (self.input_chars // 4)


@dataclass
class ProviderSummary:
    """Summary statistics for a single provider.

    Attributes:
        provider_name: Provider identifier
        total_calls: Total number of tool calls
        total_tokens: Total tokens used
        total_overhead_tokens: Total formatting overhead tokens
        avg_tokens_per_call: Average tokens per tool call
        format_distribution: Count of calls per format style
        most_used_format: Most frequently used format
    """

    provider_name: str
    total_calls: int = 0
    total_tokens: int = 0
    total_overhead_tokens: int = 0
    avg_tokens_per_call: float = 0.0
    format_distribution: Dict[str, int] = field(default_factory=dict)
    most_used_format: str = "unknown"

    def add_metric(self, metric: FormatMetric) -> None:
        """Add a metric to the summary.

        Args:
            metric: FormatMetric to incorporate
        """
        self.total_calls += 1
        self.total_tokens += metric.estimated_tokens
        self.total_overhead_tokens += metric.overhead_tokens
        self.avg_tokens_per_call = self.total_tokens / self.total_calls

        # Update format distribution
        self.format_distribution[metric.format_style] = (
            self.format_distribution.get(metric.format_style, 0) + 1
        )

        # Update most used format
        if self.format_distribution.get(metric.format_style, 0) > self.format_distribution.get(
            self.most_used_format, 0
        ):
            self.most_used_format = metric.format_style


class FormatPerformanceMonitor:
    """Monitor and report tool output formatting performance.

    This singleton class tracks all formatting operations and provides
    analytics for token efficiency and optimization opportunities.

    Usage:
        monitor = FormatPerformanceMonitor.get_instance()
        monitor.record_format_metric(
            provider_name="openai",
            format_style="plain",
            tool_name="read",
            input_chars=1000,
            output_chars=850,
            estimated_tokens=212,
        )

        # Get summary
        summary = monitor.get_provider_summary("openai")
        print(f"Avg tokens: {summary.avg_tokens_per_call:.1f}")

        # Get comparison report
        report = monitor.generate_comparison_report()
    """

    _instance: Optional["FormatPerformanceMonitor"] = None

    def __init__(self) -> None:
        """Initialize monitor (use get_instance() instead)."""
        self._metrics: List[FormatMetric] = []
        self._provider_summaries: Dict[str, ProviderSummary] = defaultdict(ProviderSummary)
        self._enabled = True

    @classmethod
    def get_instance(cls) -> "FormatPerformanceMonitor":
        """Get singleton monitor instance.

        Returns:
            FormatPerformanceMonitor instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def enable(self) -> None:
        """Enable performance monitoring."""
        self._enabled = True
        logger.info("Format performance monitoring enabled")

    def disable(self) -> None:
        """Disable performance monitoring."""
        self._enabled = False
        logger.info("Format performance monitoring disabled")

    def is_enabled(self) -> bool:
        """Check if monitoring is enabled.

        Returns:
            True if monitoring is enabled
        """
        return self._enabled

    def record_format_metric(
        self,
        provider_name: str,
        format_style: str,
        tool_name: str,
        input_chars: int,
        output_chars: int,
        estimated_tokens: int,
    ) -> None:
        """Record a formatting operation metric.

        Args:
            provider_name: Provider identifier
            format_style: Format style used (plain, xml, toon)
            tool_name: Tool that was executed
            input_chars: Character count of input
            output_chars: Character count of formatted output
            estimated_tokens: Estimated token count
        """
        if not self._enabled:
            return

        metric = FormatMetric(
            provider_name=provider_name,
            format_style=format_style,
            tool_name=tool_name,
            input_chars=input_chars,
            output_chars=output_chars,
            estimated_tokens=estimated_tokens,
        )

        self._metrics.append(metric)

        # Update provider summary
        if provider_name not in self._provider_summaries:
            self._provider_summaries[provider_name] = ProviderSummary(provider_name=provider_name)

        self._provider_summaries[provider_name].add_metric(metric)

        logger.debug(
            f"Recorded format metric: {provider_name}/{format_style}/{tool_name} "
            f"({estimated_tokens} tokens, {metric.overhead_tokens} overhead)"
        )

    def get_provider_summary(self, provider_name: str) -> Optional[ProviderSummary]:
        """Get performance summary for a specific provider.

        Args:
            provider_name: Provider identifier

        Returns:
            ProviderSummary or None if no metrics recorded
        """
        return self._provider_summaries.get(provider_name)

    def get_all_summaries(self) -> Dict[str, ProviderSummary]:
        """Get performance summaries for all providers.

        Returns:
            Dict mapping provider names to ProviderSummary
        """
        return dict(self._provider_summaries)

    def generate_comparison_report(self) -> str:
        """Generate a comparison report across all providers and formats.

        Returns:
            Formatted report string
        """
        if not self._provider_summaries:
            return "No performance data available yet."

        lines = [
            "=" * 80,
            "TOOL OUTPUT FORMAT PERFORMANCE REPORT",
            "=" * 80,
            "",
            f"Total providers tracked: {len(self._provider_summaries)}",
            f"Total formatting operations: {len(self._metrics)}",
            "",
            "-" * 80,
            "PROVIDER SUMMARIES",
            "-" * 80,
            "",
        ]

        # Sort by total tokens (most expensive first)
        sorted_providers = sorted(
            self._provider_summaries.items(),
            key=lambda x: x[1].total_tokens,
            reverse=True,
        )

        for provider_name, summary in sorted_providers:
            lines.extend(
                [
                    f"Provider: {provider_name}",
                    f"  Total calls: {summary.total_calls}",
                    f"  Total tokens: {summary.total_tokens:,}",
                    f"  Avg tokens/call: {summary.avg_tokens_per_call:.1f}",
                    f"  Total overhead: {summary.total_overhead_tokens:,} tokens",
                    f"  Format distribution: {summary.format_distribution}",
                    f"  Most used format: {summary.most_used_format}",
                    "",
                ]
            )

        # Format comparison
        lines.extend(["-" * 80, "FORMAT EFFICIENCY COMPARISON", "-" * 80, ""])

        format_stats = defaultdict(lambda: {"calls": 0, "tokens": 0, "overhead": 0})

        for metric in self._metrics:
            format_stats[metric.format_style]["calls"] += 1
            format_stats[metric.format_style]["tokens"] += metric.estimated_tokens
            format_stats[metric.format_style]["overhead"] += metric.overhead_tokens

        for format_style, stats in sorted(
            format_stats.items(), key=lambda x: x[1]["tokens"], reverse=True
        ):
            avg_tokens = stats["tokens"] / stats["calls"] if stats["calls"] > 0 else 0
            avg_overhead = stats["overhead"] / stats["calls"] if stats["calls"] > 0 else 0

            lines.extend(
                [
                    f"Format: {format_style.upper()}",
                    f"  Calls: {stats['calls']:,}",
                    f"  Total tokens: {stats['tokens']:,}",
                    f"  Avg tokens/call: {avg_tokens:.1f}",
                    f"  Avg overhead/call: {avg_overhead:.1f}",
                    "",
                ]
            )

        # Efficiency analysis
        lines.extend(["-" * 80, "EFFICIENCY INSIGHTS", "-" * 80, ""])

        if "plain" in format_stats and "xml" in format_stats:
            plain_avg = format_stats["plain"]["tokens"] / format_stats["plain"]["calls"]
            xml_avg = format_stats["xml"]["tokens"] / format_stats["xml"]["calls"]
            savings_pct = ((xml_avg - plain_avg) / xml_avg) * 100

            lines.extend(
                [
                    "Plain vs XML Efficiency:",
                    f"  Plain format avg: {plain_avg:.1f} tokens/call",
                    f"  XML format avg: {xml_avg:.1f} tokens/call",
                    f"  Savings with Plain: {savings_pct:.1f}%",
                    "",
                ]
            )

        if "toon" in format_stats:
            toon_avg = format_stats["toon"]["tokens"] / format_stats["toon"]["calls"]
            lines.append(f"TOON format avg: {toon_avg:.1f} tokens/call")
            lines.append("")

        # Optimization recommendations
        lines.extend(["-" * 80, "OPTIMIZATION RECOMMENDATIONS", "-" * 80, ""])

        recommendations = self._generate_recommendations()
        if recommendations:
            lines.extend(recommendations)
        else:
            lines.append("No optimization recommendations at this time.")

        lines.append("=" * 80)

        return "\n".join(lines)

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on metrics.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check for providers using XML that could use Plain
        for provider_name, summary in self._provider_summaries.items():
            if summary.most_used_format == "xml" and summary.total_calls > 10:
                # Calculate potential savings
                plain_overhead = 5  # Estimated overhead for plain format
                xml_overhead = summary.total_overhead_tokens / summary.total_calls
                potential_savings = (xml_overhead - plain_overhead) * summary.total_calls
                savings_pct = (potential_savings / summary.total_tokens) * 100

                if savings_pct > 10:  # Only recommend if savings > 10%
                    recommendations.append(
                        f"• {provider_name}: Consider switching from XML to Plain format "
                        f"(potential {savings_pct:.1f}% token savings, ~{potential_savings:.0f} tokens)"
                    )

        # Check for high-overhead tools
        tool_metrics = defaultdict(lambda: {"overhead": 0, "count": 0})
        for metric in self._metrics:
            tool_metrics[metric.tool_name]["overhead"] += metric.overhead_tokens
            tool_metrics[metric.tool_name]["count"] += 1

        high_overhead_tools = [
            (tool, stats["overhead"] / stats["count"])
            for tool, stats in tool_metrics.items()
            if stats["count"] >= 5
        ]
        high_overhead_tools.sort(key=lambda x: x[1], reverse=True)

        if high_overhead_tools and high_overhead_tools[0][1] > 20:
            tool, avg_overhead = high_overhead_tools[0]
            recommendations.append(
                f"• Tool '{tool}' has high formatting overhead ({avg_overhead:.1f} tokens/call). "
                f"Consider using Plain format for this tool."
            )

        return recommendations

    def get_metrics_by_format(self, format_style: str) -> List[FormatMetric]:
        """Get all metrics for a specific format style.

        Args:
            format_style: Format style to filter by (plain, xml, toon)

        Returns:
            List of FormatMetric objects
        """
        return [m for m in self._metrics if m.format_style == format_style]

    def get_metrics_by_provider(self, provider_name: str) -> List[FormatMetric]:
        """Get all metrics for a specific provider.

        Args:
            provider_name: Provider to filter by

        Returns:
            List of FormatMetric objects
        """
        return [m for m in self._metrics if m.provider_name == provider_name]

    def clear_metrics(self) -> None:
        """Clear all recorded metrics (useful for testing or reset)."""
        self._metrics.clear()
        self._provider_summaries.clear()
        logger.info("Cleared all format performance metrics")

    def get_metric_count(self) -> int:
        """Get total number of recorded metrics.

        Returns:
            Count of metrics
        """
        return len(self._metrics)
