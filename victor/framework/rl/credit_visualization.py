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

"""
Enhanced visualization and export for credit assignment.

Provides:
- HTML interactive visualizations
- Chart generation (requires matplotlib/plotly)
- Export to CSV, JSON, Markdown
- Agent attribution reports
"""

from __future__ import annotations

import io
from victor.core.json_utils import json_dumps
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from victor.framework.rl.credit_assignment import (
    CreditSignal,
    ActionMetadata,
    CreditGranularity,
    CreditMethodology,
)

logger = logging.getLogger(__name__)


# ============================================================================
# HTML Visualization
# ============================================================================


class CreditVisualizationBuilder:
    """Build HTML visualizations for credit assignment results."""

    def __init__(self, title: str = "Credit Assignment Analysis"):
        self.title = title
        self.sections: List[str] = []

    def add_header(self, text: str, level: int = 2) -> "CreditVisualizationBuilder":
        """Add a header."""
        tag = f"h{level}"
        self.sections.append(f"<{tag}>{text}</{tag}>")
        return self

    def add_paragraph(self, text: str) -> "CreditVisualizationBuilder":
        """Add a paragraph."""
        self.sections.append(f"<p>{text}</p>")
        return self

    def add_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        caption: Optional[str] = None,
    ) -> "CreditVisualizationBuilder":
        """Add a table."""
        html = ["<table>"]

        if caption:
            html.append(f"<caption>{caption}</caption>")

        # Header row
        html.append("<thead><tr>")
        for header in headers:
            html.append(f"<th>{header}</th>")
        html.append("</tr></thead>")

        # Data rows
        html.append("<tbody>")
        for row in rows:
            html.append("<tr>")
            for cell in row:
                html.append(f"<td>{cell}</td>")
            html.append("</tr>")
        html.append("</tbody>")

        html.append("</table>")
        self.sections.append("".join(html))
        return self

    def add_credit_bar_chart(
        self,
        signals: List[CreditSignal],
        max_bars: int = 50,
    ) -> "CreditVisualizationBuilder":
        """Add a simple HTML bar chart for credits."""
        # Sort by credit value
        sorted_signals = sorted(signals, key=lambda s: abs(s.credit), reverse=True)
        display_signals = sorted_signals[:max_bars]

        max_credit = max(abs(s.credit) for s in display_signals) if display_signals else 1.0

        html = ['<div class="credit-chart">']
        html.append("<h3>Credit Distribution</h3>")

        for signal in display_signals:
            credit = signal.credit
            bar_width = abs(credit) / max_credit * 100
            color = "#4CAF50" if credit >= 0 else "#f44336"

            html.append('<div class="credit-bar">')
            html.append(f'<div class="bar-label">{signal.action_id[:30]}</div>')
            html.append('<div class="bar-container">')
            html.append(
                f'<div class="bar" style="width: {bar_width}%; background: {color};">'
                f"{credit:+.3f}</div>"
            )
            html.append("</div>")
            html.append("</div>")

        html.append("</div>")

        # Add CSS
        css = """
        <style>
            .credit-chart { margin: 20px 0; }
            .credit-bar { margin: 8px 0; }
            .bar-label { font-size: 12px; margin-bottom: 2px; }
            .bar-container {
                background: #f0f0f0;
                border-radius: 4px;
                height: 24px;
                position: relative;
            }
            .bar {
                height: 100%;
                border-radius: 4px;
                display: flex;
                align-items: center;
                padding-left: 8px;
                color: white;
                font-size: 11px;
                font-weight: bold;
            }
        </style>
        """

        self.sections.append(css + "".join(html))
        return self

    def add_agent_attribution(
        self,
        attribution: Dict[str, Dict[str, float]],
    ) -> "CreditVisualizationBuilder":
        """Add agent attribution visualization."""
        html = ['<div class="attribution">']
        html.append("<h3>Agent Attribution</h3>")

        for agent, contributors in attribution.items():
            html.append(f"<h4>{agent}</h4>")
            html.append("<ul>")
            for contributor, amount in contributors.items():
                html.append(f"<li>{contributor}: {amount:+.3f}</li>")
            html.append("</ul>")

        html.append("</div>")

        self.sections.append("".join(html))
        return self

    def add_metrics_summary(
        self,
        metrics: Dict[str, float],
    ) -> "CreditVisualizationBuilder":
        """Add metrics summary."""
        html = ['<div class="metrics">']
        html.append("<h3>Metrics Summary</h3>")
        html.append("<dl>")

        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            html.append(f"<dt>{key}</dt>")
            html.append(f"<dd>{formatted_value}</dd>")

        html.append("</dl>")
        html.append("</div>")

        # Add CSS
        css = """
        <style>
            .metrics { margin: 20px 0; }
            .metrics dt { font-weight: bold; margin-top: 10px; }
            .metrics dd { margin-left: 20px; }
        </style>
        """

        self.sections.append(css + "".join(html))
        return self

    def build(self) -> str:
        """Build the complete HTML document."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1, h2, h3 {{ color: #333; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>
    {''.join(self.sections)}
    <hr>
    <footer>
        <p>Generated by Victor Credit Assignment System on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </footer>
</body>
</html>"""
        return html

    def save(self, path: Union[str, Path]) -> None:
        """Save visualization to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        html = self.build()
        with open(path, "w") as f:
            f.write(html)

        logger.info(f"Saved visualization to: {path}")


# ============================================================================
# Export Functions
# ============================================================================


@dataclass
class ExportConfig:
    """Configuration for credit assignment data export."""

    format: str = "json"  # json, csv, md, html
    include_signals: bool = True
    include_metrics: bool = True
    include_attribution: bool = True
    sort_by: Optional[str] = None  # "credit", "action_id", "confidence"
    filter_min_credit: Optional[float] = None
    filter_agent: Optional[str] = None


class CreditAssignmentExporter:
    """Export credit assignment results to various formats."""

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()

    def export(
        self,
        signals: List[CreditSignal],
        metrics: Dict[str, float],
        attribution: Optional[Dict[str, Dict[str, float]]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Export credit assignment data.

        Args:
            signals: Credit signals to export
            metrics: Computed metrics
            attribution: Optional agent attribution
            output_path: Optional path to save output

        Returns:
            Exported data as string
        """
        # Apply filters
        filtered_signals = self._filter_signals(signals)

        # Sort if specified
        if self.config.sort_by:
            filtered_signals = self._sort_signals(filtered_signals, self.config.sort_by)

        # Export in requested format
        if self.config.format == "json":
            data = self._export_json(filtered_signals, metrics, attribution)
        elif self.config.format == "csv":
            data = self._export_csv(filtered_signals, metrics, attribution)
        elif self.config.format == "md":
            data = self._export_markdown(filtered_signals, metrics, attribution)
        elif self.config.format == "html":
            data = self._export_html(filtered_signals, metrics, attribution)
        else:
            raise ValueError(f"Unknown format: {self.config.format}")

        # Save to file if specified
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(data)
            logger.info(f"Exported to: {output_path}")

        return data

    def _filter_signals(self, signals: List[CreditSignal]) -> List[CreditSignal]:
        """Apply filters to signals."""
        filtered = signals

        if self.config.filter_min_credit is not None:
            filtered = [s for s in filtered if abs(s.credit) >= self.config.filter_min_credit]

        if self.config.filter_agent is not None:
            filtered = [
                s
                for s in filtered
                if s.metadata and s.metadata.agent_id == self.config.filter_agent
            ]

        return filtered

    def _sort_signals(
        self,
        signals: List[CreditSignal],
        sort_by: str,
    ) -> List[CreditSignal]:
        """Sort signals by specified field."""
        if sort_by == "credit":
            return sorted(signals, key=lambda s: abs(s.credit), reverse=True)
        elif sort_by == "action_id":
            return sorted(signals, key=lambda s: s.action_id)
        elif sort_by == "confidence":
            return sorted(signals, key=lambda s: s.confidence, reverse=True)
        else:
            return signals

    def _export_json(
        self,
        signals: List[CreditSignal],
        metrics: Dict[str, float],
        attribution: Optional[Dict[str, Dict[str, float]]],
    ) -> str:
        """Export as JSON."""
        data: Dict[str, Any] = {
            "exported_at": datetime.now().isoformat(),
            "format": "json",
        }

        if self.config.include_signals:
            data["signals"] = [s.to_dict() for s in signals]

        if self.config.include_metrics:
            data["metrics"] = metrics

        if self.config.include_attribution and attribution:
            data["attribution"] = attribution

        return json_dumps(data, indent=2)

    def _export_csv(
        self,
        signals: List[CreditSignal],
        metrics: Dict[str, float],
        attribution: Optional[Dict[str, Dict[str, float]]],
    ) -> str:
        """Export as CSV."""
        import csv

        output = io.StringIO()
        writer = csv.writer(output)

        # Write signals
        if self.config.include_signals:
            writer.writerow(
                [
                    "action_id",
                    "raw_reward",
                    "credit",
                    "confidence",
                    "methodology",
                    "granularity",
                    "agent_id",
                    "team_id",
                    "tool_name",
                    "method_name",
                    "turn_index",
                    "step_index",
                ]
            )

            for signal in signals:
                metadata = signal.metadata or ActionMetadata(agent_id="unknown")
                writer.writerow(
                    [
                        signal.action_id,
                        signal.raw_reward,
                        signal.credit,
                        signal.confidence,
                        signal.methodology.value if signal.methodology else "",
                        signal.granularity.value,
                        metadata.agent_id,
                        metadata.team_id or "",
                        metadata.tool_name or "",
                        metadata.method_name or "",
                        metadata.turn_index,
                        metadata.step_index,
                    ]
                )

        # Write metrics as comment
        if self.config.include_metrics:
            writer.writerow([])
            writer.writerow(["# Metrics"])
            for key, value in metrics.items():
                writer.writerow([f"# {key}", value])

        return output.getvalue()

    def _export_markdown(
        self,
        signals: List[CreditSignal],
        metrics: Dict[str, float],
        attribution: Optional[Dict[str, Dict[str, float]]],
    ) -> str:
        """Export as Markdown."""
        lines = []
        lines.append("# Credit Assignment Report")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Metrics
        if self.config.include_metrics:
            lines.append("## Metrics\n")
            for key, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"- **{key}**: {value:.4f}")
                else:
                    lines.append(f"- **{key}**: {value}")
            lines.append("")

        # Signals
        if self.config.include_signals:
            lines.append("## Credit Signals\n")
            lines.append("| Action ID | Agent | Reward | Credit | Confidence | Methodology |")
            lines.append("|----------|-------|--------|--------|------------|-------------|")

            for signal in signals:
                agent_id = signal.metadata.agent_id if signal.metadata else "N/A"
                method = signal.methodology.value if signal.methodology else "N/A"
                lines.append(
                    f"| {signal.action_id} "
                    f"| {agent_id} "
                    f"| {signal.raw_reward:+.3f} "
                    f"| {signal.credit:+.3f} "
                    f"| {signal.confidence:.2f} "
                    f"| {method} |"
                )
            lines.append("")

        # Attribution
        if self.config.include_attribution and attribution:
            lines.append("## Agent Attribution\n")
            for agent, contributors in attribution.items():
                lines.append(f"### {agent}")
                for contributor, amount in contributors.items():
                    lines.append(f"- {contributor}: {amount:+.3f}")
                lines.append("")

        return "\n".join(lines)

    def _export_html(
        self,
        signals: List[CreditSignal],
        metrics: Dict[str, float],
        attribution: Optional[Dict[str, Dict[str, float]]],
    ) -> str:
        """Export as HTML."""
        builder = CreditVisualizationBuilder("Credit Assignment Report")

        # Metrics
        if self.config.include_metrics:
            builder.add_metrics_summary(metrics)

        # Signals table
        if self.config.include_signals:
            headers = [
                "Action ID",
                "Agent",
                "Reward",
                "Credit",
                "Confidence",
                "Methodology",
            ]
            rows = []
            for signal in signals[:50]:  # Limit to 50 for HTML
                agent_id = signal.metadata.agent_id if signal.metadata else "N/A"
                method = signal.methodology.value if signal.methodology else "N/A"
                rows.append(
                    [
                        signal.action_id,
                        agent_id,
                        f"{signal.raw_reward:+.3f}",
                        f"{signal.credit:+.3f}",
                        f"{signal.confidence:.2f}",
                        method,
                    ]
                )
            builder.add_table(headers, rows, caption="Credit Signals (top 50)")

        # Attribution
        if self.config.include_attribution and attribution:
            builder.add_agent_attribution(attribution)

        # Bar chart
        if self.config.include_signals:
            builder.add_credit_bar_chart(signals[:30])

        return builder.build()


# ============================================================================
# Report Generator
# ============================================================================


class CreditAssignmentReport:
    """Generate comprehensive credit assignment reports."""

    def __init__(
        self,
        title: str = "Credit Assignment Analysis",
        description: Optional[str] = None,
    ):
        self.title = title
        self.description = description
        self.sections: List[str] = []

    def add_summary(
        self,
        methodology: CreditMethodology,
        trajectory_length: int,
        total_reward: float,
        duration: Optional[float] = None,
    ) -> "CreditAssignmentReport":
        """Add executive summary."""
        lines = [
            "## Executive Summary\n",
            f"**Methodology**: {methodology.value}\n",
            f"**Trajectory Length**: {trajectory_length} actions\n",
            f"**Total Reward**: {total_reward:+.3f}\n",
        ]

        if duration is not None:
            lines.append(f"**Duration**: {duration:.2f} seconds\n")

        self.sections.extend(lines)
        return self

    def add_agent_breakdown(
        self,
        attribution: Dict[str, Dict[str, float]],
    ) -> "CreditAssignmentReport":
        """Add agent-level breakdown."""
        self.sections.append("## Agent Attribution\n")

        for agent, contributors in attribution.items():
            total_credit = sum(contributors.values())
            self.sections.append(f"### {agent}")
            self.sections.append(f"**Total Credit**: {total_credit:+.3f}\n")
            self.sections.append("**Contributors**:\n")

            for contributor, amount in sorted(
                contributors.items(), key=lambda x: abs(x[1]), reverse=True
            ):
                percentage = (amount / total_credit * 100) if total_credit != 0 else 0
                self.sections.append(f"- {contributor}: {amount:+.3f} ({percentage:+.1f}%)")

            self.sections.append("")

        return self

    def add_critical_actions(
        self,
        critical_indices: List[int],
        signals: List[CreditSignal],
    ) -> "CreditAssignmentReport":
        """Add critical action analysis."""
        self.sections.append("## Critical Actions\n")
        self.sections.append(
            f"Identified {len(critical_indices)} critical actions "
            "(bifurcation points that significantly influenced the outcome)\n"
        )

        for idx in critical_indices[:10]:  # Limit to 10
            if idx < len(signals):
                signal = signals[idx]
                self.sections.append(
                    f"**{signal.action_id}** (index {idx}): {signal.credit:+.3f}\n"
                )

        self.sections.append("")
        return self

    def add_recommendations(
        self,
        signals: List[CreditSignal],
        metrics: Dict[str, float],
    ) -> "CreditAssignmentReport":
        """Add actionable recommendations."""
        self.sections.append("## Recommendations\n")

        # Analyze patterns
        positive_ratio = metrics.get("positive_ratio", 0)
        avg_confidence = metrics.get("avg_confidence", 0)

        if positive_ratio < 0.5:
            self.sections.append(
                "- ⚠️ **Low Positive Ratio**: Consider reviewing unsuccessful actions "
                "for improvement opportunities.\n"
            )

        if avg_confidence < 0.7:
            self.sections.append(
                "- ⚠️ **Low Confidence**: Credit assignment uncertainty is high. "
                "Consider using alternative methodologies or collecting more data.\n"
            )

        # High-impact actions
        high_impact = [s for s in signals if abs(s.credit) > 2.0]
        if high_impact:
            self.sections.append(
                f"- 🔍 **{len(high_impact)} High-Impact Actions**: These actions "
                "disproportionately influenced the outcome. Consider analyzing them separately.\n"
            )

        self.sections.append("")
        return self

    def generate_markdown(self) -> str:
        """Generate report as Markdown."""
        lines = [f"# {self.title}\n"]

        if self.description:
            lines.append(f"{self.description}\n")

        lines.extend(self.sections)
        lines.append("---\n")
        lines.append("*Generated by Victor Credit Assignment System*\n")

        return "\n".join(lines)

    def generate_html(self) -> str:
        """Generate report as HTML."""
        import markdown

        md = self.generate_markdown()
        return markdown.markdown(md)


# ============================================================================
# Convenience Functions
# ============================================================================


def export_credit_report(
    signals: List[CreditSignal],
    metrics: Dict[str, float],
    attribution: Optional[Dict[str, Dict[str, float]]] = None,
    output_path: Optional[Path] = None,
    format: str = "html",
) -> str:
    """Export a comprehensive credit report.

    Args:
        signals: Credit signals
        metrics: Computed metrics
        attribution: Optional agent attribution
        output_path: Path to save report
        format: Output format (html, md, json)

    Returns:
        Report content as string
    """
    config = ExportConfig(format=format)
    exporter = CreditAssignmentExporter(config)

    return exporter.export(
        signals=signals,
        metrics=metrics,
        attribution=attribution,
        output_path=output_path,
    )


def create_interactive_report(
    signals: List[CreditSignal],
    metrics: Dict[str, float],
    attribution: Optional[Dict[str, Dict[str, float]]] = None,
    output_path: Optional[Path] = None,
) -> str:
    """Create an interactive HTML report with charts.

    Args:
        signals: Credit signals
        metrics: Computed metrics
        attribution: Optional agent attribution
        output_path: Path to save report

    Returns:
        HTML report content
    """
    builder = CreditVisualizationBuilder("Credit Assignment Analysis")

    # Add all components
    if metrics:
        builder.add_metrics_summary(metrics)

    if attribution:
        builder.add_agent_attribution(attribution)

    builder.add_credit_bar_chart(signals[:50])

    # Save if requested
    if output_path:
        builder.save(output_path)

    return builder.build()


# ============================================================================
# Exports
# ============================================================================


__all__ = [
    "CreditVisualizationBuilder",
    "ExportConfig",
    "CreditAssignmentExporter",
    "CreditAssignmentReport",
    "export_credit_report",
    "create_interactive_report",
]
