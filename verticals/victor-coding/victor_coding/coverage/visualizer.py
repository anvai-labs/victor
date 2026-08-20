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

"""Coverage visualization and reporting.

Generates text, HTML, and other visual representations of
coverage data.
"""

import html
import logging
from pathlib import Path
from typing import Optional

from victor_coding.coverage.protocol import (
    CoverageDiff,
    CoverageReport,
    CoverageStatus,
    CoverageThreshold,
    FileCoverage,
)

logger = logging.getLogger(__name__)


class CoverageVisualizer:
    """Generates coverage visualizations and reports."""

    def __init__(
        self,
        threshold: Optional[CoverageThreshold] = None,
        use_colors: bool = True,
    ):
        """Initialize the visualizer.

        Args:
            threshold: Coverage thresholds for highlighting
            use_colors: Whether to use ANSI colors in text output
        """
        self.threshold = threshold or CoverageThreshold()
        self.use_colors = use_colors

    def generate_text_report(
        self,
        report: CoverageReport,
        include_files: bool = True,
        max_files: int = 50,
    ) -> str:
        """Generate a text-based coverage report.

        Args:
            report: Coverage report to visualize
            include_files: Whether to include per-file details
            max_files: Maximum files to show in detail

        Returns:
            Text report string
        """
        lines = []

        # Header
        lines.append("=" * 70)
        lines.append("COVERAGE REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Files:       {report.file_count}")
        lines.append(
            f"Lines:       {report.covered_lines}/{report.total_lines} "
            f"({report.line_coverage_percent:.1f}%)"
        )
        if report.total_functions > 0:
            lines.append(
                f"Functions:   {report.covered_functions}/{report.total_functions} "
                f"({report.function_coverage_percent:.1f}%)"
            )
        if report.total_branches > 0:
            lines.append(
                f"Branches:    {report.covered_branches}/{report.total_branches} "
                f"({report.branch_coverage_percent:.1f}%)"
            )
        lines.append("")

        # Threshold check
        passed, failures = self.threshold.check(report)
        if failures:
            lines.append("THRESHOLD VIOLATIONS")
            lines.append("-" * 40)
            for failure in failures:
                status = self._colorize("FAIL", "red")
                lines.append(f"  {status}: {failure}")
            lines.append("")
        else:
            status = self._colorize("PASS", "green")
            lines.append(f"Threshold Check: {status}")
            lines.append("")

        # Per-file details
        if include_files:
            lines.append("FILE DETAILS")
            lines.append("-" * 70)
            lines.append(f"{'File':<50} {'Lines':>8} {'Cover':>8}")
            lines.append("-" * 70)

            sorted_files = sorted(
                report.files.values(),
                key=lambda f: f.line_coverage_percent,
            )

            for file_cov in sorted_files[:max_files]:
                filename = str(file_cov.file_path)
                if len(filename) > 48:
                    filename = "..." + filename[-45:]

                cov_str = f"{file_cov.line_coverage_percent:.1f}%"

                # Color based on threshold
                if file_cov.line_coverage_percent < self.threshold.line_coverage:
                    cov_str = self._colorize(cov_str, "red")
                elif file_cov.line_coverage_percent >= 90:
                    cov_str = self._colorize(cov_str, "green")

                lines.append(
                    f"{filename:<50} "
                    f"{file_cov.covered_lines}/{file_cov.total_lines:>5} "
                    f"{cov_str:>8}"
                )

            if len(report.files) > max_files:
                lines.append(f"  ... and {len(report.files) - max_files} more files")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def generate_file_coverage_view(
        self,
        file_coverage: FileCoverage,
        source_lines: Optional[list[str]] = None,
    ) -> str:
        """Generate a line-by-line coverage view for a file.

        Args:
            file_coverage: Coverage data for the file
            source_lines: Source code lines (reads from file if not provided)

        Returns:
            Annotated source code with coverage markers
        """
        # Load source if not provided
        if source_lines is None:
            try:
                with open(file_coverage.file_path) as f:
                    source_lines = f.read().splitlines()
            except Exception as e:
                logger.warning(f"Could not read source file: {e}")
                source_lines = []

        lines = []
        lines.append(f"Coverage for: {file_coverage.file_path}")
        lines.append(f"Coverage: {file_coverage.line_coverage_percent:.1f}%")
        lines.append("-" * 70)

        for i, source_line in enumerate(source_lines, 1):
            line_cov = file_coverage.lines.get(i)

            if line_cov is None:
                # Not executable line
                marker = " "
                color = None
            elif line_cov.status == CoverageStatus.COVERED:
                marker = "+"
                color = "green"
            elif line_cov.status == CoverageStatus.PARTIAL:
                marker = "~"
                color = "yellow"
            else:
                marker = "-"
                color = "red"

            line_num = f"{i:4d}"
            if color and self.use_colors:
                marker = self._colorize(marker, color)

            lines.append(f"{line_num} {marker} {source_line}")

        # Show uncovered ranges at the end
        uncovered_ranges = file_coverage.get_uncovered_ranges()
        if uncovered_ranges:
            lines.append("")
            lines.append("Uncovered ranges:")
            for start, end in uncovered_ranges:
                if start == end:
                    lines.append(f"  Line {start}")
                else:
                    lines.append(f"  Lines {start}-{end}")

        return "\n".join(lines)

    def generate_html_report(
        self,
        report: CoverageReport,
        output_dir: Path,
    ) -> Path:
        """Generate an HTML coverage report.

        Args:
            report: Coverage report to visualize
            output_dir: Directory to write HTML files

        Returns:
            Path to the index.html file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate index page
        index_html = self._generate_html_index(report)
        index_path = output_dir / "index.html"
        index_path.write_text(index_html)

        # Generate per-file pages
        for file_cov in report.files.values():
            file_html = self._generate_html_file(file_cov)
            safe_name = str(file_cov.file_path).replace("/", "_").replace("\\", "_")
            file_path = output_dir / f"{safe_name}.html"
            file_path.write_text(file_html)

        # Write CSS
        css_path = output_dir / "style.css"
        css_path.write_text(self._get_html_css())

        logger.info(f"HTML report written to: {index_path}")
        return index_path

    def generate_diff_report(self, diff: CoverageDiff) -> str:
        """Generate a coverage diff report.

        Args:
            diff: Coverage difference data

        Returns:
            Text report of coverage changes
        """
        lines = []

        lines.append("=" * 70)
        lines.append("COVERAGE DIFF REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Overall delta
        delta = diff.coverage_delta
        if delta > 0:
            delta_str = self._colorize(f"+{delta:.1f}%", "green")
        elif delta < 0:
            delta_str = self._colorize(f"{delta:.1f}%", "red")
        else:
            delta_str = "0.0%"

        lines.append(f"Coverage Change: {delta_str}")
        lines.append(
            f"Before: {diff.before.line_coverage_percent:.1f}%  "
            f"After: {diff.after.line_coverage_percent:.1f}%"
        )
        lines.append("")

        # New files
        new_files = set(diff.after.files.keys()) - set(diff.before.files.keys())
        if new_files:
            lines.append("NEW FILES")
            lines.append("-" * 40)
            for file_path in sorted(new_files):
                cov = diff.after.files[file_path]
                lines.append(f"  + {file_path}: {cov.line_coverage_percent:.1f}%")
            lines.append("")

        # Removed files
        removed_files = set(diff.before.files.keys()) - set(diff.after.files.keys())
        if removed_files:
            lines.append("REMOVED FILES")
            lines.append("-" * 40)
            for file_path in sorted(removed_files):
                lines.append(f"  - {file_path}")
            lines.append("")

        # Changed files
        common_files = set(diff.before.files.keys()) & set(diff.after.files.keys())
        changed_files = []
        for file_path in common_files:
            before_cov = diff.before.files[file_path]
            after_cov = diff.after.files[file_path]
            file_delta = after_cov.line_coverage_percent - before_cov.line_coverage_percent
            if abs(file_delta) > 0.1:  # More than 0.1% change
                changed_files.append((file_path, file_delta, after_cov))

        if changed_files:
            lines.append("CHANGED FILES")
            lines.append("-" * 40)
            for file_path, file_delta, after_cov in sorted(changed_files, key=lambda x: x[1]):
                if file_delta > 0:
                    delta_str = self._colorize(f"+{file_delta:.1f}%", "green")
                else:
                    delta_str = self._colorize(f"{file_delta:.1f}%", "red")
                lines.append(f"  {file_path}: {after_cov.line_coverage_percent:.1f}% ({delta_str})")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def generate_badge(
        self,
        report: CoverageReport,
        format: str = "svg",
    ) -> str:
        """Generate a coverage badge.

        Args:
            report: Coverage report
            format: Badge format (svg, markdown, text)

        Returns:
            Badge content
        """
        coverage = report.line_coverage_percent

        # Determine color
        if coverage >= 90:
            color = "brightgreen"
            hex_color = "#4c1"
        elif coverage >= 80:
            color = "green"
            hex_color = "#97ca00"
        elif coverage >= 70:
            color = "yellowgreen"
            hex_color = "#a4a61d"
        elif coverage >= 60:
            color = "yellow"
            hex_color = "#dfb317"
        else:
            color = "red"
            hex_color = "#e05d44"

        if format == "markdown":
            return (
                f"![Coverage](https://img.shields.io/badge/coverage-" f"{coverage:.0f}%25-{color})"
            )
        elif format == "text":
            return f"Coverage: {coverage:.1f}%"
        else:  # SVG
            return f"""<svg xmlns="http://www.w3.org/2000/svg" width="106" height="20">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <mask id="a">
    <rect width="106" height="20" rx="3" fill="#fff"/>
  </mask>
  <g mask="url(#a)">
    <path fill="#555" d="M0 0h63v20H0z"/>
    <path fill="{hex_color}" d="M63 0h43v20H63z"/>
    <path fill="url(#b)" d="M0 0h106v20H0z"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="31.5" y="15" fill="#010101" fill-opacity=".3">coverage</text>
    <text x="31.5" y="14">coverage</text>
    <text x="83.5" y="15" fill="#010101" fill-opacity=".3">{coverage:.0f}%</text>
    <text x="83.5" y="14">{coverage:.0f}%</text>
  </g>
</svg>"""

    def _colorize(self, text: str, color: str) -> str:
        """Apply ANSI color to text.

        Args:
            text: Text to colorize
            color: Color name

        Returns:
            Colorized text (or original if colors disabled)
        """
        if not self.use_colors:
            return text

        colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "reset": "\033[0m",
        }

        return f"{colors.get(color, '')}{text}{colors['reset']}"

    def _generate_html_index(self, report: CoverageReport) -> str:
        """Generate HTML index page."""
        passed, failures = self.threshold.check(report)

        rows = []
        for file_cov in sorted(
            report.files.values(),
            key=lambda f: f.line_coverage_percent,
        ):
            safe_name = str(file_cov.file_path).replace("/", "_").replace("\\", "_")
            cov_class = self._get_coverage_class(file_cov.line_coverage_percent)
            rows.append(f"""
                <tr class="{cov_class}">
                    <td><a href="{safe_name}.html">{html.escape(str(file_cov.file_path))}</a></td>
                    <td>{file_cov.covered_lines}/{file_cov.total_lines}</td>
                    <td>{file_cov.line_coverage_percent:.1f}%</td>
                    <td><div class="bar"><div class="fill" style="width: {file_cov.line_coverage_percent}%"></div></div></td>
                </tr>
            """)

        threshold_status = (
            '<span class="pass">PASS</span>' if passed else '<span class="fail">FAIL</span>'
        )

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Coverage Report</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <h1>Coverage Report</h1>

    <div class="summary">
        <h2>Summary</h2>
        <table class="summary-table">
            <tr><td>Files</td><td>{report.file_count}</td></tr>
            <tr><td>Lines</td><td>{report.covered_lines}/{report.total_lines} ({report.line_coverage_percent:.1f}%)</td></tr>
            <tr><td>Functions</td><td>{report.covered_functions}/{report.total_functions} ({report.function_coverage_percent:.1f}%)</td></tr>
            <tr><td>Branches</td><td>{report.covered_branches}/{report.total_branches} ({report.branch_coverage_percent:.1f}%)</td></tr>
            <tr><td>Threshold</td><td>{threshold_status}</td></tr>
        </table>
    </div>

    <div class="files">
        <h2>Files</h2>
        <table class="files-table">
            <thead>
                <tr>
                    <th>File</th>
                    <th>Lines</th>
                    <th>Coverage</th>
                    <th>Bar</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>

    <footer>
        <p>Generated by Victor Coverage</p>
    </footer>
</body>
</html>"""

    def _generate_html_file(self, file_cov: FileCoverage) -> str:
        """Generate HTML page for a single file."""
        # Try to read source
        source_lines = []
        try:
            with open(file_cov.file_path) as f:
                source_lines = f.read().splitlines()
        except Exception:
            pass

        line_rows = []
        for i, source_line in enumerate(source_lines, 1):
            line_data = file_cov.lines.get(i)

            if line_data is None:
                cov_class = ""
                hits = ""
            elif line_data.status == CoverageStatus.COVERED:
                cov_class = "covered"
                hits = str(line_data.hit_count)
            elif line_data.status == CoverageStatus.PARTIAL:
                cov_class = "partial"
                hits = str(line_data.hit_count)
            else:
                cov_class = "uncovered"
                hits = "0"

            line_rows.append(f"""
                <tr class="{cov_class}">
                    <td class="line-num">{i}</td>
                    <td class="hits">{hits}</td>
                    <td class="source"><pre>{html.escape(source_line)}</pre></td>
                </tr>
            """)

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Coverage: {html.escape(str(file_cov.file_path))}</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <h1>{html.escape(str(file_cov.file_path))}</h1>

    <div class="file-summary">
        <p>Coverage: {file_cov.line_coverage_percent:.1f}% ({file_cov.covered_lines}/{file_cov.total_lines} lines)</p>
        <p><a href="index.html">&larr; Back to index</a></p>
    </div>

    <table class="source-table">
        <tbody>
            {''.join(line_rows)}
        </tbody>
    </table>
</body>
</html>"""

    def _get_html_css(self) -> str:
        """Get CSS for HTML reports."""
        return """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 20px;
    background: #f5f5f5;
}

h1, h2 {
    color: #333;
}

.summary, .files {
    background: white;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

table {
    width: 100%;
    border-collapse: collapse;
}

th, td {
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid #eee;
}

th {
    background: #f0f0f0;
    font-weight: 600;
}

.bar {
    width: 100px;
    height: 10px;
    background: #ddd;
    border-radius: 5px;
    overflow: hidden;
}

.fill {
    height: 100%;
    background: #4caf50;
}

.high .fill { background: #4caf50; }
.medium .fill { background: #ffeb3b; }
.low .fill { background: #f44336; }

.pass { color: #4caf50; font-weight: bold; }
.fail { color: #f44336; font-weight: bold; }

.source-table {
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    font-size: 12px;
}

.source-table td {
    padding: 2px 8px;
    border: none;
}

.line-num {
    color: #999;
    text-align: right;
    width: 40px;
    user-select: none;
}

.hits {
    color: #666;
    text-align: right;
    width: 30px;
}

.source pre {
    margin: 0;
    white-space: pre-wrap;
}

tr.covered { background: #e8f5e9; }
tr.uncovered { background: #ffebee; }
tr.partial { background: #fff3e0; }

a { color: #1976d2; text-decoration: none; }
a:hover { text-decoration: underline; }

footer { color: #999; font-size: 12px; margin-top: 20px; }
"""

    def _get_coverage_class(self, coverage: float) -> str:
        """Get CSS class for coverage level."""
        if coverage >= 80:
            return "high"
        elif coverage >= 60:
            return "medium"
        return "low"
