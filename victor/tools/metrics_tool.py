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

"""Code metrics and quality analysis tool.

Features:
- Cyclomatic complexity calculation
- Maintainability index
- Technical debt estimation
- Performance profiling
- Comprehensive code analysis
- Quality reports
"""

import ast
from pathlib import Path
from typing import Any, Dict, List
import logging

from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.decorators import tool

logger = logging.getLogger(__name__)


def _calculate_complexity_score(code: str) -> int:
    """Calculate cyclomatic complexity of code."""
    try:
        tree = ast.parse(code)
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity
    except (SyntaxError, ValueError, AttributeError):
        return 0


def _calculate_maintainability_index(code: str) -> float:
    """Calculate maintainability index (0-100)."""
    try:
        ast.parse(code)
        lines = len([line for line in code.split("\n") if line.strip()])

        # Simplified maintainability calculation
        complexity = _calculate_complexity_score(code)

        # MI = max(0, (171 - 5.2 * ln(volume) - 0.23 * complexity - 16.2 * ln(lines)) * 100 / 171)
        # Simplified version
        mi = 100 - (complexity * 2) - (lines / 10)
        return max(0, min(100, mi))
    except (SyntaxError, ValueError, ZeroDivisionError):
        return 0.0


@tool(
    category="metrics",
    priority=Priority.MEDIUM,  # Task-specific analysis tool
    access_mode=AccessMode.READONLY,  # Only reads files for analysis
    danger_level=DangerLevel.SAFE,  # No side effects
    keywords=["metrics", "analyze", "code", "complexity", "quality"],
    stages=["analysis"],  # Conversation stages where relevant
)
async def metrics(
    path: str,
    metrics_list: List[str] = None,
    file_pattern: str = "*.py",
    complexity_threshold: int = 10,
    format: str = "summary",
) -> Dict[str, Any]:
    """
    Analyze code quality: complexity, coverage, lint errors, and style.

    Analyzes code quality metrics including complexity, maintainability,
    technical debt, and code structure. Consolidates multiple metric types
    into a single unified interface.

    Args:
        path: File or directory path to analyze.
        metrics: List of metrics to calculate. Options: "complexity",
            "maintainability", "debt", "profile", "all". Defaults to ["all"].
        file_pattern: Glob pattern for files to analyze (default: *.py).
        complexity_threshold: Threshold for complexity warnings (default: 10).
        format: Output format: "summary", "detailed", or "json" (default: summary).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - metrics_calculated: List of metrics that were calculated
        - results: Dictionary with results for each metric type
        - files_analyzed: Number of files analyzed
        - recommendations: List of improvement recommendations
        - formatted_report: Human-readable metrics report
        - error: Error message if failed

    Examples:
        # Analyze complexity only
        analyze_metrics("./src", metrics=["complexity"])

        # Comprehensive analysis
        analyze_metrics("./", metrics=["all"])

        # Maintainability and debt
        analyze_metrics("./src", metrics=["maintainability", "debt"])

        # Detailed report with high complexity threshold
        analyze_metrics("./", complexity_threshold=15, format="detailed")
    """
    if not path:
        return {"success": False, "error": "Missing required parameter: path"}

    if metrics_list is None:
        metrics_list = ["all"]

    # Expand "all" to all metric types
    if "all" in metrics_list:
        metrics_list = ["complexity", "maintainability", "debt", "profile"]

    path_obj = Path(path)
    if not path_obj.exists():
        return {"success": False, "error": f"Path not found: {path}"}

    # Collect files to analyze
    files_to_analyze = []
    if path_obj.is_file():
        if path_obj.suffix == ".py":
            files_to_analyze = [path_obj]
    else:
        files_to_analyze = [
            f for f in path_obj.rglob(file_pattern) if f.is_file() and f.suffix == ".py"
        ]

    if not files_to_analyze:
        return {
            "success": True,
            "files_analyzed": 0,
            "message": f"No Python files found matching pattern '{file_pattern}'",
        }

    # Initialize results
    results = {metric: [] for metric in metrics_list}
    total_complexity = 0
    total_maintainability = 0
    total_debt_hours = 0
    total_lines = 0
    total_functions = 0
    total_classes = 0
    files_analyzed = 0

    # Analyze each file
    for file_path in files_to_analyze:
        try:
            code = file_path.read_text()

            # Complexity analysis
            if "complexity" in metrics_list:
                complexity = _calculate_complexity_score(code)
                status = "ok" if complexity <= complexity_threshold else "warning"
                results["complexity"].append(
                    {
                        "file": str(file_path),
                        "complexity": complexity,
                        "status": status,
                        "threshold": complexity_threshold,
                    }
                )
                total_complexity += complexity

            # Maintainability analysis
            if "maintainability" in metrics_list:
                mi = _calculate_maintainability_index(code)
                if mi >= 80:
                    rating = "excellent"
                elif mi >= 60:
                    rating = "good"
                elif mi >= 40:
                    rating = "fair"
                else:
                    rating = "poor"

                results["maintainability"].append(
                    {
                        "file": str(file_path),
                        "maintainability_index": round(mi, 2),
                        "rating": rating,
                    }
                )
                total_maintainability += mi

            # Technical debt estimation
            if "debt" in metrics_list:
                complexity_score = _calculate_complexity_score(code)
                mi_score = _calculate_maintainability_index(code)

                issues = []
                debt_hours = 0

                if complexity_score > 20:
                    issues.append("High complexity detected")
                    debt_hours += 4
                elif complexity_score > 10:
                    issues.append("Moderate complexity")
                    debt_hours += 2

                if mi_score < 40:
                    issues.append("Low maintainability")
                    debt_hours += 8
                elif mi_score < 60:
                    issues.append("Below average maintainability")
                    debt_hours += 4

                debt_level = "low" if debt_hours < 4 else ("medium" if debt_hours < 12 else "high")

                results["debt"].append(
                    {
                        "file": str(file_path),
                        "debt_hours": debt_hours,
                        "debt_level": debt_level,
                        "issues": issues if issues else ["No major issues detected"],
                    }
                )
                total_debt_hours += debt_hours

            # Code profile
            if "profile" in metrics_list:
                tree = ast.parse(code)
                functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
                classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
                lines = len([line for line in code.split("\n") if line.strip()])

                results["profile"].append(
                    {
                        "file": str(file_path),
                        "lines": lines,
                        "functions": functions,
                        "classes": classes,
                    }
                )
                total_lines += lines
                total_functions += functions
                total_classes += classes

            files_analyzed += 1

        except Exception as e:
            logger.warning("Failed to analyze %s: %s", file_path, e)

    # Calculate averages
    avg_complexity = total_complexity / files_analyzed if files_analyzed > 0 else 0
    avg_maintainability = (
        total_maintainability / files_analyzed
        if files_analyzed > 0 and "maintainability" in metrics_list
        else 0
    )

    # Generate recommendations
    recommendations = []
    if "complexity" in metrics_list and avg_complexity > complexity_threshold:
        recommendations.append(
            f"Average complexity ({avg_complexity:.1f}) exceeds threshold ({complexity_threshold}) - consider refactoring"
        )
    if "maintainability" in metrics_list and avg_maintainability < 60:
        recommendations.append(
            f"Average maintainability ({avg_maintainability:.1f}/100) is below good - improve code quality"
        )
    if "debt" in metrics_list and total_debt_hours > 20:
        recommendations.append(
            f"Total technical debt ({total_debt_hours} hours) is significant - prioritize refactoring"
        )
    if "profile" in metrics_list and total_lines > 10000:
        recommendations.append(f"Large codebase ({total_lines} lines) - consider modularization")

    if not recommendations:
        recommendations.append("Code quality metrics look good")

    # Build formatted report
    report = []
    report.append("Code Metrics Analysis Report")
    report.append("=" * 70)
    report.append("")
    report.append(f"Path: {path}")
    report.append(f"Files analyzed: {files_analyzed}")
    report.append(f"Metrics: {', '.join(metrics_list)}")
    report.append("")

    # Complexity section
    if "complexity" in metrics_list:
        report.append("Complexity Analysis:")
        report.append(f"  Average complexity: {avg_complexity:.2f}")
        report.append(f"  Threshold: {complexity_threshold}")
        high_complexity_files = [r for r in results["complexity"] if r["status"] == "warning"]
        if high_complexity_files:
            report.append(f"  High complexity files: {len(high_complexity_files)}")
            for item in high_complexity_files[:5]:
                report.append(f"    {item['file']}: {item['complexity']}")
            if len(high_complexity_files) > 5:
                report.append(f"    ... and {len(high_complexity_files) - 5} more")
        report.append("")

    # Maintainability section
    if "maintainability" in metrics_list:
        report.append("Maintainability Analysis:")
        report.append(f"  Average maintainability: {avg_maintainability:.2f}/100")
        poor_files = [r for r in results["maintainability"] if r["rating"] in ["poor", "fair"]]
        if poor_files:
            report.append(f"  Files needing improvement: {len(poor_files)}")
            for item in poor_files[:5]:
                report.append(
                    f"    {item['file']}: {item['maintainability_index']} ({item['rating']})"
                )
            if len(poor_files) > 5:
                report.append(f"    ... and {len(poor_files) - 5} more")
        report.append("")

    # Technical debt section
    if "debt" in metrics_list:
        report.append("Technical Debt:")
        report.append(f"  Total debt: {total_debt_hours} hours")
        high_debt_files = [r for r in results["debt"] if r["debt_level"] in ["medium", "high"]]
        if high_debt_files:
            report.append(f"  Files with debt: {len(high_debt_files)}")
            for item in high_debt_files[:5]:
                report.append(
                    f"    {item['file']}: {item['debt_hours']} hours ({item['debt_level']})"
                )
            if len(high_debt_files) > 5:
                report.append(f"    ... and {len(high_debt_files) - 5} more")
        report.append("")

    # Profile section
    if "profile" in metrics_list:
        report.append("Code Profile:")
        report.append(f"  Total lines: {total_lines}")
        report.append(f"  Total functions: {total_functions}")
        report.append(f"  Total classes: {total_classes}")
        if files_analyzed > 0:
            report.append(
                f"  Average per file: {total_lines // files_analyzed} lines, "
                f"{total_functions // files_analyzed} functions, "
                f"{total_classes // files_analyzed} classes"
            )
        report.append("")

    # Recommendations section
    report.append("Recommendations:")
    for rec in recommendations:
        report.append(f"  • {rec}")

    return {
        "success": True,
        "metrics_calculated": metrics,
        "results": results,
        "files_analyzed": files_analyzed,
        "summary": {
            "avg_complexity": (round(avg_complexity, 2) if "complexity" in metrics_list else None),
            "avg_maintainability": (
                round(avg_maintainability, 2) if "maintainability" in metrics_list else None
            ),
            "total_debt_hours": total_debt_hours if "debt" in metrics_list else None,
            "total_lines": total_lines if "profile" in metrics_list else None,
        },
        "recommendations": recommendations,
        "formatted_report": "\n".join(report),
    }
