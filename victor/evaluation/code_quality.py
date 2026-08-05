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

"""Code quality analyzer for benchmark evaluation.

Analyzes generated code for quality metrics including:
- Syntax validation
- Linting (using ruff/pylint)
- Complexity metrics (cyclomatic, cognitive)
- Maintainability index
- Type coverage (for typed code)
"""

import ast
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from victor.evaluation.protocol import CodeQualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class LintResult:
    """Result from a linting tool."""

    errors: int = 0
    warnings: int = 0
    info: int = 0
    issues: list[dict] = field(default_factory=list)


class CodeQualityAnalyzer:
    """Analyzes code quality for generated solutions.

    Uses multiple tools to assess code quality:
    - ast module for syntax validation
    - ruff for fast linting
    - radon for complexity metrics (optional)

    Example:
        analyzer = CodeQualityAnalyzer()
        metrics = await analyzer.analyze(code, language="python")
        print(f"Quality score: {metrics.get_overall_score()}")
    """

    def __init__(
        self,
        use_ruff: bool = True,
        use_radon: bool = True,
        style_guide: str = "pep8",
    ):
        """Initialize the analyzer.

        Args:
            use_ruff: Whether to use ruff for linting
            use_radon: Whether to use radon for complexity
            style_guide: Style guide to follow (pep8, google)
        """
        self.use_ruff = use_ruff
        self.use_radon = use_radon
        self.style_guide = style_guide
        self._check_tools()

    def _check_tools(self) -> None:
        """Check availability of analysis tools."""
        self._has_ruff = self._tool_available("ruff")
        self._has_radon = self._tool_available("radon")

        if self.use_ruff and not self._has_ruff:
            logger.warning("ruff not available, linting will be limited")
        if self.use_radon and not self._has_radon:
            logger.warning("radon not available, complexity metrics limited")

    def _tool_available(self, tool: str) -> bool:
        """Check if a tool is available."""
        try:
            subprocess.run(
                [tool, "--version"],
                capture_output=True,
                timeout=5,
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    async def analyze(
        self,
        code: str,
        language: str = "python",
        filename: Optional[str] = None,
    ) -> CodeQualityMetrics:
        """Analyze code quality.

        Args:
            code: The code to analyze
            language: Programming language
            filename: Optional filename for context

        Returns:
            CodeQualityMetrics with all quality scores
        """
        metrics = CodeQualityMetrics()

        if not code or not code.strip():
            metrics.syntax_valid = False
            return metrics

        if language == "python":
            metrics = await self._analyze_python(code, filename)
        else:
            # Basic analysis for other languages
            metrics = self._basic_analysis(code)

        return metrics

    async def _analyze_python(
        self,
        code: str,
        filename: Optional[str] = None,
    ) -> CodeQualityMetrics:
        """Analyze Python code quality."""
        metrics = CodeQualityMetrics()

        # 1. Syntax validation
        metrics.syntax_valid = self._validate_syntax(code)
        if not metrics.syntax_valid:
            return metrics

        # 2. Basic code metrics
        metrics.lines_of_code = self._count_lines(code)
        metrics.functions_count = self._count_functions(code)
        metrics.classes_count = self._count_classes(code)
        metrics.duplicate_lines = self._detect_duplicates(code)

        # 3. Linting
        if self.use_ruff and self._has_ruff:
            lint_result = await self._run_ruff(code, filename)
            metrics.lint_errors = lint_result.errors
            metrics.lint_warnings = lint_result.warnings
            # Calculate style score based on issues per line
            total_issues = lint_result.errors + lint_result.warnings
            if metrics.lines_of_code > 0:
                issues_per_line = total_issues / metrics.lines_of_code
                metrics.style_score = max(0.0, 1.0 - (issues_per_line * 0.5))
            else:
                metrics.style_score = 1.0

        # 4. Complexity metrics
        if self.use_radon and self._has_radon:
            complexity = await self._run_radon(code)
            metrics.cyclomatic_complexity = complexity.get("cyclomatic", 0.0)
            metrics.maintainability_index = complexity.get("maintainability", 100.0)
        else:
            # Fallback: estimate complexity from code structure
            metrics.cyclomatic_complexity = self._estimate_complexity(code)
            metrics.maintainability_index = self._estimate_maintainability(code, metrics)

        # 5. Type coverage (check for type hints)
        metrics.type_coverage = self._analyze_type_coverage(code)

        return metrics

    def _validate_syntax(self, code: str) -> bool:
        """Check if code has valid Python syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _count_lines(self, code: str) -> int:
        """Count non-empty, non-comment lines."""
        lines = 0
        for line in code.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                lines += 1
        return lines

    def _count_functions(self, code: str) -> int:
        """Count function definitions."""
        try:
            tree = ast.parse(code)
            return sum(
                1
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
        except SyntaxError:
            return 0

    def _count_classes(self, code: str) -> int:
        """Count class definitions."""
        try:
            tree = ast.parse(code)
            return sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        except SyntaxError:
            return 0

    def _detect_duplicates(self, code: str) -> int:
        """Detect duplicate lines (simple heuristic)."""
        lines = [line.strip() for line in code.split("\n") if line.strip()]
        seen = set()
        duplicates = 0
        for line in lines:
            if len(line) > 10 and line in seen:  # Skip short lines
                duplicates += 1
            seen.add(line)
        return duplicates

    async def _run_ruff(
        self,
        code: str,
        filename: Optional[str] = None,
    ) -> LintResult:
        """Run ruff linter on code."""
        result = LintResult()

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            proc = subprocess.run(
                [
                    "ruff",
                    "check",
                    temp_path,
                    "--output-format=json",
                    "--quiet",
                ],
                capture_output=True,
                timeout=30,
            )

            if proc.stdout:
                import json

                try:
                    issues = json.loads(proc.stdout.decode())
                    for issue in issues:
                        severity = issue.get("code", "")
                        if severity.startswith("E") or severity.startswith("F"):
                            result.errors += 1
                        else:
                            result.warnings += 1
                        result.issues.append(issue)
                except json.JSONDecodeError:
                    pass

        except subprocess.TimeoutExpired:
            logger.warning("ruff timed out")
        except Exception as e:
            logger.warning(f"ruff failed: {e}")
        finally:
            Path(temp_path).unlink(missing_ok=True)

        return result

    async def _run_radon(self, code: str) -> dict:
        """Run radon for complexity metrics."""
        result = {
            "cyclomatic": 0.0,
            "maintainability": 100.0,
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            # Get cyclomatic complexity
            proc = subprocess.run(
                ["radon", "cc", temp_path, "-a", "-s"],
                capture_output=True,
                timeout=30,
            )

            if proc.stdout:
                output = proc.stdout.decode()
                # Parse average complexity
                avg_match = re.search(r"Average complexity: [A-F] \((\d+\.\d+)\)", output)
                if avg_match:
                    result["cyclomatic"] = float(avg_match.group(1))

            # Get maintainability index
            proc = subprocess.run(
                ["radon", "mi", temp_path, "-s"],
                capture_output=True,
                timeout=30,
            )

            if proc.stdout:
                output = proc.stdout.decode()
                # Parse MI score
                mi_match = re.search(r"- [A-F] \((\d+\.\d+)\)", output)
                if mi_match:
                    result["maintainability"] = float(mi_match.group(1))

        except subprocess.TimeoutExpired:
            logger.warning("radon timed out")
        except Exception as e:
            logger.warning(f"radon failed: {e}")
        finally:
            Path(temp_path).unlink(missing_ok=True)

        return result

    def _estimate_complexity(self, code: str) -> float:
        """Estimate cyclomatic complexity without radon.

        Uses AST to count decision points:
        - if/elif/else
        - for/while loops
        - try/except
        - and/or operators
        - ternary expressions
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0.0

        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.IfExp)):
                complexity += 1
            elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                # Each and/or adds 1
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                complexity += 1

        # Normalize by number of functions
        functions = self._count_functions(code)
        if functions > 0:
            return complexity / functions
        return float(complexity)

    def _estimate_maintainability(
        self,
        code: str,
        metrics: CodeQualityMetrics,
    ) -> float:
        """Estimate maintainability index without radon.

        Based on:
        - Halstead volume (approximated)
        - Cyclomatic complexity
        - Lines of code
        - Comment ratio
        """
        # Start with base score
        mi = 100.0

        # Penalize based on complexity
        mi -= metrics.cyclomatic_complexity * 2

        # Penalize based on length (longer code harder to maintain)
        if metrics.lines_of_code > 100:
            mi -= (metrics.lines_of_code - 100) * 0.1

        # Penalize for duplicates
        if metrics.lines_of_code > 0:
            duplicate_ratio = metrics.duplicate_lines / metrics.lines_of_code
            mi -= duplicate_ratio * 20

        # Bonus for type hints
        mi += metrics.type_coverage * 10

        # Bonus for docstrings
        docstring_bonus = self._count_docstrings(code) * 2
        mi = min(mi + docstring_bonus, 100.0)

        return max(0.0, min(100.0, mi))

    def _count_docstrings(self, code: str) -> int:
        """Count docstrings in code."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0

        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                if (
                    ast.get_docstring(node)
                    and len(node.body) > 0
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                ):
                    count += 1

        return count

    def _analyze_type_coverage(self, code: str) -> float:
        """Analyze type hint coverage in code.

        Returns ratio of typed parameters/returns to total.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0.0

        total_annotations = 0
        present_annotations = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check return type
                total_annotations += 1
                if node.returns is not None:
                    present_annotations += 1

                # Check arguments
                for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                    if arg.arg not in ("self", "cls"):
                        total_annotations += 1
                        if arg.annotation is not None:
                            present_annotations += 1

        if total_annotations == 0:
            return 0.0

        return present_annotations / total_annotations

    def _basic_analysis(self, code: str) -> CodeQualityMetrics:
        """Basic analysis for non-Python code."""
        metrics = CodeQualityMetrics()
        metrics.syntax_valid = True  # Assume valid
        metrics.lines_of_code = len([ln for ln in code.split("\n") if ln.strip()])
        return metrics


class BatchCodeAnalyzer:
    """Batch analyzer for evaluating multiple code samples."""

    def __init__(self, analyzer: Optional[CodeQualityAnalyzer] = None):
        """Initialize batch analyzer.

        Args:
            analyzer: CodeQualityAnalyzer instance (creates default if None)
        """
        self.analyzer = analyzer or CodeQualityAnalyzer()

    async def analyze_batch(
        self,
        samples: list[tuple[str, str]],  # (code, language)
    ) -> list[CodeQualityMetrics]:
        """Analyze a batch of code samples.

        Args:
            samples: List of (code, language) tuples

        Returns:
            List of CodeQualityMetrics for each sample
        """
        results = []
        for code, language in samples:
            metrics = await self.analyzer.analyze(code, language)
            results.append(metrics)
        return results

    def aggregate_metrics(
        self,
        metrics_list: list[CodeQualityMetrics],
    ) -> dict:
        """Aggregate metrics across multiple samples.

        Args:
            metrics_list: List of CodeQualityMetrics

        Returns:
            Dict with aggregated statistics
        """
        if not metrics_list:
            return {}

        total = len(metrics_list)
        syntax_valid = sum(1 for m in metrics_list if m.syntax_valid)
        avg_quality = sum(m.get_overall_score() for m in metrics_list) / total
        avg_complexity = sum(m.cyclomatic_complexity for m in metrics_list) / total
        avg_maintainability = sum(m.maintainability_index for m in metrics_list) / total
        avg_type_coverage = sum(m.type_coverage for m in metrics_list) / total

        return {
            "total_samples": total,
            "syntax_valid_count": syntax_valid,
            "syntax_valid_rate": syntax_valid / total,
            "avg_quality_score": avg_quality,
            "avg_cyclomatic_complexity": avg_complexity,
            "avg_maintainability_index": avg_maintainability,
            "avg_type_coverage": avg_type_coverage,
            "avg_lint_errors": sum(m.lint_errors for m in metrics_list) / total,
            "avg_lint_warnings": sum(m.lint_warnings for m in metrics_list) / total,
        }
