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

"""Code analyzers for automated review.

Provides various analyzers for different aspects of code quality.
"""

import ast
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

from victor_coding.review.protocol import (
    ComplexityMetrics,
    ReviewCategory,
    ReviewFinding,
    ReviewRule,
    SourceLocation,
)

logger = logging.getLogger(__name__)


class BaseAnalyzer(ABC):
    """Base class for code analyzers.

    Implements Strategy pattern for different analysis types.
    """

    @property
    @abstractmethod
    def category(self) -> ReviewCategory:
        """Get the category this analyzer handles."""
        pass

    @abstractmethod
    def analyze(
        self,
        source: str,
        file_path: Path,
        rules: list[ReviewRule],
    ) -> list[ReviewFinding]:
        """Analyze source code.

        Args:
            source: Source code to analyze
            file_path: Path to the file
            rules: Rules to apply

        Returns:
            List of findings
        """
        pass

    def _create_finding(
        self,
        rule: ReviewRule,
        message: str,
        file_path: Path,
        line: int,
        column: int = 0,
        snippet: str = "",
        suggestion: str = "",
        fix_code: str = "",
    ) -> ReviewFinding:
        """Helper to create a finding."""
        return ReviewFinding(
            rule_id=rule.id,
            message=message,
            severity=rule.severity,
            category=rule.category,
            location=SourceLocation(
                file_path=file_path,
                start_line=line,
                start_column=column,
            ),
            code_snippet=snippet,
            suggestion=suggestion,
            fix_available=bool(fix_code),
            fix_code=fix_code,
        )


class ComplexityAnalyzer(BaseAnalyzer):
    """Analyzes code complexity metrics."""

    @property
    def category(self) -> ReviewCategory:
        return ReviewCategory.COMPLEXITY

    def analyze(
        self,
        source: str,
        file_path: Path,
        rules: list[ReviewRule],
    ) -> list[ReviewFinding]:
        """Analyze code complexity."""
        findings = []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return findings

        lines = source.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_findings = self._analyze_function(node, file_path, rules, lines)
                findings.extend(func_findings)

            elif isinstance(node, ast.ClassDef):
                class_findings = self._analyze_class(node, file_path, rules, lines)
                findings.extend(class_findings)

        return findings

    def _analyze_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: Path,
        rules: list[ReviewRule],
        lines: list[str],
    ) -> list[ReviewFinding]:
        """Analyze a function for complexity issues."""
        findings = []

        # Calculate cyclomatic complexity
        cc = self._cyclomatic_complexity(node)

        # Calculate function length
        func_lines = (node.end_lineno or node.lineno) - node.lineno + 1

        # Calculate nesting depth
        max_depth = self._max_nesting_depth(node)

        # Check rules
        for rule in rules:
            if rule.id == "complexity-cyclomatic":
                threshold = rule.parameters.get("max", 10)
                if cc > threshold:
                    findings.append(
                        self._create_finding(
                            rule=rule,
                            message=f"Function '{node.name}' has cyclomatic complexity {cc} (max: {threshold})",
                            file_path=file_path,
                            line=node.lineno,
                            snippet=(lines[node.lineno - 1] if node.lineno <= len(lines) else ""),
                            suggestion="Consider breaking this function into smaller functions",
                        )
                    )

            elif rule.id == "complexity-function-length":
                threshold = rule.parameters.get("max", 50)
                if func_lines > threshold:
                    findings.append(
                        self._create_finding(
                            rule=rule,
                            message=f"Function '{node.name}' is {func_lines} lines (max: {threshold})",
                            file_path=file_path,
                            line=node.lineno,
                            suggestion="Consider extracting logic into helper functions",
                        )
                    )

            elif rule.id == "complexity-nesting":
                threshold = rule.parameters.get("max", 4)
                if max_depth > threshold:
                    findings.append(
                        self._create_finding(
                            rule=rule,
                            message=f"Function '{node.name}' has nesting depth {max_depth} (max: {threshold})",
                            file_path=file_path,
                            line=node.lineno,
                            suggestion="Consider using early returns or extracting nested logic",
                        )
                    )

            elif rule.id == "complexity-parameters":
                threshold = rule.parameters.get("max", 5)
                num_params = len([a for a in node.args.args if a.arg not in ("self", "cls")])
                if num_params > threshold:
                    findings.append(
                        self._create_finding(
                            rule=rule,
                            message=f"Function '{node.name}' has {num_params} parameters (max: {threshold})",
                            file_path=file_path,
                            line=node.lineno,
                            suggestion="Consider using a configuration object or keyword arguments",
                        )
                    )

        return findings

    def _analyze_class(
        self,
        node: ast.ClassDef,
        file_path: Path,
        rules: list[ReviewRule],
        lines: list[str],
    ) -> list[ReviewFinding]:
        """Analyze a class for complexity issues."""
        findings = []

        # Count methods
        methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

        for rule in rules:
            if rule.id == "complexity-class-methods":
                threshold = rule.parameters.get("max", 20)
                if len(methods) > threshold:
                    findings.append(
                        self._create_finding(
                            rule=rule,
                            message=f"Class '{node.name}' has {len(methods)} methods (max: {threshold})",
                            file_path=file_path,
                            line=node.lineno,
                            suggestion="Consider splitting into smaller classes using composition",
                        )
                    )

        return findings

    def _cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                complexity += len(child.ifs) + 1

        return complexity

    def _max_nesting_depth(self, node: ast.AST, depth: int = 0) -> int:
        """Calculate maximum nesting depth."""
        max_depth = depth

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                child_depth = self._max_nesting_depth(child, depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._max_nesting_depth(child, depth)
                max_depth = max(max_depth, child_depth)

        return max_depth

    def calculate_metrics(self, source: str) -> ComplexityMetrics:
        """Calculate comprehensive complexity metrics."""
        metrics = ComplexityMetrics()

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return metrics

        lines = source.split("\n")
        metrics.lines_of_code = len(
            [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        )
        metrics.lines_of_comment = len([ln for ln in lines if ln.strip().startswith("#")])

        functions = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node)
                cc = self._cyclomatic_complexity(node)
                metrics.cyclomatic_complexity = max(metrics.cyclomatic_complexity, cc)
                depth = self._max_nesting_depth(node)
                metrics.max_nesting_depth = max(metrics.max_nesting_depth, depth)
            elif isinstance(node, ast.ClassDef):
                classes.append(node)

        metrics.number_of_functions = len(functions)
        metrics.number_of_classes = len(classes)

        if functions:
            total_lines = sum((f.end_lineno or f.lineno) - f.lineno + 1 for f in functions)
            metrics.average_function_length = total_lines / len(functions)

        return metrics


class NamingAnalyzer(BaseAnalyzer):
    """Analyzes naming conventions."""

    @property
    def category(self) -> ReviewCategory:
        return ReviewCategory.NAMING

    # PEP 8 patterns
    SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")
    PASCAL_CASE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
    UPPER_CASE = re.compile(r"^[A-Z][A-Z0-9_]*$")

    def analyze(
        self,
        source: str,
        file_path: Path,
        rules: list[ReviewRule],
    ) -> list[ReviewFinding]:
        """Analyze naming conventions."""
        findings = []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return findings

        lines = source.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                findings.extend(self._check_class_name(node, file_path, rules, lines))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                findings.extend(self._check_function_name(node, file_path, rules, lines))

            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                findings.extend(self._check_variable_name(node, file_path, rules, lines))

        return findings

    def _check_class_name(
        self,
        node: ast.ClassDef,
        file_path: Path,
        rules: list[ReviewRule],
        lines: list[str],
    ) -> list[ReviewFinding]:
        """Check class naming."""
        findings = []

        for rule in rules:
            if rule.id == "naming-class-case":
                if not self.PASCAL_CASE.match(node.name):
                    findings.append(
                        self._create_finding(
                            rule=rule,
                            message=f"Class '{node.name}' should use PascalCase",
                            file_path=file_path,
                            line=node.lineno,
                            snippet=(lines[node.lineno - 1] if node.lineno <= len(lines) else ""),
                            suggestion=f"Rename to '{self._to_pascal_case(node.name)}'",
                        )
                    )

        return findings

    def _check_function_name(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: Path,
        rules: list[ReviewRule],
        lines: list[str],
    ) -> list[ReviewFinding]:
        """Check function naming."""
        findings = []

        # Skip dunder methods
        if node.name.startswith("__") and node.name.endswith("__"):
            return findings

        for rule in rules:
            if rule.id == "naming-function-case":
                if not self.SNAKE_CASE.match(node.name):
                    findings.append(
                        self._create_finding(
                            rule=rule,
                            message=f"Function '{node.name}' should use snake_case",
                            file_path=file_path,
                            line=node.lineno,
                            snippet=(lines[node.lineno - 1] if node.lineno <= len(lines) else ""),
                            suggestion=f"Rename to '{self._to_snake_case(node.name)}'",
                        )
                    )

            elif rule.id == "naming-function-length":
                min_len = rule.parameters.get("min", 2)
                max_len = rule.parameters.get("max", 30)
                if len(node.name) < min_len:
                    findings.append(
                        self._create_finding(
                            rule=rule,
                            message=f"Function name '{node.name}' is too short (min: {min_len})",
                            file_path=file_path,
                            line=node.lineno,
                            suggestion="Use a more descriptive name",
                        )
                    )
                elif len(node.name) > max_len:
                    findings.append(
                        self._create_finding(
                            rule=rule,
                            message=f"Function name '{node.name}' is too long (max: {max_len})",
                            file_path=file_path,
                            line=node.lineno,
                            suggestion="Consider a shorter, more concise name",
                        )
                    )

        return findings

    def _check_variable_name(
        self,
        node: ast.Name,
        file_path: Path,
        rules: list[ReviewRule],
        lines: list[str],
    ) -> list[ReviewFinding]:
        """Check variable naming."""
        findings = []
        name = node.id

        # Skip private/protected
        if name.startswith("_"):
            return findings

        for rule in rules:
            if rule.id == "naming-variable-case":
                # Check if it's a constant (all upper) or variable (snake_case)
                if not (self.SNAKE_CASE.match(name) or self.UPPER_CASE.match(name)):
                    findings.append(
                        self._create_finding(
                            rule=rule,
                            message=f"Variable '{name}' should use snake_case or UPPER_CASE",
                            file_path=file_path,
                            line=node.lineno,
                            snippet=(lines[node.lineno - 1] if node.lineno <= len(lines) else ""),
                        )
                    )

            elif rule.id == "naming-single-char":
                allowed = rule.parameters.get("allowed", ["i", "j", "k", "x", "y", "z", "_"])
                if len(name) == 1 and name not in allowed:
                    findings.append(
                        self._create_finding(
                            rule=rule,
                            message=f"Single character variable '{name}' should have a descriptive name",
                            file_path=file_path,
                            line=node.lineno,
                        )
                    )

        return findings

    def _to_snake_case(self, name: str) -> str:
        """Convert to snake_case."""
        result = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        result = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", result)
        return result.lower()

    def _to_pascal_case(self, name: str) -> str:
        """Convert to PascalCase."""
        parts = re.split(r"[_\s]", name)
        return "".join(p.capitalize() for p in parts)


class DocumentationAnalyzer(BaseAnalyzer):
    """Analyzes documentation completeness."""

    @property
    def category(self) -> ReviewCategory:
        return ReviewCategory.DOCUMENTATION

    def analyze(
        self,
        source: str,
        file_path: Path,
        rules: list[ReviewRule],
    ) -> list[ReviewFinding]:
        """Analyze documentation."""
        findings = []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return findings

        lines = source.split("\n")

        # Check module docstring
        if rules and any(r.id == "doc-module-docstring" for r in rules):
            if not ast.get_docstring(tree):
                rule = next(r for r in rules if r.id == "doc-module-docstring")
                findings.append(
                    self._create_finding(
                        rule=rule,
                        message="Module is missing a docstring",
                        file_path=file_path,
                        line=1,
                        suggestion='Add a module docstring at the top: """Module description."""',
                    )
                )

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                findings.extend(self._check_class_docs(node, file_path, rules, lines))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                findings.extend(self._check_function_docs(node, file_path, rules, lines))

        return findings

    def _check_class_docs(
        self,
        node: ast.ClassDef,
        file_path: Path,
        rules: list[ReviewRule],
        lines: list[str],
    ) -> list[ReviewFinding]:
        """Check class documentation."""
        findings = []

        docstring = ast.get_docstring(node)

        for rule in rules:
            if rule.id == "doc-class-docstring":
                if not node.name.startswith("_") and not docstring:
                    findings.append(
                        self._create_finding(
                            rule=rule,
                            message=f"Class '{node.name}' is missing a docstring",
                            file_path=file_path,
                            line=node.lineno,
                            suggestion='Add a class docstring: """Class description."""',
                        )
                    )

        return findings

    def _check_function_docs(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: Path,
        rules: list[ReviewRule],
        lines: list[str],
    ) -> list[ReviewFinding]:
        """Check function documentation."""
        findings = []

        # Skip private functions and dunder methods
        if node.name.startswith("_"):
            return findings

        docstring = ast.get_docstring(node)

        for rule in rules:
            if rule.id == "doc-function-docstring":
                if not docstring:
                    findings.append(
                        self._create_finding(
                            rule=rule,
                            message=f"Function '{node.name}' is missing a docstring",
                            file_path=file_path,
                            line=node.lineno,
                            suggestion="Add a docstring describing the function purpose",
                        )
                    )

            elif rule.id == "doc-function-params" and docstring:
                # Check if parameters are documented
                params = [a.arg for a in node.args.args if a.arg not in ("self", "cls")]
                for param in params:
                    if param not in docstring:
                        findings.append(
                            self._create_finding(
                                rule=rule,
                                message=f"Parameter '{param}' is not documented in '{node.name}'",
                                file_path=file_path,
                                line=node.lineno,
                                suggestion=f"Add documentation for parameter '{param}'",
                            )
                        )

            elif rule.id == "doc-function-return" and docstring:
                if node.returns and "return" not in docstring.lower():
                    findings.append(
                        self._create_finding(
                            rule=rule,
                            message=f"Return value not documented in '{node.name}'",
                            file_path=file_path,
                            line=node.lineno,
                            suggestion="Add a Returns section to the docstring",
                        )
                    )

        return findings


class SecurityAnalyzer(BaseAnalyzer):
    """Analyzes code for security issues."""

    @property
    def category(self) -> ReviewCategory:
        return ReviewCategory.SECURITY

    # Common dangerous patterns
    DANGEROUS_CALLS = {
        "eval": "CWE-95: Use of eval() can lead to code injection",
        "exec": "CWE-95: Use of exec() can lead to code injection",
        "compile": "CWE-95: Use of compile() with user input is dangerous",
        "pickle.loads": "CWE-502: Deserializing untrusted data can lead to code execution",
        "yaml.load": "CWE-502: Use yaml.safe_load() instead",
        "subprocess.call": "CWE-78: Use subprocess with shell=False and validate input",
        "os.system": "CWE-78: Use subprocess instead of os.system",
        "os.popen": "CWE-78: Use subprocess instead of os.popen",
    }

    SQL_INJECTION_PATTERNS = [
        r'execute\s*\(\s*["\'].*%.*["\']',
        r'execute\s*\(\s*f["\']',
        r'execute\s*\(\s*["\'].*\+',
    ]

    def analyze(
        self,
        source: str,
        file_path: Path,
        rules: list[ReviewRule],
    ) -> list[ReviewFinding]:
        """Analyze for security issues."""
        findings = []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return findings

        lines = source.split("\n")

        # Check for dangerous function calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                findings.extend(self._check_dangerous_call(node, file_path, rules, lines))

        # Check for SQL injection patterns
        findings.extend(self._check_sql_injection(source, file_path, rules))

        # Check for hardcoded secrets
        findings.extend(self._check_hardcoded_secrets(source, file_path, rules))

        return findings

    def _check_dangerous_call(
        self,
        node: ast.Call,
        file_path: Path,
        rules: list[ReviewRule],
        lines: list[str],
    ) -> list[ReviewFinding]:
        """Check for dangerous function calls."""
        findings = []

        # Get function name
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                func_name = f"{node.func.value.id}.{node.func.attr}"
            else:
                func_name = node.func.attr

        if not func_name:
            return findings

        for rule in rules:
            if rule.id == "security-dangerous-calls":
                if func_name in self.DANGEROUS_CALLS:
                    findings.append(
                        self._create_finding(
                            rule=rule,
                            message=self.DANGEROUS_CALLS[func_name],
                            file_path=file_path,
                            line=node.lineno,
                            snippet=(lines[node.lineno - 1] if node.lineno <= len(lines) else ""),
                        )
                    )

        return findings

    def _check_sql_injection(
        self,
        source: str,
        file_path: Path,
        rules: list[ReviewRule],
    ) -> list[ReviewFinding]:
        """Check for SQL injection vulnerabilities."""
        findings = []

        for rule in rules:
            if rule.id != "security-sql-injection":
                continue

            lines = source.split("\n")
            for i, line in enumerate(lines, 1):
                for pattern in self.SQL_INJECTION_PATTERNS:
                    if re.search(pattern, line):
                        findings.append(
                            self._create_finding(
                                rule=rule,
                                message="CWE-89: Potential SQL injection vulnerability",
                                file_path=file_path,
                                line=i,
                                snippet=line.strip(),
                                suggestion="Use parameterized queries instead of string formatting",
                            )
                        )

        return findings

    def _check_hardcoded_secrets(
        self,
        source: str,
        file_path: Path,
        rules: list[ReviewRule],
    ) -> list[ReviewFinding]:
        """Check for hardcoded secrets."""
        findings = []

        SECRET_PATTERNS = [
            (r'(?i)password\s*=\s*["\'][^"\']+["\']', "hardcoded password"),
            (r'(?i)api_?key\s*=\s*["\'][^"\']+["\']', "hardcoded API key"),
            (r'(?i)secret\s*=\s*["\'][^"\']+["\']', "hardcoded secret"),
            (r'(?i)token\s*=\s*["\'][^"\']+["\']', "hardcoded token"),
            (r'["\'][A-Za-z0-9+/]{40,}={0,2}["\']', "potential base64 encoded secret"),
        ]

        for rule in rules:
            if rule.id != "security-hardcoded-secrets":
                continue

            lines = source.split("\n")
            for i, line in enumerate(lines, 1):
                for pattern, desc in SECRET_PATTERNS:
                    if re.search(pattern, line):
                        findings.append(
                            self._create_finding(
                                rule=rule,
                                message=f"CWE-798: Found {desc}",
                                file_path=file_path,
                                line=i,
                                snippet=line.strip()[:50] + "...",
                                suggestion="Use environment variables or a secrets manager",
                            )
                        )

        return findings


class BestPracticesAnalyzer(BaseAnalyzer):
    """Analyzes code for best practices."""

    @property
    def category(self) -> ReviewCategory:
        return ReviewCategory.BEST_PRACTICES

    def analyze(
        self,
        source: str,
        file_path: Path,
        rules: list[ReviewRule],
    ) -> list[ReviewFinding]:
        """Analyze for best practices."""
        findings = []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return findings

        lines = source.split("\n")

        for node in ast.walk(tree):
            # Check exception handling
            if isinstance(node, ast.ExceptHandler):
                findings.extend(self._check_exception(node, file_path, rules, lines))

            # Check imports
            elif isinstance(node, ast.Import):
                findings.extend(self._check_import(node, file_path, rules, lines))

            elif isinstance(node, ast.ImportFrom):
                findings.extend(self._check_import_from(node, file_path, rules, lines))

            # Check comparisons
            elif isinstance(node, ast.Compare):
                findings.extend(self._check_comparison(node, file_path, rules, lines))

        return findings

    def _check_exception(
        self,
        node: ast.ExceptHandler,
        file_path: Path,
        rules: list[ReviewRule],
        lines: list[str],
    ) -> list[ReviewFinding]:
        """Check exception handling."""
        findings = []

        for rule in rules:
            if rule.id == "bp-bare-except":
                if node.type is None:
                    findings.append(
                        self._create_finding(
                            rule=rule,
                            message="Bare except clause catches all exceptions including SystemExit",
                            file_path=file_path,
                            line=node.lineno,
                            suggestion="Catch specific exceptions: except Exception:",
                        )
                    )

            elif rule.id == "bp-broad-except":
                if isinstance(node.type, ast.Name):
                    if node.type.id in ("Exception", "BaseException"):
                        # Check if exception is just being passed
                        is_pass = len(node.body) == 1 and isinstance(node.body[0], ast.Pass)
                        if is_pass:
                            findings.append(
                                self._create_finding(
                                    rule=rule,
                                    message=f"Catching {node.type.id} and passing silently hides errors",
                                    file_path=file_path,
                                    line=node.lineno,
                                    suggestion="Handle the exception or log it",
                                )
                            )

        return findings

    def _check_import(
        self,
        node: ast.Import,
        file_path: Path,
        rules: list[ReviewRule],
        lines: list[str],
    ) -> list[ReviewFinding]:
        """Check import statements."""
        findings = []

        for rule in rules:
            if rule.id == "bp-wildcard-import":
                for alias in node.names:
                    if alias.name == "*":
                        findings.append(
                            self._create_finding(
                                rule=rule,
                                message="Wildcard imports make code harder to understand",
                                file_path=file_path,
                                line=node.lineno,
                                suggestion="Import specific names instead",
                            )
                        )

        return findings

    def _check_import_from(
        self,
        node: ast.ImportFrom,
        file_path: Path,
        rules: list[ReviewRule],
        lines: list[str],
    ) -> list[ReviewFinding]:
        """Check from imports."""
        findings = []

        for rule in rules:
            if rule.id == "bp-wildcard-import":
                for alias in node.names:
                    if alias.name == "*":
                        findings.append(
                            self._create_finding(
                                rule=rule,
                                message=f"Wildcard import from {node.module} pollutes namespace",
                                file_path=file_path,
                                line=node.lineno,
                                suggestion="Import specific names instead",
                            )
                        )

        return findings

    def _check_comparison(
        self,
        node: ast.Compare,
        file_path: Path,
        rules: list[ReviewRule],
        lines: list[str],
    ) -> list[ReviewFinding]:
        """Check comparison expressions."""
        findings = []

        for rule in rules:
            if rule.id == "bp-none-comparison":
                for op, comparator in zip(node.ops, node.comparators, strict=False):
                    if isinstance(comparator, ast.Constant) and comparator.value is None:
                        if isinstance(op, (ast.Eq, ast.NotEq)):
                            findings.append(
                                self._create_finding(
                                    rule=rule,
                                    message="Use 'is None' or 'is not None' instead of '== None'",
                                    file_path=file_path,
                                    line=node.lineno,
                                )
                            )

            elif rule.id == "bp-bool-comparison":
                for op, comparator in zip(node.ops, node.comparators, strict=False):
                    if isinstance(comparator, ast.Constant):
                        if comparator.value in (True, False):
                            if isinstance(op, (ast.Eq, ast.NotEq)):
                                findings.append(
                                    self._create_finding(
                                        rule=rule,
                                        message="Avoid comparing with True/False explicitly",
                                        file_path=file_path,
                                        line=node.lineno,
                                        suggestion="Use 'if condition:' or 'if not condition:'",
                                    )
                                )

        return findings
