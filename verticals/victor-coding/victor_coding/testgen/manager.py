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

"""Test generation manager for orchestrating test creation.

Provides high-level API for automated test generation.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from victor_coding.testgen.analyzer import TestTargetAnalyzer
from victor_coding.testgen.generator import (
    BaseTestGenerator,
    PytestGenerator,
    TestCaseGenerator,
    UnittestGenerator,
)
from victor_coding.testgen.protocol import (
    GeneratedTest,
    TestFramework,
    TestGenConfig,
    TestGenResult,
)

logger = logging.getLogger(__name__)


class TestGenManager:
    """High-level manager for test generation.

    Orchestrates analysis, generation, and file writing
    with configurable options.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        config: Optional[TestGenConfig] = None,
    ):
        """Initialize the manager.

        Args:
            project_root: Root directory of the project
            config: Default generation configuration
        """
        self.project_root = project_root or Path.cwd()
        self.config = config or TestGenConfig()

        # Initialize components
        self.analyzer = TestTargetAnalyzer()
        self.generator = TestCaseGenerator(self.analyzer)

        # Register additional generators
        self._generators: dict[TestFramework, BaseTestGenerator] = {
            TestFramework.PYTEST: PytestGenerator(),
            TestFramework.UNITTEST: UnittestGenerator(),
        }

    def register_generator(self, generator: BaseTestGenerator) -> None:
        """Register a custom test generator.

        Args:
            generator: Generator to register
        """
        self._generators[generator.framework] = generator
        self.generator.register_generator(generator)

    def generate_for_file(
        self,
        file_path: Path,
        config: Optional[TestGenConfig] = None,
        write_file: bool = True,
    ) -> TestGenResult:
        """Generate tests for a single file.

        Args:
            file_path: Path to the source file
            config: Generation configuration (uses default if None)
            write_file: Whether to write the test file

        Returns:
            TestGenResult with generated tests
        """
        start_time = time.time()
        config = config or self.config
        result = TestGenResult(success=False)

        try:
            # Generate tests
            generated = self.generator.generate_for_file(file_path, config)

            if generated is None:
                result.warnings.append(f"No testable targets found in {file_path}")
                result.success = True  # Not an error, just no targets
                return result

            # Write file if requested
            if write_file:
                self._write_test_file(generated)

            result.generated_files.append(generated)
            result.total_test_cases = sum(len(suite.test_cases) for suite in generated.suites)
            result.success = True

        except Exception as e:
            logger.error(f"Test generation failed for {file_path}: {e}")
            result.errors.append(str(e))

        result.duration_ms = (time.time() - start_time) * 1000
        return result

    def generate_for_directory(
        self,
        directory: Path,
        config: Optional[TestGenConfig] = None,
        write_files: bool = True,
        recursive: bool = True,
        exclude_patterns: Optional[list[str]] = None,
    ) -> TestGenResult:
        """Generate tests for all Python files in a directory.

        Args:
            directory: Directory to scan
            config: Generation configuration
            write_files: Whether to write test files
            recursive: Whether to recurse into subdirectories
            exclude_patterns: Glob patterns to exclude

        Returns:
            TestGenResult with all generated tests
        """
        start_time = time.time()
        config = config or self.config
        result = TestGenResult(success=True)
        exclude_patterns = exclude_patterns or [
            "**/test_*.py",
            "**/*_test.py",
            "**/tests/**",
            "**/__pycache__/**",
            "**/venv/**",
            "**/.venv/**",
            "**/node_modules/**",
        ]

        # Find Python files
        pattern = "**/*.py" if recursive else "*.py"
        files = list(directory.glob(pattern))

        for file_path in files:
            # Check exclusions
            if self._should_exclude(file_path, exclude_patterns):
                continue

            # Generate tests
            file_result = self.generate_for_file(
                file_path,
                config=config,
                write_file=write_files,
            )

            # Merge results
            result.generated_files.extend(file_result.generated_files)
            result.total_test_cases += file_result.total_test_cases
            result.errors.extend(file_result.errors)
            result.warnings.extend(file_result.warnings)

            if file_result.errors:
                result.success = False

        result.duration_ms = (time.time() - start_time) * 1000
        return result

    def generate_for_module(
        self,
        module_path: Path,
        config: Optional[TestGenConfig] = None,
        write_files: bool = True,
    ) -> TestGenResult:
        """Generate tests for a Python module (package).

        Args:
            module_path: Path to the module directory
            config: Generation configuration
            write_files: Whether to write test files

        Returns:
            TestGenResult
        """
        if not module_path.is_dir():
            return TestGenResult(
                success=False,
                errors=[f"Module path {module_path} is not a directory"],
            )

        # Check for __init__.py to confirm it's a package
        init_file = module_path / "__init__.py"
        if not init_file.exists():
            logger.warning(f"{module_path} is not a Python package (no __init__.py)")

        return self.generate_for_directory(
            module_path,
            config=config,
            write_files=write_files,
            recursive=True,
        )

    def analyze_coverage_gaps(
        self,
        source_dir: Path,
        test_dir: Path,
    ) -> dict[Path, list[str]]:
        """Analyze which functions/classes lack tests.

        Args:
            source_dir: Source code directory
            test_dir: Test code directory

        Returns:
            Dict mapping source files to untested symbols
        """
        gaps: dict[Path, list[str]] = {}

        # Analyze source files
        for source_file in source_dir.rglob("*.py"):
            if self._should_exclude(source_file, ["**/__pycache__/**", "**/venv/**"]):
                continue

            functions, classes = self.analyzer.analyze_file(source_file)

            # Find corresponding test file
            test_file = test_dir / f"test_{source_file.name}"
            if not test_file.exists():
                # No test file at all
                gaps[source_file] = [f.name for f in functions] + [c.name for c in classes]
                continue

            # Check which symbols have tests
            test_content = test_file.read_text()
            untested = []

            for func in functions:
                if f"test_{func.name}" not in test_content:
                    untested.append(func.name)

            for cls in classes:
                if f"Test{cls.name}" not in test_content:
                    untested.append(cls.name)

            if untested:
                gaps[source_file] = untested

        return gaps

    def suggest_additional_tests(
        self,
        file_path: Path,
        existing_tests: Optional[Path] = None,
    ) -> list[str]:
        """Suggest additional test cases for a file.

        Args:
            file_path: Source file to analyze
            existing_tests: Path to existing test file

        Returns:
            List of suggested test names
        """
        suggestions = []

        functions, classes = self.analyzer.analyze_file(file_path)

        # Read existing tests if provided
        existing_test_names = set()
        if existing_tests and existing_tests.exists():
            content = existing_tests.read_text()
            import re

            existing_test_names = set(re.findall(r"def (test_\w+)", content))

        # Check each function
        for func in functions:
            base_test = f"test_{func.name}"

            # Suggest basic test if missing
            if base_test not in existing_test_names:
                suggestions.append(f"{base_test}_basic")

            # Suggest edge case tests
            for param_name, _type_hint in func.parameters:
                if param_name in ("self", "cls"):
                    continue

                edge_tests = [
                    f"test_{func.name}_{param_name}_empty",
                    f"test_{func.name}_{param_name}_none",
                    f"test_{func.name}_{param_name}_edge_case",
                ]
                for edge_test in edge_tests:
                    if edge_test not in existing_test_names:
                        suggestions.append(edge_test)

            # Suggest error tests
            error_conditions = self.analyzer.detect_error_conditions(func)
            for desc, _, _ in error_conditions:
                error_test = f"test_{func.name}_{desc}"
                if error_test not in existing_test_names:
                    suggestions.append(error_test)

        return suggestions

    def preview_generation(
        self,
        file_path: Path,
        config: Optional[TestGenConfig] = None,
    ) -> Optional[str]:
        """Preview generated test content without writing.

        Args:
            file_path: Source file
            config: Generation configuration

        Returns:
            Generated test content or None
        """
        config = config or self.config

        generated = self.generator.generate_for_file(file_path, config)
        if generated:
            return generated.content
        return None

    def _write_test_file(self, generated: GeneratedTest) -> None:
        """Write a generated test file.

        Args:
            generated: GeneratedTest to write
        """
        # Create directory if needed
        generated.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        generated.file_path.write_text(generated.content)
        logger.info(f"Wrote test file: {generated.file_path}")

    def _should_exclude(
        self,
        file_path: Path,
        patterns: list[str],
    ) -> bool:
        """Check if a file should be excluded.

        Args:
            file_path: File to check
            patterns: Exclusion patterns

        Returns:
            True if file should be excluded
        """
        from fnmatch import fnmatch

        str_path = str(file_path)
        for pattern in patterns:
            if fnmatch(str_path, pattern):
                return True
        return False


# Global manager singleton
_testgen_manager: Optional[TestGenManager] = None


def get_testgen_manager(
    project_root: Optional[Path] = None,
    config: Optional[TestGenConfig] = None,
) -> TestGenManager:
    """Get the global test generation manager.

    Args:
        project_root: Project root directory
        config: Default configuration

    Returns:
        TestGenManager instance
    """
    global _testgen_manager
    if _testgen_manager is None or (project_root and _testgen_manager.project_root != project_root):
        _testgen_manager = TestGenManager(project_root=project_root, config=config)
    return _testgen_manager


def reset_testgen_manager() -> None:
    """Reset the global manager."""
    global _testgen_manager
    _testgen_manager = None
