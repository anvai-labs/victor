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

"""Automated test generation module.

This module provides automated test generation capabilities using
static code analysis to create comprehensive test suites.

Example usage:
    from victor_coding.testgen import get_testgen_manager, TestGenConfig, TestFramework
    from pathlib import Path

    # Get manager
    manager = get_testgen_manager()

    # Generate tests for a file
    result = manager.generate_for_file(
        Path("my_module.py"),
        config=TestGenConfig(
            framework=TestFramework.PYTEST,
            include_edge_cases=True,
            include_error_cases=True,
        ),
    )

    # Generate tests for a directory
    result = manager.generate_for_directory(
        Path("src/"),
        write_files=True,
    )

    # Preview without writing
    content = manager.preview_generation(Path("my_module.py"))
    print(content)

    # Analyze coverage gaps
    gaps = manager.analyze_coverage_gaps(
        source_dir=Path("src/"),
        test_dir=Path("tests/"),
    )

    # Suggest additional tests
    suggestions = manager.suggest_additional_tests(
        Path("my_module.py"),
        existing_tests=Path("tests/test_my_module.py"),
    )
"""

from victor_coding.testgen.protocol import (
    AssertionType,
    ClassSignature,
    FunctionSignature,
    GeneratedTest,
    TestAssertion,
    TestCase,
    TestFramework,
    TestGenConfig,
    TestGenResult,
    TestInput,
    TestSuite,
    TestType,
)
from victor_coding.testgen.analyzer import TestTargetAnalyzer
from victor_coding.testgen.generator import (
    BaseTestGenerator,
    PytestGenerator,
    TestCaseGenerator,
    UnittestGenerator,
)
from victor_coding.testgen.manager import (
    TestGenManager,
    get_testgen_manager,
    reset_testgen_manager,
)

__all__ = [
    # Protocol types
    "AssertionType",
    "ClassSignature",
    "FunctionSignature",
    "GeneratedTest",
    "TestAssertion",
    "TestCase",
    "TestFramework",
    "TestGenConfig",
    "TestGenResult",
    "TestInput",
    "TestSuite",
    "TestType",
    # Analyzer
    "TestTargetAnalyzer",
    # Generators
    "BaseTestGenerator",
    "PytestGenerator",
    "TestCaseGenerator",
    "UnittestGenerator",
    # Manager
    "TestGenManager",
    "get_testgen_manager",
    "reset_testgen_manager",
]
