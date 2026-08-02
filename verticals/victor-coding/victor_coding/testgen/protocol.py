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

"""Test generation protocol types.

Defines data structures for automated test generation.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class TestType(Enum):
    """Types of tests to generate."""

    UNIT = "unit"  # Unit tests for functions/methods
    INTEGRATION = "integration"  # Integration tests
    PROPERTY = "property"  # Property-based tests
    PARAMETERIZED = "parameterized"  # Parameterized tests
    MOCK = "mock"  # Tests with mocks
    SNAPSHOT = "snapshot"  # Snapshot tests
    E2E = "e2e"  # End-to-end tests


class TestFramework(Enum):
    """Supported test frameworks."""

    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    MOCHA = "mocha"
    VITEST = "vitest"
    RUST_TEST = "rust_test"
    GO_TEST = "go_test"


class AssertionType(Enum):
    """Types of assertions."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    TRUE = "true"
    FALSE = "false"
    NONE = "none"
    NOT_NONE = "not_none"
    RAISES = "raises"
    IN = "in"
    NOT_IN = "not_in"
    GREATER = "greater"
    LESS = "less"
    INSTANCE = "instance"
    CONTAINS = "contains"
    MATCHES = "matches"  # Regex match
    APPROX = "approx"  # Approximate equality


@dataclass
class TestInput:
    """An input value for a test case."""

    name: str
    value: Any
    type_hint: Optional[str] = None


@dataclass
class TestAssertion:
    """An assertion in a test case."""

    assertion_type: AssertionType
    expected: Any
    actual_expr: str  # Expression to evaluate
    message: str = ""


@dataclass
class TestCase:
    """A single test case."""

    name: str
    description: str = ""
    inputs: list[TestInput] = field(default_factory=list)
    assertions: list[TestAssertion] = field(default_factory=list)
    setup_code: str = ""
    teardown_code: str = ""
    test_type: TestType = TestType.UNIT
    tags: list[str] = field(default_factory=list)
    expected_exception: Optional[str] = None


@dataclass
class TestSuite:
    """A collection of test cases for a target."""

    name: str
    target_file: Path
    target_name: str  # Function/class being tested
    test_cases: list[TestCase] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    fixtures: list[str] = field(default_factory=list)
    setup_module: str = ""
    teardown_module: str = ""


@dataclass
class GeneratedTest:
    """A generated test file."""

    file_path: Path
    content: str
    suites: list[TestSuite] = field(default_factory=list)
    framework: TestFramework = TestFramework.PYTEST
    language: str = "python"


@dataclass
class TestGenConfig:
    """Configuration for test generation."""

    framework: TestFramework = TestFramework.PYTEST
    test_types: list[TestType] = field(default_factory=lambda: [TestType.UNIT])
    include_edge_cases: bool = True
    include_error_cases: bool = True
    include_docstring_examples: bool = True
    max_test_cases_per_function: int = 10
    use_fixtures: bool = True
    use_mocks: bool = True
    generate_parameterized: bool = True
    output_dir: Optional[Path] = None
    test_file_prefix: str = "test_"


@dataclass
class FunctionSignature:
    """Signature of a function to generate tests for."""

    name: str
    parameters: list[tuple[str, Optional[str]]]  # (name, type_hint)
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    is_async: bool = False
    is_method: bool = False
    is_static: bool = False
    is_classmethod: bool = False
    decorators: list[str] = field(default_factory=list)
    source_location: Optional[tuple[int, int]] = None  # (start_line, end_line)


@dataclass
class ClassSignature:
    """Signature of a class to generate tests for."""

    name: str
    methods: list[FunctionSignature] = field(default_factory=list)
    properties: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    docstring: Optional[str] = None


@dataclass
class TestGenResult:
    """Result of test generation."""

    success: bool
    generated_files: list[GeneratedTest] = field(default_factory=list)
    total_test_cases: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
