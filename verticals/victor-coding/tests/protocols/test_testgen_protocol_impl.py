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

"""Tests for test generation protocol types and data structures."""

from pathlib import Path

import pytest

pytest.importorskip("victor_coding")

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

# =============================================================================
# ENUM TESTS
# =============================================================================


class TestTestType:
    """Tests for TestType enum."""

    def test_unit_type(self):
        """Test unit test type."""
        assert TestType.UNIT.value == "unit"

    def test_integration_type(self):
        """Test integration test type."""
        assert TestType.INTEGRATION.value == "integration"

    def test_property_type(self):
        """Test property test type."""
        assert TestType.PROPERTY.value == "property"

    def test_parameterized_type(self):
        """Test parameterized test type."""
        assert TestType.PARAMETERIZED.value == "parameterized"

    def test_mock_type(self):
        """Test mock test type."""
        assert TestType.MOCK.value == "mock"

    def test_snapshot_type(self):
        """Test snapshot test type."""
        assert TestType.SNAPSHOT.value == "snapshot"

    def test_e2e_type(self):
        """Test e2e test type."""
        assert TestType.E2E.value == "e2e"


class TestTestFramework:
    """Tests for TestFramework enum."""

    def test_pytest_framework(self):
        """Test pytest framework."""
        assert TestFramework.PYTEST.value == "pytest"

    def test_unittest_framework(self):
        """Test unittest framework."""
        assert TestFramework.UNITTEST.value == "unittest"

    def test_jest_framework(self):
        """Test jest framework."""
        assert TestFramework.JEST.value == "jest"

    def test_mocha_framework(self):
        """Test mocha framework."""
        assert TestFramework.MOCHA.value == "mocha"

    def test_vitest_framework(self):
        """Test vitest framework."""
        assert TestFramework.VITEST.value == "vitest"

    def test_rust_test_framework(self):
        """Test rust_test framework."""
        assert TestFramework.RUST_TEST.value == "rust_test"

    def test_go_test_framework(self):
        """Test go_test framework."""
        assert TestFramework.GO_TEST.value == "go_test"


class TestAssertionType:
    """Tests for AssertionType enum."""

    def test_equals_assertion(self):
        """Test equals assertion type."""
        assert AssertionType.EQUALS.value == "equals"

    def test_not_equals_assertion(self):
        """Test not_equals assertion type."""
        assert AssertionType.NOT_EQUALS.value == "not_equals"

    def test_true_assertion(self):
        """Test true assertion type."""
        assert AssertionType.TRUE.value == "true"

    def test_false_assertion(self):
        """Test false assertion type."""
        assert AssertionType.FALSE.value == "false"

    def test_none_assertion(self):
        """Test none assertion type."""
        assert AssertionType.NONE.value == "none"

    def test_not_none_assertion(self):
        """Test not_none assertion type."""
        assert AssertionType.NOT_NONE.value == "not_none"

    def test_raises_assertion(self):
        """Test raises assertion type."""
        assert AssertionType.RAISES.value == "raises"

    def test_in_assertion(self):
        """Test in assertion type."""
        assert AssertionType.IN.value == "in"

    def test_not_in_assertion(self):
        """Test not_in assertion type."""
        assert AssertionType.NOT_IN.value == "not_in"

    def test_greater_assertion(self):
        """Test greater assertion type."""
        assert AssertionType.GREATER.value == "greater"

    def test_less_assertion(self):
        """Test less assertion type."""
        assert AssertionType.LESS.value == "less"

    def test_instance_assertion(self):
        """Test instance assertion type."""
        assert AssertionType.INSTANCE.value == "instance"

    def test_contains_assertion(self):
        """Test contains assertion type."""
        assert AssertionType.CONTAINS.value == "contains"

    def test_matches_assertion(self):
        """Test matches assertion type."""
        assert AssertionType.MATCHES.value == "matches"

    def test_approx_assertion(self):
        """Test approx assertion type."""
        assert AssertionType.APPROX.value == "approx"


# =============================================================================
# TEST INPUT TESTS
# =============================================================================


class TestTestInput:
    """Tests for TestInput dataclass."""

    def test_creation_minimal(self):
        """Test minimal test input creation."""
        inp = TestInput(name="x", value=42)
        assert inp.name == "x"
        assert inp.value == 42
        assert inp.type_hint is None

    def test_creation_with_type_hint(self):
        """Test test input with type hint."""
        inp = TestInput(name="items", value=[1, 2, 3], type_hint="list[int]")
        assert inp.type_hint == "list[int]"


# =============================================================================
# TEST ASSERTION TESTS
# =============================================================================


class TestTestAssertion:
    """Tests for TestAssertion dataclass."""

    def test_creation_minimal(self):
        """Test minimal test assertion creation."""
        assertion = TestAssertion(
            assertion_type=AssertionType.EQUALS,
            expected=42,
            actual_expr="result",
        )
        assert assertion.assertion_type == AssertionType.EQUALS
        assert assertion.expected == 42
        assert assertion.actual_expr == "result"
        assert assertion.message == ""

    def test_creation_with_message(self):
        """Test test assertion with message."""
        assertion = TestAssertion(
            assertion_type=AssertionType.TRUE,
            expected=True,
            actual_expr="is_valid()",
            message="Result should be valid",
        )
        assert assertion.message == "Result should be valid"


# =============================================================================
# TEST CASE TESTS
# =============================================================================


class TestTestCase:
    """Tests for TestCase dataclass."""

    def test_creation_minimal(self):
        """Test minimal test case creation."""
        tc = TestCase(name="test_add")
        assert tc.name == "test_add"
        assert tc.description == ""
        assert tc.inputs == []
        assert tc.assertions == []
        assert tc.test_type == TestType.UNIT

    def test_creation_full(self):
        """Test full test case creation."""
        tc = TestCase(
            name="test_add_positive_numbers",
            description="Test adding two positive numbers",
            inputs=[
                TestInput("a", 2, "int"),
                TestInput("b", 3, "int"),
            ],
            assertions=[
                TestAssertion(AssertionType.EQUALS, 5, "add(a, b)"),
            ],
            setup_code="# Setup code",
            teardown_code="# Teardown code",
            test_type=TestType.PARAMETERIZED,
            tags=["unit", "arithmetic"],
            expected_exception=None,
        )
        assert len(tc.inputs) == 2
        assert len(tc.assertions) == 1
        assert tc.test_type == TestType.PARAMETERIZED
        assert "arithmetic" in tc.tags

    def test_with_expected_exception(self):
        """Test test case with expected exception."""
        tc = TestCase(
            name="test_division_by_zero",
            expected_exception="ZeroDivisionError",
        )
        assert tc.expected_exception == "ZeroDivisionError"


# =============================================================================
# TEST SUITE TESTS
# =============================================================================


class TestTestSuite:
    """Tests for TestSuite dataclass."""

    def test_creation_minimal(self):
        """Test minimal test suite creation."""
        suite = TestSuite(
            name="TestCalculator",
            target_file=Path("calculator.py"),
            target_name="Calculator",
        )
        assert suite.name == "TestCalculator"
        assert suite.target_file == Path("calculator.py")
        assert suite.test_cases == []
        assert suite.imports == []

    def test_creation_full(self):
        """Test full test suite creation."""
        tc = TestCase(name="test_add")
        suite = TestSuite(
            name="TestCalculator",
            target_file=Path("calculator.py"),
            target_name="Calculator",
            test_cases=[tc],
            imports=["from calculator import Calculator"],
            fixtures=["calculator_instance"],
            setup_module="# Module setup",
            teardown_module="# Module teardown",
        )
        assert len(suite.test_cases) == 1
        assert "from calculator import Calculator" in suite.imports
        assert "calculator_instance" in suite.fixtures


# =============================================================================
# GENERATED TEST TESTS
# =============================================================================


class TestGeneratedTest:
    """Tests for GeneratedTest dataclass."""

    def test_creation_minimal(self):
        """Test minimal generated test creation."""
        test = GeneratedTest(
            file_path=Path("test_calculator.py"),
            content="def test_example(): pass",
        )
        assert test.file_path == Path("test_calculator.py")
        assert test.suites == []
        assert test.framework == TestFramework.PYTEST
        assert test.language == "python"

    def test_creation_full(self):
        """Test full generated test creation."""
        suite = TestSuite(
            name="TestCalculator",
            target_file=Path("calculator.py"),
            target_name="Calculator",
        )
        test = GeneratedTest(
            file_path=Path("test_calculator.py"),
            content="# Test file content",
            suites=[suite],
            framework=TestFramework.UNITTEST,
            language="python",
        )
        assert len(test.suites) == 1
        assert test.framework == TestFramework.UNITTEST


# =============================================================================
# TEST GEN CONFIG TESTS
# =============================================================================


class TestTestGenConfig:
    """Tests for TestGenConfig dataclass."""

    def test_default_config(self):
        """Test default config values."""
        config = TestGenConfig()
        assert config.framework == TestFramework.PYTEST
        assert config.test_types == [TestType.UNIT]
        assert config.include_edge_cases is True
        assert config.include_error_cases is True
        assert config.include_docstring_examples is True
        assert config.max_test_cases_per_function == 10
        assert config.use_fixtures is True
        assert config.use_mocks is True
        assert config.generate_parameterized is True
        assert config.output_dir is None
        assert config.test_file_prefix == "test_"

    def test_custom_config(self):
        """Test custom config."""
        config = TestGenConfig(
            framework=TestFramework.JEST,
            test_types=[TestType.UNIT, TestType.INTEGRATION],
            include_edge_cases=False,
            max_test_cases_per_function=5,
            output_dir=Path("tests"),
            test_file_prefix="spec_",
        )
        assert config.framework == TestFramework.JEST
        assert len(config.test_types) == 2
        assert config.include_edge_cases is False
        assert config.max_test_cases_per_function == 5
        assert config.output_dir == Path("tests")
        assert config.test_file_prefix == "spec_"


# =============================================================================
# FUNCTION SIGNATURE TESTS
# =============================================================================


class TestFunctionSignature:
    """Tests for FunctionSignature dataclass."""

    def test_creation_minimal(self):
        """Test minimal function signature creation."""
        sig = FunctionSignature(
            name="add",
            parameters=[("a", "int"), ("b", "int")],
        )
        assert sig.name == "add"
        assert len(sig.parameters) == 2
        assert sig.return_type is None
        assert sig.is_async is False
        assert sig.is_method is False

    def test_creation_full(self):
        """Test full function signature creation."""
        sig = FunctionSignature(
            name="fetch_data",
            parameters=[("url", "str"), ("timeout", "float")],
            return_type="dict[str, Any]",
            docstring="Fetch data from URL.",
            is_async=True,
            is_method=False,
            is_static=False,
            is_classmethod=False,
            decorators=["@retry(3)"],
            source_location=(10, 25),
        )
        assert sig.return_type == "dict[str, Any]"
        assert sig.is_async is True
        assert sig.source_location == (10, 25)
        assert "@retry(3)" in sig.decorators

    def test_method_signature(self):
        """Test method signature."""
        sig = FunctionSignature(
            name="process",
            parameters=[("self", None), ("data", "bytes")],
            is_method=True,
        )
        assert sig.is_method is True

    def test_classmethod_signature(self):
        """Test classmethod signature."""
        sig = FunctionSignature(
            name="from_json",
            parameters=[("cls", None), ("data", "str")],
            is_classmethod=True,
            decorators=["@classmethod"],
        )
        assert sig.is_classmethod is True

    def test_staticmethod_signature(self):
        """Test staticmethod signature."""
        sig = FunctionSignature(
            name="validate",
            parameters=[("value", "Any")],
            is_static=True,
            decorators=["@staticmethod"],
        )
        assert sig.is_static is True


# =============================================================================
# CLASS SIGNATURE TESTS
# =============================================================================


class TestClassSignature:
    """Tests for ClassSignature dataclass."""

    def test_creation_minimal(self):
        """Test minimal class signature creation."""
        cls_sig = ClassSignature(name="Calculator")
        assert cls_sig.name == "Calculator"
        assert cls_sig.methods == []
        assert cls_sig.properties == []
        assert cls_sig.bases == []
        assert cls_sig.docstring is None

    def test_creation_full(self):
        """Test full class signature creation."""
        method = FunctionSignature(
            name="add",
            parameters=[("self", None), ("a", "int"), ("b", "int")],
            return_type="int",
            is_method=True,
        )
        cls_sig = ClassSignature(
            name="Calculator",
            methods=[method],
            properties=["result"],
            bases=["BaseCalculator"],
            docstring="A calculator class.",
        )
        assert len(cls_sig.methods) == 1
        assert "result" in cls_sig.properties
        assert "BaseCalculator" in cls_sig.bases
        assert cls_sig.docstring == "A calculator class."


# =============================================================================
# TEST GEN RESULT TESTS
# =============================================================================


class TestTestGenResult:
    """Tests for TestGenResult dataclass."""

    def test_creation_success_empty(self):
        """Test successful empty result."""
        result = TestGenResult(success=True)
        assert result.success is True
        assert result.generated_files == []
        assert result.total_test_cases == 0
        assert result.errors == []
        assert result.warnings == []
        assert result.duration_ms == 0.0

    def test_creation_success_with_files(self):
        """Test successful result with generated files."""
        gen_file = GeneratedTest(
            file_path=Path("test_calc.py"),
            content="# Tests",
        )
        result = TestGenResult(
            success=True,
            generated_files=[gen_file],
            total_test_cases=5,
            duration_ms=150.0,
        )
        assert len(result.generated_files) == 1
        assert result.total_test_cases == 5
        assert result.duration_ms == 150.0

    def test_creation_failure(self):
        """Test failed result."""
        result = TestGenResult(
            success=False,
            errors=["Failed to parse file"],
            warnings=["Function has no type hints"],
        )
        assert result.success is False
        assert "Failed to parse file" in result.errors
        assert "Function has no type hints" in result.warnings
