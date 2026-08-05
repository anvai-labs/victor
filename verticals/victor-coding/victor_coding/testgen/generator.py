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

"""Test case generator for automated test creation.

Uses code analysis to generate comprehensive test cases
with appropriate assertions and edge cases.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from victor_coding.testgen.analyzer import TestTargetAnalyzer
from victor_coding.testgen.protocol import (
    AssertionType,
    ClassSignature,
    FunctionSignature,
    GeneratedTest,
    TestAssertion,
    TestCase,
    TestFramework,
    TestGenConfig,
    TestInput,
    TestSuite,
    TestType,
)

logger = logging.getLogger(__name__)


class BaseTestGenerator(ABC):
    """Abstract base for test generators.

    Implements Strategy pattern for framework-specific generation.
    """

    @property
    @abstractmethod
    def framework(self) -> TestFramework:
        """Get the test framework this generator targets."""
        pass

    @abstractmethod
    def generate_test_file(
        self,
        suites: list[TestSuite],
        config: TestGenConfig,
    ) -> str:
        """Generate test file content.

        Args:
            suites: Test suites to generate
            config: Generation configuration

        Returns:
            Generated test file content
        """
        pass

    @abstractmethod
    def format_assertion(self, assertion: TestAssertion) -> str:
        """Format an assertion for this framework.

        Args:
            assertion: Assertion to format

        Returns:
            Formatted assertion code
        """
        pass


class PytestGenerator(BaseTestGenerator):
    """Pytest test file generator."""

    @property
    def framework(self) -> TestFramework:
        return TestFramework.PYTEST

    def generate_test_file(
        self,
        suites: list[TestSuite],
        config: TestGenConfig,
    ) -> str:
        """Generate pytest test file."""
        lines = []

        # Collect all imports
        imports = set()
        imports.add("import pytest")

        for suite in suites:
            imports.update(suite.imports)

        # Add imports
        lines.extend(sorted(imports))
        lines.append("")
        lines.append("")

        # Generate fixtures if needed
        if config.use_fixtures:
            fixtures = self._collect_fixtures(suites)
            for _fixture_name, fixture_code in fixtures.items():
                lines.append(fixture_code)
                lines.append("")

        # Generate test classes/functions
        for suite in suites:
            lines.extend(self._generate_suite(suite, config))
            lines.append("")

        return "\n".join(lines)

    def _collect_fixtures(
        self,
        suites: list[TestSuite],
    ) -> dict[str, str]:
        """Collect fixtures from suites."""
        fixtures = {}
        for suite in suites:
            for fixture_name in suite.fixtures:
                if fixture_name not in fixtures:
                    fixtures[fixture_name] = self._generate_fixture(fixture_name)
        return fixtures

    def _generate_fixture(self, name: str) -> str:
        """Generate a fixture definition."""
        return f'''@pytest.fixture
def {name}():
    """Fixture for {name}."""
    # TODO: Implement fixture
    yield None'''

    def _generate_suite(
        self,
        suite: TestSuite,
        config: TestGenConfig,
    ) -> list[str]:
        """Generate a test suite."""
        lines = []

        # Check if we need a class
        if any(tc.test_type == TestType.PARAMETERIZED for tc in suite.test_cases):
            # Use class for parameterized tests
            lines.append(f"class Test{suite.target_name.title().replace('_', '')}:")
            lines.append(f'    """Tests for {suite.target_name}."""')
            lines.append("")

            for test_case in suite.test_cases:
                test_lines = self._generate_test_method(test_case, config, indent=1)
                lines.extend(test_lines)
                lines.append("")
        else:
            # Use standalone functions
            for test_case in suite.test_cases:
                test_lines = self._generate_test_function(test_case, config)
                lines.extend(test_lines)
                lines.append("")

        return lines

    def _generate_test_function(
        self,
        test_case: TestCase,
        config: TestGenConfig,
    ) -> list[str]:
        """Generate a standalone test function."""
        lines = []

        # Add parameterization decorator if needed
        if test_case.test_type == TestType.PARAMETERIZED and config.generate_parameterized:
            params = self._generate_parametrize_decorator(test_case)
            if params:
                lines.append(params)

        # Add expected exception decorator
        if test_case.expected_exception:
            lines.append(f"@pytest.mark.xfail(raises={test_case.expected_exception})")

        # Function signature
        func_params = ", ".join(inp.name for inp in test_case.inputs) if test_case.inputs else ""
        lines.append(f"def {test_case.name}({func_params}):")

        # Docstring
        if test_case.description:
            lines.append(f'    """{test_case.description}"""')

        # Setup code
        if test_case.setup_code:
            for line in test_case.setup_code.split("\n"):
                lines.append(f"    {line}")

        # Assertions
        for assertion in test_case.assertions:
            lines.append(f"    {self.format_assertion(assertion)}")

        # Teardown code
        if test_case.teardown_code:
            for line in test_case.teardown_code.split("\n"):
                lines.append(f"    {line}")

        # Ensure at least one line in function
        if not test_case.assertions and not test_case.setup_code:
            lines.append("    pass  # TODO: Add assertions")

        return lines

    def _generate_test_method(
        self,
        test_case: TestCase,
        config: TestGenConfig,
        indent: int = 1,
    ) -> list[str]:
        """Generate a test method within a class."""
        prefix = "    " * indent
        lines = []

        # Add parameterization decorator if needed
        if test_case.test_type == TestType.PARAMETERIZED and config.generate_parameterized:
            params = self._generate_parametrize_decorator(test_case)
            if params:
                lines.append(f"{prefix}{params}")

        # Method signature
        func_params = ", ".join(inp.name for inp in test_case.inputs) if test_case.inputs else ""
        if func_params:
            func_params = f"self, {func_params}"
        else:
            func_params = "self"
        lines.append(f"{prefix}def {test_case.name}({func_params}):")

        # Docstring
        if test_case.description:
            lines.append(f'{prefix}    """{test_case.description}"""')

        # Setup
        if test_case.setup_code:
            for line in test_case.setup_code.split("\n"):
                lines.append(f"{prefix}    {line}")

        # Assertions
        for assertion in test_case.assertions:
            lines.append(f"{prefix}    {self.format_assertion(assertion)}")

        # Teardown
        if test_case.teardown_code:
            for line in test_case.teardown_code.split("\n"):
                lines.append(f"{prefix}    {line}")

        if not test_case.assertions and not test_case.setup_code:
            lines.append(f"{prefix}    pass  # TODO: Add assertions")

        return lines

    def _generate_parametrize_decorator(self, test_case: TestCase) -> Optional[str]:
        """Generate pytest.mark.parametrize decorator."""
        if not test_case.inputs:
            return None

        param_names = ", ".join(inp.name for inp in test_case.inputs)
        # For now, return a placeholder - actual values would come from analyzer
        return f'@pytest.mark.parametrize("{param_names}", [])'

    def format_assertion(self, assertion: TestAssertion) -> str:
        """Format assertion for pytest."""
        mapping = {
            AssertionType.EQUALS: f"assert {assertion.actual_expr} == {assertion.expected!r}",
            AssertionType.NOT_EQUALS: f"assert {assertion.actual_expr} != {assertion.expected!r}",
            AssertionType.TRUE: f"assert {assertion.actual_expr}",
            AssertionType.FALSE: f"assert not {assertion.actual_expr}",
            AssertionType.NONE: f"assert {assertion.actual_expr} is None",
            AssertionType.NOT_NONE: f"assert {assertion.actual_expr} is not None",
            AssertionType.IN: f"assert {assertion.expected!r} in {assertion.actual_expr}",
            AssertionType.NOT_IN: f"assert {assertion.expected!r} not in {assertion.actual_expr}",
            AssertionType.GREATER: f"assert {assertion.actual_expr} > {assertion.expected!r}",
            AssertionType.LESS: f"assert {assertion.actual_expr} < {assertion.expected!r}",
            AssertionType.INSTANCE: f"assert isinstance({assertion.actual_expr}, {assertion.expected})",
            AssertionType.CONTAINS: f"assert {assertion.expected!r} in {assertion.actual_expr}",
            AssertionType.APPROX: f"assert {assertion.actual_expr} == pytest.approx({assertion.expected!r})",
        }

        if assertion.assertion_type == AssertionType.RAISES:
            return f"pytest.raises({assertion.expected})"
        elif assertion.assertion_type == AssertionType.MATCHES:
            return f"assert re.match({assertion.expected!r}, {assertion.actual_expr})"

        return mapping.get(
            assertion.assertion_type,
            f"assert {assertion.actual_expr}  # {assertion.message}",
        )


class UnittestGenerator(BaseTestGenerator):
    """Unittest test file generator."""

    @property
    def framework(self) -> TestFramework:
        return TestFramework.UNITTEST

    def generate_test_file(
        self,
        suites: list[TestSuite],
        config: TestGenConfig,
    ) -> str:
        """Generate unittest test file."""
        lines = []

        # Standard imports
        lines.append("import unittest")

        # Collect other imports
        imports = set()
        for suite in suites:
            imports.update(suite.imports)

        for imp in sorted(imports):
            lines.append(imp)

        lines.append("")
        lines.append("")

        # Generate test classes
        for suite in suites:
            lines.extend(self._generate_suite(suite, config))
            lines.append("")

        # Add main block
        lines.append("")
        lines.append('if __name__ == "__main__":')
        lines.append("    unittest.main()")

        return "\n".join(lines)

    def _generate_suite(
        self,
        suite: TestSuite,
        config: TestGenConfig,
    ) -> list[str]:
        """Generate a test class for unittest."""
        lines = []

        class_name = f"Test{suite.target_name.title().replace('_', '')}"
        lines.append(f"class {class_name}(unittest.TestCase):")
        lines.append(f'    """Tests for {suite.target_name}."""')
        lines.append("")

        # Setup method
        if suite.setup_module:
            lines.append("    def setUp(self):")
            for line in suite.setup_module.split("\n"):
                lines.append(f"        {line}")
            lines.append("")

        # Test methods
        for test_case in suite.test_cases:
            lines.extend(self._generate_test_method(test_case, config))
            lines.append("")

        # Teardown method
        if suite.teardown_module:
            lines.append("    def tearDown(self):")
            for line in suite.teardown_module.split("\n"):
                lines.append(f"        {line}")
            lines.append("")

        return lines

    def _generate_test_method(
        self,
        test_case: TestCase,
        config: TestGenConfig,
    ) -> list[str]:
        """Generate a test method."""
        lines = []

        lines.append(f"    def {test_case.name}(self):")

        if test_case.description:
            lines.append(f'        """{test_case.description}"""')

        # Setup
        if test_case.setup_code:
            for line in test_case.setup_code.split("\n"):
                lines.append(f"        {line}")

        # Handle expected exceptions
        if test_case.expected_exception:
            lines.append(f"        with self.assertRaises({test_case.expected_exception}):")
            # Add indented assertions
            for assertion in test_case.assertions:
                lines.append(f"            {self.format_assertion(assertion)}")
        else:
            # Normal assertions
            for assertion in test_case.assertions:
                lines.append(f"        {self.format_assertion(assertion)}")

        # Teardown
        if test_case.teardown_code:
            for line in test_case.teardown_code.split("\n"):
                lines.append(f"        {line}")

        if not test_case.assertions and not test_case.setup_code:
            lines.append("        pass  # TODO: Add assertions")

        return lines

    def format_assertion(self, assertion: TestAssertion) -> str:
        """Format assertion for unittest."""
        mapping = {
            AssertionType.EQUALS: f"self.assertEqual({assertion.actual_expr}, {assertion.expected!r})",
            AssertionType.NOT_EQUALS: f"self.assertNotEqual({assertion.actual_expr}, {assertion.expected!r})",
            AssertionType.TRUE: f"self.assertTrue({assertion.actual_expr})",
            AssertionType.FALSE: f"self.assertFalse({assertion.actual_expr})",
            AssertionType.NONE: f"self.assertIsNone({assertion.actual_expr})",
            AssertionType.NOT_NONE: f"self.assertIsNotNone({assertion.actual_expr})",
            AssertionType.IN: f"self.assertIn({assertion.expected!r}, {assertion.actual_expr})",
            AssertionType.NOT_IN: f"self.assertNotIn({assertion.expected!r}, {assertion.actual_expr})",
            AssertionType.GREATER: f"self.assertGreater({assertion.actual_expr}, {assertion.expected!r})",
            AssertionType.LESS: f"self.assertLess({assertion.actual_expr}, {assertion.expected!r})",
            AssertionType.INSTANCE: f"self.assertIsInstance({assertion.actual_expr}, {assertion.expected})",
            AssertionType.CONTAINS: f"self.assertIn({assertion.expected!r}, {assertion.actual_expr})",
            AssertionType.APPROX: f"self.assertAlmostEqual({assertion.actual_expr}, {assertion.expected!r})",
        }

        if assertion.assertion_type == AssertionType.RAISES:
            return f"self.assertRaises({assertion.expected})"
        elif assertion.assertion_type == AssertionType.MATCHES:
            return f"self.assertRegex({assertion.actual_expr}, {assertion.expected!r})"

        return mapping.get(
            assertion.assertion_type,
            f"# {assertion.message}",
        )


class TestCaseGenerator:
    """Generates test cases from code analysis.

    Uses the analyzer to extract function signatures and
    generates appropriate test cases based on configuration.
    """

    def __init__(self, analyzer: Optional[TestTargetAnalyzer] = None):
        """Initialize the generator.

        Args:
            analyzer: Code analyzer instance
        """
        self.analyzer = analyzer or TestTargetAnalyzer()
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

    def generate_for_file(
        self,
        file_path: Path,
        config: TestGenConfig,
    ) -> Optional[GeneratedTest]:
        """Generate tests for a Python file.

        Args:
            file_path: Path to analyze
            config: Generation configuration

        Returns:
            GeneratedTest or None if no testable targets
        """
        functions, classes = self.analyzer.analyze_file(file_path)

        if not functions and not classes:
            logger.info(f"No testable targets found in {file_path}")
            return None

        suites = []

        # Generate suites for functions
        for func in functions:
            suite = self._generate_function_suite(func, file_path, config)
            if suite.test_cases:
                suites.append(suite)

        # Generate suites for classes
        for cls in classes:
            suite = self._generate_class_suite(cls, file_path, config)
            if suite.test_cases:
                suites.append(suite)

        if not suites:
            return None

        # Get generator for framework
        generator = self._generators.get(config.framework)
        if not generator:
            logger.error(f"No generator for framework: {config.framework}")
            return None

        # Generate test file content
        content = generator.generate_test_file(suites, config)

        # Determine output path
        if config.output_dir:
            test_dir = config.output_dir
        else:
            test_dir = file_path.parent / "tests"

        test_file_name = f"{config.test_file_prefix}{file_path.stem}.py"
        test_file_path = test_dir / test_file_name

        return GeneratedTest(
            file_path=test_file_path,
            content=content,
            suites=suites,
            framework=config.framework,
            language="python",
        )

    def _generate_function_suite(
        self,
        func: FunctionSignature,
        source_file: Path,
        config: TestGenConfig,
    ) -> TestSuite:
        """Generate test suite for a function."""
        suite = TestSuite(
            name=f"test_{func.name}",
            target_file=source_file,
            target_name=func.name,
            imports=[f"from {source_file.stem} import {func.name}"],
        )

        test_cases = []

        # Basic test case
        basic_test = self._create_basic_test(func)
        if basic_test:
            test_cases.append(basic_test)

        # Edge case tests
        if config.include_edge_cases:
            edge_tests = self._create_edge_case_tests(func)
            test_cases.extend(edge_tests[: config.max_test_cases_per_function])

        # Error case tests
        if config.include_error_cases:
            error_tests = self._create_error_tests(func)
            test_cases.extend(error_tests)

        # Docstring example tests
        if config.include_docstring_examples and func.docstring:
            docstring_tests = self._create_docstring_tests(func)
            test_cases.extend(docstring_tests)

        # Limit total test cases
        suite.test_cases = test_cases[: config.max_test_cases_per_function]

        return suite

    def _generate_class_suite(
        self,
        cls: ClassSignature,
        source_file: Path,
        config: TestGenConfig,
    ) -> TestSuite:
        """Generate test suite for a class."""
        suite = TestSuite(
            name=f"test_{cls.name}",
            target_file=source_file,
            target_name=cls.name,
            imports=[f"from {source_file.stem} import {cls.name}"],
        )

        test_cases = []

        # Test instantiation
        init_test = TestCase(
            name=f"test_{cls.name.lower()}_instantiation",
            description=f"Test that {cls.name} can be instantiated",
            assertions=[
                TestAssertion(
                    assertion_type=AssertionType.NOT_NONE,
                    expected=None,
                    actual_expr="instance",
                ),
                TestAssertion(
                    assertion_type=AssertionType.INSTANCE,
                    expected=cls.name,
                    actual_expr="instance",
                ),
            ],
            setup_code=f"instance = {cls.name}()",
        )
        test_cases.append(init_test)

        # Test methods
        for method in cls.methods:
            if method.name.startswith("_"):
                continue  # Skip private methods

            method_tests = self._create_method_tests(cls, method, config)
            test_cases.extend(method_tests)

        suite.test_cases = test_cases[: config.max_test_cases_per_function]

        return suite

    def _create_basic_test(self, func: FunctionSignature) -> Optional[TestCase]:
        """Create a basic test case for a function."""
        # Build call expression
        call_args = []
        inputs = []

        for param_name, type_hint in func.parameters:
            if param_name in ("self", "cls") or param_name.startswith("*"):
                continue

            # Get a representative value
            values = self.analyzer.infer_test_values(type_hint)
            if values:
                desc, value = values[0]  # Use first value
                inputs.append(TestInput(name=param_name, value=value, type_hint=type_hint))
                call_args.append(f"{param_name}={value!r}")

        call_expr = f"{func.name}({', '.join(call_args)})"

        # Determine assertion based on return type
        assertions = []
        if func.return_type:
            if "None" in func.return_type:
                assertions.append(
                    TestAssertion(
                        assertion_type=AssertionType.NONE,
                        expected=None,
                        actual_expr="result",
                    )
                )
            elif "bool" in func.return_type.lower():
                assertions.append(
                    TestAssertion(
                        assertion_type=AssertionType.INSTANCE,
                        expected="bool",
                        actual_expr="result",
                    )
                )
            else:
                assertions.append(
                    TestAssertion(
                        assertion_type=AssertionType.NOT_NONE,
                        expected=None,
                        actual_expr="result",
                    )
                )
        else:
            # No return type hint - just check it runs
            assertions.append(
                TestAssertion(
                    assertion_type=AssertionType.TRUE,
                    expected=True,
                    actual_expr="True",
                    message="Function executed without error",
                )
            )

        return TestCase(
            name=f"test_{func.name}_basic",
            description=f"Basic test for {func.name}",
            inputs=inputs,
            assertions=assertions,
            setup_code=f"result = {call_expr}",
            test_type=TestType.UNIT,
        )

    def _create_edge_case_tests(
        self,
        func: FunctionSignature,
    ) -> list[TestCase]:
        """Create edge case tests."""
        test_cases = []

        for param_name, type_hint in func.parameters:
            if param_name in ("self", "cls") or param_name.startswith("*"):
                continue

            values = self.analyzer.infer_test_values(type_hint)

            for desc, value in values[1:]:  # Skip first (basic) value
                test_name = f"test_{func.name}_{param_name}_{desc}"

                # Build call with this specific value
                call_args = []
                for p_name, p_hint in func.parameters:
                    if p_name in ("self", "cls") or p_name.startswith("*"):
                        continue
                    if p_name == param_name:
                        call_args.append(f"{p_name}={value!r}")
                    else:
                        # Use default value
                        default_values = self.analyzer.infer_test_values(p_hint)
                        if default_values:
                            call_args.append(f"{p_name}={default_values[0][1]!r}")

                call_expr = f"{func.name}({', '.join(call_args)})"

                test_cases.append(
                    TestCase(
                        name=test_name,
                        description=f"Test {func.name} with {param_name}={desc}",
                        assertions=[
                            TestAssertion(
                                assertion_type=AssertionType.NOT_NONE,
                                expected=None,
                                actual_expr="result",
                                message=f"Should handle {desc} {param_name}",
                            )
                        ],
                        setup_code=f"result = {call_expr}",
                        test_type=TestType.UNIT,
                        tags=["edge_case"],
                    )
                )

        return test_cases

    def _create_error_tests(
        self,
        func: FunctionSignature,
    ) -> list[TestCase]:
        """Create error condition tests."""
        test_cases = []

        error_conditions = self.analyzer.detect_error_conditions(func)

        for desc, input_expr, exception_type in error_conditions:
            test_name = f"test_{func.name}_{desc}"

            test_cases.append(
                TestCase(
                    name=test_name,
                    description=f"Test that {func.name} raises {exception_type.__name__}",
                    expected_exception=exception_type.__name__,
                    setup_code=f"{func.name}({input_expr})",
                    test_type=TestType.UNIT,
                    tags=["error_case"],
                )
            )

        return test_cases

    def _create_docstring_tests(
        self,
        func: FunctionSignature,
    ) -> list[TestCase]:
        """Create tests from docstring examples."""
        if not func.docstring:
            return []

        test_cases = []
        examples = self.analyzer.extract_docstring_examples(func.docstring)

        for i, (expr, expected) in enumerate(examples):
            test_name = f"test_{func.name}_docstring_example_{i + 1}"

            test_cases.append(
                TestCase(
                    name=test_name,
                    description=f"Docstring example: {expr} -> {expected}",
                    assertions=[
                        TestAssertion(
                            assertion_type=AssertionType.EQUALS,
                            expected=expected,
                            actual_expr="result",
                        )
                    ],
                    setup_code=f"result = {expr}",
                    test_type=TestType.UNIT,
                    tags=["docstring_example"],
                )
            )

        return test_cases

    def _create_method_tests(
        self,
        cls: ClassSignature,
        method: FunctionSignature,
        config: TestGenConfig,
    ) -> list[TestCase]:
        """Create tests for a class method."""
        test_cases = []

        # Basic method test
        call_args = []
        for param_name, type_hint in method.parameters:
            if param_name in ("self", "cls"):
                continue
            values = self.analyzer.infer_test_values(type_hint)
            if values:
                call_args.append(f"{param_name}={values[0][1]!r}")

        if method.is_static:
            call_expr = f"{cls.name}.{method.name}({', '.join(call_args)})"
            setup = f"result = {call_expr}"
        elif method.is_classmethod:
            call_expr = f"{cls.name}.{method.name}({', '.join(call_args)})"
            setup = f"result = {call_expr}"
        else:
            call_expr = f"instance.{method.name}({', '.join(call_args)})"
            setup = f"instance = {cls.name}()\nresult = {call_expr}"

        test_cases.append(
            TestCase(
                name=f"test_{cls.name.lower()}_{method.name}",
                description=f"Test {cls.name}.{method.name}",
                assertions=[
                    TestAssertion(
                        assertion_type=(
                            AssertionType.NOT_NONE if method.return_type else AssertionType.TRUE
                        ),
                        expected=None if method.return_type else True,
                        actual_expr="result" if method.return_type else "True",
                    )
                ],
                setup_code=setup,
                test_type=TestType.UNIT,
            )
        )

        return test_cases
