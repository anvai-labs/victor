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

"""Tests for the correction submodule."""

from datetime import datetime

import pytest

from victor.evaluation.correction import (
    # Types
    Language,
    CodeValidationResult,
    CorrectionFeedback,
    # Base
    ValidatorCapabilities,
    # Detection
    LanguageDetector,
    detect_language,
    get_detector,
    # Registry
    CodeValidatorRegistry,
    FeedbackGenerator,
    RetryPromptBuilder,
    SelfCorrector,
    create_self_corrector,
    # Metrics
    CorrectionAttempt,
    CorrectionMetrics,
    CorrectionMetricsCollector,
    get_metrics_collector,
    reset_metrics,
    # Validators
    GenericCodeValidator,
    PythonCodeValidator,
)


class TestLanguageEnum:
    """Tests for Language enum."""

    def test_language_values(self):
        """Test language enum has expected values."""
        assert Language.PYTHON is not None
        assert Language.JAVASCRIPT is not None
        assert Language.TYPESCRIPT is not None
        assert Language.GO is not None
        assert Language.RUST is not None
        assert Language.UNKNOWN is not None

    def test_from_string_python(self):
        """Test from_string conversion for Python."""
        assert Language.from_string("python") == Language.PYTHON
        assert Language.from_string("PYTHON") == Language.PYTHON
        assert Language.from_string("Python") == Language.PYTHON

    def test_from_string_javascript(self):
        """Test from_string conversion for JavaScript."""
        assert Language.from_string("javascript") == Language.JAVASCRIPT

    def test_from_string_csharp(self):
        """Test from_string conversion for C#."""
        assert Language.from_string("c#") == Language.CSHARP
        assert Language.from_string("C#") == Language.CSHARP

    def test_from_string_cpp(self):
        """Test from_string conversion for C++."""
        assert Language.from_string("c++") == Language.CPP

    def test_from_string_unknown(self):
        """Test from_string returns UNKNOWN for invalid language."""
        assert Language.from_string("unknown_lang") == Language.UNKNOWN
        assert Language.from_string("") == Language.UNKNOWN


class TestCodeValidationResult:
    """Tests for CodeValidationResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = CodeValidationResult(valid=True)
        assert result.valid is True
        assert result.language == Language.UNKNOWN
        assert result.syntax_valid is True
        assert result.imports_valid is True
        assert result.errors == ()
        assert result.warnings == ()
        assert result.syntax_error is None
        assert result.used_ast_validation is False

    def test_success_factory(self):
        """Test success factory method."""
        result = CodeValidationResult.success(Language.PYTHON, used_ast=True)
        assert result.valid is True
        assert result.language == Language.PYTHON
        assert result.used_ast_validation is True

    def test_failure_factory(self):
        """Test failure factory method."""
        result = CodeValidationResult.failure(
            ["Error 1", "Error 2"],
            Language.PYTHON,
            syntax_error="SyntaxError: invalid syntax",
        )
        assert result.valid is False
        assert result.language == Language.PYTHON
        assert result.errors == ("Error 1", "Error 2")
        assert result.syntax_error == "SyntaxError: invalid syntax"
        assert result.syntax_valid is False

    def test_with_warnings(self):
        """Test with_warnings method."""
        result = CodeValidationResult.success(Language.PYTHON)
        result_with_warnings = result.with_warnings(["Warning 1", "Warning 2"])
        assert result_with_warnings.warnings == ("Warning 1", "Warning 2")
        assert result.warnings == ()  # Original unchanged

    def test_immutability(self):
        """Test that CodeValidationResult is frozen/immutable."""
        result = CodeValidationResult(valid=True)
        with pytest.raises(AttributeError):
            result.valid = False


class TestCorrectionFeedback:
    """Tests for CorrectionFeedback dataclass."""

    def test_default_values(self):
        """Test default values."""
        feedback = CorrectionFeedback(has_issues=False)
        assert feedback.has_issues is False
        assert feedback.language == Language.UNKNOWN
        assert feedback.syntax_feedback is None
        assert feedback.import_feedback is None
        assert feedback.test_feedback is None
        assert feedback.general_feedback is None

    def test_no_issues_factory(self):
        """Test no_issues factory method."""
        feedback = CorrectionFeedback.no_issues(Language.PYTHON)
        assert feedback.has_issues is False
        assert feedback.language == Language.PYTHON

    def test_to_prompt_with_syntax_error(self):
        """Test to_prompt with syntax feedback."""
        feedback = CorrectionFeedback(has_issues=True, syntax_feedback="Line 5: SyntaxError")
        prompt = feedback.to_prompt()
        assert "SYNTAX ERROR" in prompt
        assert "Line 5: SyntaxError" in prompt

    def test_to_prompt_with_import_feedback(self):
        """Test to_prompt with import feedback."""
        feedback = CorrectionFeedback(has_issues=True, import_feedback="Missing import: os")
        prompt = feedback.to_prompt()
        assert "IMPORT ISSUES" in prompt
        assert "Missing import: os" in prompt

    def test_to_prompt_with_test_feedback(self):
        """Test to_prompt with test feedback."""
        feedback = CorrectionFeedback(has_issues=True, test_feedback="Tests: 3/10 passed")
        prompt = feedback.to_prompt()
        assert "TEST FAILURES" in prompt
        assert "3/10 passed" in prompt

    def test_to_prompt_empty(self):
        """Test to_prompt with no feedback."""
        feedback = CorrectionFeedback(has_issues=False)
        prompt = feedback.to_prompt()
        assert prompt == ""


class TestValidatorCapabilities:
    """Tests for ValidatorCapabilities."""

    def test_default_values(self):
        """Test default capability values."""
        caps = ValidatorCapabilities()
        assert caps.has_ast_parsing is False
        assert caps.has_import_detection is False
        assert caps.has_type_checking is False
        assert caps.has_auto_fix is True

    def test_custom_values(self):
        """Test custom capability values."""
        caps = ValidatorCapabilities(
            has_ast_parsing=True,
            has_import_detection=True,
        )
        assert caps.has_ast_parsing is True
        assert caps.has_import_detection is True

    def test_repr(self):
        """Test repr output."""
        caps = ValidatorCapabilities(has_ast_parsing=True)
        assert "ast=True" in repr(caps)


class TestLanguageDetector:
    """Tests for LanguageDetector."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = LanguageDetector()

    def test_detect_from_extension_python(self):
        """Test detection from .py extension."""
        result = self.detector.detect("", filename="test.py")
        assert result == Language.PYTHON

    def test_detect_from_extension_javascript(self):
        """Test detection from .js extension."""
        result = self.detector.detect("", filename="test.js")
        assert result == Language.JAVASCRIPT

    def test_detect_from_extension_typescript(self):
        """Test detection from .ts extension."""
        result = self.detector.detect("", filename="test.ts")
        assert result == Language.TYPESCRIPT

    def test_detect_from_extension_go(self):
        """Test detection from .go extension."""
        result = self.detector.detect("", filename="test.go")
        assert result == Language.GO

    def test_detect_from_extension_rust(self):
        """Test detection from .rs extension."""
        result = self.detector.detect("", filename="main.rs")
        assert result == Language.RUST

    def test_detect_from_shebang_python(self):
        """Test detection from Python shebang."""
        code = "#!/usr/bin/env python3\nprint('hello')"
        result = self.detector.detect(code)
        assert result == Language.PYTHON

    def test_detect_from_shebang_node(self):
        """Test detection from Node shebang."""
        code = "#!/usr/bin/env node\nconsole.log('hello')"
        result = self.detector.detect(code)
        assert result == Language.JAVASCRIPT

    def test_detect_from_content_python(self):
        """Test detection from Python content patterns."""
        code = """
def hello():
    print("Hello, World!")

if __name__ == "__main__":
    hello()
"""
        result = self.detector.detect(code)
        assert result == Language.PYTHON

    def test_detect_from_content_go(self):
        """Test detection from Go content patterns."""
        code = """
package main

func main() {
    fmt.Println("Hello")
}
"""
        result = self.detector.detect(code)
        assert result == Language.GO

    def test_detect_from_content_rust(self):
        """Test detection from Rust content patterns."""
        code = """
fn main() {
    let mut x = 5;
    println!("{}", x);
}
"""
        result = self.detector.detect(code)
        assert result == Language.RUST

    def test_detect_unknown(self):
        """Test detection returns UNKNOWN for unrecognizable code."""
        result = self.detector.detect("random text")
        assert result == Language.UNKNOWN

    def test_detect_empty_code(self):
        """Test detection with empty code."""
        result = self.detector.detect("")
        assert result == Language.UNKNOWN

    def test_supported_extensions(self):
        """Test supported_extensions property."""
        extensions = self.detector.supported_extensions
        assert ".py" in extensions
        assert ".js" in extensions
        assert ".go" in extensions

    def test_detectable_languages(self):
        """Test detectable_languages property."""
        languages = self.detector.detectable_languages
        assert Language.PYTHON in languages
        assert Language.JAVASCRIPT in languages


class TestDetectLanguageFunction:
    """Tests for detect_language convenience function."""

    def test_detect_language_python(self):
        """Test detect_language with Python file."""
        result = detect_language("", filename="test.py")
        assert result == Language.PYTHON

    def test_detect_language_content(self):
        """Test detect_language with Python content."""
        code = "def foo():\n    return 42"
        result = detect_language(code)
        assert result == Language.PYTHON


class TestGetDetector:
    """Tests for get_detector singleton function."""

    def test_get_detector_returns_detector(self):
        """Test get_detector returns LanguageDetector."""
        detector = get_detector()
        assert isinstance(detector, LanguageDetector)

    def test_get_detector_singleton(self):
        """Test get_detector returns same instance."""
        d1 = get_detector()
        d2 = get_detector()
        assert d1 is d2


class TestCodeValidatorRegistry:
    """Tests for CodeValidatorRegistry."""

    def setup_method(self):
        """Set up test fixtures."""
        # Reset singleton for clean test state
        CodeValidatorRegistry.reset_singleton()
        self.registry = CodeValidatorRegistry()

    def teardown_method(self):
        """Clean up after tests."""
        CodeValidatorRegistry.reset_singleton()

    def test_singleton_pattern(self):
        """Test registry uses singleton pattern."""
        registry1 = CodeValidatorRegistry()
        registry2 = CodeValidatorRegistry()
        assert registry1 is registry2

    def test_register_validator(self):
        """Test registering a validator."""
        validator = PythonCodeValidator()
        self.registry.register(validator)
        assert self.registry.has_validator(Language.PYTHON)

    def test_get_validator_returns_registered(self):
        """Test get_validator returns registered validator for the language."""
        validator = PythonCodeValidator()
        self.registry.register(validator)
        retrieved = self.registry.get_validator(Language.PYTHON)
        # The registry may have auto-discovered another PythonCodeValidator,
        # so we check that we get a PythonCodeValidator instance
        assert isinstance(retrieved, PythonCodeValidator)

    def test_get_validator_fallback(self):
        """Test get_validator returns fallback for unknown language."""
        # Clear and force discovery
        self.registry.reset()
        self.registry._discovered = True
        validator = self.registry.get_validator(Language.UNKNOWN)
        assert isinstance(validator, GenericCodeValidator)

    def test_discover_validators(self):
        """Test discover_validators finds validators."""
        self.registry.reset()
        count = self.registry.discover_validators()
        assert count > 0
        assert self.registry.has_validator(Language.PYTHON)

    def test_registered_languages(self):
        """Test registered_languages property."""
        self.registry.discover_validators()
        languages = self.registry.registered_languages
        assert Language.PYTHON in languages

    def test_reset(self):
        """Test reset clears registry."""
        self.registry.discover_validators()
        self.registry.reset()
        assert len(self.registry.registered_languages) == 0


class TestGenericCodeValidator:
    """Tests for GenericCodeValidator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = GenericCodeValidator()

    def test_supported_languages(self):
        """Test supported languages includes UNKNOWN."""
        assert Language.UNKNOWN in self.validator.supported_languages

    def test_validate_returns_success(self):
        """Test validate returns success for any code."""
        result = self.validator.validate("any code here")
        assert result.valid is True

    def test_fix_returns_normalized(self):
        """Test fix returns code with normalized trailing newline."""
        validation = CodeValidationResult.success()
        fixed = self.validator.fix("original code", validation)
        # GenericCodeValidator.fix() ensures single trailing newline
        assert fixed == "original code\n"


class TestPythonCodeValidator:
    """Tests for PythonCodeValidator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = PythonCodeValidator()

    def test_supported_languages(self):
        """Test supported languages includes Python."""
        assert Language.PYTHON in self.validator.supported_languages

    def test_validate_valid_code(self):
        """Test validate with valid Python code."""
        code = "def hello():\n    return 'world'"
        result = self.validator.validate(code)
        assert result.valid is True
        assert result.syntax_valid is True
        assert result.used_ast_validation is True

    def test_validate_syntax_error(self):
        """Test validate detects syntax errors."""
        code = "def hello(\n    return 'world'"
        result = self.validator.validate(code)
        assert result.valid is False
        assert result.syntax_valid is False
        assert result.syntax_error is not None

    def test_clean_markdown(self):
        """Test markdown cleaning."""
        code = "```python\ndef hello():\n    pass\n```"
        cleaned = self.validator.clean_markdown(code)
        assert "```" not in cleaned
        assert "def hello():" in cleaned

    def test_clean_markdown_preserves_document(self):
        """Regression: a markdown document (prose + code blocks) is returned verbatim.

        Previously clean_markdown used an unanchored re.search and returned only the
        first fenced block's interior, truncating documents to one code block.
        """
        doc = (
            "# Title\n\nSome prose here.\n\n"
            "```python\ndef h1():\n    return 1\n```\n\n"
            "More prose.\n\n"
            "```text\nsandhi keys add\n```\n"
        )
        assert self.validator.clean_markdown(doc) == doc

    def test_clean_markdown_preserves_multiple_blocks(self):
        """Two adjacent fenced blocks (no prose) are NOT collapsed to one."""
        code = "```python\ndef a():\n    pass\n```\n```python\ndef b():\n    pass\n```"
        assert self.validator.clean_markdown(code) == code

    def test_clean_markdown_generic_single_fence(self):
        """A single generic (untagged) fence wrapping the whole input is unwrapped."""
        code = "```\ndef a():\n    pass\n```"
        cleaned = self.validator.clean_markdown(code)
        assert cleaned == "def a():\n    pass"


class TestFeedbackGenerator:
    """Tests for FeedbackGenerator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = FeedbackGenerator()

    def test_generate_no_issues(self):
        """Test generate with no issues."""
        validation = CodeValidationResult.success(Language.PYTHON)
        feedback = self.generator.generate("code", validation)
        assert feedback.has_issues is False
        assert feedback.language == Language.PYTHON

    def test_generate_with_syntax_error(self):
        """Test generate with syntax error."""
        validation = CodeValidationResult.failure(
            ["Syntax error"], Language.PYTHON, syntax_error="Line 5: SyntaxError"
        )
        feedback = self.generator.generate("code", validation)
        assert feedback.has_issues is True
        assert feedback.syntax_feedback is not None
        assert "syntax error" in feedback.syntax_feedback.lower()

    def test_generate_with_test_failures(self):
        """Test generate with test failures."""
        validation = CodeValidationResult.success(Language.PYTHON)
        feedback = self.generator.generate(
            "code",
            validation,
            test_passed=3,
            test_total=10,
            test_stderr="AssertionError: expected 5 got 3",
        )
        assert feedback.has_issues is True
        assert feedback.test_feedback is not None
        assert "3/10 passed" in feedback.test_feedback

    def test_generate_extract_error_info(self):
        """Test error info extraction."""
        stderr = """
Traceback (most recent call last):
  File "test.py", line 10
AssertionError: expected 5, got 3
More output here
"""
        error_info = self.generator._extract_error_info(stderr)
        assert "AssertionError" in error_info
        assert len(error_info) <= self.generator.max_error_length + 50


class TestRetryPromptBuilder:
    """Tests for RetryPromptBuilder."""

    def setup_method(self):
        """Set up test fixtures."""
        self.builder = RetryPromptBuilder()

    def test_build_retry_prompt(self):
        """Test building a retry prompt."""
        feedback = CorrectionFeedback(
            has_issues=True,
            language=Language.PYTHON,
            syntax_feedback="Line 5 has syntax error",
        )
        prompt = self.builder.build(
            original_prompt="Write a function",
            previous_code="def foo()",
            feedback=feedback,
            iteration=1,
        )
        assert "Write a function" in prompt
        assert "def foo()" in prompt
        assert "syntax error" in prompt.lower()
        assert "```python" in prompt

    def test_language_markers(self):
        """Test language markers are set correctly."""
        assert self.builder.LANGUAGE_MARKERS[Language.PYTHON] == "python"
        assert self.builder.LANGUAGE_MARKERS[Language.JAVASCRIPT] == "javascript"


class TestSelfCorrector:
    """Tests for SelfCorrector orchestrator."""

    def setup_method(self):
        """Set up test fixtures."""
        # Reset singletons
        CodeValidatorRegistry.reset_singleton()

    def teardown_method(self):
        """Clean up after tests."""
        CodeValidatorRegistry.reset_singleton()

    def test_init_defaults(self):
        """Test default initialization."""
        corrector = SelfCorrector()
        assert corrector.max_iterations == 3
        assert corrector.auto_fix is True
        assert corrector.validate_before_test is True

    def test_init_custom(self):
        """Test custom initialization."""
        corrector = SelfCorrector(max_iterations=5, auto_fix=False)
        assert corrector.max_iterations == 5
        assert corrector.auto_fix is False

    def test_validate_and_fix_valid_code(self):
        """Test validate_and_fix with valid Python code."""
        corrector = SelfCorrector()
        code = "def hello():\n    return 42"
        fixed_code, validation = corrector.validate_and_fix(code, Language.PYTHON)
        assert validation.valid is True
        assert "def hello():" in fixed_code

    def test_validate_and_fix_with_markdown(self):
        """Test validate_and_fix strips markdown."""
        corrector = SelfCorrector()
        code = "```python\ndef hello():\n    return 42\n```"
        fixed_code, validation = corrector.validate_and_fix(code, Language.PYTHON)
        assert "```" not in fixed_code
        assert "def hello():" in fixed_code

    def test_validate_and_fix_does_not_truncate_document(self):
        """Regression: a markdown document passed to validate_and_fix is not truncated.

        The destructive clean_markdown previously collapsed this to one ~20-byte block.
        """
        corrector = SelfCorrector()
        doc = (
            "# TD\n\nIntro prose.\n\n"
            "```rust\nfn h1() -> i32 { 1 }\n```\n\nMore prose.\n\n"
            "```text\nsandhi keys add\n```\n"
        )
        fixed_code, _validation = corrector.validate_and_fix(doc, Language.UNKNOWN)
        assert fixed_code == doc  # document preserved verbatim, no truncation

    def test_validate_is_read_only(self):
        """validate() returns diagnostics without mutating/fixing the code."""
        corrector = SelfCorrector()
        code = "def hello():\n    return 42"
        snapshot = code
        validation = corrector.validate(code, Language.PYTHON)
        assert isinstance(validation, CodeValidationResult)
        # The input string is unchanged (validate never returns mutated code).
        assert code == snapshot

    def test_validate_does_not_truncate_document(self):
        """validate() on a markdown document returns diagnostics without truncating."""
        corrector = SelfCorrector()
        doc = "# Title\n\nProse.\n\n```python\ndef a():\n    pass\n```\n\nMore.\n"
        # Should not raise and should not depend on truncation; we only assert it returns.
        validation = corrector.validate(doc, Language.UNKNOWN)
        assert isinstance(validation, CodeValidationResult)

    def test_should_retry_under_limit(self):
        """Test should_retry under iteration limit."""
        corrector = SelfCorrector(max_iterations=3)
        validation = CodeValidationResult.failure(["Error"])
        assert corrector.should_retry(0, validation) is True
        assert corrector.should_retry(1, validation) is True
        assert corrector.should_retry(2, validation) is True

    def test_should_retry_at_limit(self):
        """Test should_retry at iteration limit."""
        corrector = SelfCorrector(max_iterations=3)
        validation = CodeValidationResult.failure(["Error"])
        assert corrector.should_retry(3, validation) is False

    def test_should_retry_validation_passed(self):
        """Test should_retry when validation passes."""
        corrector = SelfCorrector()
        validation = CodeValidationResult.success()
        assert corrector.should_retry(0, validation) is False

    def test_should_retry_tests_failed(self):
        """Test should_retry when tests fail."""
        corrector = SelfCorrector()
        validation = CodeValidationResult.success()
        assert corrector.should_retry(0, validation, test_passed=3, test_total=10) is True

    def test_should_retry_tests_passed(self):
        """Test should_retry when all tests pass."""
        corrector = SelfCorrector()
        validation = CodeValidationResult.success()
        assert corrector.should_retry(0, validation, test_passed=10, test_total=10) is False

    def test_generate_feedback(self):
        """Test generate_feedback method."""
        corrector = SelfCorrector()
        validation = CodeValidationResult.failure(
            ["Error"], Language.PYTHON, syntax_error="SyntaxError"
        )
        feedback = corrector.generate_feedback("code", validation)
        assert feedback.has_issues is True
        assert feedback.language == Language.PYTHON

    def test_build_retry_prompt(self):
        """Test build_retry_prompt method."""
        corrector = SelfCorrector()
        feedback = CorrectionFeedback(
            has_issues=True, language=Language.PYTHON, syntax_feedback="Error on line 5"
        )
        prompt = corrector.build_retry_prompt(
            original_prompt="Write code",
            previous_code="def foo()",
            feedback=feedback,
            iteration=1,
        )
        assert "Write code" in prompt
        assert "def foo()" in prompt

    def test_has_ast_support(self):
        """Test has_ast_support method."""
        corrector = SelfCorrector()
        assert corrector.has_ast_support(Language.PYTHON) is True

    def test_supported_languages(self):
        """Test supported_languages property."""
        corrector = SelfCorrector()
        languages = corrector.supported_languages
        assert Language.PYTHON in languages


class TestCreateSelfCorrector:
    """Tests for create_self_corrector factory."""

    def teardown_method(self):
        """Clean up after tests."""
        CodeValidatorRegistry.reset_singleton()

    def test_create_with_defaults(self):
        """Test factory with defaults."""
        corrector = create_self_corrector()
        assert corrector.max_iterations == 3
        assert corrector.auto_fix is True

    def test_create_with_custom(self):
        """Test factory with custom values."""
        corrector = create_self_corrector(max_iterations=5, auto_fix=False)
        assert corrector.max_iterations == 5
        assert corrector.auto_fix is False


class TestCorrectionAttempt:
    """Tests for CorrectionAttempt dataclass."""

    def test_create_attempt(self):
        """Test creating an attempt record."""
        attempt = CorrectionAttempt(
            task_id="task_1",
            language=Language.PYTHON,
            iteration=0,
            timestamp=datetime.now(),
            duration_ms=100.0,
            validation_before=CodeValidationResult.failure(["Error"]),
            validation_after=CodeValidationResult.success(),
            test_passed_before=3,
            test_passed_after=10,
            test_total=10,
            success=True,
            auto_fixed=True,
        )
        assert attempt.task_id == "task_1"
        assert attempt.language == Language.PYTHON
        assert attempt.success is True
        assert attempt.auto_fixed is True


class TestCorrectionMetrics:
    """Tests for CorrectionMetrics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = CorrectionMetrics()

    def test_default_values(self):
        """Test default metric values."""
        assert self.metrics.total_validations == 0
        assert self.metrics.total_corrections == 0
        assert self.metrics.successful_corrections == 0

    def test_record_validation(self):
        """Test recording a validation."""
        result = CodeValidationResult.failure(["Error"], syntax_error="Syntax error")
        self.metrics.record_validation(Language.PYTHON, result)
        assert self.metrics.total_validations == 1
        assert self.metrics.language_validations[Language.PYTHON] == 1
        assert self.metrics.syntax_errors == 1

    def test_record_correction(self):
        """Test recording a correction."""
        attempt = CorrectionAttempt(
            task_id="task_1",
            language=Language.PYTHON,
            iteration=0,
            timestamp=datetime.now(),
            duration_ms=100.0,
            validation_before=CodeValidationResult.failure(["Error"]),
            validation_after=CodeValidationResult.success(),
            test_passed_before=None,
            test_passed_after=None,
            test_total=None,
            success=True,
            auto_fixed=True,
        )
        self.metrics.record_correction(attempt)
        assert self.metrics.total_corrections == 1
        assert self.metrics.successful_corrections == 1
        assert self.metrics.auto_fixes_applied == 1

    def test_correction_success_rate(self):
        """Test success rate calculation."""
        # Record 2 corrections, 1 success
        for success in [True, False]:
            attempt = CorrectionAttempt(
                task_id=f"task_{success}",
                language=Language.PYTHON,
                iteration=0,
                timestamp=datetime.now(),
                duration_ms=100.0,
                validation_before=CodeValidationResult.failure(["Error"]),
                validation_after=None,
                test_passed_before=None,
                test_passed_after=None,
                test_total=None,
                success=success,
            )
            self.metrics.record_correction(attempt)
        assert self.metrics.correction_success_rate == 0.5

    def test_success_rate_by_language(self):
        """Test per-language success rate."""
        attempt = CorrectionAttempt(
            task_id="task_1",
            language=Language.PYTHON,
            iteration=0,
            timestamp=datetime.now(),
            duration_ms=100.0,
            validation_before=CodeValidationResult.failure(["Error"]),
            validation_after=CodeValidationResult.success(),
            test_passed_before=None,
            test_passed_after=None,
            test_total=None,
            success=True,
        )
        self.metrics.record_correction(attempt)
        rate = self.metrics.success_rate_by_language(Language.PYTHON)
        assert rate == 1.0

    def test_to_dict(self):
        """Test JSON serialization."""
        result = self.metrics.to_dict()
        assert "summary" in result
        assert "errors" in result
        assert "timing" in result
        assert "by_language" in result
        assert "by_iteration" in result

    def test_report(self):
        """Test human-readable report generation."""
        report = self.metrics.report()
        assert "CORRECTION METRICS REPORT" in report
        assert "SUMMARY" in report
        assert "ERROR TYPES" in report
        assert "TIMING" in report


class TestCorrectionMetricsCollector:
    """Tests for CorrectionMetricsCollector."""

    def setup_method(self):
        """Set up test fixtures."""
        self.collector = CorrectionMetricsCollector()

    def test_record_validation(self):
        """Test recording validation through collector."""
        result = CodeValidationResult.success(Language.PYTHON)
        self.collector.record_validation(Language.PYTHON, result)
        assert self.collector.metrics.total_validations == 1

    def test_track_correction_context_manager(self):
        """Test correction tracking context manager."""
        validation_before = CodeValidationResult.failure(["Error"])

        with self.collector.track_correction(
            task_id="task_1",
            language=Language.PYTHON,
            iteration=0,
            validation_before=validation_before,
        ) as tracker:
            tracker.set_result(success=True, auto_fixed=True)

        assert self.collector.metrics.total_corrections == 1
        assert self.collector.metrics.successful_corrections == 1
        assert self.collector.metrics.auto_fixes_applied == 1


class TestCorrectionTracker:
    """Tests for CorrectionTracker context manager."""

    def test_tracker_records_timing(self):
        """Test that tracker records duration."""
        collector = CorrectionMetricsCollector()
        validation_before = CodeValidationResult.failure(["Error"])

        with collector.track_correction(
            task_id="task_1",
            language=Language.PYTHON,
            iteration=0,
            validation_before=validation_before,
        ) as tracker:
            # Simulate some work
            tracker.set_result(success=True)

        assert len(collector.metrics.attempts) == 1
        assert collector.metrics.attempts[0].duration_ms > 0

    def test_tracker_set_result(self):
        """Test setting result on tracker."""
        collector = CorrectionMetricsCollector()
        validation_before = CodeValidationResult.failure(["Error"])
        validation_after = CodeValidationResult.success()

        with collector.track_correction(
            task_id="task_1",
            language=Language.PYTHON,
            iteration=0,
            validation_before=validation_before,
        ) as tracker:
            tracker.set_result(
                success=True,
                validation_after=validation_after,
                test_passed_after=10,
                auto_fixed=True,
            )

        attempt = collector.metrics.attempts[0]
        assert attempt.success is True
        assert attempt.validation_after is validation_after
        assert attempt.test_passed_after == 10
        assert attempt.auto_fixed is True


class TestGetMetricsCollector:
    """Tests for get_metrics_collector singleton function."""

    def teardown_method(self):
        """Clean up after tests."""
        reset_metrics()

    def test_get_metrics_collector_returns_collector(self):
        """Test get_metrics_collector returns collector."""
        collector = get_metrics_collector()
        assert isinstance(collector, CorrectionMetricsCollector)

    def test_get_metrics_collector_singleton(self):
        """Test get_metrics_collector returns same instance."""
        c1 = get_metrics_collector()
        c2 = get_metrics_collector()
        assert c1 is c2

    def test_reset_metrics(self):
        """Test reset_metrics clears global collector."""
        c1 = get_metrics_collector()
        reset_metrics()
        c2 = get_metrics_collector()
        assert c1 is not c2
