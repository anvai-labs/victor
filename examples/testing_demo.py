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

"""Demo of Victor's Testing Tool.

Demonstrates running pytest and getting structured test results:
- Run all tests in a directory
- Run specific test files
- Run with custom pytest arguments
- Get structured JSON report with pass/fail counts

Usage:
    python examples/testing_demo.py
"""

import asyncio
import tempfile
from pathlib import Path
from victor.tools.testing_tool import test


def setup_demo_files(temp_dir: Path) -> tuple[Path, Path]:
    """Create demo source and test files."""
    # Create source module
    src_code = '''
def add(a, b):
    """Add two numbers."""
    return a + b


def subtract(a, b):
    """Subtract b from a."""
    return a - b


def divide(a, b):
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def validate_email(email):
    """Validate email address."""
    if not email or "@" not in email:
        raise ValueError("Invalid email")
    return email.lower()
'''

    # Create test file
    test_code = '''
import pytest
from calc import add, subtract, divide, validate_email


class TestMath:
    """Tests for math functions."""

    def test_add_positive(self):
        assert add(2, 3) == 5

    def test_add_negative(self):
        assert add(-1, -1) == -2

    def test_add_zero(self):
        assert add(0, 0) == 0

    def test_subtract(self):
        assert subtract(5, 3) == 2

    def test_divide(self):
        assert divide(10, 2) == 5

    def test_divide_by_zero(self):
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(10, 0)


class TestValidation:
    """Tests for validation functions."""

    def test_valid_email(self):
        assert validate_email("Test@Example.com") == "test@example.com"

    def test_invalid_email_no_at(self):
        with pytest.raises(ValueError, match="Invalid email"):
            validate_email("notanemail")

    def test_invalid_email_empty(self):
        with pytest.raises(ValueError, match="Invalid email"):
            validate_email("")

    def test_invalid_email_none(self):
        with pytest.raises(ValueError, match="Invalid email"):
            validate_email(None)


# Intentionally failing test for demo
def test_failing_example():
    """This test intentionally fails to demonstrate failure reporting."""
    assert 1 == 2, "Intentional failure for demo"
'''

    # Write files
    src_file = temp_dir / "calc.py"
    src_file.write_text(src_code.strip())

    test_file = temp_dir / "test_calc.py"
    test_file.write_text(test_code.strip())

    return src_file, test_file


async def demo_run_all_tests(temp_dir: Path):
    """Demo running all tests in a directory."""
    print("\n\n🧪 Run All Tests Demo")
    print("=" * 70)

    print("\n1️⃣ Running all tests in directory...")
    result = await test(path=str(temp_dir))

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        if "stdout" in result:
            print(f"\nStdout:\n{result['stdout']}")
        if "stderr" in result:
            print(f"\nStderr:\n{result['stderr']}")
    else:
        summary = result.get("summary", {})
        print("\n✅ Test run complete!")
        print("\n📊 Results Summary:")
        print(f"   Total tests: {summary.get('total_tests', 0)}")
        print(f"   ✓ Passed: {summary.get('passed', 0)}")
        print(f"   ✗ Failed: {summary.get('failed', 0)}")
        print(f"   ⊘ Skipped: {summary.get('skipped', 0)}")

        failures = result.get("failures", [])
        if failures:
            print("\n❌ Failed Tests:")
            for failure in failures:
                print(f"\n   {failure['test_name']}:")
                print(f"      {failure['error_message']}")


async def demo_run_specific_file(temp_dir: Path):
    """Demo running tests from a specific file."""
    print("\n\n🎯 Run Specific Test File Demo")
    print("=" * 70)

    test_file = temp_dir / "test_calc.py"
    print(f"\n1️⃣ Running tests in: {test_file.name}")

    result = await test(path=str(test_file))

    if "error" not in result:
        summary = result.get("summary", {})
        print(f"\n📊 Results: {summary.get('passed', 0)} passed, {summary.get('failed', 0)} failed")


async def demo_run_with_args(temp_dir: Path):
    """Demo running tests with custom pytest arguments."""
    print("\n\n⚙️  Run Tests with Custom Arguments Demo")
    print("=" * 70)

    print("\n1️⃣ Running only TestMath class with verbose output...")
    result = await test(path=str(temp_dir), pytest_args=["-k", "TestMath", "-v"])

    if "error" not in result:
        summary = result.get("summary", {})
        print("\n📊 TestMath Results:")
        print(f"   Total: {summary.get('total_tests', 0)}")
        print(f"   Passed: {summary.get('passed', 0)}")
        print(f"   Failed: {summary.get('failed', 0)}")


async def demo_run_passing_only(temp_dir: Path):
    """Demo running only passing tests."""
    print("\n\n✓ Run Passing Tests Only Demo")
    print("=" * 70)

    print("\n1️⃣ Excluding the intentionally failing test...")
    result = await test(path=str(temp_dir), pytest_args=["-k", "not failing_example"])

    if "error" not in result:
        summary = result.get("summary", {})
        print("\n📊 Results (excluding failing test):")
        print(f"   Total: {summary.get('total_tests', 0)}")
        print(f"   Passed: {summary.get('passed', 0)}")
        print(f"   Failed: {summary.get('failed', 0)}")

        if summary.get("failed", 0) == 0:
            print("\n   🎉 All tests passed!")


async def main():
    """Run all testing demos."""
    print("🎯 Victor Testing Tool Demo")
    print("=" * 70)
    print("\nDemonstrating pytest integration with structured JSON reports")
    print("\nNote: Requires pytest and pytest-json-report to be installed:")
    print("      pip install pytest pytest-json-report")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        print("\n\n📁 Setting Up Demo Files")
        print("=" * 70)
        src_file, test_file = setup_demo_files(temp_path)
        print(f"   Created: {src_file.name}")
        print(f"   Created: {test_file.name}")

        # Show test file content
        print("\n📝 Test file content:")
        print("-" * 70)
        content = test_file.read_text()
        print(content[:800] + "..." if len(content) > 800 else content)

        # Run demos
        await demo_run_all_tests(temp_path)
        await demo_run_specific_file(temp_path)
        await demo_run_with_args(temp_path)
        await demo_run_passing_only(temp_path)

    print("\n\n✨ Demo Complete!")
    print("\nVictor's Testing Tool provides:")
    print("  • Run pytest with structured JSON output")
    print("  • Target specific files or directories")
    print("  • Pass custom pytest arguments (-k, -v, -x, etc.)")
    print("  • Get detailed failure reports with error messages")
    print("  • Automatic cleanup of report files")
    print("\nPerfect for:")
    print("  • CI/CD integration")
    print("  • Test-driven development")
    print("  • Automated test monitoring")
    print("  • Quality gate enforcement")


if __name__ == "__main__":
    asyncio.run(main())
