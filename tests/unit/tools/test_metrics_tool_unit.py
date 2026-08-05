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

"""Tests for metrics_tool module."""

import pytest
import tempfile
from pathlib import Path

from victor.tools.metrics_tool import (
    _calculate_complexity_score,
    _calculate_maintainability_index,
    metrics,
)


class TestCalculateComplexityScore:
    """Tests for _calculate_complexity_score function."""

    def test_simple_function(self):
        """Test complexity of simple function."""
        code = """
def hello():
    print("hello")
"""
        score = _calculate_complexity_score(code)
        assert score == 1  # Base complexity

    def test_function_with_if(self):
        """Test complexity with if statement."""
        code = """
def check(x):
    if x > 0:
        return True
    return False
"""
        score = _calculate_complexity_score(code)
        assert score >= 2  # Base + 1 for if

    def test_function_with_loop(self):
        """Test complexity with for loop."""
        code = """
def iterate(items):
    for item in items:
        print(item)
"""
        score = _calculate_complexity_score(code)
        assert score >= 2  # Base + 1 for for

    def test_function_with_while(self):
        """Test complexity with while loop."""
        code = """
def loop():
    while True:
        pass
"""
        score = _calculate_complexity_score(code)
        assert score >= 2  # Base + 1 for while

    def test_function_with_except(self):
        """Test complexity with exception handler."""
        code = """
def safe():
    try:
        pass
    except Exception:
        pass
"""
        score = _calculate_complexity_score(code)
        assert score >= 2  # Base + 1 for except

    def test_invalid_syntax(self):
        """Test complexity with invalid code."""
        code = "def broken("
        score = _calculate_complexity_score(code)
        assert score == 0


class TestCalculateMaintainabilityIndex:
    """Tests for _calculate_maintainability_index function."""

    def test_simple_code(self):
        """Test maintainability of simple code."""
        code = """
def hello():
    print("hello")
"""
        mi = _calculate_maintainability_index(code)
        assert mi > 0
        assert mi <= 100

    def test_complex_code(self):
        """Test maintainability of complex code."""
        code = """
def complex_function(a, b, c, d, e):
    if a > 0:
        if b > 0:
            if c > 0:
                for i in range(d):
                    while e > 0:
                        e -= 1
                        if e % 2 == 0:
                            continue
    return a + b + c + d + e
"""
        mi = _calculate_maintainability_index(code)
        # Complex code should have lower maintainability
        assert mi >= 0
        assert mi <= 100

    def test_invalid_syntax(self):
        """Test maintainability with invalid code."""
        code = "def broken("
        mi = _calculate_maintainability_index(code)
        assert mi == 0.0


class TestAnalyzeMetrics:
    """Tests for analyze_metrics function."""

    @pytest.mark.asyncio
    async def test_analyze_file(self):
        """Test analyzing a single file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
def hello():
    print("hello")

def goodbye():
    print("goodbye")
""")
            temp_path = f.name

        try:
            result = await metrics(path=temp_path)
            assert result["success"] is True
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_analyze_directory(self):
        """Test analyzing a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(f"{tmpdir}/file1.py").write_text("def foo(): pass")
            Path(f"{tmpdir}/file2.py").write_text("def bar(): pass")

            result = await metrics(path=tmpdir)
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_analyze_nonexistent_path(self):
        """Test analyzing nonexistent path."""
        result = await metrics(path="/nonexistent/path")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_analyze_specific_metrics(self):
        """Test analyzing specific metrics."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello(): print('hello')")
            temp_path = f.name

        try:
            result = await metrics(path=temp_path, metrics_list=["complexity", "maintainability"])
            assert result["success"] is True
        finally:
            Path(temp_path).unlink(missing_ok=True)
