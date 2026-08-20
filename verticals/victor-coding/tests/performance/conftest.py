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

"""Configuration for performance tests."""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "benchmark: marks tests as benchmarks (deselect with '-m \"not benchmark\"')",
    )


@pytest.fixture
def sample_code():
    """Provide sample Python code for testing."""
    return """
def calculate_sum(numbers):
    '''Calculate the sum of a list of numbers.'''
    total = 0
    for num in numbers:
        total += num
    return total


class Calculator:
    '''A simple calculator class.'''

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
"""


@pytest.fixture
def temp_code_file(sample_code, tmp_path):
    """Provide a temporary file with sample code."""

    test_file = tmp_path / "test.py"
    test_file.write_text(sample_code)
    return str(test_file)
