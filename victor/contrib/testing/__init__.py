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

"""Testing utilities for Victor verticals.

Provides base classes and utilities for implementing tests
in verticals. Verticals can use these utilities to avoid
duplicating test code across verticals.

Usage:
    from victor.contrib.testing import VerticalTestCase, MockProviderMixin

    class TestMyVertical(VerticalTestCase, MockProviderMixin):
        vertical_name = \"myvertical\"

        def test_vertical_initialization(self):
            assistant = self.create_vertical_assistant()
            self.assertIsNotNone(assistant)
"""

from victor.contrib.testing.base_test import VerticalTestCase
from victor.contrib.testing.fixtures import (
    MockProviderMixin,
    TestAssistantMixin,
    MockToolMixin,
)

__all__ = [
    "VerticalTestCase",
    "MockProviderMixin",
    "TestAssistantMixin",
    "MockToolMixin",
]
