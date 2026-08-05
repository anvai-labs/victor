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

"""Fix for mirrored test directory structure causing import confusion.

Problem: When pytest runs tests from tests/unit/agent/, it adds this directory
to sys.path. Python sees subdirectories like presentation/ and thinks 'agent' is a
top-level package. This causes ImportError when importing victor.agent.presentation
because Python tries to import from the wrong 'agent' package.

Solution: Remove test directories from sys.path to prevent package shadowing.
"""

import sys
from pathlib import Path

# Get the path to this conftest file
test_dir = Path(__file__).parent

# Remove this test directory and parent test directories from sys.path
# to prevent them from shadowing the actual victor.agent package
paths_to_remove = [str(test_dir)]
if test_dir.name == "agent":
    # Also remove parent directories that might cause issues
    for parent in [test_dir.parent, test_dir.parent.parent]:
        if "tests" in str(parent):
            paths_to_remove.append(str(parent))

for path in paths_to_remove:
    if path in sys.path:
        sys.path.remove(path)
