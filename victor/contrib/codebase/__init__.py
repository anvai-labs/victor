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

"""
Codebase analysis utilities for Victor verticals.

This contrib package provides shared codebase analysis functionality that can
be used by multiple verticals without creating framework-to-vertical dependencies.

Components:
- BasicCodebaseAnalyzer: Simple file-based codebase analyzer
- Pattern matching for common constructs (classes, functions, imports)

Usage:
    from victor.contrib.codebase import BasicCodebaseAnalyzer

    analyzer = BasicCodebaseAnalyzer()
    analysis = await analyzer.analyze_codebase(
        root_path=Path("/path/to/code"),
        include_patterns=["**/*.py"],
    )
"""

from victor.contrib.codebase.analyzer import BasicCodebaseAnalyzer

__all__ = [
    "BasicCodebaseAnalyzer",
]
