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
Language Server Protocol utilities for Victor verticals.

This contrib package provides shared LSP functionality that can
be used by multiple verticals without creating framework-to-vertical dependencies.

Components:
- BasicLSPClient: Simple LSP client stub for when full LSP is unavailable

Note: Full LSP integration requires victor-coding package. This package
provides protocol definitions and basic stubs for graceful degradation.

Usage:
    from victor.contrib.lsp import BasicLSPClient

    lsp = BasicLSPClient()
    # Returns empty results - full LSP requires victor-coding
    completions = await lsp.get_completions(file_path, line, char)
"""

from victor.contrib.lsp.client import BasicLSPClient

__all__ = [
    "BasicLSPClient",
]
